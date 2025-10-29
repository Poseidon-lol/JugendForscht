"""
Active learning loop orchestrating surrogate, generator and DFT interface.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the project root (with src/) is on sys.path
PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Could not locate project root containing src/")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.active_learn.acq import AcquisitionConfig, acquisition_score
from src.active_learn.sched import ActiveLearningScheduler, SchedulerConfig
from src.data.dataset import split_dataframe
from src.data.dft_int import DFTInterface, DFTJobSpec
from src.data.featurization import mol_to_graph
from src.models.ensemble import SurrogateEnsemble
from src.models.jtvae_extended import JTVAE, sample_conditional
from src.utils.log import get_logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKit_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    RDKit_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class LoopConfig:
    batch_size: int = 8
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    target_columns: Sequence[str] = ("HOMO", "LUMO")
    maximise: Sequence[bool] = (False, True)
    generator_samples: int = 32
    results_dir: Path = Path("experiments")
    assemble: Dict[str, object] = field(default_factory=dict)
    diversity_threshold: float = 0.85
    diversity_metric: str = "tanimoto"
    generator_refresh: Dict[str, object] = field(default_factory=dict)


class ActiveLearningLoop:
    def __init__(
        self,
        surrogate: SurrogateEnsemble,
        labelled: pd.DataFrame,
        pool: pd.DataFrame,
        config: LoopConfig,
        *,
        generator: Optional[JTVAE] = None,
        fragment_vocab: Optional[Dict[str, int]] = None,
        dft: Optional[DFTInterface] = None,
    ) -> None:
        self.surrogate = surrogate
        self.config = config
        self.labelled = labelled.reset_index(drop=True)
        self.pool = pool.reset_index(drop=True)
        self.generator = generator
        self.fragment_vocab = fragment_vocab or {}
        self.dft = dft
        self.scheduler = ActiveLearningScheduler(config.scheduler)
        self.history: List[pd.DataFrame] = []
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.assemble_kwargs = dict(config.assemble)
        self.diversity_threshold = float(getattr(config, "diversity_threshold", 0.0))
        self.diversity_metric = getattr(config, "diversity_metric", "tanimoto").lower()
        self.generator_refresh_kwargs = dict(getattr(config, "generator_refresh", {}))
        self._fingerprint_cache: Dict[str, Optional[object]] = {}
        self._fingerprints: List[object] = []
        if RDKit_AVAILABLE and self.diversity_threshold > 0:
            initial_smiles = pd.concat(
                [self.labelled.get("smiles", pd.Series(dtype=str)), self.pool.get("smiles", pd.Series(dtype=str))],
                axis=0,
            ).dropna().unique()
            for smi in initial_smiles:
                fp = self._fingerprint(smi)
                if fp is not None:
                    self._fingerprint_cache[smi] = fp
                    self._fingerprints.append(fp)

        if len(self.config.target_columns) != len(self.config.maximise):
            raise ValueError("target_columns and maximise length mismatch.")

    def _current_best(self) -> Optional[np.ndarray]:
        if self.labelled.empty:
            return None
        arr = self.labelled[self.config.target_columns].to_numpy(dtype=float)
        best = []
        for dim, maximise in enumerate(self.config.maximise):
            column = arr[:, dim]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                best.append(0.0)
            else:
                best.append(finite.max() if maximise else finite.min())
        return np.array(best)

    def _fingerprint(self, smiles: str):
        if not RDKit_AVAILABLE:
            return None
        if smiles in self._fingerprint_cache:
            return self._fingerprint_cache[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        self._fingerprint_cache[smiles] = fp
        return fp

    def _passes_diversity(self, smiles: str) -> bool:
        if self.diversity_threshold <= 0 or not RDKit_AVAILABLE:
            return True
        fp = self._fingerprint(smiles)
        if fp is None:
            return False
        if not self._fingerprints:
            self._fingerprints.append(fp)
            return True
        sims = [DataStructs.TanimotoSimilarity(fp, existing) for existing in self._fingerprints]
        if sims and max(sims) >= self.diversity_threshold:
            return False
        self._fingerprints.append(fp)
        return True

    def _normalise_predictions(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_targets = mean.shape[1]
        directions = np.array([1.0 if maximise else -1.0 for maximise in self.config.maximise])
        mus = np.zeros(n_targets)
        sigmas = np.ones(n_targets)
        for i, target in enumerate(self.config.target_columns):
            values = self.labelled[target].dropna().to_numpy(dtype=float)
            if values.size >= 2:
                oriented = values * directions[i]
                mus[i] = oriented.mean()
                sigma = oriented.std()
                sigmas[i] = sigma if sigma > 1e-6 else 1.0
            elif values.size == 1:
                mus[i] = values[0] * directions[i]
                sigmas[i] = 1.0
            else:
                mus[i] = 0.0
                sigmas[i] = 1.0
        mean_norm = ((mean * directions) - mus) / sigmas
        std_norm = std / sigmas
        return mean_norm, std_norm, mus, sigmas, directions

    def _save_diagnostics(self, pool_slice: pd.DataFrame, iteration: int) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            logger.debug("Matplotlib not available; skipping diagnostics plot for iteration %d.", iteration)
            return
        diag_dir = self.results_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        n_targets = len(self.config.target_columns)
        fig, axes = plt.subplots(n_targets, 1, figsize=(6, 3 * n_targets), squeeze=False)
        for idx, target in enumerate(self.config.target_columns):
            ax = axes[idx, 0]
            ax.scatter(
                pool_slice[f"pred_{target}"],
                pool_slice["acquisition_score"],
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel(f"Predicted {target}")
            ax.set_ylabel("Acquisition")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Acquisition diagnostics – iteration {iteration}")
        fig.tight_layout()
        fig.savefig(diag_dir / f"diag_iter_{iteration:03d}.png", dpi=150)
        plt.close(fig)

    def _refresh_generator(self) -> None:
        if self.generator is None or not self.fragment_vocab:
            return
        if len(self.labelled) < 5:
            return
        try:
            from src.data.jt_preprocess import JTPreprocessConfig, prepare_jtvae_examples
            from src.models.jtvae_extended import JTVDataset, train_jtvae
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Skipping generator refresh: preprocessing utilities unavailable (%s).", exc)
            return
        df = self.labelled.dropna(subset=["smiles", *self.config.target_columns])
        if df.empty:
            return
        df = df[["smiles", *self.config.target_columns]].drop_duplicates(subset="smiles")
        config = JTPreprocessConfig(
            max_fragments=getattr(self.generator, "max_tree_nodes", 12),
            condition_columns=self.config.target_columns,
        )
        try:
            examples = prepare_jtvae_examples(df, self.fragment_vocab, config=config)
        except Exception as exc:
            logger.warning("Failed to prepare JT-VAE examples for refresh: %s", exc)
            return
        dataset = JTVDataset(examples)
        if len(dataset) == 0:
            logger.debug("Generator refresh skipped: no valid examples.")
            return
        refresh_cfg = {
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-4,
            "kl_weight": 0.5,
            "property_weight": 0.0,
            "adj_weight": 1.0,
            "save_dir": self.results_dir / "generator_refresh",
        }
        refresh_cfg.update(self.generator_refresh_kwargs)
        refresh_cfg["epochs"] = int(refresh_cfg.get("epochs", 1))
        refresh_cfg["batch_size"] = int(refresh_cfg.get("batch_size", 16))
        refresh_cfg["lr"] = float(refresh_cfg.get("lr", 1e-4))
        refresh_cfg["kl_weight"] = float(refresh_cfg.get("kl_weight", 0.5))
        refresh_cfg["property_weight"] = float(refresh_cfg.get("property_weight", 0.0))
        refresh_cfg["adj_weight"] = float(refresh_cfg.get("adj_weight", 1.0))
        device = next(self.generator.parameters()).device
        save_dir = Path(refresh_cfg.pop("save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Refreshing generator on %d molecules for %d epochs (lr=%s)",
            len(dataset),
            refresh_cfg.get("epochs", 1),
            refresh_cfg.get("lr", 1e-4),
        )
        train_jtvae(
            self.generator,
            dataset,
            self.fragment_vocab,
            device=str(device),
            save_dir=str(save_dir),
            **refresh_cfg,
        )

    def _ensure_pool(self, min_size: int, cond: Optional[np.ndarray], assemble_kwargs: Optional[Dict]) -> int:
        if self.generator is None or not self.fragment_vocab:
            return 0
        generated = 0
        existing = set(self.pool["smiles"]).union(set(self.labelled["smiles"]))
        while len(self.pool) < min_size:
            samples = sample_conditional(
                self.generator,
                self.fragment_vocab,
                cond=cond,
                n_samples=self.config.generator_samples,
                assembler="beam",
                assemble_kwargs=assemble_kwargs or self.assemble_kwargs,
            )
            new_rows = []
            for sample in samples:
                smiles = sample.get("smiles")
                status = sample.get("status")
                if not smiles or smiles in existing:
                    continue
                if not self._passes_diversity(smiles):
                    logger.debug("Filtered out %s due to diversity threshold.", smiles)
                    continue
                existing.add(smiles)
                new_rows.append({"smiles": smiles, "assembly_status": status})
            if not new_rows:
                break
            self.pool = pd.concat([self.pool, pd.DataFrame(new_rows)], ignore_index=True)
            generated += len(new_rows)
        return generated

    def _featurize_pool(self) -> List:
        graphs = []
        valid_indices = []
        for idx, row in self.pool.iterrows():
            try:
                data = mol_to_graph(row["smiles"], y=None)
            except Exception as exc:
                logger.warning("Skipping invalid SMILES %s: %s", row["smiles"], exc)
                continue
            graphs.append(data)
            valid_indices.append(idx)
        return graphs, valid_indices

    def _predict_pool(self, graphs: List):
        mean, std, _ = self.surrogate.predict(graphs, batch_size=self.config.batch_size)
        return mean, std

    def _score_candidates(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        norm_mean, norm_std, mus, sigmas, directions = self._normalise_predictions(mean, std)
        best = self._current_best()
        norm_best = None
        if best is not None:
            norm_best = ((best * directions) - mus) / sigmas
        acq_cfg = self.config.acquisition
        cfg = AcquisitionConfig(
            kind=acq_cfg.kind,
            beta=acq_cfg.beta,
            xi=acq_cfg.xi,
            maximise=acq_cfg.maximise,
            weights=acq_cfg.weights,
        )
        if cfg.kind in {"pareto", "pareto_ucb"}:
            cfg.maximise = [True] * norm_mean.shape[1]
        scores = acquisition_score(norm_mean, norm_std, cfg, best_so_far=norm_best)
        return scores

    def _label_with_dft(self, selected: pd.DataFrame) -> pd.DataFrame:
        if self.dft is None:
            return selected
        jobs = [
            DFTJobSpec(smiles=row["smiles"], properties=list(self.config.target_columns))
            for _, row in selected.iterrows()
        ]
        ids = self.dft.submit_batch(jobs)
        results = []
        for job_id in ids:
            res = self.dft.fetch(job_id, block=True, poll_interval=1.0)
            results.append(res)
        for df_row, res in zip(selected.itertuples(index=True), results):
            for prop, value in res.properties.items():
                selected.at[df_row.Index, prop] = value
        return selected

    def _retrain_surrogate(self) -> None:
        if len(self.labelled) < len(self.config.target_columns) + 5:
            return
        train_df = self.labelled[["smiles", *self.config.target_columns]].dropna()
        if train_df.empty:
            return
        split = split_dataframe(train_df, val_fraction=0.1, test_fraction=0.0, seed=self.scheduler.iteration + 42)
        logger.info("Retraining surrogate on %d molecules", len(split.train))
        self.surrogate.fit(split.train, split.val)

    def run_iteration(
        self,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if self.scheduler.should_stop():
            raise RuntimeError("Maximum number of iterations reached.")

        if assemble_kwargs is None:
            assemble_kwargs = self.assemble_kwargs
        generated = self._ensure_pool(self.config.batch_size, cond, assemble_kwargs)
        graphs, valid_idx = self._featurize_pool()
        if not graphs:
            raise RuntimeError("No valid candidates in pool to evaluate.")

        mean, std = self._predict_pool(graphs)
        scores = self._score_candidates(mean, std)

        pool_slice = self.pool.iloc[valid_idx].copy()
        for i, name in enumerate(self.config.target_columns):
            pool_slice[f"pred_{name}"] = mean[:, i]
            pool_slice[f"pred_std_{name}"] = std[:, i]
        pool_slice["acquisition_score"] = scores
        iteration_idx = self.scheduler.iteration + 1
        self._save_diagnostics(pool_slice, iteration_idx)

        selected = (
            pool_slice.sort_values("acquisition_score", ascending=False)
            .head(self.config.batch_size)
            .copy()
        )
        self.pool = self.pool.drop(selected.index).reset_index(drop=True)

        labelled = self._label_with_dft(selected)
        labelled["iteration"] = self.scheduler.iteration + 1
        self.labelled = pd.concat([self.labelled, labelled], ignore_index=True)
        self.history.append(labelled)

        self.scheduler.step(num_labelled=len(labelled), num_generated=generated)

        if self.scheduler.should_retrain_surrogate():
            self._retrain_surrogate()

        if self.scheduler.should_refresh_generator():
            self._refresh_generator()

        return labelled

    def run(
        self,
        n_iterations: int,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> List[pd.DataFrame]:
        if assemble_kwargs is None:
            assemble_kwargs = {}
        merged_kwargs = {**self.assemble_kwargs, **assemble_kwargs}
        for _ in range(n_iterations):
            if self.scheduler.should_stop():
                break
            self.run_iteration(cond=cond, assemble_kwargs=merged_kwargs)
        return self.history

    def save_history(self) -> None:
        if not self.history:
            return
        path = self.results_dir / "active_learning_history.csv"
        pd.concat(self.history, ignore_index=True).to_csv(path, index=False)
        logger.info("Saved active learning history to %s", path)
