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
                assemble_kwargs=assemble_kwargs,
            )
            new_rows = []
            for sample in samples:
                smiles = sample.get("smiles")
                if not smiles or smiles in existing:
                    continue
                existing.add(smiles)
                new_rows.append({"smiles": smiles})
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
        best = self._current_best()
        return acquisition_score(mean, std, self.config.acquisition, best_so_far=best)

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
            logger.info("Generator refresh triggered (hook not implemented).")

        return labelled

    def run(
        self,
        n_iterations: int,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> List[pd.DataFrame]:
        for _ in range(n_iterations):
            if self.scheduler.should_stop():
                break
            self.run_iteration(cond=cond, assemble_kwargs=assemble_kwargs)
        return self.history

    def save_history(self) -> None:
        if not self.history:
            return
        path = self.results_dir / "active_learning_history.csv"
        pd.concat(self.history, ignore_index=True).to_csv(path, index=False)
        logger.info("Saved active learning history to %s", path)







