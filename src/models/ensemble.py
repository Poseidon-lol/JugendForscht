"""Utilities for training and using surrogate model ensembles."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

# Ensure the project root (with src/) is on sys.path
import sys

PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:  # pragma: no cover - defensive
    raise RuntimeError("Could not locate project root containing src/")

from src.models.mpnn import MPModel, MoleculeDataset, evaluate, train_one
from src.utils.device import DeviceSpec, ensure_state_dict_on_cpu, get_device, move_to_device

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "ensemble_config.json"
META_FILENAME = "ensemble_meta.json"
MODEL_PATTERN = "mpnn_member_{:02d}.pt"


@dataclass
class EnsembleConfig:
    """Configuration options for surrogate ensemble training."""

    n_models: int = 5
    epochs: int = 120
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    loss: str = "mae"
    dropout: float = 0.0
    hidden_dim: int = 128
    message_layers: int = 3
    readout_dim: int = 128
    save_dir: Path = Path("models/surrogate")
    device: Optional[str] = "cpu"
    seed: int = 1337

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["save_dir"] = str(self.save_dir)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "EnsembleConfig":
        payload = dict(data)
        if "save_dir" in payload and payload["save_dir"] is not None:
            payload["save_dir"] = Path(payload["save_dir"])
        return cls(**payload)  # type: ignore[arg-type]


class SurrogateEnsemble:
    """High-level wrapper around multiple MPNN surrogate models."""

    def __init__(self, config: EnsembleConfig, *, device: Optional[str | torch.device | DeviceSpec] = None) -> None:
        self.config = config
        chosen_device = device if device is not None else config.device
        self.device: DeviceSpec = get_device(chosen_device)
        self.target_columns: Sequence[str] = ()
        self.node_dim: Optional[int] = None
        self.edge_dim: Optional[int] = None
        self.out_dim: Optional[int] = None

        self._members: List[torch.nn.Module] = []
        self._member_states: List[Dict[str, torch.Tensor]] = []
        self._member_scores: List[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, train_df, val_df=None) -> None:
        """Train the ensemble on train_df (and optional val_df)."""

        import pandas as pd

        if not isinstance(train_df, pd.DataFrame):
            raise TypeError("train_df must be a pandas DataFrame.")

        if val_df is not None and not isinstance(val_df, pd.DataFrame):
            raise TypeError("val_df must be a pandas DataFrame or None.")

        if train_df.empty:
            raise ValueError("train_df is empty.")

        target_cols = [c for c in train_df.columns if c not in {"smiles", "id"}]
        if not target_cols:
            raise ValueError("train_df must contain at least one target column.")
        self.target_columns = tuple(target_cols)

        train_dataset = MoleculeDataset(train_df)
        val_dataset = MoleculeDataset(val_df) if val_df is not None and not val_df.empty else None

        train_loader = PyGDataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = (
            PyGDataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            if val_dataset is not None
            else None
        )

        sample = train_dataset[0]
        self.node_dim = sample.x.size(1)
        self.edge_dim = sample.edge_attr.size(1)
        self.out_dim = sample.y.size(1) if sample.y is not None else len(self.target_columns)

        self._members.clear()
        self._member_states.clear()
        self._member_scores.clear()

        base_seed = self.config.seed

        for idx in range(self.config.n_models):
            logger.info("Training ensemble member %d/%d", idx + 1, self.config.n_models)
            self._set_all_seeds(base_seed + idx)

            model = self._build_model().to(self.device.target)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
            loss_fn = self._resolve_loss()

            best_state: Optional[Dict[str, torch.Tensor]] = None
            best_val = float("inf")
            epochs_without_improve = 0

            for epoch in range(1, self.config.epochs + 1):
                train_loss = train_one(model, train_loader, optimizer, self.device, loss_fn=loss_fn)
                val_metric = train_loss
                if val_loader is not None:
                    val_metric, _, _ = evaluate(model, val_loader, self.device, loss_fn=loss_fn)

                improved = val_metric < best_val - 1e-6
                if improved or best_state is None:
                    best_val = val_metric
                    best_state = ensure_state_dict_on_cpu(model, self.device)
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1

                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        "Member %d epoch %03d train_loss=%.4f val_metric=%.4f",
                        idx + 1,
                        epoch,
                        train_loss,
                        val_metric,
                    )

                if self.config.patience and epochs_without_improve >= self.config.patience:
                    logger.info(
                        "Member %d early stopped after %d epochs (best_val=%.4f)",
                        idx + 1,
                        epoch,
                        best_val,
                    )
                    break

            if best_state is None:
                best_state = ensure_state_dict_on_cpu(model, self.device)

            member = self._build_model().to(self.device.target)
            member.load_state_dict(best_state)
            member.eval()

            self._members.append(member)
            self._member_states.append(best_state)
            self._member_scores.append(best_val)

        logger.info("Finished training %d surrogate members.", len(self._members))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_all(self, *, save_dir: Optional[Path] = None) -> None:
        """Persist ensemble configuration, metadata, and weights."""

        target_dir = Path(save_dir) if save_dir is not None else self.config.save_dir
        target_dir = target_dir.resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        self.config.save_dir = target_dir

        config_path = target_dir / CONFIG_FILENAME
        with config_path.open("w", encoding="utf-8") as fh:
            json.dump(self.config.to_dict(), fh, indent=2)

        meta = {
            "target_columns": list(self.target_columns),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "out_dim": self.out_dim,
            "member_scores": self._member_scores,
        }
        meta_path = target_dir / META_FILENAME
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        if not self._member_states:
            raise RuntimeError("No trained members to save. Did you call fit()?")

        for idx, state in enumerate(self._member_states):
            ckpt_path = target_dir / MODEL_PATTERN.format(idx)
            torch.save(state, ckpt_path)
            logger.info("Saved ensemble member %d to %s", idx, ckpt_path)

    @classmethod
    def from_directory(
        cls, directory: Path, *, device: Optional[str | torch.device | DeviceSpec] = None
    ) -> "SurrogateEnsemble":
        """Load ensemble configuration and weights from a directory."""

        directory = Path(directory).resolve()
        config_path = directory / CONFIG_FILENAME
        if not config_path.exists():
            raise FileNotFoundError(f"Missing ensemble configuration file at {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            cfg_dict = json.load(fh)

        config = EnsembleConfig.from_dict(cfg_dict)
        config.save_dir = directory

        ensemble = cls(config, device=device)
        ensemble._load_metadata(directory)
        ensemble._load_members_from_directory(directory)
        return ensemble

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(
        self,
        graphs: Sequence,
        *,
        batch_size: Optional[int] = None,
        return_member_predictions: bool = False,
    ):
        """Predict mean/std across ensemble for PyG Data objects.

        When ``return_member_predictions`` is ``True`` the third element is the
        raw predictions with shape ``[n_members, n_samples, out_dim]``.
        """

        if not self._members:
            raise RuntimeError("Surrogate ensemble has no loaded members.")

        effective_batch = batch_size or self.config.batch_size
        preds_collect: List[np.ndarray] = []

        for member in self._members:
            member.eval()
            loader = PyGDataLoader(graphs, batch_size=effective_batch, shuffle=False)
            member_preds: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    batch = move_to_device(batch, self.device)
                    out = member(batch)
                    member_preds.append(out.detach().cpu().numpy())
            preds_collect.append(np.concatenate(member_preds, axis=0))

        preds_stack = np.stack(preds_collect, axis=0)
        mean = preds_stack.mean(axis=0)
        std = preds_stack.std(axis=0)
        if return_member_predictions:
            return mean, std, preds_stack
        return mean, std

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_model(self) -> MPModel:
        if self.node_dim is None or self.edge_dim is None or self.out_dim is None:
            raise RuntimeError("Model dimensions unknown. Train the ensemble or load metadata first.")
        return MPModel(
            node_in_dim=self.node_dim,
            edge_in_dim=self.edge_dim,
            hidden_dim=self.config.hidden_dim,
            num_message_layers=self.config.message_layers,
            readout_dim=self.config.readout_dim,
            out_dim=self.out_dim,
            dropout=self.config.dropout,
        )

    def _resolve_loss(self):
        if self.config.loss.lower() == "mae":
            return None
        if self.config.loss.lower() == "mse":
            return torch.nn.MSELoss()
        raise ValueError(f"Unsupported loss function '{self.config.loss}'.")

    @staticmethod
    def _set_all_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_metadata(self, directory: Path) -> None:
        meta_path = directory / META_FILENAME
        if not meta_path.exists():
            logger.warning("No metadata file found at %s. Will attempt to infer dimensions.", meta_path)
            return
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self.target_columns = tuple(meta.get("target_columns", ()))
        self.node_dim = meta.get("node_dim")
        self.edge_dim = meta.get("edge_dim")
        self.out_dim = meta.get("out_dim")
        self._member_scores = meta.get("member_scores", [])

    def _load_members_from_directory(self, directory: Path) -> None:
        ckpts = sorted(directory.glob("mpnn_member_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No ensemble checkpoints found in {directory}")

        self._members.clear()
        self._member_states.clear()

        for idx, ckpt in enumerate(ckpts):
            state = torch.load(ckpt, map_location="cpu")
            self._member_states.append(state)

            if self.node_dim is None or self.edge_dim is None or self.out_dim is None:
                self._infer_dims_from_state(state)

            member = self._build_model().to(self.device.target)
            member.load_state_dict(state)
            member.eval()
            self._members.append(member)
            logger.info("Loaded ensemble member %d from %s", idx, ckpt)

    def _infer_dims_from_state(self, state: Dict[str, torch.Tensor]) -> None:
        if self.node_dim is None and "node_encoder.weight" in state:
            self.node_dim = state["node_encoder.weight"].shape[1]
        if self.edge_dim is None and "edge_encoder.weight" in state:
            self.edge_dim = state["edge_encoder.weight"].shape[1]
        if self.out_dim is None and "readout.2.weight" in state:
            self.out_dim = state["readout.2.weight"].shape[0]
