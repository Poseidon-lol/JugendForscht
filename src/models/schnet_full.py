"""echtes schnet surrogate mit pyg schnet"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet as PygSchNet

from src.utils.device import get_device
import random
import numpy as np


@dataclass
class RealSchNetConfig:
    hidden_channels: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: float = 10.0
    readout: str = "add"  # add | mean | max
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    epochs: int = 80
    patience: int = 10
    device: str = "auto"
    save_dir: Path = Path("models/surrogate_3d_full")
    n_models: int = 1
    seed: int = 1337


class RealSchNetModel(torch.nn.Module):
    """wrapper um pyg schnet mit multi target head"""

    def __init__(self, cfg: RealSchNetConfig, out_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        base = PygSchNet(
            hidden_channels=cfg.hidden_channels,
            num_filters=cfg.num_filters,
            num_interactions=cfg.num_interactions,
            num_gaussians=cfg.num_gaussians,
            cutoff=cfg.cutoff,
            readout=cfg.readout,
        )
        # Replace final linear layer to support arbitrary target dimensions.
        head_in = base.lin1.out_features
        base.lin2 = torch.nn.Linear(head_in, out_dim)
        self.model = base

    def forward(self, z, pos, batch=None):
        out = self.model(z, pos, batch)
        if out.dim() == 1:
            return out.view(-1, 1)
        return out


def _resolve_device(device_spec: str):
    dev = get_device(device_spec)
    # torch_scatter passt nicht zu directml also cpu fallback
    if dev.type == "directml":
        dev = get_device("cpu")
    return dev


def train_schnet_full(
    train_ds,
    val_ds=None,
    target_dim: int = 1,
    config: Optional[RealSchNetConfig] = None,
    *,
    save_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> Tuple[RealSchNetModel, List[float]]:
    cfg = config or RealSchNetConfig()
    device = _resolve_device(cfg.device)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    model = RealSchNetModel(cfg, out_dim=target_dim).to(device.target)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds is not None else None

    history: List[float] = []
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device.target)
            opt.zero_grad()
            pred = model(batch.z, batch.pos, getattr(batch, "batch", None))
            y = batch.y.view(pred.size(0), -1)
            loss = F.l1_loss(pred, y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
        train_loss = total / max(1, count)

        val_loss = None
        if val_loader is not None:
            model.eval()
            vtotal = 0.0
            vcount = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device.target)
                    pred = model(batch.z, batch.pos, getattr(batch, "batch", None))
                    y = batch.y.view(pred.size(0), -1)
                    loss = F.l1_loss(pred, y)
                    vtotal += loss.item() * y.size(0)
                    vcount += y.size(0)
            val_loss = vtotal / max(1, vcount)

        metric = val_loss if val_loss is not None else train_loss
        history.append(metric)

        if metric < best_loss:
            best_loss = metric
            bad_epochs = 0
            target_path = save_path if save_path is not None else (cfg.save_dir / "schnet_full.pt")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), target_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    return model, history


def load_schnet_full(path: Path, target_dim: int, cfg: Optional[RealSchNetConfig] = None) -> RealSchNetModel:
    cfg = cfg or RealSchNetConfig()
    model = RealSchNetModel(cfg, out_dim=target_dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def train_schnet_full_ensemble(
    train_ds,
    val_ds=None,
    target_dim: int = 1,
    config: Optional[RealSchNetConfig] = None,
) -> Tuple[List[RealSchNetModel], List[List[float]]]:
    cfg = config or RealSchNetConfig()
    models: List[RealSchNetModel] = []
    histories: List[List[float]] = []
    base_seed = getattr(cfg, "seed", 1337)
    n_models = max(1, int(getattr(cfg, "n_models", 1)))
    for idx in range(n_models):
        member_seed = base_seed + idx
        save_path = cfg.save_dir / f"schnet_full_member_{idx:02d}.pt"
        model, hist = train_schnet_full(
            train_ds,
            val_ds,
            target_dim=target_dim,
            config=cfg,
            save_path=save_path,
            seed=member_seed,
        )
        models.append(model)
        histories.append(hist)
    return models, histories
