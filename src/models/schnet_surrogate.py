"""leichter 3d surrogate ohne schwere schnet deps"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool


@dataclass
class SchNetConfig:
    hidden_channels: int = 128
    num_filters: int = 128  # legacy feld unused
    num_interactions: int = 6  # legacy feld unused
    num_gaussians: int = 50  # legacy feld unused
    cutoff: float = 10.0  # legacy feld unused
    readout: str = "mean"
    num_embeddings: int = 120  # max atomnummer fuer embedding
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 16
    epochs: int = 80
    patience: int = 10
    device: str = "cpu"  # auto cpu cuda oder explizit
    save_dir: Path = Path("models/surrogate_3d")


class SchNetModel(torch.nn.Module):
    """simpler point cloud encoder fuer 3d molekuele"""

    def __init__(self, cfg: SchNetConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.atom_emb = nn.Embedding(cfg.num_embeddings, cfg.hidden_channels)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, cfg.hidden_channels),
            nn.SiLU(),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_channels, cfg.hidden_channels),
            nn.SiLU(),
        )
        self.head = nn.Linear(cfg.hidden_channels, out_dim)

    def forward(self, z, pos, batch=None, mask=None):
        # clip embedding range falls exotische atome
        z_clamped = torch.clamp(z, max=self.cfg.num_embeddings - 1)
        h = self.atom_emb(z_clamped) + self.pos_mlp(pos)
        h = self.encoder(h)
        if mask is not None:
            h = h * mask.unsqueeze(-1).float()
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        pooled = global_mean_pool(h, batch)
        return self.head(pooled)


def train_schnet(
    train_ds,
    val_ds=None,
    target_dim: int = 1,
    config: Optional[SchNetConfig] = None,
) -> Tuple[SchNetModel, List[float]]:
    cfg = config or SchNetConfig()
    device = _resolve_device(cfg.device)
    model = _build_schnet(cfg, target_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss = float("inf")
    bad_epochs = 0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size) if val_ds is not None else None
    history: List[float] = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch.z, batch.pos, getattr(batch, "batch", None))
            loss = torch.nn.functional.l1_loss(pred, batch.y)
            loss.backward()
            opt.step()
            total += loss.item() * batch.y.size(0)
            count += batch.y.size(0)
        train_loss = total / max(1, count)
        val_loss = None
        if val_loader is not None:
            model.eval()
            vtotal = 0.0
            vcount = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch.z, batch.pos, getattr(batch, "batch", None))
                    loss = torch.nn.functional.l1_loss(pred, batch.y)
                    vtotal += loss.item() * batch.y.size(0)
                    vcount += batch.y.size(0)
            val_loss = vtotal / max(1, vcount)
        history.append(train_loss if val_loss is None else val_loss)
        metric = val_loss if val_loss is not None else train_loss
        if metric < best_loss:
            best_loss = metric
            bad_epochs = 0
            cfg.save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.save_dir / "schnet.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    return model, history


def load_schnet(path: Path, target_dim: int, cfg: Optional[SchNetConfig] = None) -> SchNetModel:
    cfg = cfg or SchNetConfig()
    model = _build_schnet(cfg, target_dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    return model


def _build_schnet(cfg: SchNetConfig, target_dim: int) -> SchNetModel:
    return SchNetModel(cfg, target_dim)


def _resolve_device(device_spec: str | torch.device) -> torch.device:
    if isinstance(device_spec, torch.device):
        return device_spec
    if isinstance(device_spec, str) and device_spec.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)
