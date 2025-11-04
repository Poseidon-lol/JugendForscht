# src/models/mpnn.py
"""
Surrogate MPNN for predicting molecular properties (HOMO/LUMO)
---------------------------------------------------------------
This module provides:
- A lightweight Message Passing Neural Network (MPNN) based on PyTorch Geometric
- Dataset wrapper utilities for PyG Data objects
- Training / evaluation loops (supports ensemble training for uncertainty)

Usage:
    from src.data.featurization import mol_to_graph
    from src.models.mpnn import MPModel, train_one, evaluate, train_ensemble

Requirements:
    torch, torch_geometric, scikit-learn, pandas, numpy

Notes:
- This is a starter implementation intended for extension.
- For production: add learning rate schedulers, advanced regularization, logging (MLflow),
  and better calibration for uncertainty estimates.
"""

import os
import math
import random
from typing import List, Tuple, Optional
import contextlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

# PyG imports
try:
    from torch_geometric.data import Data, Dataset as PyGDataset
    from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception as e:
    raise ImportError("This module requires torch_geometric. Install it before using MPModel.")

# local featurization (assumes file exists at src/data/featurization.py)
try:
    from src.data.featurization import mol_to_graph
except Exception:
    # fallback if running from different working dir
    from src.data.featurization import mol_to_graph

from src.utils.device import DeviceSpec, ensure_state_dict_on_cpu, get_device, move_to_device


# -----------------------------
# MPNN building blocks
# -----------------------------
class ResidualEdgeGatedMPNNLayer(MessagePassing):
    """Edge-gated message passing with residual connections and layer norm."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        self.msg_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        combined = torch.cat([x, out], dim=-1)
        residual = self.residual_mlp(combined)
        residual = self.dropout_layer(residual)
        return self.layer_norm(x + residual)

    def message(self, x_j, edge_attr):
        concat = torch.cat([x_j, edge_attr], dim=-1)
        gated = self.gate_proj(concat) * self.msg_proj(concat)
        return gated


class MPModel(nn.Module):
    def __init__(self, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 128,
                 num_message_layers: int = 3, readout_dim: int = 128, out_dim: int = 2,
                 dropout: float = 0.0, readout_type: str = "mlp", pooling: str = "mean"):
        """A small MPNN for regression.

        Args:
            node_in_dim: dimension of node feature vector
            edge_in_dim: dimension of edge feature vector
            hidden_dim: hidden dimension for message/update networks
            num_message_layers: how many message-passing steps
            readout_dim: dimension after pooling
            out_dim: number of regression targets (e.g., HOMO,LUMO)
            dropout: dropout probability applied after message layers (MC dropout support)
        """
        super().__init__()
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)
        self.pooling = pooling.lower()
        valid_poolings = {"mean", "sum", "max"}
        if self.pooling not in valid_poolings:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose from {valid_poolings}.")

        self.layers = nn.ModuleList(
            ResidualEdgeGatedMPNNLayer(hidden_dim, dropout=self.dropout) for _ in range(num_message_layers)
        )

        if readout_type.lower() == "mlp":
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, readout_dim),
                nn.ReLU(),
                nn.Linear(readout_dim, out_dim),
            )
        elif readout_type.lower() == "linear":
            self.readout = nn.Linear(hidden_dim, out_dim)
        else:
            raise ValueError("Unsupported readout_type '{}'. Use 'mlp' or 'linear'.".format(readout_type))

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, None
        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_norm(self.node_encoder(x))
        edge_attr = self.edge_norm(self.edge_encoder(edge_attr))

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # global pooling
        if self.pooling == "mean":
            g = global_mean_pool(x, batch)
        elif self.pooling == "sum":
            g = global_add_pool(x, batch)
        else:  # max
            g = global_max_pool(x, batch)
        out = self.readout(g)
        return out


# -----------------------------
# Dataset wrapper
# -----------------------------
class MoleculeDataset(PyGDataset):
    def __init__(self, dataframe, transform=None, pre_transform=None):
        """Expects a pandas DataFrame with columns: 'smiles' and target columns (e.g. 'HOMO','LUMO')."""
        super().__init__(None, transform, pre_transform)
        self.df = dataframe.reset_index(drop=True)
        # infer numeric target columns (exclude identifier / string columns)
        candidate_cols = [c for c in self.df.columns if c not in ("smiles", "id")]
        self.target_cols = [
            c for c in candidate_cols if pd.api.types.is_numeric_dtype(self.df[c])
        ]

    def len(self):
        return len(self.df)

    def get(self, idx):
        row = self.df.iloc[idx]
        smiles = row['smiles']
        y = row[self.target_cols].values.astype(float) if len(self.target_cols) > 0 else None
        data = mol_to_graph(smiles, y=y)
        # attach batch index when needed (PyG dataloader will set this)
        return data


# -----------------------------
# Training & evaluation utilities
# -----------------------------

def train_one(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: DeviceSpec | torch.device | str,
    *,
    loss_fn=None,
    grad_clip: float = 0.0,
    use_amp: bool = False,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
):
    device_spec = get_device(device)
    target = device_spec.target
    amp_enabled = bool(use_amp) and device_spec.supports_amp
    scaler = scaler if amp_enabled else None
    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else contextlib.nullcontext
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = move_to_device(batch, device_spec)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            pred = model(batch)
            y = batch.y.view(pred.size(0), -1).to(target)
            if loss_fn is None:
                loss = F.l1_loss(pred, y)  # MAE
            else:
                loss = loss_fn(pred, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * pred.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: DeviceSpec | torch.device | str,
             loss_fn=None, use_amp: bool = False) -> Tuple[float, np.ndarray, np.ndarray]:
    device_spec = get_device(device)
    target = device_spec.target
    amp_enabled = bool(use_amp) and device_spec.supports_amp
    autocast_ctx = torch.cuda.amp.autocast if amp_enabled else contextlib.nullcontext
    model.eval()
    preds = []
    trues = []
    loss_accum = 0.0
    n_total = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device_spec)
            with autocast_ctx():
                out = model(batch)
            preds.append(out.cpu().numpy())
            trues.append(batch.y.view(out.size(0), -1).cpu().numpy())
            if loss_fn is not None:
                loss_val = loss_fn(out, batch.y.view(out.size(0), -1).to(target))
                loss_accum += loss_val.item() * out.size(0)
                n_total += out.size(0)
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    mae = np.mean(np.abs(preds - trues))
    if loss_fn is not None and n_total > 0:
        mae = loss_accum / n_total
    return mae, preds, trues


# -----------------------------
# Ensemble training (deep ensembles for uncertainty)
# -----------------------------

def train_ensemble(df, model_save_dir: str, n_models: int = 5, epochs: int = 50, batch_size: int = 32,
                   lr: float = 1e-3, device: str = None, weight_decay: float = 0.0,
                   loss: str = 'mae', dropout: float = 0.0):
    """Train an ensemble of MPNNs and save them to disk.

    Args:
        df: pandas DataFrame with 'smiles' and target columns
        model_save_dir: directory to save checkpoints
        n_models: number of ensemble members
    """
    device_spec = get_device(device)
    os.makedirs(model_save_dir, exist_ok=True)

    # simple split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    train_ds = MoleculeDataset(train_df)
    val_ds = MoleculeDataset(val_df)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)

    node_dim = train_ds[0].x.size(1)
    edge_dim = train_ds[0].edge_attr.size(1)
    out_dim = train_ds[0].y.size(1)

    for i in range(n_models):
        torch.manual_seed(1000 + i)
        model = MPModel(node_in_dim=node_dim, edge_in_dim=edge_dim, out_dim=out_dim, dropout=dropout)
        model = model.to(device_spec.target)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if loss == 'mae':
            loss_fn = None
        elif loss == 'mse':
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss '{loss}'.")

        best_val = 1e9
        for epoch in range(1, epochs + 1):
            train_loss = train_one(model, train_loader, optimizer, device_spec, loss_fn=loss_fn, grad_clip=0.0)
            val_mae, _, _ = evaluate(model, val_loader, device_spec, loss_fn=loss_fn)
            if val_mae < best_val:
                best_val = val_mae
                torch.save(
                    ensure_state_dict_on_cpu(model, device_spec),
                    os.path.join(model_save_dir, f'mpnn_member_{i}.pt')
                )
            if epoch % 10 == 0 or epoch == 1:
                print(f"Model {i} Epoch {epoch:03d} train_loss={train_loss:.4f} val_mae={val_mae:.4f}")

    print(f"Ensemble training completed. Models saved to {model_save_dir}")


# -----------------------------
# Ensemble inference util
# -----------------------------

def ensemble_predict(model_dir: str, dataset, device: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load all model checkpoints in model_dir and return mean prediction and std (uncertainty).

    Returns:
        mean_pred: [N, out_dim], std_pred: [N, out_dim]
    """
    device_spec = get_device(device)
    ckpts = sorted([f for f in os.listdir(model_dir) if f.endswith('.pt')])
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No .pt files found in {model_dir}")

    loader = PyGDataLoader(dataset, batch_size=32, shuffle=False)
    preds_collect = []
    for ckpt in ckpts:
        model = MPModel(node_in_dim=dataset[0].x.size(1), edge_in_dim=dataset[0].edge_attr.size(1),
                        out_dim=dataset[0].y.size(1))
        state = torch.load(os.path.join(model_dir, ckpt), map_location=device_spec.map_location)
        model.load_state_dict(state)
        model = model.to(device_spec.target)
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, device_spec)
                out = model(batch)
                all_preds.append(out.cpu().numpy())
        preds_collect.append(np.vstack(all_preds))

    preds_stack = np.stack(preds_collect, axis=0)  # [ensemble, N, out_dim]
    mean_pred = preds_stack.mean(axis=0)
    std_pred = preds_stack.std(axis=0)
    return mean_pred, std_pred


# -----------------------------
# Quick demo / CLI
# -----------------------------
if __name__ == "__main__":
    import pandas as pd

    # small demo dataframe using a few SMILES
    demo = pd.DataFrame({
        'smiles': ['c1ccccc1', 'c1ccncc1', 'C1=CC=CC=C1O', 'c1ccoc1'],
        'HOMO': [-6.5, -6.2, -6.8, -6.0],
        'LUMO': [-2.5, -2.6, -2.3, -2.0]
    })

    # train a tiny ensemble (1 model, few epochs) for testing
    train_ensemble(demo, model_save_dir='./models_demo', n_models=1, epochs=5, batch_size=2, lr=1e-3)

    # load dataset and predict
    ds = MoleculeDataset(demo)
    mean, std = ensemble_predict('./models_demo', ds)
    print('Mean predictions:\n', mean)
    print('Std predictions:\n', std)
