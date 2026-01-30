"""
Simple conditional 3D VAE for generating atom coordinates given atomic numbers.
Fixed-size padding to max_atoms; masks handled during loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class VAE3DConfig:
    max_atoms: int
    z_dim: int = 64
    hidden_dim: int = 256
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 50
    patience: int = 5
    device: str = "auto"  # "auto" selects CUDA if available
    save_path: str = "models/generator_3d/vae3d.pt"


class VAE3D(nn.Module):
    def __init__(self, max_atoms: int, hidden_dim: int = 256, z_dim: int = 64, num_atom_types: int = 120):
        super().__init__()
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * max_atoms + 3 * max_atoms, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + hidden_dim * max_atoms, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * max_atoms),
        )

    def encode(self, atom_z: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor):
        # atom_z: [B, max_atoms], pos: [B, max_atoms, 3], mask: [B, max_atoms]
        emb = self.atom_embed(atom_z) * mask.unsqueeze(-1)  # [B, max_atoms, hidden]
        flat = torch.cat([emb.reshape(emb.size(0), -1), pos.reshape(pos.size(0), -1)], dim=-1)
        h = self.encoder(flat)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, atom_z: torch.Tensor, z: torch.Tensor):
        emb = self.atom_embed(atom_z).reshape(atom_z.size(0), -1)
        dec_in = torch.cat([emb, z], dim=-1)
        coords = self.decoder(dec_in)
        coords = coords.view(atom_z.size(0), self.max_atoms, 3)
        return coords

    def forward(self, atom_z: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor):
        mu, logvar = self.encode(atom_z, pos, mask)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(atom_z, z)
        return recon, mu, logvar


def vae3d_loss(recon_pos, true_pos, mask, mu, logvar):
    # mask over atoms
    mse = ((recon_pos - true_pos) ** 2).sum(dim=-1) * mask
    recon_loss = mse.sum() / mask.sum().clamp(min=1.0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return recon_loss + 1e-3 * kl, recon_loss, kl


def train_vae3d(dataset, config: VAE3DConfig) -> VAE3D:
    device = _resolve_device(config.device)
    model = VAE3D(max_atoms=config.max_atoms, hidden_dim=config.hidden_dim, z_dim=config.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    best_loss = float("inf")
    bad = 0
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            atom_z = batch["z"].to(device)
            pos = batch["pos"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad()
            recon, mu, logvar = model(atom_z, pos, mask)
            loss, recon_l, kl = vae3d_loss(recon, pos, mask, mu, logvar)
            loss.backward()
            opt.step()
            total += loss.item() * atom_z.size(0)
            count += atom_z.size(0)
        epoch_loss = total / max(1, count)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            bad = 0
            Path(config.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.save_path)
        else:
            bad += 1
            if bad >= config.patience:
                break
    return model


def sample_vae3d(model: VAE3D, atom_z_template: torch.Tensor, mask: torch.Tensor, device: str | torch.device = "cpu"):
    model.eval()
    device_resolved = _resolve_device(device)
    atom_z = atom_z_template.to(device_resolved)
    mask = mask.to(device_resolved)
    mu = torch.zeros(atom_z.size(0), model.z_dim, device=device_resolved)
    logvar = torch.zeros_like(mu)
    z = model.reparameterize(mu, logvar)
    with torch.no_grad():
        coords = model.decode(atom_z, z)
    coords = coords * mask.unsqueeze(-1)
    return coords


def _resolve_device(device_spec: str | torch.device) -> torch.device:
    if isinstance(device_spec, torch.device):
        return device_spec
    if isinstance(device_spec, str) and device_spec.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)
