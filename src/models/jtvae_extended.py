# src/models/jtvae_extended.py
"""
Simplified, GPU-ready conditional JT-VAE scaffold for molecular generation
------------------------------------------------------------------------
Goal:
- Provide a production-oriented, PyTorch/PyG-compatible skeleton of a
  Junction-Tree VAE (JT-VAE)-style generator that is conditional on target
  properties (e.g., HOMO, LUMO for organic semiconductors).

Disclaimer:
- A full, research-grade JT-VAE (as in Jin et al. ICML'18) includes complex
  routines for valid tree assembly, subgraph matching and discrete decoding
  steps. Implementing every detail robustly would be lengthy; here I provide
  a complete, GPU-capable scaffold with working graph encoders/decoders,
  conditional latent handling, training loop and detailed TODOs where
  project-specific research code must be filled in (e.g., tree assembly).

Features implemented:
- RDKit-based fragmentation to obtain candidate building blocks (rings, scaffolds)
- PyG Message-Passing encoders for both junction-tree nodes (fragments)
  and original molecular graph
- Conditional latent concatenation for decoder conditioning
- Sampling utilities to produce candidate molecules from latent + cond
- Training loop (reconstruction + KL) and checkpoints

What you still need to add/verify for research use:
- The decoder's exact chemical-validity enforcing assembly (subgraph matching),
  and the loss terms specific to JT-VAE (tree reconstruction loss, assembly loss)
- Advanced scheduling, beam search or MCTS for decoding assembly
- Optional chemically-aware priors and valence checks

Requirements:
  torch, torch_geometric, rdkit, numpy

Usage (overview):
  from src.models.jtvae_extended import JTVAE, train_jtvae, sample_conditional

"""

import os
import math
import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    from torch_geometric.data import Data
    from torch_geometric.nn import MessagePassing, global_mean_pool
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:
    raise ImportError("torch_geometric is required for jtvae_extended module")

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

from src.utils.device import ensure_state_dict_on_cpu, get_device, move_to_device

# -------------------------
# Fragmentation utilities (very simplified)
# -------------------------

def extract_fragments(smiles: str) -> List[str]:
    """Extract ring fragments / Murcko scaffolds as candidate-junction-nodes.

    This is a simplified fragmenter: for production replace with more robust
    ring-decomposition + BRICS / retrosynthetic fragmentation.
    """
    if not RDKit_AVAILABLE:
        raise RuntimeError("RDKit required for fragmentation")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    frags = set()
    # use ring info
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        atom_idxs = list(ring)
        sub = Chem.PathToSubmol(mol, atom_idxs)
        s = Chem.MolToSmiles(sub)
        frags.add(s)
    # add Murcko scaffold
    try:
        ms = rdMolDescriptors.CalcMurckoScaffoldSmiles(mol)
        frags.add(ms)
    except Exception:
        pass
    return [f for f in frags if len(f) > 0]

# -------------------------
# Simple GNN building blocks
# -------------------------
class GNNLayer(MessagePassing):
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_dim, out_dim)
        self.msg = nn.Linear(in_dim + in_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [N, in_dim]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        m = torch.cat([x_i, x_j], dim=-1)
        return F.relu(self.msg(m))

    def update(self, aggr_out):
        return F.relu(self.lin(aggr_out))


class SimpleGNNEncoder(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=128, n_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden_dim)
        self.layers = nn.ModuleList([GNNLayer(hidden_dim, hidden_dim) for _ in range(n_layers)])

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, edge_index)
        if batch is None:
            # assume single graph -> global mean pool
            return h.mean(dim=0, keepdim=True)
        else:
            return global_mean_pool(h, batch)

# -------------------------
# JT-VAE core classes (simplified)
# -------------------------
class JTEncoder(nn.Module):
    """Encodes both junction tree (fragment-level) and molecular graph into latents."""
    def __init__(self, node_feat_dim, hidden_dim=128, z_dim=56, cond_dim=0):
        super().__init__()
        # tree-level encoder (fragments as nodes)
        self.tree_encoder = SimpleGNNEncoder(node_feat_dim, hidden_dim)
        # graph-level encoder (full molecule)
        self.graph_encoder = SimpleGNNEncoder(node_feat_dim, hidden_dim)
        # combine
        self.fc_mu = nn.Linear(2 * hidden_dim + cond_dim, z_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim + cond_dim, z_dim)
        self.cond_dim = cond_dim

    def forward(self, tree_x, tree_edge_index, graph_x, graph_edge_index, batch_tree=None, batch_graph=None, cond=None):
        tvec = self.tree_encoder(tree_x, tree_edge_index, batch_tree)  # [B, H]
        gvec = self.graph_encoder(graph_x, graph_edge_index, batch_graph)
        vec = torch.cat([tvec, gvec], dim=-1)
        if cond is not None:
            vec = torch.cat([vec, cond], dim=-1)
        mu = self.fc_mu(vec)
        logvar = self.fc_logvar(vec)
        return mu, logvar


class JTDecoder(nn.Module):
    """Decoder skeleton: reconstructs junction tree and then assembles graph.

    NOTE: The assembly step is non-trivial. This decoder returns a reconstructed
    probability over fragments and a placeholder assembly routine. Replace with
    rigorous assembly logic for production.
    """
    def __init__(self, fragment_vocab_size, z_dim=56, hidden_dim=128, cond_dim=0):
        super().__init__()
        self.z_to_hidden = nn.Linear(z_dim + cond_dim, hidden_dim)
        # predict fragment logits for upto K nodes
        self.frag_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fragment_vocab_size)
        )
        # optional: project to atom-level reconstruction (placeholder)
        self.node_decoder = nn.Linear(hidden_dim, 32)  # e.g. atom feature logits

    def forward(self, z, max_tree_nodes=12, cond=None):
        # z: [B, z_dim]
        if cond is not None:
            z = torch.cat([z, cond], dim=-1)
        h = F.relu(self.z_to_hidden(z))
        # predict fragment probabilities per tree position
        frags_logits = []
        for _ in range(max_tree_nodes):
            frags_logits.append(self.frag_predictor(h))
        # frags_logits: list of [B, V] -> stack -> [B, max_nodes, V]
        frags_logits = torch.stack(frags_logits, dim=1)
        # placeholder for assembly: return fragment logits and node-level features
        node_feats = self.node_decoder(h)
        return frags_logits, node_feats


class JTVAE(nn.Module):
    def __init__(self, node_feat_dim, fragment_vocab_size, z_dim=56, hidden_dim=128, cond_dim=0):
        super().__init__()
        self.encoder = JTEncoder(node_feat_dim=node_feat_dim, hidden_dim=hidden_dim, z_dim=z_dim, cond_dim=cond_dim)
        self.decoder = JTDecoder(fragment_vocab_size=fragment_vocab_size, z_dim=z_dim, hidden_dim=hidden_dim, cond_dim=cond_dim)
        self.z_dim = z_dim
        self.cond_dim = cond_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, tree_x, tree_edge_index, graph_x, graph_edge_index, batch_tree=None, batch_graph=None, cond=None):
        mu, logvar = self.encoder(tree_x, tree_edge_index, graph_x, graph_edge_index, batch_tree, batch_graph, cond=cond)
        z = self.reparameterize(mu, logvar)
        frags_logits, node_feats = self.decoder(z, cond=cond)
        return frags_logits, node_feats, mu, logvar

    def sample(self, n_samples=32, cond=None, max_tree_nodes=12, fragment_idx_to_smiles=None, device=None):
        device_spec = get_device(device)
        target = device_spec.target
        z = torch.randn(n_samples, self.z_dim, device=target)
        cond_t = None
        if cond is not None:
            if torch.is_tensor(cond):
                cond_t = cond
            else:
                cond_np = cond if isinstance(cond, np.ndarray) else np.asarray(cond, dtype=np.float32)
                cond_t = torch.from_numpy(np.atleast_1d(cond_np)).float()
            if cond_t.dim() == 0:
                cond_t = cond_t.view(1, 1)
            elif cond_t.dim() == 1:
                cond_t = cond_t.unsqueeze(0)
            cond_t = cond_t.repeat(n_samples, 1)
            cond_t = cond_t.to(target)
        frags_logits, node_feats = self.decoder(z, max_tree_nodes=max_tree_nodes, cond=cond_t)
        # convert top-k fragment indices into SMILES via fragment_idx_to_smiles dict
        samples = []
        probs = F.softmax(frags_logits, dim=-1)
        top_idx = torch.argmax(probs, dim=-1)  # [B, max_nodes]
        for b in range(n_samples):
            frag_idxs = top_idx[b].cpu().tolist()
            # map to smiles
            if fragment_idx_to_smiles is None:
                # return raw fragment idx list
                samples.append(frag_idxs)
            else:
                frag_smiles = [fragment_idx_to_smiles.get(i, '') for i in frag_idxs if fragment_idx_to_smiles.get(i, '')]
                # TODO: assembly - naive concatenation (placeholder)
                mol_smi = ''.join(frag_smiles)
                samples.append(mol_smi)
        return samples

# -------------------------
# Loss and training utilities
# -------------------------

def jtvae_loss(frags_logits, node_feats, mu, logvar, target_frag_idxs=None, beta=0.5):
    """Compute simplified JT-VAE loss: fragment reconstruction + KL

    - frags_logits: [B, max_nodes, V]
    - target_frag_idxs: [B, max_nodes] (optional)
    """
    recon_loss = 0.0
    if target_frag_idxs is not None:
        B = frags_logits.size(0)
        V = frags_logits.size(-1)
        # flatten
        logits = frags_logits.view(-1, V)
        targets = target_frag_idxs.view(-1)
        recon_loss = F.cross_entropy(logits, targets, ignore_index=-1)
    else:
        recon_loss = torch.tensor(0.0, device=mu.device)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


def train_jtvae(model: JTVAE, dataset, fragment_vocab: Dict[int,str], device: str = None,
                epochs: int = 100, batch_size: int = 16, lr: float = 1e-3, save_dir: str = './jtvae_models'):
    os.makedirs(save_dir, exist_ok=True)
    device_spec = get_device(device)
    model.to(device_spec.target)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # dataset should provide precomputed: tree_x, tree_edge_index, graph_x, graph_edge_index, target_frag_idxs, cond
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            # batch must contain fields as described above
            batch = move_to_device(batch, device_spec)
            cond = batch.cond if hasattr(batch, 'cond') else None
            target_frag_idxs = batch.target_frag_idxs if hasattr(batch, 'target_frag_idxs') else None
            frags_logits, node_feats, mu, logvar = model(batch.tree_x, batch.tree_edge_index, batch.graph_x, batch.graph_edge_index,
                                                         batch_tree=batch.tree_batch if hasattr(batch, 'tree_batch') else None,
                                                         batch_graph=batch.batch if hasattr(batch, 'batch') else None,
                                                         cond=cond)
            loss, recon, kl = jtvae_loss(frags_logits, node_feats, mu, logvar, target_frag_idxs=target_frag_idxs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.num_graphs
        print(f"Epoch {epoch:03d} Loss: {epoch_loss/len(dataset):.4f}")
        torch.save(ensure_state_dict_on_cpu(model, device_spec), os.path.join(save_dir, f'jtvae_epoch_{epoch}.pt'))
    return model


def sample_conditional(
    model: JTVAE,
    fragment_vocab: Dict[str, int],
    *,
    cond: Optional[np.ndarray] = None,
    n_samples: int = 32,
    assembler: str = "beam",
    assemble_kwargs: Optional[Dict] = None,
) -> List[Dict[str, str]]:
    """Convenience wrapper that converts JT-VAE samples into dicts with SMILES strings.

    Parameters
    ----------
    model:
        Trained JTVAE instance.
    fragment_vocab:
        Mapping from fragment SMILES to integer indices.
    cond:
        Optional conditioning vector (e.g. normalised properties).
    n_samples:
        Number of candidate molecules to sample.
    assembler:
        Currently unused placeholder to keep API compatibility with future assemblers.
    assemble_kwargs:
        Optional overrides, e.g. ``{"max_tree_nodes": 16}``.
    """

    del assembler  # placeholder for compatibility

    if assemble_kwargs is None:
        assemble_kwargs = {}

    max_tree_nodes = assemble_kwargs.get("max_tree_nodes", 12)
    idx_to_frag = {idx: frag for frag, idx in fragment_vocab.items()}

    raw_samples = model.sample(
        n_samples=n_samples,
        cond=cond,
        max_tree_nodes=max_tree_nodes,
        fragment_idx_to_smiles=idx_to_frag,
    )

    formatted: List[Dict[str, str]] = []
    for entry in raw_samples:
        if isinstance(entry, str):
            smiles = entry
        elif isinstance(entry, (list, tuple)):
            smiles = "".join(entry)
        else:
            smiles = str(entry)
        formatted.append({"smiles": smiles})
    return formatted

# -------------------------
# Dataset adapter for JT-VAE
# -------------------------
class JTVDataset(torch.utils.data.Dataset):
    """Dataset should provide per-example:
    - tree_x, tree_edge_index, tree_batch (fragment-level graphs)
    - graph_x, graph_edge_index, batch (full molecule graph)
    - target_frag_idxs (indices into fragment vocab)
    - cond (conditioning vector, e.g., normalized HOMO/LUMO)

    You'll need a preprocessing routine to build fragment vocab and per-molecule
    fragment index arrays. This class assumes preprocessing already performed.
    """
    def __init__(self, examples: List[Dict]):
        super().__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        # return a simple dict-like object; the training loop expects attributes
        class B:
            pass
        b = B()
        b.tree_x = ex['tree_x']
        b.tree_edge_index = ex['tree_edge_index']
        b.graph_x = ex['graph_x']
        b.graph_edge_index = ex['graph_edge_index']
        b.target_frag_idxs = ex.get('target_frag_idxs', None)
        b.cond = ex.get('cond', None)
        b.num_graphs = 1
        return b

# -------------------------
# Quick usage notes
# -------------------------
# Preprocessing required:
# 1) Build fragment vocabulary across dataset -> fragment_idx mapping
# 2) For each molecule: extract fragments, map to indices -> target_frag_idxs
# 3) Create fragment-level graph (nodes = fragments, edges = adjacency between fragments)
# 4) Create atom-level graph (nodes = atoms, edges=bonds) features for graph_x
# 5) Create conditioning vector (e.g., normalized HOMO/LUMO) per example

# I recommend building a preprocessing script 'src/data/jt_preprocess.py' to
# create these example dicts and save them as torch files for fast loading.

if __name__ == '__main__':
    print('JT-VAE extended scaffold created.\nReminder: replace placeholder assembly with robust JT assembly code for chemical validity.')


