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
import logging
from typing import Dict, List, Optional, Tuple

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

def _load_mol_no_kekulize(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    try:
        Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops)
    except Exception as exc:  # pragma: no cover - RDKit warnings only
        logging.getLogger(__name__).debug("SanitizeMol skipped kekulize for %s: %s", smiles, exc)
    return mol


def extract_fragments(smiles: str) -> List[str]:
    """Extract ring fragments / Murcko scaffolds as candidate-junction-nodes.

    This is a simplified fragmenter: for production replace with more robust
    ring-decomposition + BRICS / retrosynthetic fragmentation.
    """
    if not RDKit_AVAILABLE:
        raise RuntimeError("RDKit required for fragmentation")
    mol = _load_mol_no_kekulize(smiles)
    if mol is None:
        return []
    frags = set()
    try:
        ri = mol.GetRingInfo()
        for ring in ri.AtomRings():
            atom_idxs = list(ring)
            try:
                sub = Chem.PathToSubmol(mol, atom_idxs)
                if sub is None:
                    continue
                s = Chem.MolToSmiles(sub, canonical=True, kekuleSmiles=False)
                if s:
                    frags.add(s)
            except Exception:
                continue
    except Exception:
        logging.getLogger(__name__).debug("Ring extraction failed for %s", smiles)
    # add Murcko scaffold
    try:
        ms = rdMolDescriptors.CalcMurckoScaffoldSmiles(mol)
        if ms:
            frags.add(ms)
    except Exception:
        logging.getLogger(__name__).debug("Murcko scaffold failed for %s", smiles)
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
        if isinstance(batch, torch.Tensor) and (batch.device.type == "privateuseone" or h.device.type == "privateuseone"):
            pooled = global_mean_pool(h.to("cpu"), batch.to("cpu"))
            return pooled.to(h.device)
        return global_mean_pool(h, batch)

# -------------------------
# JT-VAE core classes (simplified)
# -------------------------
class JTEncoder(nn.Module):
    """Encodes both junction tree (fragment-level) and molecular graph into latents."""

    def __init__(self, tree_feat_dim, graph_feat_dim, hidden_dim=128, z_dim=56, cond_dim=0):
        super().__init__()
        # tree-level encoder (fragments as nodes)
        self.tree_encoder = SimpleGNNEncoder(tree_feat_dim, hidden_dim)
        # graph-level encoder (full molecule)
        self.graph_encoder = SimpleGNNEncoder(graph_feat_dim, hidden_dim)
        # combine
        self.fc_mu = nn.Linear(2 * hidden_dim + cond_dim, z_dim)
        self.fc_logvar = nn.Linear(2 * hidden_dim + cond_dim, z_dim)
        self.cond_dim = cond_dim

    def forward(
        self,
        tree_x,
        tree_edge_index,
        graph_x,
        graph_edge_index,
        batch_tree=None,
        batch_graph=None,
        cond=None,
    ):
        tvec = self.tree_encoder(tree_x, tree_edge_index, batch_tree)  # [B, H]
        gvec = self.graph_encoder(graph_x, graph_edge_index, batch_graph)
        fused = torch.cat([tvec, gvec], dim=-1)
        vec = fused
        if cond is not None:
            vec = torch.cat([vec, cond], dim=-1)
        mu = self.fc_mu(vec)
        logvar = self.fc_logvar(vec)
        return mu, logvar, fused


class JTDecoder(nn.Module):
    """Decoder skeleton: reconstructs junction tree and then assembles graph.

    NOTE: The assembly step is non-trivial. This decoder returns a reconstructed
    probability over fragments and a placeholder assembly routine. Replace with
    rigorous assembly logic for production.
    """

    def __init__(self, fragment_vocab_size, z_dim=56, hidden_dim=128, cond_dim=0, max_tree_nodes: int = 12):
        super().__init__()
        self.max_tree_nodes = max_tree_nodes
        self.z_to_hidden = nn.Linear(z_dim + cond_dim, hidden_dim)
        # positional embeddings for each tree node slot
        self.positional = nn.Parameter(torch.randn(max_tree_nodes, hidden_dim))
        # predict fragment logits for each node
        self.frag_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fragment_vocab_size),
        )
        # optional: project to atom-level reconstruction (placeholder)
        self.node_decoder = nn.Linear(hidden_dim, 32)  # e.g. atom feature logits
        self.adj_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z, max_tree_nodes=None, cond=None):
        if max_tree_nodes is None:
            max_tree_nodes = self.max_tree_nodes
        if max_tree_nodes > self.max_tree_nodes:
            raise ValueError(
                f"Decoder initialised for {self.max_tree_nodes} tree nodes but received {max_tree_nodes}."
            )
        # z: [B, z_dim]
        if cond is not None:
            z = torch.cat([z, cond], dim=-1)
        h = F.relu(self.z_to_hidden(z))

        node_states = []
        frag_logits = []
        for idx in range(max_tree_nodes):
            node_input = h + self.positional[idx].unsqueeze(0)
            node_states.append(node_input)
            frag_logits.append(self.frag_predictor(node_input))

        frags_logits = torch.stack(frag_logits, dim=1)  # [B, max_nodes, V]
        node_states = torch.stack(node_states, dim=1)  # [B, max_nodes, hidden]
        node_feats = self.node_decoder(node_states)  # [B, max_nodes, F]

        adj_repr = self.adj_proj(node_states)
        adj_logits = torch.matmul(adj_repr, adj_repr.transpose(1, 2))
        diag = torch.diagonal(adj_logits, dim1=1, dim2=2)
        adj_logits = adj_logits - torch.diag_embed(diag)

        return frags_logits, node_feats, adj_logits


def _candidate_attachment_atoms(mol: "Chem.Mol") -> List[int]:
    atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not atoms:
        return [atom.GetIdx() for atom in mol.GetAtoms()]
    return atoms


def assemble_fragments(
    fragment_smiles: List[str],
    adjacency: Optional[np.ndarray],
    *,
    threshold: float = 0.5,
) -> Tuple[str, str]:
    """Attempt to assemble fragment SMILES into a single molecule using adjacency."""

    if not fragment_smiles:
        return "", "no_fragments"

    if not RDKit_AVAILABLE:
        return ".".join([s for s in fragment_smiles if s]), "rdkit_unavailable"

    valid_entries = [(idx, smi) for idx, smi in enumerate(fragment_smiles) if smi]
    if not valid_entries:
        return "", "no_valid_fragments"

    indices = [idx for idx, _ in valid_entries]
    smiles_list = []
    mols = []
    for idx, smi in valid_entries:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.warning("Skipping invalid fragment SMILES '%s' during assembly.", smi)
            continue
        smiles_list.append(smi)
        mols.append(Chem.Mol(mol))

    if not mols:
        return "", "no_valid_fragments"

    if len(mols) == 1 or adjacency is None:
        try:
            base = Chem.Mol(mols[0])
            Chem.SanitizeMol(base)
            return Chem.MolToSmiles(base), "single_fragment"
        except Exception:
            return smiles_list[0], "single_fragment"

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2:
        adjacency = np.zeros((len(mols), len(mols)), dtype=float)
    # reduce to valid indices
    adjacency = adjacency[np.ix_(indices, indices)]
    adjacency = np.nan_to_num(adjacency)
    adjacency = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adjacency, 0.0)

    combined = Chem.Mol(mols[0])
    anchor_indices = [_candidate_attachment_atoms(combined)[0]]
    status = "assembled"

    for i in range(1, len(mols)):
        new_mol = mols[i]
        base_atoms = combined.GetNumAtoms()
        combo = Chem.CombineMols(combined, new_mol)
        rw = Chem.RWMol(combo)
        new_candidates = _candidate_attachment_atoms(new_mol)
        neighbors = [j for j in range(i) if adjacency[i, j] > threshold]
        if not neighbors:
            neighbors = [0]
        success = False
        for neigh in neighbors:
            base_anchor = anchor_indices[min(neigh, len(anchor_indices) - 1)]
            for cand in new_candidates:
                rw_trial = Chem.RWMol(rw)
                new_anchor = base_atoms + cand
                try:
                    rw_trial.AddBond(base_anchor, new_anchor, Chem.BondType.SINGLE)
                    Chem.SanitizeMol(rw_trial)
                    combined = rw_trial.GetMol()
                    anchor_indices.append(new_anchor)
                    success = True
                    break
                except Exception:
                    # revert bond if sanitization fails and try next candidate
                    continue
            if success:
                break
        if not success:
            logger.warning("Unable to connect fragment %d during assembly; keeping disconnected.", i)
            status = "partial"
            combined = Chem.Mol(combo)
            fallback_anchor = new_candidates[0] if new_candidates else 0
            anchor_indices.append(base_atoms + fallback_anchor)

    try:
        Chem.SanitizeMol(combined)
        smiles = Chem.MolToSmiles(combined, isomericSmiles=True)
        return smiles, status
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Assembly sanitization failed: %s. Falling back to dot-joined fragments.", exc)
        fallback = ".".join(smiles_list)
        return fallback, "failed"


def _score_status(status: str, penalties: Dict[str, float]) -> float:
    return penalties.get(status, penalties.get("default", -5.0))


def beam_search_fragments(
    logits: torch.Tensor,
    adjacency: np.ndarray,
    idx_to_smiles: Dict[int, str],
    *,
    beam_width: int,
    topk_per_node: int,
    threshold: float,
    max_nodes: int,
    penalties: Dict[str, float],
) -> Dict[str, object]:
    # logits: [max_nodes, vocab]
    device = logits.device
    log_probs = torch.log_softmax(logits, dim=-1)
    max_nodes = min(max_nodes, logits.size(0))
    initial = {
        "indices": [],
        "smiles": [],
        "log_prob": 0.0,
        "status": "start",
    }
    beams = [initial]

    for position in range(max_nodes):
        node_log_prob = log_probs[position]
        top_log_prob, top_indices = torch.topk(node_log_prob, k=min(topk_per_node, node_log_prob.size(0)))
        new_beams = []
        for beam in beams:
            for lp, idx in zip(top_log_prob.tolist(), top_indices.tolist()):
                frag_smiles = idx_to_smiles.get(idx)
                new_indices = beam["indices"] + [idx]
                new_smiles = beam["smiles"] + ([frag_smiles] if frag_smiles else [""])
                score_log_prob = beam["log_prob"] + lp
                valid_positions = [i for i, s in enumerate(new_smiles) if s]
                adj_subset = None
                if adjacency is not None and len(valid_positions) > 0:
                    adj_subset = adjacency[np.ix_(valid_positions, valid_positions)]
                assembled_smiles, status = assemble_fragments(new_smiles, adj_subset, threshold=threshold)
                status_score = _score_status(status, penalties)
                new_beams.append(
                    {
                        "indices": new_indices,
                        "smiles": new_smiles,
                        "assembled": assembled_smiles,
                        "status": status,
                        "log_prob": score_log_prob + status_score,
                    }
                )
        # prune
        new_beams.sort(key=lambda x: x["log_prob"], reverse=True)
        beams = new_beams[:beam_width]
        if not beams:
            break
    if not beams:
        return {
            "smiles": "",
            "status": "beam_empty",
            "fragments": [],
            "score": float("-inf"),
        }
    best = beams[0]
    return {
        "smiles": best.get("assembled") or ".".join([s for s in best["smiles"] if s]),
        "status": best.get("status", "unknown"),
        "fragments": best["smiles"],
        "score": best["log_prob"],
    }


class JTVAE(nn.Module):
    def __init__(
        self,
        tree_feat_dim,
        graph_feat_dim,
        fragment_vocab_size,
        z_dim=56,
        hidden_dim=128,
        cond_dim=0,
        max_tree_nodes: int = 12,
    ):
        super().__init__()
        self.max_tree_nodes = max_tree_nodes
        self.encoder = JTEncoder(
            tree_feat_dim=tree_feat_dim,
            graph_feat_dim=graph_feat_dim,
            hidden_dim=hidden_dim,
            z_dim=z_dim,
            cond_dim=cond_dim,
        )
        self.decoder = JTDecoder(
            fragment_vocab_size=fragment_vocab_size,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            max_tree_nodes=max_tree_nodes,
        )
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.property_head = None
        if cond_dim and cond_dim > 0:
            self.property_head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, cond_dim),
            )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        tree_x,
        tree_edge_index,
        graph_x,
        graph_edge_index,
        batch_tree=None,
        batch_graph=None,
        cond=None,
    ):
        mu, logvar, fused = self.encoder(
            tree_x,
            tree_edge_index,
            graph_x,
            graph_edge_index,
            batch_tree,
            batch_graph,
            cond=cond,
        )
        z = self.reparameterize(mu, logvar)
        frags_logits, node_feats, adj_logits = self.decoder(z, cond=cond)
        property_pred = None
        if self.property_head is not None:
            property_pred = self.property_head(fused)
        return frags_logits, node_feats, adj_logits, mu, logvar, property_pred

    def sample(
        self,
        n_samples=32,
        cond=None,
        max_tree_nodes=12,
        fragment_idx_to_smiles=None,
        device=None,
        assemble_kwargs: Optional[Dict] = None,
    ):
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
        frags_logits, node_feats, adj_logits = self.decoder(z, max_tree_nodes=max_tree_nodes, cond=cond_t)
        if assemble_kwargs is None:
            assemble_kwargs = {}
        samples = []
        adj_probs = torch.sigmoid(adj_logits).detach().cpu().numpy()
        for b in range(n_samples):
            if fragment_idx_to_smiles is None:
                samples.append(
                    {
                        "smiles": "",
                        "status": "no_vocab_mapping",
                        "fragments": [],
                        "score": float("-inf"),
                    }
                )
                continue

            penalties = assemble_kwargs.get(
                "status_penalties",
                {
                    "assembled": 0.0,
                    "single_fragment": 0.0,
                    "partial": -1.0,
                    "failed": -5.0,
                    "no_fragments": -10.0,
                    "rdkit_unavailable": -2.0,
                    "beam_empty": -10.0,
                    "default": -5.0,
                },
            )
            beam_width = assemble_kwargs.get("beam_width", 5)
            topk_per_node = assemble_kwargs.get("topk_per_node", 5)
            adj_threshold = assemble_kwargs.get("adjacency_threshold", 0.5)

            beam_result = beam_search_fragments(
                frags_logits[b],
                adj_probs[b],
                fragment_idx_to_smiles,
                beam_width=beam_width,
                topk_per_node=topk_per_node,
                threshold=adj_threshold,
                max_nodes=max_tree_nodes,
                penalties=penalties,
            )
            samples.append(beam_result)
        return samples

# -------------------------
# Loss and training utilities
# -------------------------

def jtvae_loss(
    frags_logits,
    node_feats,
    mu,
    logvar,
    *,
    target_frag_idxs=None,
    property_pred=None,
    cond_target=None,
    adj_logits=None,
    adj_target=None,
    beta: float = 0.5,
    aux_weight: float = 0.0,
    adj_weight: float = 1.0,
):
    """Compute JT-VAE loss: fragment + adjacency reconstruction, KL, and auxiliary property loss.

    Parameters
    ----------
    frags_logits:
        Tensor of shape [B, max_nodes, vocab] with fragment logits.
    target_frag_idxs:
        Tensor of shape [B, max_nodes] with target fragment indices (padded with -1).
    property_pred:
        Tensor of shape [B, cond_dim] containing predicted conditioning properties.
    cond_target:
        Tensor of shape [B, cond_dim] with target conditioning properties.
    adj_logits:
        Tensor of shape [B, max_nodes, max_nodes] with adjacency logits.
    adj_target:
        Tensor of shape [B, max_nodes, max_nodes] containing adjacency targets in {0,1}.
    beta:
        Weight on KL divergence term.
    aux_weight:
        Weight on auxiliary property prediction loss.
    adj_weight:
        Weight on adjacency reconstruction loss.
    """

    recon_loss = torch.tensor(0.0, device=mu.device)
    if target_frag_idxs is not None:
        B = frags_logits.size(0)
        V = frags_logits.size(-1)
        logits = frags_logits.view(-1, V)
        targets = target_frag_idxs.view(-1)
        recon_loss = F.cross_entropy(logits, targets, ignore_index=-1)

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    property_loss = torch.tensor(0.0, device=mu.device)
    if (
        aux_weight
        and aux_weight > 0.0
        and property_pred is not None
        and cond_target is not None
    ):
        property_loss = F.mse_loss(property_pred, cond_target)

    adj_loss = torch.tensor(0.0, device=mu.device)
    if (
        adj_weight
        and adj_weight > 0.0
        and adj_logits is not None
        and adj_target is not None
        and target_frag_idxs is not None
    ):
        valid_mask = (target_frag_idxs != -1)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        mask = mask & (~torch.eye(mask.size(1), device=mask.device, dtype=torch.bool).unsqueeze(0))
        logits_masked = adj_logits[mask]
        target_masked = adj_target.to(adj_logits.device)[mask]
        if logits_masked.numel() > 0:
            adj_loss = F.binary_cross_entropy_with_logits(logits_masked, target_masked)

    total = recon_loss + beta * kl + aux_weight * property_loss + adj_weight * adj_loss
    return total, recon_loss, kl, property_loss, adj_loss


def train_jtvae(
    model: JTVAE,
    dataset,
    fragment_vocab: Dict[int, str],
    device: str = None,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    save_dir: str = './jtvae_models',
    kl_weight: float = 0.5,
    property_weight: float = 0.0,
    adj_weight: float = 1.0,
):
    os.makedirs(save_dir, exist_ok=True)
    device_spec = get_device(device)
    if device_spec.type == "directml":
        logging.getLogger(__name__).warning(
            "DirectML backend lacks required scatter kernels for PyG; falling back to CPU for JT-VAE training."
        )
        device_spec = get_device("cpu")
    model.to(device_spec.target)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # dataset should provide precomputed: tree_x, tree_edge_index, graph_x, graph_edge_index, target_frag_idxs, cond
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        epoch_prop = 0.0
        epoch_adj = 0.0
        for batch in loader:
            # batch must contain fields as described above
            batch = move_to_device(batch, device_spec)
            cond = batch.cond if hasattr(batch, 'cond') else None
            target_frag_idxs = batch.target_frag_idxs if hasattr(batch, 'target_frag_idxs') else None
            target_adj = batch.tree_adj if hasattr(batch, 'tree_adj') else None
            if cond is not None:
                cond = cond.to(device_spec.target).float()
            if target_frag_idxs is not None:
                target_frag_idxs = target_frag_idxs.to(device_spec.target).long()
            if target_adj is not None:
                target_adj = target_adj.to(device_spec.target).float()
            frags_logits, node_feats, adj_logits, mu, logvar, prop_pred = model(
                batch.tree_x,
                batch.tree_edge_index,
                batch.graph_x,
                batch.graph_edge_index,
                batch_tree=batch.tree_batch if hasattr(batch, 'tree_batch') else None,
                batch_graph=batch.batch if hasattr(batch, 'batch') else None,
                cond=cond,
            )
            loss, recon, kl, prop_loss, adj_loss = jtvae_loss(
                frags_logits,
                node_feats,
                mu,
                logvar,
                target_frag_idxs=target_frag_idxs,
                property_pred=prop_pred,
                cond_target=cond,
                adj_logits=adj_logits,
                adj_target=target_adj,
                beta=kl_weight,
                aux_weight=property_weight,
                adj_weight=adj_weight,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.num_graphs
            epoch_recon += recon.item() * batch.num_graphs
            epoch_kl += kl.item() * batch.num_graphs
            epoch_prop += prop_loss.item() * batch.num_graphs
            epoch_adj += adj_loss.item() * batch.num_graphs
        denom = len(dataset) if len(dataset) > 0 else 1
        print(
            f"Epoch {epoch:03d} total={epoch_loss/denom:.4f} "
            f"recon={epoch_recon/denom:.4f} kl={epoch_kl/denom:.4f} "
            f"prop={epoch_prop/denom:.4f} adj={epoch_adj/denom:.4f}"
        )
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
        assemble_kwargs=assemble_kwargs,
    )

    formatted: List[Dict[str, str]] = []
    for entry in raw_samples:
        if isinstance(entry, dict):
            formatted.append(entry)
        elif isinstance(entry, str):
            formatted.append({"smiles": entry, "status": "raw"})
        elif isinstance(entry, (list, tuple)):
            formatted.append(
                {"smiles": "".join(entry), "status": "raw_fragments", "fragments": list(entry)}
            )
        else:
            formatted.append({"smiles": str(entry), "status": "unknown"})
    return formatted

# -------------------------
# Dataset adapter for JT-VAE
# -------------------------
class JTData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "tree_edge_index":
            return self.tree_x.size(0)
        if key == "graph_edge_index":
            return self.graph_x.size(0)
        if key == "tree_batch":
            # increment fragment batch indices by one per example
            return value.new_full((1,), 1, dtype=value.dtype, device=value.device)
        if key == "batch":
            return value.new_full((1,), 1, dtype=value.dtype, device=value.device)
        if key in {"target_frag_idxs", "tree_adj"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {"tree_edge_index", "graph_edge_index"}:
            return 1
        if key in {"tree_batch", "batch"}:
            return 0
        if key in {"target_frag_idxs", "tree_adj"}:
            # keep per-graph tensors in separate batch dimension
            return None
        if key == "cond":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


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
        data = JTData()
        data.tree_x = ex["tree_x"]
        data.tree_edge_index = ex["tree_edge_index"]
        data.graph_x = ex["graph_x"]
        data.graph_edge_index = ex["graph_edge_index"]
        data.target_frag_idxs = ex.get("target_frag_idxs", None)
        cond = ex.get("cond", None)
        if cond is not None:
            data.cond = cond.unsqueeze(0) if cond.dim() == 1 else cond
        data.tree_adj = ex.get("tree_adj", None)
        data.num_graphs = 1
        data.tree_num_nodes = data.tree_x.size(0)
        data.graph_num_nodes = data.graph_x.size(0)
        data.tree_batch = torch.zeros(data.tree_num_nodes, dtype=torch.long)
        data.batch = torch.zeros(data.graph_num_nodes, dtype=torch.long)
        data.num_nodes = data.graph_num_nodes
        return data

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


logger = logging.getLogger(__name__)
