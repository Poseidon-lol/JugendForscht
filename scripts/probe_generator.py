"""
Quick JT-VAE sampling probe to debug generator outputs before running the active loop.

Usage (example):
  python scripts/probe_generator.py --ckpt models/generator_test/jtvae_epoch_20.pt \
    --vocab models/generator_test/fragment_vocab.json --n-samples 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
import sys

import torch

# ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.jtvae_extended import JTVAE, sample_conditional


def _infer_cond_dim(state: dict) -> int:
    if "property_head.2.weight" in state:
        return state["property_head.2.weight"].shape[0]
    fused_dim = state["encoder.fc_mu.weight"].shape[1]
    hidden_dim = state["encoder.tree_encoder.input_proj.weight"].shape[0]
    return max(fused_dim - 2 * hidden_dim, 0)


def _load_vocab(path: Path) -> dict[str, int]:
    raw = json.load(path.open("r", encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Fragment vocab at {path} is empty or not a mapping.")
    sample_key, sample_val = next(iter(raw.items()))

    def _intlike(x: object) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    if _intlike(sample_val):  # frag -> idx
        return {str(k): int(v) for k, v in raw.items()}
    if _intlike(sample_key):  # idx -> frag
        return {str(v): int(k) for k, v in raw.items()}
    raise ValueError(f"Unrecognised fragment_vocab format in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe JT-VAE sampling to debug generator outputs.")
    parser.add_argument("--ckpt", required=True, help="Path to JT-VAE checkpoint (.pt).")
    parser.add_argument("--vocab", required=True, help="Path to fragment_vocab.json.")
    parser.add_argument("--n-samples", type=int, default=8, help="Number of samples to generate.")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for assembly.")
    parser.add_argument("--topk-per-node", type=int, default=5, help="Top-k fragments per node.")
    parser.add_argument("--adj-threshold", type=float, default=0.6, help="Adjacency threshold for assembly.")
    parser.add_argument("--device", default="cpu", help="Device for sampling (cpu/cuda:0/etc).")
    parser.add_argument("--max-tree-nodes", type=int, default=None, help="Override decoder max_tree_nodes during sampling.")
    parser.add_argument("--max-heavy", type=int, default=None, help="Filter samples with heavy atom count above this.")
    parser.add_argument("--max-length", type=int, default=None, help="Filter samples with SMILES length above this.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    vocab_path = Path(args.vocab)
    state = torch.load(ckpt_path, map_location="cpu")
    frag_vocab = _load_vocab(vocab_path)

    hidden_dim = state["encoder.tree_encoder.input_proj.weight"].shape[0]
    node_feat_dim = state["encoder.tree_encoder.input_proj.weight"].shape[1]
    graph_feat_dim = state["encoder.graph_encoder.input_proj.weight"].shape[1]
    z_dim = state["encoder.fc_mu.weight"].shape[0]
    cond_dim = _infer_cond_dim(state)
    positional_key = next((k for k in state if k.endswith("decoder.positional")), None)
    max_tree_nodes = state[positional_key].shape[0] if positional_key else 12

    model = JTVAE(
        tree_feat_dim=node_feat_dim,
        graph_feat_dim=graph_feat_dim,
        fragment_vocab_size=len(frag_vocab),
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        cond_dim=cond_dim,
        max_tree_nodes=max_tree_nodes,
    )
    model.load_state_dict(state)
    model.eval()

    samples = sample_conditional(
        model,
        frag_vocab,
        n_samples=args.n_samples,
        assemble_kwargs={
            "beam_width": args.beam_width,
            "topk_per_node": args.topk_per_node,
            "adjacency_threshold": args.adj_threshold,
            "max_tree_nodes": args.max_tree_nodes or max_tree_nodes,
        },
        device=args.device,
    )

    counts = Counter()
    filtered = []

    def _heavy_atoms(smiles: str) -> int:
        try:
            from rdkit import Chem  # type: ignore

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        except Exception:
            return 0

    for s in samples:
        smi = s.get("smiles") or ""
        status = s.get("status") or "unknown"
        if smi and args.max_length and len(smi) > args.max_length:
            status = "filtered_len"
        if smi and args.max_heavy:
            hcount = _heavy_atoms(smi)
            if hcount > args.max_heavy:
                status = "filtered_heavy"
        counts[status] += 1
        filtered.append((status, smi))

    print(f"cond_dim={cond_dim} max_tree_nodes={max_tree_nodes} vocab={len(frag_vocab)}")
    print("status counts:", dict(counts))
    for idx, (status, smi) in enumerate(filtered, 1):
        print(f"{idx:02d} {status}: {smi}")


if __name__ == "__main__":
    main()
