# src/data/jt_preprocess.py
"""
Preprocessing script for JT-VAE dataset
--------------------------------------
Creates a processed dataset for JT-VAE training from a CSV in data/raw.

Input CSV (data/raw/osc_data.csv) expected columns:
 - smiles (SMILES string)
 - HOMO, LUMO (float)  -- used for conditioning
 - optional: id, dopant, other metadata

Outputs (saved to data/processed/):
 - jt_examples.pt : list of dicts with keys:
     - tree_x (torch.FloatTensor) [num_frags, frag_feat_dim]
     - tree_edge_index (torch.LongTensor) [2, num_tree_edges]
     - graph_x (torch.FloatTensor) [num_atoms, atom_feat_dim]
     - graph_edge_index (torch.LongTensor) [2, num_bonds]
     - target_frag_idxs (torch.LongTensor) [max_tree_nodes] (padded with -1)
     - cond (torch.FloatTensor) [cond_dim]
 - fragment_vocab.json : mapping idx -> fragment_smiles
 - preprocessing_stats.json : normalization stats (mean/std) for conditioning

Notes / limitations:
 - Fragment extraction is simplified: uses ring fragments + Murcko scaffold.
 - Fragment adjacency: fragments considered connected if they share any heavy atom index.
 - Fragment features: Morgan fingerprint (radius=2) converted to float vector.
 - You should review/sample produced fragments for quality before large-scale runs.

Requirements:
 - rdkit
 - torch
 - rdkit.Chem.rdMolDescriptors
 - src.data.featurization.mol_to_graph (for atom-level graph)

Usage:
 python src/data/jt_preprocess.py --input data/raw/osc_data.csv --out_dir data/processed --max_frags 12

"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# RDKit
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
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

# local featurizer
try:
    from src.data.featurization import mol_to_graph
except Exception:
    from data.featurization import mol_to_graph


# -------------------------
# Fragment utilities
# -------------------------

def extract_fragments(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    frags = set()
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        atom_idxs = list(ring)
        sub = Chem.PathToSubmol(mol, atom_idxs)
        s = Chem.MolToSmiles(sub)
        frags.add(s)
    try:
        ms = rdMolDescriptors.CalcMurckoScaffoldSmiles(mol)
        if ms and ms.strip():
            frags.add(ms)
    except Exception:
        pass
    # fallback: whole molecule
    if len(frags) == 0:
        frags.add(smiles)
    return list(frags)


def frag_to_fp_vector(frag_smiles: str, n_bits=512):
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# -------------------------
# Build fragment vocab across dataset
# -------------------------

def build_fragment_vocab(smiles_list, min_count=1):
    counter = defaultdict(int)
    for smi in smiles_list:
        frags = extract_fragments(smi)
        for f in frags:
            counter[f] += 1
    # filter by min_count
    items = [f for f,c in counter.items() if c >= min_count]
    items = sorted(items)
    idx2frag = {i: frag for i, frag in enumerate(items)}
    frag2idx = {frag: i for i, frag in idx2frag.items()}
    return frag2idx, idx2frag


# -------------------------
# Fragment adjacency
# -------------------------

def fragment_adjacency_from_mol(smiles: str, fragments: list):
    """Return adjacency list between fragments: if fragments share atom indices -> connected.
    Also return mapping frag -> atom idx set for possible later use.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], {}
    frag_atom_sets = {}
    # find occurrences (naive) by matching substructure
    for frag in fragments:
        pattern = Chem.MolFromSmiles(frag)
        if pattern is None:
            frag_atom_sets[frag] = set()
            continue
        matches = mol.GetSubstructMatches(pattern)
        atom_idxs = set()
        for m in matches:
            atom_idxs.update(m)
        frag_atom_sets[frag] = atom_idxs
    # build adjacency
    adj = defaultdict(set)
    for i, fi in enumerate(fragments):
        for j, fj in enumerate(fragments):
            if i >= j:
                continue
            if len(frag_atom_sets.get(fi, set()) & frag_atom_sets.get(fj, set())) > 0:
                adj[i].add(j)
                adj[j].add(i)
    # convert to edge_index
    edges = []
    for i, neighs in adj.items():
        for j in neighs:
            edges.append((i, j))
            edges.append((j, i))
    if len(edges) == 0:
        # make a trivial chain if only one fragment
        if len(fragments) > 1:
            edges = [(k, k+1) for k in range(len(fragments)-1)]
            edges = [(i,j) for (i,j) in edges for _ in (0,1)]  # both directions
    return edges, frag_atom_sets


# -------------------------
# Preprocess single molecule
# -------------------------

def process_one(smiles: str, homo: float, lumo: float, frag2idx: dict, max_frags: int = 12, fp_bits: int = 512):
    # 1) extract fragments for this mol
    frags = extract_fragments(smiles)
    # map to indices (filter unknown frags)
    frag_idxs = [frag2idx[f] for f in frags if f in frag2idx]
    # limit to max_frags
    if len(frag_idxs) > max_frags:
        frag_idxs = frag_idxs[:max_frags]
    # target vector padded
    targ = np.full((max_frags,), -1, dtype=np.int64)
    targ[:len(frag_idxs)] = frag_idxs

    # 2) fragment features (fp)
    frag_feats = []
    frag_smiles_list = [list(frag2idx.keys())[list(frag2idx.values()).index(idx)] if True else '' for idx in frag_idxs]
    # above is inefficient; instead build reverse mapping
    idx2frag = {v:k for k,v in frag2idx.items()}
    frag_smiles_list = [idx2frag[idx] for idx in frag_idxs]
    for fsm in frag_smiles_list:
        vec = frag_to_fp_vector(fsm, n_bits=fp_bits)
        frag_feats.append(vec)
    if len(frag_feats) == 0:
        # fallback: use whole-molecule fingerprint
        vec = frag_to_fp_vector(smiles, n_bits=fp_bits)
        frag_feats = [vec]
        targ = np.full((max_frags,), -1, dtype=np.int64)
        targ[0] = frag2idx.get(frag_smiles_list[0], 0) if len(frag_smiles_list)>0 else 0
    # pad fragment features to max_frags
    while len(frag_feats) < max_frags:
        frag_feats.append(np.zeros((fp_bits,), dtype=np.float32))
    frag_feats = np.vstack(frag_feats).astype(np.float32)

    # 3) fragment adjacency -> edge_index
    # map fragment SMILES list to their occurrences in this mol (use original extracted list)
    # For adjacency we need the fragment list corresponding to the non-padded ones
    real_frags = [idx2frag[idx] for idx in frag_idxs]
    edges, frag_atom_sets = fragment_adjacency_from_mol(smiles, real_frags)
    if len(edges) == 0:
        # create trivial self-loop
        edge_index = np.array([[0,0]], dtype=np.int64).T
    else:
        edge_index = np.array(edges, dtype=np.int64).T

    # pad edge_index if necessary (not required for PyG)

    # 4) atom-level graph via existing featurizer
    atom_data = mol_to_graph(smiles)
    # atom_data.x (torch.FloatTensor), atom_data.edge_index (torch.LongTensor)

    example = {
        'tree_x': torch.tensor(frag_feats, dtype=torch.float32),
        'tree_edge_index': torch.tensor(edge_index, dtype=torch.long) if edge_index.size>0 else torch.zeros((2,0), dtype=torch.long),
        'graph_x': atom_data.x,  # already torch tensors
        'graph_edge_index': atom_data.edge_index,
        'target_frag_idxs': torch.tensor(targ, dtype=torch.long),
        'cond_raw': np.array([homo, lumo], dtype=np.float32),
        'smiles': smiles
    }
    return example


# -------------------------
# Full preprocessing
# -------------------------

def preprocess(input_csv: str, out_dir: str, max_frags: int = 12, fp_bits: int = 512):
    import pandas as pd
    df = pd.read_csv(input_csv)
    assert 'smiles' in df.columns, 'input CSV must contain smiles column'
    if not ('HOMO' in df.columns and 'LUMO' in df.columns):
        raise ValueError('CSV must contain HOMO and LUMO columns for conditioning')

    os.makedirs(out_dir, exist_ok=True)
    raw_smiles = df['smiles'].dropna().unique().tolist()
    frag2idx, idx2frag = build_fragment_vocab(raw_smiles)
    # save fragment vocab
    with open(os.path.join(out_dir, 'fragment_vocab.json'), 'w') as f:
        json.dump(idx2frag, f, indent=2)

    examples = []
    conds = []
    print('Processing molecules...')
    for _, row in tqdm(df.iterrows(), total=len(df)):
        smi = row['smiles']
        try:
            ex = process_one(smi, float(row['HOMO']), float(row['LUMO']), frag2idx, max_frags=max_frags, fp_bits=fp_bits)
            examples.append(ex)
            conds.append(ex['cond_raw'])
        except Exception as e:
            print(f'Error processing {smi}: {e}')

    # normalize cond (HOMO,LUMO)
    conds = np.vstack(conds).astype(np.float32)
    mean = conds.mean(axis=0)
    std = conds.std(axis=0) + 1e-8
    preprocessing_stats = {'cond_mean': mean.tolist(), 'cond_std': std.tolist()}
    with open(os.path.join(out_dir, 'preprocessing_stats.json'), 'w') as f:
        json.dump(preprocessing_stats, f, indent=2)

    # attach normalized cond and remove raw
    for ex in examples:
        raw = ex.pop('cond_raw')
        norm = (raw - mean) / std
        ex['cond'] = torch.tensor(norm, dtype=torch.float32)

    # save examples as torch file
    torch.save(examples, os.path.join(out_dir, 'jt_examples.pt'))
    print(f'Saved {len(examples)} examples to {os.path.join(out_dir, "jt_examples.pt")}')


# -------------------------
# CLI
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/raw/osc_data.csv')
    parser.add_argument('--out_dir', type=str, default='data/processed')
    parser.add_argument('--max_frags', type=int, default=12)
    parser.add_argument('--fp_bits', type=int, default=512)
    args = parser.parse_args()
    preprocess(args.input, args.out_dir, max_frags=args.max_frags, fp_bits=args.fp_bits)
