# src/data/featurization.py
"""
Featurization utilities for organic semiconductor molecules.
------------------------------------------------------------
This module converts SMILES strings into graph tensors compatible
with PyTorch Geometric.

Each molecule becomes a torch_geometric.data.Data object with:
- x: [num_atoms, num_features] atom features
- edge_index: [2, num_bonds] connectivity matrix
- edge_attr: [num_bonds, num_bond_features]
- y: target property (e.g. HOMO, LUMO, conductivity)
"""

import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

# -------------------------------------------
# 1. Define feature dictionaries
# -------------------------------------------
ATOM_LIST = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "H"]
BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

def one_hot_encoding(x, allowable_set):
    """Return one-hot encoding of x among allowable_set."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

# -------------------------------------------
# 2. Atom featurization
# -------------------------------------------
def atom_features(atom):
    return torch.tensor(
        one_hot_encoding(atom.GetSymbol(), ATOM_LIST)
        + [atom.GetDegree(), atom.GetTotalNumHs(), atom.GetIsAromatic()],
        dtype=torch.float
    )

# -------------------------------------------
# 3. Bond featurization
# -------------------------------------------
def bond_features(bond):
    bt = [0, 0, 0, 0]
    bt[BOND_TYPES[bond.GetBondType()]] = 1
    return torch.tensor(bt, dtype=torch.float)

# -------------------------------------------
# 4. Main function: SMILES -> graph tensor
# -------------------------------------------
def mol_to_graph(smiles: str, y=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # --- atoms ---
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    # --- bonds ---
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # undirected graph: add both directions
        edge_index += [[i, j], [j, i]]
        edge_attr += [bf, bf]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_TYPES)), dtype=torch.float)

    # --- target (optional) ---
    y_tensor = None
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float).view(1, -1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)

# -------------------------------------------
# 5. Example usage (for debugging)
# -------------------------------------------
if __name__ == "__main__":
    smiles = "c1ccccc1"  # benzene
    data = mol_to_graph(smiles, y=[-5.5, -2.5])  # example HOMO/LUMO
    print(data)
    print("Node features:", data.x.shape)
    print("Edge index:", data.edge_index.shape)
