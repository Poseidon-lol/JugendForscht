#!/usr/bin/env python3
"""
Build dimer/trimer oligomers from monomers that contain attachment points.

Monomers typically need two dummy atoms ([*]) to define a linear chain.
With --attachment-policy, you can optionally pick two from many or allow
single-attachment dimers.
Outputs a CSV with monomer_smiles, oligomer_smiles, n_units, status.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem


LOGGER = logging.getLogger("build_oligomer_proxy")


def _find_attachments(mol: Chem.Mol) -> List[Tuple[int, int, Chem.BondType]]:
    attachments: List[Tuple[int, int, Chem.BondType]] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1:
            continue
        nb = neighbors[0]
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx())
        if bond is None:
            continue
        attachments.append((atom.GetIdx(), nb.GetIdx(), bond.GetBondType()))
    return attachments


def _pick_two_attachments(
    mol: Chem.Mol, attachments: Sequence[Tuple[int, int, Chem.BondType]]
) -> List[Tuple[int, int, Chem.BondType]]:
    if len(attachments) <= 2:
        return list(attachments)
    dist = Chem.GetDistanceMatrix(mol)
    best_pair = None
    best_dist = -1.0
    for i in range(len(attachments)):
        for j in range(i + 1, len(attachments)):
            _, nb_i, _ = attachments[i]
            _, nb_j, _ = attachments[j]
            d = float(dist[nb_i, nb_j])
            if d > best_dist:
                best_dist = d
                best_pair = (attachments[i], attachments[j])
    if best_pair is None:
        return list(attachments[:2])
    return [best_pair[0], best_pair[1]]


def _mark_methyl_attachments(mol: Chem.Mol) -> Tuple[Chem.Mol, int]:
    """Convert aromatic-methyl substituents into dummy attachment points."""
    rw = Chem.RWMol(mol)
    updated = 0
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if atom.GetDegree() != 1:
            continue
        if atom.GetTotalNumHs() != 3:
            continue
        nb = atom.GetNeighbors()[0]
        if nb.GetAtomicNum() != 6:
            continue
        if not nb.GetIsAromatic():
            continue
        atom.SetAtomicNum(0)
        atom.SetIsotope(0)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        updated += 1
    if updated > 0:
        Chem.SanitizeMol(rw, catchErrors=True)
    return rw.GetMol(), updated


def _remove_dummies(mol: Chem.Mol) -> Tuple[Chem.Mol, dict]:
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 0]
    mapping = {}
    rw = Chem.RWMol()
    for old_idx in keep:
        atom = mol.GetAtomWithIdx(old_idx)
        new_idx = rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        mapping[old_idx] = new_idx
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 not in mapping or a2 not in mapping:
            continue
        rw.AddBond(mapping[a1], mapping[a2], bond.GetBondType())
    return rw.GetMol(), mapping


def _build_chain(core: Chem.Mol, attach: List[Tuple[int, Chem.BondType]], n_units: int) -> Optional[Chem.Mol]:
    if n_units < 2:
        return None
    attach_sorted = sorted(attach, key=lambda x: x[0])
    head_idx, head_bond = attach_sorted[0]
    tail_idx, tail_bond = attach_sorted[1]
    current = Chem.Mol(core)
    current_tail = tail_idx
    current_tail_bond = tail_bond
    for _ in range(1, n_units):
        new = Chem.Mol(core)
        combo = Chem.CombineMols(current, new)
        offset = current.GetNumAtoms()
        rw = Chem.RWMol(combo)
        bond_type = current_tail_bond
        if bond_type == Chem.BondType.AROMATIC:
            bond_type = Chem.BondType.SINGLE
        rw.AddBond(current_tail, head_idx + offset, bond_type)
        current = rw.GetMol()
        current_tail = tail_idx + offset
        current_tail_bond = tail_bond
    try:
        Chem.SanitizeMol(current)
    except Exception:
        return None
    return current


def _build_oligomer(
    smiles: str,
    n_units: int,
    *,
    methyl_as_attach: bool,
    attachment_policy: str,
) -> Tuple[Optional[str], str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    attachments = _find_attachments(mol)
    if len(attachments) != 2 and methyl_as_attach:
        mol, updated = _mark_methyl_attachments(mol)
        if updated > 0:
            attachments = _find_attachments(mol)
    if len(attachments) != 2:
        if attachment_policy in ("pick_two", "allow_single") and len(attachments) > 2:
            attachments = _pick_two_attachments(mol, attachments)
        elif attachment_policy == "allow_single" and len(attachments) == 1:
            if n_units != 2:
                return None, "attachments_insufficient"
            attachments = [attachments[0], attachments[0]]
        else:
            return None, "attachments_not_two"
    core, mapping = _remove_dummies(mol)
    attach_core: List[Tuple[int, Chem.BondType]] = []
    for _, nb_idx, bond_type in attachments:
        if nb_idx in mapping:
            attach_core.append((mapping[nb_idx], bond_type))
    if len(attach_core) != 2:
        return None, "attachments_mapping_failed"
    chain = _build_chain(core, attach_core, n_units)
    if chain is None:
        return None, "build_failed"
    smi = Chem.MolToSmiles(chain, isomericSmiles=True)
    return smi, "ok"


def _detect_smiles_column(columns: Sequence[str], override: Optional[str]) -> str:
    if override and override in columns:
        return override
    for candidate in ("smiles", "smile", "SMILES"):
        if candidate in columns:
            return candidate
    raise KeyError("No smiles column found.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build oligomer SMILES from monomers with dummy attachments.")
    parser.add_argument("--input", required=True, help="Monomer CSV with attachment points.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--smiles-col", default=None, help="SMILES column name.")
    parser.add_argument("--units", nargs="+", type=int, default=[2, 3], help="Oligomer sizes to build.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--methyl-attachments",
        action="store_true",
        help="Treat aromatic methyl substituents as attachment points.",
    )
    parser.add_argument(
        "--attachment-policy",
        choices=("strict", "pick_two", "allow_single"),
        default="strict",
        help="How to handle attachment counts other than two.",
    )
    parser.add_argument("--log-every", type=int, default=1000, help="Progress log cadence.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(input_path)
    smiles_col = _detect_smiles_column(df.columns, args.smiles_col)
    if args.max_rows:
        df = df.head(args.max_rows)

    rows = []
    total = 0
    ok = 0
    for smi in df[smiles_col].dropna().astype(str):
        total += 1
        for n_units in args.units:
            oligomer, status = _build_oligomer(
                smi,
                int(n_units),
                methyl_as_attach=args.methyl_attachments,
                attachment_policy=args.attachment_policy,
            )
            if status == "ok":
                ok += 1
            rows.append(
                {
                    "monomer_smiles": smi,
                    "oligomer_smiles": oligomer,
                    "n_units": int(n_units),
                    "status": status,
                }
            )
        if args.log_every and total % int(args.log_every) == 0:
            LOGGER.info("Processed %d monomers | oligomers ok=%d", total, ok)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    LOGGER.info("Wrote %d oligomer rows to %s", len(out_df), output_path)


if __name__ == "__main__":
    main()
