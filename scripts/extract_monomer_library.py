#!/usr/bin/env python3
"""
Extract a monomer library from a polymer dataset by fragmenting polymer bonds.

Outputs:
  - monomer_library.csv: unique fragments with counts
  - monomer_dataset.csv: unique fragments for JT-VAE training
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, Recap


LOGGER = logging.getLogger("extract_monomer_library")


def _canonicalize_mol(mol: Optional[Chem.Mol]) -> Optional[str]:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def _heavy_atom_count(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _strip_dummy_atoms(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None
    dummy_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_idxs:
        return mol
    rw = Chem.RWMol(mol)
    for idx in sorted(dummy_idxs, reverse=True):
        rw.RemoveAtom(idx)
    stripped = rw.GetMol()
    if stripped.GetNumAtoms() == 0:
        return None
    return stripped


def _filter_fragments(
    frags: Iterable[Chem.Mol],
    *,
    min_heavy: int,
    max_heavy: Optional[int],
    keep_dummies: bool,
) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for frag in frags:
        if frag is None:
            continue
        if not keep_dummies:
            frag = _strip_dummy_atoms(frag)
            if frag is None:
                continue
        heavy = _heavy_atom_count(frag)
        if min_heavy > 0 and heavy < min_heavy:
            continue
        if max_heavy is not None and max_heavy > 0 and heavy > max_heavy:
            continue
        smi = _canonicalize_mol(frag)
        if smi:
            rows.append((smi, heavy))
    return rows


def _fragment_brics(
    mol: Chem.Mol,
    *,
    min_heavy: int,
    max_heavy: Optional[int],
    keep_dummies: bool,
) -> List[Tuple[str, int]]:
    try:
        frags = BRICS.BRICSDecompose(
            mol,
            minFragmentSize=max(1, int(min_heavy)),
            keepNonLeafNodes=True,
            returnMols=False,
        )
    except Exception:
        return []
    mols = []
    for smi in frags or []:
        frag_mol = Chem.MolFromSmiles(smi)
        if frag_mol is not None:
            mols.append(frag_mol)
    return _filter_fragments(mols, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)


def _fragment_recap(
    mol: Chem.Mol,
    *,
    min_heavy: int,
    max_heavy: Optional[int],
    keep_dummies: bool,
) -> List[Tuple[str, int]]:
    try:
        root = Recap.RecapDecompose(mol, minFragmentSize=max(0, int(min_heavy)))
    except Exception:
        return []
    nodes = root.GetAllChildren() if root is not None else {}
    mols = []
    for node in nodes.values():
        smi = getattr(node, "smiles", None)
        if not smi:
            continue
        frag_mol = Chem.MolFromSmiles(smi)
        if frag_mol is not None:
            mols.append(frag_mol)
    return _filter_fragments(mols, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)


def _fragment_rotatable(
    mol: Chem.Mol,
    *,
    min_heavy: int,
    max_heavy: Optional[int],
    keep_dummies: bool,
) -> List[Tuple[str, int]]:
    bond_indices = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1:
            continue
        bond_indices.append(bond.GetIdx())
    if not bond_indices:
        return []
    try:
        frag_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return []
    return _filter_fragments(frags, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)


def _bond_indices_from_smarts(mol: Chem.Mol, patterns: Sequence[str]) -> List[int]:
    bond_indices = set()
    for smarts in patterns:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            LOGGER.warning("Invalid SMARTS pattern skipped: %s", smarts)
            continue
        if patt.GetNumAtoms() != 2:
            LOGGER.warning("Bond SMARTS must have 2 atoms; skipping: %s", smarts)
            continue
        for match in mol.GetSubstructMatches(patt):
            if len(match) != 2:
                continue
            bond = mol.GetBondBetweenAtoms(match[0], match[1])
            if bond is not None:
                bond_indices.add(bond.GetIdx())
    return sorted(bond_indices)


def _fragment_smarts(
    mol: Chem.Mol,
    *,
    bond_smarts: Sequence[str],
    min_heavy: int,
    max_heavy: Optional[int],
    keep_dummies: bool,
) -> List[Tuple[str, int]]:
    bond_indices = _bond_indices_from_smarts(mol, bond_smarts)
    if not bond_indices:
        return []
    try:
        frag_mol = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return []
    return _filter_fragments(frags, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)


def _detect_smiles_column(df: pd.DataFrame, override: Optional[str]) -> str:
    if override:
        return override
    for candidate in ("smiles", "smile", "SMILES"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Input CSV must contain a SMILES column (smiles/smile/SMILES).")


def _fragment_molecule(
    mol: Chem.Mol,
    method: str,
    *,
    min_heavy: int,
    max_heavy: Optional[int],
    keep_dummies: bool,
    bond_smarts: Sequence[str],
) -> List[Tuple[str, int]]:
    method_key = method.lower().strip()
    if method_key == "brics":
        return _fragment_brics(mol, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)
    if method_key == "recap":
        return _fragment_recap(mol, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)
    if method_key == "rotatable":
        return _fragment_rotatable(mol, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)
    if method_key == "smarts":
        return _fragment_smarts(
            mol,
            bond_smarts=bond_smarts,
            min_heavy=min_heavy,
            max_heavy=max_heavy,
            keep_dummies=keep_dummies,
        )
    if method_key == "hybrid":
        frags = _fragment_brics(mol, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)
        if len(frags) < 2:
            frags = _fragment_rotatable(mol, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)
        return frags
    LOGGER.warning("Unknown method '%s'; falling back to BRICS.", method)
    return _fragment_brics(mol, min_heavy=min_heavy, max_heavy=max_heavy, keep_dummies=keep_dummies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract monomer fragments from a polymer dataset.")
    parser.add_argument("--input", required=True, help="Input CSV containing polymer SMILES.")
    parser.add_argument("--output-dir", default="data/processed/monomers", help="Output directory.")
    parser.add_argument("--smiles-col", default=None, help="SMILES column name (optional).")
    parser.add_argument(
        "--method",
        default="brics",
        choices=("brics", "recap", "rotatable", "hybrid", "smarts"),
        help="Fragmentation method.",
    )
    parser.add_argument(
        "--bond-smarts",
        action="append",
        default=[],
        help="Bond SMARTS (2-atom pattern) for method=smarts. Can be repeated.",
    )
    parser.add_argument("--min-frag-heavy-atoms", type=int, default=1, help="Minimum heavy atoms per fragment.")
    parser.add_argument("--max-frag-heavy-atoms", type=int, default=None, help="Maximum heavy atoms per fragment.")
    parser.add_argument(
        "--keep-dummies",
        action="store_true",
        help="Keep dummy atoms ([*]) at cut points (default).",
    )
    parser.add_argument(
        "--strip-dummies",
        action="store_true",
        help="Strip dummy atoms after fragmentation.",
    )
    parser.add_argument("--min-count", type=int, default=1, help="Min occurrences to keep a fragment.")
    parser.add_argument("--max-per-parent", type=int, default=None, help="Cap fragments per parent molecule.")
    parser.add_argument("--log-every", type=int, default=5000, help="Progress log cadence (rows).")
    parser.add_argument("--log-every-seconds", type=float, default=30.0, help="Progress log cadence (seconds).")
    parser.add_argument("--max-heavy-atoms", type=int, default=None, help="Skip molecules above this heavy-atom count.")
    parser.add_argument("--warn-slow", type=float, default=5.0, help="Warn if fragmentation exceeds this time (s).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    keep_dummies = True
    if args.strip_dummies:
        keep_dummies = False
    elif args.keep_dummies:
        keep_dummies = True

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    smiles_col = _detect_smiles_column(df, args.smiles_col)
    LOGGER.info("Loaded %d rows from %s (smiles column '%s').", len(df), input_path, smiles_col)

    counts = defaultdict(int)
    parent_counts = defaultdict(set)
    examples = {}
    total = 0
    invalid = 0
    no_frags = 0
    start = time.time()
    last_log = start
    for smi in df[smiles_col].dropna().astype(str):
        total += 1
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid += 1
            continue
        if args.max_heavy_atoms is not None:
            heavy = _heavy_atom_count(mol)
            if heavy > int(args.max_heavy_atoms):
                no_frags += 1
                continue
        frag_start = time.time()
        fragments = _fragment_molecule(
            mol,
            args.method,
            min_heavy=args.min_frag_heavy_atoms,
            max_heavy=args.max_frag_heavy_atoms,
            keep_dummies=keep_dummies,
            bond_smarts=args.bond_smarts,
        )
        frag_elapsed = time.time() - frag_start
        if args.warn_slow and frag_elapsed > float(args.warn_slow):
            LOGGER.warning(
                "Slow fragmentation (%.2fs) for SMILES length %d",
                frag_elapsed,
                len(smi),
            )
        if not fragments:
            no_frags += 1
            continue
        if args.max_per_parent is not None:
            fragments = fragments[: max(1, int(args.max_per_parent))]
        for frag_smiles, _heavy in fragments:
            counts[frag_smiles] += 1
            parent_counts[frag_smiles].add(smi)
            if frag_smiles not in examples:
                examples[frag_smiles] = smi
        if args.log_every and total % int(args.log_every) == 0:
            elapsed = time.time() - start
            rate = total / elapsed if elapsed > 0 else 0.0
            LOGGER.info(
                "Processed %d rows | invalid=%d no_frags=%d | fragments=%d | %.1f rows/s",
                total,
                invalid,
                no_frags,
                len(counts),
                rate,
            )
        if args.log_every_seconds:
            now = time.time()
            if now - last_log >= float(args.log_every_seconds):
                elapsed = now - start
                rate = total / elapsed if elapsed > 0 else 0.0
                LOGGER.info(
                    "Processed %d rows | invalid=%d no_frags=%d | fragments=%d | %.1f rows/s",
                    total,
                    invalid,
                    no_frags,
                    len(counts),
                    rate,
                )
                last_log = now

    rows = []
    for smi, count in counts.items():
        if count < args.min_count:
            continue
        rows.append(
            {
                "smiles": smi,
                "count": int(count),
                "parent_count": int(len(parent_counts.get(smi, set()))),
                "method": args.method,
                "min_frag_heavy_atoms": args.min_frag_heavy_atoms,
                "max_frag_heavy_atoms": args.max_frag_heavy_atoms,
                "keep_dummies": keep_dummies,
                "example_parent": examples.get(smi, ""),
            }
        )

    lib_df = pd.DataFrame(rows).sort_values(["count", "parent_count"], ascending=False)
    library_path = output_dir / "monomer_library.csv"
    lib_df.to_csv(library_path, index=False)
    LOGGER.info("Wrote %d monomer fragments to %s", len(lib_df), library_path)

    dataset_path = output_dir / "monomer_dataset.csv"
    dataset_df = lib_df[["smiles", "count", "parent_count"]].copy()
    dataset_df.to_csv(dataset_path, index=False)
    LOGGER.info("Wrote monomer dataset to %s", dataset_path)


if __name__ == "__main__":
    main()
