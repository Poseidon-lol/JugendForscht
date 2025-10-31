#!/usr/bin/env python3
"""
Build merged OSC datasets from QM9, QMugs, and df_62k sources.

The script canonicalises SMILES, harmonises energy units, and writes two
CSV files:

1. JT-VAE pool (`jtvae_pool.csv`): union of all molecules with the best
   available frontier-orbital estimates and optional low-cost gaps.
2. Surrogate labelled (`surrogate_labelled.csv`): subset containing
   high-quality HOMO/LUMO/gap labels filtered to OSC-relevant ranges.

Usage:
    python scripts/build_osc_datasets.py \
        --output-dir data/processed
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG = 0.52917721092
BOHR3_TO_A3 = BOHR_TO_ANG**3


def canonicalise_smiles(smiles: Optional[str]) -> Optional[str]:
    """Return canonical isomer-aware SMILES or None if invalid."""
    if not isinstance(smiles, str):
        return None
    stripped = smiles.strip()
    if not stripped:
        return None
    mol = Chem.MolFromSmiles(stripped)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _clean_numeric_list(values: Iterable[float | int | None]) -> List[float]:
    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            fval = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(fval):
            continue
        cleaned.append(fval)
    return cleaned


def load_qm9(path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path)
    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        smiles = canonicalise_smiles(row.get("smiles"))
        if smiles is None:
            continue
        homo = row.get("HOMO")
        lumo = row.get("LUMO")
        gap = row.get("gap")
        dipole = row.get("dipole_moment")
        polar = row.get("polarizability")

        records.append(
            {
                "smiles": smiles,
                "source_tag": "qm9_dft_b3lyp",
                "quality_rank": 2,
                "HOMO_eV": float(homo) * HARTREE_TO_EV if pd.notna(homo) else None,
                "LUMO_eV": float(lumo) * HARTREE_TO_EV if pd.notna(lumo) else None,
                "gap_eV": float(gap) * HARTREE_TO_EV if pd.notna(gap) else None,
                "gap_lowcost_eV": None,
                "IE_eV": float(-homo) * HARTREE_TO_EV if pd.notna(homo) else None,
                "EA_eV": float(-lumo) * HARTREE_TO_EV if pd.notna(lumo) else None,
                "dipole_D": float(dipole) if pd.notna(dipole) else None,
                "polarizability_A3": float(polar) * BOHR3_TO_A3 if pd.notna(polar) else None,
                "n_atoms": None,
                "notes": "QM9 B3LYP/6-31G(2df,p)",
            }
        )
    return records


def load_qmugs(path: Path) -> List[Dict[str, object]]:
    df = pd.read_csv(path)
    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        smiles = canonicalise_smiles(row.get("smiles"))
        if smiles is None:
            continue
        dft_gap = row.get("DFT_HOMO_LUMO_GAP")
        gfn_gap = row.get("GFN2_HOMO_LUMO_GAP")

        records.append(
            {
                "smiles": smiles,
                "source_tag": "qmugs_dft_gap",
                "quality_rank": 2,
                "HOMO_eV": None,
                "LUMO_eV": None,
                "gap_eV": float(dft_gap) * HARTREE_TO_EV if pd.notna(dft_gap) else None,
                "gap_lowcost_eV": float(gfn_gap) * HARTREE_TO_EV if pd.notna(gfn_gap) else None,
                "dipole_D": None,
                "polarizability_A3": None,
                "n_atoms": None,
                "IE_eV": None,
                "EA_eV": None,
                "notes": "QMugs DFT (gap only)",
            }
        )
    return records


def load_df62k(path: Path) -> List[Dict[str, object]]:
    df = pd.read_json(path, orient="split")
    records: List[Dict[str, object]] = []

    priority_fields: List[Tuple[str, str, str, int]] = [
        ("energies_occ_gw_qzvp", "energies_unocc_gw_qzvp", "df62k_gw_qzvp", 4),
        ("energies_occ_pbe0_vac_qzvp", "energies_unocc_pbe0_vac_qzvp", "df62k_pbe0_qzvp", 3),
        ("energies_occ_pbe0_vac_tzvp", "energies_unocc_pbe0_vac_tzvp", "df62k_pbe0_tzvp", 3),
        ("energies_occ_pbe", "energies_unocc_pbe", "df62k_pbe", 2),
    ]

    for _, row in df.iterrows():
        raw_smiles = row.get("canonical_smiles")
        if isinstance(raw_smiles, str):
            raw_smiles = raw_smiles.strip()
        smiles = canonicalise_smiles(raw_smiles)
        if smiles is None:
            continue

        homo_val = lumo_val = gap_val = None
        source_tag = None
        quality_rank = 0

        for occ_field, unocc_field, tag, rank in priority_fields:
            occ_values = row.get(occ_field)
            unocc_values = row.get(unocc_field)
            if not isinstance(occ_values, (list, tuple)) or not isinstance(unocc_values, (list, tuple)):
                continue
            occ_clean = _clean_numeric_list(occ_values)
            unocc_clean = _clean_numeric_list(unocc_values)
            if not occ_clean or not unocc_clean:
                continue
            homo_candidate = max(occ_clean)
            lumo_candidate = min(unocc_clean)
            if homo_candidate is None or lumo_candidate is None:
                continue
            homo_val = homo_candidate
            lumo_val = lumo_candidate
            gap_val = lumo_candidate - homo_candidate
            source_tag = tag
            quality_rank = rank
            break

        if homo_val is None or lumo_val is None:
            continue

        records.append(
            {
                "smiles": smiles,
                "source_tag": source_tag or "df62k_unknown",
                "quality_rank": quality_rank or 2,
                "HOMO_eV": homo_val,
                "LUMO_eV": lumo_val,
                "gap_eV": gap_val,
                "gap_lowcost_eV": None,
                "IE_eV": -homo_val if homo_val is not None else None,
                "EA_eV": -lumo_val if lumo_val is not None else None,
                "dipole_D": None,
                "polarizability_A3": None,
                "n_atoms": int(row.get("number_of_atoms")) if pd.notna(row.get("number_of_atoms")) else None,
                "notes": "DFT/MBPT frontier energies",
            }
        )
    return records


def aggregate_records(records: Iterable[Dict[str, object]]) -> pd.DataFrame:
    merged: Dict[str, Dict[str, object]] = {}

    def update(entry: Dict[str, object], key: str, value: Optional[float], rank: int, source: str) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        current_rank = entry.get(f"_{key}_rank", -1)
        if rank > current_rank or entry.get(key) is None:
            entry[key] = float(value)
            entry[f"{key}_source"] = source
            entry[f"_{key}_rank"] = rank

    for record in records:
        smiles = record["smiles"]
        entry = merged.setdefault(
            smiles,
            {
                "smiles": smiles,
                "source_tags": set(),
                "HOMO_eV": None,
                "LUMO_eV": None,
                "gap_eV": None,
                "gap_lowcost_eV": None,
                "IE_eV": None,
                "EA_eV": None,
                "dipole_D": None,
                "polarizability_A3": None,
                "n_atoms": None,
            },
        )
        source_tag = str(record.get("source_tag", "unknown"))
        quality_rank = int(record.get("quality_rank", 0))
        entry["source_tags"].add(source_tag)

        update(entry, "HOMO_eV", record.get("HOMO_eV"), quality_rank, source_tag)
        update(entry, "LUMO_eV", record.get("LUMO_eV"), quality_rank, source_tag)
        update(entry, "gap_eV", record.get("gap_eV"), quality_rank, source_tag)
        update(entry, "gap_lowcost_eV", record.get("gap_lowcost_eV"), quality_rank, source_tag)
        update(entry, "IE_eV", record.get("IE_eV"), quality_rank, source_tag)
        update(entry, "EA_eV", record.get("EA_eV"), quality_rank, source_tag)
        update(entry, "dipole_D", record.get("dipole_D"), quality_rank, source_tag)
        update(entry, "polarizability_A3", record.get("polarizability_A3"), quality_rank, source_tag)
        update(entry, "n_atoms", record.get("n_atoms"), quality_rank, source_tag)

    rows: List[Dict[str, object]] = []
    for entry in merged.values():
        homo = entry.get("HOMO_eV")
        lumo = entry.get("LUMO_eV")
        gap = entry.get("gap_eV")
        if gap is None and homo is not None and lumo is not None:
            entry["gap_eV"] = lumo - homo
            entry["gap_eV_source"] = (
                "computed_from_frontier"
                if entry.get("HOMO_eV_source") != entry.get("LUMO_eV_source")
                else entry.get("HOMO_eV_source")
            )
        if entry.get("IE_eV") is None and homo is not None:
            entry["IE_eV"] = -homo
            entry["IE_eV_source"] = entry.get("HOMO_eV_source")
        if entry.get("EA_eV") is None and lumo is not None:
            entry["EA_eV"] = -lumo
            entry["EA_eV_source"] = entry.get("LUMO_eV_source")
        entry["source_tags"] = ",".join(sorted(entry["source_tags"]))
        for cleanup_key in list(entry.keys()):
            if cleanup_key.startswith("_") and cleanup_key.endswith("_rank"):
                entry.pop(cleanup_key)
        rows.append(entry)
    return pd.DataFrame(rows)


def build_labelled(df: pd.DataFrame) -> pd.DataFrame:
    labelled = df.dropna(subset=["HOMO_eV", "LUMO_eV", "gap_eV"]).copy()

    # OSC-oriented filters (loose enough to cover both p- and n-type)
    mask_gap = labelled["gap_eV"].between(1.0, 4.0)
    mask_homo = labelled["HOMO_eV"].between(-8.0, -3.0)
    mask_lumo = labelled["LUMO_eV"].between(-5.5, 1.0)

    labelled = labelled[mask_gap & mask_homo & mask_lumo]
    labelled = labelled.reset_index(drop=True)
    return labelled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge QM9/QMugs/df_62k into OSC datasets.")
    parser.add_argument("--qm9-path", type=Path, default=Path("data/raw/qm9_dataset.csv"))
    parser.add_argument("--qmugs-path", type=Path, default=Path("data/raw/QMugs_curatedDFT.csv"))
    parser.add_argument("--df62k-path", type=Path, default=Path("m1507656/df_62k.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pool-name", default="jtvae_pool.csv")
    parser.add_argument("--labelled-name", default="surrogate_labelled.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_records: List[Dict[str, object]] = []
    all_records.extend(load_qm9(args.qm9_path))
    all_records.extend(load_qmugs(args.qmugs_path))
    all_records.extend(load_df62k(args.df62k_path))

    combined_df = aggregate_records(all_records)
    labelled_df = build_labelled(combined_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = args.output_dir / args.pool_name
    labelled_path = args.output_dir / args.labelled_name

    combined_df.sort_values("smiles").reset_index(drop=True).to_csv(pool_path, index=False)
    labelled_df.sort_values("smiles").reset_index(drop=True).to_csv(labelled_path, index=False)

    print(f"JT-VAE pool written to: {pool_path} ({len(combined_df)} molecules)")
    print(f"Surrogate labelled written to: {labelled_path} ({len(labelled_df)} molecules)")


if __name__ == "__main__":
    main()
