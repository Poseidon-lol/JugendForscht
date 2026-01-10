#!/usr/bin/env python3
"""
Aggregate oligomer QC labels back to monomer-level proxy targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _pick_columns(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for name in preferred:
        if name in df.columns:
            return name
    return None


def _select_best_group(group: pd.DataFrame, prefer_units: List[int]) -> pd.DataFrame:
    for n in prefer_units:
        subset = group[group["n_units"] == n]
        if not subset.empty:
            return subset
    return group


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate oligomer labels to monomer proxy targets.")
    parser.add_argument("--input", required=True, help="Labeled oligomer CSV.")
    parser.add_argument("--output", required=True, help="Output monomer proxy CSV.")
    parser.add_argument("--prefer-units", nargs="+", type=int, default=[3, 2], help="Prefer these oligomer sizes.")
    parser.add_argument("--require-success", action="store_true", help="Use only qc_status==success.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input))
    if "monomer_smiles" not in df.columns or "oligomer_smiles" not in df.columns:
        raise KeyError("Input must contain monomer_smiles and oligomer_smiles columns.")

    if args.require_success and "qc_status" in df.columns:
        df = df[df["qc_status"].fillna("").str.lower().eq("success")]

    homo_col = _pick_columns(df, ["homo", "HOMO_eV", "HOMO"])
    lumo_col = _pick_columns(df, ["lumo", "LUMO_eV", "LUMO"])
    gap_col = _pick_columns(df, ["gap", "gap_eV"])
    if homo_col is None or lumo_col is None:
        raise KeyError("Missing homo/lumo columns in labelled oligomer dataset.")

    df["homo_val"] = pd.to_numeric(df[homo_col], errors="coerce")
    df["lumo_val"] = pd.to_numeric(df[lumo_col], errors="coerce")
    if gap_col:
        df["gap_val"] = pd.to_numeric(df[gap_col], errors="coerce")
    else:
        df["gap_val"] = df["lumo_val"] - df["homo_val"]

    grouped = []
    for monomer, group in df.groupby("monomer_smiles"):
        picked = _select_best_group(group, [int(x) for x in args.prefer_units])
        # aggregate by mean in case multiple oligomers of same size exist
        row = {
            "smiles": monomer,
            "homo": float(np.nanmean(picked["homo_val"])) if not picked["homo_val"].isna().all() else None,
            "lumo": float(np.nanmean(picked["lumo_val"])) if not picked["lumo_val"].isna().all() else None,
            "gap": float(np.nanmean(picked["gap_val"])) if not picked["gap_val"].isna().all() else None,
            "n_units": int(picked["n_units"].iloc[0]) if not picked.empty else None,
            "oligomer_smiles": picked["oligomer_smiles"].iloc[0] if not picked.empty else None,
        }
        grouped.append(row)

    out_df = pd.DataFrame(grouped)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} monomer proxy rows to {args.output}")


if __name__ == "__main__":
    main()
