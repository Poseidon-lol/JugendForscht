#!/usr/bin/env python3
"""
Filter QC results based on configurable property windows.

Usage:
    python scripts/filter_qc_results.py \
        --qc data/processed/qc_results.csv \
        --filter configs/filter_rules.yaml \
        --output data/processed/surrogate_labelled_filtered.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def filter_dataframe(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    targets: Dict[str, Dict[str, float]] = cfg.get("targets", {})
    result = df.copy()

    def extract_col(row: pd.Series, column: str):
        if column in row:
            return row[column]
        try:
            props = json.loads(row.get("properties_json", "{}"))
            if column in props:
                return props[column]
            # legacy keys from ORCA: HOMO_eV etc.
            for alt_key in ("HOMO_eV", "LUMO_eV", "gap_eV", "IE_eV", "EA_eV"):
                if column == alt_key and alt_key in props:
                    return props[alt_key]
        except Exception:
            pass
        return None

    for column in targets.keys():
        result[column] = result.apply(lambda r: extract_col(r, column), axis=1)

    mask = pd.Series(True, index=result.index)
    for column, bounds in targets.items():
        col_series = pd.to_numeric(result[column], errors="coerce")
        min_val = bounds.get("min", float("-inf"))
        max_val = bounds.get("max", float("inf"))
        mask &= (col_series >= min_val) & (col_series <= max_val)

    if "qc_status" in result.columns:
        mask &= result["qc_status"].fillna("").str.lower().eq("success")

    return result[mask].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter QC results.")
    parser.add_argument("--qc", required=True, help="CSV with QC results (from loop or store).")
    parser.add_argument("--filter", required=True, help="YAML with target ranges.")
    parser.add_argument("--output", required=True, help="Output CSV with filtered rows.")
    args = parser.parse_args()

    qc_path = Path(args.qc)
    filter_path = Path(args.filter)
    output_path = Path(args.output)

    df = pd.read_csv(qc_path)
    cfg = load_config(filter_path)
    filtered = filter_dataframe(df, cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    print(f"Filtered {len(filtered)} rows out of {len(df)} into {output_path}")


if __name__ == "__main__":
    main()
