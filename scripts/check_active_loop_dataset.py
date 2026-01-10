from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None  # type: ignore


def _resolve_smiles_col(columns: Iterable[str], preferred: Optional[str]) -> str:
    cols = list(columns)
    if preferred and preferred in cols:
        return preferred
    for candidate in ("smiles", "smile"):
        if candidate in cols:
            return candidate
    raise KeyError(f"No SMILES column found. Available columns: {cols}")


def _canonicalize_smiles(smiles: str, *, use_rdkit: bool) -> Optional[str]:
    if smiles is None:
        return None
    smi = str(smiles).strip()
    if not smi:
        return None
    if not use_rdkit or Chem is None:
        return smi
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _load_smiles_set(
    path: Path,
    *,
    smiles_col: Optional[str],
    use_rdkit: bool,
) -> tuple[set[str], int]:
    header = pd.read_csv(path, nrows=0)
    col = _resolve_smiles_col(header.columns, smiles_col)
    df = pd.read_csv(path, usecols=[col])
    invalid = 0
    result: set[str] = set()
    for smi in df[col].dropna().astype(str):
        canon = _canonicalize_smiles(smi, use_rdkit=use_rdkit)
        if canon is None:
            invalid += 1
            continue
        result.add(canon)
    return result, invalid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check if active loop molecules already exist in a dataset."
    )
    parser.add_argument(
        "--history",
        default="experiments/active_runs/active_learning_history.csv",
        help="Active loop history CSV.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset CSV to check against.",
    )
    parser.add_argument(
        "--history-smiles-col",
        default=None,
        help="SMILES column name in the history CSV (default: auto-detect).",
    )
    parser.add_argument(
        "--dataset-smiles-col",
        default=None,
        help="SMILES column name in the dataset CSV (default: auto-detect).",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Skip RDKit canonicalization (string match only).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path for overlapping rows.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of overlapping rows to print (default: 5).",
    )
    parser.add_argument(
        "--fail-on-duplicates",
        action="store_true",
        help="Exit with code 2 if duplicates are found.",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    dataset_path = Path(args.dataset)

    use_rdkit = not args.no_canonicalize

    history_df = pd.read_csv(history_path)
    history_col = _resolve_smiles_col(history_df.columns, args.history_smiles_col)

    dataset_set, dataset_invalid = _load_smiles_set(
        dataset_path,
        smiles_col=args.dataset_smiles_col,
        use_rdkit=use_rdkit,
    )

    invalid_hist = 0
    canonical = []
    for smi in history_df[history_col].dropna().astype(str):
        canon = _canonicalize_smiles(smi, use_rdkit=use_rdkit)
        if canon is None:
            invalid_hist += 1
            canonical.append(None)
        else:
            canonical.append(canon)
    history_df["__canonical_smiles"] = canonical
    overlap_mask = history_df["__canonical_smiles"].isin(dataset_set)
    overlap_df = history_df[overlap_mask].copy()

    total_hist = len(history_df)
    unique_hist = history_df["__canonical_smiles"].dropna().nunique()
    overlap_unique = overlap_df["__canonical_smiles"].dropna().nunique()

    print(f"History rows: {total_hist} | unique: {unique_hist}")
    print(f"Dataset unique: {len(dataset_set)}")
    if invalid_hist:
        print(f"History invalid/empty SMILES: {invalid_hist}")
    if dataset_invalid:
        print(f"Dataset invalid/empty SMILES: {dataset_invalid}")
    print(f"Overlap (unique): {overlap_unique} | rows: {len(overlap_df)}")

    if args.show > 0 and not overlap_df.empty:
        print("Sample overlaps:")
        print(overlap_df.head(args.show))

    if args.output:
        output_path = Path(args.output)
        overlap_df.to_csv(output_path, index=False)
        print(f"Wrote overlap rows to {output_path}")

    if args.fail_on_duplicates and overlap_unique > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
