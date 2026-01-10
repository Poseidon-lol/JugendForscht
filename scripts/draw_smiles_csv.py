#!/usr/bin/env python
"""
Read a CSV with SMILES, draw 2D structures, and annotate canonical SMILES/IUPAC names.

Default behavior:
  - auto-detects the SMILES column (smiles/smile)
  - writes PNGs to an output directory
  - writes a CSV report with canonical SMILES + (optional) IUPAC name

IUPAC names are fetched from PubChem and require network access.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
except Exception as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(f"RDKit is required for drawing structures: {exc}") from exc


def _resolve_smiles_col(columns: Iterable[str], preferred: Optional[str]) -> str:
    if preferred and preferred in columns:
        return preferred
    for candidate in ("smiles", "smile", "SMILES", "Smiles"):
        if candidate in columns:
            return candidate
    raise ValueError("Could not find a SMILES column; pass --smiles-col to specify it.")


def _parse_size(text: str) -> Tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("Invalid --image-size. Use e.g. '300' or '300,300'.")


def _fetch_iupac_pubchem(smiles: str, *, timeout: float) -> Optional[str]:
    encoded = quote(smiles, safe="")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{encoded}/property/IUPACName/JSON"
    )
    req = Request(url, headers={"User-Agent": "BLLAmen/SMILES-IUPAC"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        props = payload.get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        return props[0].get("IUPACName")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw structures from SMILES in a CSV and annotate IUPAC/SMILES.",
    )
    parser.add_argument("input_csv", help="Path to input CSV containing SMILES.")
    parser.add_argument("--smiles-col", default=None, help="Name of the SMILES column.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs (default: <input_stem>_structures).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: <input_stem>_structures.csv).",
    )
    parser.add_argument(
        "--image-size",
        default="300,300",
        help="Image size in pixels (e.g. 300 or 300,300).",
    )
    parser.add_argument(
        "--no-iupac",
        action="store_true",
        help="Skip IUPAC lookups (canonical SMILES still included).",
    )
    parser.add_argument("--iupac-timeout", type=float, default=10.0, help="IUPAC lookup timeout (s).")
    parser.add_argument("--iupac-delay", type=float, default=0.0, help="Delay between IUPAC requests (s).")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    df = pd.read_csv(input_path)
    smiles_col = _resolve_smiles_col(df.columns, args.smiles_col)

    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit)

    out_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(f"{input_path.stem}_structures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output_csv) if args.output_csv else input_path.with_name(f"{input_path.stem}_structures.csv")
    img_size = _parse_size(args.image_size)

    results = []
    iupac_cache: Dict[str, Optional[str]] = {}
    for idx, row in df.iterrows():
        raw = row.get(smiles_col)
        smi = "" if pd.isna(raw) else str(raw).strip()
        if not smi:
            results.append(
                {
                    "input_smiles": smi,
                    "canonical_smiles": None,
                    "iupac_name": None,
                    "image_path": None,
                    "status": "missing_smiles",
                }
            )
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(
                {
                    "input_smiles": smi,
                    "canonical_smiles": None,
                    "iupac_name": None,
                    "image_path": None,
                    "status": "invalid_smiles",
                }
            )
            continue

        rdDepictor.Compute2DCoords(mol)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        img_path = out_dir / f"{idx:06d}.png"
        Draw.MolToFile(mol, str(img_path), size=img_size)

        iupac = None
        if not args.no_iupac:
            iupac = iupac_cache.get(canonical)
            if iupac is None:
                iupac = _fetch_iupac_pubchem(canonical, timeout=args.iupac_timeout)
                iupac_cache[canonical] = iupac
                if args.iupac_delay:
                    time.sleep(args.iupac_delay)

        results.append(
            {
                "input_smiles": smi,
                "canonical_smiles": canonical,
                "iupac_name": iupac,
                "image_path": str(img_path),
                "status": "ok",
            }
        )

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(out_df)} rows to {out_csv}")
    print(f"Images saved to {out_dir}")


if __name__ == "__main__":
    main()
