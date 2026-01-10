#!/usr/bin/env python3
"""
Run QC (ORCA) on monomer fragments to generate conditioning targets.

Writes a labeled CSV with HOMO/LUMO/gap (and other QC properties) plus metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from concurrent.futures import as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dft_int import DFTJobSpec
from src.qc.config import GeometryConfig, PipelineConfig, QuantumTaskConfig
from src.qc.pipeline import AsyncQCManager, QCPipeline


LOGGER = logging.getLogger("label_monomer_dataset")


PROPERTY_COLUMNS = [
    "HOMO_eV",
    "LUMO_eV",
    "gap_eV",
    "IE_eV",
    "EA_eV",
    "lambda_hole",
    "lambda_electron",
    "dipole",
    "polarizability",
    "lambda_max_nm",
    "oscillator_strength",
    "packing_score",
    "stability_index",
]

QC_COLUMNS = [
    "qc_status",
    "qc_wall_time",
    "qc_error",
    "qc_metadata",
    "basis",
    "total_energy",
    "level_of_theory",
]


def _detect_smiles_column(columns: Iterable[str], override: Optional[str]) -> str:
    if override and override in columns:
        return override
    for candidate in ("smiles", "smile", "SMILES"):
        if candidate in columns:
            return candidate
    raise KeyError("Input CSV must contain a smiles/smile/SMILES column.")


def _is_missing_smiles(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value).strip()
    return not text or text.lower() in ("nan", "none")


def _load_pipeline_config(path: Path) -> PipelineConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = PipelineConfig()
    geometry_cfg = GeometryConfig(**data.get("geometry", {}))
    quantum_cfg = QuantumTaskConfig(**data.get("quantum", {}))
    pipeline = PipelineConfig(
        geometry=geometry_cfg,
        quantum=quantum_cfg,
        work_dir=Path(data.get("work_dir", defaults.work_dir)),
        max_workers=data.get("max_workers", defaults.max_workers),
        poll_interval=data.get("poll_interval", defaults.poll_interval),
        cleanup_workdir=data.get("cleanup_workdir", defaults.cleanup_workdir),
        store_metadata=data.get("store_metadata", defaults.store_metadata),
        allow_fallback=data.get("allow_fallback", defaults.allow_fallback),
        tracked_properties=tuple(data.get("tracked_properties", defaults.tracked_properties)),
    )
    if isinstance(pipeline.work_dir, str):
        pipeline.work_dir = Path(pipeline.work_dir)
    if isinstance(pipeline.quantum.scratch_dir, str):
        pipeline.quantum.scratch_dir = Path(pipeline.quantum.scratch_dir)
    pipeline.quantum.properties = tuple(pipeline.quantum.properties)
    pipeline.tracked_properties = tuple(pipeline.tracked_properties)
    return pipeline


def _build_job(smiles: str, pipe_cfg: PipelineConfig) -> DFTJobSpec:
    level = pipe_cfg.quantum.level_of_theory or f"{pipe_cfg.quantum.method}/{pipe_cfg.quantum.basis}"
    return DFTJobSpec(
        smiles=smiles,
        properties=list(pipe_cfg.quantum.properties),
        charge=pipe_cfg.quantum.charge,
        multiplicity=pipe_cfg.quantum.multiplicity,
        metadata={
            "engine": pipe_cfg.quantum.engine,
            "level_of_theory": level,
        },
    )


def _merge_result(row: Dict[str, object], result) -> Dict[str, object]:
    out = dict(row)
    out["qc_status"] = result.status
    out["qc_wall_time"] = result.wall_time
    out["qc_error"] = result.error_message
    if result.metadata:
        out["basis"] = result.metadata.get("basis")
        out["total_energy"] = result.metadata.get("total_energy")
        out["level_of_theory"] = result.metadata.get("level_of_theory")
        out["qc_metadata"] = json.dumps(result.metadata, ensure_ascii=False)
    for key, value in result.properties.items():
        out[key] = value
    homo = result.properties.get("HOMO_eV") or result.properties.get("HOMO")
    lumo = result.properties.get("LUMO_eV") or result.properties.get("LUMO")
    gap = result.properties.get("gap_eV") or result.properties.get("gap")
    if homo is not None:
        out["homo"] = homo
    if lumo is not None:
        out["lumo"] = lumo
    if gap is None and homo is not None and lumo is not None:
        gap = lumo - homo
    if gap is not None:
        out["gap"] = gap
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Label monomer fragments with QC properties.")
    parser.add_argument("--input", required=True, help="Monomer CSV (from extract_monomer_library.py).")
    parser.add_argument("--output", required=True, help="Output labeled CSV.")
    parser.add_argument("--qc-config", default="configs/qc_pipeline.yaml", help="QC pipeline config.")
    parser.add_argument("--smiles-col", default=None, help="SMILES column name (optional).")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--workers", type=int, default=None, help="Override max_workers from qc config.")
    parser.add_argument("--resume", action="store_true", help="Skip SMILES already in output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    input_path = Path(args.input)
    output_path = Path(args.output)
    qc_path = Path(args.qc_config)

    if output_path.exists():
        if args.overwrite:
            output_path.unlink()
        elif not args.resume:
            raise FileExistsError(f"Output exists: {output_path}. Use --resume or --overwrite.")

    df = pd.read_csv(input_path)
    smiles_col = _detect_smiles_column(df.columns, args.smiles_col)
    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    if "status" in df.columns:
        before = len(df)
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()
        dropped = before - len(df)
        if dropped:
            LOGGER.info("Filtered %d rows with status != ok.", dropped)
    before = len(df)
    df = df[~df[smiles_col].apply(_is_missing_smiles)].copy()
    dropped = before - len(df)
    if dropped:
        LOGGER.info("Dropped %d rows with missing SMILES in '%s'.", dropped, smiles_col)
    LOGGER.info("Loaded %d rows from %s (smiles column '%s').", len(df), input_path, smiles_col)

    done = set()
    if output_path.exists() and args.resume:
        try:
            prev = pd.read_csv(output_path, usecols=[smiles_col])
            done = set(prev[smiles_col].dropna().astype(str))
            LOGGER.info("Resuming: %d SMILES already labeled.", len(done))
        except Exception as exc:
            LOGGER.warning("Failed to load resume file (%s); proceeding without resume.", exc)

    pipe_cfg = _load_pipeline_config(qc_path)
    if args.workers is not None:
        pipe_cfg.max_workers = int(args.workers)
    pipeline = QCPipeline(pipe_cfg)
    manager = AsyncQCManager(pipeline, max_workers=pipe_cfg.max_workers)

    base_cols = list(df.columns)
    out_fields = list(base_cols)
    for col in QC_COLUMNS + PROPERTY_COLUMNS + ["homo", "lumo", "gap"]:
        if col not in out_fields:
            out_fields.append(col)

    writer = None
    out_handle = None
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = output_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_handle, fieldnames=out_fields)
        writer.writeheader()

    futures = {}
    submitted = 0
    for _, row in df.iterrows():
        if _is_missing_smiles(row[smiles_col]):
            continue
        smi = str(row[smiles_col]).strip()
        if smi in done:
            continue
        job = _build_job(smi, pipe_cfg)
        future = manager.submit(job)
        futures[future] = row.to_dict()
        submitted += 1

    LOGGER.info("Submitted %d QC jobs.", submitted)

    try:
        for idx, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                row.update(
                    {
                        "qc_status": "error",
                        "qc_wall_time": None,
                        "qc_error": str(exc),
                    }
                )
                out_row = row
            else:
                out_row = _merge_result(row, result)

            if writer is None:
                out_handle = output_path.open("a", newline="", encoding="utf-8")
                writer = csv.DictWriter(out_handle, fieldnames=out_fields)
            writer.writerow({k: out_row.get(k) for k in out_fields})
            if out_handle is not None:
                out_handle.flush()

            if idx % 10 == 0:
                LOGGER.info("Completed %d/%d QC jobs.", idx, submitted)
    finally:
        manager.shutdown()
        if out_handle is not None:
            out_handle.close()


if __name__ == "__main__":
    main()
