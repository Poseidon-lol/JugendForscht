from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Iterable, Union

from src.data.dft_int import DFTResult

from .pipeline import QCResult


class QCResultStore:
    """Append-only CSV/JSONL storage for QC results."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fieldnames = [
            "job_id",
            "smiles",
            "status",
            "wall_time",
            "level_of_theory",
            "properties_json",
            "metadata_json",
        ]

    def append(self, result: Union[QCResult, DFTResult]) -> None:
        row = self._convert_row(result)
        with self._lock:
            file_exists = self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

    def append_many(self, results: Iterable[Union[QCResult, DFTResult]]) -> None:
        rows = [self._convert_row(res) for res in results]
        if not rows:
            return
        with self._lock:
            file_exists = self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self._fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)

    def _convert_row(self, result: Union[QCResult, DFTResult]) -> dict:
        job = result.job
        level = job.metadata.get("level_of_theory") if job.metadata else None
        if level is None and result.metadata:
            level = result.metadata.get("level_of_theory") or result.metadata.get("method")
        return {
            "job_id": job.job_id,
            "smiles": job.smiles,
            "status": result.status,
            "wall_time": f"{result.wall_time:.3f}",
            "level_of_theory": level,
            "properties_json": json.dumps(result.properties, ensure_ascii=False),
            "metadata_json": json.dumps(result.metadata, ensure_ascii=False),
        }
