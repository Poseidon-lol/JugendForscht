from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass
class GeometryConfig:
    """Settings for SMILES → 3D geometry preparation."""

    force_field: str = "MMFF94"
    max_iterations: int = 500
    embed_tries: int = 10
    random_seed: Optional[int] = 42
    optimize: bool = True
    use_etkdg: bool = True


@dataclass
class QuantumTaskConfig:
    """Describe a single-point (or trajectory) calculation."""

    engine: str = "psi4"
    executable: Optional[str] = None
    method: str = "wb97xd"
    basis: str = "def2-SVP"
    level_of_theory: Optional[str] = None
    dispersion: Optional[str] = None
    properties: Tuple[str, ...] = ("HOMO", "LUMO", "gap", "IE", "EA", "dipole", "polarizability")
    charge: int = 0
    multiplicity: int = 1
    solvent_model: Optional[str] = None
    keywords: Mapping[str, Any] = field(default_factory=dict)
    scratch_dir: Path = Path("qc_runs")
    walltime_limit: Optional[int] = None  # seconds
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Bundle geometry and quantum task settings with orchestration options."""

    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    quantum: QuantumTaskConfig = field(default_factory=QuantumTaskConfig)
    work_dir: Path = Path("qc_runs")
    max_workers: int = 2
    poll_interval: float = 10.0
    cleanup_workdir: bool = False
    store_metadata: bool = True
    allow_fallback: bool = True
    tracked_properties: Sequence[str] = (
        "HOMO",
        "LUMO",
        "gap",
        "IE",
        "EA",
        "lambda_hole",
        "lambda_electron",
        "dipole",
        "polarizability",
        "packing_score",
        "stability_index",
    )
