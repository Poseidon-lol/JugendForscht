"""
Interfaces to external quantum-chemistry (DFT) evaluations.
===========================================================

During active learning we periodically validate surrogate predictions with
high-fidelity calculations (DFT / GW / experimental measurements).  The
project eventually aims to connect to an ASE + FireWorks queue; for now we
provide a mock interface that mimics asynchronous job submission while
remaining completely self-contained for development/testing.

Key abstractions
----------------

``DFTJobSpec``
    describes a pending quantum-chemistry evaluation (SMILES, level of
    theory, properties to compute).

``DFTResult``
    container for completed calculations (energies, metadata, wall-clock).

``DFTInterface``
    orchestrates submission, polling, and retrieval.  It can run either
    with a user-provided executor (hook into actual infrastructure) or
    fall back to the included ``PseudoDFTSolver`` which produces
    deterministic pseudo-results derived from SMILES hashes.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

__all__ = ["DFTJobSpec", "DFTResult", "DFTInterface", "PseudoDFTSolver"]


@dataclass
class DFTJobSpec:
    smiles: str
    properties: List[str] = field(default_factory=lambda: ["HOMO", "LUMO"])
    level_of_theory: str = "B3LYP/6-31G*"
    charge: int = 0
    multiplicity: int = 1
    metadata: Dict[str, str] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class DFTResult:
    job: DFTJobSpec
    properties: Dict[str, float]
    wall_time: float
    status: str = "success"
    error_message: Optional[str] = None


def _smiles_hash(smiles: str) -> int:
    digest = hashlib.md5(smiles.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class PseudoDFTSolver:
    """Fast deterministic surrogate for quantum-chemistry evaluations.

    The ``run`` method returns property values derived from SMILES hash
    ensuring that repeated calls are stable (useful for testing the active
    learning loop without real DFT workloads).
    """

    def __init__(self, noise_level: float = 0.05, seed: Optional[int] = None) -> None:
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def run(self, job: DFTJobSpec) -> Dict[str, float]:
        base = _smiles_hash(job.smiles) / 0xFFFFFFFF
        props: Dict[str, float] = {}
        for i, name in enumerate(job.properties):
            offset = (i + 1) * 0.3
            deterministic = -7.0 + 4.0 * (base + 0.13 * i) % 1.0
            noise = float(self.rng.normal(0.0, self.noise_level))
            props[name] = deterministic + offset + noise
        if "dot_energy" in job.properties:
            props["dot_energy"] = props.get("LUMO", deterministic) - props.get("HOMO", deterministic)
        return props


class DFTInterface:
    """Queue-like interface to an external DFT executor."""

    def __init__(self, executor: Optional[Callable[[DFTJobSpec], Dict[str, float]]] = None) -> None:
        self.executor = executor or PseudoDFTSolver().run
        self._pending: Dict[str, float] = {}
        self._results: Dict[str, DFTResult] = {}

    # --- public API -----------------------------------------------------
    def submit(self, job: DFTJobSpec) -> str:
        if job.job_id in self._pending or job.job_id in self._results:
            raise ValueError(f"Job with id {job.job_id} already submitted.")
        self._pending[job.job_id] = time.time()
        # For the pseudo solver we execute synchronously; real interfaces
        # could offload to background threads or cluster queues.
        properties = self.executor(job)
        result = DFTResult(job=job, properties=properties, wall_time=time.time() - self._pending[job.job_id])
        self._results[job.job_id] = result
        self._pending.pop(job.job_id, None)
        return job.job_id

    def submit_batch(self, jobs: Iterable[DFTJobSpec]) -> List[str]:
        return [self.submit(job) for job in jobs]

    def fetch(self, job_id: str, *, block: bool = False, poll_interval: float = 5.0) -> Optional[DFTResult]:
        """Retrieve result. For the pseudo solver results are immediate."""

        if job_id in self._results:
            return self._results[job_id]
        if not block:
            return None
        while job_id not in self._results:
            time.sleep(poll_interval)
        return self._results[job_id]

    def pop_completed(self) -> List[DFTResult]:
        """Return all completed results and clear internal storage."""

        results = list(self._results.values())
        self._results.clear()
        return results

    def pending_ids(self) -> List[str]:
        return list(self._pending.keys())

    def reset(self) -> None:
        self._pending.clear()
        self._results.clear()
