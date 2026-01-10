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
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

__all__ = ["DFTJobSpec", "DFTResult", "DFTInterface", "PseudoDFTSolver"]


def _is_timeout_exception(exc: Exception) -> bool:
    """Return True if exc looks like a timeout from futures/threads."""
    if isinstance(exc, TimeoutError):
        return True
    return exc.__class__.__name__ == "TimeoutError"


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
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        self._pending: Dict[str, Future | float] = {}
        self._submitted_at: Dict[str, float] = {}
        self._results: Dict[str, DFTResult] = {}
        self._lock = threading.Lock()
        self._logger = logging.getLogger(self.__class__.__name__)

    # --- public API -----------------------------------------------------
    def submit(self, job: DFTJobSpec) -> str:
        with self._lock:
            if job.job_id in self._pending or job.job_id in self._results:
                raise ValueError(f"Job with id {job.job_id} already submitted.")
            submitted_at = time.time()
            self._submitted_at[job.job_id] = submitted_at

            if hasattr(self.executor, "submit"):
                future = getattr(self.executor, "submit")(job)
                if not isinstance(future, Future):
                    raise TypeError("Executor.submit must return concurrent.futures.Future")
                self._pending[job.job_id] = future
                future.add_done_callback(lambda fut, jid=job.job_id: self._collect_future(jid, fut))
            else:
                self._pending[job.job_id] = submitted_at
                properties = self.executor(job)  # type: ignore[operator]
                result = self._coerce_result(job, properties, submitted_at)
                self._results[job.job_id] = result
                self._pending.pop(job.job_id, None)
                self._submitted_at.pop(job.job_id, None)
        return job.job_id

    def submit_batch(self, jobs: Iterable[DFTJobSpec]) -> List[str]:
        return [self.submit(job) for job in jobs]

    def fetch(self, job_id: str, *, block: bool = False, poll_interval: float = 5.0) -> Optional[DFTResult]:
        """Retrieve result. For the pseudo solver results are immediate."""

        while True:
            with self._lock:
                if job_id in self._results:
                    return self._results[job_id]
                future = self._pending.get(job_id)
            if not block or future is None:
                return None
            if isinstance(future, Future):
                try:
                    raw = future.result(timeout=poll_interval)
                    self._collect_future(job_id, future, raw_value=raw)
                except Exception as exc:  # pragma: no cover - propagation
                    if _is_timeout_exception(exc):
                        if not future.done():
                            continue
                        # The future completed right as we timed out; re-fetch without timeout.
                        try:
                            raw = future.result()
                        except Exception as final_exc:
                            self._collect_exception(job_id, final_exc)
                        else:
                            self._collect_future(job_id, future, raw_value=raw)
                        continue
                    self._collect_exception(job_id, exc)
            else:
                time.sleep(poll_interval)

    def pop_completed(self) -> List[DFTResult]:
        """Return all completed results and clear internal storage."""

        with self._lock:
            results = list(self._results.values())
            self._results.clear()
        return results

    def pending_ids(self) -> List[str]:
        with self._lock:
            return list(self._pending.keys())

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._results.clear()
            self._submitted_at.clear()

    # --- internal helpers -----------------------------------------------
    def _collect_future(self, job_id: str, future: Future, *, raw_value: Any = None) -> None:
        try:
            value = raw_value if raw_value is not None else future.result()
        except Exception as exc:  # pragma: no cover - executor exceptions
            self._collect_exception(job_id, exc)
            return
        with self._lock:
            job = future.job if hasattr(future, "job") else None  # type: ignore[attr-defined]
            job_spec = job if isinstance(job, DFTJobSpec) else None
            if job_spec is None:
                # fallback to metadata we stored on submission
                job_spec = getattr(future, "job_spec", None)
            if job_spec is None:
                self._logger.warning("Future completed without attached job spec (%s)", job_id)
                submitted_at = self._submitted_at.get(job_id, time.time())
                job_spec = DFTJobSpec(smiles="UNKNOWN", job_id=job_id)  # coarse fallback
            else:
                submitted_at = self._submitted_at.get(job_id, time.time())
            result = self._coerce_result(job_spec, value, submitted_at)
            self._results[job_id] = result
            self._pending.pop(job_id, None)
            self._submitted_at.pop(job_id, None)

    def _collect_exception(self, job_id: str, exc: Exception) -> None:
        with self._lock:
            submitted_at = self._submitted_at.get(job_id, time.time())
            job_spec = None
            pending = self._pending.get(job_id)
            if isinstance(pending, Future):
                job_spec = getattr(pending, "job_spec", None)
            if job_spec is None:
                job_spec = DFTJobSpec(smiles="UNKNOWN", job_id=job_id)
            result = DFTResult(
                job=job_spec,
                properties={},
                wall_time=time.time() - submitted_at,
                status="error",
                error_message=str(exc),
                metadata={},
            )
            self._results[job_id] = result
            self._pending.pop(job_id, None)
            self._submitted_at.pop(job_id, None)

    def _coerce_result(self, job: DFTJobSpec, raw_value: Any, submitted_at: float) -> DFTResult:
        wall_time = time.time() - submitted_at
        if isinstance(raw_value, DFTResult):
            return raw_value
        if hasattr(raw_value, "properties") and hasattr(raw_value, "status"):
            metadata = getattr(raw_value, "metadata", {})
            status = getattr(raw_value, "status", "success")
            error = getattr(raw_value, "error_message", None)
            return DFTResult(job=job, properties=dict(raw_value.properties), wall_time=wall_time, status=status, error_message=error, metadata=dict(metadata))
        if isinstance(raw_value, dict):
            return DFTResult(job=job, properties=raw_value, wall_time=wall_time, status="success", metadata={})
        raise TypeError(f"Executor returned unsupported type: {type(raw_value)!r}")
