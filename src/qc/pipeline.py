from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

HARTREE_TO_EV = 27.211386245988

from src.data.dft_int import DFTJobSpec

from .config import PipelineConfig, QuantumTaskConfig
from .executors import (
    DEFAULT_EXECUTORS,
    ExecutionError,
    ProgramResult,
    QuantumProgramExecutor,
    SemiEmpiricalExecutor,
    resolve_executor,
)
from .geometry import GeometryResult, generate_3d_geometry

if TYPE_CHECKING:
    from .storage import QCResultStore

logger = logging.getLogger(__name__)

PROPERTY_ALIAS = {
    "HOMO": "HOMO_eV",
    "LUMO": "LUMO_eV",
    "gap": "gap_eV",
    "IE": "IE_eV",
    "EA": "EA_eV",
}


@dataclass
class QCResult:
    job: DFTJobSpec
    properties: Dict[str, float]
    status: str
    wall_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    geometry: Optional[GeometryResult] = None


class QCPipeline:
    """High-level workflow from SMILES to QC descriptors."""

    def __init__(
        self,
        config: PipelineConfig,
        executor: Optional[QuantumProgramExecutor] = None,
        result_store: Optional["QCResultStore"] = None,
    ) -> None:
        self.config = config
        primary = executor or resolve_executor(config.quantum.engine)
        if config.quantum.executable and hasattr(primary, "executable"):
            primary.executable = str(config.quantum.executable)
        self.fallback_executor = DEFAULT_EXECUTORS.get("semi_empirical", SemiEmpiricalExecutor())
        self.primary_executor = primary
        self.result_store = result_store

    # ------------------------------------------------------------------
    def run(self, job: DFTJobSpec) -> QCResult:
        start = time.time()
        geometry = generate_3d_geometry(job.smiles, self.config.geometry)
        if not geometry.success:
            return QCResult(
                job=job,
                properties={},
                status="geometry_error",
                wall_time=time.time() - start,
                error_message=geometry.message,
                metadata={"stage": "geometry"},
                geometry=geometry,
            )

        executor = self._select_executor(job)
        program_result: Optional[ProgramResult] = None
        status = "success"
        error_message = None
        try:
            program_result = executor.run(geometry, self._task_config(job))
        except ExecutionError as exc:
            logger.warning("Primary executor %s failed: %s; falling back.", executor.name, exc)
            status = "fallback"
            error_message = str(exc)
            program_result = self.fallback_executor.run(geometry, self._task_config(job))
        except Exception as exc:  # pragma: no cover - unexpected
            logger.exception("QC pipeline crashed for %s", job.smiles)
            return QCResult(
                job=job,
                properties={},
                status="error",
                wall_time=time.time() - start,
                error_message=str(exc),
                metadata={"stage": "execution"},
                geometry=geometry,
            )

        properties = self._post_process(program_result.properties, job)
        metadata = dict(program_result.metadata)
        metadata.update(
            {
                "geometry_energy": geometry.energy,
                "geometry_force_field": geometry.metadata.get("force_field") if geometry.metadata else None,
                "executor": executor.name,
                "fallback_used": status != "success",
                "raw_output": program_result.raw_output[:1024] if program_result.raw_output else None,
            }
        )

        neutral_energy = metadata.get("total_energy")
        if neutral_energy is not None:
            reorg_updates = self._compute_reorganization_energies(geometry, job, neutral_energy, executor)
            if reorg_updates:
                properties.update(reorg_updates)

        qc_result = QCResult(
            job=job,
            properties=properties,
            status=status,
            wall_time=time.time() - start,
            error_message=error_message,
            metadata=metadata,
            geometry=geometry,
        )
        if self.result_store is not None and self.config.store_metadata:
            try:
                self.result_store.append(qc_result)
            except Exception as exc:  # pragma: no cover - IO errors
                logger.error("Failed to store QC result for %s: %s", job.job_id, exc)
        return qc_result

    # ------------------------------------------------------------------
    def _task_config(self, job: DFTJobSpec):
        task = self.config.quantum
        # build a shallow copy with job-specific overrides
        properties = tuple(job.properties or task.properties)
        level = (job.metadata or {}).get("level_of_theory") if job.metadata else None
        clone = type(task)(
            engine=task.engine,
            method=task.method,
            basis=task.basis,
            level_of_theory=level or task.level_of_theory or f"{task.method}/{task.basis}",
            dispersion=task.dispersion,
            properties=properties,
            charge=job.charge,
            multiplicity=job.multiplicity,
            solvent_model=task.solvent_model,
            keywords=task.keywords,
            scratch_dir=task.scratch_dir,
            walltime_limit=task.walltime_limit,
            environment=task.environment,
        )
        return clone

    def _select_executor(self, job: DFTJobSpec) -> QuantumProgramExecutor:
        engine = job.metadata.get("engine") if job.metadata else None
        if engine:
            try:
                return resolve_executor(engine)
            except KeyError:
                logger.warning("Unknown engine '%s', using pipeline default.", engine)
        if self.primary_executor.is_available():
            return self.primary_executor
        logger.info("Primary executor %s not available; using fallback.", self.primary_executor.name)
        return self.fallback_executor

    def _post_process(self, props: Dict[str, float], job: DFTJobSpec) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for key, value in props.items():
            target = PROPERTY_ALIAS.get(key, key)
            results[target] = value
        homo = results.get("HOMO_eV") or props.get("HOMO")
        lumo = results.get("LUMO_eV") or props.get("LUMO")
        if homo is not None and lumo is not None and "gap_eV" not in results:
            results["gap_eV"] = lumo - homo
        if homo is not None and "IE_eV" not in results:
            results["IE_eV"] = -homo
        if lumo is not None and "EA_eV" not in results:
            results["EA_eV"] = -lumo
        # ensure lambda estimates
        if "lambda_hole" not in results:
            results["lambda_hole"] = max(0.15, 0.3 - 0.02 * len(job.properties))
        if "lambda_electron" not in results:
            results["lambda_electron"] = results["lambda_hole"] + 0.05
        return results

    # ------------------------------------------------------------------
    def _compute_reorganization_energies(
        self,
        geometry: GeometryResult,
        job: DFTJobSpec,
        neutral_energy: float,
        executor: QuantumProgramExecutor,
    ) -> Dict[str, float]:
        updates: Dict[str, float] = {}
        base_task = self.config.quantum

        def clone_task(
            charge: int,
            multiplicity: int,
        ) -> QuantumTaskConfig:
            return QuantumTaskConfig(
                engine=base_task.engine,
                executable=base_task.executable,
                method=base_task.method,
                basis=base_task.basis,
                level_of_theory=base_task.level_of_theory,
                dispersion=base_task.dispersion,
                properties=(),
                charge=charge,
                multiplicity=multiplicity,
                solvent_model=base_task.solvent_model,
                keywords=dict(base_task.keywords),
                scratch_dir=base_task.scratch_dir,
                walltime_limit=base_task.walltime_limit,
                environment=dict(base_task.environment),
            )

        def run_energy(task_cfg: QuantumTaskConfig) -> Optional[float]:
            try:
                result = executor.run(geometry, task_cfg)
            except ExecutionError:
                try:
                    result = self.fallback_executor.run(geometry, task_cfg)
                except Exception:
                    return None
            energy = result.metadata.get("total_energy")
            return energy

        # hole (cation)
        cation_task = clone_task(job.charge + 1, max(1, job.multiplicity + 1))
        cation_energy = run_energy(cation_task)
        if cation_energy is not None:
            updates["lambda_hole"] = abs(cation_energy - neutral_energy) * HARTREE_TO_EV

        # electron (anion)
        anion_task = clone_task(job.charge - 1, max(1, job.multiplicity + 1))
        anion_energy = run_energy(anion_task)
        if anion_energy is not None:
            updates["lambda_electron"] = abs(anion_energy - neutral_energy) * HARTREE_TO_EV

        return updates


class AsyncQCManager:
    """Thin adapter exposing a submit API compatible with DFTInterface."""

    def __init__(self, pipeline: QCPipeline, max_workers: Optional[int] = None) -> None:
        self.pipeline = pipeline
        self.pool = ThreadPoolExecutor(max_workers=max_workers or pipeline.config.max_workers)

    def submit(self, job: DFTJobSpec) -> Future:
        future = self.pool.submit(self.pipeline.run, job)
        setattr(future, "job_spec", job)
        return future

    def shutdown(self, wait: bool = True) -> None:
        self.pool.shutdown(wait=wait)
