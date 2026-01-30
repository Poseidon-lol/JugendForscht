from __future__ import annotations

import logging
import re
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from rdkit import Chem

from .config import QuantumTaskConfig
from .geometry import GeometryResult

logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Raised when an external QC program fails."""


@dataclass
class ProgramResult:
    properties: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[str] = None


class QuantumProgramExecutor(Protocol):
    name: str

    def is_available(self) -> bool:
        ...

    def run(self, geometry: GeometryResult, task: QuantumTaskConfig) -> ProgramResult:
        ...


class ExternalProgramExecutor:
    """Base helper for external QC engines."""

    name: str = "external"
    executable: str = ""
    work_dir: Optional[Path] = None
    cleanup: bool = True

    def __init__(self, command: Optional[str] = None) -> None:
        if command is not None:
            self.executable = command

    # -- public API -----------------------------------------------------
    def is_available(self) -> bool:
        return bool(shutil.which(self.executable))

    def run(self, geometry: GeometryResult, task: QuantumTaskConfig) -> ProgramResult:
        if not self.is_available():
            raise ExecutionError(f"Executable '{self.executable}' not found on PATH.")
        cleanup = getattr(task, "cleanup_workdir", getattr(self, "cleanup", True))
        base_dir = getattr(task, "work_dir", None) or self.work_dir
        tmp_ctx = None
        if base_dir:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            workdir = Path(tempfile.mkdtemp(prefix=f"{self.name}_", dir=base_dir))
        else:
            tmp_ctx = tempfile.TemporaryDirectory(prefix=f"{self.name}_")
            workdir = Path(tmp_ctx.__enter__())
        try:
            input_path = workdir / "input.inp"
            output_path = workdir / "output.out"
            self._write_input(input_path, geometry, task)
            self._execute(task, input_path, output_path, workdir)
            properties, metadata = self._parse_output(output_path, task)
            metadata["engine"] = self.name
            metadata["method"] = task.method
            metadata["basis"] = task.basis
            metadata["level_of_theory"] = task.level_of_theory if hasattr(task, "level_of_theory") else f"{task.method}/{task.basis}"
            metadata["workdir"] = str(workdir)
            raw_text = output_path.read_text(encoding="utf-8", errors="replace")
            return ProgramResult(properties=properties, metadata=metadata, raw_output=raw_text)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.__exit__(None, None, None)
            elif cleanup:
                shutil.rmtree(workdir, ignore_errors=True)

    # -- hooks ----------------------------------------------------------
    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        raise NotImplementedError

    def _execute(self, task: QuantumTaskConfig, input_path: Path, output_path: Path, workdir: Path) -> None:
        env = dict(os.environ)
        if task.environment:
            env.update(task.environment)
        exe_parent = Path(self.executable).parent
        env["PATH"] = f"{exe_parent};{env.get('PATH', '')}"
        cmd = [self.executable, input_path.name]
        logger.debug("Running %s with command: %s", self.name, cmd)
        try:
            with output_path.open("w", encoding="utf-8") as out_f:
                completed = subprocess.run(
                    cmd,
                    cwd=workdir,
                    check=True,
                    stdout=out_f,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env if env else None,
                    timeout=task.walltime_limit,
                )
        except FileNotFoundError as exc:  # pragma: no cover - depends on system
            raise ExecutionError(f"{self.executable} not found: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(f"{self.name} exceeded walltime limit ({task.walltime_limit}s)") from exc
        except subprocess.CalledProcessError as exc:
            raise ExecutionError(f"{self.name} failed with return code {exc.returncode}: {exc.stderr}") from exc

        if completed.stderr:
            (workdir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raise NotImplementedError


class Psi4Executor(ExternalProgramExecutor):
    name = "psi4"
    executable = "psi4"

    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        coords = ""
        if geometry.xyz:
            lines = geometry.xyz.splitlines()[2:]
            coords = "\n".join(line for line in lines if line.strip())
        props = ", ".join(f"'{p}'" for p in task.properties)
        threads = 1
        try:
            import multiprocessing

            threads = max(1, multiprocessing.cpu_count() // 2)
        except Exception:
            threads = 1

        template = """memory 2 GB
set_num_threads {threads}
set scf_type df
set basis {basis}
set reference {reference}
molecule {{
{charge} {multiplicity}
{coords}
}}
set {{
    basis {basis}
}}
energy('{method}')
properties('{method}', properties=[{props}])
"""
        script = template.format(
            threads=threads,
            basis=task.basis,
            reference="uhf" if task.multiplicity != 1 else "rhf",
            charge=task.charge,
            multiplicity=task.multiplicity,
            coords=coords,
            method=task.method,
            props=props,
        )
        path.write_text(script.strip(), encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        props: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        for line in text.splitlines():
            if "Total Energy =" in line:
                try:
                    energy = float(line.split()[-2])
                    metadata["total_energy"] = energy
                except Exception:
                    continue
            if "Dipole Moment" in line and "Debye" in line:
                try:
                    dipole = float(line.split()[-2])
                    props["dipole"] = dipole
                except Exception:
                    continue
        return props, metadata


class OrcaExecutor(ExternalProgramExecutor):
    name = "orca"
    executable = "orca"

    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        keywords = [task.method, task.basis, "TightSCF"]
        use_tddft = any(
            p.lower().startswith("lambda")
            or p.lower().startswith("absorption")
            or "oscillator" in p.lower()
            for p in task.properties
        )
        use_polar = any(p.lower() == "polarizability" for p in task.properties)
        header = "! " + " ".join(keywords)
        if task.dispersion:
            header += f" {task.dispersion}"
        lines = [header]
        if use_tddft:
            lines.extend(["%tddft", "nroots 10", "end"])
        if use_polar:
            lines.extend(["%elprop", "Polar 1", "end"])
        lines.append(f"* xyz {task.charge} {task.multiplicity}")
        if geometry.xyz:
            for row in geometry.xyz.splitlines()[2:]:
                if row.strip():
                    lines.append(row)
        lines.append("*")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        props: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        orbitals = []
        in_orbital_block = False
        tddft_states = []
        polar = None
        for line in text.splitlines():
            stripped = line.strip()
            if "TOTAL SCF ENERGY" in line:
                continue
            if "Total Energy" in line and ":" in line and "Eh" in line:
                tokens = line.replace(":", " ").split()
                for token in tokens:
                    try:
                        metadata["total_energy"] = float(token)
                        break
                    except Exception:
                        continue
            if "Isotropic polarizability" in line:
                try:
                    polar = float(line.split()[-2])
                    props["polarizability"] = polar
                except Exception:
                    pass
            if stripped.startswith("Excited State"):
                # ORCA TDDFT excitation line: Excited State  1:   Singlet-A      3.12 eV 397.5 nm  f=0.1234
                ev_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*eV", stripped)
                nm_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*nm", stripped)
                f_match = re.search(r"f\s*=\s*([0-9]+(?:\.[0-9]+)?)", stripped)
                try:
                    state = {
                        "ev": float(ev_match.group(1)) if ev_match else None,
                        "nm": float(nm_match.group(1)) if nm_match else None,
                        "f": float(f_match.group(1)) if f_match else None,
                    }
                    if state["ev"] is not None or state["nm"] is not None:
                        tddft_states.append(state)
                except Exception:
                    continue
            if stripped.startswith("ORBITAL ENERGIES"):
                in_orbital_block = True
                continue
            if in_orbital_block:
                if not stripped or stripped.startswith("---") or stripped.startswith("NO"):
                    continue
                tokens = stripped.split()
                if len(tokens) >= 4 and tokens[0].isdigit():
                    try:
                        occ = float(tokens[1])
                        energy_ev = float(tokens[3])
                        orbitals.append((occ, energy_ev))
                        continue
                    except Exception:
                        pass
                # end of block when non-parsable line encountered after data started
                if orbitals:
                    in_orbital_block = False
        occupied = [energy for occ, energy in orbitals if occ > 0.0]
        virtual = [energy for occ, energy in orbitals if occ <= 0.0]
        if occupied:
            props["HOMO"] = occupied[-1]
        if virtual:
            props["LUMO"] = virtual[0]
        if "HOMO" in props and "LUMO" in props:
            props["gap"] = props["LUMO"] - props["HOMO"]
        # TDDFT excitations: choose strongest oscillator if available, else first
        if tddft_states:
            with_f = [s for s in tddft_states if s.get("f") is not None]
            best = max(with_f, key=lambda s: s["f"]) if with_f else tddft_states[0]
            if best.get("nm") is not None:
                props["lambda_max_nm"] = float(best["nm"])
            if best.get("f") is not None:
                props["oscillator_strength"] = float(best["f"])
        return props, metadata


class GaussianExecutor(ExternalProgramExecutor):
    name = "gaussian"
    executable = "g16"

    def _write_input(self, path: Path, geometry: GeometryResult, task: QuantumTaskConfig) -> None:
        header = f"%Chk=job.chk\n#P {task.method}/{task.basis} Pop=Full"
        if task.dispersion:
            header += f" EmpiricalDispersion={task.dispersion}"
        lines = [header, "", "Generated by qc.executor", "", f"{task.charge} {task.multiplicity}"]
        if geometry.xyz:
            for row in geometry.xyz.splitlines()[2:]:
                if row.strip():
                    lines.append(row)
        lines.append("")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")

    def _parse_output(self, path: Path, task: QuantumTaskConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        props: Dict[str, float] = {}
        metadata: Dict[str, Any] = {}
        for line in text.splitlines():
            if "SCF Done:" in line:
                try:
                    metadata["total_energy"] = float(line.split()[4])
                except Exception:
                    continue
        return props, metadata


class SemiEmpiricalExecutor(QuantumProgramExecutor):
    """Fallback executor using RDKit-derived heuristics."""

    name = "semi_empirical"

    def is_available(self) -> bool:
        return True

    def run(self, geometry: GeometryResult, task: QuantumTaskConfig) -> ProgramResult:
        from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

        mol = geometry.mol or Chem.MolFromSmiles(geometry.smiles)  # type: ignore[name-defined]
        if mol is None:
            raise ExecutionError("Cannot construct RDKit molecule for surrogate executor.")

        mw = Descriptors.MolWt(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        logp = Crippen.MolLogP(mol)
        frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        num_rot = Descriptors.NumRotatableBonds(mol)

        # crude heuristics for demonstration / fast fallback
        homo = -4.8 - 0.002 * (mw - 300) - 0.18 * aromatic + 0.05 * frac_csp3
        lumo = homo + 2.2 - 0.004 * rings + 0.1 * frac_csp3
        gap = lumo - homo
        ie = -homo
        ea = -lumo
        lam_hole = 0.22 + 0.005 * num_rot + 0.015 * aromatic
        lam_electron = 0.25 + 0.004 * rings + 0.01 * tpsa / 100
        dipole = 1.5 + 0.02 * tpsa
        polar = 3.0 + 0.1 * mw / 100
        packing = max(0.0, 1.0 - (tpsa / 150 + abs(logp) / 7.0))
        stability = 0.5 * ie + 0.2 * polar - 0.1 * lam_hole

        props = {
            "HOMO": homo,
            "LUMO": lumo,
            "gap": gap,
            "IE": ie,
            "EA": ea,
            "lambda_hole": lam_hole,
            "lambda_electron": lam_electron,
            "dipole": dipole,
            "polarizability": polar,
            "packing_score": packing,
            "stability_index": stability,
        }
        filtered = {k: v for k, v in props.items() if k in task.properties or not task.properties}
        metadata = {
            "engine": self.name,
            "heuristic": True,
            "mw": mw,
            "rings": rings,
            "aromatic_rings": aromatic,
            "tpsa": tpsa,
            "logp": logp,
        }
        return ProgramResult(properties=filtered, metadata=metadata)


DEFAULT_EXECUTORS: Mapping[str, QuantumProgramExecutor] = {
    "psi4": Psi4Executor(),
    "gaussian": GaussianExecutor(),
    "orca": OrcaExecutor(),
    "semi_empirical": SemiEmpiricalExecutor(),
}


def resolve_executor(name: str | None) -> QuantumProgramExecutor:
    if not name:
        return DEFAULT_EXECUTORS["semi_empirical"]
    lowered = name.lower()
    if lowered in DEFAULT_EXECUTORS:
        return DEFAULT_EXECUTORS[lowered]
    raise KeyError(f"Unbekannter QC executor '{name}'")
