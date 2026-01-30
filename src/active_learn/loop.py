"""
Active learning loop dirigiert surrogate, generator und DFT interface
"""

from __future__ import annotations

import contextlib
from collections import deque
from pathlib import Path
import sys


PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("project roout nicht auf src/")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json

import numpy as np
import pandas as pd

from src.active_learn.acq import AcquisitionConfig, acquisition_score
from src.active_learn.sched import ActiveLearningScheduler, SchedulerConfig
from src.data.dataset import split_dataframe
from src.data.dft_int import DFTInterface, DFTJobSpec, DFTResult
from src.data.featurization import mol_to_graph
from src.data.featurization_3d import molblock_to_data
from src.models.ensemble import SurrogateEnsemble
from src.models.jtvae_extended import JTVAE, sample_conditional
from src.utils.log import get_logger

try:
    _sa_score = None  # optional SA scorer
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import AllChem, DataStructs, Lipinski, rdMolDescriptors, Crippen
    try:
        from rdkit.Chem import rdFingerprintGenerator
    except Exception:  # auch optional
        rdFingerprintGenerator = None  # type: ignore
    try:
        # optionaler synthetic accessibility scorer (nicht in allen RDKits glaub) 
        from rdkit.Chem import rdMolDescriptors as _rdm
        from rdkit.Chem import Descriptors as _desc
        import sascorer as _sascorer  # type: ignore

        def _sa_score(mol: "Chem.Mol") -> float:
            return float(_sascorer.calculateScore(mol))

    except Exception:  # optional wie davor
        _sa_score = None  # type: ignore
    RDKit_AVAILABLE = True
except Exception:  # wieder optional, werden safe nie benutzt
    RDKit_AVAILABLE = False

logger = get_logger(__name__)

PROPERTY_DEFAULT_ALIASES = {
    "HOMO": "HOMO_eV",
    "LUMO": "LUMO_eV",
    "gap": "gap_eV",
    "IE": "IE_eV",
    "EA": "EA_eV",
}



@dataclass
class LoopConfig:
    batch_size: int = 8
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    target_columns: Sequence[str] = ("HOMO", "LUMO")
    maximise: Sequence[bool] = (False, True)
    generator_samples: int = 32
    generator_attempts: int = 5  # batches to sample wenn pool keine candidates hat
    results_dir: Path = Path("experiments")
    assemble: Dict[str, object] = field(default_factory=dict)
    diversity_threshold: float = 0.85
    diversity_metric: str = "tanimoto"
    generator_refresh: Dict[str, object] = field(default_factory=dict)
    property_aliases: Dict[str, str] = field(default_factory=dict)
    max_pool_eval: Optional[int] = None  # cap number von pool candidates evaluated pro iteration
    max_generated_heavy_atoms: Optional[int] = None  # skipt generated SMILES mit zu vielen schweren atome
    max_generated_smiles_len: Optional[int] = None  # skipt generated SMILES mit zu langer Länge
    generated_smiles_len_factor: Optional[float] = 1.5  # fallback length cap als factor von median length wenn max nicht gesetted
    require_conjugation: bool = True  # macht basic OSC conjugation filter auf die generierten SMILES
    min_conjugated_bonds: int = 2  # selbserklärend
    min_alternating_conjugated_bonds: int = 3  # macht das alternating single/double conjugated path von dieser länge sind
    min_pi_conjugated_fraction: Optional[float] = None  # selbsterklärend
    min_aromatic_rings: int = 1  # selbsterklärend
    max_rotatable_bonds: Optional[int] = None  # cap flexibility
    max_rotatable_bonds_conjugated: Optional[int] = None  # cap rotatable bonds mit einem conjugated subgraph
    max_branch_points: Optional[int] = None  # cap anzahl von heavy atoms mit degree >= 3
    max_branch_degree: Optional[int] = None  # cap max heavy-atom degree
    max_charged_atoms: Optional[int] = None  # cap anzahl von geladenen atomen
    property_filters: Dict[str, Sequence[float]] = field(default_factory=dict)  # min/max per property
    require_neutral: bool = True  #selbserklärend
    sa_score_max: Optional[float] = None  # optional siehe oben
    physchem_filters: Dict[str, Sequence[float]] = field(default_factory=dict)  # also clogp, tpsa, frac_csp3
    scaffold_unique: bool = False  # macht das man unique Murcko scaffolds hat
    exclude_smiles_paths: Sequence[str] = tuple()  # optional CSV/TXT files with SMILES to exclude
    auto_relax_filters: bool = True  #falls nichts generated wird, werden alle filter relaxed
    dft_job_defaults: Dict[str, object] = field(default_factory=dict)  # also charge, multiplicity, metadata etc.


@contextlib.contextmanager
def _suppress_rdkit_errors():
    """RDKit Fehler spam unterdrücken."""

    if not RDKit_AVAILABLE:
        yield
        return
    try:
        RDLogger.DisableLog("rdApp.error")
        yield
    finally:
        try:
            RDLogger.EnableLog("rdApp.error")
        except Exception:
            pass


class ActiveLearningLoop:
    def __init__(
        self,
        surrogate: SurrogateEnsemble,
        labelled: pd.DataFrame,
        pool: pd.DataFrame,
        config: LoopConfig,
        *,
        generator: Optional[JTVAE] = None,
        generator3d: Optional[object] = None,
        generator3d_template: Optional[Dict[str, object]] = None,
        fragment_vocab: Optional[Dict[str, int]] = None,
        dft: Optional[DFTInterface] = None,
        generator_device: Optional[str] = None,
        dft_job_defaults: Optional[Dict[str, object]] = None,
    ) -> None:
        self.surrogate = surrogate
        self.config = config
        self.labelled = labelled.reset_index(drop=True)
        self.pool = pool.reset_index(drop=True)
        self.generator = generator
        self.generator3d = generator3d
        self.generator3d_template = generator3d_template or {}
        self.generator_device = generator_device
        self.fragment_vocab = fragment_vocab or {}
        self.dft = dft
        self.scheduler = ActiveLearningScheduler(config.scheduler)
        self.history: List[pd.DataFrame] = []
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.assemble_kwargs = dict(config.assemble)
        self.diversity_threshold = float(getattr(config, "diversity_threshold", 0.0))
        self.diversity_metric = getattr(config, "diversity_metric", "tanimoto").lower()
        self.generator_refresh_kwargs = dict(getattr(config, "generator_refresh", {}))
        self.property_aliases = dict(PROPERTY_DEFAULT_ALIASES)
        self.property_aliases.update(getattr(config, "property_aliases", {}))
        self._rng = np.random.default_rng()
        self._smiles_cache: Dict[str, Optional[str]] = {}
        self._fingerprint_cache: Dict[str, Optional[object]] = {}
        self._fingerprints: List[object] = []
        self._morgan_generator = None
        self._target_indices: List[int] = []
        self._refresh_target_indices()
        self._filter_indices: Dict[str, int] = {}
        self._init_property_filters()
        self._is_schnet = bool(getattr(self.surrogate, "is_schnet", False) or getattr(self.surrogate, "schnet_like", False))
        self._scaffolds_seen: set[str] = set()
        self._excluded_smiles: set[str] = self._load_excluded_smiles(getattr(config, "exclude_smiles_paths", ()))
        self._excluded_smiles = self._canonicalize_smiles_set(self._excluded_smiles, "exclude list")
        self.labelled = self._canonicalize_dataframe(self.labelled, "labelled")
        self.pool = self._canonicalize_dataframe(self.pool, "pool")
        self._filter_pool_overlaps()
        self._median_smiles_len = self._compute_median_smiles_len()
        self._dft_job_defaults: Dict[str, object] = dict(dft_job_defaults or {})
        if self.generator is not None:
            self.generator.eval()
            if self.generator_device is None:
                try:
                    self.generator_device = str(next(self.generator.parameters()).device)
                except StopIteration:
                    self.generator_device = None
        if RDKit_AVAILABLE and self.diversity_threshold > 0:
            logger.info(
                "Precomputet diversity fingerprints (threshold=%.2f) fuer %d molecules...",
                self.diversity_threshold,
                len(pd.concat(
                    [self.labelled.get("smiles", pd.Series(dtype=str)), self.pool.get("smiles", pd.Series(dtype=str))],
                    axis=0,
                ).dropna().unique()),
            )
            initial_smiles = pd.concat(
                [self.labelled.get("smiles", pd.Series(dtype=str)), self.pool.get("smiles", pd.Series(dtype=str))],
                axis=0,
            ).dropna().unique()
            for idx, smi in enumerate(initial_smiles, 1):
                fp = self._fingerprint(smi)
                if fp is not None:
                    self._fingerprint_cache[smi] = fp
                    self._fingerprints.append(fp)
                if idx % 5000 == 0:
                    logger.debug("Processed %d/%d fingerprints...", idx, len(initial_smiles))
            logger.info("beendet fingerprint precompute (%d cached).", len(self._fingerprints))

        if len(self.config.target_columns) != len(self.config.maximise):
            raise ValueError("target_columns und maximise length mismatch.")

    def _current_best(self) -> Optional[np.ndarray]:
        if self.labelled.empty:
            return None
        target_cols = list(self.config.target_columns)
        arr = self.labelled[target_cols].to_numpy(dtype=float)
        best = []
        for dim, maximise in enumerate(self.config.maximise):
            column = arr[:, dim]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                best.append(0.0)
            else:
                best.append(finite.max() if maximise else finite.min())
        return np.array(best)

    def _fingerprint(self, smiles: str):
        if not RDKit_AVAILABLE:
            return None
        if smiles in self._fingerprint_cache:
            return self._fingerprint_cache[smiles]
        with _suppress_rdkit_errors():
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = None
        if rdFingerprintGenerator is not None:
            if self._morgan_generator is None:
                self._morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            try:
                fp = self._morgan_generator.GetFingerprint(mol)
            except Exception:
                fp = None
        if fp is None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        self._fingerprint_cache[smiles] = fp
        return fp

    def _refresh_target_indices(self) -> None:
        """Aligned desired target columns mit surrogate outputs (handles retrains/ordering)"""
        surrogate_targets = list(getattr(self.surrogate, "target_columns", ()))
        if not surrogate_targets:
            raise ValueError("Surrogate keine target_columns defined.")
        self._target_indices = []
        for t in self.config.target_columns:
            if t not in surrogate_targets:
                raise ValueError(f"Target column '{t}' nicht gefudnen in surrogate outputs: {surrogate_targets}")
            self._target_indices.append(surrogate_targets.index(t))

    def _parse_smiles(self, smiles: str):
        """Parset SMILES waehrend suppressed RDKit stderr spam auf invalid inputs"""
        if not RDKit_AVAILABLE or not smiles:
            return None
        with _suppress_rdkit_errors():
            return Chem.MolFromSmiles(smiles)

    def _init_property_filters(self) -> None:
        """Map property_filters keys zum surrogate output indices"""
        self._filter_indices.clear()
        if not self.config.property_filters:
            return
        surrogate_targets = list(getattr(self.surrogate, "target_columns", ()))
        missing = []
        for prop in self.config.property_filters.keys():
            if prop in surrogate_targets:
                self._filter_indices[prop] = surrogate_targets.index(prop)
            else:
                missing.append(prop)
        if missing:
            logger.warning("property_filters entries nicht da in surrogate outputs und wird ignoriert: %s", missing)

    def _load_excluded_smiles(self, paths: Sequence[str]) -> set[str]:
        excluded: set[str] = set()
        for p in paths or ():
            try:
                path = Path(p)
                if not path.exists():
                    logger.warning("Exclude SMILES path not found: %s", path)
                    continue
                if path.suffix.lower() in {".csv", ".tsv"}:
                    import pandas as pd  # lazy import

                    df = pd.read_csv(path)
                    if "smiles" in df.columns:
                        excluded.update(df["smiles"].dropna().astype(str).tolist())
                    else:
                        logger.warning("Exclude SMILES CSV missing 'smiles' column: %s", path)
                else:
                    # als plain text (1 SMILES pro linie)
                    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line = line.strip()
                        if line:
                            excluded.add(line)
                if excluded:
                    logger.info("Loaded %d SMILES auszuschließen von %s", len(excluded), path)
            except Exception as exc:
                logger.warning("fehler beim laden exclude SMILES von %s: %s", p, exc)
        return excluded

    def _sample_generator3d_templates(self, n_samples: int):
        if not self.generator3d_template:
            return None
        try:
            import torch
        except Exception:
            return None
        template = self.generator3d_template
        count = max(1, int(n_samples))
        if isinstance(template, dict):
            if "z_pool" in template and "mask_pool" in template:
                z_pool = template.get("z_pool")
                mask_pool = template.get("mask_pool")
                if z_pool is None or mask_pool is None:
                    return None
                pool_size = int(z_pool.size(0))
                if pool_size == 0:
                    return None
                idxs = self._rng.integers(0, pool_size, size=count)
                return z_pool[idxs], mask_pool[idxs]
            if "templates" in template:
                templates = template.get("templates") or []
                if not templates:
                    return None
                idxs = self._rng.integers(0, len(templates), size=count)
                z = [templates[i]["z"] for i in idxs]
                mask = [templates[i]["mask"] for i in idxs]
                return torch.stack(z, dim=0), torch.stack(mask, dim=0)
            if "z" in template and "mask" in template:
                z = template.get("z")
                mask = template.get("mask")
                if z is None or mask is None:
                    return None
                if count == 1:
                    return z.unsqueeze(0), mask.unsqueeze(0)
                return z.unsqueeze(0).repeat(count, 1), mask.unsqueeze(0).repeat(count, 1)
        if isinstance(template, (list, tuple)):
            if not template:
                return None
            idxs = self._rng.integers(0, len(template), size=count)
            z = [template[i]["z"] for i in idxs]
            mask = [template[i]["mask"] for i in idxs]
            return torch.stack(z, dim=0), torch.stack(mask, dim=0)
        return None

    def _canonical_smiles(self, smiles: str) -> Optional[str]:
        if smiles is None:
            return None
        text = str(smiles).strip()
        if not text:
            return None
        if text in self._smiles_cache:
            return self._smiles_cache[text]
        if not RDKit_AVAILABLE:
            self._smiles_cache[text] = text
            return text
        mol = self._parse_smiles(text)
        if mol is None:
            self._smiles_cache[text] = None
            return None
        try:
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            canonical = None
        self._smiles_cache[text] = canonical
        if canonical is not None:
            self._smiles_cache.setdefault(canonical, canonical)
        return canonical

    def _canonicalize_smiles_set(self, smiles: Sequence[str], name: str) -> set[str]:
        canonical: set[str] = set()
        invalid = 0
        for smi in smiles:
            canon = self._canonical_smiles(smi)
            if canon:
                canonical.add(canon)
            else:
                invalid += 1
        if invalid:
            logger.warning("Dropped %d invalid SMILES von %s", invalid, name)
        return canonical

    def _canonicalize_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "smiles" not in df.columns:
            return df
        canonical = []
        invalid = 0
        for smi in df["smiles"].tolist():
            canon = self._canonical_smiles(smi)
            if canon is None:
                invalid += 1
            canonical.append(canon)
        if invalid:
            logger.warning("Dropping %d rows mit invaliden SMILES von %s dataset", invalid, name)
        cleaned = df.copy()
        cleaned["smiles"] = canonical
        cleaned = cleaned.dropna(subset=["smiles"])
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=["smiles"]).reset_index(drop=True)
        dropped = before - len(cleaned)
        if dropped:
            logger.info("entfernt %d duplicate SMILES von %s dataset", dropped, name)
        return cleaned

    def _filter_pool_overlaps(self) -> None:
        if "smiles" not in self.pool.columns or self.pool.empty:
            return
        if "smiles" not in self.labelled.columns:
            return
        known = set(self.labelled["smiles"]).union(self._excluded_smiles)
        if not known:
            return
        before = len(self.pool)
        self.pool = self.pool[~self.pool["smiles"].isin(known)].reset_index(drop=True)
        removed = before - len(self.pool)
        if removed:
            logger.info(
                "entfernt %d pool entries schon enthalten in labelled/excluded datasets",
                removed,
            )

    def _passes_diversity(self, smiles: str) -> bool:
        if self.diversity_threshold <= 0 or not RDKit_AVAILABLE:
            return True
        fp = self._fingerprint(smiles)
        if fp is None:
            return False
        if not self._fingerprints:
            self._fingerprints.append(fp)
            return True
        sims = [DataStructs.TanimotoSimilarity(fp, existing) for existing in self._fingerprints]
        if sims and max(sims) >= self.diversity_threshold:
            return False
        self._fingerprints.append(fp)
        return True

    def _has_conjugation(self, mol: "Chem.Mol") -> bool:
        """Basic OSC filter, checkt fuer conjugated path length"""
        if mol is None:
            return False
        min_len = int(getattr(self.config, "min_conjugated_bonds", 0) or 0)
        if min_len <= 0:
            return True
        return self._longest_conjugated_path(mol) >= min_len

    def _count_aromatic_rings(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        rings = Chem.GetSymmSSSR(mol)
        aromatic = 0
        for ring in rings:
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                aromatic += 1
        return aromatic

    def _pi_conjugated_fraction(self, mol: "Chem.Mol") -> float:
        if mol is None:
            return 0.0
        atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        if not atoms:
            return 0.0
        conj_atoms = 0
        for atom in atoms:
            if atom.GetIsAromatic():
                conj_atoms += 1
                continue
            for bond in atom.GetBonds():
                if bond.GetIsConjugated() or bond.GetIsAromatic():
                    conj_atoms += 1
                    break
        return conj_atoms / len(atoms)

    def _conjugated_atom_indices(self, mol: "Chem.Mol") -> set[int]:
        if mol is None:
            return set()
        conj_atoms: set[int] = set()
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() > 1 and atom.GetIsAromatic():
                conj_atoms.add(atom.GetIdx())
        for bond in mol.GetBonds():
            if bond.GetIsConjugated() or bond.GetIsAromatic():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                if a1.GetAtomicNum() > 1:
                    conj_atoms.add(a1.GetIdx())
                if a2.GetAtomicNum() > 1:
                    conj_atoms.add(a2.GetIdx())
        return conj_atoms

    def _is_rotatable_bond(self, bond: "Chem.Bond") -> bool:
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
        if bond.IsInRing() or bond.GetIsAromatic():
            return False
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetAtomicNum() <= 1 or a2.GetAtomicNum() <= 1:
            return False
        # exclude terminal heavy-atom bonds
        if sum(1 for n in a1.GetNeighbors() if n.GetAtomicNum() > 1) <= 1:
            return False
        if sum(1 for n in a2.GetNeighbors() if n.GetAtomicNum() > 1) <= 1:
            return False
        return True

    def _rotatable_bonds_in_conjugated(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        conj_atoms = self._conjugated_atom_indices(mol)
        if not conj_atoms:
            return 0
        count = 0
        for bond in mol.GetBonds():
            if not (bond.GetIsConjugated() or bond.GetIsAromatic()):
                continue
            if not self._is_rotatable_bond(bond):
                continue
            if bond.GetBeginAtomIdx() in conj_atoms and bond.GetEndAtomIdx() in conj_atoms:
                count += 1
        return count

    def _charged_atom_count(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        return sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() != 0)

    def _branch_points(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
            if heavy_neighbors >= 3:
                count += 1
        return count

    def _max_heavy_degree(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        max_degree = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() <= 1:
                continue
            heavy_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() > 1)
            if heavy_neighbors > max_degree:
                max_degree = heavy_neighbors
        return max_degree

    def _longest_conjugated_path(self, mol: "Chem.Mol") -> int:
        if mol is None:
            return 0
        heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
        if not heavy:
            return 0
        adjacency: Dict[int, List[int]] = {idx: [] for idx in heavy}
        for bond in mol.GetBonds():
            if not (bond.GetIsConjugated() or bond.GetIsAromatic()):
                continue
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if a1 not in heavy or a2 not in heavy:
                continue
            adjacency[a1].append(a2)
            adjacency[a2].append(a1)
        if not any(adjacency.values()):
            return 0
        max_len = 0
        for start in adjacency:
            if not adjacency[start]:
                continue
            distances = {start: 0}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adjacency.get(node, []):
                    if nb in distances:
                        continue
                    distances[nb] = distances[node] + 1
                    queue.append(nb)
            if distances:
                max_len = max(max_len, max(distances.values()))
        return max_len

    def _smiles_from_atoms_positions(self, atom_z: List[int], pos: List[List[float]], mask: Optional[List[float]] = None) -> Optional[str]:
        """Buildet SMILES von atomic numbers und 3D positions mit RDKit bond perception"""
        if not RDKit_AVAILABLE:
            return None
        try:
            mol = Chem.RWMol()
            positions = []
            for idx, z in enumerate(atom_z):
                if mask is not None and mask[idx] <= 0:
                    continue
                if idx >= len(pos):
                    break
                a = Chem.Atom(int(z))
                mol_idx = mol.AddAtom(a)
                positions.append((mol_idx, pos[idx]))
            if not positions:
                return None
            base = mol.GetMol()
            conf = Chem.Conformer(base.GetNumAtoms())
            for mol_idx, p in positions:
                conf.SetAtomPosition(mol_idx, p)
            base.AddConformer(conf, assignId=True)
            Chem.SanitizeMol(base, catchErrors=True)
            from rdkit.Chem import rdDetermineBonds
            rdDetermineBonds.DetermineBonds(base, charge=0)
            Chem.SanitizeMol(base)
            return Chem.MolToSmiles(base)
        except Exception:
            return None

    def _murcko_scaffold(self, mol: "Chem.Mol") -> Optional[str]:
        if mol is None:
            return None
        try:
            return rdMolDescriptors.CalcMurckoScaffoldSmiles(mol)
        except Exception:
            return None

    def _physchem_ok(self, mol: "Chem.Mol") -> bool:
        """Checkt lightweight physchem/processability windows"""

        if mol is None or not RDKit_AVAILABLE:
            return False
        cfg = getattr(self.config, "physchem_filters", {}) or {}

        def _in_range(val: float, bounds: Sequence[float]) -> bool:
            if not bounds or len(bounds) != 2:
                return True
            lo, hi = bounds
            if lo is not None and val < lo:
                return False
            if hi is not None and val > hi:
                return False
            return True

        # TPSA, logP, HBA/HBD, fractionCSP3
        try:
            tpsa = rdMolDescriptors.CalcTPSA(mol)
            clogp = Crippen.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            frac_csp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        except Exception as exc:
            logger.debug("Physchem calc failed: %s", exc)
            return False

        if not _in_range(tpsa, cfg.get("tpsa", ())):
            return False
        if not _in_range(clogp, cfg.get("clogp", ())):
            return False
        if not _in_range(hbd, cfg.get("hbd", ())):
            return False
        if not _in_range(hba, cfg.get("hba", ())):
            return False
        if not _in_range(frac_csp3, cfg.get("frac_csp3", ())):
            return False
        return True

    def _has_alternating_conjugation(self, mol: "Chem.Mol", min_bonds: int) -> bool:
        """Checkt für einen conjugated path mit alternating single/double bonds"""
        if mol is None or min_bonds <= 0:
            return False

        aromatic_bonds = {b.GetIdx() for b in mol.GetBonds() if b.GetIsAromatic()}
        # Kekulize a copy so aromatic systems become an explicit single/double pattern.
        try:
            work = Chem.Mol(mol)
            Chem.Kekulize(work, clearAromaticFlags=True)
        except Exception:
            work = mol

        conj_bonds = []
        for bond in work.GetBonds():
            if bond.GetIdx() in aromatic_bonds:
                continue
            if not bond.GetIsConjugated():
                continue
            btype = bond.GetBondType()
            if btype not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE):
                continue
            conj_bonds.append((bond.GetIdx(), btype, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if not conj_bonds:
            return False

        bond_adj: Dict[int, List[int]] = {idx: [] for idx, *_ in conj_bonds}
        for idx, _, a1, a2 in conj_bonds:
            for jdx, _, b1, b2 in conj_bonds:
                if idx == jdx:
                    continue
                if a1 in (b1, b2) or a2 in (b1, b2):
                    bond_adj[idx].append(jdx)

        classes = {idx: ("S" if btype == Chem.BondType.SINGLE else "D") for idx, btype, _, _ in conj_bonds}

        for start_idx, start_class in classes.items():
            stack = [(start_idx, start_class, 1)]
            visited = set()
            while stack:
                bond_idx, cls, length = stack.pop()
                if length >= min_bonds:
                    return True
                visited.add(bond_idx)
                for nb in bond_adj.get(bond_idx, []):
                    ncls = classes.get(nb)
                    if ncls is None or ncls == cls or nb in visited:
                        continue
                    stack.append((nb, ncls, length + 1))
        return False

    def _compute_median_smiles_len(self) -> Optional[float]:
        lengths = []
        if "smiles" in self.labelled:
            lengths.extend(self.labelled["smiles"].dropna().astype(str).str.len().tolist())
        if "smiles" in self.pool:
            lengths.extend(self.pool["smiles"].dropna().astype(str).str.len().tolist())
        if not lengths:
            return None
        lengths.sort()
        mid = len(lengths) // 2
        if len(lengths) % 2 == 0:
            return (lengths[mid - 1] + lengths[mid]) / 2.0
        return float(lengths[mid])

    def _build_graph(self, smiles: str, row: Optional[pd.Series] = None):
        """Featurize in den graph/point-cloud object rein based auf surrogate typ"""
        if self._is_schnet:
            mol_block = None
            if row is not None:
                try:
                    mol_block = row.get("mol", None)
                except Exception:
                    mol_block = getattr(row, "mol", None)
            try:
                return molblock_to_data(mol_block or "", smiles=smiles)
            except Exception as exc:
                logger.debug("SchNet featurization failed fuer %s: %s", smiles, exc)
                return None
        try:
            return mol_to_graph(smiles, y=None)
        except Exception as exc:
            logger.debug("Featurization failed fuer %s: %s", smiles, exc)
            return None

    def _passes_property_filters(self, smiles: str) -> bool:
        """benutzt surrogate predictions um property ranges auf generated SMILES zu enforcen"""
        if not self._filter_indices:
            return True
        graph = self._build_graph(smiles)
        if graph is None:
            logger.debug("Property filter: failed to featurize %s", smiles)
            return False
        mean, std, _ = self.surrogate.predict([graph], batch_size=1)
        for prop, idx in self._filter_indices.items():
            vmin, vmax = self.config.property_filters.get(prop, (None, None))
            if vmin is None or vmax is None:
                continue
            val = float(mean[0, idx])
            if val < vmin or val > vmax:
                return False
        return True

    def _normalise_predictions(
        self, mean: np.ndarray, std: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_targets = mean.shape[1]
        directions = np.array([1.0 if maximise else -1.0 for maximise in self.config.maximise])
        mus = np.zeros(n_targets)
        sigmas = np.ones(n_targets)
        for i, target in enumerate(self.config.target_columns):
            values = self.labelled[target].dropna().to_numpy(dtype=float)
            if values.size >= 2:
                oriented = values * directions[i]
                mus[i] = oriented.mean()
                sigma = oriented.std()
                sigmas[i] = sigma if sigma > 1e-6 else 1.0
            elif values.size == 1:
                mus[i] = values[0] * directions[i]
                sigmas[i] = 1.0
            else:
                mus[i] = 0.0
                sigmas[i] = 1.0
        mean_norm = ((mean * directions) - mus) / sigmas
        std_norm = std / sigmas
        return mean_norm, std_norm, mus, sigmas, directions

    def _save_diagnostics(self, pool_slice: pd.DataFrame, iteration: int) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            logger.debug("Matplotlib nicht available; kein diagnostics plot fuer iteration %d.", iteration)
            return
        diag_dir = self.results_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        n_targets = len(self.config.target_columns)
        fig, axes = plt.subplots(n_targets, 1, figsize=(6, 3 * n_targets), squeeze=False)
        for idx, target in enumerate(self.config.target_columns):
            ax = axes[idx, 0]
            ax.scatter(
                pool_slice[f"pred_{target}"],
                pool_slice["acquisition_score"],
                alpha=0.6,
                edgecolors="none",
            )
            ax.set_xlabel(f"Predicted {target}")
            ax.set_ylabel("Acquisition")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Acquisition diagnostics (iteration {iteration})")
        fig.tight_layout()
        fig.savefig(diag_dir / f"diag_iter_{iteration:03d}.png", dpi=150)
        plt.close(fig)

    def _refresh_generator(self) -> None:
        if self.generator is None or not self.fragment_vocab:
            return
        if len(self.labelled) < 5:
            return
        try:
            from src.data.jt_preprocess import JTPreprocessConfig, prepare_jtvae_examples
            from src.models.jtvae_extended import JTVDataset, train_jtvae
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Skippen von generator refresh: preprocessing utilities nicht da (%s).", exc)
            return
        df = self.labelled.dropna(subset=["smiles", *self.config.target_columns])
        if df.empty:
            return
        df = df[["smiles", *self.config.target_columns]].drop_duplicates(subset="smiles")
        config = JTPreprocessConfig(
            max_fragments=getattr(self.generator, "max_tree_nodes", 12),
            condition_columns=self.config.target_columns,
        )
        max_heavy_atoms = self.generator_refresh_kwargs.get("max_heavy_atoms")
        if max_heavy_atoms is None:
            max_heavy_atoms = getattr(self.config, "max_generated_heavy_atoms", None)
        try:
            examples = prepare_jtvae_examples(
                df,
                self.fragment_vocab,
                config=config,
                max_heavy_atoms=max_heavy_atoms,
            )
        except Exception as exc:
            logger.warning("Failed prepare JT-VAE examples für refresh: %s", exc)
            return
        dataset = JTVDataset(examples)
        if len(dataset) == 0:
            logger.debug("Generator refresh skipped: no valid examples.")
            return
        refresh_cfg = {
            "epochs": 1,
            "batch_size": 16,
            "lr": 1e-4,
            "kl_weight": 0.5,
            "property_weight": 0.0,
            "adj_weight": 1.0,
            "save_dir": self.results_dir / "generator_refresh",
        }
        refresh_cfg.update(self.generator_refresh_kwargs)
        refresh_cfg.pop("max_heavy_atoms", None)
        refresh_cfg["epochs"] = int(refresh_cfg.get("epochs", 1))
        refresh_cfg["batch_size"] = int(refresh_cfg.get("batch_size", 16))
        refresh_cfg["lr"] = float(refresh_cfg.get("lr", 1e-4))
        refresh_cfg["kl_weight"] = float(refresh_cfg.get("kl_weight", 0.5))
        refresh_cfg["property_weight"] = float(refresh_cfg.get("property_weight", 0.0))
        refresh_cfg["adj_weight"] = float(refresh_cfg.get("adj_weight", 1.0))
        device = next(self.generator.parameters()).device
        save_dir = Path(refresh_cfg.pop("save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Refreshing generator on %d molecules fuer %d epochs (lr=%s)",
            len(dataset),
            refresh_cfg.get("epochs", 1),
            refresh_cfg.get("lr", 1e-4),
        )
        train_jtvae(
            self.generator,
            dataset,
            self.fragment_vocab,
            device=str(device),
            save_dir=str(save_dir),
            **refresh_cfg,
        )

    def _ensure_pool(self, min_size: int, cond: Optional[np.ndarray], assemble_kwargs: Optional[Dict]) -> int:
        if self.generator is None and self.generator3d is None:
            return 0
        generated = 0
        rejected = {"invalid": 0, "duplicate": 0, "filtered": 0}
        max_attempts = max(1, int(getattr(self.config, "generator_attempts", 5)))
        existing = set(self.pool["smiles"]).union(set(self.labelled["smiles"])).union(self._excluded_smiles)
        # dynamic length cap based on dataset median if explicit cap not set
        effective_len_cap = self.config.max_generated_smiles_len
        if self._median_smiles_len is not None:
            factor = getattr(self.config, "generated_smiles_len_factor", None)
            if factor is not None and factor > 0:
                dynamic_cap = int(self._median_smiles_len * factor)
                if effective_len_cap is None:
                    effective_len_cap = dynamic_cap
                    logger.info(
                        "Using dynamic SMILES length cap: median %.1f * %.2f -> %d",
                        self._median_smiles_len,
                        factor,
                        effective_len_cap,
                    )
                else:
                    capped = min(effective_len_cap, dynamic_cap)
                    if capped != effective_len_cap:
                        logger.info(
                            "Tightening SMILES length cap: min(config=%d, dynamic=%d) -> %d",
                            effective_len_cap,
                            dynamic_cap,
                            capped,
                        )
                        effective_len_cap = capped
        attempts = 0
        relaxed_used = False
        use_relaxed = False
        skip_property_filters = False
        skip_diversity = False
        relax_structural = False
        while len(self.pool) < min_size and attempts < max_attempts:
            attempts += 1
            assemble_kwargs_current = assemble_kwargs or self.assemble_kwargs
            if use_relaxed and self.generator is not None:
                assemble_kwargs_current = dict(assemble_kwargs_current)
                assemble_kwargs_current["adjacency_threshold"] = min(
                    assemble_kwargs_current.get("adjacency_threshold", 0.7), 0.4
                )
                assemble_kwargs_current["max_tree_nodes"] = min(
                    assemble_kwargs_current.get("max_tree_nodes", 8) or 8, 6
                )
                assemble_kwargs_current["beam_width"] = max(assemble_kwargs_current.get("beam_width", 3), 5)
            samples = []
            if self.generator is not None and self.fragment_vocab:
                samples = sample_conditional(
                    self.generator,
                    self.fragment_vocab,
                    cond=cond,
                    n_samples=self.config.generator_samples,
                    assembler="beam",
                    assemble_kwargs=assemble_kwargs_current,
                    device=self.generator_device,
                )
            elif self.generator3d is not None and self.generator3d_template:
                try:
                    from src.models.vae3d import sample_vae3d
                    template_batch = self._sample_generator3d_templates(
                        int(getattr(self.config, "generator_samples", 1))
                    )
                    if template_batch is None:
                        logger.warning("3D generator templates nicht da; skipping generation")
                        samples = []
                    else:
                        atom_z, mask = template_batch
                        coords = sample_vae3d(
                            self.generator3d,
                            atom_z,
                            mask,
                            device=self.generator_device or "cpu",
                        )
                        coords = coords.cpu().numpy()
                        z_list = atom_z.cpu().numpy()
                        mask_list = mask.cpu().numpy()
                        samples = []
                        for idx in range(coords.shape[0]):
                            smi = self._smiles_from_atoms_positions(
                                z_list[idx].tolist(),
                                coords[idx].tolist(),
                                mask_list[idx].tolist(),
                            )
                            if smi:
                                samples.append({"smiles": smi, "status": "generated3d"})
                except Exception as exc:
                    logger.warning("3D generator sampling failed: %s", exc)
                    samples = []
            new_rows = []
            for sample in samples:
                smiles = sample.get("smiles")
                status = sample.get("status")
                if not smiles:
                    rejected["invalid"] += 1
                    continue
                smiles = self._canonical_smiles(smiles)
                if not smiles:
                    rejected["invalid"] += 1
                    continue
                if smiles in existing:
                    rejected["duplicate"] += 1
                    continue
                if effective_len_cap and len(smiles) > effective_len_cap:
                    logger.debug(
                        "Skipping generated SMILES (len %d > cap %d): %s",
                        len(smiles),
                        effective_len_cap,
                        smiles,
                    )
                    rejected["filtered"] += 1
                    continue
                mol = self._parse_smiles(smiles) if RDKit_AVAILABLE else None
                if mol is not None and RDKit_AVAILABLE:
                    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                    if len(frags) > 1:
                        # keept den largest connected component um disconnected assemblies zu vermeiden
                        frags_sorted = sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)
                        mol = frags_sorted[0]
                        smiles_new = Chem.MolToSmiles(mol, isomericSmiles=True)
                        if smiles_new != smiles:
                            smiles = smiles_new
                            if smiles in existing:
                                rejected["duplicate"] += 1
                                continue
                            if effective_len_cap and len(smiles) > effective_len_cap:
                                rejected["filtered"] += 1
                                continue
                heavy_atoms = None
                if RDKit_AVAILABLE:
                    if mol is None:
                        logger.debug("Skipping invalid generated SMILES: %s", smiles)
                        rejected["invalid"] += 1
                        continue
                    heavy_atoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
                    if self.config.require_neutral and Chem.GetFormalCharge(mol) != 0:
                        logger.debug(
                            "Skipping charged generated SMILES (charge %d): %s",
                            Chem.GetFormalCharge(mol),
                            smiles,
                        )
                        rejected["filtered"] += 1
                        continue
                    if getattr(self.config, "max_charged_atoms", None) is not None:
                        charged_atoms = self._charged_atom_count(mol)
                        if charged_atoms > int(self.config.max_charged_atoms):
                            logger.debug(
                                "Skipping generated SMILES (charged atoms %d > %d): %s",
                                charged_atoms,
                                self.config.max_charged_atoms,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if (
                        self.config.max_generated_heavy_atoms is not None
                        and heavy_atoms is not None
                        and heavy_atoms > self.config.max_generated_heavy_atoms
                    ):
                        logger.debug(
                            "Skipping generated SMILES (heavy atoms %d > %d): %s",
                            heavy_atoms,
                            self.config.max_generated_heavy_atoms,
                            smiles,
                        )
                        rejected["filtered"] += 1
                        continue
                    if self.config.require_conjugation and not self._has_conjugation(mol):
                        logger.debug("Skipping generated SMILES (failt conjugation filter): %s", smiles)
                        rejected["filtered"] += 1
                        continue
                    if (
                        not relax_structural
                        and getattr(self.config, "min_pi_conjugated_fraction", None) is not None
                        and self._pi_conjugated_fraction(mol) < float(self.config.min_pi_conjugated_fraction)
                    ):
                        logger.debug("Skipping generated SMILES (pi-conjugated fraction unter threshold): %s", smiles)
                        rejected["filtered"] += 1
                        continue
                    if (
                        (self.config.min_alternating_conjugated_bonds and not relax_structural)
                        and not self._has_alternating_conjugation(
                            mol, min_bonds=self.config.min_alternating_conjugated_bonds
                        )
                    ):
                        logger.debug(
                            "Skipping generated SMILES (fails alternating conjugation filter): %s", smiles
                        )
                        rejected["filtered"] += 1
                        continue
                    if self.config.min_aromatic_rings:
                        aromatic_rings = self._count_aromatic_rings(mol)
                        if aromatic_rings < self.config.min_aromatic_rings:
                            logger.debug(
                                "Skipping generated SMILES (aromatic rings %d < %d): %s",
                                aromatic_rings,
                                self.config.min_aromatic_rings,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "max_branch_points", None) is not None and not relax_structural:
                        branches = self._branch_points(mol)
                        if branches > int(self.config.max_branch_points):
                            logger.debug(
                                "Skipping generated SMILES (branch points %d > %d): %s",
                                branches,
                                self.config.max_branch_points,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "max_branch_degree", None) is not None and not relax_structural:
                        max_deg = self._max_heavy_degree(mol)
                        if max_deg > int(self.config.max_branch_degree):
                            logger.debug(
                                "Skipping generated SMILES (max heavy degree %d > %d): %s",
                                max_deg,
                                self.config.max_branch_degree,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if self.config.max_rotatable_bonds is not None:
                        rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
                        if rotb > self.config.max_rotatable_bonds:
                            logger.debug(
                                "Skipping generated SMILES (rotatable bonds %d > %d): %s",
                                rotb,
                                self.config.max_rotatable_bonds,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "max_rotatable_bonds_conjugated", None) is not None:
                        rotb_conj = self._rotatable_bonds_in_conjugated(mol)
                        if rotb_conj > int(self.config.max_rotatable_bonds_conjugated):
                            logger.debug(
                                "Skipping generated SMILES (rotatable bonds in conjugated core %d > %d): %s",
                                rotb_conj,
                                self.config.max_rotatable_bonds_conjugated,
                                smiles,
                            )
                            rejected["filtered"] += 1
                            continue
                if mol is not None:
                    if self.config.physchem_filters and not self._physchem_ok(mol):
                        logger.debug("Skipping generated SMILES (fails physchem filters): %s", smiles)
                        rejected["filtered"] += 1
                        continue
                    if getattr(self.config, "sa_score_max", None) is not None and _sa_score is not None:
                        sa = _sa_score(mol)
                        if sa > float(self.config.sa_score_max):
                            logger.debug("Skipping generated SMILES (SA %.2f > %.2f): %s", sa, self.config.sa_score_max, smiles)
                            rejected["filtered"] += 1
                            continue
                    if getattr(self.config, "scaffold_unique", False):
                        scaffold = self._murcko_scaffold(mol)
                        if scaffold and scaffold in self._scaffolds_seen:
                            logger.debug("Skipping generated SMILES (duplicate scaffold): %s", smiles)
                            rejected["duplicate"] += 1
                            continue
                        if scaffold:
                            self._scaffolds_seen.add(scaffold)
                if mol is None and RDKit_AVAILABLE:
                    rejected["invalid"] += 1
                    continue
                if not skip_diversity and not self._passes_diversity(smiles):
                    logger.debug("Filtered out %s wegen diversity threshold.", smiles)
                    rejected["filtered"] += 1
                    continue
                if self._filter_indices and not skip_property_filters and not self._passes_property_filters(smiles):
                    logger.debug("Skipping generated SMILES (failt property filters): %s", smiles)
                    rejected["filtered"] += 1
                    continue
                existing.add(smiles)
                new_rows.append({"smiles": smiles, "assembly_status": status})
            if not new_rows:
                #  relaxed assembly/sample wenn nichts dazugekommen ist
                if not relaxed_used and attempts >= max(1, max_attempts // 2):
                    relaxed_used = True
                    use_relaxed = True
                    attempts -= 1  #zählt nicht als vollwertiger versuch
                    relaxed_adj = min((assemble_kwargs or self.assemble_kwargs).get("adjacency_threshold", 0.7), 0.4)
                    relaxed_nodes = min((assemble_kwargs or self.assemble_kwargs).get("max_tree_nodes", 8) or 8, 6)
                    relaxed_beam = max((assemble_kwargs or self.assemble_kwargs).get("beam_width", 3), 5)
                    logger.info(
                        "keine accepted samples bis jetzt; nochmal mit relaxed assembly (adjacency_threshold=%s, max_tree_nodes=%s, beam_width=%s).",
                        relaxed_adj,
                        relaxed_nodes,
                        relaxed_beam,
                    )
                    continue
                # wenn immernoch nichts accepted, dann property filter ausschalten
                if self._filter_indices and not skip_property_filters and attempts >= max_attempts:
                    logger.info("No candidates accepted with property filters; retrying once with property filters disabled.")
                    skip_property_filters = True
                    attempts = 0
                    continue
                # finaler structural relaxation fallback
                if (
                    getattr(self.config, "auto_relax_filters", False)
                    and not relax_structural
                    and attempts >= max_attempts
                    and generated == 0
                ):
                    relax_structural = True
                    skip_diversity = True
                    logger.info(
                        "Generation stalled;  structural relaxation (disable pi_fraction/branch caps, lower alternation requirement, skip diversity) und nochaml"
                    )
                    attempts = 0
                    continue
                continue
            use_relaxed = False
            self.pool = pd.concat([self.pool, pd.DataFrame(new_rows)], ignore_index=True)
            generated += len(new_rows)
        if len(self.pool) < min_size:
            logger.warning(
                "Generator sampling exhausted after %d attempt(s): pool size %d (target %d). "
                "Generated %d new entries; rejected invalid=%d duplicate=%d filtered=%d. "
                "filter sind kacke oder unleeren seed pool geben",
                attempts,
                len(self.pool),
                min_size,
                generated,
                rejected["invalid"],
                rejected["duplicate"],
                rejected["filtered"],
            )
        return generated

    def _featurize_pool(self) -> List:
        graphs = []
        valid_indices = []
        invalid_indices = []
        for idx, row in self.pool.iterrows():
            smiles = row["smiles"]
            if RDKit_AVAILABLE:
                mol = self._parse_smiles(smiles)
                if mol is None:
                    logger.warning("Skipping invalid SMILES %s during featurization", smiles)
                    invalid_indices.append(idx)
                    continue
            data = self._build_graph(smiles, row=row)
            if data is None:
                logger.warning("Skipping invalid SMILES %s: featurization failed", smiles)
                invalid_indices.append(idx)
                continue
            graphs.append(data)
            valid_indices.append(idx)
        if invalid_indices:
            self.pool = self.pool.drop(index=invalid_indices).reset_index(drop=True)
            valid_indices = list(range(len(graphs)))
            logger.info(
                "Dropped %d invalid pool entries nach featurization; %d candidates remain.",
                len(invalid_indices),
                len(graphs),
            )
        return graphs, valid_indices

    def _predict_pool(self, graphs: List):
        mean, std, _ = self.surrogate.predict(graphs, batch_size=self.config.batch_size)
        if self._target_indices:
            mean = mean[:, self._target_indices]
            std = std[:, self._target_indices]
        logger.debug("Mapped surrogate outputs to %d targets mit indices %s.", mean.shape[1], self._target_indices)
        return mean, std

    def _score_candidates(self, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        norm_mean, norm_std, mus, sigmas, directions = self._normalise_predictions(mean, std)
        best = self._current_best()
        norm_best = None
        if best is not None:
            norm_best = ((best * directions) - mus) / sigmas
        acq_cfg = self.config.acquisition
        cfg = AcquisitionConfig(
            kind=acq_cfg.kind,
            beta=acq_cfg.beta,
            xi=acq_cfg.xi,
            maximise=acq_cfg.maximise,
            weights=acq_cfg.weights,
        )
        if cfg.kind in {"pareto", "pareto_ucb"}:
            cfg.maximise = [True] * norm_mean.shape[1]
        scores = acquisition_score(norm_mean, norm_std, cfg, best_so_far=norm_best)
        return scores

    def _label_with_dft(self, selected: pd.DataFrame) -> pd.DataFrame:
        if self.dft is None:
            return selected
        inverse_alias = {v: k for k, v in self.property_aliases.items()}
        requested_props_unique: List[str] = []
        for column in self.config.target_columns:
            base = inverse_alias.get(column, column)
            if base not in requested_props_unique:
                requested_props_unique.append(base)
        jobs = [
            DFTJobSpec(smiles=row["smiles"], properties=requested_props_unique, **self._dft_job_defaults)
            for _, row in selected.iterrows()
        ]
        ids = self.dft.submit_batch(jobs)
        results = []
        for job_id in ids:
            res = self.dft.fetch(job_id, block=True, poll_interval=1.0)
            results.append(res)
        for df_row, res in zip(selected.itertuples(index=True), results):
            if res is None:
                continue
            if res.status != "success":
                if res.error_message:
                    logger.warning(
                        "QC job %s returned status %s: %s",
                        res.job.job_id,
                        res.status,
                        res.error_message,
                    )
                else:
                    logger.warning("QC job %s returned status %s", res.job.job_id, res.status)
            self._apply_result(df_row.Index, selected, res)
            selected.at[df_row.Index, "qc_status"] = res.status
            selected.at[df_row.Index, "qc_wall_time"] = res.wall_time
            selected.at[df_row.Index, "qc_error"] = res.error_message
            if res.metadata:
                for meta_key in ("total_energy", "basis"):
                    if meta_key in res.metadata:
                        selected.at[df_row.Index, meta_key] = res.metadata[meta_key]
                selected.at[df_row.Index, "qc_metadata"] = json.dumps(res.metadata, ensure_ascii=False)
        return selected

    def _apply_result(self, row_index: int, frame: pd.DataFrame, result: DFTResult) -> Dict[str, float]:
        mapped = {}
        for prop, value in result.properties.items():
            column = self.property_aliases.get(prop)
            if column is None and prop.endswith("_eV"):
                column = self.property_aliases.get(prop[:-3])
            if column is None and prop.endswith("_nm"):
                column = self.property_aliases.get(prop[:-3])
            if column is None:
                column = prop
            frame.at[row_index, column] = value
            mapped[column] = value
        return mapped

    def _retrain_surrogate(self) -> None:
        if len(self.labelled) < len(self.config.target_columns) + 5:
            return
        cols = ["smiles", *self.config.target_columns]
        if "mol" in self.labelled.columns:
            cols = ["mol"] + cols
        train_df = self.labelled[cols].dropna()
        if train_df.empty:
            return
        split = split_dataframe(train_df, val_fraction=0.1, test_fraction=0.0, seed=self.scheduler.iteration + 42)
        logger.info("Retraining surrogate on %d molecules", len(split.train))
        if self._is_schnet and hasattr(self.surrogate, "fit"):
            self.surrogate.fit(split.train, split.val)
        else:
            self.surrogate.fit(split.train, split.val)
        self._refresh_target_indices()

    def run_iteration(
        self,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        if self.scheduler.should_stop():
            raise RuntimeError("Maximum number of iterations erreicht")

        if assemble_kwargs is None:
            assemble_kwargs = self.assemble_kwargs
        logger.info(
            "Starting iteration %d | labelled=%d pool=%d | assemble_kwargs=%s",
            self.scheduler.iteration + 1,
            len(self.labelled),
            len(self.pool),
            assemble_kwargs,
        )
        generated = self._ensure_pool(self.config.batch_size, cond, assemble_kwargs)
        logger.debug("Pool replenished mit %d new samples (wenn ueberhaupt). Current pool=%d", generated, len(self.pool))
        graphs, valid_idx = self._featurize_pool()
        if not graphs:
            raise RuntimeError(
                "Keine valid candidates im pool nach filtering von invalid SMILES. "
                "Generation/seed pool hat 0 usable molecules; relax filters oder fuege mehr seed molecules hinzu"
            )
        if self.config.max_pool_eval is not None and len(graphs) > self.config.max_pool_eval:
            logger.info(
                "Capping pool evaluation zu first %d of %d candidates.",
                self.config.max_pool_eval,
                len(graphs),
            )
            graphs = graphs[: self.config.max_pool_eval]
            valid_idx = valid_idx[: self.config.max_pool_eval]
        logger.debug("Featurized %d pool candidates (valid_idx=%d).", len(graphs), len(valid_idx))

        mean, std = self._predict_pool(graphs)
        logger.debug("Predictions ready: mean shape %s, std shape %s", mean.shape, std.shape)
        scores = self._score_candidates(mean, std)
        logger.debug("Acquisition scores computed")

        pool_slice = self.pool.iloc[valid_idx].copy()
        for i, name in enumerate(self.config.target_columns):
            pool_slice[f"pred_{name}"] = mean[:, i]
            pool_slice[f"pred_std_{name}"] = std[:, i]
        pool_slice["acquisition_score"] = scores
        iteration_idx = self.scheduler.iteration + 1
        self._save_diagnostics(pool_slice, iteration_idx)

        selected = (
            pool_slice.sort_values("acquisition_score", ascending=False)
            .head(self.config.batch_size)
            .copy()
        )
        self.pool = self.pool.drop(selected.index).reset_index(drop=True)

        labelled = self._label_with_dft(selected)
        labelled["iteration"] = self.scheduler.iteration + 1
        self.labelled = pd.concat([self.labelled, labelled], ignore_index=True)
        self.history.append(labelled)

        self.scheduler.step(num_labelled=len(labelled), num_generated=generated)

        if self.scheduler.should_retrain_surrogate():
            self._retrain_surrogate()

        if self.scheduler.should_refresh_generator():
            self._refresh_generator()

        return labelled

    def run(
        self,
        n_iterations: int,
        *,
        cond: Optional[np.ndarray] = None,
        assemble_kwargs: Optional[Dict] = None,
    ) -> List[pd.DataFrame]:
        if assemble_kwargs is None:
            assemble_kwargs = {}
        merged_kwargs = {**self.assemble_kwargs, **assemble_kwargs}
        for i in range(n_iterations):
            if self.scheduler.should_stop():
                break
            logger.info(
                "Loop progress: iteration %d/%d (labelled=%d, pool=%d)",
                i + 1,
                n_iterations,
                len(self.labelled),
                len(self.pool),
            )
            self.run_iteration(cond=cond, assemble_kwargs=merged_kwargs)
        return self.history

    def save_history(self) -> None:
        if not self.history:
            return
        path = self.results_dir / "active_learning_history.csv"
        pd.concat(self.history, ignore_index=True).to_csv(path, index=False)
        logger.info("Fertig, active learning history (ergebnnisse) in %s", path)
