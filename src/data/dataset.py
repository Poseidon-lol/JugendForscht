"""
Data loading helpers for surrogate and generative models.


The project works primarily with CSV/Parquet files that contain SMILES rows
and target properties (HOMO/LUMO, electron affinity, ionisation energy,
doping metrics, etc.).  This module provides convenient primitives to:

* ingest tabular datasets and ensure consistent column naming
* compute / store normalisation statistics for target columns
* perform deterministic train/val/test splits
* create PyTorch-Geometric compatible dataloaders

The functions are intentionally lightweight and avoid imposing a specific
framework (PyTorch Lightning, Hydra, ...).  They are used by both the
surrogate training scripts and the JT-VAE preprocessing pipeline.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure the project root (with src/) is on sys.path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = None
for candidate in [_THIS_FILE.parent, *_THIS_FILE.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if _PROJECT_ROOT is None and (candidate / "src").is_dir():
            _PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate project root containing src/")

if _PROJECT_ROOT is None:
    # Fallback: assume the parent of the `src` package is the project root.
    for candidate in _THIS_FILE.parents:
        if (candidate / "src").exists():
            _PROJECT_ROOT = candidate
            break
    else:  # pragma: no cover - defensive guard
        _PROJECT_ROOT = _THIS_FILE.parent

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
except Exception:  # pragma: no cover - optional dependency is validated elsewhere
    PyGDataLoader = None

from src.data.featurization import mol_to_graph

__all__ = [
    "NormalizationStats",
    "TrainValTestSplit",
    "load_dataframe",
    "split_dataframe",
    "compute_normalization",
    "apply_normalization",
    "create_property_dataset",
    "build_pyg_dataloaders",
]


@dataclass
class NormalizationStats:
    """Simple container für normalisations parameter (mean/std pro column)"""

    mean: pd.Series
    std: pd.Series

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {"mean": self.mean.to_dict(), "std": self.std.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Mapping[str, float]]) -> "NormalizationStats":
        return cls(mean=pd.Series(data["mean"]), std=pd.Series(data["std"]))

    def transform(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        df = df.copy()
        df[columns] = (df[columns] - self.mean[columns]) / self.std[columns]
        return df

    def inverse_transform(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        df = df.copy()
        df[columns] = df[columns] * self.std[columns] + self.mean[columns]
        return df


@dataclass
class TrainValTestSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


#resolve dataset paths wurde mit chatgpt gemacht weil ich kein bock hatte sowas zu implementieren
def resolve_dataset_path(candidate: Path) -> Path:
    """Resolve dataset paths in bezug zum project root und den common data folders"""

    candidate = candidate.expanduser()
    if candidate.exists():
        return candidate.resolve()

    if candidate.is_absolute():
        raise FileNotFoundError(f"Dataset path irgendwie falsch: {candidate}")

    direct = (_PROJECT_ROOT / candidate).resolve()
    if direct.exists():
        return direct

    data_root = _PROJECT_ROOT / "data"
    if data_root.exists():
        parts = candidate.parts
        trimmed = candidate
        if parts and parts[0].lower() == "data":
            trimmed = Path(*parts[1:]) if len(parts) > 1 else Path()

        variants = {candidate, trimmed}
        if trimmed != Path():
            variants.add(Path(trimmed.name))

        subdirs = [Path(), Path("raw"), Path("processed")]
        for sub in subdirs:
            base = data_root / sub
            for variant in variants:
                if variant == Path():
                    maybe = base
                else:
                    maybe = (base / variant).resolve()
                if maybe.exists() and maybe.is_file():
                    return maybe

        name_matches = [p for p in data_root.rglob(candidate.name) if p.is_file()] if candidate.name else []
        if name_matches:
            return name_matches[0].resolve()

        tokens = [token for token in candidate.stem.replace("-", "_").split("_") if token]
        if tokens:
            pattern = f"*{candidate.suffix}" if candidate.suffix else "*"
            fuzzy_matches = []
            for entry in data_root.rglob(pattern):
                if not entry.is_file():
                    continue
                entry_tokens = [token for token in entry.stem.replace("-", "_").split("_") if token]
                if all(token in entry_tokens for token in tokens):
                    fuzzy_matches.append(entry)
            if fuzzy_matches:
                fuzzy_matches.sort(key=lambda p: (len(p.stem), p.stat().st_size if p.exists() else 0))
                return fuzzy_matches[0].resolve()

    raise FileNotFoundError(f"Dataset path gibts nicht: {candidate}")

def load_dataframe(path: Path | str) -> pd.DataFrame:
    """dataset in ein DataFrame reinloaden"""

    resolved_path = resolve_dataset_path(Path(path))
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved_path)
    raise ValueError(f"Datei Format nachgucken '{resolved_path.suffix}'")



def split_dataframe(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.1, # portion of data for validation set
    test_fraction: float = 0.1, # portion of data for test set
    seed: int = 42, # random seed for reproducibility
) -> TrainValTestSplit:
    """Deterministic random split for train/val/test portions."""

    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1.0: 
        raise ValueError("werte iwie falsch") # Keine der Anteile darf negativ sein; Zusammen dürfen sie nicht ≥ 1 sein, weil sonst kein Platz mehr für Training bleibt

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True) # shuffle dataframe rows, damit die splits random sind Das ist wichtig und damit beim Split später die Reihenfolge keine Rolle spielt
    n = len(df)
    n_test = int(n * test_fraction) #anzahl der zeilen für testsatz
    n_val = int(n * val_fraction) #anzahl der zeilen für validierungssatz

    test_df = df.iloc[:n_test].reset_index(drop=True) #test datensatz
    val_df = df.iloc[n_test : n_test + n_val].reset_index(drop=True) #validierungs datensatz
    train_df = df.iloc[n_test + n_val :].reset_index(drop=True) #train datensatz
    return TrainValTestSplit(train=train_df, val=val_df, test=test_df)  #return the splits


def compute_normalization(df: pd.DataFrame, target_cols: Sequence[str]) -> NormalizationStats: #rechnet mean und std für columns aus
    if len(target_cols) == 0:
        raise ValueError("du brauchst mehr als ein target column um normalisierung zu machen")
    mean = df[target_cols].mean()
    std = df[target_cols].std().replace(0, 1.0)
    return NormalizationStats(mean=mean, std=std)


def apply_normalization(df: pd.DataFrame, stats: NormalizationStats, target_cols: Sequence[str]) -> pd.DataFrame: #apply normalisierung auf dataframe
    return stats.transform(df, target_cols)


def create_property_dataset(df: pd.DataFrame, *, cache_graphs: bool = False): 
    """Convert dataframe into a PyG dataset mit ``mol_to_graph`` aus src.featurization.""" 

    target_cols = [c for c in df.columns if c not in {"smiles", "id"}]
    graphs = []
    for _, row in df.iterrows():
        y = row[target_cols].values.astype(float) if target_cols else None
        data = mol_to_graph(row["smiles"], y=y)
        graphs.append(data)
        #Hier passiert der Hauptteil: 

        #Iteration über jede Zeile im DataFrame
        #→ row["smiles"] ist ein Molekülstring.

        #Zielwerte extrahieren
       # →  y enthält die Zielgrößen als numpy.float-Array.

        ##Molekül in Graph umwandeln
        #→ mol_to_graph() ist eine Funktion (aus src.featurization),
        #die den SMILES-String in ein PyG Data-Objekt umwandelt:

        #Knoten = Atome

        #Kanten = Bindungen

        #Features = atomare und chemische Eigenschaften

        #y = Zielwert(e) für das Molekül

        #Graph speichern
        # → Der Graph wird der Liste graphs hinzugefügt.

    if not cache_graphs:
        # return lazy dataset to avoid storing all Data objects at once
        from src.models.mpnn import MoleculeDataset  # local import to avoid circular on module load

        return MoleculeDataset(df)
    return graphs


def build_pyg_dataloaders(
    split: TrainValTestSplit,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create PyG dataloaders for train/val/test splits."""

    if PyGDataLoader is None:
        raise ImportError("torch_geometric is required to build graph dataloaders.")

    train_ds = create_property_dataset(split.train)
    val_ds = create_property_dataset(split.val)
    test_ds = create_property_dataset(split.test)

    loaders = {
        "train": PyGDataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers),
        "val": PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders
