"""scoring utils fuer kandidaten flexibles weighting + pareto helper"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

__all__ = [
    "PropertyObjective",
    "score_properties",
    "pareto_front",
]


@dataclass
class PropertyObjective:
    """beschreibung eines property ziels"""

    name: str
    target: float
    weight: float = 1.0
    tolerance: float = 0.0
    direction: str = "max"  # "max", "min", or "target"
    penalty: float = 0.0  # optional hard penalty if violated

    def evaluate(self, value: float) -> float:
        if self.direction not in {"max", "min", "target"}:
            raise ValueError(f"Unknown direction {self.direction}")

        if self.direction == "target":
            diff = abs(value - self.target)
            score = -diff
        elif self.direction == "max":
            score = value - self.target
        else:  # min
            score = self.target - value

        if self.tolerance > 0 and abs(score) > self.tolerance:
            score -= self.penalty

        return self.weight * score


def score_properties(predictions: Mapping[str, float], objectives: Sequence[PropertyObjective]) -> float:
    """aggregiert surrogate vorhersagen zu einem scalar"""

    score = 0.0
    for obj in objectives:
        if obj.name not in predictions:
            raise KeyError(f"Prediction missing objective '{obj.name}'")
        score += obj.evaluate(predictions[obj.name])
    return score


def pareto_front(points: np.ndarray, maximise: Sequence[bool]) -> np.ndarray:
    """liefert boolean maske fuer pareto front eines multi ziel arrays

    Parameters
    ----------
    points:
        array [N, D] mit property werten
    maximise:
        sequenz ob jede dimension maximiert werden soll
    """

    if points.ndim != 2:
        raise ValueError("points must be 2D array")
    n, d = points.shape
    maximise = list(maximise)
    if len(maximise) != d:
        raise ValueError("maximise length mismatch")

    adjusted = points.copy()
    for j, do_max in enumerate(maximise):
        if not do_max:
            adjusted[:, j] = -adjusted[:, j]

    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominance = np.all(adjusted[i] >= adjusted, axis=1) & np.any(adjusted[i] > adjusted, axis=1)
        dominance[i] = False
        mask[dominance] = False
    return mask
