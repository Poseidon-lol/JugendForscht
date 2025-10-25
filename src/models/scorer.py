"""
Scoring utilities for candidate molecules.
==========================================

The active-learning loop needs scalar scores to rank molecules according to
multiple surrogate predictions (HOMO, LUMO, EA, IE, conductivity, etc.).
This module implements a flexible weighting/constraint system together with
Pareto-front helpers.
"""

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
    """Specification of a single property objective."""

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
    """Aggregate surrogate predictions into a scalar score."""

    score = 0.0
    for obj in objectives:
        if obj.name not in predictions:
            raise KeyError(f"Prediction missing objective '{obj.name}'")
        score += obj.evaluate(predictions[obj.name])
    return score


def pareto_front(points: np.ndarray, maximise: Sequence[bool]) -> np.ndarray:
    """Return boolean mask for Pareto front of multi-objective array.

    Parameters
    ----------
    points:
        Array of shape [N, D] with property values.
    maximise:
        Sequence indicating whether each dimension should be maximised.
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
