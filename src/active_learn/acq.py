"""acquisition funktionen für active learning"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from pathlib import Path
import sys


PROJECT_ROOT = Path().resolve()
for candidate in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (candidate / "src").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("project root ohne src gefunden")

import numpy as np
from src.models.scorer import pareto_front

__all__ = [
    "AcquisitionConfig",
    "acquisition_score",
    "upper_confidence_bound",
    "expected_improvement",
    "probability_improvement",
    "pareto_rank",
]


@dataclass
class AcquisitionConfig:
    kind: str = "ucb"  # oder "ei", "pi", "pareto"
    beta: float = 1.0
    xi: float = 0.01
    maximise: Optional[Sequence[bool]] = None
    weights: Optional[Sequence[float]] = None
    targets: Optional[Sequence[float]] = None
    tolerances: Optional[Sequence[float]] = None


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return mean + beta * std


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: np.ndarray, xi: float = 0.01) -> np.ndarray:
    improvement = mean - best - xi
    with np.errstate(divide="ignore"):
        Z = improvement / (std + 1e-12)
    ei = improvement * _norm_cdf(Z) + std * _norm_pdf(Z)
    ei[std < 1e-12] = 0.0
    return ei


def probability_improvement(mean: np.ndarray, std: np.ndarray, best: np.ndarray, xi: float = 0.01) -> np.ndarray:
    with np.errstate(divide="ignore"):
        Z = (mean - best - xi) / (std + 1e-12)
    return _norm_cdf(Z)


def pareto_rank(mean: np.ndarray, maximise: Sequence[bool]) -> np.ndarray:
    mask = pareto_front(mean, maximise)
    scores = np.zeros(mean.shape[0])
    scores[mask] = 1.0
    return scores


def acquisition_score(
    mean: np.ndarray,
    std: np.ndarray,
    config: AcquisitionConfig,
    best_so_far: Optional[np.ndarray] = None,
) -> np.ndarray:
    """berechnet acquisition scores wie in der configuration"""

    if mean.ndim == 1:
        mean = mean[:, None]
    if std.ndim == 1:
        std = std[:, None]

    if config.kind == "ucb":
        scores = upper_confidence_bound(mean, std, beta=config.beta)
        return scores.mean(axis=1)
    if config.kind == "multi_ucb":
        scores = upper_confidence_bound(mean, std, beta=config.beta)
        return scores.sum(axis=1)
    if config.kind == "pareto_ucb":
        scores = upper_confidence_bound(mean, std, beta=config.beta)
        maximise = config.maximise if config.maximise is not None else [True] * scores.shape[1]
        return pareto_rank(scores, maximise)
    if config.kind == "ei":
        if best_so_far is None:
            raise ValueError("best_so_far gebraucht für erwartete verbesserung")
        best = np.asarray(best_so_far)
        if best.ndim == 0:
            best = np.full(mean.shape[1], best)
        ei = expected_improvement(mean, std, best, xi=config.xi)
        return ei.mean(axis=1)
    if config.kind == "pi":
        if best_so_far is None:
            raise ValueError("best_so_far gebrtaucht für wahrschienlichkeit von verbesserung")
        best = np.asarray(best_so_far)
        if best.ndim == 0:
            best = np.full(mean.shape[1], best)
        pi = probability_improvement(mean, std, best, xi=config.xi)
        return pi.mean(axis=1)
    if config.kind == "pareto":
        if config.maximise is None:
            raise ValueError("maximise gebraucht für pareto acquisition")
        return pareto_rank(mean, config.maximise)

    if config.kind == "target":
        if config.targets is None:
            raise ValueError("targets gebraucht target acquisition")
        targets = np.asarray(config.targets, dtype=float)
        if targets.shape[0] != mean.shape[1]:
            raise ValueError("targets length ungleich mit mean dimension")
        diff = np.abs(mean - targets)
        if config.tolerances is not None:
            tol = np.asarray(config.tolerances, dtype=float)
            if tol.shape[0] != mean.shape[1]:
                raise ValueError("tolerances length ungleich mit mean dimension")
            tol = np.where(tol <= 0, 1.0, tol)
            diff = diff / tol
        if config.weights is not None:
            weights = np.asarray(config.weights, dtype=float)
            if weights.shape[0] != mean.shape[1]:
                raise ValueError("weights length ungleich mit mean dimension")
            diff = diff * weights
        score = -diff.sum(axis=1)
        if config.beta:
            score = score + float(config.beta) * std.mean(axis=1)
        return score

    if config.weights is not None:
        weights = np.asarray(config.weights)
        if weights.shape[0] != mean.shape[1]:
            raise ValueError("weights length ungleich mit mean dimension")
        return (weights * mean).sum(axis=1)

    raise ValueError(f"unbekannter acquisition art'{config.kind}'")
SQRT_2 = np.sqrt(2.0)
SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.erf(z / SQRT_2))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return (1.0 / SQRT_2PI) * np.exp(-0.5 * z ** 2)
