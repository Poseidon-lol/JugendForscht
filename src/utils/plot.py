"""
Plotting helpers
================

Utility routines for quick-look visualisations (loss curves, property
histograms) used throughout notebooks and experiment scripts.  The module
degrades gracefully when :mod:`matplotlib` is not available by returning
``None`` so that callers can decide how to proceed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, TYPE_CHECKING

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

try:
    import seaborn as sns

    if _HAS_MPL:
        sns.set_style("whitegrid")
    _HAS_SNS = True
except Exception:
    sns = None
    _HAS_SNS = False

__all__ = [
    "plot_learning_curve",
    "plot_property_histogram",
]

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _require_matplotlib() -> None:
    if not _HAS_MPL:
        raise RuntimeError(
            "matplotlib is required for plotting. Install via `pip install matplotlib seaborn`."
        )


def plot_learning_curve(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    *,
    title: str = "Training Curve",
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """Plot loss curves and optionally persist figure."""

    if not _HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(train_losses, label="train")
    if val_losses is not None:
        ax.plot(val_losses, label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig


def plot_property_histogram(
    values: Sequence[float],
    *,
    title: str = "Property Distribution",
    xlabel: str = "Value",
    save_path: Optional[str] = None,
) -> Optional["Figure"]:
    """Plot histogram of property values."""

    if not _HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if _HAS_SNS:
        sns.histplot(values, kde=True, ax=ax)  # type: ignore[arg-type]
    else:
        ax.hist(values, bins=30, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    return fig
