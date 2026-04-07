"""Optional matplotlib plots for diagnostics (bonus)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _get_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Install matplotlib for plotting: pip install matplotlib") from e
    return plt


def plot_rmse_comparison(
    cv_rmse: dict[str, float],
    *,
    ensemble_rmse: float | None = None,
    title: str = "CV RMSE by model",
    figsize: tuple[float, float] = (8, 4),
    save_path: str | Path | None = None,
) -> Any:
    """
    Bar chart of per-model OOF RMSE; optional horizontal line for ensemble RMSE.

    Returns matplotlib figure (caller may ``plt.show()``).
    """
    plt = _get_matplotlib()
    names = list(cv_rmse.keys())
    values = [cv_rmse[m] for m in names]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(names, values, color="steelblue", alpha=0.85)
    ax.set_ylabel("RMSE (OOF)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    if ensemble_rmse is not None:
        ax.axhline(ensemble_rmse, color="darkred", linestyle="--", label=f"Gated ensemble: {ensemble_rmse:.4f}")
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_disagreement_distribution(
    disagreement: np.ndarray,
    *,
    tau: float | None = None,
    title: str = "Distribution of prediction disagreement (OOF)",
    figsize: tuple[float, float] = (7, 4),
    save_path: str | Path | None = None,
    bins: int = 40,
) -> Any:
    """Histogram of per-sample std across model predictions."""
    plt = _get_matplotlib()
    d = np.asarray(disagreement, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(d, bins=bins, color="seagreen", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Disagreement (std across models)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    if tau is not None:
        ax.axvline(tau, color="darkred", linestyle="--", label=f"τ = {tau:.4f}")
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
