"""Grid search for disagreement threshold τ on OOF gated predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rgde.ensemble import compute_ensemble_oof
from rgde.evaluation import evaluate_rmse


def tune_tau_grid(
    oof_predictions: pd.DataFrame,
    cv_rmse: dict[str, float],
    y_true: np.ndarray | pd.Series,
    tau_candidates: np.ndarray | list[float],
) -> tuple[float, dict[float, float]]:
    """
    Select τ minimizing ensemble CV RMSE on out-of-fold gated predictions.

    Note
    ----
    Tuning τ on the same OOF matrix used to build weights is standard for
    meta-parameters in stacking but can be slightly optimistic; for strict
    reporting, reserve a hold-out set or use nested CV.
    """
    scores: dict[float, float] = {}
    y_arr = y_true.to_numpy(dtype=float) if isinstance(y_true, pd.Series) else np.asarray(y_true, dtype=float).ravel()
    for tau in tau_candidates:
        gated, _, _, _ = compute_ensemble_oof(oof_predictions, cv_rmse, y_arr, float(tau))
        scores[float(tau)] = evaluate_rmse(y_arr, gated)
    best_tau = min(scores, key=scores.get)  # type: ignore[arg-type]
    return best_tau, scores


def default_tau_grid(disagreement: np.ndarray, n_points: int = 25) -> np.ndarray:
    """Quantile-spaced grid from observed OOF disagreement (includes a small epsilon min)."""
    d = np.asarray(disagreement, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.linspace(0.01, 1.0, n_points)
    lo = max(float(np.min(d)), 1e-8)
    hi = max(float(np.max(d)), lo * 1.01)
    return np.unique(
        np.concatenate(
            [
                np.linspace(lo, hi, n_points),
                np.quantile(d, np.linspace(0.05, 0.95, min(10, n_points))),
            ]
        )
    )
