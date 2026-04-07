"""RMSE weights, disagreement, and gated combination of OOF or test predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from rgde.evaluation import evaluate_rmse


def rmse_dict_to_weights(cv_rmse: dict[str, float], eps: float = 1e-12) -> dict[str, float]:
    """
    weight_i = 1 / RMSE_i, normalized to sum to 1.
    """
    names = list(cv_rmse.keys())
    inv = np.array([1.0 / max(cv_rmse[m], eps) for m in names], dtype=float)
    s = inv.sum()
    if s <= 0:
        raise ValueError("Invalid RMSE values for weighting")
    return {m: float(inv[i] / s) for i, m in enumerate(names)}


def prediction_disagreement(pred_matrix: np.ndarray | pd.DataFrame, axis: int = 1) -> np.ndarray:
    """
    Per-sample standard deviation across model predictions.

    Parameters
    ----------
    pred_matrix : array-like, shape (n_samples, n_models)
    """
    if isinstance(pred_matrix, pd.DataFrame):
        M = pred_matrix.to_numpy(dtype=float)
    else:
        M = np.asarray(pred_matrix, dtype=float)
    return np.std(M, axis=axis, ddof=0)


def compute_gated_predictions(
    pred_matrix: np.ndarray | pd.DataFrame,
    weights: dict[str, float],
    best_model_name: str,
    tau: float,
) -> np.ndarray:
    """
    If disagreement < tau: weighted average of all models.
    Else: prediction from the single best model (lowest CV RMSE).
    """
    if isinstance(pred_matrix, pd.DataFrame):
        names = list(pred_matrix.columns)
        M = pred_matrix.to_numpy(dtype=float)
    else:
        raise TypeError("pred_matrix must be a DataFrame with column names matching weights keys")

    w_vec = np.array([weights[m] for m in names], dtype=float)
    if not np.isclose(w_vec.sum(), 1.0, rtol=1e-5):
        w_vec = w_vec / w_vec.sum()

    weighted = M @ w_vec
    if best_model_name not in names:
        raise ValueError(f"best_model_name {best_model_name!r} not in prediction columns")
    best_col = names.index(best_model_name)
    best_only = M[:, best_col]

    d = prediction_disagreement(M)
    mask = d < tau
    return np.where(mask, weighted, best_only)


def compute_ensemble_oof(
    oof_predictions: pd.DataFrame,
    cv_rmse: dict[str, float],
    y_true: np.ndarray | pd.Series,
    tau: float,
) -> tuple[np.ndarray, dict[str, float], str, float]:
    """
    Apply gating to out-of-fold predictions; return gated OOF vector and ensemble CV RMSE.
    """
    y_arr = y_true.to_numpy(dtype=float) if isinstance(y_true, pd.Series) else np.asarray(y_true, dtype=float).ravel()
    best_model_name = min(cv_rmse, key=cv_rmse.get)  # type: ignore[arg-type]
    weights = rmse_dict_to_weights(cv_rmse)
    gated = compute_gated_predictions(oof_predictions, weights, best_model_name, tau)
    ensemble_cv_rmse = evaluate_rmse(y_arr, gated)
    return gated, weights, best_model_name, ensemble_cv_rmse


def compute_ensemble(
    oof_predictions: pd.DataFrame,
    cv_rmse: dict[str, float],
    y_true: np.ndarray | pd.Series,
    tau: float,
) -> dict[str, object]:
    """
    Convenience wrapper returning a report dict (OOF gated predictions + metadata).
    """
    gated, weights, best_name, ens_rmse = compute_ensemble_oof(
        oof_predictions, cv_rmse, y_true, tau
    )
    return {
        "gated_oof_predictions": gated,
        "weights": weights,
        "best_model_name": best_name,
        "ensemble_cv_rmse": ens_rmse,
        "disagreement_oof": prediction_disagreement(oof_predictions),
    }


def compute_test_predictions_matrix(
    fitted_models: dict[str, BaseEstimator],
    X_test: np.ndarray | pd.DataFrame,
) -> pd.DataFrame:
    """Stack test predictions from each fitted model into a DataFrame."""
    X_arr = X_test.to_numpy(dtype=float) if isinstance(X_test, pd.DataFrame) else np.asarray(X_test, dtype=float)
    cols: dict[str, np.ndarray] = {}
    for name, model in fitted_models.items():
        cols[name] = model.predict(X_arr)
    return pd.DataFrame(cols)
