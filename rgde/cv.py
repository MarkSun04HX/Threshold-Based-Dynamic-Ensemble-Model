"""Out-of-fold cross-validation with no leakage between train and validation folds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

from rgde.config import N_CV_FOLDS, RANDOM_STATE
from rgde.evaluation import evaluate_rmse


def cross_validate_model(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_folds: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, float]:
    """
    Generate out-of-fold predictions and CV RMSE.

    Each fold: clone(estimator).fit(train).predict(val). No information from
    validation folds enters training for that row's OOF prediction.
    """
    X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    y_arr = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float).ravel()
    n = X_arr.shape[0]
    oof = np.full(n, np.nan, dtype=float)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X_arr):
        est: Any = clone(estimator)
        est.fit(X_arr[train_idx], y_arr[train_idx])
        oof[val_idx] = est.predict(X_arr[val_idx])

    if np.any(np.isnan(oof)):
        raise RuntimeError("OOF predictions contain NaN; check CV splits and model fits")

    rmse = evaluate_rmse(y_arr, oof)
    return oof, rmse


def cross_validate_all_models(
    estimators: dict[str, BaseEstimator],
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_folds: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Run OOF CV for every base learner.

    Returns
    -------
    oof_predictions : DataFrame, shape (n_samples, n_models)
    cv_rmse : dict mapping model name -> RMSE on OOF predictions
    """
    cols: dict[str, np.ndarray] = {}
    rmses: dict[str, float] = {}
    for name, est in estimators.items():
        oof, rmse = cross_validate_model(est, X, y, n_folds=n_folds, random_state=random_state)
        cols[name] = oof
        rmses[name] = rmse
    oof_df = pd.DataFrame(cols)
    return oof_df, rmses
