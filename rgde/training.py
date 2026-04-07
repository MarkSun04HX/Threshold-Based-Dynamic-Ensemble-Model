"""Fit estimators on full training data (no leakage)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone


def train_model(
    estimator: BaseEstimator,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> BaseEstimator:
    """
    Clone and fit ``estimator`` on (X, y).

    Cloning avoids mutating a template pipeline passed from a factory.
    """
    X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    y_arr = y.to_numpy(dtype=float) if isinstance(y, pd.Series) else np.asarray(y, dtype=float).ravel()
    fitted = clone(estimator)
    fitted.fit(X_arr, y_arr)
    return fitted
