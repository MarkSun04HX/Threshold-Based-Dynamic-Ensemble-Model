"""
Threshold-Based Dynamic Ensemble (TBDE) — stub models + inner CV + coalition selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

Selection = Literal["top_k", "threshold"]


def _fold_ids(n_rows: int, k_folds: int) -> np.ndarray:
    """Sequential fold assignment (same idea as R cut(seq_len(n), k_folds))."""
    if k_folds < 1:
        raise ValueError("k_folds must be >= 1")
    return (np.arange(n_rows) * k_folds) // n_rows + 1


def _numeric_X(df: pd.DataFrame, target: str) -> pd.DataFrame:
    X = df.drop(columns=[target])
    num = X.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("No numeric feature columns after dropping target")
    return num


class _LinearFit:
    """OLS via numpy (matches R lm for numeric features; no sklearn dependency)."""

    def __init__(self, coef: np.ndarray, feature_cols: list[str]):
        self.coef = coef  # length p+1: intercept, then features
        self.feature_cols = feature_cols

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        X = test[self.feature_cols].astype(float).values
        X1 = np.column_stack([np.ones(len(X)), X])
        return X1 @ self.coef


def _model_definitions(target: str) -> dict[str, Callable[[pd.DataFrame], object | float]]:
    def xgboost(train: pd.DataFrame) -> float:
        return float(train[target].mean())

    def catboost(train: pd.DataFrame) -> float:
        return float(train[target].median())

    def neural_net(train: pd.DataFrame) -> float:
        return float(train[target].mean() * 1.05)

    def linear_reg(train: pd.DataFrame) -> _LinearFit:
        X = _numeric_X(train, target)
        y = train[target].values.astype(float)
        Xm = X.values.astype(float)
        X1 = np.column_stack([np.ones(len(Xm)), Xm])
        coef, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
        return _LinearFit(coef, list(X.columns))

    def knn(train: pd.DataFrame) -> float:
        s = train[target].tail(3)
        return float(s.mean())

    def random_forest(train: pd.DataFrame) -> float:
        return float(train[target].mean())

    def elastic_net(train: pd.DataFrame) -> float:
        return float(train[target].mean() * 0.98)

    def svm(train: pd.DataFrame) -> float:
        return float(train[target].median())

    def lightgbm(train: pd.DataFrame) -> float:
        return float(train[target].mean())

    def decision_tree(train: pd.DataFrame) -> float:
        return float(train[target].mean())

    return {
        "XGBoost": xgboost,
        "CatBoost": catboost,
        "NeuralNet": neural_net,
        "LinearReg": linear_reg,
        "KNN": knn,
        "RandomForest": random_forest,
        "ElasticNet": elastic_net,
        "SVM": svm,
        "LightGBM": lightgbm,
        "DecisionTree": decision_tree,
    }


def predict_tbde_ensemble(
    train: pd.DataFrame,
    test: pd.DataFrame,
    coalition: list[str],
    target: str,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Unweighted mean of coalition predictions, or weighted combination."""
    models = _model_definitions(target)
    if not coalition:
        raise ValueError("coalition must be non-empty")
    n = len(test)
    pred_mat = np.empty((n, len(coalition)))
    for j, name in enumerate(coalition):
        if name not in models:
            raise ValueError(f"unknown model in coalition: {name}")
        fit = models[name](train)
        if isinstance(fit, (int, float, np.floating)):
            pred_mat[:, j] = float(fit)
        else:
            pred_mat[:, j] = fit.predict(test)
    if weights is None:
        return pred_mat.mean(axis=1)
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(coalition),):
        raise ValueError("weights must have one value per coalition member")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    s = w.sum()
    if s <= 0:
        return pred_mat.mean(axis=1)
    return pred_mat @ (w / s)


def _cv_rmse_per_model(
    data: pd.DataFrame,
    models: dict[str, Callable[[pd.DataFrame], object | float]],
    target: str,
    k_folds: int,
) -> dict[str, float]:
    n = len(data)
    folds = _fold_ids(n, k_folds)
    rmse: dict[str, float] = {}
    for m_name, fn in models.items():
        errors: list[float] = []
        for i in range(1, k_folds + 1):
            test_mask = folds == i
            train_df = data.loc[~test_mask].reset_index(drop=True)
            test_df = data.loc[test_mask].reset_index(drop=True)
            pred = fn(train_df)
            if isinstance(pred, (int, float, np.floating)):
                actual = np.full(len(test_df), float(pred))
            else:
                actual = pred.predict(test_df)
            y = test_df[target].values
            errors.extend(((y - actual) ** 2).tolist())
        rmse[m_name] = float(np.sqrt(np.mean(errors)))
    return rmse


@dataclass
class CoalitionResult:
    """Output of :func:`build_coalition`."""

    models: list[str]
    cv_rmse: Optional[dict[str, float]] = None


def build_coalition(
    data: pd.DataFrame,
    threshold: float = 5.0,
    k_folds: int = 3,
    seed: int = 123,
    target: str = "quality",
    verbose: bool = True,
    return_rmse: bool = False,
    selection: Selection = "top_k",
    top_k: int = 3,
) -> CoalitionResult:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if target not in data.columns:
        raise ValueError(f"data must contain target column '{target}'")
    if top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if selection == "threshold" and threshold <= 0:
        raise ValueError('threshold must be positive when selection == "threshold"')

    _ = seed  # reserved if folds become stochastic

    models = _model_definitions(target)
    rmse_map = _cv_rmse_per_model(data, models, target, k_folds)
    rows = sorted(rmse_map.items(), key=lambda x: x[1])
    results = pd.DataFrame(rows, columns=["Model", "RMSE"])
    n_models = len(results)

    if selection == "top_k":
        nk = min(top_k, n_models)
        selected = results.iloc[:nk]
    else:
        selected = results[results["RMSE"] < threshold]
        if selected.empty:
            if verbose:
                print(f"⚠️ No models met threshold. Picking top {top_k}.")
            nk = min(top_k, n_models)
            selected = results.iloc[:nk]

    if verbose:
        print("\n✅ Final Coalition:")
        print(selected.to_string(index=False))

    names = selected["Model"].tolist()
    cv = {m: rmse_map[m] for m in names} if return_rmse else None
    return CoalitionResult(models=names, cv_rmse=cv)


def evaluate_coalition(
    coalition: list[str],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target: str = "quality",
    weights: Optional[np.ndarray] = None,
) -> float:
    if not coalition:
        raise ValueError("Coalition cannot be empty")
    if target not in test_data.columns or target not in train_data.columns:
        raise ValueError(f"train and test must contain '{target}'")
    y_hat = predict_tbde_ensemble(train_data, test_data, coalition, target, weights=weights)
    y = test_data[target].values
    return float(np.mean(np.abs(y - y_hat)))
