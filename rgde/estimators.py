"""Sklearn ``Pipeline`` definitions for base learners (scaling where appropriate)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from rgde.config import RANDOM_STATE


def _scaled_linear(name: str, model: Any) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (name, model),
        ]
)


def build_base_estimators(
    *,
    include_random_forest: bool = True,
    include_xgboost: bool = True,
) -> dict[str, Pipeline]:
    """
    Factory for named sklearn Pipelines.

    Linear models use ``StandardScaler`` inside the pipeline (fit per fold in CV — no leakage).
    Tree-based models are unscaled.
    """
    estimators: dict[str, Pipeline] = {
        "linear_regression": _scaled_linear(
            "lr",
            LinearRegression(),
        ),
        "ridge": _scaled_linear(
            "ridge",
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
        ),
        "elastic_net": _scaled_linear(
            "enet",
            ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=RANDOM_STATE,
                max_iter=5000,
            ),
        ),
        "decision_tree": Pipeline(
            [
                (
                    "dt",
                    DecisionTreeRegressor(
                        max_depth=8,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if include_random_forest:
        estimators["random_forest"] = Pipeline(
            [
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=12,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if include_xgboost:
        try:
            from xgboost import XGBRegressor

            estimators["xgboost"] = Pipeline(
                [
                    (
                        "xgb",
                        XGBRegressor(
                            n_estimators=200,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        except ImportError:
            pass

    return estimators
