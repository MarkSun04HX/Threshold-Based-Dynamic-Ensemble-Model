"""End-to-end RMSE-gated dynamic ensemble."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from rgde.config import N_CV_FOLDS, RANDOM_STATE
from rgde.cv import cross_validate_all_models
from rgde.ensemble import (
    compute_gated_predictions,
    compute_ensemble_oof,
    compute_test_predictions_matrix,
    prediction_disagreement,
)
from rgde.estimators import build_base_estimators
from rgde.evaluation import evaluate_rmse
from rgde.training import train_model
from rgde.tuning import default_tau_grid, tune_tau_grid


@dataclass
class GatedEnsembleReport:
    """Structured training summary (portfolio-friendly)."""

    cv_rmse_per_model: dict[str, float]
    best_model_name: str
    best_cv_rmse: float
    ensemble_cv_rmse: float
    improvement_vs_best_pct: float
    weights: dict[str, float]
    tau: float
    oof_predictions: pd.DataFrame = field(repr=False)
    gated_oof_predictions: np.ndarray = field(repr=False)
    disagreement_oof: np.ndarray = field(repr=False)
    tau_search_scores: dict[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cv_rmse_per_model": dict(self.cv_rmse_per_model),
            "best_model_name": self.best_model_name,
            "best_cv_rmse": self.best_cv_rmse,
            "ensemble_cv_rmse": self.ensemble_cv_rmse,
            "improvement_vs_best_pct": self.improvement_vs_best_pct,
            "weights": dict(self.weights),
            "tau": self.tau,
            "tau_search_scores": dict(self.tau_search_scores) if self.tau_search_scores else None,
        }


class RMSEGatedDynamicEnsemble:
    """
    RMSE-weighted mixture with disagreement gating (mixture-of-experts style).

    * 10-fold CV (shuffle=True) produces OOF predictions and per-model RMSE.
    * Weights ∝ 1 / RMSE (normalized).
    * Per-sample disagreement = std across model predictions.
    * If disagreement < τ: weighted average; else prediction from best single model.
    """

    def __init__(
        self,
        tau: float = 0.15,
        *,
        n_folds: int = N_CV_FOLDS,
        random_state: int = RANDOM_STATE,
        include_random_forest: bool = True,
        include_xgboost: bool = True,
        tune_tau: bool = False,
        tau_grid: np.ndarray | list[float] | None = None,
    ) -> None:
        self.tau = tau
        self.n_folds = n_folds
        self.random_state = random_state
        self.include_random_forest = include_random_forest
        self.include_xgboost = include_xgboost
        self.tune_tau = tune_tau
        self.tau_grid = tau_grid

        self.estimator_templates_: dict[str, BaseEstimator] | None = None
        self.fitted_models_: dict[str, BaseEstimator] | None = None
        self.cv_rmse_: dict[str, float] | None = None
        self.weights_: dict[str, float] | None = None
        self.best_model_name_: str | None = None
        self.tau_: float | None = None
        self.oof_predictions_: pd.DataFrame | None = None
        self.gated_oof_: np.ndarray | None = None
        self.disagreement_oof_: np.ndarray | None = None
        self.report_: GatedEnsembleReport | None = None
        self.tau_search_scores_: dict[float, float] | None = None

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> RMSEGatedDynamicEnsemble:
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        y_ser = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=float).ravel())

        self.estimator_templates_ = build_base_estimators(
            include_random_forest=self.include_random_forest,
            include_xgboost=self.include_xgboost,
        )
        if not self.estimator_templates_:
            raise RuntimeError("No estimators available (check optional deps)")

        self.oof_predictions_, self.cv_rmse_ = cross_validate_all_models(
            self.estimator_templates_,
            X_df,
            y_ser,
            n_folds=self.n_folds,
            random_state=self.random_state,
        )

        self.best_model_name_ = min(self.cv_rmse_, key=self.cv_rmse_.get)  # type: ignore[arg-type]
        best_cv = self.cv_rmse_[self.best_model_name_]

        self.disagreement_oof_ = prediction_disagreement(self.oof_predictions_)

        tau_use = self.tau
        tau_scores = None
        if self.tune_tau:
            grid = self.tau_grid
            if grid is None:
                grid = default_tau_grid(self.disagreement_oof_)
            tau_use, tau_scores = tune_tau_grid(
                self.oof_predictions_,
                self.cv_rmse_,
                y_ser,
                grid,
            )
            self.tau_search_scores_ = tau_scores

        self.tau_ = float(tau_use)

        self.gated_oof_, self.weights_, self.best_model_name_, ens_rmse = compute_ensemble_oof(
            self.oof_predictions_,
            self.cv_rmse_,
            y_ser,
            self.tau_,
        )

        improvement_pct = (best_cv - ens_rmse) / best_cv * 100.0 if best_cv > 0 else 0.0

        self.report_ = GatedEnsembleReport(
            cv_rmse_per_model=dict(self.cv_rmse_),
            best_model_name=self.best_model_name_,
            best_cv_rmse=best_cv,
            ensemble_cv_rmse=ens_rmse,
            improvement_vs_best_pct=improvement_pct,
            weights=dict(self.weights_),
            tau=self.tau_,
            oof_predictions=self.oof_predictions_.copy(),
            gated_oof_predictions=self.gated_oof_.copy(),
            disagreement_oof=self.disagreement_oof_.copy(),
            tau_search_scores=dict(tau_scores) if tau_scores is not None else None,
        )

        # Refit all models on full training data for inference
        self.fitted_models_ = {}
        for name, tmpl in self.estimator_templates_.items():
            self.fitted_models_[name] = train_model(tmpl, X_df, y_ser)

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.fitted_models_ is None or self.weights_ is None or self.best_model_name_ is None or self.tau_ is None:
            raise RuntimeError("Call fit before predict")
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        pred_df = compute_test_predictions_matrix(self.fitted_models_, X_df)
        return compute_gated_predictions(pred_df, self.weights_, self.best_model_name_, self.tau_)

    def get_report(self) -> GatedEnsembleReport:
        if self.report_ is None:
            raise RuntimeError("Call fit first")
        return self.report_
