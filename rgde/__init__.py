"""
RMSE-Gated Dynamic Ensemble (RGDE).

Mixture-of-experts style ensemble: inverse-RMSE weights + disagreement gating.
"""

from rgde.config import N_CV_FOLDS, RANDOM_STATE
from rgde.cv import cross_validate_all_models, cross_validate_model
from rgde.ensemble import (
    compute_ensemble,
    compute_gated_predictions,
    compute_ensemble_oof,
    prediction_disagreement,
    rmse_dict_to_weights,
)
from rgde.estimators import build_base_estimators
from rgde.evaluation import evaluate_rmse
from rgde.pipeline import GatedEnsembleReport, RMSEGatedDynamicEnsemble
from rgde.training import train_model
from rgde.tuning import default_tau_grid, tune_tau_grid

__all__ = [
    "N_CV_FOLDS",
    "RANDOM_STATE",
    "GatedEnsembleReport",
    "RMSEGatedDynamicEnsemble",
    "build_base_estimators",
    "cross_validate_all_models",
    "cross_validate_model",
    "train_model",
    "evaluate_rmse",
    "rmse_dict_to_weights",
    "prediction_disagreement",
    "compute_ensemble",
    "compute_gated_predictions",
    "compute_ensemble_oof",
    "tune_tau_grid",
    "default_tau_grid",
]

__version__ = "1.0.0"
