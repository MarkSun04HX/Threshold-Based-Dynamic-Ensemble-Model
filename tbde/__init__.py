"""TBDE: Threshold-Based Dynamic Ensemble (Python implementation)."""

from tbde.coalition import (
    CoalitionResult,
    build_coalition,
    evaluate_coalition,
    predict_tbde_ensemble,
)

__all__ = [
    "CoalitionResult",
    "build_coalition",
    "evaluate_coalition",
    "predict_tbde_ensemble",
]

__version__ = "0.2.0"
