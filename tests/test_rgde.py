"""Unit tests for RMSE-gated dynamic ensemble."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rgde.cv import cross_validate_model  # noqa: E402
from rgde.estimators import build_base_estimators  # noqa: E402
from rgde.evaluation import evaluate_rmse  # noqa: E402
from rgde.pipeline import RMSEGatedDynamicEnsemble  # noqa: E402
from rgde.training import train_model  # noqa: E402


def _make_regression(n: int = 200, p: int = 5, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    coef = rng.standard_normal(p)
    y = X @ coef + rng.normal(0, 0.5, n)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    return df, pd.Series(y, name="y")


class TestRGDE(unittest.TestCase):
    def test_cross_validate_model_no_nan(self) -> None:
        X, y = _make_regression(n=80, p=3)
        est = build_base_estimators(include_random_forest=False, include_xgboost=False)["ridge"]
        oof, rmse = cross_validate_model(est, X, y, n_folds=5, random_state=42)
        self.assertFalse(np.any(np.isnan(oof)))
        self.assertGreater(rmse, 0)

    def test_train_model_predict(self) -> None:
        X, y = _make_regression(n=50, p=2)
        tmpl = build_base_estimators(include_random_forest=False, include_xgboost=False)["linear_regression"]
        fitted = train_model(tmpl, X, y)
        pred = fitted.predict(X.to_numpy(float))
        self.assertEqual(pred.shape, (50,))

    def test_full_pipeline(self) -> None:
        X, y = _make_regression(n=120, p=4)
        m = RMSEGatedDynamicEnsemble(
            tau=0.2,
            tune_tau=False,
            include_random_forest=True,
            include_xgboost=False,
            n_folds=5,
        )
        m.fit(X, y)
        rep = m.get_report()
        self.assertIn(rep.best_model_name, rep.cv_rmse_per_model)
        self.assertGreater(rep.best_cv_rmse, 0)
        self.assertGreater(rep.ensemble_cv_rmse, 0)
        pred = m.predict(X)
        self.assertEqual(pred.shape, (120,))

    def test_evaluate_rmse(self) -> None:
        e = evaluate_rmse(np.array([0.0, 2.0]), np.array([0.0, 0.0]))
        self.assertAlmostEqual(e, np.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
