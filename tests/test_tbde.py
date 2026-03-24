"""Tests for TBDE Python implementation (stdlib unittest)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tbde.coalition import build_coalition, evaluate_coalition  # noqa: E402


def _synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "quality": rng.normal(50, 15, n),
            "feature1": rng.standard_normal(n),
            "feature2": rng.standard_normal(n),
            "feature3": rng.random(n),
        }
    )


class TestTBDE(unittest.TestCase):
    def test_build_top_k(self) -> None:
        res = build_coalition(_synthetic(), top_k=3, verbose=False)
        self.assertEqual(len(res.models), 3)

    def test_build_threshold(self) -> None:
        res = build_coalition(_synthetic(), selection="threshold", threshold=8.0, verbose=False)
        self.assertGreaterEqual(len(res.models), 1)

    def test_missing_target(self) -> None:
        bad = pd.DataFrame({"x": range(10)})
        with self.assertRaises(ValueError):
            build_coalition(bad, verbose=False)

    def test_evaluate(self) -> None:
        d = _synthetic()
        rng = np.random.default_rng(99)
        idx = rng.choice(len(d), size=int(0.7 * len(d)), replace=False)
        train = d.iloc[idx].reset_index(drop=True)
        test = d.drop(idx).reset_index(drop=True)
        res = build_coalition(train, selection="threshold", threshold=9.0, verbose=False)
        mae = evaluate_coalition(res.models, train, test)
        self.assertGreaterEqual(mae, 0)
        self.assertTrue(np.isfinite(mae))


if __name__ == "__main__":
    unittest.main()
