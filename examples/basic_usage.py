"""TBDE usage examples (Python). Run from repo root: python examples/basic_usage.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tbde.coalition import build_coalition, evaluate_coalition  # noqa: E402


def main() -> None:
    print("TBDE examples (Python)\n")

    simple = pd.DataFrame(
        {
            "quality": [10, 15, 12, 40, 11, 14, 13, 20, 18, 22, 25, 19],
            "feature_a": [1, 2, 1.5, 8, 1.2, 2.1, 1.8, 3, 2.5, 3.2, 4, 3.1],
            "feature_b": [100, 105, 102, 200, 101, 106, 103, 150, 140, 155, 180, 145],
        }
    )
    res = build_coalition(simple, top_k=3)
    print("Example 1 coalition:", res.models)

    rng = np.random.default_rng(123)
    realistic = pd.DataFrame(
        {
            "quality": rng.normal(500, 100, 150),
            "feature_1": rng.normal(50, 10, 150),
            "feature_2": rng.normal(100, 20, 150),
        }
    )
    realistic["quality"] = realistic["quality"] + 0.3 * realistic["feature_1"] + 0.1 * realistic["feature_2"]
    res2 = build_coalition(realistic, k_folds=5, top_k=3, verbose=False)
    print("Example 2 coalition size:", len(res2.models))

    train_path = ROOT / "data" / "train.csv"
    if train_path.is_file():
        train = pd.read_csv(train_path, sep=";", check_names=False)
        res3 = build_coalition(train, top_k=3, verbose=False)
        print("Wine data coalition:", res3.models)


if __name__ == "__main__":
    main()
