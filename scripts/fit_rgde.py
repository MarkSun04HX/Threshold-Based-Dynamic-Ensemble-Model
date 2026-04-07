#!/usr/bin/env python3
"""
Train RMSE-gated dynamic ensemble on data/train.csv and print the report.

Usage (repo root):
  PYTHONPATH=. python scripts/fit_rgde.py
  PYTHONPATH=. python scripts/fit_rgde.py --tune-tau --no-xgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rgde.pipeline import RMSEGatedDynamicEnsemble  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, default=ROOT / "data" / "train.csv")
    p.add_argument("--target", default="quality")
    p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--tune-tau", action="store_true")
    p.add_argument("--no-rf", action="store_true")
    p.add_argument("--no-xgb", action="store_true")
    p.add_argument("--plot", action="store_true", help="Save plots under artifacts/")
    args = p.parse_args()

    df = pd.read_csv(args.file, sep=";", check_names=False)
    feature_cols = [c for c in df.columns if c != args.target]
    X = df[feature_cols]
    y = df[args.target]

    model = RMSEGatedDynamicEnsemble(
        tau=args.tau,
        tune_tau=args.tune_tau,
        include_random_forest=not args.no_rf,
        include_xgboost=not args.no_xgb,
    )
    model.fit(X, y)
    rep = model.get_report()

    print("\n=== RMSE-Gated Dynamic Ensemble ===\n")
    print("CV RMSE (out-of-fold) per model:")
    for name, rmse in sorted(rep.cv_rmse_per_model.items(), key=lambda x: x[1]):
        print(f"  {name:20s} {rmse:.4f}")
    print(f"\nBest single model: {rep.best_model_name} (RMSE = {rep.best_cv_rmse:.4f})")
    print(f"Gated ensemble CV RMSE: {rep.ensemble_cv_rmse:.4f}")
    print(f"Improvement vs best single: {rep.improvement_vs_best_pct:.2f}%")
    print(f"\nτ (disagreement threshold): {rep.tau:.6f}")
    print("Weights (normalized 1/RMSE):")
    for name, w in sorted(rep.weights.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {w:.4f}")

    if args.plot:
        try:
            from rgde.plots import plot_disagreement_distribution, plot_rmse_comparison
        except ImportError:
            print("matplotlib not installed; skip plots", file=sys.stderr)
            return
        art = ROOT / "artifacts"
        art.mkdir(exist_ok=True)
        plot_rmse_comparison(
            rep.cv_rmse_per_model,
            ensemble_rmse=rep.ensemble_cv_rmse,
            save_path=art / "rgde_rmse_bars.png",
        )
        plot_disagreement_distribution(
            rep.disagreement_oof,
            tau=rep.tau,
            save_path=art / "rgde_disagreement.png",
        )
        print(f"\nSaved plots to {art}/")


if __name__ == "__main__":
    main()
