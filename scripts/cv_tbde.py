#!/usr/bin/env python3
"""Outer k-fold CV for TBDE (unweighted mean of top-k coalition)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tbde.coalition import build_coalition, predict_tbde_ensemble


def main() -> None:
    p = argparse.ArgumentParser(description="TBDE outer cross-validation")
    p.add_argument("--file", default=str(_REPO / "data" / "train.csv"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--target", default="quality")
    p.add_argument("--selection", choices=("top_k", "threshold"), default="top_k")
    p.add_argument("--top-k", type=int, default=3, dest="top_k")
    args = p.parse_args()

    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"Data file not found: {path}")

    data = pd.read_csv(path, sep=";", check_names=False)
    if args.target not in data.columns:
        raise SystemExit(f"Column '{args.target}' not found")

    n = len(data)
    if n < args.folds:
        raise SystemExit(f"Need n >= folds ({n} < {args.folds})")

    rng = np.random.default_rng(args.seed)
    # Like R: sample(rep(1:k, length.out=n)) — cycle then shuffle
    fold_id = np.tile(np.arange(1, args.folds + 1), int(np.ceil(n / args.folds)))[:n]
    rng.shuffle(fold_id)

    fold_rmse = np.zeros(args.folds)
    fold_mae = np.zeros(args.folds)
    fold_exact = np.zeros(args.folds)
    fold_w1 = np.zeros(args.folds)
    fold_r2 = np.zeros(args.folds)

    for k in range(1, args.folds + 1):
        test_mask = fold_id == k
        train_df = data.loc[~test_mask].reset_index(drop=True)
        test_df = data.loc[test_mask].reset_index(drop=True)

        res = build_coalition(
            train_df,
            threshold=args.threshold,
            k_folds=args.inner_folds,
            seed=args.seed + k,
            target=args.target,
            verbose=False,
            selection=args.selection,
            top_k=args.top_k,
        )
        y_hat = predict_tbde_ensemble(train_df, test_df, res.models, args.target)
        y = test_df[args.target].values

        fold_rmse[k - 1] = np.sqrt(np.mean((y - y_hat) ** 2))
        fold_mae[k - 1] = np.mean(np.abs(y - y_hat))
        fold_exact[k - 1] = np.mean(np.round(y_hat) == y)
        fold_w1[k - 1] = np.mean(np.abs(np.round(y_hat) - y) <= 1)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        fold_r2[k - 1] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print("\n=== TBDE outer cross-validation (Python) ===")
    print(
        f"Data: {path} | n = {n} | outer folds = {args.folds} | inner folds = {args.inner_folds} | selection = {args.selection}",
        end="",
    )
    if args.selection == "top_k":
        print(f" | top_k = {args.top_k}")
    else:
        print(f" | threshold = {args.threshold} | top_k (fallback) = {args.top_k}")
    print("Ensemble: unweighted mean of coalition predictions.\n")
    print("Note: exact match is strict; 'within-1' is common for wine quality.\n")
    print(f"Mean CV accuracy (exact match, rounded pred): {fold_exact.mean():.4f}")
    print(f"Mean CV accuracy (within 1 point):        {fold_w1.mean():.4f}")
    print(f"SD across folds (within-1):                {fold_w1.std(ddof=1):.4f}")
    print(f"Mean CV RMSE:                              {fold_rmse.mean():.4f}")
    print(f"Mean CV MAE:                               {fold_mae.mean():.4f}")
    print(f"Mean CV R-squared:                       {np.nanmean(fold_r2):.4f}")
    print("\nPer-fold within-1 accuracy:", ", ".join(f"{x:.4f}" for x in fold_w1))


if __name__ == "__main__":
    main()
