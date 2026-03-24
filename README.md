# Threshold-Based Dynamic Ensemble (TBDE) Model

The Threshold-Based Dynamic Ensemble (TBDE) “auditions” multiple named models via inner cross-validation. **By default** the **top 3** models with the lowest inner-CV RMSE form the coalition, and predictions are the **unweighted mean** of their outputs. You can switch to a **threshold** rule (`selection = "threshold"`).

**Recommended implementation: Python** (`tbde/`) — only **NumPy** and **pandas**; easy to extend and test. **R** sources remain available under `R/` for compatibility.

## Repository structure

```
├── tbde/                          # Python package (primary)
│   ├── __init__.py
│   └── coalition.py
├── R/
│   └── build_coalition.R          # R port of the same logic
├── data/
│   └── train.csv
├── data-raw/
│   └── generate_sample_data.R
├── examples/
│   ├── basic_usage.py             # Python examples
│   └── basic_usage.R
├── tests/
│   ├── test_tbde.py               # Python (unittest)
│   └── test_ensemble.R
├── scripts/
│   ├── cv_tbde.py                 # Outer CV (Python)
│   └── cv_tbde.R
├── pyproject.toml
├── requirements.txt
├── model.R                        # Older R script
└── LICENSE
```

## Prerequisites (Python)

- **Python 3.10+**
- `pip install -e .` or `pip install -r requirements.txt` (numpy, pandas)

Run commands from the **repository root** with `PYTHONPATH=.` or after `pip install -e .`.

## Quick start (Python)

```python
from pathlib import Path
import pandas as pd
from tbde.coalition import build_coalition, predict_tbde_ensemble

train = pd.read_csv("data/train.csv", sep=";", check_names=False)
result = build_coalition(train, top_k=3)   # CoalitionResult
print(result.models)                       # e.g. ['LinearReg', 'XGBoost', ...]

# Predict on a holdout frame (same columns as train, including target)
# y_hat = predict_tbde_ensemble(train, test, result.models, target="quality")
```

```bash
python examples/basic_usage.py
python -m unittest tests.test_tbde -v
python scripts/cv_tbde.py --folds 5 --top-k 3
```

### Python API

- **`build_coalition(data, ...)`** → **`CoalitionResult`** with `.models: list[str]` and optional `.cv_rmse: dict[str, float]` when `return_rmse=True`.
- **`predict_tbde_ensemble(train, test, coalition, target, weights=None)`** → row-wise mean (or weighted sum).
- **`evaluate_coalition(coalition, train_data, test_data, target, weights=None)`** → MAE.

Key parameters: **`selection`**: `"top_k"` | `"threshold"`, **`top_k`**, **`threshold`**, **`k_folds`**, **`target`** (default `"quality"`).

## Quick start (R, optional)

```r
source("R/build_coalition.R")
# build_coalition returns a character vector; use selection / top_k as in Python
coalition <- build_coalition(train, top_k = 3)
```

```bash
Rscript examples/basic_usage.R
Rscript tests/test_ensemble.R
Rscript scripts/cv_tbde.R --folds 5
```

## Models (stub implementations)

Ten named slots are evaluated (XGBoost, CatBoost, …). **Each slot uses a small stub** (means/medians and an **OLS linear model** via NumPy `lstsq` in Python / `lm` in R) so the pipeline runs without heavy ML libraries. Swap in real estimators in `tbde/coalition.py` (or `R/build_coalition.R`) when you move to production.

## How it works

1. **Inner CV:** each model is scored by RMSE across folds.
2. **Selection:** default **top_k** by lowest RMSE; optional **threshold** mode.
3. **Prediction:** **unweighted mean** of coalition members’ predictions per row.

## Cross-validation (outer folds)

- **Python:** `python scripts/cv_tbde.py` — reports exact-match and within-1 “accuracy”, RMSE, MAE, R².
- **R:** `Rscript scripts/cv_tbde.R`

## Sample / synthetic data

```bash
Rscript -e 'source("data-raw/generate_sample_data.R"); str(synthetic_data)'
```

## Best practices

- Prefer **within-1** accuracy for ordinal wine scores; exact match is strict for regression.
- Tune **`top_k`** or **threshold** and outer CV folds for your sample size.
- Validate on held-out data not used to build the coalition.

## License

MIT License © 2026 Mark Sun — see `LICENSE`.

## Contributing

- Python: run `python -m unittest tests.test_tbde -v`.
- R: run `Rscript tests/test_ensemble.R`.
- Match behavior between languages when changing selection or stub logic.
