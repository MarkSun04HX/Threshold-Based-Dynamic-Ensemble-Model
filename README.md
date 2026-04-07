# Threshold-Based Dynamic Ensemble & RMSE-Gated Ensemble

Production-quality **Python** stack for regression ensembles, plus legacy **TBDE** (top‑k / threshold coalitions) and **R** scripts.

---

## RMSE-Gated Dynamic Ensemble (`rgde/`) — recommended

A **stacking-inspired** system with **inverse-RMSE weights** and **disagreement gating** (mixture-of-experts style).

### Behavior

1. **10-fold CV** (`KFold(shuffle=True, random_state=42)`): for each base learner, **out-of-fold predictions** and **CV RMSE** (no leakage between train/val folds).
2. **Weights:** \(w_i \propto 1/\mathrm{RMSE}_i\), normalized to sum to 1.
3. **Disagreement** per row: standard deviation of predictions across models (on OOF or test).
4. **Gating** with threshold **τ** (tunable):
   - if **disagreement < τ** → **weighted average** of all models;
   - else → prediction from the **best** model (lowest CV RMSE).

### Base learners (sklearn `Pipeline` where appropriate)

| Model | Implementation |
|--------|------------------|
| Linear regression | `StandardScaler` + `LinearRegression` |
| Ridge | `StandardScaler` + `Ridge` |
| Elastic Net | `StandardScaler` + `ElasticNet` |
| Decision tree | `DecisionTreeRegressor` |
| Random forest | `RandomForestRegressor` (optional) |
| XGBoost | `XGBRegressor` in `Pipeline` if `pip install xgboost` |

### Modular API

| Module | Role |
|--------|------|
| `rgde.training.train_model` | Clone + fit on full data |
| `rgde.cv.cross_validate_model` | OOF preds + CV RMSE for one estimator |
| `rgde.cv.cross_validate_all_models` | OOF matrix + RMSE dict |
| `rgde.ensemble` | `rmse_dict_to_weights`, `prediction_disagreement`, `compute_gated_predictions`, `compute_ensemble` / `compute_ensemble_oof` |
| `rgde.evaluation.evaluate_rmse` | RMSE |
| `rgde.pipeline.RMSEGatedDynamicEnsemble` | End-to-end `fit` / `predict` |
| `rgde.tuning.tune_tau_grid` | Optional **τ** grid search on OOF |
| `rgde.plots` | Optional RMSE bar chart & disagreement histogram (`matplotlib`) |

### `fit` output (`GatedEnsembleReport`)

- CV RMSE per model  
- Best model name and best CV RMSE  
- Gated **ensemble CV RMSE** (on OOF)  
- **Improvement %** vs best single model  
- Normalized weights, τ, OOF prediction matrix, gated OOF vector, disagreement  

### Install

```bash
pip install -e .
# optional: pip install xgboost matplotlib
```

### Quick start

```python
import pandas as pd
from rgde import RMSEGatedDynamicEnsemble

df = pd.read_csv("data/train.csv", sep=";", check_names=False)
y = df["quality"]
X = df.drop(columns=["quality"])

model = RMSEGatedDynamicEnsemble(tau=0.15, tune_tau=False)
model.fit(X, y)
report = model.get_report()
print(report.to_dict())

y_test_hat = model.predict(X_test)  # same feature columns as X
```

### CLI & plots

```bash
PYTHONPATH=. python scripts/fit_rgde.py
PYTHONPATH=. python scripts/fit_rgde.py --tune-tau --plot   # needs matplotlib
```

### Tests

```bash
PYTHONPATH=. python -m unittest tests.test_rgde -v
```

### Repository layout (Python)

```
rgde/
  __init__.py
  config.py          # RANDOM_STATE=42, N_CV_FOLDS=10
  estimators.py
  training.py
  cv.py
  ensemble.py
  evaluation.py
  pipeline.py
  tuning.py
  plots.py
scripts/fit_rgde.py
tests/test_rgde.py
```

**Note:** Tuning τ on the same OOF matrix used for weights is convenient but can be mildly optimistic; for publications use a hold-out or nested CV.

---

## Legacy: TBDE (`tbde/`)

Top‑**k** or **threshold** coalitions with **stub** models (NumPy OLS + constants). See earlier sections in git history or `tbde/coalition.py`.

```bash
PYTHONPATH=. python examples/basic_usage.py
PYTHONPATH=. python scripts/cv_tbde.py
python -m unittest tests.test_tbde -v
```

---

## R (optional)

```bash
Rscript examples/basic_usage.R
Rscript tests/test_ensemble.R
```

---

## Contributing

- `PYTHONPATH=. python -m unittest tests.test_rgde -v`
- `PYTHONPATH=. python -m unittest tests.test_tbde -v` (legacy TBDE)

## License

MIT License © 2026 Mark Sun — see `LICENSE`.
