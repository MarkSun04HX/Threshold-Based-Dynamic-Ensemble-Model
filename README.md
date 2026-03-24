# Threshold-Based Dynamic Ensemble (TBDE) Model

The Threshold-Based Dynamic Ensemble (TBDE) is an AutoML architecture that "auditions" multiple models (XGBoost, NN, etc.) via Cross-Validation. Only models beating a specific RMSE threshold join the voting coalition. If none pass, it picks the top 3. This ensures a robust, self-optimizing consensus that filters out weak predictors.

## 📋 Repository Structure

```
├── R/
│   └── build_coalition.R          # Core ensemble logic
├── data/
│   └── train.csv                  # Sample tabular data (see note below)
├── data-raw/
│   └── generate_sample_data.R     # Defines synthetic_data for tests / experiments
├── examples/
│   └── basic_usage.R              # Usage examples
├── tests/
│   └── test_ensemble.R            # Testing suite
├── model.R                        # Legacy implementation
├── README.md                      # This file
└── LICENSE                        # MIT License
```

## Prerequisites

- **R** (3.5+ recommended). No extra R packages are required for `build_coalition()` itself. The example script loads **dplyr** (install with `install.packages("dplyr")` if needed).

Run scripts from the **repository root** so paths like `R/build_coalition.R` resolve correctly.

## 🚀 Quick Start

### Run the ensemble in R

```r
# From the project directory, or setwd() to it first
source("R/build_coalition.R")

# Data must include a numeric target column named 'cost'
data <- data.frame(
  cost = rnorm(100, mean = 50, sd = 15),
  feature1 = rnorm(100),
  feature2 = rnorm(100)
)

# Returns a character vector of selected model names
coalition <- build_coalition(data, threshold = 8.0)
```

### Command-line examples

From the project root:

```bash
# Full example workflows (threshold sweeps, train/test, etc.)
Rscript examples/basic_usage.R

# Test suite
Rscript tests/test_ensemble.R
```

### Key Parameters

- **data**: Data frame with required `cost` column (target) and any feature columns
- **threshold**: RMSE threshold for model inclusion (default: 5.0)
- **k_folds**: Number of cross-validation folds (default: 3)
- **seed**: Random seed for reproducibility (default: 123)

### Data files

- **`data/train.csv`**: Wine-quality style features in CSV form. The API expects the target under the name **`cost`**, not `quality`. Rename or derive a column, e.g. `cost <- quality`, before calling `build_coalition()`.

## 📊 Models Evaluated

The coalition logic is written for ten named model slots (e.g. XGBoost, CatBoost, LightGBM). **In the current R implementation, each slot uses a small stub trainer/predictor** so the CV and threshold selection pipeline can run without installing heavy ML stacks. The list below describes the *intended* model types for a full production setup:

1. **XGBoost** - Gradient boosting framework
2. **CatBoost** - Categorical boosting
3. **NeuralNet** - Neural network regressor
4. **LinearReg** - Linear regression baseline
5. **KNN** - K-nearest neighbors
6. **RandomForest** - Random forest ensemble
7. **ElasticNet** - Regularized linear regression
8. **SVM** - Support vector machine
9. **LightGBM** - Light gradient boosting
10. **DecisionTree** - Single decision tree

## 🎯 How It Works

### Phase 1: Model Audition
- Each model is trained via k-fold cross-validation
- RMSE error is computed for each fold
- Final RMSE for each model is the average across folds

### Phase 2: Coalition Selection
- Models with RMSE < threshold are selected
- If no models meet threshold, top 3 performers are chosen
- Ensures robust voting ensemble

### Phase 3: Prediction
- Selected models vote on predictions
- Coalition output is more stable than individual models

## 📈 Example Workflow

See `examples/basic_usage.R` for complete examples including:

- Simple dataset usage
- Larger realistic datasets
- Threshold sensitivity analysis
- Train-test split evaluation
- Custom cross-validation configurations

## 🧪 Testing

Run the full test suite from the repository root:

```bash
Rscript tests/test_ensemble.R
```

Tests cover:
- Basic coalition building
- Strict vs. lenient thresholds
- Different k-fold configurations
- Coalition evaluation
- Input validation

## 📊 Generate Sample Data

Create in-memory synthetic data used by the test script (defines `synthetic_data` when sourced):

```bash
Rscript -e 'source("data-raw/generate_sample_data.R"); str(synthetic_data)'
```

Or in R: `source("data-raw/generate_sample_data.R")`.

## 🔍 API Reference

### `build_coalition(data, threshold = 5.0, k_folds = 3, seed = 123)`

Builds a threshold-based ensemble coalition.

**Returns:** Character vector of selected model names

### `evaluate_coalition(coalition, test_data)`

Evaluates coalition performance on test data.

**Returns:** Numeric mean absolute error

## 💡 Best Practices

1. **Choose appropriate threshold**: 
   - Lower thresholds (5-10) for stricter selection
   - Higher thresholds (15+) for more inclusive coalitions

2. **Use cross-validation**: 
   - Default 3-fold is fast; 5-fold recommended for robustness
   - 10-fold for very small datasets

3. **Scale your data**: 
   - Consider normalizing features for better model performance

4. **Validate results**: 
   - Always test on held-out test set (not used during coalition building)

## 📝 Legacy Code

`model.R` contains the original implementation. The refactored version in `R/build_coalition.R` provides better structure, documentation, and extensibility.

## 📄 License

MIT License © 2026 Mark Sun

See LICENSE file for details.

## 🤝 Contributing

Issues and pull requests welcome! Please ensure:
- Code follows R style guide
- Tests pass: `Rscript tests/test_ensemble.R`
- Documentation is updated
