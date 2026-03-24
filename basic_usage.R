#' Basic Usage Examples for TBDE Model
#'
#' This file demonstrates common workflows for using the
#' Threshold-Based Dynamic Ensemble model

# Load required libraries
library(dplyr)

# Source the model functions
source("R/build_coalition.R")

cat("===========================================\n")
cat("Threshold-Based Dynamic Ensemble Examples\n")
cat("===========================================\n\n")

# ------- EXAMPLE 1: Using Built-in Data -------
cat("EXAMPLE 1: Simple Dataset\n")
cat("---------\n\n")

# Create a minimal dataset
simple_data <- data.frame(
  cost = c(10, 15, 12, 40, 11, 14, 13, 20, 18, 22, 25, 19),
  feature_a = c(1, 2, 1.5, 8, 1.2, 2.1, 1.8, 3, 2.5, 3.2, 4, 3.1),
  feature_b = c(100, 105, 102, 200, 101, 106, 103, 150, 140, 155, 180, 145)
)

cat("Dataset shape:", nrow(simple_data), "rows,", ncol(simple_data), "columns\n\n")

# Build coalition with default parameters
coalition_simple <- build_coalition(simple_data, threshold = 10.0)

cat("Models in coalition:", paste(coalition_simple, collapse = ", "), "\n\n\n")

# ------- EXAMPLE 2: Larger Realistic Dataset -------
cat("EXAMPLE 2: Larger Realistic Dataset\n")
cat("---------\n\n")

# Generate a more complex dataset
set.seed(123)
realistic_data <- data.frame(
  cost = rnorm(150, mean = 500, sd = 100),
  feature_1 = rnorm(150, mean = 50, sd = 10),
  feature_2 = rnorm(150, mean = 100, sd = 20),
  feature_3 = rbinom(150, size = 1, prob = 0.6),
  feature_4 = runif(150, min = 0, max = 1000),
  feature_5 = rnorm(150, mean = 25, sd = 5)
)

# Add some correlation
realistic_data$cost <- realistic_data$cost + 0.3 * realistic_data$feature_1 +
                       0.1 * realistic_data$feature_2

cat("Dataset shape:", nrow(realistic_data), "rows,", ncol(realistic_data), "columns\n")
cat("Cost range: [", min(realistic_data$cost), ", ", max(realistic_data$cost), "]\n\n")

# Build coalition with custom threshold
coalition_realistic <- build_coalition(realistic_data, threshold = 25.0, k_folds = 5)

cat("Selected coalition size:", length(coalition_realistic), "\n\n\n")

# ------- EXAMPLE 3: Comparing Different Thresholds -------
cat("EXAMPLE 3: Threshold Sensitivity Analysis\n")
cat("---------\n\n")

thresholds <- c(5, 10, 15, 20, 25)

cat("Threshold | Coalition Size | Models\n")
cat("-----------|----------------|--------\n")

for (thresh in thresholds) {
  coalition <- build_coalition(realistic_data, threshold = thresh)
  cat(sprintf("%8.1f  |      %2d        | %s\n",
              thresh, length(coalition), paste(coalition, collapse = ", ")))
}

cat("\n\n")

# ------- EXAMPLE 4: Train-Test Split -------
cat("EXAMPLE 4: Train-Test Evaluation\n")
cat("---------\n\n")

set.seed(456)

# Split data 80-20
split_idx <- sample(1:nrow(realistic_data), size = 0.8 * nrow(realistic_data))
train_data <- realistic_data[split_idx, ]
test_data <- realistic_data[-split_idx, ]

cat("Train set:", nrow(train_data), "samples\n")
cat("Test set:", nrow(test_data), "samples\n\n")

# Build coalition on training data only
coalition_final <- build_coalition(train_data, threshold = 15.0)

# Evaluate on test set
mae <- evaluate_coalition(coalition_final, test_data)

cat("Test set MAE:", round(mae, 4), "\n\n\n")

# ------- EXAMPLE 5: Custom k-Fold Configuration -------
cat("EXAMPLE 5: Different Cross-Validation Folds\n")
cat("---------\n\n")

cat("Testing different k-fold values:\n\n")

for (k in c(3, 5, 10)) {
  coalition <- build_coalition(realistic_data, threshold = 15.0, k_folds = k)
  cat("k =", k, "-> Coalition size:", length(coalition), "\n")
}

cat("\n")
cat("===========================================\n")
cat("Examples completed!\n")
cat("===========================================\n")
