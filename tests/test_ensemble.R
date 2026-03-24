#' Test Suite for Threshold-Based Dynamic Ensemble
#'
#' This script tests the core TBDE functionality with realistic training data

# Source the main model functions
source("R/build_coalition.R")
source("data-raw/generate_sample_data.R")

cat("\n========================================\n")
cat("TBDE Model Testing\n")
cat("========================================\n\n")

# Test 1: Basic Coalition Building
cat("Test 1: Building coalition with threshold = 8.0\n")
cat("-------------------------------------------\n")

coalition_1 <- build_coalition(synthetic_data, selection = "threshold", threshold = 8.0)
cat("Selected models:", paste(coalition_1, collapse = ", "), "\n\n")

# Test 2: Strict Threshold (forces Top-3 selection)
cat("Test 2: Strict threshold forcing Top-3 selection\n")
cat("-------------------------------------------\n")

coalition_2 <- build_coalition(synthetic_data, selection = "threshold", threshold = 1.0)
cat("Selected models:", paste(coalition_2, collapse = ", "), "\n\n")

# Test 3: Lenient Threshold
cat("Test 3: Lenient threshold allowing more models\n")
cat("-------------------------------------------\n")

coalition_3 <- build_coalition(synthetic_data, selection = "threshold", threshold = 15.0)
cat("Selected models:", paste(coalition_3, collapse = ", "), "\n\n")

# Test 4: Different k-fold configurations
cat("Test 4: Testing with different k-fold values\n")
cat("-------------------------------------------\n")

coalition_5fold <- build_coalition(synthetic_data, selection = "threshold", threshold = 8.0, k_folds = 5)
cat("5-fold selected:", paste(coalition_5fold, collapse = ", "), "\n\n")

# Test 5: Evaluate coalition on test split
cat("Test 5: Evaluating coalition on test data\n")
cat("-------------------------------------------\n")

# Split data into train/test
set.seed(99)
train_idx <- sample(1:nrow(synthetic_data), size = floor(0.7 * nrow(synthetic_data)))
train_set <- synthetic_data[train_idx, ]
test_set <- synthetic_data[-train_idx, ]

# Build coalition on training data
final_coalition <- build_coalition(train_set, selection = "threshold", threshold = 9.0)

# Evaluate on test data
mae <- evaluate_coalition(final_coalition, train_set, test_set)
cat("Mean Absolute Error on test set:", round(mae, 4), "\n\n")

# Test 6: Input validation
cat("Test 6: Input validation\n")
cat("-------------------------------------------\n")

tryCatch({
  bad_data <- data.frame(x = 1:10)  # Missing 'quality' column
  build_coalition(bad_data)
}, error = function(e) {
  cat("Caught expected error:", conditionMessage(e), "\n\n")
})

cat("========================================\n")
cat("All tests completed successfully!\n")
cat("========================================\n")
