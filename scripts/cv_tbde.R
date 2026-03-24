#!/usr/bin/env Rscript
# Outer k-fold cross-validation for the TBDE coalition:
# each fold builds the coalition on the training split (with inner CV inside
# build_coalition), then scores the ensemble on the held-out fold.
#
# Default: top_k models by inner-CV RMSE, unweighted mean prediction (see build_coalition).
#
# Usage (from repository root):
#   Rscript scripts/cv_tbde.R
#   Rscript scripts/cv_tbde.R --file data/train.csv --folds 5 --top-k 3
#   Rscript scripts/cv_tbde.R --selection threshold --threshold 1.0

args <- commandArgs(trailingOnly = TRUE)
args_all <- commandArgs(trailingOnly = FALSE)
file_line <- args_all[grepl("^--file=", args_all)]
script_path <- if (length(file_line)) sub("^--file=", "", file_line[1]) else NA_character_
repo_root <- if (!is.na(script_path) && nzchar(script_path)) {
  dirname(dirname(normalizePath(script_path)))
} else {
  getwd()
}

get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i >= length(args)) return(default)
  args[i + 1]
}

data_path <- get_arg("--file", file.path(repo_root, "data", "train.csv"))
k_outer <- as.integer(get_arg("--folds", "5"))
inner_k <- as.integer(get_arg("--inner-folds", "3"))
threshold <- as.numeric(get_arg("--threshold", "1.0"))
seed <- as.integer(get_arg("--seed", "123"))
target <- get_arg("--target", "quality")
sel_raw <- get_arg("--selection", "top_k")
selection <- tryCatch(
  match.arg(sel_raw, c("top_k", "threshold")),
  error = function(e) stop("Invalid --selection (use top_k or threshold)", call. = FALSE)
)
top_k_arg <- as.integer(get_arg("--top-k", "3"))

if (any(is.na(c(k_outer, inner_k, seed, top_k_arg)))) {
  stop("Invalid numeric arguments for --folds, --inner-folds, --seed, or --top-k")
}
if (selection == "threshold" && (is.na(threshold) || threshold <= 0)) {
  stop("When using --selection threshold, pass a positive --threshold", call. = FALSE)
}

source(file.path(repo_root, "R", "build_coalition.R"))

if (!file.exists(data_path)) {
  stop("Data file not found: ", data_path, call. = FALSE)
}

data <- read.csv(data_path, sep = ";", check.names = FALSE, stringsAsFactors = FALSE)
if (!(target %in% names(data))) {
  stop("Column '", target, "' not found in data", call. = FALSE)
}

n <- nrow(data)
if (n < k_outer) {
  stop("Need at least as many rows as outer folds (n = ", n, ", folds = ", k_outer, ")", call. = FALSE)
}

set.seed(seed)
fold_id <- sample(rep(seq_len(k_outer), length.out = n))

fold_rmse <- numeric(k_outer)
fold_mae <- numeric(k_outer)
fold_acc_exact <- numeric(k_outer)
fold_acc_1 <- numeric(k_outer)
fold_r2 <- numeric(k_outer)

for (k in seq_len(k_outer)) {
  test_idx <- which(fold_id == k)
  train_df <- data[-test_idx, , drop = FALSE]
  test_df <- data[test_idx, , drop = FALSE]

  coalition <- build_coalition(
    train_df,
    threshold = threshold,
    k_folds = inner_k,
    seed = seed + k,
    target = target,
    verbose = FALSE,
    selection = selection,
    top_k = top_k_arg
  )

  y_hat <- predict_tbde_ensemble(train_df, test_df, coalition, target)
  y_true <- test_df[[target]]

  fold_rmse[k] <- sqrt(mean((y_true - y_hat)^2))
  fold_mae[k] <- mean(abs(y_true - y_hat))
  fold_acc_exact[k] <- mean(round(y_hat) == y_true)
  fold_acc_1[k] <- mean(abs(round(y_hat) - y_true) <= 1)
  ss_res <- sum((y_true - y_hat)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  fold_r2[k] <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
}

cat("\n=== TBDE outer cross-validation ===\n")
cat("Data:", data_path, "| n =", n, "| outer folds =", k_outer,
    "| inner folds (build_coalition) =", inner_k,
    "| selection =", selection)
if (selection == "top_k") {
  cat(" | top_k =", top_k_arg, "\n")
} else {
  cat(" | threshold =", threshold, "| top_k (fallback) =", top_k_arg, "\n")
}
cat("Ensemble: unweighted mean of coalition predictions.\n\n")

cat("Note: exact match (rounded pred == label) is strict for regression.\n")
cat("      'Within-1' is the usual reported score for wine quality (ordinal 3-9).\n\n")

cat(sprintf("Mean CV accuracy (exact match, rounded pred): %.4f\n", mean(fold_acc_exact)))
cat(sprintf("Mean CV accuracy (within 1 point):        %.4f   <- main wine-quality metric\n", mean(fold_acc_1)))
cat(sprintf("SD across folds (within-1):                %.4f\n", stats::sd(fold_acc_1)))
cat(sprintf("Mean CV RMSE:                              %.4f\n", mean(fold_rmse)))
cat(sprintf("Mean CV MAE:                               %.4f\n", mean(fold_mae)))
cat(sprintf("Mean CV R-squared:                       %.4f\n", mean(fold_r2, na.rm = TRUE)))
cat("\nPer-fold within-1 accuracy:", paste(round(fold_acc_1, 4), collapse = ", "), "\n")
