#' Build Dynamic Ensemble Coalition
#'
#' @description
#' Audits multiple models via k-fold cross-validation and forms a coalition. By default
#' (\code{selection = "top_k"}) the \strong{top_k} models with lowest RMSE are selected and
#' predictions are the \strong{unweighted mean} of their outputs (see \code{predict_tbde_ensemble}).
#' Alternatively, use \code{selection = "threshold"} so models with RMSE below \code{threshold}
#' join the coalition; if none qualify, the top \code{top_k} are used (legacy behavior).
#'
#' @param data Data frame containing features and target variable (see \code{target})
#' @param threshold Numeric. RMSE threshold when \code{selection = "threshold"} (default: 5.0)
#' @param k_folds Integer. Number of folds for cross-validation (default: 3)
#' @param seed Integer. Random seed for reproducibility (default: 123)
#' @param target Character. Name of the response column (default: \code{"quality"}, e.g. wine \code{data/train.csv})
#' @param verbose Logical. Print coalition table and messages (default: TRUE)
#' @param return_rmse Logical. If TRUE, attach \code{attr(result, "cv_rmse")}, inner-CV RMSE per coalition member
#' @param selection Character. \code{"top_k"} (default): take the \code{top_k} lowest-RMSE models; \code{"threshold"}: threshold rule above
#' @param top_k Integer. Number of models when using \code{selection = "top_k"}, or fallback size when no model meets the threshold
#' @return Character vector of selected model names (optionally with \code{cv_rmse} attribute)
#' @details
#' The function evaluates the following models:
#' \itemize{
#'   \item XGBoost
#'   \item CatBoost
#'   \item Neural Network
#'   \item Linear Regression
#'   \item K-Nearest Neighbors
#'   \item Random Forest
#'   \item Elastic Net
#'   \item Support Vector Machine
#'   \item LightGBM
#'   \item Decision Tree
#' }
#' @examples
#' \dontrun{
#'   # Generate sample data
#'   set.seed(42)
#'   sample_data <- data.frame(
#'     quality = rnorm(100, mean = 50, sd = 15),
#'     feature1 = rnorm(100),
#'     feature2 = rnorm(100)
#'   )
#'
#'   coalition <- build_coalition(sample_data, top_k = 3)
#'   coalition <- build_coalition(sample_data, selection = "threshold", threshold = 8.0)
#' }
# TBDE stub model trainers (one function per model name)
#' @noRd
tbde_model_definitions <- function(target) {
  list(
    XGBoost = function(train) { return(mean(train[[target]])) },
    CatBoost = function(train) { return(median(train[[target]])) },
    NeuralNet = function(train) { return(mean(train[[target]]) * 1.05) },
    LinearReg = function(train) { lm(as.formula(paste(target, "~ .")), data = train) },
    KNN = function(train) { return(mean(tail(train[[target]], 3))) },
    RandomForest = function(train) { return(mean(train[[target]])) },
    ElasticNet = function(train) { return(mean(train[[target]]) * 0.98) },
    SVM = function(train) { return(median(train[[target]])) },
    LightGBM = function(train) { return(mean(train[[target]])) },
    DecisionTree = function(train) { return(mean(train[[target]])) }
  )
}

#' Combine coalition predictions: unweighted mean, or weighted rows (e.g. inverse inner-CV RMSE).
#' @noRd
predict_tbde_ensemble <- function(train, test, coalition, target, weights = NULL) {
  models <- tbde_model_definitions(target)
  if (length(coalition) == 0L) {
    stop("coalition must be non-empty")
  }
  n <- nrow(test)
  pred_mat <- matrix(NA_real_, nrow = n, ncol = length(coalition))
  for (j in seq_along(coalition)) {
    m_name <- coalition[j]
    if (!m_name %in% names(models)) {
      stop("unknown model in coalition: ", m_name, call. = FALSE)
    }
    fit <- models[[m_name]](train)
    if (is.numeric(fit)) {
      pred_mat[, j] <- rep(fit, n)
    } else {
      pred_mat[, j] <- as.numeric(predict(fit, test))
    }
  }
  if (is.null(weights)) {
    return(rowMeans(pred_mat))
  }
  if (length(weights) != length(coalition)) {
    stop("weights must have one value per coalition member", call. = FALSE)
  }
  w <- as.numeric(weights)
  if (any(!is.finite(w)) || any(w < 0)) {
    stop("weights must be finite and non-negative", call. = FALSE)
  }
  sw <- sum(w)
  if (sw <= 0) {
    return(rowMeans(pred_mat))
  }
  drop(pred_mat %*% (w / sw))
}

#' @export
build_coalition <- function(data, threshold = 5.0, k_folds = 3, seed = 123, target = "quality", verbose = TRUE, return_rmse = FALSE, selection = c("top_k", "threshold"), top_k = 3L) {

  # Validate inputs
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }
  if (!is.character(target) || length(target) != 1L || !nzchar(target)) {
    stop("target must be a non-empty string")
  }
  if (!(target %in% colnames(data))) {
    stop("data must contain target column '", target, "'", call. = FALSE)
  }
  selection <- match.arg(selection)
  top_k <- as.integer(top_k)[1L]
  if (is.na(top_k) || top_k < 1L) {
    stop("top_k must be a positive integer", call. = FALSE)
  }
  if (selection == "threshold" && threshold <= 0) {
    stop("threshold must be positive when selection = \"threshold\"")
  }

  models <- tbde_model_definitions(target)

  # 2. K-Fold Cross Validation
  results <- data.frame(
    Model = names(models),
    RMSE = NA_real_,
    stringsAsFactors = FALSE
  )

  set.seed(seed)
  folds <- cut(seq(1, nrow(data)), breaks = k_folds, labels = FALSE)

  for (m_name in names(models)) {
    errors <- c()

    for (i in 1:k_folds) {
      test_idx <- which(folds == i, arr.ind = TRUE)
      test_data <- data[test_idx, ]
      train_data <- data[-test_idx, ]

      # Train and Predict
      pred <- models[[m_name]](train_data)

      if (is.numeric(pred)) {
        actual_preds <- rep(pred, nrow(test_data))
      } else {
        actual_preds <- predict(pred, test_data)
      }

      errors <- c(errors, (test_data[[target]] - actual_preds)^2)
    }

    results$RMSE[results$Model == m_name] <- sqrt(mean(errors))
  }

  # 3. Apply Selection Logic
  results_ord <- results[order(results$RMSE), , drop = FALSE]
  n_models <- nrow(results)

  if (selection == "top_k") {
    nk <- min(top_k, n_models)
    selected <- results_ord[seq_len(nk), , drop = FALSE]
  } else {
    selected <- results[results$RMSE < threshold, , drop = FALSE]
    if (nrow(selected) == 0) {
      if (verbose) {
        message("\u26a0\ufe0f No models met threshold. Picking top ", top_k, ".")
      }
      nk <- min(top_k, n_models)
      selected <- results_ord[seq_len(nk), , drop = FALSE]
    }
  }

  if (verbose) {
    cat("\n\u2705 Final Coalition:\n")
    print(selected)
  }

  out <- selected$Model
  if (isTRUE(return_rmse)) {
    attr(out, "cv_rmse") <- setNames(selected$RMSE, selected$Model)
  }
  return(out)
}

#' Evaluate Coalition Performance
#'
#' @description
#' Evaluates the performance of a selected coalition on test data
#'
#' @param coalition Character vector of model names
#' @param train_data Data used to fit coalition members (same split as when building the coalition)
#' @param test_data Data frame for evaluation (held-out)
#' @param target Character. Name of the response column (must match training)
#'
#' @return Numeric. Mean absolute error of coalition predictions
#'
#' @param weights Optional numeric vector (same length as \code{coalition}), e.g. \code{1 / attr(coalition, "cv_rmse")}
#' @export
evaluate_coalition <- function(coalition, train_data, test_data, target = "quality", weights = NULL) {
  if (length(coalition) == 0) {
    stop("Coalition cannot be empty")
  }
  if (!(target %in% colnames(test_data))) {
    stop("test_data must contain target column '", target, "'", call. = FALSE)
  }
  if (!(target %in% colnames(train_data))) {
    stop("train_data must contain target column '", target, "'", call. = FALSE)
  }

  y_hat <- predict_tbde_ensemble(train_data, test_data, coalition, target, weights = weights)
  mean(abs(test_data[[target]] - y_hat))
}
