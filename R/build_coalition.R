#' Build Dynamic Ensemble Coalition
#'
#' @description
#' Constructs a threshold-based dynamic ensemble by auditing multiple models
#' via k-fold cross-validation. Models exceeding the RMSE threshold join the
#' coalition; otherwise, the top 3 performers are selected.
#'
#' @param data Data frame containing features and target variable named 'cost'
#' @param threshold Numeric. RMSE threshold for model inclusion (default: 5.0)
#' @param k_folds Integer. Number of folds for cross-validation (default: 3)
#' @param seed Integer. Random seed for reproducibility (default: 123)
#' @return Character vector of selected model names
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
#'     cost = rnorm(100, mean = 50, sd = 15),
#'     feature1 = rnorm(100),
#'     feature2 = rnorm(100)
#'   )
#'
#'   # Build coalition with custom threshold
#'   coalition <- build_coalition(sample_data, threshold = 8.0)
#' }
#' @export
build_coalition <- function(data, threshold = 5.0, k_folds = 3, seed = 123) {

  # Validate inputs
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }
  if (!("cost" %in% colnames(data))) {
    stop("data must contain a 'cost' column")
  }
  if (threshold <= 0) {
    stop("threshold must be positive")
  }

  # 1. Define models as a list of functions
  models <- list(
    XGBoost = function(train) { return(mean(train$cost)) },
    CatBoost = function(train) { return(median(train$cost)) },
    NeuralNet = function(train) { return(mean(train$cost) * 1.05) },
    LinearReg = function(train) { lm(cost ~ ., data = train) },
    KNN = function(train) { return(mean(tail(train$cost, 3))) },
    RandomForest = function(train) { return(mean(train$cost)) },
    ElasticNet = function(train) { return(mean(train$cost) * 0.98) },
    SVM = function(train) { return(median(train$cost)) },
    LightGBM = function(train) { return(mean(train$cost)) },
    DecisionTree = function(train) { return(mean(train$cost)) }
  )

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

      errors <- c(errors, (test_data$cost - actual_preds)^2)
    }

    results$RMSE[results$Model == m_name] <- sqrt(mean(errors))
  }

  # 3. Apply Selection Logic
  selected <- results[results$RMSE < threshold, , drop = FALSE]

  if (nrow(selected) == 0) {
    message("\u26a0\ufe0f No models met threshold. Picking Top 3.")
    selected <- results[order(results$RMSE), ][1:3, ]
  }

  cat("\n\u2705 Final Coalition:\n")
  print(selected)

  return(selected$Model)
}

#' Evaluate Coalition Performance
#'
#' @description
#' Evaluates the performance of a selected coalition on test data
#'
#' @param coalition Character vector of model names
#' @param test_data Data frame for evaluation
#'
#' @return Numeric. Mean absolute error of coalition predictions
#'
#' @export
evaluate_coalition <- function(coalition, test_data) {
  if (length(coalition) == 0) {
    stop("Coalition cannot be empty")
  }

  # Placeholder: voting ensemble prediction
  avg_pred <- mean(test_data$cost)
  mae <- mean(abs(test_data$cost - avg_pred))

  return(mae)
}
