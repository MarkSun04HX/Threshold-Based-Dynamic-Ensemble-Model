# AutoML Coalition Script in R
library(dplyr)

build_coalition <- function(data, threshold = 5.0) {
  
  # 1. Define our "Lite" models as a list of functions
  # In a real scenario, you'd call 'xgboost()', 'randomForest()', etc.
  models <- list(
    XGBoost = function(train) { return(mean(train$cost)) }, # Placeholder logic
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
  
  # 2. K-Fold Cross Validation (3-Fold)
  results <- data.frame(Model = names(models), RMSE = NA, stringsAsFactors = FALSE)
  
  set.seed(123)
  folds <- cut(seq(1, nrow(data)), breaks = 3, labels = FALSE)
  
  for (m_name in names(models)) {
    errors <- c()
    
    for (i in 1:3) {
      test_idx <- which(folds == i, arr.ind = TRUE)
      test_data <- data[test_idx, ]
      train_data <- data[-test_idx, ]
      
      # Train and Predict (Simplified)
      pred <- models[[m_name]](train_data)
      # If the model returns a complex object (like lm), we'd use predict()
      if(is.numeric(pred)) {
        actual_preds <- rep(pred, nrow(test_data))
      } else {
        actual_preds <- predict(pred, test_data)
      }
      
      errors <- c(errors, (test_data$cost - actual_preds)^2)
    }
    results$RMSE[results$Model == m_name] <- sqrt(mean(errors))
  }
  
  # 3. Apply Selection Logic
  selected <- results %>% filter(RMSE < threshold)
  
  if (nrow(selected) == 0) {
    message("⚠️ No models met threshold. Picking Top 3.")
    selected <- results %>% arrange(RMSE) %>% head(3)
  }
  
  print("✅ Final Coalition:")
  print(selected)
  
  return(selected$Model)
}

# --- Example Usage ---
# dummy_data <- data.frame(cost = c(10, 15, 12, 40, 11, 14, 13))
# my_coalition <- build_coalition(dummy_data, threshold = 10)
