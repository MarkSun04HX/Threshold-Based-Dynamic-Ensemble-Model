# Threshold-Based-Dynamic-Ensemble-Model
The Threshold-Based Dynamic Ensemble (TBDE) is an AutoML architecture that "auditions" multiple models (XGBoost, NN, etc.) via Cross-Validation. Only models beating a specific RMSE threshold join the voting coalition. If none pass, it picks the top 3. This ensures a robust, self-optimizing consensus that filters out weak predictors.
