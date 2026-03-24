# Synthetic data for examples and tests (source this file; defines synthetic_data)
set.seed(42)
n <- 200
synthetic_data <- data.frame(
  quality = rnorm(n, mean = 50, sd = 15),
  feature1 = rnorm(n),
  feature2 = rnorm(n),
  feature3 = runif(n)
)
