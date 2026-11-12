## Unit 5 - GP vs RVM vs SVR for lesion severity

| Model | Basis vectors kept | R2 | RMSE | 95% interval coverage (%) | Fit time (s) |
|---|---|---|---|---|---|
| Gaussian process (ARD RBF + noise, hyper-parameters by marginal likelihood) | 600 | 0.84 | 0.64 | 94.25 | 13.25 |
| Relevance Vector Machine (from scratch) | 35 | 0.76 | 0.79 | 90.50 | 3.49 |
| Support Vector Regression (epsilon-insensitive) | 447 | 0.80 | 0.72 | nan | nan |
