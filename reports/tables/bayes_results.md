## Bayesian logistic regression, healthy vs diseased (alpha = 0.03 chosen by evidence)

| Model | Accuracy (%) | Balanced acc (%) | Log-loss | Brier | ECE |
|---|---|---|---|---|---|
| sklearn LogisticRegression (C = 1/alpha) | 99.48 | 99.17 | 0.02 | 0.00 | 0.00 |
| Bayesian LR - MAP plug-in sigma(w^T x) | 99.48 | 99.17 | 0.02 | 0.00 | 0.00 |
| Bayesian LR - predictive (probit approx.) | 99.48 | 99.17 | 0.02 | 0.00 | 0.01 |
