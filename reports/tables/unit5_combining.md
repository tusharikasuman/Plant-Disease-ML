## Unit 5 - boosting, committees and stacking

| Model | Kind | Valid accuracy (%) |
|---|---|---|
| Logistic Regression | member | 89.29 |
| LDA | member | 85.04 |
| Gaussian Naive Bayes | member | 71.07 |
| KNN (k = 7) | member | 87.70 |
| Decision Tree (depth 10) | member | 84.82 |
| Committee - hard (majority) vote | committee | 89.55 |
| Committee - soft vote (average probability) | committee | 88.69 |
| Stacking (logistic regression on members' outputs) | committee | 91.83 |
| AdaBoost from scratch (300 stumps) | boosting | 83.96 |
