## Feature ablation - 5-fold CV accuracy as feature groups are added

| Feature set | Groups | # features | SVM (RBF kernel) CV acc (%) | SVM (RBF kernel) std | Logistic Regression CV acc (%) | Logistic Regression std |
|---|---|---|---|---|---|---|
| RGB stats only | rgb | 6 | 73.12 | 2.29 | 61.24 | 2.50 |
| + indices + HSV histogram | rgb + indices + hsv | 42 | 86.08 | 0.85 | 79.72 | 1.58 |
| + texture (GLCM) | rgb + indices + hsv + texture | 48 | 88.08 | 0.84 | 83.20 | 1.56 |
| + shape (Hu moments) = all | rgb + indices + hsv + texture + shape | 55 | 87.40 | 1.11 | 84.00 | 1.76 |
