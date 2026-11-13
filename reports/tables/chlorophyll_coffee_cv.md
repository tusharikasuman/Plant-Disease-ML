## Coffee SPAD - 10-fold CV on 340 training leaves (ArabicaLens - Radhakrishnan, Joseph & Kumar (2026), Zenodo, doi:10.5281/zenodo.19447467, CC BY 4.0)

| Features | # features | Model | Syllabus | CV R2 | CV RMSE (SPAD) | CV MAE (SPAD) | Time (s) |
|---|---|---|---|---|---|---|---|
| colour + HSV histogram | 38 | Random forest | Combining models: bagged trees | 0.89 | 5.81 | 4.14 | 6.37 |
| colour | 6 | Gaussian process | Unit 5: kernel method with error bars | 0.89 | 5.81 | 4.26 | 1.50 |
| all 55 features | 55 | Random forest | Combining models: bagged trees | 0.89 | 5.82 | 4.19 | 9.04 |
| colour | 6 | Polynomial (degree 2) + ridge | Unit 2: linear model on non-linear basis functions | 0.89 | 5.88 | 4.33 | 0.10 |
| rgb (Paper 2) | 3 | Polynomial (degree 2) + ridge | Unit 2: linear model on non-linear basis functions | 0.88 | 6.00 | 4.38 | 0.07 |
| rgb (Paper 2) | 3 | Gaussian process | Unit 5: kernel method with error bars | 0.88 | 6.02 | 4.39 | 1.88 |
| colour | 6 | Random forest | Combining models: bagged trees | 0.87 | 6.18 | 4.47 | 4.38 |
| colour | 6 | Least squares (OLS) | Unit 2: b = A X (Paper 2) | 0.87 | 6.27 | 4.59 | 0.04 |
| colour | 6 | Ridge (lambda by CV) | Unit 2: L2 penalty against collinear colours | 0.87 | 6.27 | 4.59 | 0.07 |
| colour | 6 | Bayesian ridge | Unit 2: prior strength from the evidence | 0.87 | 6.30 | 4.58 | 0.05 |
| all 55 features | 55 | Ridge (lambda by CV) | Unit 2: L2 penalty against collinear colours | 0.87 | 6.33 | 4.63 | 0.15 |
| all 55 features | 55 | Gaussian process | Unit 5: kernel method with error bars | 0.87 | 6.36 | 4.54 | 2.15 |
| rgb (Paper 2) | 3 | Random forest | Combining models: bagged trees | 0.86 | 6.40 | 4.57 | 3.92 |
| colour + HSV histogram | 38 | Bayesian ridge | Unit 2: prior strength from the evidence | 0.86 | 6.40 | 4.68 | 0.07 |
| colour + HSV histogram | 38 | Gaussian process | Unit 5: kernel method with error bars | 0.86 | 6.42 | 4.56 | 2.09 |
| colour + HSV histogram | 38 | Ridge (lambda by CV) | Unit 2: L2 penalty against collinear colours | 0.86 | 6.44 | 4.72 | 0.10 |
| rgb (Paper 2) | 3 | Least squares (OLS) | Unit 2: b = A X (Paper 2) | 0.86 | 6.45 | 4.67 | 0.04 |
| rgb (Paper 2) | 3 | Ridge (lambda by CV) | Unit 2: L2 penalty against collinear colours | 0.86 | 6.45 | 4.67 | 0.07 |
| rgb (Paper 2) | 3 | Bayesian ridge | Unit 2: prior strength from the evidence | 0.86 | 6.45 | 4.67 | 0.05 |
| all 55 features | 55 | Bayesian ridge | Unit 2: prior strength from the evidence | 0.84 | 6.92 | 4.76 | 0.09 |
| colour + HSV histogram | 38 | Least squares (OLS) | Unit 2: b = A X (Paper 2) | 0.83 | 7.14 | 4.89 | 0.06 |
| all 55 features | 55 | Least squares (OLS) | Unit 2: b = A X (Paper 2) | 0.54 | 11.82 | 5.43 | 0.07 |
