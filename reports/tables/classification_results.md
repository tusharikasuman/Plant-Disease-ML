## Disease classification - 5-fold stratified CV and held-out valid set

| Model | Syllabus | Best params | CV accuracy (%) | CV std | Valid accuracy (%) | Valid balanced acc (%) | Valid macro F1 (%) | Train size | Time (s) | Best | Topic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Unit 2 | C=100 (lambda=0.01) | 84.00 | 1.76 | 89.29 | 85.01 | 85.57 | 9302 | 22.52 |  | Probabilistic discriminative model; softmax fitted by MLE with L2 penalty (lambda = 1/C) |
| LDA | Unit 2 | shrinkage=None | 79.92 | 1.66 | 85.04 | 78.90 | 79.45 | 9302 | 0.76 |  | Linear discriminant functions; probabilistic generative model (Gaussians, shared covariance) |
| Gaussian Naive Bayes | Unit 1 | var_smoothing=1e-06 | 64.52 | 1.39 | 71.07 | 64.19 | 60.47 | 9302 | 0.40 |  | Bayes' rule with class-conditional independence of features |
| SVM (linear kernel) | Unit 2 | C=10 (lambda=0.1) | 83.32 | 1.51 | 88.52 | 82.79 | 83.90 | 9302 | 38.40 |  | Maximum-margin classifier, linear kernel (LinearSVC solver) |
| SVM (RBF kernel) | Unit 2 | C=10 (lambda=0.1), gamma=scale | 87.40 | 1.11 | 93.25 | 91.63 | 91.03 | 4800 | 7.02 |  | Kernel trick: RBF kernel = maximum margin in an implicit feature space |
| KNN | Baseline (Paper 1) | n_neighbors=9, weights=distance | 80.36 | 1.67 | 87.79 | 81.02 | 82.97 | 9302 | 1.68 |  | Instance-based: majority vote of the k nearest leaves |
| Decision Tree | Baseline (Paper 1) | max_depth=10, min_samples_leaf=1 | 75.76 | 1.40 | 84.82 | 79.55 | 79.79 | 9302 | 2.38 |  | Axis-aligned if/else splits chosen by Gini impurity |
| Random Forest | Baseline (Paper 1) | max_depth=20, min_samples_leaf=1 | 89.40 | 0.78 | 94.07 | 90.35 | 91.44 | 9302 | 19.21 |  | Bagged ensemble of decision trees with random feature subsets |
| Gradient Boosting | Ensemble: boosting | learning_rate=0.1, max_leaf_nodes=31 | 92.24 | 1.02 | 96.30 | 94.13 | 94.34 | 9302 | 102.64 | * | Trees added one after another, each fitting the errors of the ensemble so far |
