# Explainable Plant Leaf Disease Detection and Chlorophyll Estimation using Classical Machine Learning

**Design, Implementation, and Performance Evaluation of Machine Learning Techniques**
21CSC305P – MACHINE LEARNING PROJECT

Submitted by: Name [Reg No] · Name [Reg No]
Under the Guidance of: Faculty Name (Designation, Department of Computational Intelligence)

> **Notes for the final Word file (delete this box before submission)**
> - Every number in this report comes from the files in `reports/tables/` of the project; nothing is estimated.
> - Figures marked **[Insert: …]** give the PNG to paste. Unit 5 and chlorophyll figures are already in
>   `reports/figures/15_…` to `19_…`. The Chapter 2 and 3 figures are not kept in the repository; generate them
>   once with `python -m src.evaluate --all`, `python -m src.hmm` and
>   `python -m src.leaf_unsupervised samples/tomato_early_blight.jpg`.
> - Fill in names, register numbers, guide name and the Google Colab links (Appendix A).
> - Apply the formatting rules: Times New Roman, body 12 pt justified, 1.5 spacing, captions 11 pt bold, tables
>   above-captioned, figures below-captioned, code in Consolas 10 pt.

---

## ABSTRACT

Plant diseases reduce crop yield, while expert diagnosis is slow and deep-learning tools are opaque. This project
builds an explainable, CPU-only system that diagnoses tomato leaf diseases from a single photograph and estimates
leaf chlorophyll using only classical machine learning. The objectives were to convert each image into
interpretable features, compare probabilistic and linear classifiers, discover structure without labels, model
pixel sequences, and improve accuracy with advanced and ensemble methods. The primary dataset is the PlantVillage
tomato subset of 11,628 images in five classes, from which 55 colour, texture and shape features were extracted
after HSV segmentation. Unit 1 applied Bayes' rule through Gaussian Naive Bayes; Unit 2 compared logistic
regression, linear discriminant analysis, linear and RBF support vector machines and a from-scratch Bayesian
logistic regression; Unit 3 applied PCA, k-means and Gaussian mixtures trained with expectation–maximisation;
Unit 4 modelled image rows as sequences with a hidden Markov model decoded by forward–backward and Viterbi
algorithms; and Unit 5 implemented a neural network, AdaBoost, committees, stacking, tree ensembles, kernel methods
and sampling-based graphical models. The ArabicaLens dataset of 425 coffee leaves with SPAD readings was used for
chlorophyll regression. Gradient boosting achieved the best held-out accuracy of 96.3 percent with 94.3 percent
macro F1, against 71.1 percent for Naive Bayes. Bayesian logistic regression separated healthy and diseased
leaves with 99.5 percent accuracy, and chlorophyll was estimated with an R² of 0.91. Interpretable features with
classical models therefore give accurate, explainable plant health assessment.

*(242 words)*

---

## TABLE OF CONTENTS

| Chapter | Title | Page |
|---|---|---|
| | Abstract | v |
| | Table of Contents | vi |
| | List of Figures | vii |
| | List of Tables | vii |
| | Abbreviations | viii |
| 1 | Introduction | 1 |
| 2 | Data Analysis and Fundamental Machine Learning Techniques (Units 1 and 2) | |
| 3 | Unsupervised Learning and Sequential Data Modelling (Units 3 and 4) | |
| 4 | Advanced Predictive Modelling and Ensemble Learning (Unit 5) | |
| 5 | Integrated Discussion and Findings | |
| 6 | Overall Conclusion and Future Scope | |
| | References | |
| | Appendix A – Coding | |

*(Fill page numbers after the Word layout is final.)*

## LIST OF FIGURES

| No. | Title | Source file |
|---|---|---|
| 2.1 | Leaf segmentation examples (original, leaf mask, lesion mask) | `01_feature_representation/1_segmentation_examples.png` |
| 2.2 | Key features by class (lesion fraction, greenness, texture) | `01_feature_representation/2_key_features_by_class.png` |
| 2.3 | Proposed workflow for Units 1 and 2 | draw from Section 2.5 |
| 2.4 | Model comparison: CV and held-out accuracy | `10_evaluation_metrics/1_model_comparison.png` |
| 2.5 | SVM grid search over C and γ | `09_cv_model_selection/2_svm_grid_search.png` |
| 2.6 | Cumulative feature-group ablation | `11_ablation/1_cumulative_ablation.png` |
| 2.7 | Bayesian LR: confidence versus error | `12_bayesian_logistic_regression/2_confidence_vs_error.png` |
| 3.1 | PCA scatter and scree plot | `02_pca/1_pca_scatter_scree.png` |
| 3.2 | Accuracy versus number of principal components | `02_pca/2_accuracy_vs_components.png` |
| 3.3 | k-means elbow and silhouette | `03_kmeans/1_elbow_silhouette.png` |
| 3.4 | GMM BIC and EM log-likelihood | `04_gmm_em/1_bic_and_em.png` |
| 3.5 | Clusters versus true classes | `03_kmeans/2_clusters_vs_classes.png` |
| 3.6 | Unit 3 workflow | draw from Section 3.5 |
| 3.7 | Unit 4 workflow | draw from Section 3.10 |
| 3.8 | Viterbi and forward–backward decoding of leaves | `14_hmm_viterbi_forward_backward/1_viterbi_and_forward_backward_leaves.png` |
| 3.9 | One pixel row: forward–backward posteriors and Viterbi path | `14_hmm_viterbi_forward_backward/2_one_row_forward_backward_viterbi.png` |
| 4.1 | Unit 5 workflow | draw from Section 4.4 |
| 4.2 | Decision-tree depth versus over-fitting | `05_decision_trees/1_depth_vs_overfitting.png` |
| 4.3 | Bagging versus boosting | `07_boosting/2_bagging_vs_boosting.png` |
| 4.4 | Neural network training curves | `15_neural_networks/1_training_curves.png` |
| 4.5 | Accuracy versus hidden units | `15_neural_networks/2_accuracy_vs_hidden_units.png` |
| 4.6 | Decision boundaries: logistic regression vs neural network | `15_neural_networks/3_decision_boundaries_2d.png` |
| 4.7 | AdaBoost error and α per round | `16_combining_models/1_adaboost_rounds.png` |
| 4.8 | Committees versus single members | `16_combining_models/2_committee_vs_members.png` |
| 4.9 | SVM support vectors versus C | `17_kernel_methods/1_svm_support_vectors_vs_C.png` |
| 4.10 | Gaussian process regression with error bars | `17_kernel_methods/2_gaussian_process_regression.png` |
| 4.11 | RVM versus SVR sparsity | `17_kernel_methods/3_rvm_vs_svr_sparsity.png` |
| 4.12 | MCMC versus Laplace posterior | `18_sampling_graphical_models/1_mcmc_vs_laplace.png` |
| 4.13 | Gibbs sampling in an Ising MRF: lesion-mask denoising | `18_sampling_graphical_models/2_gibbs_mrf_lesion_denoising.png` |
| 4.14 | Bayesian network inference: exact vs sampled | `18_sampling_graphical_models/3_bayes_net_inference.png` |
| 4.15 | Confusion matrix of the best model | `10_evaluation_metrics/1_confusion_matrix.png` |
| 4.16 | Chlorophyll: predicted vs measured SPAD | `19_chlorophyll_spad/1_predicted_vs_measured.png` |
| 4.17 | Chlorophyll model comparison | `19_chlorophyll_spad/2_model_comparison.png` |
| 4.18 | Greenness vs SPAD and GP error bars | `19_chlorophyll_spad/3_greenness_vs_spad_and_error_bars.png` |

## LIST OF TABLES

| No. | Title |
|---|---|
| 1.1 | Unit-wise topic and dataset allocation |
| 2.1 | Class distribution and train/test split |
| 2.2 | The 55 features |
| 2.3 | Mean lesion fraction per class |
| 2.4 | Correlation of colour features |
| 2.5 | Variance inflation factors |
| 2.6 | Hyper-parameter grids and selected values |
| 2.7 | Classification results of Unit 1 and Unit 2 models |
| 2.8 | Cumulative feature ablation |
| 2.9 | Leave-one-group-out ablation |
| 2.10 | Bayesian logistic regression results |
| 2.11 | Bayesian LR on an unseen disease |
| 3.1 | PCA variance kept and accuracy on top-k components |
| 3.2 | k-means inertia and silhouette |
| 3.3 | GMM BIC for 1 to 12 components |
| 3.4 | Clustering quality (ARI, NMI, purity) |
| 3.5 | Sequential dataset summary |
| 3.6 | HMM transition matrix |
| 3.7 | HMM decoding results on 40 held-out leaves |
| 3.8 | Single-leaf Unit 3 and Unit 4 results |
| 4.1 | Tree ensembles compared with a single tree |
| 4.2 | Neural network results |
| 4.3 | Accuracy versus hidden units |
| 4.4 | Committees, stacking and AdaBoost |
| 4.5 | SVM support vectors versus C |
| 4.6 | GP, RVM and SVR for lesion severity |
| 4.7 | Metropolis–Hastings vs Laplace |
| 4.8 | Bayesian network posterior: exact vs sampled |
| 4.9 | Chlorophyll: 10-fold CV (top models) |
| 4.10 | Chlorophyll: held-out test |
| 4.11 | All models on the same held-out test set |
| 4.12 | Per-class report of the best model |

## ABBREVIATIONS

| Abbreviation | Expansion |
|---|---|
| ARD | Automatic Relevance Determination |
| ARI | Adjusted Rand Index |
| ASM | Angular Second Moment |
| BIC | Bayesian Information Criterion |
| CART | Classification and Regression Trees |
| CV | Cross-Validation |
| ECE | Expected Calibration Error |
| EDA | Exploratory Data Analysis |
| EM | Expectation–Maximisation |
| ExG | Excess Green index |
| GLCM | Grey-Level Co-occurrence Matrix |
| GLI | Green Leaf Index |
| GMM | Gaussian Mixture Model |
| GP | Gaussian Process |
| HMM | Hidden Markov Model |
| HSV | Hue, Saturation, Value |
| KNN | k-Nearest Neighbours |
| LDA | Linear Discriminant Analysis |
| MAP | Maximum A Posteriori |
| MCMC | Markov Chain Monte Carlo |
| MLE | Maximum Likelihood Estimation |
| MLP | Multilayer Perceptron |
| MRF | Markov Random Field |
| NMI | Normalised Mutual Information |
| OLS | Ordinary Least Squares |
| PCA | Principal Component Analysis |
| RBF | Radial Basis Function |
| RMSE | Root Mean Squared Error |
| RVM | Relevance Vector Machine |
| SAMME | Stagewise Additive Modelling using a Multi-class Exponential loss |
| SPAD | Soil Plant Analysis Development (chlorophyll meter units) |
| SVM / SVR | Support Vector Machine / Regression |
| VIF | Variance Inflation Factor |

---

# CHAPTER 1 – INTRODUCTION

## 1.1 Background, Motivation and Scope

Plant diseases cause large yield losses every season, and tomato is among the most affected vegetable crops.
Early blight, late blight, Septoria leaf spot and tomato yellow leaf curl virus can spread across a field within
days, while diagnosis normally needs a trained plant pathologist. A smartphone photograph is the cheapest sensor
available to a farmer, so automatic diagnosis from a leaf photo can make early detection widely accessible.

Most published systems use deep convolutional networks. They are accurate but act as black boxes, need GPUs and
large labelled datasets, and give no reason for a decision. This project takes the opposite route: each photograph
is converted into **55 measurable, human-readable features** (colour, greenness, lesion area, texture and shape),
and classical machine-learning models from the syllabus learn from those features. Every model runs on an ordinary
laptop CPU, and every prediction can be explained in terms of the features.

Leaf chlorophyll content is a second indicator of plant health. It is normally measured with a SPAD meter, a
contact device that is not available to most farmers. The project therefore also estimates SPAD values from leaf
colour, trained on real meter readings.

**Stakeholders:** farmers and extension workers (fast field diagnosis), agronomists and plant pathologists
(screening support, with an "uncertain – refer to expert" flag), and researchers (a transparent, reproducible
baseline).

**Scope:** single-leaf laboratory photographs on a plain background; five tomato classes; chlorophyll estimation
calibrated for coffee leaves (the only crop with public paired photo–SPAD data). Field images with several leaves,
soil and strong shadows, and deep learning, are outside the scope.

## 1.2 Integrated Problem Statement

*Given a single photograph of a tomato leaf, determine whether it is healthy or diseased, identify the disease,
measure the damaged area, and state how confident the system is, using only interpretable features and classical
machine learning; additionally, estimate the leaf's chlorophyll (SPAD) value from its colour.*

The integrated problem is broken into linked sub-problems, all on the original PlantVillage tomato dataset:

- **Units 1 and 2 (PBL 1):** five-class disease classification and binary healthy/diseased classification from the
  55 features, using probabilistic (Bayes) and linear/kernel classifiers, with calibrated uncertainty.
- **Unit 3 (PBL 2):** can the same features reveal disease groups **without labels**, and how many dimensions are
  really needed?
- **Unit 4 (PBL 2):** treat each row of pixels of a leaf image as an ordered sequence and infer the hidden tissue
  state (background, healthy tissue, lesion) of every pixel with a hidden Markov model. The sequential data are
  derived from the same PlantVillage images (sequence = image row, position index = pixel column); no separate
  time-series dataset was used.
- **Unit 5 (PBL 3):** improve on the PBL 1 baseline with neural networks, ensembles (CART, bagging, boosting,
  committees), kernel methods, and sampling/graphical models; and estimate chlorophyll with regression on the
  additional ArabicaLens coffee-leaf dataset.

## 1.3 Project Objectives

- **Objective 1 (Unit 1 – data analysis and Bayesian decision making):** extract 55 explainable features, study
  their distributions, correlations and collinearity, and build a Gaussian Naive Bayes classifier using Bayes'
  rule. *Result:* 71.1% held-out accuracy; showed that the independence assumption fails (R–G correlation 0.91).
- **Objective 2 (Unit 2 – linear models for classification and regression):** compare logistic regression, LDA,
  linear and RBF SVMs, and implement Bayesian logistic regression from scratch with a Laplace approximation and
  evidence-based prior selection. *Result:* RBF SVM 93.3% (five classes); Bayesian LR 99.5% healthy vs diseased
  with honest confidence.
- **Objective 3 (Unit 3 – unsupervised learning and dimensionality reduction):** apply PCA, k-means and
  Gaussian mixtures with EM. *Result:* 30 of 55 components keep 95% of the variance; clustering ARI 0.22
  (random = 0.00).
- **Objective 4 (Unit 4 – sequential modelling):** build an HMM over pixel rows and decode it with forward–backward
  and Viterbi. *Result:* 95.1% pixel agreement with the reference segmentation and fewer spurious state changes
  (2.48 per row vs 3.97 without the HMM).
- **Objective 5 (Unit 5 – advanced models and ensembles):** implement a neural network, AdaBoost, committees,
  stacking, tree ensembles, kernel methods and sampling methods. *Result:* gradient boosting 96.3% held-out
  accuracy (best overall); neural network 93.2%; chlorophyll R² 0.91.

## 1.4 Unit-wise Topic and Dataset Allocation

**Table 1.1: Unit-wise topic and dataset allocation**

| PBL | Unit | Allotted topic / algorithm | Dataset |
|---|---|---|---|
| PBL 1 | 1 | Probability, Bayes' rule, Gaussian Naive Bayes; EDA, covariance/correlation, collinearity (VIF) | Original (PlantVillage tomato) |
| PBL 1 | 2 | Linear models: logistic regression, LDA, SVM (linear, RBF kernel), Bayesian logistic regression; linear regression baselines | Original |
| PBL 2 | 3 | PCA, k-means clustering, Gaussian mixture model with EM | Original |
| PBL 2 | 4 | Hidden Markov model, forward–backward, Viterbi | Sequences derived from the original images (pixel rows) |
| PBL 3 | 5 | Neural networks (MLP + backpropagation), combining models (CART, random forest, gradient boosting, AdaBoost, committees, stacking), kernel methods (SVM dual, GP, RVM), sampling and graphical models (MCMC, Gibbs/MRF, Bayesian network) | Original + ArabicaLens (chlorophyll) |

> *Check the "Allotted topic" column against your official allotment sheet and edit the wording if it differs.*

---

# CHAPTER 2 – DATA ANALYSIS AND FUNDAMENTAL MACHINE LEARNING TECHNIQUES (UNITS 1 AND 2)

## 2.1 Overview of Allotted Topics

**Unit 1 – Probability and Bayesian classification.** Bayes' rule turns class-conditional densities into a
posterior class probability. Gaussian Naive Bayes additionally assumes that, within a class, the features are
independent. It is the natural first baseline: it trains in milliseconds and is fully interpretable. Unit 1 data
analysis (means, covariance, correlation and VIF) is also used to test whether its assumption holds for leaf
features.

**Unit 2 – Linear models.**
- *Logistic regression* (probabilistic discriminative model) models P(class | features) directly with a softmax,
  fitted by maximum likelihood with an L2 penalty. Its weights explain which features push a leaf towards a
  disease, so it powers the "why this prediction?" panel.
- *LDA* (probabilistic generative model) models each class as a Gaussian with a shared covariance and applies
  Bayes' rule, giving linear decision boundaries.
- *SVM* finds the maximum-margin separating hyperplane; the *RBF kernel* lets it draw curved boundaries in an
  implicit high-dimensional space.
- *Bayesian logistic regression* places a Gaussian prior on the weights, finds the MAP solution with Newton's
  method, approximates the posterior with a Laplace (Gaussian) approximation, chooses the prior strength by the
  model evidence, and predicts with the probit approximation. This gives a calibrated confidence and an
  "unsure – refer to an expert" signal.
- *KNN* (k-nearest neighbours) is kept as a non-parametric baseline from the reference paper [1].

These methods fit the problem because the 55 features are continuous, mostly unimodal per class and partly
collinear, which suits Gaussian and linear models and lets their assumptions be tested.

## 2.2 Dataset Description

- **Name:** PlantVillage – tomato subset (colour images) [5].
- **Source:** PlantVillage open repository (Hughes and Salathé, arXiv:1511.08060); the same images are included in
  the Kaggle "New Plant Diseases Dataset" (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset).
- **Size:** 11,628 RGB images, 256 × 256 pixels, JPG, ≈ 200 MB; one leaf per image on a plain background.
- **Target:** disease class (five classes); a derived binary target healthy/diseased.
- **Features:** 55 numeric features engineered from each image (Table 2.2). All are continuous (float).
- **Missing values:** none. Images where the leaf mask covers less than 1% of the picture fall back to the whole
  image, so every image yields all 55 values.
- **Limitations:** laboratory conditions (single leaf, uniform background); class imbalance (yellow leaf curl virus
  is 46% of the data); two different photo backgrounds exist within some classes.

**Table 2.1: Class distribution and train/test split (stratified 80/20, seed 42)**

| Class | Images | Training | Held-out test |
|---|---|---|---|
| Early blight | 1,000 | 800 | 200 |
| Late blight | 1,909 | 1,527 | 382 |
| Septoria leaf spot | 1,771 | 1,417 | 354 |
| Yellow leaf curl virus | 5,357 | 4,285 | 1,072 |
| Healthy | 1,591 | 1,273 | 318 |
| **Total** | **11,628** | **9,302** | **2,326** |

*Interpretation:* the classes are imbalanced by a factor of five, so the report gives balanced accuracy and macro
F1 next to plain accuracy, and the split is stratified so the test set keeps the same proportions.

## 2.3 Data Preprocessing

1. **Resize and denoise:** every image is resized to 256 × 256 and smoothed with a 3 × 3 Gaussian blur to remove
   sensor noise.
2. **Segmentation in HSV space:** HSV separates colour (hue) from brightness, so green tissue
   (H 25–95, S ≥ 30, V ≥ 30) and brown lesions (H 5–24, S ≥ 70, V ≥ 20) are found with colour thresholds.
   A morphological closing (7 × 7) fills small gaps, the largest connected region is kept, holes are filled so dark
   lesions inside the leaf are kept, and dark brown pixels outside the green outline are rejected as shadows
   (V < 60). This shadow rule was added after shadows were found to be counted as disease.
3. **Feature extraction (leaf pixels only):** the 55 features of Table 2.2 are measured inside the leaf mask, so
   the background cannot influence the model.
4. **Caching:** features are extracted in parallel (≈ 4.4 ms per image on 8 CPU workers) and cached in
   `cache/features.npz`.
5. **Train/test split:** stratified 80/20 split with seed 42; the 2,326 test leaves are used **once**, at the end.
6. **Scaling:** every model is a `StandardScaler → model` pipeline, so the scaler is fitted inside each CV fold on
   training data only (no leakage).
7. **Feature selection:** no automatic selection; instead the contribution of each feature group is measured by
   ablation (Section 2.7).

**Table 2.2: The 55 features**

| Group | Count | Features | What it captures |
|---|---|---|---|
| RGB statistics | 6 | mean and standard deviation of R, G, B | average colour and blotchiness |
| Indices | 4 | GLI, G/R ratio, normalised ExG, lesion fraction | greenness; lesion % = brown pixels ÷ leaf pixels |
| HSV histogram | 32 | 8 hue × 2 saturation × 2 value bins (fractions) | yellowing and browning patterns |
| GLCM texture | 6 | contrast, dissimilarity, homogeneity, energy, correlation, ASM (distances 1, 2; 4 angles; 32 grey levels) | surface roughness caused by spots |
| Hu moments | 7 | log-scaled Hu invariant moments of the leaf mask | leaf outline shape (curling) |

**Essential code (from `src/features.py`):**

```python
img = cv2.GaussianBlur(cv2.resize(img, (256, 256)), (3, 3), 0)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
green = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
brown = cv2.inRange(hsv, (5, 70, 20), (24, 255, 255))
mask  = cv2.morphologyEx(green | brown, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
mask  = fill_holes(largest_component(mask))
lesion_fraction = (brown & mask).sum() / mask.sum()
```

**[Insert: Figure 2.1 – `01_feature_representation/1_segmentation_examples.png`]**
**Figure 2.1: Leaf segmentation examples (original photo, leaf mask, detected lesions)**
*Interpretation:* the mask follows the leaf outline and excludes the background, and the brown lesion mask marks
the spots that drive the lesion-fraction feature.

## 2.4 Exploratory Data Analysis (EDA)

**Table 2.3: Mean lesion fraction per class (training set)**

| Healthy | Yellow leaf curl | Septoria | Early blight | Late blight |
|---|---|---|---|---|
| 1.0% | 2.3% | 5.1% | 9.0% | 12.6% |

*Interpretation:* lesion area alone orders the classes by severity, so it is a strong feature; but Septoria, early
blight and late blight overlap, so colour and texture are needed to separate them.

**[Insert: Figure 2.2 – `01_feature_representation/2_key_features_by_class.png`]**
**Figure 2.2: Distribution of key features by class**
*Interpretation:* healthy leaves have high greenness and low lesion fraction; yellow leaf curl virus shifts the hue
histogram toward yellow; the blights have high GLCM contrast.

**Table 2.4: Correlation of the colour features**

| | R mean | G mean | B mean | GLI | G/R | ExG |
|---|---|---|---|---|---|---|
| R mean | 1.00 | **0.91** | 0.54 | −0.23 | −0.55 | −0.23 |
| G mean | 0.91 | 1.00 | 0.47 | 0.08 | −0.18 | 0.07 |
| B mean | 0.54 | 0.47 | 1.00 | −0.71 | −0.34 | −0.72 |
| GLI | −0.23 | 0.08 | −0.71 | 1.00 | 0.69 | **1.00** |
| G/R | −0.55 | −0.18 | −0.34 | 0.69 | 1.00 | 0.68 |
| ExG | −0.23 | 0.07 | −0.72 | 1.00 | 0.68 | 1.00 |

**Table 2.5: Variance inflation factors**

| Feature | R mean | G mean | B mean | GLI | G/R ratio | ExG |
|---|---|---|---|---|---|---|
| VIF | 62.2 | 62.7 | 15.9 | 807.3 | 12.9 | 715.1 |

*Interpretation:* R and G are 91% correlated and GLI and ExG are almost identical (r = 1.00). Every VIF is above
10, the usual limit for stable least-squares coefficients. Two consequences follow: (i) the Naive Bayes
independence assumption is clearly violated, which predicts its weak result; (ii) linear models need L2
regularisation (logistic regression with λ = 1/C, ridge regression).

## 2.5 Proposed System Architecture / Workflow

**[Draw: Figure 2.3 – boxes connected left to right]**

```
Leaf photo ─► Preprocess ─► Segment (HSV) ─► 55 features ─► Stratified 80/20 split
 (256×256)    resize, blur    green+brown masks,   colour, indices,      9,302 train / 2,326 test
                              closing, holes,      HSV hist, GLCM, Hu
                              shadow removal
                                                         │
                         ┌───────────────────────────────┘
                         ▼
     StandardScaler → model pipeline   (GNB · LR · LDA · SVM-lin · SVM-RBF · KNN ;  Bayesian LR)
                         │
     Grid search + 5-fold stratified CV  ─►  choose by CV score  ─►  test ONCE on held-out set
                         │
     Outputs: disease + confidence, lesion %, top-3, "why this prediction?", refer-to-expert flag
```
**Figure 2.3: Proposed workflow for Units 1 and 2**

*Stages:* (1) input image; (2) preprocessing removes noise and fixes size; (3) segmentation isolates leaf pixels;
(4) feature extraction gives 55 interpretable numbers; (5) splitting holds out a test set; (6) each model is tuned
by grid search with 5-fold stratified CV, scaling learned inside each fold; (7) the selected model is evaluated
once on the test set; (8) the output layer reports the diagnosis with confidence and an explanation.

## 2.6 Algorithm Theory and Mathematical Formulation

Notation: **x** ∈ ℝ^55 is a standardised feature vector, C_k one of K = 5 classes, **w** weights, N training leaves.

**Bayes' rule and Gaussian Naive Bayes (Unit 1)**

  P(C_k | **x**) = p(**x** | C_k) P(C_k) / Σ_j p(**x** | C_j) P(C_j)                              (2.1)

  p(**x** | C_k) = ∏_{d=1}^{55} 𝒩(x_d | μ_{kd}, σ²_{kd})   (class-conditional independence)          (2.2)

Prediction is argmax_k P(C_k | **x**). *Assumption:* features are independent given the class (tested in 2.4).

**Logistic (softmax) regression (Unit 2)**

  P(C_k | **x**) = exp(**w**_kᵀ**x** + b_k) / Σ_j exp(**w**_jᵀ**x** + b_j)                           (2.3)

  E(**W**) = −Σ_n Σ_k t_{nk} ln y_{nk} + (λ/2)‖**W**‖²,  λ = 1/C                                     (2.4)

**Linear discriminant analysis**

  δ_k(**x**) = **x**ᵀ Σ⁻¹ μ_k − ½ μ_kᵀ Σ⁻¹ μ_k + ln P(C_k)   (shared covariance Σ)                    (2.5)

**Support vector machine (soft margin) and RBF kernel**

  min_{**w**,b,ξ} ½‖**w**‖² + C Σ_n ξ_n  s.t. t_n(**w**ᵀφ(**x**_n) + b) ≥ 1 − ξ_n, ξ_n ≥ 0          (2.6)

  k(**x**, **x**′) = exp(−γ ‖**x** − **x**′‖²)                                                       (2.7)

**Bayesian logistic regression (binary: healthy vs diseased)**

  Prior p(**w**) = 𝒩(**0**, α⁻¹**I**); MAP by Newton's method:
  **w**_new = **w** − **H**⁻¹∇E,  ∇E = Φᵀ(**y** − **t**) + α**w**,  **H** = ΦᵀRΦ + α**I**, R = diag(y_n(1−y_n))   (2.8)

  Laplace approximation: p(**w** | 𝒟) ≈ 𝒩(**w**_MAP, **S**_N), **S**_N = **H**⁻¹                       (2.9)

  Predictive (probit approximation): μ_a = **w**_MAPᵀφ, σ²_a = φᵀ**S**_Nφ,
  p(healthy | φ) ≈ σ( μ_a / √(1 + πσ²_a/8) )                                                         (2.10)

  Evidence for choosing α: ln p(𝒟 | α) ≈ ln p(𝒟 | **w**_MAP) − (α/2)‖**w**_MAP‖² + (M/2) ln α − ½ ln|**H**|   (2.11)

*Why appropriate:* (2.10) shrinks the confidence of leaves far from the training data, which is exactly what is
needed for a "refer to expert" rule.

**Evaluation metrics**

  Accuracy = (correct)/N;  balanced accuracy = (1/K) Σ_k recall_k;  macro F1 = (1/K) Σ_k 2P_kR_k/(P_k+R_k)   (2.12)

  Log-loss = −(1/N) Σ_n [t_n ln p_n + (1−t_n) ln(1−p_n)];  Brier = (1/N) Σ_n (p_n − t_n)²              (2.13)

## 2.7 Results and Performance Evaluation

**Table 2.6: Hyper-parameter grids and selected values (5-fold stratified CV)**

| Model | Tuned parameter(s) | Selected |
|---|---|---|
| Gaussian Naive Bayes | var_smoothing | 1e-6 |
| Logistic regression | C (λ = 1/C) | C = 100 (λ = 0.01) |
| LDA | shrinkage | none |
| SVM (linear) | C | C = 10 |
| SVM (RBF) | C, γ | C = 10, γ = scale |
| KNN | k, weights | k = 9, distance-weighted |

**Table 2.7: Classification results of the Unit 1 and Unit 2 models**

| Model | Unit | CV accuracy (%) | Test accuracy (%) | Test balanced acc. (%) | Test macro F1 (%) |
|---|---|---|---|---|---|
| Gaussian Naive Bayes | 1 | 64.5 ± 1.4 | 71.1 | 64.2 | 60.5 |
| LDA | 2 | 79.9 ± 1.7 | 85.0 | 78.9 | 79.5 |
| KNN (baseline) | – | 80.4 ± 1.7 | 87.8 | 81.0 | 83.0 |
| SVM (linear) | 2 | 83.3 ± 1.5 | 88.5 | 82.8 | 83.9 |
| Logistic regression | 2 | 84.0 ± 1.8 | 89.3 | 85.0 | 85.6 |
| **SVM (RBF)** | 2 | **87.4 ± 1.1** | **93.3** | **91.6** | **91.0** |

*Interpretation:* all Unit 2 models beat Naive Bayes by 14 to 22 points, confirming that the independence
assumption costs accuracy on correlated features. The RBF kernel adds about 4 points over the linear SVM, showing
that the class boundaries are curved in feature space. Test accuracy is higher than CV accuracy because CV used a
balanced subsample (500 leaves per class) while the test set contains many easy yellow-leaf-curl leaves; balanced
accuracy removes this effect.

**[Insert: Figure 2.4 – `10_evaluation_metrics/1_model_comparison.png`]**
**Figure 2.4: CV and held-out accuracy of every model**
*Interpretation:* the ranking on CV and on the test set is the same, so model selection by CV is reliable.

**[Insert: Figure 2.5 – `09_cv_model_selection/2_svm_grid_search.png`]**
**Figure 2.5: RBF SVM grid search over C and γ**
*Interpretation:* the best cell is C = 10 with γ = scale; check after regenerating the figure how flat the region
around it is, and describe it here.

**Table 2.8: Cumulative feature ablation (5-fold CV accuracy, %)**

| Feature set | # features | RBF SVM | Logistic regression |
|---|---|---|---|
| RGB statistics only | 6 | 73.1 | 61.2 |
| + indices + HSV histogram | 42 | 86.1 | 79.7 |
| + GLCM texture | 48 | **88.1** | 83.2 |
| + Hu shape = all | 55 | 87.4 | **84.0** |

**Table 2.9: Leave-one-group-out ablation (RBF SVM, all features = 87.4%)**

| Removed group | # removed | CV accuracy (%) | Drop (points) |
|---|---|---|---|
| HSV histogram | 32 | 82.2 | 5.2 |
| GLCM texture | 6 | 84.8 | 2.6 |
| RGB statistics | 6 | 86.4 | 1.0 |
| Indices | 4 | 86.8 | 0.6 |
| Hu shape | 7 | 88.1 | −0.7 |

**[Insert: Figure 2.6 – `11_ablation/1_cumulative_ablation.png`]**
**Figure 2.6: Accuracy as feature groups are added**
*Interpretation:* colour information (indices and HSV histogram) adds 13 points, texture adds 2 more, and shape
adds nothing for the SVM (removing it even helps by 0.7 points), though it helps logistic regression slightly.

**Table 2.10: Bayesian logistic regression, healthy vs diseased (α = 0.03 chosen by evidence)**

| Model | Accuracy (%) | Balanced acc. (%) | Log-loss | Brier |
|---|---|---|---|---|
| scikit-learn logistic regression (C = 1/α) | 99.48 | 99.17 | 0.02 | 0.00 |
| Bayesian LR – MAP plug-in | 99.48 | 99.17 | 0.02 | 0.00 |
| Bayesian LR – predictive (probit) | 99.48 | 99.17 | 0.02 | 0.00 |

*Verification:* the from-scratch MAP weights match scikit-learn's to within 0.0004; the analytic Hessian matches
finite differences; the probit approximation matches a Monte Carlo average.
*Uncertainty:* wrong predictions have an average confidence of 69%, correct ones 99%. Sending the 20% least
confident leaves to an expert gives 100% accuracy on the remaining 80%.

**Table 2.11: Bayesian LR when one disease is hidden from training**

| Hidden disease | Leaves | Detected as diseased (%) | MAP log-loss | Bayes log-loss |
|---|---|---|---|---|
| Early blight | 200 | 99.0 | 0.03 | 0.04 |
| Late blight | 382 | 94.5 | 0.18 | **0.13** |
| Septoria leaf spot | 354 | 90.4 | 0.33 | **0.25** |
| Yellow leaf curl virus | 1,072 | 100.0 | 0.00 | 0.00 |

*Interpretation:* on a disease never seen in training, the Bayesian predictive distribution is less over-confident
and cuts log-loss by 25–30% for late blight and Septoria.

**[Insert: Figure 2.7 – `12_bayesian_logistic_regression/2_confidence_vs_error.png`]**
**Figure 2.7: Confidence of correct versus wrong predictions**
*Interpretation:* errors concentrate at low confidence, which justifies the refer-to-expert threshold of 0.90.

## 2.8 Discussion and Findings

- **What worked:** HSV segmentation plus hand-made features already give 93.3% accuracy with an RBF SVM and 99.5%
  for healthy vs diseased. Pipelines with in-fold scaling and a single final test kept the evaluation honest.
- **Naive Bayes is last (71.1%)** because the strong correlations (Table 2.4) break its central assumption — a
  result predicted by the Unit 1 data analysis.
- **Hardest class:** early blight, often confused with late blight (both are brown-spot diseases).
- **Challenges:** shadows counted as lesions (fixed by the shadow rule); beige backgrounds looked brown (fixed by
  S ≥ 70 for brown); strong collinearity (handled by regularisation).
- **Honest negative result:** on leaves similar to the training set, plain logistic regression is already well
  calibrated; on small training sets (20–100 leaves) the extra Bayesian caution slightly increased log-loss
  (e.g. 0.35 vs 0.28 with 20 leaves). The Bayesian benefit appears on unfamiliar inputs (Table 2.11).
- **Limitations:** laboratory images only; class imbalance; segmentation labels are rule-based.

## 2.9 Chapter Conclusion

PBL 1 established a complete, explainable pipeline from photo to diagnosis. The best Unit 2 model, the RBF SVM,
reached 93.3% test accuracy and 91.0% macro F1, compared with 71.1% for the Unit 1 Naive Bayes baseline, and the
Bayesian logistic regression added a trustworthy confidence for healthy/diseased screening (99.5%). These
results, the fixed 2,326-leaf test set and the 55 features are the baseline for Units 3 to 5.

---

# CHAPTER 3 – UNSUPERVISED LEARNING AND SEQUENTIAL DATA MODELLING (UNITS 3 AND 4)

## 3.1 Overview of Allotted Unit 3 and Unit 4 Topics

**Unit 3:** dimensionality reduction with **PCA**, and clustering with **k-means** and a **Gaussian mixture model
trained by EM**, plus model selection with the elbow method, silhouette score and BIC. The labels are used only
after clustering, to score the clusters.

**Unit 4:** a **hidden Markov model** with discrete observations, trained on labelled pixel sequences, and
inference with the **forward–backward** algorithm (most probable state per pixel) and the **Viterbi** algorithm
(most probable whole state path).

## 3.2 Unit 3 – Dataset Description and Preprocessing

The existing dataset and the 55 features of Chapter 2 are used (9,302 training leaves). Additional preprocessing
specific to Unit 3:
- Features are standardised (zero mean, unit variance) because PCA and k-means are distance/variance based and
  would otherwise be dominated by large-scale features.
- Clustering is run on the first **30 principal components** (95% of variance) to remove redundant, correlated
  directions found in Table 2.4.
- A second, pixel-level analysis is run on a single photograph (`samples/tomato_early_blight.jpg`): 65,536 pixels
  with 7 colour values each (R, G, B, H, S, V and a greenness index).

## 3.3 Unit 3 – Additional EDA

The main EDA is in Section 2.4. The additional analysis relevant to Unit 3 is the variance structure of the
features: because many features are strongly correlated (VIF up to 807), much of the 55-dimensional space is
redundant, which motivates PCA.

**[Insert: Figure 3.1 – `02_pca/1_pca_scatter_scree.png`]**
**Figure 3.1: Leaves on the first two principal components and the scree plot**
*Interpretation:* PC1 explains 16.8% and PC2 10.0% of the variance. Healthy leaves form two separate groups, which
matches the two photo backgrounds in the dataset — the largest variation is partly photographic, not disease.

## 3.4 Unit 3 – Algorithm Theory and Mathematical Formulation

**PCA.** With standardised data matrix **X** (N × D) and covariance **S** = (1/N)**X**ᵀ**X**:

  **S u**_i = λ_i **u**_i,  projection **z** = **U**_Mᵀ**x**,  variance kept = Σ_{i≤M} λ_i / Σ_i λ_i        (3.1)

**k-means.** Minimises the within-cluster sum of squares (inertia):

  J = Σ_n Σ_k r_{nk} ‖**x**_n − μ_k‖²,  alternating assignment r_{nk} and update μ_k = mean of cluster k     (3.2)

  Silhouette s(n) = (b_n − a_n)/max(a_n, b_n)                                                         (3.3)

**Gaussian mixture and EM.**

  p(**x**) = Σ_k π_k 𝒩(**x** | μ_k, Σ_k)                                                             (3.4)

  E-step: γ(z_{nk}) = π_k 𝒩(**x**_n | μ_k, Σ_k) / Σ_j π_j 𝒩(**x**_n | μ_j, Σ_j)                        (3.5)

  M-step: N_k = Σ_n γ(z_{nk}), μ_k = (1/N_k) Σ_n γ(z_{nk}) **x**_n,
  Σ_k = (1/N_k) Σ_n γ(z_{nk})(**x**_n − μ_k)(**x**_n − μ_k)ᵀ, π_k = N_k/N                             (3.6)

  BIC = −2 ln L̂ + p ln N   (p = number of free parameters; lower is better)                           (3.7)

EM never decreases the log-likelihood, which is checked in the results.

**Cluster quality** (labels used only here): Adjusted Rand Index (ARI), Normalised Mutual Information (NMI) and
purity = (1/N) Σ_k max_j |cluster_k ∩ class_j|.

## 3.5 Unit 3 Proposed System Architecture / Workflow

```
55 features (9,302 train) ─► StandardScaler ─► PCA ──► variance curve; LR on top-k PCs (5-fold CV)
                                                 │
                                                 └─► 30 PCs ─► k-means (K = 2..12: elbow, silhouette)
                                                           └─► GMM + EM (K = 1..12: BIC; EM trace)
                                                                     │
                                 labels used ONLY now ─► ARI, NMI, purity vs random assignment
```
**Figure 3.6: Unit 3 workflow**

## 3.6 Unit 3 – Results and Performance Evaluation

**Table 3.1: PCA variance kept and logistic-regression accuracy on the top-k components (5-fold CV)**

| Components | Variance kept (%) | LR CV accuracy (%) |
|---|---|---|
| 2 | 26.8 | 47.4 ± 0.9 |
| 5 | 46.8 | 62.0 ± 1.7 |
| 10 | 65.2 | 68.4 ± 1.5 |
| 20 | 84.9 | 75.2 ± 1.6 |
| 30 | 95.3 | 80.4 ± 1.4 |
| 55 (all) | 100.0 | 83.1 ± 1.3 |

*Interpretation:* 17 components keep 80% and 30 keep 95% of the variance — the data are far from 55-dimensional.
However, accuracy keeps rising up to all 55 features: the low-variance directions still carry disease information,
so PCA is useful for visualisation and clustering but not for compressing the classifier input here.

**[Insert: Figure 3.2 – `02_pca/2_accuracy_vs_components.png`]**
**Figure 3.2: Accuracy versus number of principal components**
*Interpretation:* the curve rises steeply up to 10 components and keeps improving slowly afterwards.

**Table 3.2: k-means inertia and silhouette (on 30 PCs)**

| K | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| Inertia (×10³) | 403.6 | 373.5 | 348.9 | 330.4 | 314.3 | 293.3 | 270.1 | 253.6 |
| Silhouette | 0.16 | 0.13 | 0.14 | 0.15 | 0.16 | 0.14 | 0.15 | 0.15 |

**[Insert: Figure 3.3 – `03_kmeans/1_elbow_silhouette.png`]**
**Figure 3.3: k-means elbow and silhouette curves**
*Interpretation:* inertia falls smoothly with no clear elbow and the silhouette stays near 0.15, meaning the
leaves do not form well-separated spherical clusters.

**Table 3.3: GMM BIC for 1 to 12 components (lower is better)**

| K | 1 | 3 | 5 | 7 | 8 | 9 | 10 | **11** | 12 |
|---|---|---|---|---|---|---|---|---|---|
| BIC (×10³) | 836.0 | 365.0 | 220.0 | 144.5 | 136.9 | 140.6 | 141.4 | **104.3** | 107.6 |

**[Insert: Figure 3.4 – `04_gmm_em/1_bic_and_em.png`]**
**Figure 3.4: BIC per number of components and the EM log-likelihood per iteration**
*Interpretation:* BIC prefers 11 components rather than 5: the data contain more natural groups than diseases
(e.g. two healthy groups from two backgrounds). EM increased the average log-likelihood at every iteration
(from −21.8 to −10.6), as the theory guarantees.

**Table 3.4: Clustering into K = 5 groups without labels (on 30 PCs)**

| Method | ARI | NMI | Purity (%) |
|---|---|---|---|
| k-means | 0.22 | 0.25 | 59 |
| GMM (EM) | 0.21 | 0.25 | 58 |
| Random assignment (reference) | 0.00 | 0.00 | 46 |

**[Insert: Figure 3.5 – `03_kmeans/2_clusters_vs_classes.png`]**
**Figure 3.5: Contingency of clusters versus true classes**
*Interpretation:* the clusters are clearly better than random (ARI 0.22 vs 0.00) but far from the classes. Healthy
leaves cluster best (84% fall into one GMM cluster); the three brown-spot diseases mix together.

## 3.7 Unit 4 – Sequential Dataset Description

No separate time-series dataset was used. Sequences are **derived from the PlantVillage images**: each image row
is read from left to right like a sentence, and the hidden state of each pixel is the tissue type. The ordering is
spatial (pixel column), not time.

**Table 3.5: Sequential dataset summary**

| Property | Value |
|---|---|
| Source | PlantVillage tomato images (Section 2.2) |
| Sequence identifier | (image, row index) |
| "Time" index | pixel column t = 1 … 128 |
| Sampling rate | 1 pixel; images down-sampled from 256 to 128 × 128 for speed |
| Sequence length | T = 128 (fixed) |
| Hidden states | 3: background, healthy leaf, lesion |
| Observation | pixel colour quantised into 18 hue × 8 saturation × 4 value = 576 symbols |
| Training data | 75 leaves (15 per class) = 9,600 rows = 1,228,800 labelled pixels |
| Test data | 40 held-out leaves (8 per class) = 5,120 rows |
| Missing positions | none (every row is complete) |
| State labels | from the HSV-threshold segmentation of Chapter 2 (not hand-drawn) |

*Limitations:* the "ground truth" states are produced by our own segmentation rule, so evaluation measures
agreement with that rule rather than absolute accuracy; the sequences are spatial, so the Markov assumption means
"neighbouring pixels are usually the same tissue".

## 3.8 Unit 4 – Sequential Preprocessing and EDA

- **Order preserved:** pixels inside a row are never shuffled; each row is one independent sequence.
- **Boundaries:** sequences never cross rows or images. Training and test leaves come from the training and
  held-out sets of Chapter 2, so no leaf contributes to both (no leakage between sequences of the same leaf).
- **Quantisation:** the 32 coarse HSV bins of the classification features cannot separate beige background from
  brown lesions (both are low hue), so a finer 576-symbol code with 8 saturation steps is used.
- **Parameter estimation from training sequences only:** π, A and B are counted on training rows only, with
  Laplace (+1) smoothing to avoid zero probabilities for unseen colours.
- **EDA:** the empirical transition matrix (Table 3.6) shows strong self-transitions — long runs of the same
  tissue — which is exactly the structure an HMM exploits. Lesion runs are the shortest (self-transition 0.79),
  matching the small size of disease spots.

**Table 3.6: Estimated HMM transition matrix A (row = from, column = to)**

| from \ to | background | healthy leaf | lesion |
|---|---|---|---|
| background | 0.99 | 0.01 | 0.00 |
| healthy leaf | 0.02 | 0.97 | 0.01 |
| lesion | 0.05 | 0.16 | 0.79 |

*Interpretation:* expected run length is 1/(1 − a_kk): about 100 pixels for background, 33 for healthy tissue and
5 for lesions. A lesion most often returns to healthy tissue (0.16), consistent with spots surrounded by leaf.

## 3.9 Unit 4 – Mathematical Formulation

States z_t ∈ {1, 2, 3}, observations x_t ∈ {1, …, 576}, parameters θ = (π, A, B):

  π_k = p(z_1 = k),  A_{jk} = p(z_t = k | z_{t−1} = j),  B_{km} = p(x_t = m | z_t = k)                 (3.8)

  Joint: p(**x**, **z** | θ) = π_{z_1} B_{z_1 x_1} ∏_{t=2}^{T} A_{z_{t−1} z_t} B_{z_t x_t}                   (3.9)

**Supervised MLE (counts with Laplace smoothing):**

  A_{jk} = (n_{jk} + 1) / Σ_k′ (n_{jk′} + 1),  B_{km} = (n_{km} + 1) / Σ_m′ (n_{km′} + 1)                (3.10)

**Forward (scaled):**  α̂_1(k) = π_k B_{k x_1} / c_1;  α̂_t(k) = B_{k x_t} Σ_j α̂_{t−1}(j) A_{jk} / c_t     (3.11)

  Log-likelihood of a row: ln p(x_1 … x_T) = Σ_t ln c_t                                               (3.12)

**Backward:**  β̂_T(k) = 1;  β̂_t(j) = Σ_k A_{jk} B_{k x_{t+1}} β̂_{t+1}(k) / c_{t+1}                      (3.13)

**Posterior (forward–backward):** γ_t(k) = p(z_t = k | **x**) = α̂_t(k) β̂_t(k)                          (3.14)

**Viterbi (log space):**  ω_t(k) = ln B_{k x_t} + max_j [ω_{t−1}(j) + ln A_{jk}],  then back-tracking   (3.15)

*Inference goals:* γ gives the most probable state of each pixel and its uncertainty; Viterbi gives the single most
probable whole path, which is spatially consistent. *Assumptions:* first-order Markov states along a row,
observations conditionally independent given the state, and stationary parameters for all rows and leaves.
Scaling by c_t prevents numerical underflow over 128 steps.

## 3.10 Unit 4 Proposed System Architecture / Workflow

```
Leaf images ─► resize 128×128 ─► segmentation → state per pixel (0 bg / 1 healthy / 2 lesion)
     │                └─► HSV quantisation → symbol per pixel (576)
     ▼
split by LEAF (75 train / 40 test, from the Chapter 2 split) ─► each row = 1 sequence of 128
     ▼
train: count π, A, B (+1 smoothing) ─► decode test rows:
     • per-pixel colour only (no HMM)   • forward–backward γ   • Viterbi path
     ▼
compare with reference: pixel agreement, lesion recall / precision, state changes per row
```
**Figure 3.7: Unit 4 workflow**

## 3.11 Results and Performance Evaluation

### Unit 3 summary
(see Tables 3.1–3.4) – PCA: 30/55 components for 95% variance; clustering ARI 0.22, NMI 0.25, purity 59%
(random 0.00 / 0.00 / 46%); BIC optimum K = 11; EM monotone.

### Unit 4 results

**Table 3.7: HMM decoding of 40 held-out leaves (5,120 rows), compared with the reference segmentation**

| Decoder | Pixel agreement (%) | Lesion recall (%) | Lesion precision (%) | State changes per row |
|---|---|---|---|---|
| Per-pixel colour only (no HMM) | 93.08 | 34.45 | 71.99 | 3.97 |
| Forward–backward (argmax γ) | **95.12** | **97.19** | 37.07 | 2.54 |
| Viterbi (best path) | **95.12** | 96.99 | 37.10 | **2.48** |
| Reference segmentation | 100 | 100 | 100 | 2.85 |

*Interpretation:* adding the Markov chain raises pixel agreement from 93.1% to 95.1% and reduces noisy state
changes from 3.97 to 2.48 per row, close to the 2.85 of the reference. Lesion recall rises from 34% to 97%:
context lets the HMM recognise lesion pixels whose colour alone is ambiguous. The cost is lower lesion precision
(37%): the HMM extends lesion runs into neighbouring discoloured tissue. Forward–backward and Viterbi give almost
identical maps, showing that the posterior is sharply peaked.

**[Insert: Figure 3.8 – `14_hmm_viterbi_forward_backward/1_viterbi_and_forward_backward_leaves.png`]**
**Figure 3.8: Reference segmentation, per-pixel decoding, forward–backward and Viterbi maps of held-out leaves**
*Interpretation:* the HMM maps are smooth, without the salt-and-pepper speckle of per-pixel decoding.

**[Insert: Figure 3.9 – `14_hmm_viterbi_forward_backward/2_one_row_forward_backward_viterbi.png`]**
**Figure 3.9: One pixel row – posterior probabilities γ_t(k) and the Viterbi path**
*Interpretation:* posterior uncertainty is concentrated at tissue boundaries; elsewhere one state has probability
close to 1.

**Table 3.8: Unit 3 and Unit 4 methods on the pixels of one leaf (`tomato_early_blight.jpg`, 65,536 pixels)**

| Method | Result |
|---|---|
| PCA (7 colour values per pixel) | 3 of 7 components keep 95% of the variance (PC1 76.2%) |
| k-means | best silhouette at K = 2 (0.62); with K = 3: lesion = 29.8% of the leaf |
| GMM + EM (K = 3, from k-means centres) | log-likelihood increased in all 35 iterations; lesion = 22.0% of the leaf; 0.8% unsure pixels |
| HMM (transitions from k-means labels) | Viterbi relabelled 7.2% of pixels; state changes in row 231: k-means 22 → Viterbi 5; lesion = 26.8% of the leaf |

*Interpretation:* on a single image, unsupervised colour clustering separates background, tissue and lesion well,
and the HMM removes speckle from the clustering.

## 3.12 Discussion, Findings and Limitations

- **PCA** showed strong redundancy (30 of 55 components for 95%) but also that discarding low-variance directions
  loses accuracy (80.4% vs 83.1%) — variance is not the same as class relevance.
- **Clustering** finds real structure (well above random) but not the disease structure: the largest differences
  between leaves are lighting, background and leaf age. BIC choosing 11 components confirms sub-groups.
- **HMM state definition:** three tissue states are natural for a leaf row, and the estimated transitions are
  physically sensible. The first-order chain along rows ignores vertical neighbours — this is addressed by the 2-D
  Markov random field in Unit 5 (Section 4.6).
- **Sequential validation:** split by leaf, parameters learned only on training rows; but the reference labels are
  rule-based, and the evaluation set is small (40 leaves), so results show relative improvement, not absolute
  segmentation accuracy.
- The over-extension of lesions (low precision) suggests adding a fourth "border/yellowing" state or learning
  emissions with Baum–Welch.

## 3.13 Chapter Conclusion

Unit 3 showed that the 55 features are highly redundant but that unsupervised methods recover only part of the
disease structure (ARI 0.22), which justifies supervised learning for diagnosis. Unit 4 showed that modelling
spatial order with an HMM makes pixel labelling more consistent (95.1% agreement, 37% fewer state changes, lesion
recall 97%). The two units address different questions and their numbers are not compared with each other.

---

# CHAPTER 4 – ADVANCED PREDICTIVE MODELLING AND ENSEMBLE LEARNING (UNIT 5)

## 4.1 Overview of Allotted Unit 5 Topics

1. **Tree-based models and ensembles:** CART decision tree, bagging/random forest, gradient boosting.
2. **Combining models:** AdaBoost (SAMME) from scratch, committees (hard and soft voting), stacking.
3. **Neural networks:** multilayer perceptron with error back-propagation, gradient check, Adam, weight decay.
4. **Kernel methods and sparse kernel machines:** SVM dual representation, Gaussian process regression with ARD,
   relevance vector machine from scratch.
5. **Sampling and graphical models:** Metropolis–Hastings MCMC, Gibbs sampling in an Ising Markov random field,
   Bayesian network with exact and sampled inference.
6. **Application on a new dataset:** chlorophyll (SPAD) regression with random forest and Gaussian process versus
   linear (Unit 2) baselines.

Key formulas: CART Gini impurity G = 1 − Σ_k p_k²; boosting as an additive model F_M(**x**) = Σ_m α_m h_m(**x**);
committee prediction y_COM = (1/M) Σ_m y_m(**x**) (detailed in Section 4.5).

## 4.2 Dataset Description and Preprocessing

**Original dataset (classification):** the same 55 features, 9,302 training and 2,326 test leaves as Chapter 2;
features standardised with training statistics only. Additional Unit 5 preprocessing:
- Neural network: standardised inputs, one-hot targets.
- SVM dual analysis: a balanced subsample of 1,500 training leaves (300 per class), binary healthy/diseased target.
- Lesion-severity regression (GP, RVM, SVR): target √(lesion %) (less skewed than lesion %), predicted from 12
  colour and texture features **without** the lesion mask; 600 training and 400 test leaves (stratified).
- Bayesian network: lesion fraction, yellowing and texture contrast discretised into three levels (low / medium /
  high) at the training-set tertiles (yellowing = share of pixels in the yellow hue bins).
- MCMC: 40 leaves per class of the binary target (80 leaves) and two features (lesion fraction, GLCM contrast), small on purpose
  so the posterior is wide and visible.

**New dataset (chlorophyll regression): ArabicaLens [4]**
- **Source:** Radhakrishnan, Joseph and Kumar, Zenodo, doi:10.5281/zenodo.19447467, licence CC BY 4.0.
- **Records:** 425 photographs of coffee (*Coffea arabica*) leaves on a white background.
- **Target:** SPAD value — the mean of five SPAD-meter readings per leaf (range 6–70 SPAD).
- **Features:** the same 55 features from the same segmentation pipeline; four feature sets compared: RGB means
  (3, as in reference [2]), colour (6), colour + HSV histogram (38) and all 55.
- **Relevance:** the tomato images have no chlorophyll measurements, so a model can only be calibrated on a
  dataset with paired photo–meter readings.
- **Preprocessing:** no missing values; split 340 training / 85 held-out test leaves; scaling inside each CV fold;
  10-fold CV for model selection; the test set is used once.
- **Measurement noise:** the meter itself varies by 3.2 SPAD between repeated readings of the same leaf — a floor
  for achievable error.

## 4.3 EDA and Feature Analysis

The main EDA is in Section 2.4. Additional analyses relevant to Unit 5:
- **Feature importance (ablation, Tables 2.8–2.9):** colour histogram is the most valuable group, texture second;
  shape adds little. This explains why tree ensembles, which select informative features at each split, work well.
- **Collinearity (Table 2.5):** VIFs of 13–807 explain why least squares on all 55 features fails for chlorophyll
  (CV R² 0.54) while ridge, random forest and GP remain stable.
- **Chlorophyll vs greenness (Figure 4.18):** SPAD rises with greenness and falls with red/blue brightness, but the
  relation is non-linear and saturates for very dark leaves — a reason to try non-linear models.

## 4.4 Proposed Workflow / Architecture

```
                  55 standardised features (train 9,302 / test 2,326)
                                     │
   ┌──────────────┬──────────────────┼──────────────────┬─────────────────────────┐
   ▼              ▼                  ▼                  ▼                         ▼
 Trees &       Neural network     Combining         Kernel methods          Sampling & graphical
 ensembles     55→32 tanh→5       AdaBoost (300     SVM dual (support       MH-MCMC on Bayesian LR
 CART, RF,     backprop, grad     stumps), hard/    vectors vs C); GP,      Gibbs in Ising MRF
 gradient      check, Adam,       soft voting,      RVM, SVR for lesion     Bayesian network:
 boosting      weight decay       stacking          severity                exact vs sampled
   │              │                  │                  │                         │
   └──────────────┴────── grid search / CV / verification against sklearn ───────┘
                                     ▼
          held-out test (same 2,326 leaves as Chapter 2) ─► comparison with PBL 1 models
                                     ▼
          ArabicaLens: 55 features ─► OLS / ridge / Bayesian ridge / polynomial / RF / GP
                    ─► 10-fold CV ─► best on 85-leaf test ─► SPAD ± 95% error bar
```
**Figure 4.1: Unit 5 workflow**

*Stages:* the same features and test set as PBL 1 are used so that results are directly comparable; every
from-scratch implementation is verified against scikit-learn or a numerical check before results are reported.

## 4.5 Algorithm Theory and Mathematical Formulation

**CART decision tree.** Each split chooses the feature and threshold that maximise the decrease in Gini impurity

  G(node) = 1 − Σ_k p_k²                                                                             (4.1)

**Random forest (bagging).** B trees on bootstrap samples with random feature subsets; prediction by majority vote.
Averaging reduces variance.

**Gradient boosting.** Additive model fitted stage-wise to the negative gradient of the loss:

  F_m(**x**) = F_{m−1}(**x**) + ν h_m(**x**),  h_m ≈ −∂L(t, F)/∂F |_{F_{m−1}}                          (4.2)

Each tree corrects the errors of the ensemble so far, reducing bias. Settings: learning rate ν = 0.1, max 31 leaf
nodes per tree (histogram gradient boosting).

**AdaBoost (SAMME, K classes).** Sample weights w_n, weak learner h_m (decision stump):

  ε_m = Σ_n w_n I(h_m(**x**_n) ≠ t_n) / Σ_n w_n,  α_m = ln((1 − ε_m)/ε_m) + ln(K − 1)                (4.3)

  w_n ← w_n exp(α_m I(h_m(**x**_n) ≠ t_n)),  H(**x**) = argmax_k Σ_m α_m I(h_m(**x**) = k)                (4.4)

The ln(K − 1) term lets a weak learner with error up to 1 − 1/K = 0.8 still contribute.

**Committees and stacking.**

  y_COM(**x**) = (1/M) Σ_m y_m(**x**),  E_COM ≤ E_AV = (1/M) Σ_m E_m                                   (4.5)

Stacking learns a logistic-regression meta-model on out-of-fold (5-fold) member probabilities.

**Multilayer perceptron.** 55 inputs → 32 tanh hidden units → 5 softmax outputs (1,957 weights):

  **h** = tanh(**W**₁**x** + **b**₁),  **y** = softmax(**W**₂**h** + **b**₂)                                  (4.6)

  E = −Σ_n Σ_k t_{nk} ln y_{nk} + (λ/2)(‖**W**₁‖² + ‖**W**₂‖²)                                           (4.7)

Back-propagation: δ⁽²⁾ = **y** − **t**;  δ⁽¹⁾ = (1 − **h**²) ⊙ (**W**₂ᵀ δ⁽²⁾);  ∂E/∂**W**₂ = δ⁽²⁾**h**ᵀ, ∂E/∂**W**₁ = δ⁽¹⁾**x**ᵀ   (4.8)

Settings: Adam, learning rate 3 × 10⁻³, mini-batch 128, 60 epochs, weight decay 10⁻⁴. Gradients are verified
against central finite differences (ε = 10⁻⁶).

**SVM dual.** y(**x**) = Σ_{n∈SV} a_n t_n k(**x**, **x**_n) + b                                          (4.9)
Only support vectors (a_n > 0) contribute; C controls how many.

**Gaussian process regression.** With kernel matrix **K** + σ²**I** and ARD-RBF kernel
k(**x**, **x**′) = θ₀ exp(−½ Σ_d (x_d − x′_d)²/ℓ_d²):

  m(**x**_*) = **k**_*ᵀ(**K** + σ²**I**)⁻¹**t**,  σ²(**x**_*) = k(**x**_*, **x**_*) − **k**_*ᵀ(**K** + σ²**I**)⁻¹**k**_* + σ²   (4.10)

Hyper-parameters (θ₀, ℓ_d, σ²) maximise the log marginal likelihood; a very large ℓ_d switches input d off (ARD).

**Relevance vector machine.** y(**x**) = Σ_n w_n k(**x**, **x**_n), prior w_n ~ 𝒩(0, α_n⁻¹), re-estimated by

  γ_i = 1 − α_i Σ_ii,  α_i ← γ_i / m_i²,  β⁻¹ ← ‖**t** − Φ**m**‖² / (N − Σ_i γ_i)                          (4.11)

Basis functions whose α_i → ∞ are pruned, giving a sparse model with error bars.

**Metropolis–Hastings.** Gaussian random-walk proposal **w**′ = **w** + 𝒩(0, s²**I**), s = 0.45:

  accept with probability min(1, p(**w**′ | 𝒟) / p(**w** | 𝒟))                                          (4.12)

30,000 samples, first 3,000 discarded as burn-in; effective sample size from the autocorrelation.

**Ising MRF and Gibbs sampling.** Labels s_i ∈ {−1, +1} (lesion / not), observed noisy labels y_i:

  E(**s**) = −β Σ_{i~j} s_i s_j − η Σ_i s_i y_i,  p(s_i = +1 | rest) = σ(2(β Σ_{j∈N(i)} s_j + η y_i))       (4.13)

β = 1, η = ½ ln((1 − 0.15)/0.15) = 0.87 (exact log-odds for 15% label noise), 40 sweeps, 10 burn-in.

**Bayesian network.** D → L, D → Y, D → T, L → T (Disease, Lesion, Yellowing, Texture; three levels each):

  p(D, L, Y, T) = p(D) p(L | D) p(Y | D) p(T | D, L)                                                  (4.14)

Tables are learned by counting; queries such as P(D | T = high) are answered exactly by enumeration and
approximately by rejection sampling, likelihood weighting and Gibbs sampling.

**Model selection decisions.** Tree models were tuned by grid search with 5-fold stratified CV (decision tree depth
10; random forest depth 20; gradient boosting ν = 0.1, 31 leaves). Hidden-unit count was studied from 1 to 128.
SVM C was swept from 0.01 to 1000. The chlorophyll model was chosen by 10-fold CV over 6 models × 4 feature sets.

## 4.6 Results and Performance Evaluation

### 4.6.1 Tree-based models and ensembles

**Table 4.1: Single tree versus ensembles (same features and test set)**

| Model | Best parameters | CV accuracy (%) | Test accuracy (%) | Balanced acc. (%) | Macro F1 (%) |
|---|---|---|---|---|---|
| Decision tree (CART) | depth 10 | 75.8 ± 1.4 | 84.8 | 79.6 | 79.8 |
| Random forest (bagging) | depth 20 | 89.4 ± 0.8 | 94.1 | 90.4 | 91.4 |
| **Gradient boosting** | ν = 0.1, 31 leaves | **92.2 ± 1.0** | **96.3** | **94.1** | **94.3** |

*Interpretation:* bagging adds 13.6 CV points over a single tree by reducing variance; boosting adds a further
2.8 points by reducing bias. Gradient boosting is the best model of the whole project.

**[Insert: Figure 4.2 – `05_decision_trees/1_depth_vs_overfitting.png`]**
**Figure 4.2: Training and CV accuracy versus tree depth**
*Interpretation:* training accuracy keeps rising with depth while CV accuracy levels off — a single tree over-fits.

**[Insert: Figure 4.3 – `07_boosting/2_bagging_vs_boosting.png`]**
**Figure 4.3: Bagging versus boosting as trees are added**
*Interpretation:* boosting continues improving with more trees, while the random forest saturates early.

### 4.6.2 Neural network

**Table 4.2: Neural network results (held-out test, 2,326 leaves)**

| Model | Test accuracy (%) | Train accuracy (%) | Fit time (s) |
|---|---|---|---|
| MLP from scratch (NumPy, 32 tanh units) | **93.21** | 97.89 | 5.5 |
| scikit-learn MLPClassifier (same architecture) | 92.99 | 97.83 | 5.2 |
| Logistic regression (no hidden layer) | 89.29 | 90.10 | 2.9 |

*Verification:* worst relative error between back-propagation and finite-difference gradients = 2 × 10⁻⁶.

**Table 4.3: Accuracy versus number of hidden units (40 epochs)**

| Hidden units | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| Train (%) | 68.2 | 81.1 | 88.7 | 92.4 | 95.1 | 97.1 | 98.8 | 99.7 |
| Test (%) | 66.8 | 80.9 | 87.6 | 90.8 | 92.0 | 93.6 | 94.2 | 94.3 |

**[Insert: Figure 4.4 – `15_neural_networks/1_training_curves.png`]**
**Figure 4.4: Training loss and accuracy per epoch**
*Interpretation:* loss decreases smoothly; train and test curves stay close, so weight decay controls over-fitting.

**[Insert: Figure 4.5 – `15_neural_networks/2_accuracy_vs_hidden_units.png`]**
**Figure 4.5: Accuracy versus hidden units**
*Interpretation:* beyond 32 units training accuracy approaches 100% while test accuracy levels off at ≈ 94% —
the gap is over-fitting.

**[Insert: Figure 4.6 – `15_neural_networks/3_decision_boundaries_2d.png`]**
**Figure 4.6: Decision boundaries in two dimensions — logistic regression (straight) vs neural network (curved)**
*Interpretation:* the hidden layer learns non-linear features, which is why the network beats logistic regression
by 3.9 points.

### 4.6.3 Combining models

**Table 4.4: Committees, stacking and AdaBoost (held-out test)**

| Model | Kind | Test accuracy (%) |
|---|---|---|
| Logistic regression | member | 89.29 |
| KNN (k = 7) | member | 87.70 |
| LDA | member | 85.04 |
| Decision tree (depth 10) | member | 84.82 |
| Gaussian Naive Bayes | member | 71.07 |
| Committee – hard (majority) vote | committee | 89.55 |
| Committee – soft vote | committee | 88.69 |
| **Stacking (LR on members' outputs)** | committee | **91.83** |
| Single decision stump | weak learner | 52.1 |
| AdaBoost from scratch (300 stumps) | boosting | 83.96 |

*Interpretation:* the average member error is 16.4% and the soft-vote committee error 11.3%, confirming
E_COM ≤ E_AV (4.5). A plain vote is pulled down by the weak Naive Bayes member; stacking learns how much to trust
each member and beats every single member. AdaBoost turns 52.1% stumps into 84.0% and gives the same predictions
as scikit-learn's AdaBoost on 100% of test leaves.

**[Insert: Figure 4.7 – `16_combining_models/1_adaboost_rounds.png`]**
**Figure 4.7: AdaBoost weighted error, α and train/test error per round**
*Interpretation:* test error falls from 47.9% at round 1 to 16.0% at round 300 without rising, a typical
boosting behaviour; weighted errors stay below 0.8 so every α_m is positive.

**[Insert: Figure 4.8 – `16_combining_models/2_committee_vs_members.png`]**
**Figure 4.8: Committees compared with their members**
*Interpretation:* stacking is the only combiner above the best member (logistic regression).

### 4.6.4 Kernel methods

**Table 4.5: SVM (RBF) support vectors versus C (healthy vs diseased, 1,500 training leaves)**

| C | 0.01 | 0.1 | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|---|
| Support vectors | 611 | 457 | 229 | 162 | 144 | 144 |
| % of leaves | 40.7 | 30.5 | 15.3 | 10.8 | 9.6 | 9.6 |
| Test accuracy (%) | 86.3 | 98.1 | 99.1 | 99.2 | 99.2 | 99.2 |

*Verification:* the decision function rebuilt by hand from the dual coefficients (4.9) matches scikit-learn to
10⁻¹³ while using only 162 of 1,500 training leaves.

**[Insert: Figure 4.9 – `17_kernel_methods/1_svm_support_vectors_vs_C.png`]**
**Figure 4.9: Support vectors and accuracy versus C**
*Interpretation:* a larger C penalises margin violations, so fewer points become support vectors and the model
becomes sparser without losing accuracy.

**Table 4.6: Lesion-severity regression, √(lesion %) from 12 colour + texture features (400 test leaves)**

| Model | Basis vectors kept | R² | RMSE | 95% interval coverage (%) |
|---|---|---|---|---|
| **Gaussian process (ARD RBF + noise)** | 600 | **0.84** | **0.64** | 94.3 |
| Relevance vector machine (from scratch) | **35** | 0.76 | 0.79 | 90.5 |
| Support vector regression | 447 | 0.80 | 0.72 | – |

**[Insert: Figure 4.10 – `17_kernel_methods/2_gaussian_process_regression.png`]**
**Figure 4.10: GP predictions with 95% error bars versus true severity**
*Interpretation:* 94.3% of test leaves fall inside the 95% interval, so the GP error bars are honest; ARD switched
off three GLCM inputs.

**[Insert: Figure 4.11 – `17_kernel_methods/3_rvm_vs_svr_sparsity.png`]**
**Figure 4.11: Relevance vectors versus support vectors**
*Interpretation:* the RVM keeps 35 training leaves versus 447 for SVR — 13 times sparser — at a small loss in R².

### 4.6.5 Sampling methods and graphical models

**Table 4.7: Metropolis–Hastings versus Laplace approximation (Bayesian LR, 80 leaves, 2 features)**

| Parameter | MCMC mean | MCMC std | Laplace mean | Laplace std | Effective samples |
|---|---|---|---|---|---|
| bias | 0.81 | 0.37 | 0.75 | 0.37 | 917 |
| w (lesion fraction) | 1.85 | 0.68 | 1.69 | 0.70 | 575 |
| w (GLCM contrast) | −1.02 | 0.24 | −0.97 | 0.23 | 3,049 |

*Interpretation:* with an acceptance rate of 29% the sampler agrees with the Laplace approximation in both means
and standard deviations, which validates the approximation used in Chapter 2.

**[Insert: Figure 4.12 – `18_sampling_graphical_models/1_mcmc_vs_laplace.png`]**
**Figure 4.12: MCMC samples and the Laplace Gaussian**
*Interpretation:* the sample cloud and the Laplace ellipse overlap; the slight skew of the lesion weight is the
only visible difference.

**Gibbs sampling in an Ising MRF:** 15% of the lesion labels of a real leaf were flipped at random. After 40 Gibbs
sweeps the wrong-pixel rate fell from **15.0% to 2.1%**.

**[Insert: Figure 4.13 – `18_sampling_graphical_models/2_gibbs_mrf_lesion_denoising.png`]**
**Figure 4.13: True lesion mask, noisy mask, posterior P(lesion) and cleaned mask**
*Interpretation:* the "neighbours agree" prior removes isolated noise while keeping real lesion shapes — the 2-D
extension of the row-wise HMM of Unit 4.

**Table 4.8: Bayesian network – P(Disease | texture contrast = high), exact versus sampled (%; 30,000 samples)**

| Disease | Exact | Rejection | Likelihood weighting | Gibbs |
|---|---|---|---|---|
| Early blight | 9.25 | 9.40 | 9.20 | 9.27 |
| Late blight | 20.02 | 20.98 | 19.86 | 20.42 |
| Septoria leaf spot | 21.12 | 20.98 | 21.43 | 21.18 |
| Yellow leaf curl virus | 24.27 | 24.10 | 23.54 | 24.36 |
| Healthy | 25.35 | 24.53 | 25.97 | 24.77 |

**[Insert: Figure 4.14 – `18_sampling_graphical_models/3_bayes_net_inference.png`]**
**Figure 4.14: Network structure, posterior estimates and error versus number of samples**
*Interpretation:* all three samplers are within one percentage point of the exact answer; the maximum error falls
from 0.07–0.09 at 100 samples to ≤ 0.01 at 30,000, at the expected 1/√N rate.

### 4.6.6 Chlorophyll (SPAD) estimation on ArabicaLens

**Table 4.9: 10-fold CV on 340 coffee leaves (top models and baselines)**

| Features | Model | CV R² | CV RMSE (SPAD) | CV MAE (SPAD) |
|---|---|---|---|---|
| colour + HSV (38) | **Random forest** | **0.89** | **5.81** | **4.14** |
| colour (6) | Gaussian process | 0.89 | 5.81 | 4.26 |
| colour (6) | Polynomial (deg 2) + ridge | 0.89 | 5.88 | 4.33 |
| colour (6) | Ridge | 0.87 | 6.27 | 4.59 |
| RGB (3) | Least squares (reference [2]) | 0.86 | 6.45 | 4.67 |
| all 55 | Least squares | 0.54 | 11.82 | 5.43 |

**Table 4.10: Held-out test on 85 coffee leaves**

| Model | R² | RMSE (SPAD) | MAE (SPAD) |
|---|---|---|---|
| **Random forest on colour + HSV histogram (chosen by CV)** | **0.91** | **5.51** | **4.23** |
| Reference [2] baseline: least squares on mean R, G, B | 0.86 | 6.65 | 4.79 |
| No model (always the training mean) | 0.00 | 18.04 | 15.69 |

**[Insert: Figure 4.16 – `19_chlorophyll_spad/1_predicted_vs_measured.png`]**
**Figure 4.16: Predicted versus measured SPAD on the held-out leaves**
*Interpretation:* points lie close to the diagonal; very pale and very dark leaves are pulled toward the middle
because few training leaves are that extreme.

**[Insert: Figure 4.17 – `19_chlorophyll_spad/2_model_comparison.png`]**
**Figure 4.17: CV error of every model and feature set**
*Interpretation:* non-linear models (random forest, GP, polynomial) beat linear least squares, and OLS on all 55
collinear features collapses (R² 0.54), as the VIF analysis predicted.

**[Insert: Figure 4.18 – `19_chlorophyll_spad/3_greenness_vs_spad_and_error_bars.png`]**
**Figure 4.18: Greenness versus SPAD, and GP 95% error bars**
*Interpretation:* 95% of held-out leaves fall inside the GP interval. On a tomato leaf (outside the training
distribution) the error bar becomes about four times wider, so the system flags the estimate as untrustworthy.

## 4.7 Comparison with Previous Models

All models below use the **same task (5-class disease classification), the same 55 features, the same 2,326-leaf
held-out test set and the same metrics**, so they are directly comparable.

**Table 4.11: All classifiers on the same held-out test set**

| Rank | Model | PBL / Unit | Test accuracy (%) | Macro F1 (%) |
|---|---|---|---|---|
| 1 | **Gradient boosting** | PBL 3 / 5 | **96.30** | **94.34** |
| 2 | Random forest | PBL 3 / 5 | 94.07 | 91.44 |
| 3 | SVM (RBF) | PBL 1 / 2 | 93.25 | 91.03 |
| 4 | MLP (from scratch) | PBL 3 / 5 | 93.21 | – |
| 5 | Stacking | PBL 3 / 5 | 91.83 | – |
| 6 | Committee (hard vote) | PBL 3 / 5 | 89.55 | – |
| 7 | Logistic regression | PBL 1 / 2 | 89.29 | 85.57 |
| 8 | SVM (linear) | PBL 1 / 2 | 88.52 | 83.90 |
| 9 | KNN | baseline | 87.79 | 82.97 |
| 10 | LDA | PBL 1 / 2 | 85.04 | 79.45 |
| 11 | Decision tree | PBL 3 / 5 | 84.82 | 79.79 |
| 12 | AdaBoost (stumps) | PBL 3 / 5 | 83.96 | – |
| 13 | Gaussian Naive Bayes | PBL 1 / 1 | 71.07 | 60.47 |

**Table 4.12: Per-class report of the best model (gradient boosting)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Early blight | 0.882 | 0.825 | 0.853 | 200 |
| Late blight | 0.912 | 0.950 | 0.931 | 382 |
| Septoria leaf spot | 0.957 | 0.952 | 0.955 | 354 |
| Yellow leaf curl virus | 0.991 | 0.989 | 0.990 | 1,072 |
| Healthy | 0.987 | 0.991 | 0.989 | 318 |
| **Macro average** | 0.946 | 0.941 | 0.943 | 2,326 |

**[Insert: Figure 4.15 – `10_evaluation_metrics/1_confusion_matrix.png`]**
**Figure 4.15: Confusion matrix of gradient boosting on the held-out test set**
*Interpretation:* almost all errors are between early and late blight; healthy and yellow leaf curl leaves are
recognised at about 99%.

*Improvements and declines:* gradient boosting improves on the best PBL 1 model (RBF SVM) by 3.1 accuracy points
and 3.3 macro-F1 points, and on the Naive Bayes baseline by 25.2 points. The neural network matches the RBF SVM.
AdaBoost with stumps is weaker than full gradient boosting because a stump can use only one feature per round.
For chlorophyll, the random forest improves on the published least-squares baseline from R² 0.86 to 0.91 and
reduces RMSE from 6.65 to 5.51 SPAD.

## 4.8 Discussion, Findings and Limitations

- **Performance and generalisation:** ensembles give the largest gain; the CV standard deviation of gradient
  boosting (1.0) is small, so the gain is stable. The hidden-unit sweep and tree-depth study show where
  over-fitting starts.
- **Interpretability:** tree ensembles provide feature importances; the GP's ARD length-scales and the Bayesian
  network's tables are directly readable; the MLP is the least interpretable model.
- **Uncertainty:** GP, RVM and MCMC give error bars; the GP's coverage (94–95%) shows they are calibrated, and the
  GP widens its interval for out-of-distribution leaves.
- **Training cost (CPU):** gradient boosting 103 s for the full grid search; RBF SVM 7 s; MLP 5.5 s; GP 13 s on
  600 leaves (cost grows as N³, so GPs do not scale to all 9,302 leaves); RVM 3.5 s.
- **Correctness:** every from-scratch method was verified (MLP gradient check 2 × 10⁻⁶; AdaBoost 100% agreement
  with scikit-learn; SVM dual 10⁻¹³; MCMC vs Laplace); 102 automated tests cover the pipeline.
- **Limitations:** laboratory images only; the Bayesian network uses three coarse variables and is meant to
  demonstrate inference, not to compete as a classifier; the chlorophyll model is calibrated only for coffee and
  regresses extreme leaves toward the mean.

## 4.9 Chapter Conclusion

Unit 5 delivered the best classifier of the project (gradient boosting, 96.3% accuracy, 94.3% macro F1), a
verified from-scratch neural network (93.2%), and showed through committees and stacking that combining diverse
models reduces error. Kernel and sampling methods added calibrated uncertainty and spatial denoising, and the new
ArabicaLens dataset extended the system to chlorophyll estimation (R² 0.91, RMSE 5.5 SPAD).

---

# CHAPTER 5 – INTEGRATED DISCUSSION AND FINDINGS

**What was learned across PBL 1, 2 and 3**

1. **Features first.** Careful segmentation and 55 interpretable features made every later method work on a CPU.
   Colour histogram and texture carry most of the information; shape adds little.
2. **Assumptions matter, and the data analysis predicts them.** The correlation and VIF analysis of Unit 1
   explained the failure of Naive Bayes (71.1%) and of unregularised least squares on all features (R² 0.54), and
   motivated regularisation throughout.
3. **Model capacity ladder (same test set, comparable):** Naive Bayes 71.1% → linear models 85–89% → RBF SVM and
   MLP ≈ 93% → random forest 94.1% → gradient boosting 96.3%. Non-linear boundaries and ensembles give the main
   gains.
4. **Labels are essential for diagnosis.** Unsupervised clustering reached ARI 0.22, far below supervised
   performance, because the dominant variation in the images is lighting, background and leaf age, not disease.
5. **Spatial structure helps segmentation.** The row-wise HMM (Unit 4) and the 2-D Ising MRF (Unit 5) both use the
   assumption that neighbouring pixels agree: the HMM reduced spurious state changes from 3.97 to 2.48 per row and
   the MRF reduced label noise from 15.0% to 2.1%. (These are separate experiments with different metrics and are
   not compared numerically.)
6. **Honest uncertainty.** Bayesian logistic regression, the GP and MCMC give confidence values that match
   reality; this enables a practical "refer to expert" rule (100% accuracy on the 80% most confident leaves).

**Comparable versus non-comparable results.** Only the classifiers in Table 4.11 share the task, data split and
metrics. Clustering scores (Unit 3), HMM pixel agreement (Unit 4), lesion-severity R² and chlorophyll R² are on
different tasks or datasets and are reported separately.

**Critical findings:** gradient boosting is the best model; early blight vs late blight is the hardest
distinction; test accuracy is higher than CV accuracy because of class imbalance, so balanced accuracy and macro F1
must always be reported.

**Study limitations:** laboratory photographs only; five tomato classes; HMM reference labels generated by our own
segmentation; chlorophyll calibrated for coffee only; the 38-class run of the code has not yet been performed.

---

# CHAPTER 6 – OVERALL CONCLUSION AND FUTURE SCOPE

**Conclusion.** The project built a complete, explainable, CPU-only plant health system covering all five units.
From a single photograph it diagnoses five tomato classes with 96.3% held-out accuracy and 94.3% macro F1
(gradient boosting), screens healthy versus diseased leaves with 99.5% accuracy and calibrated confidence
(Bayesian logistic regression), measures lesion area, and estimates chlorophyll with R² 0.91 and 5.5 SPAD error on
coffee leaves. Unsupervised and sequential methods explained the structure of the data and improved pixel-level
segmentation. Every from-scratch algorithm was verified against a reference implementation, and the whole project
runs from a guided terminal demo and a Streamlit web application, with 102 automated tests.

**Future scope**

1. **More data:** run the prepared 38-class Kaggle dataset; add field photographs with complex backgrounds.
2. **Tomato chlorophyll:** collect SPAD readings of tomato leaves (template in `spad/spad_readings.csv`) to
   calibrate a tomato-specific model.
3. **Better sequential model:** learn HMM emissions with Baum–Welch, add a "border/yellowing" state, or use a 2-D
   HMM/MRF jointly with the classifier.
4. **Tuning:** Bayesian hyper-parameter optimisation for gradient boosting and the MLP; class-weighted training for
   early blight.
5. **Deployment:** a mobile application with on-device feature extraction (the features need only ≈ 4.4 ms per
   image).
6. **Interpretability:** SHAP values for the gradient-boosting model to complement the logistic-regression
   explanations.

---

## REFERENCES

[1] S. M. Kiran and D. N. Chandrappa, "Plant leaf disease detection using efficient image processing and machine
learning algorithms," *Journal of Robotics and Control*, vol. 4, no. 6, 2023, doi: 10.18196/jrc.v4i6.20342.

[2] J. Anderson *et al.*, "Image-based leaf SPAD value and chlorophyll measurement using a mobile phone,"
*Precision Agriculture*, 2026, doi: 10.1007/s11119-026-10386-x.

[3] C. M. Bishop, *Pattern Recognition and Machine Learning*. New York, NY, USA: Springer, 2006.

[4] S. Radhakrishnan, J. Joseph, and P. Kumar, "ArabicaLens: A multimodal dataset of high-resolution leaf imagery
and SPAD-based chlorophyll readings," Zenodo, 2026, doi: 10.5281/zenodo.19447467.

[5] D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of
mobile disease diagnostics," arXiv:1511.08060, 2015.

[6] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*,
vol. 12, pp. 2825–2830, 2011.

[7] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," *IEEE
Transactions on Systems, Man, and Cybernetics*, vol. SMC-3, no. 6, pp. 610–621, 1973.

[8] M.-K. Hu, "Visual pattern recognition by moment invariants," *IRE Transactions on Information Theory*, vol. 8,
no. 2, pp. 179–187, 1962.

[9] L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition,"
*Proceedings of the IEEE*, vol. 77, no. 2, pp. 257–286, 1989.

[10] J. Zhu, H. Zou, S. Rosset, and T. Hastie, "Multi-class AdaBoost," *Statistics and Its Interface*, vol. 2,
no. 3, pp. 349–360, 2009.

[11] M. E. Tipping, "Sparse Bayesian learning and the relevance vector machine," *Journal of Machine Learning
Research*, vol. 1, pp. 211–244, 2001.

---

## APPENDIX A – CODING

**Repository:** GitHub link: ______  ·  **Google Colab notebooks:** Unit 1–2: ______ · Unit 3: ______ ·
Unit 4: ______ · Unit 5: ______

Environment: Python 3.13, CPU only; `pip install -r requirements.txt`. All settings and the random seed (42) are in
`config.py`. Paste each file below in Consolas 10 pt, single spacing.

### Chapter 2 – Implementation Methodology and Algorithm Configuration

| File | Content |
|---|---|
| `config.py` | paths, seed, HSV thresholds, feature and CV settings |
| `src/features.py` | preprocessing, segmentation, the 55 features |
| `src/dataset.py` | dataset discovery, parallel extraction, cache, stratified split |
| `src/classify.py` | Naive Bayes, LR, LDA, SVMs, KNN pipelines; grid search + 5-fold CV; ablation |
| `src/bayes_logreg.py` | Bayesian logistic regression (Newton, Laplace, evidence, probit) |
| `src/evaluate.py`, `src/figures.py` | metrics, tables, graphs |

Run: `python -m src.dataset` → `python -m src.classify` → `python -m src.bayes_logreg` → `python -m src.evaluate --all`

### Chapter 3 – Unit 3: Proposed Workflow and Implementation Methodology

| File | Content |
|---|---|
| `src/unsupervised.py` | PCA, k-means, GMM + EM, BIC, ARI/NMI/purity |
| `src/leaf_unsupervised.py` | PCA, k-means, GMM and HMM on the pixels of one photo |

Run: `python -m src.unsupervised` and `python -m src.leaf_unsupervised samples/tomato_early_blight.jpg`

### Chapter 3 – Unit 4: Proposed Workflow and Implementation Methodology

| File | Content |
|---|---|
| `src/hmm.py` | HMM class: supervised fitting, scaled forward, backward, posteriors, Viterbi; evaluation |

Run: `python -m src.hmm`

Key code (from `src/hmm.py`):

```python
def viterbi(self, obs):
    logA, logB = np.log(self.A), np.log(self.B)
    omega = np.log(self.pi) + logB[:, obs[:, 0]].T          # (rows, K)
    back = np.empty(obs.shape + (len(self.pi),), dtype=int)
    for t in range(1, obs.shape[1]):
        scores = omega[:, :, None] + logA[None]              # from j to k
        back[:, t] = scores.argmax(1)
        omega = scores.max(1) + logB[:, obs[:, t]].T
    # back-track the most probable path ...
```

*(Replace this excerpt with the exact function from the file when pasting.)*

### Chapter 4 – Implementation Methodology and Hyper-parameter Configuration

| File | Content |
|---|---|
| `src/classify.py` | decision tree, random forest, gradient boosting (grid search) |
| `src/unit5.py` | MLP + backprop + gradient check; AdaBoost SAMME; committees and stacking; SVM dual; GP; RVM; Metropolis–Hastings; Ising Gibbs; Bayesian network |
| `src/chlorophyll_coffee.py` | ArabicaLens chlorophyll regression (6 models × 4 feature sets, 10-fold CV, GP error bars) |

Run: `python -m src.unit5` and `python -m src.chlorophyll_coffee`

**Hyper-parameter configuration**

| Model | Setting |
|---|---|
| Decision tree | max_depth 10, min_samples_leaf 1 (grid search) |
| Random forest | max_depth 20, min_samples_leaf 1 (grid search) |
| Gradient boosting | learning_rate 0.1, max_leaf_nodes 31 (grid search) |
| MLP | 55-32-5, tanh, Adam lr 3e-3, batch 128, 60 epochs, weight decay 1e-4 |
| AdaBoost | 300 decision stumps (depth 1), SAMME |
| Committee members | LR, LDA, Gaussian NB, KNN (k = 7), decision tree (depth 10); stacking meta-model LR on 5-fold out-of-fold probabilities |
| SVM (dual study) | RBF, γ = scale, C ∈ {0.01 … 1000}; 1,500 leaves |
| Gaussian process | constant × ARD-RBF (initial ℓ = 3) + white noise; hyper-parameters by marginal likelihood |
| RVM | RBF γ = 1/12, max 300 iterations, prune α > 10⁹ |
| SVR | RBF γ = 1/12, C = 3·std(t), ε = 0.1·std(t) |
| Metropolis–Hastings | step 0.45, 30,000 samples, 3,000 burn-in, prior α = 1 |
| Ising MRF | β = 1, η = 0.87, 40 sweeps, 10 burn-in |
| Bayesian network | D→L, D→Y, D→T, L→T; 3 levels; samplers with 100 to 30,000 samples |
| Chlorophyll | OLS, ridge (λ by CV), Bayesian ridge, polynomial deg 2 + ridge, random forest, GP; 10-fold CV |
