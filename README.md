# Plant Disease Detection & Chlorophyll Estimation using Classical Machine Learning

Detect plant leaf diseases from a photo and estimate leaf greenness with **classical machine learning only**
(no deep learning). Every feature is a hand-made, explainable number (colour, texture, shape).

**Full explanation of the approach, results and likely questions:** [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)

## Folder structure

```
plant-disease-ml/
├── app.py                    Streamlit web app (streamlit run app.py)
├── config.py                 every path, seed and setting
├── requirements.txt
├── src/                      the code, one file per step
│   ├── features.py           preprocessing, leaf segmentation, the 55 features
│   ├── dataset.py            find the dataset, extract features for all images, cache them
│   ├── classify.py           9 disease classifiers, grid search + 5-fold CV, ablation
│   ├── unsupervised.py       PCA, k-means, Gaussian mixture (EM)
│   ├── bayes_logreg.py       Bayesian logistic regression from scratch (NumPy)
│   ├── chlorophyll.py        SPAD regression (trains only on real meter readings)
│   ├── hmm.py                Hidden Markov Model on leaf image rows: forward-backward + Viterbi (NumPy)
│   ├── leaf_unsupervised.py  PCA, k-means, GMM/EM and HMM on the pixels of ONE leaf photo
│   ├── figures.py            all supervised-learning graphs
│   ├── evaluate.py           metrics, tables, graph style; `python -m src.evaluate` makes every graph
│   ├── predict.py            predict one leaf photo
│   └── assistant.py          Q&A: TF-IDF + nearest-neighbour question matching, live answers
├── tests/                    pytest tests (python -m pytest -q)
├── docs/
│   ├── PROJECT_OVERVIEW.md   architecture, results, presentation guide
│   └── spad_collection.md    how to collect SPAD readings
├── samples/                  3 leaf photos for the demo and tests
├── spad/
│   ├── spad_readings.csv     your real SPAD readings go here (empty template)
│   └── photos/               the leaf photos that go with them
├── legacy/                   the original course scripts this project grew from
│
│   ── generated, not in git ──────────────────────────────────────────
├── data/                     put the downloaded Kaggle dataset here
├── cache/                    features.npz (extracted features)
├── models/                   trained models (.joblib)
└── reports/
    ├── figures/              every graph, one folder per topic:
    │   ├── 01_feature_representation/   ├── 08_knn/
    │   ├── 02_pca/                      ├── 09_cv_model_selection/
    │   ├── 03_kmeans/                   ├── 10_evaluation_metrics/
    │   ├── 04_gmm_em/                   ├── 11_ablation/
    │   ├── 05_decision_trees/           ├── 12_bayesian_logistic_regression/
    │   ├── 06_bagging_random_forest/    └── 13_chlorophyll/
    │   ├── 07_boosting/                 └── 14_hmm_viterbi_forward_backward/
    └── tables/               every result as CSV + Markdown
```

## Setup

**Windows / VS Code:** open the `plant-disease-ml` folder (File → Open Folder), then in the terminal:
```
pip install -r requirements.txt
```
**Google Colab:** upload the folder (or clone the repo), then `!pip install -r requirements.txt`.

**Dataset:** Kaggle "New Plant Diseases Dataset (Augmented)" -
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset. Unzip it into `data/` (it is found
automatically). Without it, `python -m src.dataset` prints download instructions.

## How to run

| Step | Command | What it makes |
|---|---|---|
| 1. Features | `python -m src.dataset` (`--limit 50` for a quick test) | `cache/features.npz` |
| 2. Classifiers | `python -m src.classify` | `models/best_model.joblib`, `reports/tables/classification_*` |
| 3. All graphs | `python -m src.evaluate --all` | every graph in `reports/figures/` (also runs steps 4–6) |
| 4. Unsupervised | `python -m src.unsupervised` | PCA, k-means, GMM graphs and tables |
| 5. Bayesian LR | `python -m src.bayes_logreg` | `models/bayes_logreg.joblib`, graphs, tables |
| 6. Chlorophyll | `python -m src.chlorophyll` | collinearity graph; a SPAD model only with real readings |
| 7. HMM | `python -m src.hmm` | forward-backward and Viterbi on leaf image rows |
| One photo, Unit 3 | `python -m src.leaf_unsupervised samples/tomato_early_blight.jpg` | PCA, k-means, GMM, HMM on that photo's pixels |
| Web app | `streamlit run app.py` | opens http://localhost:8501 |
| Ask questions | `python -m src.assistant --image samples/tomato_early_blight.jpg` | chat: answers from the models and result tables |
| One photo | `python -m src.predict samples/tomato_early_blight.jpg` | top-3 diseases, lesion %, greenness |
| Tests | `python -m pytest -q` | 75 tests |

## Current results (5-class tomato subset, 11,628 images)

| Model | 5-fold CV accuracy | Held-out accuracy |
|---|---|---|
| **Gradient Boosting** (best) | **92.2 ± 1.0%** | **96.3%** |
| Random Forest | 89.4 ± 0.8% | 94.1% |
| SVM (RBF kernel) | 87.4 ± 1.1% | 93.3% |
| Logistic Regression | 84.0 ± 1.8% | 89.3% |

All 9 models, the Bayesian, unsupervised and chlorophyll results: [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md).
The full 38-class run needs the Kaggle dataset.

## Limitations
- Trained on lab photos (one leaf, plain background); field photos may be less accurate.
- Chlorophyll needs real SPAD meter readings, calibrated per crop. Until then only an
  *uncalibrated greenness index* is shown - never a made-up SPAD value.

## References
1. S. M. Kiran and D. N. Chandrappa, "Plant Leaf Disease Detection Using Efficient Image Processing and Machine
   Learning Algorithms," *Journal of Robotics and Control* 4(6), 2023. doi:10.18196/jrc.v4i6.20342
2. J. Anderson et al., "Image-based leaf SPAD value and chlorophyll measurement using a mobile phone,"
   *Precision Agriculture*, 2026. doi:10.1007/s11119-026-10386-x
