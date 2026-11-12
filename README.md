# Plant Disease Detection & Chlorophyll Estimation using Classical Machine Learning

Detect plant leaf diseases from a photo and estimate leaf greenness with **classical machine learning only**
(no deep learning). Every feature is a hand-made, explainable number (colour, texture, shape).

**Full explanation of the approach, results and likely questions:** [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)

## Quick start: the guided terminal demo

```
pip install -r requirements.txt
python demo.py
```

`demo.py` walks through the **whole project in the terminal**, one step per topic (Units 1, 2, 3 and 5,
the classifiers, and the finished app). For every step it explains what is done and why (with the formulas),
prints the real code from `src/` with file and line numbers, runs it live on the leaf data, draws the key result
as a chart in the terminal (Unit 5), and lists the Unit 5 graphs with a note on how to read each one. Press Enter
to go to the next step. Everything printed is also saved to `reports/demo_log.txt`.

| Option | What it does |
|---|---|
| `--list` / `--step 6` | list the 12 steps / run only some of them |
| `--no-pause` | run straight through (about 5 minutes) |
| `--quick` | show saved results for the two slowest live runs (Unit 3) |
| `--open` | also open every graph in the image viewer |
| `--no-code`, `--no-colour`, `--ascii` | shorter / plain-text output |

## Folder structure

```
plant-disease-ml/
├── demo.py                   guided terminal demo of everything (python demo.py)
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
│   ├── unit5.py              Unit 5: neural network (backprop), AdaBoost, committees, SVM dual,
│   │                         Gaussian process, RVM, MCMC, Gibbs/MRF, Bayesian network
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
│   ── generated, kept in git so the graphs are visible ───────────────
└── reports/
    ├── figures/              the project's graphs - Unit 5 only, one folder per topic:
    │   ├── 15_neural_networks/
    │   ├── 16_combining_models/
    │   ├── 17_kernel_methods/
    │   └── 18_sampling_graphical_models/
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
| 3. Earlier-unit graphs | `python -m src.evaluate --all` | (optional) re-creates the Units 1-3 graphs, which the project no longer keeps |
| 4. Unsupervised | `python -m src.unsupervised` | PCA, k-means, GMM tables |
| 5. Bayesian LR | `python -m src.bayes_logreg` | `models/bayes_logreg.joblib`, tables |
| 6. Chlorophyll | `python -m src.chlorophyll` | collinearity table; a SPAD model only with real readings |
| 7. HMM | `python -m src.hmm` | forward-backward and Viterbi on leaf image rows |
| 8. Unit 5 | `python -m src.unit5` (or `--part nn` / `combining` / `kernels` / `sampling`) | 11 graphs + tables, about 1.5 min |
| **Guided demo** | `python demo.py` | everything above explained and run step by step in the terminal |
| One photo, Unit 3 | `python -m src.leaf_unsupervised samples/tomato_early_blight.jpg` | PCA, k-means, GMM, HMM on that photo's pixels |
| Web app | `streamlit run app.py` | opens http://localhost:8501 |
| Ask questions | `python -m src.assistant --image samples/tomato_early_blight.jpg` | chat: answers from the models and result tables |
| One photo | `python -m src.predict samples/tomato_early_blight.jpg` | top-3 diseases, lesion %, greenness |
| Tests | `python -m pytest -q` | 96 tests |

## Current results (5-class tomato subset, 11,628 images)

| Model | 5-fold CV accuracy | Held-out accuracy |
|---|---|---|
| **Gradient Boosting** (best) | **92.2 ± 1.0%** | **96.3%** |
| Random Forest | 89.4 ± 0.8% | 94.1% |
| SVM (RBF kernel) | 87.4 ± 1.1% | 93.3% |
| Logistic Regression | 84.0 ± 1.8% | 89.3% |

### Unit 5 results (same leaves)

| Topic | Result |
|---|---|
| Neural network (NumPy, backprop) | 93.2% held-out (sklearn MLP 93.0%, logistic regression 89.3%); gradient check error 2e-6 |
| AdaBoost from scratch (300 stumps) | 84.0%, identical predictions to sklearn; one stump alone 52.1% |
| Committees | stacking 91.8% beats every member (best member 89.3%) |
| SVM dual | decision function rebuilt from 162 of 1,500 leaves (support vectors), matches sklearn to 1e-13 |
| Gaussian process / RVM / SVR | lesion severity R² 0.84 / 0.76 / 0.80; RVM keeps 35 leaves, SVR 447; GP 95% interval covers 94% |
| Metropolis-Hastings | posterior mean and std match the Laplace approximation |
| Gibbs sampling in an MRF | noisy lesion mask: 15.0% wrong pixels → 2.1% |
| Bayesian network | exact inference = rejection sampling = likelihood weighting = Gibbs (error ∝ 1/√N) |

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
