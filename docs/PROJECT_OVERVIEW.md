# Plant Disease Detection & Chlorophyll Estimation using Classical Machine Learning

**Project overview and presentation guide**

> We detect plant leaf diseases from a photo, and estimate how green or healthy the leaf is, using
> **classical machine learning only** (no deep learning), so every decision can be explained with real,
> measurable features.

---

## 1. The problem and why this approach

- Farmers need early, cheap disease detection. A phone photo is the cheapest sensor there is.
- Deep learning (CNNs) is accurate but a "black box", and it needs GPUs and huge amounts of data.
- **Our approach:** turn each photo into **55 meaningful numbers** (colour, texture, shape), then use
  classical ML models from our syllabus. They run on a normal laptop CPU, and we can explain *why*
  each prediction was made.

---

## 2. Architecture: 5 stages

```
 Leaf photo
    │
    ▼
 ① PREPROCESS     resize 256×256 → Gaussian blur 3×3 (remove noise)
    │
    ▼
 ② SEGMENT        HSV colour threshold (green + brown) → morphological closing
    │             → keep largest connected region → fill holes → remove shadows
    │             = "leaf mask" (which pixels are leaf)
    ▼
 ③ FEATURES       55 hand-made numbers from LEAF PIXELS ONLY
    │             colour (6) · indices (4) · HSV histogram (32) · texture (6) · shape (7)
    ▼
 ④ MODELS         StandardScaler → classifier   (8 models compared)
    │             + Bayesian logistic regression (healthy vs diseased, with uncertainty)
    │             + chlorophyll regression (SPAD, needs real meter readings)
    ▼
 ⑤ APP            Streamlit web app: upload photo → diagnosis, confidence,
                  lesion %, top-3, "why this prediction?"
```

| Stage | Code |
|---|---|
| ① Preprocess, ② Segment, ③ Features | `src/features.py` |
| Find dataset, extract all features, cache | `src/dataset.py` |
| ④ Disease classification (8 models) | `src/classify.py` |
| ④ Bayesian logistic regression (from scratch) | `src/bayes_logreg.py` |
| ④ Chlorophyll / SPAD regression | `src/chlorophyll.py` |
| Metrics, tables, graphs | `src/evaluate.py` |
| Predict one photo | `src/predict.py` |
| ⑤ Web app | `app.py` |

### ① + ② Preprocessing and segmentation (image processing)

- Converting to **HSV** colour space separates *colour* (hue) from *brightness*, so "green" and "brown"
  are easy to threshold.
- **Morphological closing** fills small gaps. **Largest connected component** removes stray specks.
  **Hole filling** keeps dark lesion spots inside the leaf.
- **Shadow rejection:** brown pixels *outside* the leaf outline that are dark get removed. We added this
  after finding that shadows were being counted as disease.
- **Why it matters:** features come only from the leaf, so the background cannot fool the model.

### ③ The 55 features (each one explainable)

| Group | Count | What it measures | Why it helps |
|---|---|---|---|
| RGB mean and std | 6 | Average colour, and how blotchy the leaf is | Diseased leaves are more uneven |
| Indices | 4 | GLI, G/R ratio, ExG (greenness), **lesion fraction** | Lesion % = brown pixels ÷ leaf pixels |
| HSV histogram | 32 | Share of pixels in 8 hue × 2 saturation × 2 brightness bins | Captures yellowing and browning patterns |
| GLCM texture | 6 | Contrast, dissimilarity, homogeneity, energy, correlation, ASM | Spots make the surface "rough" |
| Hu moments | 7 | Shape of the leaf outline | Curled leaves (leaf curl virus) look different |

Lesion fraction alone already separates the classes (mean per class):
**healthy 1.0% → yellow leaf curl 2.3% → Septoria 5.1% → early blight 9.0% → late blight 12.6%.**

### ④ Models, mapped to the syllabus

| Model | Syllabus | The idea in one line |
|---|---|---|
| Gaussian Naive Bayes | **Unit 1**: Bayes rule, conditional independence | Assumes features are independent; simplest |
| Logistic Regression | **Unit 2**: discriminative model, MLE | Learns P(disease \| features) directly |
| LDA | **Unit 2**: generative model, discriminant functions | Models each class as a Gaussian, then applies Bayes rule |
| SVM (linear kernel) | **Unit 2**: maximum margin | Widest gap between classes |
| SVM (RBF kernel) | **Unit 2**: kernel trick | Curved boundaries via an implicit high-dimensional space |
| KNN, Decision Tree, Random Forest | Baselines from **Paper 1** | Comparison with the published paper |
| Bayesian Logistic Regression | **Unit 2**: written from scratch in NumPy | Gaussian prior, Newton's method, Laplace approximation, probit predictive |
| OLS / Ridge / Bayesian Ridge / polynomial | **Unit 2**: regression | Chlorophyll from colour (**Paper 2**) |

**How we trained them fairly:**
- Every model goes in a pipeline: **StandardScaler → model** (scaling is learned inside each CV fold,
  so there is no data leakage).
- Settings (C, gamma, λ) are tuned with **grid search + 5-fold stratified cross-validation**.
- The best model is chosen by its **CV score**, then tested **once** on held-out images it never saw.
- A fixed random seed (42) everywhere, so results are reproducible.

---

## 3. Dataset

- **Currently:** PlantVillage tomato leaves, **5 classes, 11,628 images**: healthy, early blight,
  late blight, Septoria leaf spot, yellow leaf curl virus.
- Split 80/20 (stratified): **9,302 for training, 2,326 held out for testing.**
- **Target:** Kaggle "New Plant Diseases Dataset (Augmented)" - 38 classes, ~88,000 images.
  The code is ready; it only needs the download.
- Feature extraction speed: about **4.4 ms per image** with 8 CPU workers.

---

## 4. Results

### Disease classification (5 tomato classes)

| Model | CV accuracy | Test accuracy | Test balanced accuracy |
|---|---|---|---|
| **Random Forest** (best by CV) | **89.4 ± 0.8%** | **94.1%** | 90.4% |
| SVM (RBF kernel) | 87.4 ± 1.1% | 93.3% | **91.6%** |
| Logistic Regression | 84.0 ± 1.8% | 89.3% | 85.0% |
| SVM (linear kernel) | 83.3 ± 1.5% | 88.5% | 82.8% |
| KNN (k = 9) | 80.4 ± 1.7% | 87.8% | 81.0% |
| LDA | 79.9 ± 1.7% | 85.0% | 78.9% |
| Decision Tree | 75.8 ± 1.4% | 84.8% | 79.6% |
| Gaussian Naive Bayes | 64.5 ± 1.4% | 71.1% | 64.2% |

**Key messages**
1. **The kernel trick works:** RBF SVM beats linear SVM by about 4 points.
2. **Ensembles help:** Random Forest beats a single decision tree by about 14 points.
3. **Naive Bayes is last** because its independence assumption is false: R and G are 91% correlated.
4. **Hardest class: early blight** (70% recall vs 99% for healthy). It gets confused with late blight,
   since both are brown-spot diseases.

### Feature ablation (what each feature group adds, RBF SVM, 5-fold CV)

| Features | Count | Accuracy |
|---|---|---|
| RGB averages only | 6 | 73.1% |
| + greenness indices + HSV histogram | 42 | 86.1% |
| + texture (GLCM) | 48 | **88.1%** |
| + shape (Hu moments) | 55 | 87.4% |

Colour features matter most (+13 points), and texture adds a further 2 points.

### Bayesian logistic regression (healthy vs diseased)

- **99.5% accuracy** on the held-out test set.
- **The maths is verified:** our from-scratch MAP weights match scikit-learn's to within **0.0004**.
  The Hessian matches finite differences, and the probit shortcut matches a Monte Carlo average.
- **It knows when it is unsure:** wrong predictions average **69%** confidence, correct ones **99%**.
  Sending the least-confident 20% of leaves to an expert gives **100% accuracy** on the rest.
- **On a disease it never saw in training**, it is less overconfident: log-loss is **25–30% lower**
  for late blight (0.18 → 0.13) and Septoria (0.33 → 0.25).
- **Honest negative result:** on leaves like the training leaves, plain logistic regression is already
  well calibrated, and the extra Bayesian caution costs a little.

### Chlorophyll (SPAD)

- The colour features are highly collinear (variance inflation factor): **R 62, G 63, B 16,
  GLI 807, ExG 715**. R and G correlate at **r = 0.91**.
- VIF above 10 means least-squares coefficients are unstable, so **Ridge** regression is needed
  (Unit 1 covariance → Unit 2 ridge).
- **Not calibrated yet.** The model needs real SPAD meter readings (`spad_readings.csv`,
  see `docs/spad_collection.md`). Until then the app shows only an *uncalibrated greenness index*,
  clearly labelled **not chlorophyll**. We never fake data.

---

## 5. The web app (live demo)

Run inside the `plant-disease-ml` folder:

```
streamlit run app.py
```

| Page | What it shows |
|---|---|
| **Diagnose a leaf** | Original / mask / lesions, diagnosis, confidence, lesion %, top-3 chart, Bayesian second opinion, "Why this prediction?" |
| **Model results** | All 8 models compared, ablation chart, per-class precision / recall / F1 |
| **Bayesian LR** | 7 graphs with plain-English explanations |
| **Chlorophyll** | Calibration status, how to collect SPAD readings, "Why Ridge?" collinearity graph |
| **How it works** | Pipeline, the 55 features, syllabus mapping, references |

**Demo tip:** pick the sample "tomato early blight" and show the mask, the top-3 chart and the
"Why this prediction?" panel.

---

## 6. Technology

| Tool | Used for |
|---|---|
| Python | Everything |
| OpenCV | Reading images, resizing, blur, HSV masks, morphology, Hu moments |
| scikit-image | GLCM (Haralick) texture features |
| scikit-learn | All classifiers and regressors, cross-validation, metrics |
| NumPy / SciPy | Bayesian logistic regression written from scratch |
| matplotlib | All graphs (white background, colour-blind-safe colours) |
| Streamlit | Web app |
| pytest | 59 automated tests |
| Git / GitHub | Version control |

Runs on a normal laptop CPU (8 GB RAM) and in Google Colab.

---

## 7. Limitations

1. Trained on **lab photos** (one leaf, plain background); field photos may be less accurate.
2. Currently **5 tomato classes**; the 38-class run is pending the dataset download.
3. Chlorophyll needs **real SPAD readings**, calibrated separately for each crop.
4. Shadows near the leaf stem can still be counted as lesion.

---

## 8. Likely questions

**Why not deep learning?**
Our course is classical ML. Our features are explainable, it runs on a CPU, and it still reaches
about 94% accuracy.

**Why is test accuracy higher than CV accuracy?**
The test set has many easy yellow-leaf-curl leaves, and the final model trains on more data.
That is why we also report **balanced accuracy**, which weighs every class equally.

**Why is Random Forest the best?**
Averaging 200 trees, each trained on random samples and random features, reduces variance,
which a single decision tree has a lot of.

**What does the Bayesian model add?**
An honest confidence. It knows when a leaf looks unfamiliar, so the app can say "refer to an expert".

**How do you explain a prediction?**
Logistic regression weight × standardised feature value shows how much each feature group pushed
towards the diagnosis. It is shown on the app's "Why this prediction?" panel.

**Is the chlorophyll value real?**
Not yet. Without real SPAD meter readings we only show an uncalibrated greenness index, labelled as
not chlorophyll. Once readings are collected, `python -m src.chlorophyll` trains the model.

---

## 9. How to run everything

```
pip install -r requirements.txt        # once
python -m src.dataset                  # extract the 55 features (cached)
python -m src.classify                 # train and compare the 8 models
python -m src.bayes_logreg             # Bayesian logistic regression
python -m src.chlorophyll              # collinearity analysis; trains only with real SPAD data
python -m src.evaluate                 # regenerate graphs
streamlit run app.py                   # web app
python -m pytest -q                    # tests
```

---

## References

1. S. M. Kiran and D. N. Chandrappa, "Plant Leaf Disease Detection Using Efficient Image Processing
   and Machine Learning Algorithms," *Journal of Robotics and Control*, 4(6), 2023.
   doi:10.18196/jrc.v4i6.20342
2. J. Anderson et al., "Image-based leaf SPAD value and chlorophyll measurement using a mobile phone,"
   *Precision Agriculture*, 2026. doi:10.1007/s11119-026-10386-x
