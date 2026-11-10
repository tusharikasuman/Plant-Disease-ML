"""
app.py - Streamlit web app: diagnose a leaf photo and browse every result of the project.

Run (from the plant-disease-ml folder):
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import features as F
from src import predict as P
from src import assistant as A

st.set_page_config(page_title="Leaf Doctor - classical ML", page_icon="🌿", layout="wide")

# Okabe-Ito colour-blind safe palette
GREEN, ORANGE, BLUE, VERMILION, GREY = "#009E73", "#E69F00", "#0072B2", "#D55E00", "#9E9E9E"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  .caption-note {color: #5f6b5f; font-size: 0.85rem;}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading models...")
def get_models():
    try:
        return P.load_models()
    except P.ModelsNotTrainedError:
        return None


@st.cache_data
def read_csv(path: str, mtime: float) -> pd.DataFrame:      # mtime -> reload when the file changes
    return pd.read_csv(path)


def report(name: str) -> pd.DataFrame | None:
    path = config.tables_dir() / name
    return read_csv(str(path), path.stat().st_mtime) if path.exists() else None


def pretty(name: str) -> str:
    return config.pretty_class(name)


# ----------------------------------------------------------------------------- page: diagnose
def page_diagnose(models):
    st.title("🌿 Diagnose a leaf")
    st.write("Upload a photo of a **single leaf**. The app cuts the leaf out of the background, measures "
             "55 explainable features (colour, texture, shape) and a classical ML model predicts the disease.")
    if models is None:
        st.error("No trained model yet. In the terminal run:  `python -m src.classify`  then reload this page.")
        return

    left, right = st.columns([2, 1])
    upload = left.file_uploader("Leaf photo (JPG / PNG)", type=["jpg", "jpeg", "png"])
    samples = sorted(config.SAMPLES_DIR.glob("*.jpg"))
    choice = right.selectbox("...or try a sample leaf", ["-"] + [s.stem.replace("_", " ") for s in samples])

    if upload is not None:
        data, source = upload.getvalue(), upload.name
    elif choice != "-":
        path = samples[[s.stem.replace("_", " ") for s in samples].index(choice)]
        data, source = path.read_bytes(), path.name
    else:
        st.info("⬆️ Upload a leaf photo or pick a sample to start.")
        return

    img = F.decode_image(data)
    if img is None:
        st.error("That file could not be read as an image.")
        return
    with st.spinner("Analysing leaf..."):
        pred = P.predict(img, models)
        views = F.segmented_views(img)
    st.session_state["leaf"] = (source, pred)          # the "Ask questions" page talks about this leaf

    # ---- the three images
    st.subheader("1 · Segmentation")
    c1, c2, c3 = st.columns(3)
    c1.image(views["image"], caption=f"Original ({source}), resized to 256×256", width="stretch")
    c2.image(views["mask"], caption="Leaf mask (white = leaf)", width="stretch")
    c3.image(views["lesions"], caption="Segmented leaf - brown lesion pixels in orange", width="stretch")
    if not pred.leaf_found:
        st.warning("No leaf was detected, so the whole image was used. The prediction is unreliable.")

    # ---- prediction
    st.subheader("2 · Prediction")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Diagnosis", "Healthy" if pred.is_healthy else "Diseased")
    m2.metric("Confidence", f"{100 * pred.confidence:.0f}%")
    m3.metric("Lesion area", f"{pred.lesion_percent:.1f}%", help="Brown pixels as a share of all leaf pixels")
    if pred.spad is not None:
        m4.metric("Chlorophyll (SPAD)", f"{pred.spad:.1f}", help="Calibrated with real SPAD meter readings")
    else:
        m4.metric("Greenness index (GLI)", f"{pred.greenness_index:.3f}",
                  help="Green Leaf Index (2G−R−B)/(2G+R+B), range −1 to 1. Uncalibrated: NOT chlorophyll.")

    st.markdown(f"**Most likely:** {pretty(pred.label)}  ·  model: {pred.model_name}")
    fig, ax = plt.subplots(figsize=(7, 1.9))
    labels = [pretty(n) for n, _ in pred.top3][::-1]
    probs = [100 * p for _, p in pred.top3][::-1]
    bars = ax.barh(labels, probs, color=[GREY, GREY, GREEN] if pred.is_healthy else [GREY, GREY, VERMILION])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlim(0, 112); ax.set_xlabel("probability (%)"); ax.set_title("Top-3 predictions", loc="left")
    st.pyplot(fig, width="content"); plt.close(fig)

    if pred.low_confidence:
        st.warning(f"⚠️ Low confidence ({100 * pred.confidence:.0f}% < {100 * config.LOW_CONFIDENCE:.0f}%). "
                   "The model is unsure - check the top-3 list and consider a clearer photo.")
    if pred.spad is not None:
        c = models["chlorophyll"]
        st.caption(f"SPAD model: {c['model_name']}, calibrated on {c['n_leaves']} real leaves "
                   f"({', '.join(c['crops'])}), SPAD range {c['spad_range'][0]:.1f}–{c['spad_range'][1]:.1f}.")
        if pred.spad_outside_range:
            st.warning("This SPAD estimate is outside the range of the calibration readings, so it is unreliable.")
    if pred.spad is None:
        st.caption("The greenness index is an **uncalibrated** colour measure, not a chlorophyll value. "
                   "A SPAD estimate appears only after the chlorophyll model is calibrated with real SPAD "
                   "meter readings (Phase 4).")
    st.info("ℹ️ The model was trained on **lab photos** (PlantVillage: one leaf, plain background). "
            "Field photos with several leaves, soil or strong shadows may be less accurate.")

    if pred.bayes_p_diseased is not None:
        st.subheader("Bayesian second opinion")
        b1, b2, b3 = st.columns(3)
        b1.metric("P(diseased), Bayesian", f"{100 * pred.bayes_p_diseased:.1f}%",
                  help="Bayesian logistic regression: averages over all plausible weight vectors (Laplace "
                       "posterior), so it is less sure when the leaf looks unlike the training leaves.")
        b2.metric("P(diseased), plain LR", f"{100 * pred.bayes_p_diseased_map:.1f}%",
                  help="Same model using only the single best weight vector (MAP), ignoring weight uncertainty.")
        b3.metric("Bayesian confidence", f"{100 * pred.bayes_confidence:.0f}%")
        if pred.refer_to_expert:
            st.warning(f"🔎 The Bayesian model is unsure (confidence < {100 * config.BAYES_REFER_BELOW:.0f}%). "
                       "Refer this leaf to an expert.")
        bayes_diseased = pred.bayes_p_diseased >= 0.5
        if bayes_diseased == pred.is_healthy:
            st.caption("Note: the Bayesian healthy/diseased model disagrees with the main disease model above.")

    # ---- explanation
    st.subheader("3 · Why this prediction?")
    if not pred.group_explanation:
        st.write("Train the logistic regression model (`python -m src.classify`) to see explanations.")
        return
    st.write(f"How much each feature group pushed the score **towards** (+) or **away from** (−) "
             f"*{pretty(pred.label)}*, measured with the interpretable logistic regression model "
             "(weight × standardised feature value).")
    g = pd.Series(pred.group_explanation).sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.barh(g.index, g.values, color=[GREEN if v > 0 else VERMILION for v in g.values])
    ax.axvline(0, color="#333", lw=0.8); ax.set_xlabel("push on the class score")
    st.pyplot(fig, width="content"); plt.close(fig)

    with st.expander("Most influential single features"):
        df = pd.DataFrame(pred.explanation)[["feature", "group", "value", "z_score", "contribution"]]
        df["effect"] = np.where(df["contribution"] > 0, "supports", "argues against")
        st.dataframe(df.rename(columns={"z_score": "value vs average (z)"}).round(3),
                     hide_index=True, width="stretch")
        st.caption("Correlated features (e.g. GLCM contrast and dissimilarity) can get large opposite weights "
                   "that cancel out - that is why the group totals above are the more reliable summary.")


# ----------------------------------------------------------------------------- page: results
def page_results(models):
    st.title("📊 Model results")
    res = report("classification_results.csv")
    if res is None:
        st.error("No results yet. Run  `python -m src.classify`  in the terminal.")
        return

    if models is not None:
        meta = models["best"].get("dataset", {})
        n_cls = len(models["best"]["class_names"])
        msg = f"Trained on **{n_cls} classes** from `{meta.get('train_dir', '?')}`."
        if n_cls < len(config.CLASS_NAMES):
            msg += (f" This is a subset of the full {len(config.CLASS_NAMES)}-class Kaggle dataset - download it "
                    "and re-run `python -m src.dataset` and `python -m src.classify` for the final results.")
        st.info(msg)

    best = res.loc[res["Best"] == "*", "Model"]
    best_name = best.iloc[0] if len(best) else res.iloc[res["CV accuracy (%)"].idxmax()]["Model"]
    b = res[res["Model"] == best_name].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model (by CV)", best_name)
    c2.metric("5-fold CV accuracy", f"{b['CV accuracy (%)']:.1f}%", f"± {b['CV std']:.1f}", delta_color="off")
    c3.metric("Held-out accuracy", f"{b['Valid accuracy (%)']:.1f}%")
    c4.metric("Held-out balanced accuracy", f"{b['Valid balanced acc (%)']:.1f}%")

    st.subheader("All models")
    order = res.sort_values("CV accuracy (%)")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(order))
    ax.barh(y - 0.2, order["CV accuracy (%)"], 0.4, xerr=order["CV std"], color=GREEN,
            label="5-fold CV accuracy", error_kw={"ecolor": "#555", "capsize": 2})
    ax.barh(y + 0.2, order["Valid balanced acc (%)"], 0.4, color=BLUE, label="held-out balanced accuracy")
    ax.set_yticks(y, [f"{m}  ({u})" for m, u in zip(order["Model"], order["Syllabus"])])
    ax.set_xlim(max(0, order["Valid balanced acc (%)"].min() - 15), 100); ax.set_xlabel("%")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    st.pyplot(fig); plt.close(fig)
    st.dataframe(res.drop(columns=["Topic"]).round(2), hide_index=True, width="stretch")
    st.caption("CV = grid search + 5-fold stratified cross-validation on a class-balanced sample. "
               "Held-out = one final test on images never used for training or tuning. Balanced accuracy "
               "is the average recall over classes, so a large easy class cannot inflate it.")

    abl = report("ablation.csv")
    if abl is not None:
        st.subheader("Feature ablation - what each feature group adds")
        cols = [c for c in abl.columns if c.endswith("CV acc (%)")]
        fig, ax = plt.subplots(figsize=(8, 3))
        x = np.arange(len(abl))
        for i, (col, color) in enumerate(zip(cols, [GREEN, BLUE])):
            bars = ax.bar(x + (i - (len(cols) - 1) / 2) * 0.38, abl[col], 0.38, color=color,
                          label=col.replace(" CV acc (%)", ""))
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(x, [f"{s}\n({n} features)" for s, n in zip(abl["Feature set"], abl["# features"])], fontsize=8)
        ax.set_ylim(max(0, abl[cols].min().min() - 15), 100); ax.set_ylabel("5-fold CV accuracy (%)")
        ax.legend(frameon=False, fontsize=8)
        st.pyplot(fig); plt.close(fig)

    rep = config.tables_dir() / "classification_report_best.txt"
    if rep.exists():
        with st.expander("Per-class precision / recall / F1 of the best model"):
            st.code(rep.read_text(encoding="utf-8"))
    st.caption("Every graph, by topic, is on the **All graphs** page.")


# ----------------------------------------------------------------------------- page: bayesian
BAYES_GRAPHS = [
    ("1_uncertainty_2d", "What the Bayesian model 'thinks', on 2 features",
     "Left: the predictive probability of *diseased*; the black line is the decision boundary (P = 0.5). "
     "Right: the uncertainty s = √(xᵀS_N x) of the score. It is smallest where the training leaves are and grows "
     "as we move away from them: the model knows what it has not seen."),
    ("3_unseen_disease", "A disease the model never saw",
     "One disease is removed from training, then its leaves are predicted. On the harder hidden diseases the "
     "Bayesian predictive has lower log-loss and is less over-confident on the leaves it gets wrong."),
    ("2_confidence_vs_error", "Confidence tells us when to ask an expert",
     "Wrong predictions come with low confidence. Sending the least confident leaves to an expert raises the "
     "accuracy on the rest."),
    ("4_weights", "Weights with 95% credible intervals",
     "The Laplace posterior gives every weight an interval. Wide intervals crossing 0 mean the data cannot pin "
     "that weight down - usually because the feature is strongly correlated with others."),
]


def page_bayes():
    st.title("🎲 Bayesian logistic regression")
    st.markdown("""
Healthy vs diseased, written **from scratch in NumPy** (Unit 2: probabilistic discriminative models):

* **Prior** - Gaussian on the weights, w ~ N(0, α⁻¹I).
* **MAP** - the most probable weights, found with **Newton's method** (iteratively reweighted least squares).
* **Laplace approximation** - the Hessian at the MAP is the inverse posterior covariance S_N.
* **Predictive** - averages the sigmoid over all plausible weights, using the **probit approximation**
  σ(κ(s²)·μ), κ = (1 + πs²/8)^-½. This pulls uncertain predictions towards 50 %.
""")
    res = report("bayes_results.csv")
    if res is None:
        st.error("No Bayesian results yet. Run  `python -m src.bayes_logreg`  in the terminal.")
        return
    weights = report("bayes_weights.csv")
    if weights is not None:
        diff = (weights["w_MAP"] - weights["w_sklearn"]).abs().max()
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (held-out)", f"{res['Accuracy (%)'].iloc[-1]:.2f}%")
        c2.metric("Max |w_MAP − w_sklearn|", f"{diff:.1e}", help="Our NumPy MAP equals sklearn with C = 1/α")
        c3.metric("Features", f"{len(weights)} + bias")
    st.dataframe(res.round(4), hide_index=True, width="stretch")

    for name, title, text in BAYES_GRAPHS:
        path = config.figure_dir("bayes") / f"{name}.png"
        if path.exists():
            st.subheader(title)
            st.write(text)
            st.image(str(path), width="stretch")
            if name == "3_unseen_disease" and report("bayes_unseen_disease.csv") is not None:
                with st.expander("Table"):
                    st.dataframe(report("bayes_unseen_disease.csv").round(3), hide_index=True, width="stretch")


# ----------------------------------------------------------------------------- page: chlorophyll
def page_chlorophyll(models):
    st.title("🧪 Chlorophyll (SPAD)")
    st.markdown("""
Following **Paper 2** (Anderson et al., 2026), chlorophyll is estimated with a linear model **b = A X**
from leaf colour features X to the reading b of a SPAD chlorophyll meter. Models compared: least squares
(OLS), **Ridge** (λ by cross-validation), Bayesian Ridge, polynomial regression (degree 2) and Random Forest.
""")
    calibrated = models is not None and models.get("chlorophyll") is not None
    if calibrated:
        c = models["chlorophyll"]
        st.success(f"Calibrated: **{c['model_name']}** on {c['n_leaves']} real leaves ({', '.join(c['crops'])}), "
                   f"{c['cv']} cross-validation.")
        res = report("chlorophyll_results.csv")
        if res is not None:
            st.dataframe(res.round(3), hide_index=True, width="stretch")
        for name, title in [("2_pred_vs_actual", "Predicted vs measured SPAD"),
                            ("3_coefficient_stability", "OLS vs Ridge: coefficient stability")]:
            path = config.figure_dir("chlorophyll") / f"{name}.png"
            if path.exists():
                st.subheader(title)
                st.image(str(path), width="stretch")
    else:
        st.warning("**Not calibrated yet.** There are no real SPAD meter readings, so the app shows only an "
                   "*uncalibrated greenness index* (GLI) - it is **not** a chlorophyll value. No SPAD data is "
                   "ever made up.")
        st.markdown("""
**How to calibrate**
1. Photograph 30–50 leaves of one crop (same light and background), covering pale to dark green.
2. Measure each leaf 3–5 times with a SPAD meter and write the readings in `spad/spad_readings.csv`.
3. Run `python -m src.chlorophyll` and restart the app.
""")
        if config.SPAD_CSV.exists():
            st.download_button("Download the empty spad_readings.csv template", config.SPAD_CSV.read_bytes(),
                               file_name="spad_readings.csv", mime="text/csv")
        guide = config.PROJECT_ROOT / "docs" / "spad_collection.md"
        if guide.exists():
            with st.expander("Full guide: collecting SPAD readings"):
                st.markdown(guide.read_text(encoding="utf-8"))

    st.subheader("Why Ridge? The colour features are strongly collinear")
    st.write("Measured on the real leaves of the disease dataset (no SPAD data needed). A variance inflation "
             "factor (VIF) above 10 means a feature is almost a combination of the others, so least-squares "
             "coefficients swing wildly from sample to sample. Ridge regression's L2 penalty stabilises them.")
    path = config.figure_dir("chlorophyll") / "1_collinearity.png"
    if path.exists():
        st.image(str(path), width="stretch")
    else:
        st.info("Run `python -m src.chlorophyll` to compute the correlation matrix and VIF.")


# ----------------------------------------------------------------------------- page: all graphs
TOPIC_TITLES = {
    "features": "Feature representation", "pca": "PCA", "kmeans": "k-means", "gmm": "GMM + EM",
    "tree": "Decision trees", "rf": "Bagging / Random Forest", "boosting": "Boosting", "knn": "KNN",
    "cv": "CV & model selection", "metrics": "Evaluation metrics", "ablation": "Ablation",
    "bayes": "Bayesian LR", "chlorophyll": "Chlorophyll", "hmm": "HMM (Viterbi, forward-backward)",
}


def page_graphs():
    st.title("🖼️ All graphs")
    st.write("Every graph of the project, one tab per topic. The same PNG files are in "
             "`reports/figures/<topic>/` for slides. Regenerate them with `python -m src.evaluate`.")
    topics = [t for t in config.FIG if any(config.figure_dir(t).glob("*.png"))]
    if not topics:
        st.error("No graphs yet. Run  `python -m src.evaluate`  in the terminal.")
        return
    for tab, topic in zip(st.tabs([TOPIC_TITLES[t] for t in topics]), topics):
        with tab:
            for g in sorted(config.figure_dir(topic).glob("*.png")):
                st.markdown(f"**{g.stem.replace('_', ' ')}**  ·  `reports/figures/{config.FIG[topic]}/{g.name}`")
                st.image(str(g), width="stretch")


# ----------------------------------------------------------------------------- page: how it works
def page_how():
    st.title("🧪 How it works")
    st.markdown("""
**Classical machine learning only** - no deep learning. Every number the model sees is a hand-made,
explainable measurement of the leaf.

1. **Preprocess** - resize to 256×256, 3×3 Gaussian blur to remove camera noise.
2. **Segment** - keep pixels that are green (healthy tissue) or brown (lesions) in HSV colour space,
   close small gaps, keep the largest connected region, fill enclosed holes, and drop dark shadows.
3. **Measure 55 features** on leaf pixels only (table below).
4. **Classify** - StandardScaler + classifier pipelines, tuned with grid search and 5-fold stratified
   cross-validation, then tested once on held-out images.
""")
    rows = [
        ("Colour averages (RGB)", 6, "Mean and spread of red, green, blue", "R mean, G std"),
        ("Greenness & lesion indices", 4, "GLI, G/R ratio, Excess Green, lesion fraction", "Lesion fraction = brown / leaf pixels"),
        ("Colour histogram (HSV)", 32, "Share of pixels in 8 hue × 2 saturation × 2 brightness bins", "HSV h0 s1 v0 = dull dark orange-brown"),
        ("Texture (GLCM)", 6, "How grey levels of neighbouring pixels relate (Haralick)", "contrast is high for spotty leaves"),
        ("Shape (Hu moments)", 7, "Outline descriptors, unchanged by moving / rotating / resizing", "curled leaves differ"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Group", "# features", "What it measures", "Example"]),
                 hide_index=True, width="stretch")

    mapping = report("syllabus_mapping.csv")
    if mapping is not None:
        st.subheader("Syllabus mapping")
        st.dataframe(mapping, hide_index=True, width="stretch")

    st.subheader("References")
    st.markdown("""
1. Kiran & Chandrappa, *Plant Leaf Disease Detection Using Efficient Image Processing and Machine Learning
   Algorithms*, Journal of Robotics and Control 4(6), 2023. doi:10.18196/jrc.v4i6.20342
2. Anderson et al., *Image-based leaf SPAD value and chlorophyll measurement using a mobile phone*,
   Precision Agriculture, 2026. doi:10.1007/s11119-026-10386-x
""")


# ----------------------------------------------------------------------------- page: ask questions
@st.cache_resource
def get_assistant():
    return A.get_assistant()


def page_ask(models):
    st.title("💬 Ask Leaf Doctor")
    st.write("Type a question about the leaf, the diseases or the project. Answers are computed live from the "
             "trained models and result tables (TF-IDF + nearest-neighbour matching, classical ML).")
    leaf = st.session_state.get("leaf")
    samples = sorted(config.SAMPLES_DIR.glob("*.jpg"))
    with st.expander("Leaf to ask about", expanded=leaf is None):
        upload = st.file_uploader("Leaf photo (JPG / PNG)", type=["jpg", "jpeg", "png"], key="ask_upload")
        choice = st.selectbox("...or a sample leaf", ["-"] + [s.name for s in samples], key="ask_sample")
        if models is not None and (upload is not None or choice != "-"):
            data, source = (upload.getvalue(), upload.name) if upload is not None else \
                ((config.SAMPLES_DIR / choice).read_bytes(), choice)
            if leaf is None or leaf[0] != source:
                img = F.decode_image(data)
                if img is not None:
                    leaf = (source, P.predict(img, models))
                    st.session_state["leaf"] = leaf
    if leaf is not None:
        st.caption(f"Talking about **{leaf[0]}** - predicted {pretty(leaf[1].label)}.")
    ctx = A.Context(prediction=leaf[1] if leaf else None, models=models, source=leaf[0] if leaf else None)

    history = st.session_state.setdefault("chat", [])
    for role, text in history:
        st.chat_message(role).markdown(text)
    sugg = A.SUGGESTIONS_LEAF if leaf else A.SUGGESTIONS_PROJECT
    cols = st.columns(len(sugg))
    clicked = next((s for col, s in zip(cols, sugg) if col.button(s, key=f"sugg_{s}")), None)
    question = st.chat_input("Ask a question...") or clicked
    if question:
        ans = get_assistant().ask(question, ctx)
        text = ans.text + ("".join(f"\n- {s}" for s in ans.suggestions))
        history += [("user", question), ("assistant", text)]
        st.rerun()
    if history and st.button("Clear chat"):
        st.session_state["chat"] = []
        st.rerun()


# ----------------------------------------------------------------------------- layout
def main():
    models = get_models()
    with st.sidebar:
        st.markdown("## 🌿 Leaf Doctor")
        st.caption("Plant disease detection & chlorophyll estimation with classical machine learning")
        page = st.radio("Page", ["Diagnose a leaf", "Ask questions", "Model results", "All graphs", "Bayesian LR", "Chlorophyll",
                                 "How it works"], label_visibility="collapsed")
        st.divider()
        if models is not None:
            b = models["best"]
            st.markdown(f"**Model:** {b['model_name']}  \n**Classes:** {len(b['class_names'])}  \n"
                        f"**Trained on:** {b['train_size']:,} leaves")
            st.markdown("**Chlorophyll:** " + ("calibrated (SPAD)" if models["chlorophyll"] is not None
                                               else "not calibrated - greenness index only"))
        else:
            st.warning("No trained model yet.")

    if page == "Diagnose a leaf":
        page_diagnose(models)
    elif page == "Ask questions":
        page_ask(models)
    elif page == "Model results":
        page_results(models)
    elif page == "All graphs":
        page_graphs()
    elif page == "Bayesian LR":
        page_bayes()
    elif page == "Chlorophyll":
        page_chlorophyll(models)
    else:
        page_how()


main()
