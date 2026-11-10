"""
src/assistant.py - "Ask Leaf Doctor": answer typed questions using the project's own models and results.

How it works (classical ML, no deep learning, no internet):
  1. Every kind of question the assistant can answer is an INTENT, described by a few example phrasings.
  2. The question and all example phrasings are turned into TF-IDF vectors (word 1-2 grams + character
     3-5 grams, so small typos still match).
  3. The intent of the most similar example (cosine similarity) wins - a 1-nearest-neighbour classifier.
     Below a similarity threshold the assistant says it did not understand and suggests questions.
  4. Names in the question (a disease, a model, a feature group) are picked out by keyword matching.
  5. The intent's answer function builds the reply DYNAMICALLY from
       - the current leaf prediction (src/predict.py: disease, confidence, lesion %, Bayesian opinion, ...)
       - the result tables in reports/tables/ (re-read from disk, so retraining updates the answers)
       - a small built-in knowledge base about the diseases (cause, symptoms, management).

Usage (from the plant-disease-ml folder):
    python -m src.assistant                                   # interactive chat about the project
    python -m src.assistant --image samples/tomato_early_blight.jpg      # chat about one leaf
    python -m src.assistant --image samples/tomato_healthy.jpg "is this leaf healthy?"
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):          # allow `python src/assistant.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

import config

MIN_SIMILARITY = 0.18          # below this cosine similarity the question is "not understood"

# ============================================================================= disease knowledge base
# General plant-pathology facts for the classes the model was trained on. Not a substitute for an
# agronomist - the app says so.
DISEASES = {
    "Tomato___Early_blight": {
        "aliases": ["early blight", "alternaria"],
        "cause": "the fungus *Alternaria solani*",
        "symptoms": "brown spots with dark **concentric rings** (a 'target' pattern), often with a yellow halo, "
                    "starting on the **older, lower leaves**",
        "spread": "spores spread by rain splash, wind and tools; it survives in infected plant debris and soil",
        "manage": ["remove and destroy infected lower leaves", "mulch the soil so spores do not splash up",
                   "water at the base, not over the leaves, and space plants for airflow",
                   "rotate crops (no tomato/potato in the same bed for 2-3 years)",
                   "if needed, a protective fungicide (e.g. chlorothalonil, mancozeb or copper), as per the label"],
    },
    "Tomato___Late_blight": {
        "aliases": ["late blight", "phytophthora"],
        "cause": "the water mould *Phytophthora infestans* (an oomycete, the same pathogen as the Irish potato famine)",
        "symptoms": "large, irregular, **greasy grey-green to dark brown patches**, with white fuzzy growth on the "
                    "underside of the leaf in humid weather; it spreads **very fast**",
        "spread": "airborne spores in cool, wet, humid weather; it can destroy a field within days",
        "manage": ["remove and bag (do not compost) infected plants immediately",
                   "keep leaves dry and improve airflow", "plant resistant varieties",
                   "protective fungicides before/at the first signs in wet weather, as per the label",
                   "warn neighbouring growers - it spreads between farms"],
    },
    "Tomato___Septoria_leaf_spot": {
        "aliases": ["septoria", "leaf spot", "septoria leaf spot"],
        "cause": "the fungus *Septoria lycopersici*",
        "symptoms": "many **small round spots** (2-5 mm) with a dark border and a grey/tan centre, often with tiny "
                    "black dots in the middle; lower leaves first, then they yellow and drop",
        "spread": "rain splash and overhead watering in warm, wet weather; survives on plant debris",
        "manage": ["remove infected leaves", "mulch and avoid overhead watering",
                   "clean up plant debris after the season and rotate crops",
                   "a protective fungicide if the disease keeps spreading, as per the label"],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "aliases": ["yellow leaf curl", "leaf curl", "tylcv", "curl virus", "yellow curl", "virus"],
        "cause": "*Tomato yellow leaf curl virus* (a begomovirus), carried by the **silverleaf whitefly** "
                 "(*Bemisia tabaci*)",
        "symptoms": "leaves **curl upwards**, are small and have **yellow edges**; the plant is stunted and sets "
                    "few fruit",
        "spread": "only by whiteflies feeding on the plant - not by touch or seed",
        "manage": ["there is **no cure** for an infected plant - remove and destroy it",
                   "control whiteflies: yellow sticky traps, insect-proof nets, reflective mulch",
                   "remove weeds that host whiteflies", "plant virus-resistant varieties"],
    },
    "Tomato___healthy": {
        "aliases": ["healthy"],
        "cause": "no disease",
        "symptoms": "uniform green colour, no spots, no curling",
        "spread": "-",
        "manage": ["keep monitoring the lower leaves every few days", "water at the base, mulch, rotate crops",
                   "remove any leaf that develops spots early"],
    },
}

# ============================================================================= names used in questions
MODEL_ALIASES = {
    "Gradient Boosting": ["gradient boosting", "boosting", "gbm", "gb", "hist gradient", "xgboost"],
    "Random Forest": ["random forest", "forest", "rf", "bagging"],
    "SVM (RBF kernel)": ["rbf", "rbf svm", "kernel svm", "svm rbf", "kernel trick"],
    "SVM (linear kernel)": ["linear svm", "linear kernel", "svm linear"],
    "Logistic Regression": ["logistic regression", "logistic", "logreg", "lr"],
    "LDA": ["lda", "linear discriminant"],
    "Gaussian Naive Bayes": ["naive bayes", "gaussian nb", "nb", "naive"],
    "KNN": ["knn", "k nearest", "nearest neighbour", "nearest neighbor", "k-nn"],
    "Decision Tree": ["decision tree", "tree", "cart"],
}
GROUP_ALIASES = {
    "rgb": ["rgb", "colour average", "color average", "mean colour", "mean color"],
    "indices": ["indices", "index", "gli", "greenness index", "lesion fraction", "exg"],
    "hsv": ["hsv", "histogram", "hue"],
    "texture": ["texture", "glcm", "haralick"],
    "shape": ["shape", "hu moment", "hu moments", "outline"],
}


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def _find(text: str, aliases: dict[str, list[str]]) -> str | None:
    """Return the key whose alias appears in the text (longest alias wins, whole words only)."""
    t = _norm(text)
    best, best_len = None, 0
    for key, words in aliases.items():
        for w in words:
            if f" {_norm(w).strip()} " in t and len(w) > best_len:
                best, best_len = key, len(w)
    return best


def find_disease(text: str) -> str | None:
    return _find(text, {k: v["aliases"] for k, v in DISEASES.items()})


def find_model(text: str) -> str | None:
    return _find(text, MODEL_ALIASES)


def find_group(text: str) -> str | None:
    return _find(text, GROUP_ALIASES)


# ============================================================================= result tables (live)
def table(name: str, reports_dir: Path | None = None) -> pd.DataFrame | None:
    path = config.tables_dir(reports_dir) / name
    return pd.read_csv(path) if path.exists() else None


def per_class_report(reports_dir: Path | None = None) -> pd.DataFrame | None:
    """Parse classification_report_best.txt into a table (class, precision, recall, f1, support)."""
    path = config.tables_dir(reports_dir) / "classification_report_best.txt"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(.+?)\s+([01]\.\d+)\s+([01]\.\d+)\s+([01]\.\d+)\s+(\d+)\s*$", line)
        if m and m.group(1).strip() not in ("macro avg", "weighted avg"):
            rows.append({"class": m.group(1).strip(), "precision": float(m.group(2)),
                         "recall": float(m.group(3)), "f1": float(m.group(4)), "support": int(m.group(5))})
    return pd.DataFrame(rows) if rows else None


# ============================================================================= answers
@dataclass
class Context:
    """Everything an answer may use."""
    prediction: object | None = None     # src.predict.Prediction of the current leaf, or None
    models: dict | None = None           # output of src.predict.load_models(), or None
    reports_dir: Path | None = None
    source: str | None = None            # file name of the current leaf photo


@dataclass
class Answer:
    text: str
    intent: str
    similarity: float
    suggestions: list[str] = field(default_factory=list)


NO_LEAF = ("I need a leaf photo for that. Upload one (or pick a sample) above, then ask again. "
           "Without a photo I can answer questions about the project, the models and the diseases.")
NO_TABLES = "There are no results yet - run `python -m src.classify` (and `python -m src.evaluate --all`) first."


def pretty(name: str) -> str:
    return config.pretty_class(name)


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


# ---------------------------------------------------------------- about the current leaf
def a_diagnosis(q: str, c: Context) -> str:
    p = c.prediction
    if p is None:
        return NO_LEAF
    status = "**healthy**" if p.is_healthy else "**diseased**"
    out = (f"The leaf looks {status}. Most likely: **{pretty(p.label)}** with **{pct(p.confidence, 0)}** "
           f"probability ({p.model_name}).\n\nTop-3: " +
           ", ".join(f"{pretty(n)} {pct(pr)}" for n, pr in p.top3) + ".")
    if p.low_confidence:
        out += f"\n\n⚠️ Confidence is below {pct(config.LOW_CONFIDENCE, 0)}, so the model is unsure."
    if not p.leaf_found:
        out += "\n\n⚠️ No leaf was detected in the photo, so this prediction is unreliable."
    return out


def a_healthy(q: str, c: Context) -> str:
    p = c.prediction
    if p is None:
        return NO_LEAF
    out = (f"{'Yes' if p.is_healthy else 'No'} - the model says **{pretty(p.label)}** "
           f"({pct(p.confidence, 0)}).")
    if p.healthy_probability is not None:
        out += f" Total probability of the healthy class: **{pct(p.healthy_probability)}**."
    if p.bayes_p_diseased is not None:
        out += (f"\n\nSecond opinion, Bayesian logistic regression: P(diseased) = **{pct(p.bayes_p_diseased)}**.")
        if (p.bayes_p_diseased >= 0.5) == p.is_healthy:
            out += " Note: the two models **disagree**, so treat this leaf with caution."
    out += f"\n\nLesion (brown) area: {p.lesion_percent:.1f}% of the leaf."
    return out


def a_confidence(q: str, c: Context) -> str:
    p = c.prediction
    if p is None:
        return NO_LEAF
    out = (f"The {p.model_name} model gives **{pretty(p.label)}** a probability of **{pct(p.confidence)}**. "
           f"The runner-up is {pretty(p.top3[1][0])} at {pct(p.top3[1][1])}.")
    out += (" That is **low** confidence - I would take a clearer photo or ask an expert."
            if p.low_confidence else " That is above our 60% warning threshold.")
    if p.bayes_confidence is not None:
        out += (f"\n\nThe Bayesian model's healthy-vs-diseased confidence is **{pct(p.bayes_confidence, 0)}**"
                + (f" - below {pct(config.BAYES_REFER_BELOW, 0)}, so it suggests **referring the leaf to an expert**."
                   if p.refer_to_expert else ", so it is sure."))
    return out


def a_lesion(q: str, c: Context) -> str:
    p = c.prediction
    if p is None:
        return NO_LEAF
    lp = p.lesion_percent
    level = ("very little - typical of healthy leaves (about 1%)" if lp < 2 else
             "mild" if lp < 6 else "moderate" if lp < 12 else "severe")
    return (f"**{lp:.1f}%** of the leaf area is brown lesion tissue ({level}).\n\n"
            "It is measured as brown pixels (HSV hue 5-24, saturation ≥ 70) divided by all leaf pixels, after "
            "segmenting the leaf. Average per class in our data: healthy 1.0%, yellow leaf curl 2.3%, "
            "Septoria 5.1%, early blight 9.0%, late blight 12.6%.")


def a_chlorophyll(q: str, c: Context) -> str:
    p = c.prediction
    calibrated = c.models is not None and c.models.get("chlorophyll") is not None
    if p is not None and p.spad is not None:
        out = f"Estimated chlorophyll: **{p.spad:.1f} SPAD** (calibrated model)."
        if p.spad_outside_range:
            out += " ⚠️ This is outside the range of the calibration readings, so it is unreliable."
        return out
    out = ""
    if p is not None:
        g = p.greenness_index
        out = (f"Greenness index (GLI) of this leaf: **{g:.3f}** (range −1 to 1, higher = greener). ")
    if not calibrated:
        out += ("This is an **uncalibrated colour index, not a chlorophyll value**. A real SPAD (chlorophyll) "
                "estimate needs real SPAD meter readings in `spad/spad_readings.csv`, then "
                "`python -m src.chlorophyll`. We never make up SPAD data.")
    return out.strip()


def a_why(q: str, c: Context) -> str:
    p = c.prediction
    if p is None:
        return ("Each prediction is explained with the logistic regression model: for a linear model the class "
                "score is b + Σ wⱼ·zⱼ, so wⱼ·zⱼ (weight × standardised feature) is exactly how much feature j pushed "
                "the score. Upload a leaf to see the explanation for that leaf.")
    if not p.group_explanation:
        return "Train the logistic regression model (`python -m src.classify`) to get explanations."
    groups = sorted(p.group_explanation.items(), key=lambda kv: -abs(kv[1]))
    out = [f"Why **{pretty(p.label)}**? Push of each feature group on that class score "
           "(logistic regression, weight × standardised value):"]
    for g, v in groups:
        out.append(f"- {g}: **{v:+.2f}** ({'supports' if v > 0 else 'argues against'})")
    if p.explanation:
        top = p.explanation[:3]
        out.append("\nStrongest single features: " + "; ".join(
            f"{e['feature']} = {e['value']:.3f} ({'supports' if e['contribution'] > 0 else 'against'}, "
            f"{e['contribution']:+.2f})" for e in top) + ".")
    out.append(f"\nLesion area {p.lesion_percent:.1f}% and greenness {p.greenness_index:.3f} are part of the "
               "'Greenness & lesion indices' group.")
    return "\n".join(out)


def a_bayes_leaf(q: str, c: Context) -> str:
    p = c.prediction
    if p is None or p.bayes_p_diseased is None:
        return a_bayes(q, c)
    out = (f"Bayesian logistic regression on this leaf: P(diseased) = **{pct(p.bayes_p_diseased)}** "
           f"(plain MAP logistic regression: {pct(p.bayes_p_diseased_map)}). Confidence "
           f"**{pct(p.bayes_confidence, 0)}**.")
    out += (f"\n\n🔎 Below {pct(config.BAYES_REFER_BELOW, 0)} - **refer this leaf to an expert**."
            if p.refer_to_expert else "\n\nThe model is confident, no expert referral needed.")
    return out


def _disease_for(q: str, c: Context) -> str | None:
    """The disease the question is about: named in it, else the current leaf's prediction."""
    d = find_disease(q)
    if d is None and c.prediction is not None:
        d = c.prediction.label
    return d


def a_treatment(q: str, c: Context) -> str:
    d = _disease_for(q, c)
    if d is None:
        return ("Which disease? Ask e.g. *'how do I treat early blight?'*, or upload a leaf and ask "
                "*'what should I do?'*. I know: " + ", ".join(pretty(k) for k in DISEASES) + ".")
    info = DISEASES.get(d)
    if info is None:
        return (f"The model predicted {pretty(d)}, but my knowledge base only covers the 5 tomato classes the "
                "model was trained on. Please check with a local agricultural extension officer.")
    head = (f"For **{pretty(d)}** ({info['cause']}):" if d != "Tomato___healthy"
            else "The leaf looks healthy. To keep it that way:")
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(info["manage"], 1))
    tail = ""
    if c.prediction is not None and find_disease(q) is None and c.prediction.low_confidence:
        tail = "\n\n⚠️ The diagnosis has low confidence - confirm it before treating."
    return (f"{head}\n{steps}{tail}\n\n*General guidance only - follow product labels and ask a local "
            "agricultural expert before spraying.*")


def a_disease_info(q: str, c: Context) -> str:
    d = _disease_for(q, c)
    if d is None:
        return ("The model knows these classes: " + ", ".join(pretty(k) for k in DISEASES) +
                ". Ask e.g. *'what is late blight?'*.")
    info = DISEASES.get(d)
    if info is None:
        return f"{pretty(d)} is outside my knowledge base (it covers the 5 trained tomato classes)."
    if d == "Tomato___healthy":
        return "A healthy tomato leaf: " + info["symptoms"] + "."
    out = (f"**{pretty(d)}** is caused by {info['cause']}.\n\n**Symptoms:** {info['symptoms']}.\n\n"
           f"**Spread:** {info['spread']}.")
    rep = per_class_report(c.reports_dir)
    if rep is not None:
        row = rep[rep["class"] == pretty(d)]
        if len(row):
            r = row.iloc[0]
            out += (f"\n\n**How well our model detects it:** recall {pct(r['recall'])}, precision "
                    f"{pct(r['precision'])} on {r['support']} held-out leaves.")
    out += "\n\nAsk *'how do I treat it?'* for management steps."
    return out


def a_classes(q: str, c: Context) -> str:
    names = list(DISEASES)
    if c.models is not None:
        names = c.models["best"]["class_names"]
    out = f"The current model recognises **{len(names)} classes**: " + ", ".join(pretty(n) for n in names) + "."
    if len(names) < len(config.CLASS_NAMES):
        out += (f" The code also supports the full {len(config.CLASS_NAMES)}-class Kaggle dataset "
                "(14 crops) once it is downloaded.")
    return out


# ---------------------------------------------------------------- about the project's results
def _results(c: Context) -> pd.DataFrame | None:
    return table("classification_results.csv", c.reports_dir)


def a_best_model(q: str, c: Context) -> str:
    res = _results(c)
    if res is None:
        return NO_TABLES
    res = res.sort_values("CV accuracy (%)", ascending=False)
    b = res.iloc[0]
    second = res.iloc[1]
    return (f"The best model is **{b['Model']}** ({b['Best params']}), chosen by 5-fold cross-validation: "
            f"**{b['CV accuracy (%)']:.1f} ± {b['CV std']:.1f}%** CV accuracy, **{b['Valid accuracy (%)']:.1f}%** "
            f"held-out accuracy, {b['Valid balanced acc (%)']:.1f}% balanced accuracy.\n\n"
            f"Runner-up: {second['Model']} at {second['CV accuracy (%)']:.1f}% CV. Boosting wins because it adds "
            "trees one after another, each correcting the errors of the trees before it (it reduces bias), "
            "whereas Random Forest averages independent trees (it reduces variance).")


def a_model_accuracy(q: str, c: Context) -> str:
    res = _results(c)
    if res is None:
        return NO_TABLES
    m = find_model(q)
    if m is None:
        return a_compare(q, c)
    row = res[res["Model"] == m]
    if not len(row):
        return f"{m} is not in the results table."
    r = row.iloc[0]
    rank = int((res["CV accuracy (%)"] > r["CV accuracy (%)"]).sum()) + 1
    return (f"**{m}** ({r['Syllabus']}): {r['Topic']}.\n\n"
            f"- Best settings (grid search): {r['Best params']}\n"
            f"- 5-fold CV accuracy: **{r['CV accuracy (%)']:.1f} ± {r['CV std']:.1f}%** (rank {rank} of {len(res)})\n"
            f"- Held-out accuracy: **{r['Valid accuracy (%)']:.1f}%**, balanced {r['Valid balanced acc (%)']:.1f}%, "
            f"macro F1 {r['Valid macro F1 (%)']:.1f}%\n"
            f"- Training time: {r['Time (s)']:.1f} s on {int(r['Train size']):,} leaves")


def a_compare(q: str, c: Context) -> str:
    res = _results(c)
    if res is None:
        return NO_TABLES
    res = res.sort_values("CV accuracy (%)", ascending=False)
    lines = ["| Model | CV accuracy | Held-out | Balanced |", "|---|---|---|---|"]
    for _, r in res.iterrows():
        lines.append(f"| {r['Model']} | {r['CV accuracy (%)']:.1f} ± {r['CV std']:.1f}% | "
                     f"{r['Valid accuracy (%)']:.1f}% | {r['Valid balanced acc (%)']:.1f}% |")
    return ("All models, ranked by 5-fold CV accuracy:\n\n" + "\n".join(lines) +
            "\n\nEnsembles win (single tree → Random Forest → Boosting), the RBF kernel beats the linear SVM, "
            "and Naive Bayes is last because its independence assumption is false (R and G are 91% correlated).")


def a_hardest(q: str, c: Context) -> str:
    rep = per_class_report(c.reports_dir)
    if rep is None:
        return NO_TABLES
    rep = rep.sort_values("recall")
    w, b = rep.iloc[0], rep.iloc[-1]
    lines = "\n".join(f"- {r['class']}: recall {pct(r['recall'])}, precision {pct(r['precision'])}"
                      for _, r in rep.iterrows())
    return (f"Hardest class: **{w['class']}** (recall {pct(w['recall'])}); easiest: **{b['class']}** "
            f"(recall {pct(b['recall'])}).\n\n{lines}\n\nEarly blight is mostly confused with late blight - both "
            "are brown-spot diseases, so their colour features overlap.")


def a_cv_vs_test(q: str, c: Context) -> str:
    return ("Why is held-out accuracy higher than CV accuracy?\n\n"
            "1. CV runs on a **class-balanced sample** (500 leaves per class), the test set keeps the natural class "
            "mix, which has many easy yellow-leaf-curl leaves (1,072 of 2,326).\n"
            "2. The final model is trained on **all 9,302** training leaves, more than each CV fold.\n\n"
            "That is why we also report **balanced accuracy** (mean recall over classes), which weighs every "
            "class equally.")


def a_training(q: str, c: Context) -> str:
    return ("How the models are trained and compared fairly:\n"
            "- Every model is a pipeline **StandardScaler → classifier**, so scaling is learned inside each "
            "CV fold (no data leakage).\n"
            "- Settings (C, γ, k, depth, learning rate...) are tuned by **grid search + 5-fold stratified "
            "cross-validation**.\n"
            "- The best model is picked by its **CV score**, then tested **once** on 2,326 held-out images.\n"
            "- Fixed random seed 42 everywhere, so results are reproducible.")


def a_features(q: str, c: Context) -> str:
    g = find_group(q)
    detail = {
        "rgb": "6 features: mean and standard deviation of R, G and B over leaf pixels (blotchy leaves have a high std).",
        "indices": "4 features: GLI = (2G−R−B)/(2G+R+B), G/R ratio, normalised Excess Green, and the lesion fraction "
                   "(brown pixels ÷ leaf pixels).",
        "hsv": "32 features: share of leaf pixels in 8 hue × 2 saturation × 2 brightness bins - captures yellowing "
               "and browning.",
        "texture": "6 GLCM (Haralick) features: contrast, dissimilarity, homogeneity, energy, correlation, ASM - "
                   "spots make the surface rough.",
        "shape": "7 Hu moments of the leaf outline, unchanged by moving, rotating or resizing - curled leaves differ.",
    }
    if g is not None:
        out = f"**{g.upper() if g in ('rgb', 'hsv') else g.capitalize()}** - {detail[g]}"
        lo = table("ablation_leave_one_group_out.csv", c.reports_dir)
        if lo is not None and g in set(lo["Removed group"]):
            r = lo[lo["Removed group"] == g].iloc[0]
            out += (f"\n\nRemoving this group changes RBF-SVM CV accuracy by **{-r['Drop (points)']:+.1f} points** "
                    f"({r['CV acc (%)']:.1f}% without it).")
        return out
    return ("Each leaf becomes **55 hand-made, explainable numbers**, measured on leaf pixels only:\n" +
            "\n".join(f"- {d}" for d in detail.values()) +
            "\n\nAsk *'what does texture add?'* for the ablation result of one group.")


def a_ablation(q: str, c: Context) -> str:
    g = find_group(q)
    if g is not None:
        return a_features(q, c)
    ab = table("ablation.csv", c.reports_dir)
    lo = table("ablation_leave_one_group_out.csv", c.reports_dir)
    if ab is None:
        return NO_TABLES
    lines = [f"- {r['Feature set']} ({r['# features']} features): **{r['SVM (RBF kernel) CV acc (%)']:.1f}%**"
             for _, r in ab.iterrows()]
    out = "Feature ablation (RBF SVM, 5-fold CV), adding groups one by one:\n" + "\n".join(lines)
    if lo is not None:
        lo = lo.sort_values("Drop (points)", ascending=False)
        out += ("\n\nLeave-one-group-out (drop in points when removed): " +
                ", ".join(f"{r['Removed group']} {r['Drop (points)']:+.1f}" for _, r in lo.iterrows()) +
                ". The HSV colour histogram matters most; shape adds almost nothing.")
    return out


def a_pipeline(q: str, c: Context) -> str:
    return ("The pipeline has 5 stages:\n"
            "1. **Preprocess** - resize to 256×256, 3×3 Gaussian blur to remove camera noise.\n"
            "2. **Segment** - HSV threshold for green + brown pixels → morphological closing → keep the largest "
            "connected region → fill holes → remove dark shadows = the leaf mask.\n"
            "3. **Features** - 55 numbers from leaf pixels only (colour, indices, HSV histogram, GLCM texture, "
            "Hu shape moments).\n"
            "4. **Models** - StandardScaler → classifier (9 compared, Gradient Boosting best), plus Bayesian "
            "logistic regression (healthy vs diseased with uncertainty) and chlorophyll regression.\n"
            "5. **App** - this Streamlit app: diagnosis, confidence, lesion %, explanation, and this Q&A.")


def a_segmentation(q: str, c: Context) -> str:
    out = ("Segmentation (`src/features.py`): convert to **HSV** (separates colour from brightness), keep pixels "
           "that are green (healthy, hue 25-95) or brown (lesions, hue 5-24, saturation ≥ 70), **morphological "
           "closing** to fill gaps, keep the **largest connected component**, **fill holes** so dark spots stay "
           "inside the leaf, and **remove shadows** (dark brown pixels outside the green outline). Features are "
           "then measured on leaf pixels only, so the background cannot fool the model.")
    if c.prediction is not None and not c.prediction.leaf_found:
        out += "\n\n⚠️ On this photo no leaf was found, so the whole image was used."
    return out


def a_dataset(q: str, c: Context) -> str:
    n = c.models["best"]["train_size"] if c.models is not None else 9302
    return (f"Dataset: **PlantVillage tomato leaves, 5 classes, 11,628 images** (lab photos: one leaf, plain "
            f"background). Stratified 80/20 split: **{n:,} training**, 2,326 held-out test images.\n\n"
            "Target: Kaggle *New Plant Diseases Dataset (Augmented)* - 38 classes, ~88,000 images. The code finds "
            "it automatically in `data/` once downloaded. Feature extraction takes about 4.4 ms per image.")


def a_unsupervised(q: str, c: Context) -> str:
    pca = table("unsup_pca.csv", c.reports_dir)
    cl = table("unsup_clustering.csv", c.reports_dir)
    bic = table("unsup_bic.csv", c.reports_dir)
    if pca is None or cl is None:
        return "Run `python -m src.unsupervised` first."
    k95 = pca[pca["Variance kept (%)"] >= 95].iloc[0]
    out = [f"**PCA:** {int(k95['PCA components'])} of 55 components keep {k95['Variance kept (%)']:.0f}% of the "
           f"variance (the features are correlated). Logistic regression on 2 PCs gets "
           f"{pca.iloc[0]['LR CV accuracy (%)']:.1f}%, on all 55 features {pca.iloc[-1]['LR CV accuracy (%)']:.1f}%."]
    out.append("**Clustering into 5 groups without labels:** " + "; ".join(
        f"{r['Method']} ARI {r['ARI']:.2f}, purity {pct(r['Purity'], 0)}" for _, r in cl.iterrows()) + ".")
    if bic is not None:
        best = bic.loc[bic["BIC"].idxmin()]
        out.append(f"**BIC** prefers **{int(best['GMM components'])} Gaussian components**, not 5 - the data has "
                   "more natural groups (e.g. two photo backgrounds) than diseases. EM raises the log-likelihood "
                   "every iteration, as theory says.")
    out.append("Message: the features separate diseases well **with** labels (92% supervised) but only weakly "
               "**without** them.")
    return "\n\n".join(out)


def a_bayes(q: str, c: Context) -> str:
    res = table("bayes_results.csv", c.reports_dir)
    acc = f"{res['Accuracy (%)'].iloc[-1]:.1f}%" if res is not None else "99.5%"
    return (f"**Bayesian logistic regression** (healthy vs diseased), written from scratch in NumPy:\n"
            "- Gaussian prior on the weights, w ~ N(0, α⁻¹I); α chosen by the evidence.\n"
            "- MAP weights by **Newton's method** (IRLS); they match scikit-learn to within 0.0004.\n"
            "- **Laplace approximation**: the Hessian at the MAP gives the posterior covariance S_N.\n"
            "- **Predictive** with the probit approximation σ(κ(s²)·μ), which pulls uncertain predictions "
            "towards 50%.\n\n"
            f"Held-out accuracy **{acc}**. Wrong predictions average 69% confidence, correct ones 99%, so "
            "sending the least-confident 20% of leaves to an expert gives 100% on the rest. On a disease it never "
            "saw, its log-loss is 25-30% lower than plain logistic regression.")


def a_hmm(q: str, c: Context) -> str:
    dec = table("hmm_decoding.csv", c.reports_dir)
    out = ("**Hidden Markov Model** (`src/hmm.py`, from scratch): each image row is read left to right like a "
           "sentence. Hidden state = background / healthy leaf / lesion; observation = the pixel's HSV colour bin. "
           "Transitions say neighbouring pixels are usually the same tissue. **Forward-backward** gives the "
           "probability of each state per pixel; **Viterbi** finds the single most likely state path.")
    if dec is not None:
        rows = {r["Decoder"].split(" (")[0]: r for _, r in dec.iterrows()}
        if "Per-pixel colour only" in rows and "Viterbi" in rows:
            a, v = rows["Per-pixel colour only"], rows["Viterbi"]
            out += (f"\n\nResult: per-pixel colour alone finds {a['Lesion recall (%)']:.0f}% of lesion pixels; "
                    f"Viterbi finds **{v['Lesion recall (%)']:.0f}%**, with {v['Pixel agreement (%)']:.1f}% pixel "
                    "agreement, because the HMM uses the neighbours.")
    return out


def a_chlorophyll_project(q: str, c: Context) -> str:
    vif = table("chlorophyll_vif.csv", c.reports_dir)
    out = ("Chlorophyll follows **Paper 2** (Anderson et al., 2026): a linear model b = A·X from leaf colour X to "
           "a SPAD meter reading b. Models compared: OLS, **Ridge** (λ by CV), Bayesian Ridge, polynomial, "
           "Random Forest.")
    if vif is not None:
        out += ("\n\nThe colour features are strongly collinear - VIF " +
                ", ".join(f"{r['Feature']} {r['VIF']:.0f}" for _, r in vif.iterrows()) +
                " (above 10 = unstable least squares), which is why Ridge's L2 penalty is needed.")
    out += "\n\n" + a_chlorophyll(q, Context(models=c.models))
    return out


def a_why_classical(q: str, c: Context) -> str:
    return ("Why classical ML and not deep learning?\n"
            "- Our course is classical ML (Bayes, discriminative/generative models, SVMs, ensembles, EM, HMMs).\n"
            "- Every feature is an **explainable** measurement, so we can say *why* a leaf was diagnosed.\n"
            "- It runs on a normal **laptop CPU** in milliseconds - no GPU, no huge dataset.\n"
            "- It still reaches **96% held-out accuracy** (Gradient Boosting).")


def a_limitations(q: str, c: Context) -> str:
    return ("Limitations:\n1. Trained on **lab photos** (one leaf, plain background) - field photos may be less "
            "accurate.\n2. Currently **5 tomato classes**; the 38-class run needs the Kaggle download.\n"
            "3. Chlorophyll needs **real SPAD readings**, calibrated per crop - until then only a greenness index.\n"
            "4. Shadows near the stem can still be counted as lesion.\n"
            "5. This Q&A matches questions to known topics; it does not understand free text like a chatbot.")


def a_assistant(q: str, c: Context) -> str:
    return ("I am a **retrieval assistant built with classical ML**: your question is turned into a TF-IDF "
            "vector (word and character n-grams) and matched by **cosine similarity** to example questions "
            "(a 1-nearest-neighbour classifier). Disease and model names are picked out by keywords. The answer is "
            "then computed live from the trained models (for the current leaf) and the result tables in "
            "`reports/tables/` - so if you retrain, my answers change too. Code: `src/assistant.py`.")


def a_greeting(q: str, c: Context) -> str:
    leaf = f" about **{c.source}**" if c.prediction is not None and c.source else ""
    return (f"Hi! Ask me anything{leaf}: *what disease is this?*, *how confident are you?*, "
            "*how do I treat it?*, *which model is best?*, *what is the accuracy of random forest?*")


def a_thanks(q: str, c: Context) -> str:
    return "You're welcome! Ask another question any time."


# ============================================================================= intents
# (name, answer function, example phrasings). Add phrasings here to teach the assistant new wordings.
INTENTS: list[tuple[str, Callable[[str, Context], str], list[str]]] = [
    ("diagnosis", a_diagnosis, [
        "what disease is this", "what disease does this leaf have", "diagnose this leaf", "what is wrong with my plant",
        "what is the prediction", "which disease", "identify the disease", "what does the model predict",
        "what is the result", "tell me the diagnosis", "what is the top 3", "what are the top predictions",
        "what is the second most likely disease", "what is this leaf", "what is the problem with this leaf"]),
    ("healthy", a_healthy, [
        "is this leaf healthy", "is my plant healthy", "is it diseased", "is the leaf sick", "is it infected",
        "healthy or diseased", "does this plant have a disease", "is it ok", "is anything wrong"]),
    ("confidence", a_confidence, [
        "how confident are you", "how sure is the model", "what is the confidence", "what is the probability",
        "can i trust this prediction", "how reliable is this", "how certain", "is the prediction reliable",
        "should i see an expert", "is this accurate for my leaf"]),
    ("lesion", a_lesion, [
        "how much of the leaf is damaged", "what is the lesion area", "lesion percentage", "how severe is it",
        "how bad is the infection", "how many spots", "how much brown", "damage percentage", "severity"]),
    ("chlorophyll", a_chlorophyll, [
        "what is the chlorophyll", "how green is the leaf", "spad value", "greenness index", "what is gli",
        "chlorophyll content of this leaf", "nitrogen level", "is the leaf green enough"]),
    ("why", a_why, [
        "why this prediction", "why did the model say this", "explain the prediction", "which features mattered",
        "what made the model decide", "why do you think so", "how did you decide", "explain why",
        "which features support the diagnosis", "how is the prediction explained", "explainability"]),
    ("bayes_leaf", a_bayes_leaf, [
        "what does the bayesian model say", "bayesian second opinion", "bayesian probability for this leaf",
        "refer to expert", "what is p diseased", "bayesian confidence"]),
    ("treatment", a_treatment, [
        "how do i treat it", "what should i do", "how to cure", "treatment", "how to control the disease",
        "what medicine", "which fungicide", "how to prevent it", "remedy", "how to save my plant",
        "how to manage early blight", "what pesticide should i spray", "how do i get rid of it",
        "prevention tips", "cure for this disease", "how to treat late blight", "how to treat septoria"]),
    ("disease_info", a_disease_info, [
        "what is early blight", "tell me about late blight", "what causes septoria", "symptoms of this disease",
        "what causes it", "how does it spread", "what is yellow leaf curl virus", "describe the disease",
        "is it contagious", "which pathogen causes it", "what are the symptoms", "information about the disease"]),
    ("classes", a_classes, [
        "which diseases can you detect", "what classes", "how many classes", "which plants are supported",
        "list the diseases", "what crops", "can you detect potato disease", "what can the model recognise"]),
    ("best_model", a_best_model, [
        "which model is best", "best model", "which classifier performs best", "what is the top model",
        "why is gradient boosting the best", "which algorithm did you choose", "what model is used",
        "which model is used for prediction", "what is the final model"]),
    ("model_accuracy", a_model_accuracy, [
        "what is the accuracy of random forest", "how good is svm", "accuracy of logistic regression",
        "how did knn do", "naive bayes accuracy", "decision tree result", "lda performance",
        "tell me about the svm model", "how well does gradient boosting perform", "what is the accuracy",
        "how accurate is the model", "overall accuracy", "what are the parameters of random forest"]),
    ("compare", a_compare, [
        "compare all models", "model comparison", "show all results", "results table", "how do the models compare",
        "rank the models", "which models did you try", "list of algorithms", "all classifier accuracies"]),
    ("hardest", a_hardest, [
        "which class is hardest", "which disease is confused", "per class accuracy", "precision recall f1",
        "which disease is easiest", "confusion matrix", "where does the model make mistakes", "recall per class",
        "which class has lowest recall", "worst class"]),
    ("cv_vs_test", a_cv_vs_test, [
        "why is test accuracy higher than cv", "why is held out higher than cross validation",
        "what is balanced accuracy", "why balanced accuracy", "test vs cv accuracy"]),
    ("training", a_training, [
        "how did you train the models", "what is cross validation", "how was hyperparameter tuning done",
        "grid search", "is there data leakage", "how did you choose parameters", "train test split",
        "why standard scaler", "how is the model evaluated", "random seed reproducible"]),
    ("features", a_features, [
        "what features do you use", "what are the 55 features", "explain the features", "what is glcm",
        "what are hu moments", "what is the hsv histogram", "feature extraction", "what is texture feature",
        "colour features", "how is the leaf converted to numbers", "feature engineering"]),
    ("ablation", a_ablation, [
        "feature ablation", "which feature group matters most", "what does texture add",
        "importance of each feature group", "leave one group out", "which features are most important",
        "feature importance", "does shape help"]),
    ("pipeline", a_pipeline, [
        "how does it work", "explain the pipeline", "what are the steps", "architecture", "explain the project",
        "overview of the project", "what is the workflow", "methodology", "how does the system work",
        "what is this project about"]),
    ("segmentation", a_segmentation, [
        "how is the leaf segmented", "what is segmentation", "how do you remove the background", "leaf mask",
        "how are lesions found", "hsv thresholding", "morphological closing", "shadow removal", "preprocessing"]),
    ("dataset", a_dataset, [
        "what dataset", "how many images", "where is the data from", "plantvillage", "kaggle dataset",
        "training data size", "how big is the dataset", "train test split size", "data source"]),
    ("unsupervised", a_unsupervised, [
        "pca results", "k means clustering", "gaussian mixture", "em algorithm", "unsupervised learning",
        "how many principal components", "clustering results", "bic", "does clustering find diseases",
        "dimensionality reduction", "gmm", "explain pca", "pca", "kmeans", "explain pca and kmeans"]),
    ("bayes", a_bayes, [
        "explain bayesian logistic regression", "laplace approximation", "what is the prior",
        "probit approximation", "newton method irls", "bayesian results", "why bayesian",
        "how is uncertainty estimated", "posterior distribution", "tell me about bayesian", "bayesian"]),
    ("hmm", a_hmm, [
        "hidden markov model", "what is viterbi", "forward backward algorithm", "hmm results", "explain the hmm",
        "transition matrix", "emission probabilities", "markov chain on pixels"]),
    ("chlorophyll_project", a_chlorophyll_project, [
        "how is chlorophyll estimated", "spad regression", "why ridge regression", "collinearity", "vif",
        "paper 2 method", "chlorophyll model", "ridge vs ols", "is the chlorophyll value real"]),
    ("why_classical", a_why_classical, [
        "why not deep learning", "why not cnn", "why classical machine learning", "why not neural networks",
        "advantages of your approach", "why not use a cnn"]),
    ("limitations", a_limitations, [
        "what are the limitations", "drawbacks", "weaknesses", "future work", "what can be improved",
        "does it work on field photos", "problems with the project"]),
    ("assistant", a_assistant, [
        "how do you answer questions", "how does this chatbot work", "how does the q and a work",
        "are you an ai", "who are you", "what can you do", "how do you understand my question", "help"]),
    ("greeting", a_greeting, ["hi", "hello", "hey", "good morning", "namaste", "hello there"]),
    ("thanks", a_thanks, ["thank you", "thanks", "great thanks", "ok thanks", "nice"]),
]

SUGGESTIONS_LEAF = ["What disease is this?", "How confident are you?", "How do I treat it?",
                    "Why this prediction?", "How much of the leaf is damaged?"]
SUGGESTIONS_PROJECT = ["Which model is best?", "Compare all models", "What features do you use?",
                       "Why not deep learning?", "Which class is hardest?", "What is early blight?"]


class Assistant:
    """TF-IDF + cosine-similarity nearest-neighbour intent matcher with dynamic answers."""

    def __init__(self, intents=INTENTS, min_similarity: float = MIN_SIMILARITY):
        self.min_similarity = min_similarity
        self.funcs = {name: fn for name, fn, _ in intents}
        self.example_intent = [name for name, _, examples in intents for _ in examples]
        examples = [_norm(e) for _, _, examples in intents for e in examples]
        self.vectorizer = FeatureUnion([
            ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])
        self.X = self._unit(self.vectorizer.fit_transform(examples))

    @staticmethod
    def _unit(X):
        X = X.tocsr().astype(float)
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1
        from scipy.sparse import diags
        return diags(1 / norms) @ X

    def match(self, question: str) -> list[tuple[str, float]]:
        """Intents ranked by the best cosine similarity of any of their examples."""
        q = self._unit(self.vectorizer.transform([_norm(question)]))
        sims = (self.X @ q.T).toarray().ravel()
        best: dict[str, float] = {}
        for intent, s in zip(self.example_intent, sims):
            best[intent] = max(best.get(intent, 0.0), float(s))
        return sorted(best.items(), key=lambda kv: -kv[1])

    def ask(self, question: str, context: Context | None = None) -> Answer:
        c = context or Context()
        question = (question or "").strip()
        sugg = SUGGESTIONS_LEAF if c.prediction is not None else SUGGESTIONS_PROJECT
        if not question:
            return Answer("Please type a question.", "empty", 0.0, sugg)
        ranked = self.match(question)
        intent, sim = ranked[0]
        # A named model in an "accuracy"-style question should go to that model, not the overall best.
        if intent in ("best_model", "compare") and find_model(question) and not re.search(r"\bbest\b|\bcompare|\ball\b", question.lower()):
            intent = "model_accuracy"
        if re.search(r"(treat|cure|spray|remedy|fungicide|pesticide|prevent|get rid)", question.lower()):
            intent = "treatment"
        if sim < self.min_similarity:
            return Answer("Sorry, I did not understand that. I can answer questions about the uploaded leaf, "
                          "the diseases, and the project's models and results. Try one of these:",
                          "unknown", sim, sugg)
        try:
            text = self.funcs[intent](question, c)
        except Exception as e:              # never crash the app because of one answer
            text = f"I matched your question to '{intent}' but could not build the answer ({e})."
        return Answer(text, intent, sim, [])


_ASSISTANT: Assistant | None = None


def get_assistant() -> Assistant:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = Assistant()
    return _ASSISTANT


def ask(question: str, context: Context | None = None) -> Answer:
    return get_assistant().ask(question, context)


# ============================================================================= command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ask questions about a leaf photo or the project.")
    ap.add_argument("question", nargs="*", help="a question (omit for an interactive chat)")
    ap.add_argument("--image", help="leaf photo to ask about")
    args = ap.parse_args(argv)

    from src import predict as P
    ctx = Context()
    try:
        ctx.models = P.load_models()
    except P.ModelsNotTrainedError as e:
        print(f"(no trained models: {e})")
    if args.image:
        if ctx.models is None:
            print("Cannot analyse the image without trained models.")
            return 1
        ctx.prediction = P.predict(args.image, ctx.models)
        ctx.source = Path(args.image).name

    def reply(q: str) -> None:
        a = ask(q, ctx)
        print(f"\n[{a.intent}, similarity {a.similarity:.2f}]\n{a.text}")
        for s in a.suggestions:
            print(f"  - {s}")

    if args.question:
        reply(" ".join(args.question))
        return 0
    print("Ask Leaf Doctor - type a question, or 'quit' to exit.")
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        reply(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
