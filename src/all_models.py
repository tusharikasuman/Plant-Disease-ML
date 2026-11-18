"""
src/all_models.py - run ONE leaf photo through EVERY model and compare their answers.

The 9 classifiers of src/classify.py (with the settings grid search chose, from
reports/tables/classification_scores.json), the neural network and AdaBoost written from scratch in
src/unit5.py, and the Bayesian logistic regression (healthy vs diseased) are trained once on the 9,302
training leaves and saved to models/all_models.joblib (about 1-2 minutes, CPU).

Usage (from the plant-disease-ml folder):
    python -m src.all_models                                   # train and save every model
    python -m src.all_models samples/tomato_early_blight.jpg   # what does each model say about this leaf?

The Streamlit app's "Compare all models" page uses the same functions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

import config
from src import features as F

ALL_MODELS_PATH = config.MODELS_DIR / "all_models.joblib"


def short(name: str) -> str:
    """'Tomato___Tomato_Yellow_Leaf_Curl_Virus' -> 'Yellow leaf curl virus', 'Tomato___healthy' -> 'Healthy'."""
    s = config.pretty_class(name).replace("Tomato: ", "").replace("Tomato ", "")
    return s[:1].upper() + s[1:].lower()


class ScaledModel:
    """StandardScaler + a from-scratch model (MLP or AdaBoost) with a predict_proba like sklearn's."""

    def __init__(self, scaler, model, kind: str, n_classes: int):
        self.scaler, self.model, self.kind, self.n_classes = scaler, model, kind, n_classes

    def predict_proba(self, X):
        Z = self.scaler.transform(X)
        if self.kind == "mlp":
            return self.model.predict_proba(Z)
        votes = self.model.decision_function(Z)            # AdaBoost: weighted votes of the stumps
        return votes / np.maximum(votes.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def _tuned_params() -> dict:
    path = config.tables_dir() / "classification_scores.json"
    if not path.exists():
        return {}
    return {k: v["params"] for k, v in json.loads(path.read_text(encoding="utf-8"))["models"].items()}


def train_all(data: dict, path: Path = ALL_MODELS_PATH, verbose: bool = True) -> dict:
    """Fit every model on the training leaves, score it once on the held-out leaves, save the bundle."""
    from sklearn.preprocessing import StandardScaler

    from src import classify, unit5
    from src.all_models import ScaledModel        # save it as src.all_models.ScaledModel, also when run as __main__
    config.set_seeds()
    X, y, Xv, yv = data["X_train"], data["y_train"], data["X_valid"], data["y_valid"]
    k = len(data["class_names"])
    params = _tuned_params()
    models = []

    def add(name, unit, pipe, t0):
        acc = 100 * float(np.mean(pipe.predict(Xv) == yv))
        models.append({"name": name, "unit": unit, "model": pipe, "test_accuracy": acc})
        if verbose:
            print(f"  {name:38s} held-out accuracy {acc:5.1f}%   ({time.perf_counter() - t0:.0f} s)")

    if verbose:
        print(f"Training every model on {len(y):,} leaves, testing on {len(yv):,} held-out leaves...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in classify.model_specs():
            t0 = time.perf_counter()
            pipe, _, _ = classify.fit_final(spec, params.get(spec.key, {}), X, y, config.N_JOBS)
            idx = classify.balanced_sample(y, spec.max_per_class)
            pipe = classify.with_probabilities(pipe, X[idx], y[idx])     # SVMs: Platt-scaled probabilities
            add(spec.name, spec.unit.replace("Baseline (Paper 1)", "Paper 1 baseline / Unit 5 trees")
                .replace("Ensemble: boosting", "Unit 5: boosting"), pipe, t0)

        scaler = StandardScaler().fit(X)
        t0 = time.perf_counter()
        net = unit5.MLP(X.shape[1], 32, k).fit(scaler.transform(X), y, epochs=60)
        add("Neural network (from scratch, 32 units)", "Unit 5: neural networks", ScaledModel(scaler, net, "mlp", k), t0)
        t0 = time.perf_counter()
        ada = unit5.AdaBoostSAMME(n_rounds=300).fit(scaler.transform(X), y)
        add("AdaBoost (from scratch, 300 stumps)", "Unit 5: combining models", ScaledModel(scaler, ada, "ada", k), t0)

    bayes = None
    if config.BAYES_MODEL_PATH.exists():
        bayes = joblib.load(config.BAYES_MODEL_PATH)
    bundle = {"models": models, "class_names": list(data["class_names"]), "feature_version": config.FEATURE_VERSION,
              "bayes": bayes, "train_size": len(y), "test_size": len(yv)}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    if verbose:
        print(f"Saved {len(models)} models to {path}")
    return bundle


def load_all(path: Path = ALL_MODELS_PATH) -> dict | None:
    if not Path(path).exists():
        return None
    b = joblib.load(path)
    if b.get("feature_version") != config.FEATURE_VERSION:
        return None
    return b


def compare(image, bundle: dict) -> tuple[pd.DataFrame, dict]:
    """Every model's answer for one photo (path, bytes or BGR array). Returns (table, leaf info)."""
    if isinstance(image, (bytes, bytearray)):
        img = F.decode_image(bytes(image)) if len(image) else None
    elif isinstance(image, (str, Path)):
        img = F.read_image(image)
    else:
        img = image
    if img is None:
        raise ValueError("Could not read the image (is it a JPG/PNG file?)")
    x, seg = F.features_from_array(img)
    names = bundle["class_names"]
    healthy = [i for i, n in enumerate(names) if config.is_healthy(n)]
    rows = []
    for m in bundle["models"]:
        p = m["model"].predict_proba(x[None])[0]
        i = int(np.argmax(p))
        rows.append({"Model": m["name"] + (" *" if "AdaBoost" in m["name"] else ""), "Syllabus": m["unit"], "Prediction": short(names[i]),
                     "Healthy or diseased": "HEALTHY" if config.is_healthy(names[i]) else "DISEASED",
                     "Confidence (%)": round(100 * float(p[i]), 1), "P(healthy) (%)": round(100 * float(p[healthy].sum()), 1),
                     "Held-out accuracy (%)": round(m["test_accuracy"], 1)})
    b = bundle.get("bayes")
    if b is not None:
        z = b["scaler"].transform(x[None])
        pd_ = float(b["model"].predict_proba(z)[0, 1])
        rows.append({"Model": "Bayesian logistic regression (healthy vs diseased)", "Syllabus": "Unit 2: Bayesian LR",
                     "Prediction": "Diseased (any disease)" if pd_ >= 0.5 else "Healthy",
                     "Healthy or diseased": "DISEASED" if pd_ >= 0.5 else "HEALTHY",
                     "Confidence (%)": round(100 * max(pd_, 1 - pd_), 1), "P(healthy) (%)": round(100 * (1 - pd_), 1),
                     "Held-out accuracy (%)": 99.5})
    table = pd.DataFrame(rows)
    info = {"lesion_percent": 100 * float(x[F.LESION_INDEX]), "leaf_found": seg.leaf_found,
            "diseased_votes": int((table["Healthy or diseased"] == "DISEASED").sum()), "n_models": len(table),
            "majority_disease": table.loc[table["Healthy or diseased"] == "DISEASED", "Prediction"].mode().tolist()}
    return table, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one leaf photo through every model.")
    ap.add_argument("image", nargs="?", help="leaf photo; omit to (re)train and save every model")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    bundle = None if args.image is None else load_all()
    if bundle is None:
        from src import dataset
        try:
            data = dataset.load_features()
        except FileNotFoundError as e:
            print(e)
            return 1
        bundle = train_all(data)
    if args.image is None:
        return 0
    try:
        table, info = compare(args.image, bundle)
    except ValueError as e:
        print(e)
        return 1
    pd.set_option("display.width", 200, "display.max_colwidth", 55)
    print(f"\n{Path(args.image).name}: lesion area {info['lesion_percent']:.1f}% of the leaf"
          + ("" if info["leaf_found"] else "   (WARNING: no leaf found - unreliable)"))
    print(table[["Model", "Prediction", "Healthy or diseased", "Confidence (%)", "Held-out accuracy (%)"]].to_string(index=False))
    v, n = info["diseased_votes"], info["n_models"]
    print("* AdaBoost's 'confidence' is the share of its 300 stumps' weighted votes that went to that class.")
    print(f"\nVerdict: {v} of {n} models say DISEASED, {n - v} say HEALTHY"
          + (f"; most common disease: {info['majority_disease'][0]}" if info["majority_disease"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
