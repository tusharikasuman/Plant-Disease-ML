"""
src/predict.py - predict disease (and greenness / chlorophyll) for one leaf photo.

Usage (from the plant-disease-ml folder):
    python -m src.predict samples/tomato_early_blight.jpg

Output: top-3 diseases with probabilities, healthy vs diseased, lesion %, and
  - a calibrated SPAD (chlorophyll) estimate, only if models/chlorophyll_model.joblib exists
    (it is trained in Phase 4 from REAL SPAD meter readings), otherwise
  - an UNCALIBRATED greenness index (GLI). This is NOT a chlorophyll value.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/predict.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

import config
from src import features as F


@dataclass
class Prediction:
    top3: list[tuple[str, float]]          # (class folder name, probability), best first
    label: str                             # predicted class folder name
    confidence: float                      # probability of the predicted class
    healthy_probability: float | None      # sum of probabilities of all "...healthy" classes
    is_healthy: bool
    lesion_percent: float                  # brown pixels / leaf pixels x 100
    greenness_index: float                 # GLI, -1..1 (higher = greener); NOT chlorophyll
    spad: float | None                     # calibrated SPAD, only with a real chlorophyll model
    leaf_found: bool
    model_name: str
    explanation: list[dict] = field(default_factory=list)   # most influential features
    group_explanation: dict[str, float] = field(default_factory=dict)   # summed push per feature group
    features: np.ndarray | None = None

    @property
    def low_confidence(self) -> bool:
        return self.confidence < config.LOW_CONFIDENCE


class ModelsNotTrainedError(FileNotFoundError):
    pass


def load_bundle(path: Path) -> dict | None:
    return joblib.load(path) if Path(path).exists() else None


def load_models(models_dir: Path = config.MODELS_DIR) -> dict:
    """Load the saved models. Raises ModelsNotTrainedError with instructions if Phase 2 has not run."""
    models_dir = Path(models_dir)
    best = load_bundle(models_dir / config.BEST_MODEL_PATH.name)
    if best is None:
        raise ModelsNotTrainedError(
            f"No trained model in {models_dir}. Train one first:  python -m src.classify")
    if best.get("feature_version") != config.FEATURE_VERSION:
        print("WARNING: the model was trained with an older features.py - retrain with python -m src.classify")
    return {
        "best": best,
        "logreg": load_bundle(models_dir / config.LOGREG_MODEL_PATH.name),
        "chlorophyll": load_bundle(models_dir / config.CHLOROPHYLL_MODEL_PATH.name),
    }


GROUP_LABELS = {"rgb": "Colour averages (RGB)", "indices": "Greenness & lesion indices",
                "hsv": "Colour histogram (HSV)", "texture": "Texture (GLCM)", "shape": "Shape (Hu moments)"}
FEATURE_GROUP = {F.FEATURE_NAMES[j]: g for g, cols in F.FEATURE_GROUPS.items() for j in cols}


def explain(logreg: dict | None, x: np.ndarray, label: str, top_k: int = 8) -> tuple[list[dict], dict]:
    """Which features pushed the leaf towards `label`, using the interpretable logistic regression.

    For a linear model the class score is  b_k + sum_j w_kj * z_j  (z = standardised feature), so
    w_kj * z_j is exactly how much feature j added to (or took away from) the score of class k.
    Correlated features (e.g. GLCM contrast and dissimilarity) can get large opposite weights that
    cancel, so we also return the summed contribution of each feature group, which is more stable.
    """
    if logreg is None or label not in logreg["class_names"]:
        return [], {}
    pipe = logreg["pipeline"]
    z = pipe.named_steps["scaler"].transform(x[None])[0]
    coef = pipe.named_steps["clf"].coef_
    k = logreg["class_names"].index(label)
    if coef.shape[0] == 1:                       # binary problem: one weight vector for class 1
        w = coef[0] if k == 1 else -coef[0]
    else:
        w = coef[k]
    contrib = w * z
    order = np.argsort(-np.abs(contrib))[:top_k]
    top = [{"feature": F.FEATURE_NAMES[j], "group": GROUP_LABELS[FEATURE_GROUP[F.FEATURE_NAMES[j]]],
            "value": float(x[j]), "z_score": float(z[j]), "weight": float(w[j]),
            "contribution": float(contrib[j])} for j in order]
    groups = {GROUP_LABELS[g]: float(contrib[cols].sum()) for g, cols in F.FEATURE_GROUPS.items()}
    return top, groups


def predict(image, models: dict | None = None) -> Prediction:
    """Predict for a file path, image bytes, or a BGR array."""
    models = models or load_models()
    if isinstance(image, (bytes, bytearray)):
        img = F.decode_image(bytes(image))
    elif isinstance(image, (str, Path)):
        img = F.read_image(image)
    else:
        img = image
    if img is None:
        raise ValueError("Could not read the image (is it a JPG/PNG file?)")

    x, seg = F.features_from_array(img)
    best = models["best"]
    names = best["class_names"]
    proba = best["pipeline"].predict_proba(x[None])[0]
    order = np.argsort(-proba)
    top3 = [(names[i], float(proba[i])) for i in order[:3]]
    label, confidence = top3[0]

    healthy = [i for i, n in enumerate(names) if config.is_healthy(n)]
    healthy_p = float(proba[healthy].sum()) if healthy else None

    spad = None
    if models.get("chlorophyll") is not None:
        spad = float(models["chlorophyll"]["pipeline"].predict(x[None])[0])

    top_features, groups = explain(models.get("logreg"), x, label)
    return Prediction(
        top3=top3, label=label, confidence=confidence,
        healthy_probability=healthy_p,
        is_healthy=config.is_healthy(label),
        lesion_percent=100 * float(x[F.LESION_INDEX]),
        greenness_index=float(x[F.GLI_INDEX]),
        spad=spad, leaf_found=seg.leaf_found,
        model_name=best["model_name"],
        explanation=top_features, group_explanation=groups,
        features=x,
    )


def format_prediction(p: Prediction) -> str:
    lines = [f"Model: {p.model_name}", "", "Top-3 predictions:"]
    for name, prob in p.top3:
        lines.append(f"  {config.pretty_class(name):45s} {100 * prob:5.1f}%")
    status = "HEALTHY" if p.is_healthy else "DISEASED"
    extra = f" (P(healthy) = {100 * p.healthy_probability:.1f}%)" if p.healthy_probability is not None else ""
    lines += ["", f"Status:      {status}{extra}", f"Lesion area: {p.lesion_percent:.1f}% of the leaf"]
    if p.spad is not None:
        lines.append(f"Chlorophyll: {p.spad:.1f} SPAD (calibrated model)")
    else:
        lines.append(f"Greenness:   {p.greenness_index:.3f} GLI (uncalibrated greenness index, -1..1 - "
                     f"NOT a chlorophyll / SPAD value)")
    if p.low_confidence:
        lines.append(f"WARNING: low confidence ({100 * p.confidence:.0f}% < {100 * config.LOW_CONFIDENCE:.0f}%)")
    if not p.leaf_found:
        lines.append("WARNING: no leaf was detected - the whole image was used")
    if p.group_explanation:
        lines += ["", f"Why '{config.pretty_class(p.label)}'? Push of each feature group (logistic regression):"]
        for g, c in sorted(p.group_explanation.items(), key=lambda kv: -abs(kv[1])):
            lines.append(f"  {g:28s} {c:+7.2f}  {'supports' if c > 0 else 'argues against'}")
    if p.explanation:
        lines += ["", "Most influential single features:"]
        for e in p.explanation[:5]:
            arrow = "supports" if e["contribution"] > 0 else "argues against"
            lines.append(f"  {e['feature']:20s} value {e['value']:8.3f}  {arrow} ({e['contribution']:+.2f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict plant disease for one leaf photo.")
    ap.add_argument("image", help="path to a leaf photo (JPG/PNG)")
    args = ap.parse_args(argv)
    try:
        print(format_prediction(predict(args.image)))
    except (ModelsNotTrainedError, ValueError) as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
