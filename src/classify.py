"""
src/classify.py - Phase 2: disease classification with classical ML models.

Usage (from the plant-disease-ml folder):
    python -m src.classify --limit 50         # quick test on the 50-images-per-class cache
    python -m src.classify                    # full run on cache/features.npz
    python -m src.classify --models lr,svm_rbf --no-ablation

Protocol (the same for every model)
  1. Pipeline = StandardScaler -> classifier, so scaling is learned inside each CV fold (no leakage).
  2. GridSearchCV with 5-fold stratified CV on a class-balanced sample of the training set
     (<= CV_PER_CLASS leaves per class) picks C / gamma / lambda.  Its mean +- std is the CV score.
  3. The tuned pipeline is refit on the whole training set (kernel SVM: <= SVM_MAX_PER_CLASS per class)
     and scored once on the held-out valid set.
  4. The best model is chosen by CV score (not valid score, so the valid number stays an honest
     estimate) and saved with joblib, together with the logistic regression model.
  5. Feature ablation: RGB stats -> + indices + HSV -> + texture -> + shape, same CV folds.

If the feature cache does not exist yet it is built first (see src/dataset.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/classify.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import config
from src import features
from src.evaluate import classification_metrics, save_table, text_table


# ----------------------------------------------------------------------------- the models
@dataclass
class ModelSpec:
    key: str
    name: str
    unit: str            # syllabus unit
    topic: str           # what the model demonstrates
    estimator: object
    grid: dict = field(default_factory=dict)       # hyper-parameter grid (keys without the clf__ prefix)
    max_per_class: int | None = None               # cap on the final training set (kernel SVM only)


def model_specs() -> list[ModelSpec]:
    """Fresh, untrained model specifications. Small grids so tuning stays fast on a laptop."""
    s = config.SEED
    return [
        ModelSpec("lr", "Logistic Regression", "Unit 2",
                  "Probabilistic discriminative model; softmax fitted by MLE with L2 penalty (lambda = 1/C)",
                  LogisticRegression(max_iter=5000, random_state=s),
                  {"C": [0.01, 0.1, 1, 10, 100]}),
        ModelSpec("lda", "LDA", "Unit 2",
                  "Linear discriminant functions; probabilistic generative model (Gaussians, shared covariance)",
                  LinearDiscriminantAnalysis(solver="lsqr"),
                  {"shrinkage": [None, 0.01, 0.1, "auto"]}),
        ModelSpec("gnb", "Gaussian Naive Bayes", "Unit 1",
                  "Bayes' rule with class-conditional independence of features",
                  GaussianNB(),
                  {"var_smoothing": [1e-9, 1e-6, 1e-3]}),
        ModelSpec("svm_linear", "SVM (linear kernel)", "Unit 2",
                  "Maximum-margin classifier, linear kernel (LinearSVC solver)",
                  LinearSVC(dual="auto", max_iter=10000, random_state=s),
                  {"C": [0.01, 0.1, 1, 10]}),
        ModelSpec("svm_rbf", "SVM (RBF kernel)", "Unit 2",
                  "Kernel trick: RBF kernel = maximum margin in an implicit feature space",
                  SVC(kernel="rbf"),
                  {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]},
                  max_per_class=config.SVM_MAX_PER_CLASS),
        ModelSpec("knn", "KNN", "Baseline (Paper 1)",
                  "Instance-based: majority vote of the k nearest leaves",
                  KNeighborsClassifier(),
                  {"n_neighbors": [3, 5, 9, 15], "weights": ["uniform", "distance"]}),
        ModelSpec("tree", "Decision Tree", "Baseline (Paper 1)",
                  "Axis-aligned if/else splits chosen by Gini impurity",
                  DecisionTreeClassifier(random_state=s),
                  {"max_depth": [10, 20, None], "min_samples_leaf": [1, 5]}),
        ModelSpec("rf", "Random Forest", "Baseline (Paper 1)",
                  "Bagged ensemble of decision trees with random feature subsets",
                  RandomForestClassifier(n_estimators=200, random_state=s, n_jobs=1),
                  {"max_depth": [None, 20], "min_samples_leaf": [1, 2]}),
        ModelSpec("gb", "Gradient Boosting", "Ensemble: boosting",
                  "Trees added one after another, each fitting the errors of the ensemble so far",
                  HistGradientBoostingClassifier(max_iter=300, random_state=s),
                  {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    ]


MODEL_KEYS = [m.key for m in model_specs()]

# cumulative feature sets for the ablation study
ABLATION_STEPS = [
    ("RGB stats only", ["rgb"]),
    ("+ indices + HSV histogram", ["rgb", "indices", "hsv"]),
    ("+ texture (GLCM)", ["rgb", "indices", "hsv", "texture"]),
    ("+ shape (Hu moments) = all", ["rgb", "indices", "hsv", "texture", "shape"]),
]


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", clone(estimator))])


# ----------------------------------------------------------------------------- helpers
def balanced_sample(y: np.ndarray, per_class: int | None, seed: int = config.SEED) -> np.ndarray:
    """Sorted indices with at most per_class examples of every class (reproducible)."""
    if per_class is None:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(y == c), size=min(per_class, int((y == c).sum())), replace=False)
           for c in np.unique(y)]
    return np.sort(np.concatenate(idx))


def cv_splitter(y: np.ndarray) -> StratifiedKFold:
    folds = min(config.CV_FOLDS, int(np.bincount(y).min()))     # tiny test datasets may have < 5 per class
    if folds < 2:
        raise ValueError("Every class needs at least 2 training images for cross-validation.")
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.SEED)


def format_params(params: dict) -> str:
    if not params:
        return "-"
    parts = []
    for k, v in params.items():
        text = f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
        if k == "C":
            text += f" (lambda={1 / v:g})"
        parts.append(text)
    return ", ".join(parts)


def tune(spec: ModelSpec, X: np.ndarray, y: np.ndarray, cv, n_jobs: int) -> dict:
    """Grid search with stratified k-fold CV. Returns best params and their CV fold scores."""
    grid = {f"clf__{k}": v for k, v in spec.grid.items()}
    search = GridSearchCV(make_pipeline(spec.estimator), grid, cv=cv, scoring="accuracy",
                          n_jobs=n_jobs, refit=False, error_score="raise")
    t0 = time.perf_counter()
    search.fit(X, y)
    res = search.cv_results_
    best = int(np.flatnonzero(res["rank_test_score"] == 1)[0])
    folds = np.array([res[f"split{i}_test_score"][best] for i in range(cv.get_n_splits())])
    params = {k.removeprefix("clf__"): v for k, v in res["params"][best].items()}
    return {"params": params, "fold_scores": folds, "seconds": time.perf_counter() - t0}


def fit_final(spec: ModelSpec, params: dict, X: np.ndarray, y: np.ndarray, n_jobs: int):
    """Refit the tuned pipeline on the (possibly capped) training set."""
    idx = balanced_sample(y, spec.max_per_class)
    pipe = make_pipeline(spec.estimator).set_params(**{f"clf__{k}": v for k, v in params.items()})
    if isinstance(pipe.named_steps["clf"], (RandomForestClassifier, KNeighborsClassifier)):
        pipe.set_params(clf__n_jobs=n_jobs)          # these two use several cores when fitting/predicting
    t0 = time.perf_counter()
    pipe.fit(X[idx], y[idx])
    return pipe, len(idx), time.perf_counter() - t0


def with_probabilities(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Models without predict_proba (SVMs) get Platt-scaled probabilities via CalibratedClassifierCV."""
    if hasattr(pipe, "predict_proba"):
        return pipe
    clf = clone(pipe.named_steps["clf"])
    calibrated = Pipeline([("scaler", StandardScaler()),
                           ("clf", CalibratedClassifierCV(clf, method="sigmoid", cv=3, ensemble=False))])
    return calibrated.fit(X, y)


def save_bundle(path: Path, pipe: Pipeline, spec: ModelSpec, row: dict, data: dict, train_size: int) -> None:
    """Save a model together with everything predict.py needs to use and describe it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "model_key": spec.key,
        "model_name": spec.name,
        "class_names": data["class_names"],
        "feature_names": data["feature_names"],
        "feature_version": config.FEATURE_VERSION,
        "metrics": {k: row[k] for k in ("CV accuracy (%)", "CV std", "Valid accuracy (%)",
                                        "Valid balanced acc (%)", "Valid macro F1 (%)")},
        "params": row["_params"],
        "train_size": train_size,
        "dataset": data.get("meta", {}),
    }, path, compress=3)


# ----------------------------------------------------------------------------- main experiment
def run_models(data: dict, model_keys: list[str] | None = None, cv_per_class: int | None = config.CV_PER_CLASS,
               n_jobs: int = config.N_JOBS) -> tuple[pd.DataFrame, dict]:
    """Tune, cross-validate and test every model. Returns (results table, fitted pipelines by key)."""
    X, y = data["X_train"], data["y_train"]
    Xv, yv = data["X_valid"], data["y_valid"]
    cv_idx = balanced_sample(y, cv_per_class)
    Xc, yc = X[cv_idx], y[cv_idx]
    cv = cv_splitter(yc)
    print(f"Grid search + {cv.get_n_splits()}-fold CV on {len(cv_idx)} leaves "
          f"(<= {cv_per_class} per class); final fit on {len(y)} train; test on {len(yv)} valid.\n")

    specs = [s for s in model_specs() if model_keys is None or s.key in model_keys]
    rows, fitted = [], {}
    for spec in specs:
        print(f"  {spec.name:22s} tuning...", end="", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            t = tune(spec, Xc, yc, cv, n_jobs)
            pipe, n_train, fit_s = fit_final(spec, t["params"], X, y, n_jobs)
        m = classification_metrics(yv, pipe.predict(Xv))
        fitted[spec.key] = pipe
        rows.append({
            "Model": spec.name, "Syllabus": spec.unit, "Topic": spec.topic,
            "Best params": format_params(t["params"]),
            "CV accuracy (%)": 100 * t["fold_scores"].mean(), "CV std": 100 * t["fold_scores"].std(),
            "Valid accuracy (%)": m["accuracy"], "Valid balanced acc (%)": m["balanced_accuracy"],
            "Valid macro F1 (%)": m["macro_f1"], "Train size": n_train,
            "Time (s)": t["seconds"] + fit_s,
            "_key": spec.key, "_params": t["params"], "_folds": t["fold_scores"].tolist(),
        })
        print(f" CV {rows[-1]['CV accuracy (%)']:.2f} +- {rows[-1]['CV std']:.2f}   "
              f"valid {m['accuracy']:.2f}   ({rows[-1]['Time (s)']:.0f} s)")
    return pd.DataFrame(rows), fitted


def run_ablation(data: dict, params: dict[str, dict], cv_per_class: int | None = config.CV_PER_CLASS,
                 n_jobs: int = config.N_JOBS) -> pd.DataFrame:
    """CV accuracy of the tuned SVM (RBF) and Logistic Regression as feature groups are added."""
    from sklearn.model_selection import cross_val_score
    X, y = data["X_train"], data["y_train"]
    idx = balanced_sample(y, cv_per_class)
    Xc, yc = X[idx], y[idx]
    cv = cv_splitter(yc)
    specs = {s.key: s for s in model_specs()}
    rows = []
    for label, groups in ABLATION_STEPS:
        cols = [i for g in groups for i in features.FEATURE_GROUPS[g]]
        row = {"Feature set": label, "Groups": " + ".join(groups), "# features": len(cols)}
        for key in ("svm_rbf", "lr"):
            if key not in params:
                continue
            pipe = make_pipeline(specs[key].estimator).set_params(
                **{f"clf__{k}": v for k, v in params[key].items()})
            s = cross_val_score(pipe, Xc[:, cols], yc, cv=cv, n_jobs=n_jobs)
            row[f"{specs[key].name} CV acc (%)"] = 100 * s.mean()
            row[f"{specs[key].name} std"] = 100 * s.std()
        rows.append(row)
    return pd.DataFrame(rows)


def run(data: dict, model_keys: list[str] | None = None, cv_per_class: int | None = config.CV_PER_CLASS,
        ablation: bool = True, n_jobs: int = config.N_JOBS,
        reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR) -> pd.DataFrame:
    """The whole of Phase 2: models, tables, saved models, ablation."""
    config.set_seeds()
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    results, fitted = run_models(data, model_keys, cv_per_class, n_jobs)

    # ---- pick the best model by CV accuracy, save it (with probabilities) + logistic regression
    best_row = results.loc[results["CV accuracy (%)"].idxmax()]
    specs = {s.key: s for s in model_specs()}
    best_spec = specs[best_row["_key"]]
    X, y = data["X_train"], data["y_train"]
    idx = balanced_sample(y, best_spec.max_per_class)
    best_pipe = with_probabilities(fitted[best_spec.key], X[idx], y[idx])
    save_bundle(models_dir / config.BEST_MODEL_PATH.name, best_pipe, best_spec, best_row.to_dict(), data, len(idx))
    if "lr" in fitted:
        lr_row = results.loc[results["_key"] == "lr"].iloc[0].to_dict()
        save_bundle(models_dir / config.LOGREG_MODEL_PATH.name, fitted["lr"], specs["lr"], lr_row, data, len(y))

    # ---- tables
    results = results.assign(Best=np.where(results["_key"] == best_spec.key, "*", ""))
    show = ["Model", "Syllabus", "Best params", "CV accuracy (%)", "CV std", "Valid accuracy (%)",
            "Valid balanced acc (%)", "Valid macro F1 (%)", "Train size", "Time (s)", "Best"]
    print("\n" + text_table(results[show]))
    save_table(results[show + ["Topic"]], "classification_results", reports_dir,
               "Disease classification - 5-fold stratified CV and held-out valid set")
    save_table(results[["Model", "Syllabus", "Topic"]], "syllabus_mapping", reports_dir,
               "Which syllabus topic each model demonstrates")

    names = [config.pretty_class(c) for c in data["class_names"]]
    report = classification_report(data["y_valid"], best_pipe.predict(data["X_valid"]),
                                   labels=np.arange(len(names)), target_names=names, digits=3, zero_division=0)
    (config.tables_dir(reports_dir) / "classification_report_best.txt").write_text(
        f"Best model (by CV accuracy): {best_spec.name}  [{format_params(best_row['_params'])}]\n"
        f"Valid set: {len(data['y_valid'])} leaves ({data.get('valid_source', '')})\n\n{report}", encoding="utf-8")
    print(f"\nBest model by CV accuracy: {best_spec.name}  ->  saved to {models_dir / config.BEST_MODEL_PATH.name}")

    scores = {"best_model": best_spec.key, "cv_per_class": cv_per_class,
              "class_names": data["class_names"],
              "models": {r["_key"]: {"name": r["Model"], "params": r["_params"], "cv_fold_scores": r["_folds"],
                                     "valid_accuracy": r["Valid accuracy (%)"]} for r in results.to_dict("records")}}

    if ablation:
        params = {k: results.loc[results["_key"] == k, "_params"].iloc[0]
                  for k in ("svm_rbf", "lr") if k in fitted}
        if params:
            print("\nFeature ablation (cumulative feature groups, same CV folds)...")
            abl = run_ablation(data, params, cv_per_class, n_jobs)
            print(text_table(abl))
            save_table(abl, "ablation", reports_dir, "Feature ablation - 5-fold CV accuracy as feature groups are added")
            scores["ablation"] = abl.to_dict("records")

    (config.tables_dir(reports_dir) / "classification_scores.json").write_text(json.dumps(scores, indent=1, default=str),
                                                            encoding="utf-8")
    print(f"Tables written to {reports_dir}")
    return results


def load_or_build_features(limit: int | None, data_dir: str | None, n_jobs: int) -> dict | None:
    from src import dataset
    try:
        return dataset.load_features(limit)
    except FileNotFoundError:
        print("No feature cache yet - extracting features first.\n")
        loc = dataset.find_dataset(data_dir)
        if loc is None:
            return None
        dataset.build_feature_cache(loc, limit=limit, n_jobs=n_jobs)
        return dataset.load_features(limit)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train and compare disease classifiers on the cached features.")
    ap.add_argument("--limit", type=int, default=None, help="use the --limit N feature cache (images per class)")
    ap.add_argument("--data", help="dataset folder, only needed if features must be extracted first")
    ap.add_argument("--models", default=None,
                    help=f"comma-separated subset of: {','.join(MODEL_KEYS)} (default: all)")
    ap.add_argument("--cv-per-class", type=int, default=config.CV_PER_CLASS,
                    help=f"leaves per class for grid search + CV (default {config.CV_PER_CLASS}, 0 = all)")
    ap.add_argument("--no-ablation", action="store_true", help="skip the feature ablation study")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help=f"parallel jobs (default {config.N_JOBS})")
    args = ap.parse_args(argv)

    keys = None
    if args.models:
        keys = [k.strip() for k in args.models.split(",")]
        unknown = set(keys) - set(MODEL_KEYS)
        if unknown:
            ap.error(f"unknown model(s): {', '.join(sorted(unknown))}. Choose from {', '.join(MODEL_KEYS)}")

    data = load_or_build_features(args.limit, args.data, args.jobs)
    if data is None:
        return 1
    print(f"Features: {len(data['y_train'])} train + {len(data['y_valid'])} valid leaves, "
          f"{len(data['class_names'])} classes, {len(data['feature_names'])} features.")
    if data["meta"].get("feature_version") != config.FEATURE_VERSION:
        print("WARNING: the feature cache was built with an older features.py - rebuild with "
              "python -m src.dataset --force")
    run(data, keys, args.cv_per_class or None, ablation=not args.no_ablation, n_jobs=args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
