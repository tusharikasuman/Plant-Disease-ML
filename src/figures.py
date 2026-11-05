"""
src/figures.py - every graph of the supervised part of the project, one function per graph.

Each graph is saved to reports/figures/<topic folder>/ (folders are listed in config.FIG):
  01_feature_representation  segmentation examples, key features per class, 55 x 55 correlation
  05_decision_trees          depth vs over-fitting, the tree itself (depth 3)
  06_bagging_random_forest   accuracy vs number of trees, feature importance
  07_boosting                accuracy after each boosting round (3 learning rates), bagging vs boosting
  08_knn                     accuracy vs k, with and without feature scaling
  09_cv_model_selection      model comparison, per-fold CV scores, SVM grid search heatmap, learning curve
  10_evaluation_metrics      confusion matrix, precision / recall / F1 per class, ROC curves
  11_ablation                cumulative ablation, leave-one-group-out ablation
(PCA, k-means and GMM graphs are made by src/unsupervised.py, Bayesian ones by src/bayes_logreg.py,
 chlorophyll ones by src/chlorophyll.py.)

Curves that need many refits use the same class-balanced sample as the grid search in src/classify.py
(<= CV_PER_CLASS leaves per class) and 5-fold stratified CV, so they stay fast on a laptop.
Run everything with:  python -m src.evaluate
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import features as F
from src.evaluate import COLORS as C
from src.evaluate import apply_style, save_fig


def _plt():
    apply_style()
    import matplotlib.pyplot as plt
    return plt


def _short(name: str) -> str:
    return config.pretty_class(name).replace("Tomato: ", "").replace("Tomato ", "")


def _class_color(name: str) -> str:
    return C["healthy"] if config.is_healthy(name) else C["diseased"]


def _cv():
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)


def _cv_sample(data: dict):
    from src.classify import balanced_sample
    idx = balanced_sample(data["y_train"], config.CV_PER_CLASS)
    return data["X_train"][idx], data["y_train"][idx]


def _tuned_params(reports_dir: Path) -> dict:
    path = config.tables_dir(reports_dir) / "classification_scores.json"
    if not path.exists():
        return {}
    return {k: v["params"] for k, v in json.loads(path.read_text(encoding="utf-8"))["models"].items()}


def _pipe(estimator):
    from src.classify import make_pipeline
    return make_pipeline(estimator)


def _heat_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9dcf3", C["blue"], "#173f73"])


# ============================================================================ 01 feature representation
def segmentation_examples(reports_dir, image_paths=None):
    plt = _plt()
    image_paths = image_paths or sorted(config.SAMPLES_DIR.glob("*.jpg"))
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(9, 3.1 * len(image_paths)), squeeze=False)
    for r, path in enumerate(image_paths):
        v = F.segmented_views(path)
        x, _ = F.extract_features(path)
        name = Path(path).stem.replace("tomato_", "").replace("_", " ")
        for c, (img, title) in enumerate([(v["image"], f"{name}: original"),
                                          (v["mask"], "leaf mask (white = leaf)"),
                                          (v["lesions"], f"lesions in orange: {100 * x[F.LESION_INDEX]:.1f}% of leaf")]):
            axes[r, c].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[r, c].set_title(title, fontsize=9.5); axes[r, c].axis("off")
    fig.suptitle("Segmentation: HSV threshold -> closing -> largest component -> fill holes",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save_fig(fig, "features", "1_segmentation_examples", reports_dir)


def feature_boxplots(data, reports_dir):
    """Six explainable features, one box per disease: what the model 'sees'."""
    plt = _plt()
    X, y, names = data["X_train"], data["y_train"], data["class_names"]
    feats = [("Lesion fraction", 100, "lesion area (% of leaf)"), ("GLI", 1, "Green Leaf Index"),
             ("G std", 1, "spread of green (blotchiness)"), ("GLCM contrast", 1, "texture contrast"),
             ("GLCM homogeneity", 1, "texture homogeneity"), ("Hu moment 1", 1, "shape: Hu moment 1")]
    order = np.argsort([X[y == k, F.LESION_INDEX].mean() for k in range(len(names))])   # healthy first
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for ax, (feat, scale, label) in zip(axes.flat, feats):
        j = F.FEATURE_NAMES.index(feat)
        vals = [scale * X[y == k, j] for k in order]
        bp = ax.boxplot(vals, patch_artist=True, showfliers=False, widths=0.6)
        for box, k in zip(bp["boxes"], order):
            box.set_facecolor(_class_color(names[k])); box.set_alpha(0.8); box.set_edgecolor(C["ink"])
        for part in ("whiskers", "caps", "medians"):
            for line in bp[part]:
                line.set_color(C["ink"])
        ax.set_xticks(range(1, len(order) + 1), [_short(names[k]) for k in order], rotation=25, ha="right",
                      fontsize=8)
        ax.set_title(f"{feat}", fontsize=10); ax.set_ylabel(label, fontsize=8.5); ax.grid(axis="x", visible=False)
    fig.suptitle("Six of the 55 features, per class (aqua = healthy, orange = diseased; middle 50% of leaves "
                 "in each box)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save_fig(fig, "features", "2_key_features_by_class", reports_dir)


def feature_correlation(data, reports_dir):
    """55 x 55 correlation matrix, with the 5 feature groups marked."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap
    corr = np.nan_to_num(np.corrcoef(data["X_train"].T))
    div = LinearSegmentedColormap.from_list("div", [C["blue"], "#f4f3ef", C["orange"]])
    fig, ax = plt.subplots(figsize=(10.5, 9.2))
    im = ax.imshow(corr, cmap=div, vmin=-1, vmax=1)
    n = len(F.FEATURE_NAMES)
    ax.set_xticks(range(n), F.FEATURE_NAMES, rotation=90, fontsize=5.5)
    ax.set_yticks(range(n), F.FEATURE_NAMES, fontsize=5.5); ax.grid(False)
    for g, cols in F.FEATURE_GROUPS.items():
        lo, hi = cols[0] - 0.5, cols[-1] + 0.5
        ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec=C["ink"], lw=1.2))
        ax.text(hi + 0.3, lo + 0.2, g, fontsize=8, va="top", color=C["ink"])
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson correlation r")
    strong = int((np.abs(corr[np.triu_indices(n, 1)]) > 0.8).sum())
    ax.set_title(f"Correlation of all 55 features ({strong} pairs with |r| > 0.8) - boxes = feature groups")
    return save_fig(fig, "features", "3_feature_correlation_55", reports_dir)


# ============================================================================ 05 decision trees
def tree_depth_curve(data, reports_dir, n_jobs):
    from sklearn.model_selection import cross_validate
    from sklearn.tree import DecisionTreeClassifier
    plt = _plt()
    Xc, yc = _cv_sample(data)
    depths = list(range(1, 21))
    tr, te, sd = [], [], []
    for d in depths:
        r = cross_validate(_pipe(DecisionTreeClassifier(max_depth=d, random_state=config.SEED)), Xc, yc, cv=_cv(),
                           return_train_score=True, n_jobs=n_jobs)
        tr.append(100 * r["train_score"].mean()); te.append(100 * r["test_score"].mean())
        sd.append(100 * r["test_score"].std())
    best = depths[int(np.argmax(te))]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(depths, tr, color=C["blue"], marker="o", ms=4, label="training accuracy")
    ax.plot(depths, te, color=C["orange"], marker="s", ms=4, label="5-fold CV accuracy")
    ax.fill_between(depths, np.array(te) - sd, np.array(te) + sd, color=C["orange"], alpha=0.12, lw=0)
    ax.axvline(best, color=C["muted"], ls=":", lw=1)
    ax.annotate(f"best depth = {best}\n(CV {max(te):.1f}%)", (best, max(te)), xytext=(10, -40),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("maximum tree depth"); ax.set_ylabel("accuracy (%)"); ax.set_xticks(depths[::2])
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Deeper trees memorise the training leaves (over-fitting): the gap keeps growing")
    return save_fig(fig, "tree", "1_depth_vs_overfitting", reports_dir)


def tree_diagram(data, reports_dir):
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    plt = _plt()
    Xc, yc = _cv_sample(data)
    tree = DecisionTreeClassifier(max_depth=3, random_state=config.SEED).fit(Xc, yc)
    fig, ax = plt.subplots(figsize=(22, 8))
    plot_tree(tree, feature_names=F.FEATURE_NAMES, class_names=[_short(c) for c in data["class_names"]],
              filled=False, rounded=True, impurity=True, proportion=True, precision=2, fontsize=8, ax=ax)
    ax.set_title(f"A decision tree of depth 3 (raw feature values, {len(yc):,} leaves): every split is a "
                 f"readable if/else rule", fontsize=11)
    return save_fig(fig, "tree", "2_tree_depth3", reports_dir)


# ============================================================================ 06 bagging / random forest
def rf_trees_curve(data, reports_dir, n_jobs, params):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    plt = _plt()
    Xc, yc = _cv_sample(data)
    p = {k: v for k, v in params.get("rf", {}).items()}
    sizes = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    m, s = [], []
    for n in sizes:
        r = cross_val_score(_pipe(RandomForestClassifier(n_estimators=n, random_state=config.SEED, **p)), Xc, yc,
                            cv=_cv(), n_jobs=n_jobs)
        m.append(100 * r.mean()); s.append(100 * r.std())
    tparams = params.get("tree", {})
    tree = 100 * cross_val_score(_pipe(DecisionTreeClassifier(random_state=config.SEED, **tparams)), Xc, yc,
                                 cv=_cv(), n_jobs=n_jobs).mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(sizes, m, yerr=s, color=C["blue"], marker="o", ms=5, capsize=3, label="Random Forest (5-fold CV)")
    ax.axhline(tree, color=C["orange"], ls="--", lw=1.6, label=f"single tuned decision tree ({tree:.1f}%)")
    ax.set_xscale("log"); ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlabel("number of trees (each on a bootstrap sample + random feature subsets)")
    ax.set_ylabel("CV accuracy (%)"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Bagging: averaging many trees removes variance, then the curve flattens")
    return save_fig(fig, "rf", "1_accuracy_vs_number_of_trees", reports_dir)


def rf_importance(data, reports_dir, n_jobs, params):
    from sklearn.ensemble import RandomForestClassifier
    plt = _plt()
    rf = RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=n_jobs,
                                **params.get("rf", {})).fit(data["X_train"], data["y_train"])
    imp = rf.feature_importances_
    top = np.argsort(imp)[-15:]
    group_of = {j: g for g, cols in F.FEATURE_GROUPS.items() for j in cols}
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(range(len(top)), 100 * imp[top], color=C["blue"])
    ax.set_yticks(range(len(top)), [f"{F.FEATURE_NAMES[j]}  ({group_of[j]})" for j in top], fontsize=8.5)
    for i, j in enumerate(top):
        ax.text(100 * imp[j] + 0.05, i, f"{100 * imp[j]:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("share of the total impurity decrease (%)"); ax.grid(axis="y", visible=False)
    by_group = {g: 100 * imp[cols].sum() for g, cols in F.FEATURE_GROUPS.items()}
    ax.set_title("Random Forest: 15 most important features\nper group: "
                 + ", ".join(f"{g} {v:.0f}%" for g, v in sorted(by_group.items(), key=lambda kv: -kv[1])),
                 fontsize=10)
    return save_fig(fig, "rf", "2_feature_importance", reports_dir)


# ============================================================================ 07 boosting
def boosting_rounds(data, reports_dir):
    from sklearn.ensemble import HistGradientBoostingClassifier
    plt = _plt()
    Xc, yc = _cv_sample(data)
    Xv, yv = data["X_valid"], data["y_valid"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    styles = [(0.03, C["aqua"], ":"), (0.1, C["blue"], "-"), (0.3, C["orange"], "--")]
    for lr, col, ls in styles:
        gb = HistGradientBoostingClassifier(learning_rate=lr, max_iter=300, max_leaf_nodes=31, early_stopping=False,
                                            random_state=config.SEED).fit(Xc, yc)
        tr = [100 * np.mean(p == yc) for p in gb.staged_predict(Xc)]
        va = [100 * np.mean(p == yv) for p in gb.staged_predict(Xv)]
        it = np.arange(1, len(va) + 1)
        ax.plot(it, va, color=col, ls=ls, lw=2, label=f"learning rate {lr}: held-out ({va[-1]:.1f}% at {it[-1]})")
        ax.plot(it, tr, color=col, ls=ls, lw=0.9, alpha=0.5)
    ax.set_xlabel("boosting round (number of trees added one after another)"); ax.set_ylabel("accuracy (%)")
    ax.set_ylim(50, 101); ax.legend(fontsize=8.5, loc="lower right")
    ax.set_title("Boosting: each round fixes the previous errors (thin lines = training accuracy)")
    return save_fig(fig, "boosting", "1_accuracy_per_round", reports_dir)


def ensembles_compared(reports_dir):
    plt = _plt()
    path = config.tables_dir(reports_dir) / "classification_results.csv"
    if not path.exists():
        return None
    res = pd.read_csv(path).set_index("Model")
    rows = [m for m in ["Decision Tree", "Random Forest", "Gradient Boosting"] if m in res.index]
    labels = {"Decision Tree": "one decision tree", "Random Forest": "bagging\n(Random Forest)",
              "Gradient Boosting": "boosting\n(Gradient Boosting)"}
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(rows))
    b1 = ax.bar(x - 0.2, res.loc[rows, "CV accuracy (%)"], 0.38, yerr=res.loc[rows, "CV std"], color=C["blue"],
                error_kw={"ecolor": C["muted"], "capsize": 3}, label="5-fold CV accuracy")
    b2 = ax.bar(x + 0.2, res.loc[rows, "Valid balanced acc (%)"], 0.38, color=C["orange"],
                label="held-out balanced accuracy")
    ax.bar_label(b1, fmt="%.1f", padding=8, fontsize=8.5); ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8.5)
    ax.set_xticks(x, [labels[m] for m in rows]); ax.set_ylim(50, 103); ax.set_ylabel("accuracy (%)")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Bagging reduces variance, boosting reduces bias - both beat one tree")
    return save_fig(fig, "boosting", "2_bagging_vs_boosting", reports_dir)


# ============================================================================ 08 knn
def knn_k_curve(data, reports_dir, n_jobs):
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    plt = _plt()
    Xc, yc = _cv_sample(data)
    ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 45]
    scaled, raw = [], []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
        scaled.append(100 * cross_val_score(_pipe(knn), Xc, yc, cv=_cv(), n_jobs=n_jobs).mean())
        raw.append(100 * cross_val_score(knn, Xc, yc, cv=_cv(), n_jobs=n_jobs).mean())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ks, scaled, color=C["blue"], marker="o", ms=5, label="with StandardScaler")
    ax.plot(ks, raw, color=C["orange"], marker="s", ms=5, ls="--", label="raw features (no scaling)")
    b = ks[int(np.argmax(scaled))]
    ax.annotate(f"best k = {b}", (b, max(scaled)), xytext=(8, -16), textcoords="offset points", fontsize=9)
    ax.set_xlabel("k (number of neighbours, distance-weighted vote)"); ax.set_ylabel("5-fold CV accuracy (%)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("KNN: small k = noisy, large k = blurred; without scaling, big-valued features dominate")
    return save_fig(fig, "knn", "1_accuracy_vs_k", reports_dir)


# ============================================================================ 09 cv & model selection
def model_comparison(reports_dir):
    plt = _plt()
    path = config.tables_dir(reports_dir) / "classification_results.csv"
    if not path.exists():
        return None
    res = pd.read_csv(path)
    order = res.sort_values("CV accuracy (%)").reset_index(drop=True)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(order) + 1.4))
    b1 = ax.barh(y + 0.2, order["CV accuracy (%)"], 0.38, xerr=order["CV std"], color=C["blue"],
                 error_kw={"ecolor": C["muted"], "capsize": 2, "lw": 1}, label="5-fold CV accuracy (± std)")
    b2 = ax.barh(y - 0.2, order["Valid balanced acc (%)"], 0.38, color=C["orange"], label="held-out balanced accuracy")
    ax.bar_label(b1, fmt="%.1f", padding=10, fontsize=8, color=C["ink"])
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8, color=C["ink"])
    ax.set_yticks(y, [f"{m}  ·  {u}" for m, u in zip(order["Model"], order["Syllabus"])])
    ax.set_xlim(max(0, order[["CV accuracy (%)", "Valid balanced acc (%)"]].min().min() - 10), 103)
    ax.set_xlabel("accuracy (%)"); ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title(f"Disease classification: {len(order)} classical models, same 55 features")
    return save_fig(fig, "cv", "1_model_comparison", reports_dir)


def cv_fold_scores(reports_dir):
    plt = _plt()
    path = config.tables_dir(reports_dir) / "classification_scores.json"
    if not path.exists():
        return None
    models = json.loads(path.read_text(encoding="utf-8"))["models"]
    items = sorted(models.values(), key=lambda m: np.mean(m["cv_fold_scores"]))
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(items) + 1.4))
    for i, m in enumerate(items):
        s = 100 * np.array(m["cv_fold_scores"])
        ax.plot([s.min(), s.max()], [i, i], color=C["grid"], lw=6, solid_capstyle="round", zorder=1)
        ax.scatter(s, np.full(len(s), i), s=36, color=C["blue"], edgecolors="white", lw=1, zorder=2)
        ax.scatter([s.mean()], [i], s=90, marker="|", color=C["ink"], lw=2, zorder=3)
        ax.text(s.max() + 0.4, i, f"{s.mean():.1f} ± {s.std():.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(items)), [m["name"] for m in items]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy on each of the 5 CV folds (%)   |  = mean")
    ax.set_title("Cross-validation: every model is scored on 5 different held-out folds")
    return save_fig(fig, "cv", "2_cv_fold_scores", reports_dir)


def svm_grid_heatmap(data, reports_dir, n_jobs):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    plt = _plt()
    Xc, yc = _cv_sample(data)
    Cs, gammas = [0.1, 1, 10, 100, 1000], [0.001, 0.003, 0.01, 0.03, 0.1]
    gs = GridSearchCV(_pipe(SVC(kernel="rbf")), {"clf__C": Cs, "clf__gamma": gammas}, cv=_cv(), n_jobs=n_jobs,
                      refit=False).fit(Xc, yc)
    grid = 100 * gs.cv_results_["mean_test_score"].reshape(len(Cs), len(gammas))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(grid, cmap=_heat_cmap(), aspect="auto")
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if grid[i, j] > np.percentile(grid, 60) else C["ink"])
    bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec=C["orange"], lw=2.5))
    ax.set_xticks(range(len(gammas)), [str(g) for g in gammas]); ax.set_yticks(range(len(Cs)), [str(c) for c in Cs])
    ax.set_xlabel("gamma (RBF kernel width: larger = wigglier boundary)")
    ax.set_ylabel("C (larger = less regularisation)"); ax.grid(False)
    fig.colorbar(im, ax=ax, label="5-fold CV accuracy (%)")
    ax.set_title(f"Grid search for the RBF SVM: best C = {Cs[bi]}, gamma = {gammas[bj]} ({grid.max():.1f}%)")
    return save_fig(fig, "cv", "3_svm_grid_search", reports_dir)


def learning_curve_best(data, reports_dir, n_jobs):
    import joblib
    from sklearn.base import clone
    from sklearn.model_selection import learning_curve
    plt = _plt()
    if not config.BEST_MODEL_PATH.exists():
        return None
    bundle = joblib.load(config.BEST_MODEL_PATH)
    pipe = bundle["pipeline"]
    if pipe.named_steps["clf"].__class__.__name__ == "CalibratedClassifierCV":
        pipe = _pipe(pipe.named_steps["clf"].estimator)
    Xc, yc = _cv_sample(data)
    sizes, tr, te = learning_curve(clone(pipe), Xc, yc, cv=_cv(), n_jobs=n_jobs,
                                   train_sizes=np.linspace(0.1, 1.0, 6), random_state=config.SEED, shuffle=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for scores, col, lab, mk in [(tr, C["blue"], "training accuracy", "o"), (te, C["orange"], "5-fold CV accuracy", "s")]:
        m, s = 100 * scores.mean(1), 100 * scores.std(1)
        ax.plot(sizes, m, color=col, marker=mk, ms=5, label=lab)
        ax.fill_between(sizes, m - s, m + s, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("number of training leaves"); ax.set_ylabel("accuracy (%)"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"Learning curve of the best model ({bundle['model_name']}): more leaves -> higher CV accuracy")
    return save_fig(fig, "cv", "4_learning_curve_best_model", reports_dir)


# ============================================================================ 10 evaluation metrics
def _best_predictions(data):
    import joblib
    if not config.BEST_MODEL_PATH.exists():
        return None
    bundle = joblib.load(config.BEST_MODEL_PATH)
    proba = bundle["pipeline"].predict_proba(data["X_valid"])
    return bundle["model_name"], proba, proba.argmax(1)


def confusion_matrix_plot(data, reports_dir):
    from sklearn.metrics import confusion_matrix
    plt = _plt()
    out = _best_predictions(data)
    if out is None:
        return None
    model, _, pred = out
    names = [_short(c) for c in data["class_names"]]
    K = len(names)
    cm = confusion_matrix(data["y_valid"], pred, labels=np.arange(K))
    frac = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = max(6, 0.32 * K + 3)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    im = ax.imshow(frac, cmap=_heat_cmap(), vmin=0, vmax=1)
    if K <= 15:
        for i in range(K):
            for j in range(K):
                if cm[i, j]:
                    ax.text(j, i, f"{100 * frac[i, j]:.0f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8,
                            color="white" if frac[i, j] > 0.55 else C["ink"])
    fs = 8.5 if K <= 15 else 6
    ax.set_xticks(range(K), names, rotation=35 if K <= 15 else 90, ha="right" if K <= 15 else "center", fontsize=fs)
    ax.set_yticks(range(K), names, fontsize=fs); ax.grid(False)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of the true class (row)")
    ax.set_title(f"Confusion matrix - {model}, {len(pred):,} held-out leaves (diagonal = correct)")
    return save_fig(fig, "metrics", "1_confusion_matrix", reports_dir)


def per_class_prf(data, reports_dir):
    from sklearn.metrics import precision_recall_fscore_support
    plt = _plt()
    out = _best_predictions(data)
    if out is None:
        return None
    model, _, pred = out
    names = [_short(c) for c in data["class_names"]]
    p, r, f, sup = precision_recall_fscore_support(data["y_valid"], pred, labels=np.arange(len(names)),
                                                   zero_division=0)
    order = np.argsort(f)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(names) + 1.6))
    for off, vals, col, lab in [(0.27, p, C["blue"], "precision"), (0, r, C["orange"], "recall"),
                                (-0.27, f, C["aqua"], "F1")]:
        bars = ax.barh(y + off, 100 * vals[order], 0.25, color=col, label=lab)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7.5)
    ax.set_yticks(y, [f"{names[k]} (n={sup[k]})" for k in order], fontsize=8.5)
    ax.set_xlim(max(0, 100 * min(p.min(), r.min()) - 15), 104); ax.set_xlabel("%"); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8.5, loc="lower right", ncol=3)
    ax.set_title(f"Precision, recall and F1 per class - {model} (held-out set)")
    return save_fig(fig, "metrics", "2_precision_recall_f1", reports_dir)


def roc_curves(data, reports_dir, max_panels: int = 9):
    from sklearn.metrics import auc, roc_curve
    plt = _plt()
    out = _best_predictions(data)
    if out is None:
        return None
    model, proba, _ = out
    names = data["class_names"]
    yv = data["y_valid"]
    aucs = []
    for k in range(len(names)):
        fpr, tpr, _ = roc_curve(yv == k, proba[:, k])
        aucs.append((auc(fpr, tpr), k, fpr, tpr))
    aucs.sort(key=lambda t: t[0])              # hardest classes (lowest AUC) first
    show = aucs[:max_panels]
    cols = 3
    rows = int(np.ceil(len(show) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.4 * rows), squeeze=False)
    for ax, (a, k, fpr, tpr) in zip(axes.flat, show):
        ax.plot([0, 1], [0, 1], color=C["muted"], ls=":", lw=1)
        ax.plot(fpr, tpr, color=_class_color(names[k]), lw=2)
        ax.fill_between(fpr, tpr, color=_class_color(names[k]), alpha=0.1, lw=0)
        ax.set_title(f"{_short(names[k])}: AUC {a:.3f}", fontsize=10)
        ax.set_xlabel("false positive rate", fontsize=8.5); ax.set_ylabel("true positive rate", fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    extra = f" ({len(show)} hardest of {len(names)} classes)" if len(names) > max_panels else ""
    fig.suptitle(f"ROC curves, one class vs the rest - {model}{extra}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return save_fig(fig, "metrics", "3_roc_curves", reports_dir)


# ============================================================================ 11 ablation
def cumulative_ablation(reports_dir):
    plt = _plt()
    path = config.tables_dir(reports_dir) / "ablation.csv"
    if not path.exists():
        return None
    abl = pd.read_csv(path)
    cols = [c for c in abl.columns if c.endswith("CV acc (%)")]
    x = np.arange(len(abl))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.8 / max(len(cols), 1)
    for i, (col, color) in enumerate(zip(cols, [C["blue"], C["orange"]])):
        bars = ax.bar(x + (i - (len(cols) - 1) / 2) * width, abl[col], width * 0.95, color=color,
                      label=col.replace(" CV acc (%)", ""))
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2, color=C["ink"])
    ax.set_xticks(x, [f"{s}\n({n} features)" for s, n in zip(abl["Feature set"], abl["# features"])], fontsize=8.5)
    ax.set_ylim(max(0, abl[cols].min().min() - 12), 100); ax.set_ylabel("5-fold CV accuracy (%)")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Cumulative ablation: what each feature group adds")
    return save_fig(fig, "ablation", "1_cumulative_ablation", reports_dir)


def leave_one_group_out(data, reports_dir, n_jobs, params):
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    from src.evaluate import save_table
    plt = _plt()
    Xc, yc = _cv_sample(data)
    svm = lambda: _pipe(SVC(kernel="rbf", **params.get("svm_rbf", {"C": 10, "gamma": "scale"})))
    full = 100 * cross_val_score(svm(), Xc, yc, cv=_cv(), n_jobs=n_jobs).mean()
    rows = []
    for g, cols in F.FEATURE_GROUPS.items():
        keep = [j for j in range(F.N_FEATURES) if j not in cols]
        acc = 100 * cross_val_score(svm(), Xc[:, keep], yc, cv=_cv(), n_jobs=n_jobs).mean()
        rows.append({"Removed group": g, "# features removed": len(cols), "CV acc (%)": acc, "Drop (points)": full - acc})
    df = pd.DataFrame(rows).sort_values("Drop (points)")
    save_table(df.round(2), "ablation_leave_one_group_out", reports_dir,
               f"Leave-one-group-out ablation, RBF SVM (all 55 features: {full:.2f}%)")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(df["Removed group"], df["Drop (points)"],
                   color=[C["orange"] if d > 0 else C["blue"] for d in df["Drop (points)"]])
    ax.bar_label(bars, labels=[f"{d:+.1f} pts ({a:.1f}%)" for d, a in zip(df["Drop (points)"], df["CV acc (%)"])],
                 padding=3, fontsize=8.5)
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_xlabel(f"accuracy lost when the group is removed (points; all 55 features = {full:.1f}%)")
    ax.grid(axis="y", visible=False); ax.margins(x=0.25)
    ax.set_title("Leave-one-group-out ablation (RBF SVM): which group can the model not do without?")
    return save_fig(fig, "ablation", "2_leave_one_group_out", reports_dir)


# ============================================================================ all
def make_all(data: dict | None, reports_dir: Path = config.REPORTS_DIR, n_jobs: int = config.N_JOBS) -> list[Path]:
    """Make every supervised-learning graph. Returns the saved paths."""
    from sklearn.exceptions import ConvergenceWarning
    reports_dir = Path(reports_dir)
    config.set_seeds()
    params = _tuned_params(reports_dir)
    jobs = [("segmentation examples", lambda: segmentation_examples(reports_dir)),
            ("model comparison", lambda: model_comparison(reports_dir)),
            ("CV fold scores", lambda: cv_fold_scores(reports_dir)),
            ("bagging vs boosting", lambda: ensembles_compared(reports_dir)),
            ("cumulative ablation", lambda: cumulative_ablation(reports_dir))]
    if data is not None:
        jobs += [("key features by class", lambda: feature_boxplots(data, reports_dir)),
                 ("55 x 55 correlation", lambda: feature_correlation(data, reports_dir)),
                 ("tree depth curve", lambda: tree_depth_curve(data, reports_dir, n_jobs)),
                 ("tree diagram", lambda: tree_diagram(data, reports_dir)),
                 ("random forest vs trees", lambda: rf_trees_curve(data, reports_dir, n_jobs, params)),
                 ("random forest importance", lambda: rf_importance(data, reports_dir, n_jobs, params)),
                 ("boosting rounds", lambda: boosting_rounds(data, reports_dir)),
                 ("KNN k curve", lambda: knn_k_curve(data, reports_dir, n_jobs)),
                 ("SVM grid search", lambda: svm_grid_heatmap(data, reports_dir, n_jobs)),
                 ("learning curve", lambda: learning_curve_best(data, reports_dir, n_jobs)),
                 ("confusion matrix", lambda: confusion_matrix_plot(data, reports_dir)),
                 ("precision / recall / F1", lambda: per_class_prf(data, reports_dir)),
                 ("ROC curves", lambda: roc_curves(data, reports_dir)),
                 ("leave-one-group-out ablation", lambda: leave_one_group_out(data, reports_dir, n_jobs, params))]
    saved = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for label, job in jobs:
            print(f"  {label:32s}", end="", flush=True)
            path = job()
            print(f" -> {path.relative_to(reports_dir) if path else 'skipped (run src.classify first)'}")
            if path:
                saved.append(path)
    return saved
