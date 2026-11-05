"""
src/unsupervised.py - unsupervised learning on the 55 leaf features (no labels used for fitting).

  PCA         how many directions really carry information? (features are strongly correlated)
              2-D view of the leaves, and accuracy of logistic regression on the top-k components
  k-means     hard clustering into K groups
  GMM + EM    soft clustering with a Gaussian mixture fitted by Expectation-Maximisation;
              BIC chooses the number of components; EM's log-likelihood rises every iteration
  Evaluation  the true disease labels are used ONLY afterwards, to score the clusters
              (adjusted Rand index, normalised mutual information, purity)

Usage (from the plant-disease-ml folder):
    python -m src.unsupervised              # uses cache/features.npz
    python -m src.unsupervised --limit 100
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/unsupervised.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config

PCA_VARIANCE = 0.95             # clustering runs on the PCs that keep this share of the variance
BIC_K_RANGE = range(1, 13)
ELBOW_K_RANGE = range(2, 13)
PCA_K_LIST = (2, 5, 10, 20, 30)


def purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """Share of leaves whose cluster's majority class is their own class (1 = every cluster is pure)."""
    total = 0
    for c in np.unique(clusters):
        total += np.bincount(y_true[clusters == c]).max()
    return total / len(y_true)


def cluster_scores(y_true, clusters) -> dict[str, float]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return {"ARI": adjusted_rand_score(y_true, clusters),
            "NMI": normalized_mutual_info_score(y_true, clusters),
            "Purity": purity(y_true, clusters)}


def em_trace(Z: np.ndarray, k: int, max_iter: int = 60) -> list[float]:
    """Run EM one iteration at a time and record the average log-likelihood (lower bound) after each."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture
    gm = GaussianMixture(n_components=k, covariance_type="full", max_iter=1, warm_start=True,
                         random_state=config.SEED, init_params="kmeans")
    trace = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(max_iter):
            gm.fit(Z)
            trace.append(float(gm.lower_bound_))
            if len(trace) > 1 and abs(trace[-1] - trace[-2]) < 1e-4:
                break
    return trace


def run(data: dict, reports_dir: Path = config.REPORTS_DIR, n_jobs: int = config.N_JOBS) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.classify import balanced_sample
    from src.evaluate import save_table, text_table

    config.set_seeds()
    reports_dir = Path(reports_dir)
    X, y = data["X_train"], data["y_train"]
    names = [config.pretty_class(c).replace("Tomato: ", "").replace("Tomato ", "") for c in data["class_names"]]
    K = len(names)
    Xs = StandardScaler().fit_transform(X)

    # ---- PCA
    pca = PCA(random_state=config.SEED).fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    need = {f"{int(t * 100)}%": int(np.searchsorted(cum, t) + 1) for t in (0.80, 0.90, 0.95, 0.99)}
    print(f"PCA on {len(Xs):,} leaves x {Xs.shape[1]} standardised features. Components needed for "
          + ", ".join(f"{k} variance: {v}" for k, v in need.items()))
    print(f"PC1 explains {100 * pca.explained_variance_ratio_[0]:.1f}%, PC2 {100 * pca.explained_variance_ratio_[1]:.1f}%")

    idx = balanced_sample(y, config.CV_PER_CLASS)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    acc_rows = []
    for k in [k for k in PCA_K_LIST if k < Xs.shape[1]] + [Xs.shape[1]]:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=config.SEED),
                             LogisticRegression(max_iter=5000))
        s = cross_val_score(pipe, X[idx], y[idx], cv=cv, n_jobs=n_jobs)
        acc_rows.append({"PCA components": k, "Variance kept (%)": 100 * cum[k - 1],
                         "LR CV accuracy (%)": 100 * s.mean(), "CV std": 100 * s.std()})
    pca_table = pd.DataFrame(acc_rows)
    print("\nLogistic regression on the top-k principal components:\n" + text_table(pca_table))

    # ---- clustering on the PCs that keep 95% of the variance
    n95 = need["95%"]
    Z = PCA(n_components=n95, random_state=config.SEED).fit_transform(Xs)
    km = KMeans(n_clusters=K, n_init=10, random_state=config.SEED).fit(Z)
    gm = GaussianMixture(n_components=K, covariance_type="full", n_init=3, random_state=config.SEED).fit(Z)
    bic = [GaussianMixture(n_components=k, covariance_type="full", random_state=config.SEED).fit(Z).bic(Z)
           for k in BIC_K_RANGE]
    best_k = list(BIC_K_RANGE)[int(np.argmin(bic))]
    trace = em_trace(Z, K)
    clusters = {"k-means": km.labels_, "GMM (EM)": gm.predict(Z)}
    responsibilities = gm.predict_proba(Z)

    # k-means elbow (inertia) and silhouette for several K
    from sklearn.metrics import silhouette_score
    sil_idx = np.random.default_rng(config.SEED).permutation(len(Z))[:3000]   # silhouette is O(n^2)
    elbow = []
    for k in ELBOW_K_RANGE:
        m = KMeans(n_clusters=k, n_init=5, random_state=config.SEED).fit(Z)
        elbow.append({"K": k, "Inertia": m.inertia_, "Silhouette": silhouette_score(Z[sil_idx], m.labels_[sil_idx])})
    elbow = pd.DataFrame(elbow)
    rows = [{"Method": m, "Clusters": K, **cluster_scores(y, c)} for m, c in clusters.items()]
    rows.append({"Method": "Random assignment (reference)", "Clusters": K,
                 **cluster_scores(y, np.random.default_rng(config.SEED).integers(0, K, len(y)))})
    clus_table = pd.DataFrame(rows)
    print(f"\nClustering {len(Z):,} leaves on {n95} PCs (95% variance), K = {K} = number of diseases. "
          f"Labels used only for scoring:\n" + text_table(clus_table))
    print(f"GMM BIC is lowest at {best_k} components (tried {BIC_K_RANGE.start}-{BIC_K_RANGE.stop - 1}). "
          f"EM ran {len(trace)} iterations; average log-likelihood {trace[0]:.2f} -> {trace[-1]:.2f} "
          f"(never decreased: {bool(np.all(np.diff(trace) >= -1e-9))})")

    save_table(pca_table.round(2), "unsup_pca", reports_dir,
               "PCA: variance kept and logistic regression accuracy on the top-k components")
    save_table(clus_table.round(3), "unsup_clustering", reports_dir,
               f"Clustering without labels (K = {K}, on {n95} PCs); labels used only to score")
    save_table(pd.DataFrame({"GMM components": list(BIC_K_RANGE), "BIC": np.round(bic, 1)}), "unsup_bic",
               reports_dir, "Gaussian mixture: BIC for each number of components (lower = better)")

    save_table(elbow.round(4), "unsup_kmeans_elbow", reports_dir, "k-means: inertia and silhouette for each K")

    make_graphs(pca, Xs, y, data["class_names"], names, cum, need, pca_table, bic, best_k, trace, clusters,
                responsibilities, elbow, K, reports_dir)
    return {"need": need, "pca": pca_table, "clustering": clus_table, "bic_best_k": best_k, "em_trace": trace}


# ----------------------------------------------------------------------------- graphs
def _contingency(ax, y, c, names, title, cmap, C):
    tab = pd.crosstab(pd.Series(y, name="class"), pd.Series(c, name="cluster"))
    frac = tab.div(tab.sum(axis=1), axis=0).to_numpy()
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    if frac.shape[0] * frac.shape[1] <= 300:
        for i in range(frac.shape[0]):
            for j in range(frac.shape[1]):
                ax.text(j, i, f"{100 * frac[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if frac[i, j] > 0.55 else C["ink"])
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.set_xticks(range(frac.shape[1]), [f"C{j + 1}" for j in tab.columns], fontsize=8.5)
    ax.set_xlabel("cluster found without labels"); ax.grid(False)
    sc = cluster_scores(y, c)
    ax.set_title(f"{title}: ARI {sc['ARI']:.2f}, NMI {sc['NMI']:.2f}, purity {100 * sc['Purity']:.0f}%", fontsize=10)
    return im


def make_graphs(pca, Xs, y, class_names, names, cum, need, pca_table, bic, best_k, trace, clusters,
                responsibilities, elbow, K, reports_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from src import features as F
    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9dcf3", C["blue"], "#173f73"])
    healthy = np.array([config.is_healthy(c) for c in class_names])
    P = pca.transform(Xs)[:, :2]
    v1, v2 = 100 * pca.explained_variance_ratio_[:2]

    # ===== PCA 1: one small panel per class (class highlighted over all leaves in grey) + scree
    show_classes = list(range(len(names)))[:8]
    n_panels = len(show_classes) + 1
    cols = 3
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.4 * rows), squeeze=False)
    show = np.random.default_rng(config.SEED).permutation(len(P))[:4000]
    for i, k in enumerate(show_classes):
        ax = axes.flat[i]
        ax.scatter(P[show, 0], P[show, 1], s=4, color="#d9d6cc", lw=0)
        m = show[y[show] == k]
        ax.scatter(P[m, 0], P[m, 1], s=6, color=C["healthy"] if healthy[k] else C["diseased"], lw=0, alpha=0.8)
        ax.set_title(names[k], fontsize=10)
        ax.set_xlabel(f"PC1 ({v1:.0f}% of variance)", fontsize=8.5); ax.set_ylabel(f"PC2 ({v2:.0f}%)", fontsize=8.5)
    ax = axes.flat[len(show_classes)]
    k = np.arange(1, len(cum) + 1)
    ax.bar(k, 100 * pca.explained_variance_ratio_, color=C["blue"], width=0.8, label="each component")
    ax.plot(k, 100 * cum, color=C["ink"], lw=1.6, label="cumulative")
    ax.axhline(95, color=C["muted"], ls=":", lw=1)
    ax.annotate(f"{need['95%']} PCs keep 95%", (need["95%"], 95), xytext=(8, -18), textcoords="offset points",
                fontsize=8.5)
    ax.set_xlabel("principal component"); ax.set_ylabel("variance explained (%)"); ax.set_ylim(0, 102)
    ax.set_title("Scree plot: 55 features, far fewer real directions", fontsize=10); ax.legend(fontsize=8)
    for a in axes.flat[n_panels:]:
        a.axis("off")
    fig.suptitle(f"PCA of {len(Xs):,} leaves (each class highlighted over all leaves in grey)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "pca", "1_pca_scatter_scree", reports_dir)

    # ===== PCA 2: accuracy vs number of components
    fig, ax = plt.subplots(figsize=(8, 4))
    t = pca_table
    ax.plot(t["PCA components"], t["LR CV accuracy (%)"], color=C["blue"], marker="o", ms=6,
            label="logistic regression, 5-fold CV accuracy")
    ax.fill_between(t["PCA components"], t["LR CV accuracy (%)"] - t["CV std"],
                    t["LR CV accuracy (%)"] + t["CV std"], color=C["blue"], alpha=0.12, lw=0)
    ax.plot(t["PCA components"], t["Variance kept (%)"], color=C["muted"], ls="--", lw=1.4, label="variance kept")
    for _, r in t.iterrows():
        ax.annotate(f"{r['LR CV accuracy (%)']:.1f}%", (r["PCA components"], r["LR CV accuracy (%)"]),
                    xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlabel("number of principal components kept"); ax.set_ylabel("%"); ax.set_ylim(0, 103)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Dimensionality reduction: fewer components -> less information -> lower accuracy")
    save_fig(fig, "pca", "2_accuracy_vs_components", reports_dir)

    # ===== PCA 3: loadings of PC1 and PC2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, pc in zip(axes, (0, 1)):
        load = pca.components_[pc]
        top = np.argsort(np.abs(load))[-10:]
        ax.barh(range(10), load[top], color=[C["orange"] if v > 0 else C["blue"] for v in load[top]])
        ax.set_yticks(range(10), [F.FEATURE_NAMES[j] for j in top], fontsize=8.5)
        ax.axvline(0, color=C["ink"], lw=0.8); ax.grid(axis="y", visible=False)
        ax.set_xlabel("loading (weight of the feature in this component)")
        ax.set_title(f"PC{pc + 1} ({100 * pca.explained_variance_ratio_[pc]:.1f}% of variance): top 10 features",
                     fontsize=10)
    fig.suptitle("What do the principal components measure?", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "pca", "3_loadings_pc1_pc2", reports_dir)

    # ===== k-means 1: elbow + silhouette
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(elbow["K"], elbow["Inertia"], color=C["blue"], marker="o", ms=6)
    a1.set_xlabel("number of clusters K"); a1.set_ylabel("inertia (within-cluster sum of squares)")
    a1.set_title("Elbow method: inertia always falls - look for the bend"); a1.set_xticks(elbow["K"])
    a2.plot(elbow["K"], elbow["Silhouette"], color=C["orange"], marker="s", ms=6)
    bk = int(elbow.loc[elbow["Silhouette"].idxmax(), "K"])
    a2.annotate(f"best silhouette: K = {bk}", (bk, elbow["Silhouette"].max()), xytext=(10, -14),
                textcoords="offset points", fontsize=9)
    a2.axvline(K, color=C["muted"], ls=":", lw=1)
    a2.text(K, elbow["Silhouette"].min(), f" {K} = number of classes", fontsize=8, color=C["muted"])
    a2.set_xlabel("number of clusters K"); a2.set_ylabel("silhouette score (higher = better separated)")
    a2.set_title("Silhouette: how well separated are the clusters?"); a2.set_xticks(elbow["K"])
    save_fig(fig, "kmeans", "1_elbow_silhouette", reports_dir)

    # ===== k-means 2 and GMM 3: clusters vs true classes
    for topic, name, method in [("kmeans", "2_clusters_vs_classes", "k-means"),
                                ("gmm", "3_clusters_vs_classes", "GMM (EM)")]:
        fig, ax = plt.subplots(figsize=(7.4, 0.5 * len(names) + 2.2))
        im = _contingency(ax, y, clusters[method], names, method, seq, C)
        fig.colorbar(im, ax=ax, shrink=0.8, label="% of the true class in each cluster")
        fig.suptitle("Do the clusters match the diseases? (row = true class)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        save_fig(fig, topic, name, reports_dir)

    # ===== GMM 1: BIC + EM log-likelihood per iteration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ks = list(BIC_K_RANGE)
    a1.plot(ks, bic, color=C["blue"], marker="o", ms=6)
    a1.scatter([best_k], [min(bic)], s=120, facecolors="none", edgecolors=C["orange"], linewidths=2, zorder=3)
    a1.annotate(f"lowest BIC: {best_k} components", (best_k, min(bic)), xytext=(-10, 18),
                textcoords="offset points", ha="right", fontsize=9)
    a1.axvline(K, color=C["muted"], ls=":", lw=1)
    a1.text(K, max(bic), f" {K} = number of classes", fontsize=8, color=C["muted"], va="top")
    a1.set_xlabel("number of Gaussian components"); a1.set_ylabel("BIC (lower = better)")
    a1.set_title("Choosing the number of components with BIC"); a1.set_xticks(ks)
    a2.plot(np.arange(1, len(trace) + 1), trace, color=C["orange"], marker="o", ms=4)
    a2.set_xlabel("EM iteration"); a2.set_ylabel("average log-likelihood")
    a2.set_title(f"EM never decreases the likelihood ({K} components)")
    save_fig(fig, "gmm", "1_bic_and_em", reports_dir)

    # ===== GMM 2: soft assignments (how sure is each leaf's cluster membership?)
    top = responsibilities.max(axis=1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.linspace(1 / K, 1, 30)
    is_h = healthy[y]
    ax.hist(top[~is_h], bins=bins, color=C["diseased"], alpha=0.75, label="diseased leaves")
    ax.hist(top[is_h], bins=bins, color=C["healthy"], alpha=0.75, label="healthy leaves")
    ax.set_yscale("log")
    ax.set_xlabel("largest responsibility  max_k p(cluster k | leaf)   (1 = certain)")
    ax.set_ylabel("number of leaves (log scale)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(f"GMM soft clustering: {100 * (top > 0.9).mean():.0f}% of leaves are > 90% sure of their cluster "
                 f"(k-means is always 100%)", fontsize=10)
    save_fig(fig, "gmm", "2_soft_assignments", reports_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PCA, k-means and Gaussian mixture (EM) on the leaf features.")
    ap.add_argument("--limit", type=int, default=None, help="use the --limit N feature cache")
    args = ap.parse_args(argv)
    from src import dataset
    try:
        data = dataset.load_features(args.limit)
    except FileNotFoundError as e:
        print(e)
        return 1
    run(data)
    print(f"\nTables and graphs written to {config.REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
