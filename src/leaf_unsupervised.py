"""
Unit 3 on a single leaf photo: PCA, k-means, GMM (EM), and an HMM
(forward-backward + Viterbi) run on the photo's pixels.

Usage (from the plant-disease-ml folder):
    python -m src.leaf_unsupervised samples/tomato_early_blight.jpg
    python -m src.leaf_unsupervised path/to/leaf.jpg --out results --size 256

Results (4 figures + results.txt) go to reports/single_leaf/<photo name>/ unless --out is given.

Needs: pip install numpy opencv-python scikit-learn matplotlib
"""

import argparse
import os
import sys
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURE_NAMES = ["R", "G", "B", "H", "S", "V", "ExG"]
STATE_NAMES = ["background", "healthy", "lesion"]
STATE_COLOURS = np.array([[40, 40, 40], [46, 125, 50], [230, 160, 0]], dtype=np.uint8)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def heading(text):
    say()
    say("=" * 64)
    say(text)
    say("=" * 64)


def load_image(path, size):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return rgb, hsv


def pixel_features(rgb, hsv):
    r, g, b = [rgb[..., i].astype(float) / 255 for i in range(3)]
    h = hsv[..., 0].astype(float) / 179
    s = hsv[..., 1].astype(float) / 255
    v = hsv[..., 2].astype(float) / 255
    exg = 2 * g - r - b
    return np.stack([r, g, b, h, s, v, exg], axis=-1).reshape(-1, len(FEATURE_NAMES))


def name_clusters(raw_features, labels, k):
    """Map arbitrary cluster ids to background / healthy / lesion.
    Background = lowest saturation, healthy = greenest of the rest, lesion = the last one."""
    means = np.array([raw_features[labels == c].mean(axis=0) for c in range(k)])
    sat = means[:, FEATURE_NAMES.index("S")]
    exg = means[:, FEATURE_NAMES.index("ExG")]
    background = int(np.argmin(sat))
    rest = [c for c in range(k) if c != background]
    healthy = max(rest, key=lambda c: exg[c])
    lesion = [c for c in rest if c != healthy][0]
    mapping = np.zeros(k, dtype=int)
    mapping[background], mapping[healthy], mapping[lesion] = 0, 1, 2
    return mapping


def share_table(state_map):
    total = state_map.size
    leaf = np.isin(state_map, [1, 2]).sum()
    for s, name in enumerate(STATE_NAMES):
        say(f"  {name:<11} {100 * (state_map == s).sum() / total:5.1f}% of image")
    lesion_pct = 100 * (state_map == 2).sum() / max(leaf, 1)
    say(f"  lesion area = {lesion_pct:.1f}% of the leaf")
    return lesion_pct


def run_pca(x_scaled, size, out):
    heading("1. PCA  (7 colour features per pixel)")
    pca = PCA(random_state=SEED).fit(x_scaled)
    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    say(f"  {'component':<10}{'variance':>10}{'cumulative':>12}")
    for i, (r, c) in enumerate(zip(ratio, cum), 1):
        say(f"  PC{i:<8}{100 * r:9.1f}%{100 * c:11.1f}%")
    say(f"  -> {n95} of 7 components keep 95% of the variance")
    top = np.argsort(-np.abs(pca.components_[0]))[:3]
    say("  PC1 is mostly: " + ", ".join(f"{FEATURE_NAMES[i]} ({pca.components_[0][i]:+.2f})" for i in top))

    scores = pca.transform(x_scaled)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].bar(range(1, 8), 100 * ratio, color="#2E7D32", label="each component")
    ax[0].plot(range(1, 8), 100 * cum, "o-", color="#E65100", label="cumulative")
    ax[0].axhline(95, ls=":", color="grey")
    ax[0].set(xlabel="principal component", ylabel="% variance explained", title="Scree plot")
    ax[0].legend()
    ax[1].imshow(scores[:, 0].reshape(size, size), cmap="viridis")
    ax[1].set_title("PC1 score of every pixel")
    ax[1].axis("off")
    ax[2].imshow(scores[:, 1].reshape(size, size), cmap="viridis")
    ax[2].set_title("PC2 score of every pixel")
    ax[2].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_pca.png"), dpi=150)
    plt.close(fig)
    return n95


def run_kmeans(x_scaled, raw, rgb, size, out):
    heading("2. k-means  (cluster the pixels by colour, no labels)")
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(x_scaled), size=min(4000, len(x_scaled)), replace=False)
    ks, inertias, sils = list(range(2, 9)), [], []
    say(f"  {'K':<4}{'inertia':>12}{'silhouette':>12}")
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(x_scaled[sample], km.labels_[sample]))
        say(f"  {k:<4}{km.inertia_:12.0f}{sils[-1]:12.3f}")
    say(f"  best silhouette: K = {ks[int(np.argmax(sils))]}")

    km3 = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(x_scaled)
    mapping = name_clusters(raw, km3.labels_, 3)
    state_map = mapping[km3.labels_].reshape(size, size)
    say("  With K = 3 (background / healthy / lesion):")
    lesion_pct = share_table(state_map)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(rgb)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(STATE_COLOURS[state_map])
    ax[1].set_title(f"k-means K=3: grey=background, green=healthy,\norange=lesion ({lesion_pct:.1f}% of leaf)")
    ax[1].axis("off")
    ax2 = ax[2].twinx()
    ax[2].plot(ks, inertias, "o-", color="#1565C0")
    ax2.plot(ks, sils, "s-", color="#E65100")
    ax[2].set(xlabel="number of clusters K", title="Elbow (blue) and silhouette (orange)")
    ax[2].set_ylabel("inertia (lower = tighter)", color="#1565C0")
    ax2.set_ylabel("silhouette (higher = better separated)", color="#E65100")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_kmeans.png"), dpi=150)
    plt.close(fig)
    return state_map, km3.cluster_centers_


def run_gmm(x_scaled, raw, rgb, size, out, km_centres):
    heading("3. GMM trained with EM  (soft clusters)")
    rng = np.random.default_rng(SEED)
    fit_idx = rng.choice(len(x_scaled), size=min(20000, len(x_scaled)), replace=False)
    xs = x_scaled[fit_idx]
    ks, bics = list(range(1, 9)), []
    say(f"  {'K':<4}{'BIC (lower = better)':>22}")
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=SEED).fit(xs)
        bics.append(g.bic(xs))
        say(f"  {k:<4}{bics[-1]:22.0f}")
    say(f"  lowest BIC: K = {ks[int(np.argmin(bics))]}")

    gmm = GaussianMixture(n_components=3, covariance_type="full", max_iter=1,
                          warm_start=True, means_init=km_centres, random_state=SEED)
    trace = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(200):
            gmm.fit(x_scaled)
            trace.append(gmm.lower_bound_)
            if len(trace) > 1 and abs(trace[-1] - trace[-2]) < 1e-4:
                break
    drops = sum(1 for a, b in zip(trace, trace[1:]) if b < a - 1e-9)
    say("  EM with K = 3, started from the k-means centres:")
    say(f"  average log-likelihood {trace[0]:.3f} -> {trace[-1]:.3f} "
        f"in {len(trace)} iterations (times it went down: {drops})")

    probs = gmm.predict_proba(x_scaled)
    labels = probs.argmax(axis=1)
    mapping = name_clusters(raw, labels, 3)
    state_map = mapping[labels].reshape(size, size)
    confidence = probs.max(axis=1).reshape(size, size)
    say("  With K = 3:")
    share_table(state_map)
    say(f"  unsure pixels (top probability < 0.9): {100 * (confidence < 0.9).mean():.1f}%")

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    ax[0].plot(ks, bics, "o-", color="#1565C0")
    ax[0].set(xlabel="number of Gaussian components", ylabel="BIC (lower = better)", title="Choosing K with BIC")
    ax[1].plot(range(1, len(trace) + 1), trace, "o-", color="#E65100", ms=3)
    ax[1].set(xlabel="EM iteration", ylabel="average log-likelihood", title="EM never decreases the likelihood")
    ax[2].imshow(STATE_COLOURS[state_map])
    ax[2].set_title("GMM K=3 hard labels")
    ax[2].axis("off")
    im = ax[3].imshow(confidence, cmap="magma", vmin=0.33, vmax=1)
    ax[3].set_title("How sure GMM is (dark = unsure)")
    ax[3].axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "3_gmm_em.png"), dpi=150)
    plt.close(fig)


def forward_backward(obs, pi, A, B):
    """Scaled forward-backward on many sequences at once. obs: (rows, T) symbol ids."""
    rows, T = obs.shape
    N = len(pi)
    alpha = np.zeros((rows, T, N))
    c = np.zeros((rows, T))
    alpha[:, 0] = pi * B[:, obs[:, 0]].T
    c[:, 0] = alpha[:, 0].sum(1)
    alpha[:, 0] /= c[:, 0, None]
    for t in range(1, T):
        alpha[:, t] = (alpha[:, t - 1] @ A) * B[:, obs[:, t]].T
        c[:, t] = alpha[:, t].sum(1)
        alpha[:, t] /= c[:, t, None]
    beta = np.ones((rows, T, N))
    for t in range(T - 2, -1, -1):
        beta[:, t] = ((B[:, obs[:, t + 1]].T * beta[:, t + 1]) @ A.T) / c[:, t + 1, None]
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    loglik = np.log(c).sum(1)
    return gamma, loglik


def viterbi(obs, pi, A, B):
    """Most likely hidden-state path for many sequences at once (log space)."""
    rows, T = obs.shape
    N = len(pi)
    logA, logB = np.log(A), np.log(B)
    delta = np.log(pi) + logB[:, obs[:, 0]].T
    back = np.zeros((rows, T, N), dtype=int)
    for t in range(1, T):
        scores = delta[:, :, None] + logA[None]
        back[:, t] = scores.argmax(1)
        delta = scores.max(1) + logB[:, obs[:, t]].T
    path = np.zeros((rows, T), dtype=int)
    path[:, -1] = delta.argmax(1)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[np.arange(rows), t, path[:, t]]
    return path, delta.max(1)


def run_hmm(x_scaled, km_states, rgb, size, out, n_symbols=8):
    heading("4. HMM along each pixel row  (forward-backward + Viterbi)")
    say("  Hidden states: background / healthy / lesion")
    say(f"  Observations: each pixel's colour, rounded to 1 of {n_symbols} colour codes")
    codebook = KMeans(n_clusters=n_symbols, n_init=5, random_state=SEED).fit(x_scaled)
    obs = codebook.labels_.reshape(size, size)
    states = km_states

    N = 3
    pi = np.bincount(states[:, 0], minlength=N) + 1.0
    pi /= pi.sum()
    A = np.ones((N, N))
    np.add.at(A, (states[:, :-1].ravel(), states[:, 1:].ravel()), 1)
    A /= A.sum(1, keepdims=True)
    B = np.ones((N, n_symbols))
    np.add.at(B, (states.ravel(), obs.ravel()), 1)
    B /= B.sum(1, keepdims=True)
    say("  Transition matrix A (estimated from the k-means labels, row = from, col = to):")
    say("  " + " " * 12 + "".join(f"{n:>12}" for n in STATE_NAMES))
    for i, n in enumerate(STATE_NAMES):
        say(f"  {n:<12}" + "".join(f"{A[i, j]:12.3f}" for j in range(N)))

    gamma, loglik = forward_backward(obs, pi, A, B)
    path, _ = viterbi(obs, pi, A, B)

    leafy = np.isin(states, [1, 2]).sum(1)
    row = int(np.argmax(leafy))
    say(f"  Example row {row} (the row with the most leaf):")
    say(f"    forward-backward: log P(row's colours | model) = {loglik[row]:.1f}")
    switches_raw = int((states[row, 1:] != states[row, :-1]).sum())
    switches_vit = int((path[row, 1:] != path[row, :-1]).sum())
    say(f"    state changes along the row: k-means {switches_raw}, Viterbi {switches_vit}")
    unsure = (gamma.max(2) < 0.9).mean() * 100
    say(f"  Whole image: pixels where forward-backward is unsure (top prob < 0.9): {unsure:.1f}%")
    changed = 100 * (path != states).mean()
    say(f"  Viterbi relabelled {changed:.1f}% of pixels compared with k-means (removes speckle)")
    say("  Viterbi map:")
    share_table(path)

    fig = plt.figure(figsize=(16, 8))
    g = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    a0 = fig.add_subplot(g[0, 0])
    a0.imshow(rgb)
    a0.axhline(row, color="red", lw=1.5)
    a0.set_title(f"Original (red line = example row {row})")
    a0.axis("off")
    a1 = fig.add_subplot(g[0, 1])
    a1.imshow(STATE_COLOURS[states])
    a1.set_title("k-means labels (input to HMM)")
    a1.axis("off")
    a2 = fig.add_subplot(g[0, 2])
    a2.imshow(STATE_COLOURS[path])
    a2.set_title("Viterbi: most likely states, row by row")
    a2.axis("off")
    a3 = fig.add_subplot(g[1, :])
    x = np.arange(size)
    for s, col in zip(range(N), ["#555555", "#2E7D32", "#E69F00"]):
        a3.plot(x, gamma[row, :, s], color=col, label=f"P({STATE_NAMES[s]}) forward-backward")
    a3.step(x, path[row] / 2, where="mid", color="black", ls="--", label="Viterbi state (0=bg, 0.5=healthy, 1=lesion)")
    a3.set(xlabel="pixel position along the row", ylabel="probability", ylim=(-0.05, 1.05),
           title=f"Row {row}: forward-backward probabilities and the Viterbi path")
    a3.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "4_hmm_forward_backward_viterbi.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="PCA, k-means, GMM/EM and HMM on one leaf photo")
    p.add_argument("image", help="path to a leaf photo (jpg/png)")
    p.add_argument("--out", default=None,
                   help="folder for the figures and results.txt (default: reports/single_leaf/<photo name>)")
    p.add_argument("--size", type=int, default=256, help="resize the photo to size x size pixels")
    args = p.parse_args()

    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        args.out = os.path.join(PROJECT_ROOT, "reports", "single_leaf", stem)
    os.makedirs(args.out, exist_ok=True)
    rgb, hsv = load_image(args.image, args.size)
    raw = pixel_features(rgb, hsv)
    x_scaled = StandardScaler().fit_transform(raw)
    say(f"Image: {args.image}  ->  {args.size}x{args.size} = {len(raw)} pixels, 7 features each")

    run_pca(x_scaled, args.size, args.out)
    km_states, km_centres = run_kmeans(x_scaled, raw, rgb, args.size, args.out)
    run_gmm(x_scaled, raw, rgb, args.size, args.out, km_centres)
    run_hmm(x_scaled, km_states, rgb, args.size, args.out)

    with open(os.path.join(args.out, "results.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved figures and results.txt to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
