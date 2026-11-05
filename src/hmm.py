"""
src/hmm.py - Hidden Markov Model for leaf images: forward-backward and Viterbi, from scratch in NumPy.

Idea: read every row of a leaf photo from left to right, like a sentence.
  hidden state z_t   what the pixel really is: 0 = background, 1 = healthy leaf, 2 = lesion
  observation x_t    the pixel's colour, quantised into 18 hue x 8 saturation x 4 brightness = 576 HSV bins
                     (the 32 coarse bins of the HSV-histogram features cannot tell beige background from
                     brown lesions: both are low hue; saturation needs finer steps)
  transition A       neighbouring pixels are usually the same tissue (A[i, j] = p(z_t = j | z_t-1 = i))
  emission B         which colours each tissue produces (B[k, m] = p(x_t = m | z_t = k))
  start pi           what the first pixel of a row usually is

Training (supervised): counts of states, transitions and emissions on training leaves, where the state of
every pixel comes from our HSV-threshold segmentation (src/features.py). Laplace smoothing avoids zeros.

Algorithms (Bishop, PRML, section 13.2):
  forward       alpha_t(k) = p(x_1..x_t, z_t = k), computed with scaling c_t to avoid underflow
                -> log-likelihood of a row  ln p(x_1..x_T) = sum_t ln c_t
  backward      beta_t(k)  = p(x_t+1..x_T | z_t = k)
  posterior     gamma_t(k) = p(z_t = k | whole row) = alpha_t(k) beta_t(k)   (forward-backward)
  Viterbi       the single most likely state path  argmax_z p(z | x), by dynamic programming in log space

Evaluation: decode held-out leaves and compare with the threshold segmentation. These labels are our own
segmentation, not hand-drawn ground truth - the comparison shows how the HMM's "neighbours agree" assumption
cleans up per-pixel decisions, not absolute accuracy.

Usage (from the plant-disease-ml folder):
    python -m src.hmm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/hmm.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

import config

STATES = ["background", "healthy leaf", "lesion"]
HMM_BINS = (18, 8, 4)          # hue, saturation, brightness bins of the observed colour
N_SYMBOLS = HMM_BINS[0] * HMM_BINS[1] * HMM_BINS[2]       # 576
SIZE = 128                     # rows of 128 pixels (images downsampled from 256 for speed)
TRAIN_PER_CLASS = 15
TEST_PER_CLASS = 8


# ----------------------------------------------------------------------------- the model
class HMM:
    """Discrete-observation HMM. All sequences in a batch must have the same length (image rows)."""

    def __init__(self, pi: np.ndarray, A: np.ndarray, B: np.ndarray):
        self.pi, self.A, self.B = pi, A, B

    @classmethod
    def fit_supervised(cls, obs: np.ndarray, states: np.ndarray, n_states: int, n_symbols: int,
                       smoothing: float = 1.0) -> "HMM":
        """Maximum-likelihood estimates from labelled sequences (counts + Laplace smoothing).

        obs, states: integer arrays of shape (n_sequences, T).
        """
        pi = np.bincount(states[:, 0], minlength=n_states) + smoothing
        A = np.full((n_states, n_states), smoothing)
        np.add.at(A, (states[:, :-1].ravel(), states[:, 1:].ravel()), 1)
        B = np.full((n_states, n_symbols), smoothing)
        np.add.at(B, (states.ravel(), obs.ravel()), 1)
        return cls(pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True))

    # ---- forward-backward
    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled forward pass for a batch (R, T). Returns alpha_hat (R, T, K) and scales c (R, T)."""
        R, T = obs.shape
        K = len(self.pi)
        alpha = np.empty((R, T, K))
        c = np.empty((R, T))
        a = self.pi[None, :] * self.B[:, obs[:, 0]].T
        c[:, 0] = a.sum(1)
        alpha[:, 0] = a / c[:, :1]
        for t in range(1, T):
            a = (alpha[:, t - 1] @ self.A) * self.B[:, obs[:, t]].T
            c[:, t] = a.sum(1)
            alpha[:, t] = a / c[:, t:t + 1]
        return alpha, c

    def backward(self, obs: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Scaled backward pass, using the forward scales. Returns beta_hat (R, T, K)."""
        R, T = obs.shape
        K = len(self.pi)
        beta = np.empty((R, T, K))
        beta[:, -1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[:, t] = (self.B[:, obs[:, t + 1]].T * beta[:, t + 1]) @ self.A.T / c[:, t + 1:t + 2]
        return beta

    def posteriors(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """gamma (R, T, K) = p(z_t = k | whole row) and the log-likelihood of each row (R,)."""
        alpha, c = self.forward(obs)
        beta = self.backward(obs, c)
        gamma = alpha * beta
        gamma /= gamma.sum(-1, keepdims=True)
        return gamma, np.log(c).sum(1)

    # ---- Viterbi
    def viterbi(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Most likely state path for every row (R, T) and its log-probability (R,)."""
        R, T = obs.shape
        K = len(self.pi)
        logA, logB = np.log(self.A), np.log(self.B)
        delta = np.log(self.pi)[None, :] + logB[:, obs[:, 0]].T
        back = np.empty((R, T, K), dtype=np.int8)
        for t in range(1, T):
            scores = delta[:, :, None] + logA[None, :, :]          # (R, from, to)
            back[:, t] = scores.argmax(1)
            delta = scores.max(1) + logB[:, obs[:, t]].T
        path = np.empty((R, T), dtype=np.int64)
        path[:, -1] = delta.argmax(1)
        for t in range(T - 1, 0, -1):
            path[:, t - 1] = back[np.arange(R), t, path[:, t]]
        return path, delta.max(1)

    def emission_only(self, obs: np.ndarray) -> np.ndarray:
        """Per-pixel decision WITHOUT neighbours: argmax_k p(k) p(x | k) (what the HMM improves on)."""
        prior = self.A.T @ self.pi                                  # rough state frequencies
        return (np.log(prior)[:, None] + np.log(self.B)).argmax(0)[obs]


# ----------------------------------------------------------------------------- images -> sequences
def quantise(img_bgr: np.ndarray) -> np.ndarray:
    """Pixel colour -> one of N_SYMBOLS HSV symbols (hue-major order: symbol = (h * S + s) * V + v)."""
    hb, sb, vb = HMM_BINS
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int64)
    h = np.minimum(hsv[..., 0] * hb // 180, hb - 1)
    s = (hsv[..., 1] * sb) // 256
    v = (hsv[..., 2] * vb) // 256
    return (h * sb + s) * vb + v


def leaf_rows(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(symbols, states, rgb) of one leaf at SIZE x SIZE; states come from the threshold segmentation."""
    from src import features as F
    img = F.preprocess(F.read_image(path))
    seg = F.segment_leaf(img)
    labels = np.zeros(seg.mask.shape, dtype=np.int64)
    labels[seg.mask > 0] = 1
    labels[seg.brown > 0] = 2
    small = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    labels = cv2.resize(labels.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return quantise(small), labels, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def pick_paths(paths, y, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for k in np.unique(y):
        idx = np.flatnonzero(y == k)
        out += [paths[i] for i in rng.choice(idx, size=min(per_class, len(idx)), replace=False)]
    return [p for p in out if Path(p).exists()]


# ----------------------------------------------------------------------------- experiment
def run(data: dict, reports_dir: Path = config.REPORTS_DIR) -> dict:
    from src.evaluate import save_table, text_table
    config.set_seeds()
    train_paths = pick_paths(list(data["paths_train"]), data["y_train"], TRAIN_PER_CLASS, config.SEED)
    test_paths = pick_paths(list(data["paths_valid"]), data["y_valid"], TEST_PER_CLASS, config.SEED + 1)
    if not train_paths or not test_paths:
        raise FileNotFoundError("The leaf photos listed in the feature cache were not found on this computer.")

    tr = [leaf_rows(p) for p in train_paths]
    obs_tr = np.vstack([o for o, _, _ in tr]); st_tr = np.vstack([s for _, s, _ in tr])
    hmm = HMM.fit_supervised(obs_tr, st_tr, len(STATES), N_SYMBOLS)
    print(f"Trained on {len(tr)} leaves = {len(obs_tr):,} rows of {SIZE} pixels "
          f"({obs_tr.size:,} labelled pixels).")
    print("Transition matrix A (row = from, column = to):")
    print(text_table(pd.DataFrame(hmm.A.round(4), columns=STATES).assign(**{"from": STATES})[["from"] + STATES]))

    te = [leaf_rows(p) for p in test_paths]
    obs_te = np.vstack([o for o, _, _ in te]); st_te = np.vstack([s for _, s, _ in te])
    vit, _ = hmm.viterbi(obs_te)
    gamma, loglik = hmm.posteriors(obs_te)
    post = gamma.argmax(-1)
    naive = hmm.emission_only(obs_te)

    def scores(pred):
        acc = 100 * float((pred == st_te).mean())
        les = st_te == 2
        rec = 100 * float((pred[les] == 2).mean()) if les.any() else float("nan")
        prec = 100 * float((st_te[pred == 2] == 2).mean()) if (pred == 2).any() else float("nan")
        switches = float((pred[:, 1:] != pred[:, :-1]).sum(1).mean())
        return {"Pixel agreement (%)": acc, "Lesion recall (%)": rec, "Lesion precision (%)": prec,
                "State changes per row": switches}

    rows = [{"Decoder": "Per-pixel colour only (no HMM)", **scores(naive)},
            {"Decoder": "Forward-backward (most probable state per pixel)", **scores(post)},
            {"Decoder": "Viterbi (most probable whole path)", **scores(vit)},
            {"Decoder": "Threshold segmentation (reference labels)", "Pixel agreement (%)": 100.0,
             "Lesion recall (%)": 100.0, "Lesion precision (%)": 100.0,
             "State changes per row": float((st_te[:, 1:] != st_te[:, :-1]).sum(1).mean())}]
    table = pd.DataFrame(rows)
    print(f"\nDecoding {len(te)} held-out leaves ({len(obs_te):,} rows); average row log-likelihood "
          f"{loglik.mean():.1f}:\n" + text_table(table))

    save_table(table.round(2), "hmm_decoding", reports_dir,
               f"HMM decoding of {len(te)} held-out leaves, compared with the threshold segmentation")
    save_table(pd.DataFrame(hmm.A, index=STATES, columns=STATES).round(4).reset_index().rename(
        columns={"index": "from \\ to"}), "hmm_transitions", reports_dir, "HMM transition matrix A")
    make_graphs(hmm, te, test_paths, reports_dir)
    return {"hmm": hmm, "table": table, "loglik": float(loglik.mean())}


# ----------------------------------------------------------------------------- graphs
def make_graphs(hmm: HMM, te, test_paths, reports_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    from src.evaluate import COLORS as C
    from src.evaluate import apply_style, save_fig
    apply_style()
    state_cmap = ListedColormap(["#ffffff", C["healthy"], C["diseased"]])
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#f6c9b3", C["diseased"], "#8a2f0b"])

    # pick 3 test leaves with the most lesion so the pictures are interesting
    order = np.argsort([-(s == 2).mean() for _, s, _ in te])[:3]

    # ---- 1. original | threshold labels | Viterbi | forward-backward P(lesion)
    fig, axes = plt.subplots(len(order), 4, figsize=(12, 3.2 * len(order)), squeeze=False)
    for r, i in enumerate(order):
        obs, st, rgb = te[i]
        vit, _ = hmm.viterbi(obs)
        gamma, _ = hmm.posteriors(obs)
        name = config.pretty_class(Path(test_paths[i]).parent.name)
        panels = [(rgb, None, f"{name}"), (st, state_cmap, "threshold segmentation"),
                  (vit, state_cmap, f"Viterbi path (lesion {100 * (vit == 2).mean():.1f}%)"),
                  (gamma[..., 2], seq, "forward-backward P(lesion)")]
        for c, (img, cmap, title) in enumerate(panels):
            ax = axes[r, c]
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=2 if cmap is state_cmap else 1, interpolation="nearest")
            ax.set_title(title, fontsize=9); ax.axis("off")
    fig.suptitle("HMM on every image row: white = background, aqua = healthy leaf, orange = lesion",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "hmm", "1_viterbi_and_forward_backward_leaves", reports_dir)

    # ---- 2. one row in detail: posterior probabilities along the row + Viterbi path
    obs, st, rgb = te[order[0]]
    lesion_rows = np.flatnonzero((st == 2).sum(1) > 0)
    row = int(lesion_rows[len(lesion_rows) // 2]) if len(lesion_rows) else SIZE // 2
    gamma, ll = hmm.posteriors(obs[row:row + 1])
    vit, _ = hmm.viterbi(obs[row:row + 1])
    t = np.arange(SIZE)
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True,
                                     gridspec_kw={"height_ratios": [0.6, 2.2, 0.6]})
    a0.imshow(rgb[row:row + 1].repeat(6, axis=0), aspect="auto"); a0.set_yticks([])
    a0.set_title(f"Row {row} of the leaf (the observed colours), log-likelihood ln p(x) = {ll[0]:.1f}", fontsize=10)
    for k, col, ls in [(0, C["muted"], ":"), (1, C["healthy"], "-"), (2, C["diseased"], "-")]:
        a1.plot(t, gamma[0, :, k], color=col, ls=ls, lw=2, label=f"P({STATES[k]} | whole row)")
    a1.set_ylabel("forward-backward\nposterior"); a1.set_ylim(-0.03, 1.03); a1.legend(fontsize=8.5, loc="center right")
    a2.imshow(np.vstack([st[row], vit[0]]), cmap=state_cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest")
    a2.set_yticks([0, 1], ["threshold", "Viterbi"], fontsize=8.5); a2.grid(False)
    a2.set_xlabel("pixel position along the row (left -> right)")
    fig.suptitle("Forward-backward gives a probability per pixel; Viterbi picks the single best path",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save_fig(fig, "hmm", "2_one_row_forward_backward_viterbi", reports_dir)

    # ---- 3. the learned parameters: transition matrix A and emissions B (summed over sat./brightness)
    hb, sb, vb = HMM_BINS
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 2.2]})
    im = a1.imshow(hmm.A, cmap=seq, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a1.text(j, i, f"{hmm.A[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if hmm.A[i, j] > 0.6 else C["ink"])
    a1.set_xticks(range(3), STATES, rotation=20, fontsize=8.5); a1.set_yticks(range(3), STATES, fontsize=8.5)
    a1.set_xlabel("to"); a1.set_ylabel("from"); a1.grid(False)
    a1.set_title("Transition matrix A: tissue rarely changes\nfrom one pixel to the next", fontsize=10)
    fig.colorbar(im, ax=a1, shrink=0.8)
    hue = hmm.B.reshape(3, hb, sb * vb).sum(-1)
    centres = (np.arange(hb) + 0.5) * 360 / hb                       # hue in degrees
    for k, col, mk in [(0, C["muted"], "o"), (1, C["healthy"], "s"), (2, C["diseased"], "^")]:
        a2.plot(centres, hue[k], color=col, marker=mk, ms=5, label=STATES[k])
    a2.set_xlabel("hue of the pixel (degrees: 0 = red/brown, 60 = yellow, 120 = green, 240 = blue)")
    a2.set_ylabel("p(hue | tissue)"); a2.set_xticks(range(0, 361, 30)); a2.legend(fontsize=9)
    a2.set_title("Emission probabilities B: which colours each tissue produces", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "hmm", "3_learned_parameters", reports_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HMM forward-backward and Viterbi on leaf image rows.")
    ap.add_argument("--limit", type=int, default=None, help="use the --limit N feature cache")
    args = ap.parse_args(argv)
    from src import dataset
    try:
        data = dataset.load_features(args.limit)
        run(data)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(f"\nGraphs written to {config.figure_dir('hmm')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
