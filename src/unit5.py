"""
src/unit5.py - Unit 5 of the syllabus, on the same 55 leaf features as the rest of the project.

  Part 1  Neural networks           a multilayer perceptron written from scratch in NumPy:
                                    forward pass, error backpropagation, gradient check, Adam;
                                    compared with sklearn's MLPClassifier
  Part 2  Combining models          AdaBoost (SAMME) written from scratch, committees (hard vote,
                                    soft vote) and stacking; the committee error vs the members' error
  Part 3  Kernel methods &          SVM dual / support vectors (the decision function rebuilt by hand),
          sparse kernel machines    Gaussian-process regression, and a Relevance Vector Machine written
                                    from scratch, compared with Support Vector Regression
  Part 4  Sampling methods &        Metropolis-Hastings MCMC for Bayesian logistic regression (vs the Laplace
          graphical models          approximation), Gibbs sampling in a Markov random field (Ising model) that
                                    cleans the lesion mask of a real leaf, and a Bayesian network with exact
                                    inference, rejection sampling, likelihood weighting and Gibbs sampling

Reference: C. M. Bishop, Pattern Recognition and Machine Learning (2006), chapters 5, 6, 7, 8, 11 and 14.

Usage (from the plant-disease-ml folder):
    python -m src.unit5                    # all four parts (about 1-2 minutes on a laptop)
    python -m src.unit5 --part nn          # one part: nn | combining | kernels | sampling
    python -m src.unit5 --limit 100        # use the --limit 100 feature cache

Graphs go to reports/figures/15_... to 18_..., tables to reports/tables/unit5_*.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/unit5.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.special import expit, logsumexp

import config

PARTS = ("nn", "combining", "kernels", "sampling")


def say(text: str = "") -> None:
    print(text, flush=True)


def heading(text: str) -> None:
    say()
    say("=" * 78)
    say(text)
    say("=" * 78)


def short(name: str) -> str:
    """'Tomato___Early_blight' -> 'Early blight' (and short names for the long ones)."""
    s = config.pretty_class(name).split(": ", 1)[-1]
    s = {"Tomato Yellow Leaf Curl Virus": "Yellow leaf curl virus"}.get(s, s)
    return s[:1].upper() + s[1:]


def standardise(data: dict):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data["X_train"])
    return scaler.transform(data["X_train"]), scaler.transform(data["X_valid"])


def stratified_idx(y: np.ndarray, per_class: int, seed: int = config.SEED) -> np.ndarray:
    """Up to per_class random indices of every class."""
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())),
                                              replace=False) for c in np.unique(y)]))


# =============================================================================================================
# Part 1 - Neural networks (Bishop chapter 5)
# =============================================================================================================
def softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)                 # subtract the max: same result, no overflow
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y] = 1
    return Y


class MLP:
    """Two-layer network  x (D) -> tanh hidden layer (H) -> softmax output (K).

    forward     a1 = x W1 + b1,   z = tanh(a1),   a2 = z W2 + b2,   p = softmax(a2)
    error       E = -(1/N) sum_n sum_k t_nk ln p_nk  +  (lambda/2) (||W1||^2 + ||W2||^2)
    backprop    delta2 = (p - t) / N                         output error (softmax + cross-entropy)
                delta1 = (delta2 W2^T) * (1 - z^2)            sent back through tanh'(a) = 1 - tanh(a)^2
                dE/dW2 = z^T delta2 + lambda W2,   dE/dW1 = x^T delta1 + lambda W1
    training    mini-batch gradient descent with the Adam update rule
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, weight_decay: float = 1e-4, seed: int = config.SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_hidden))       # small random weights, so the
        self.b1 = np.zeros(n_hidden)                                       # tanh units start in their
        self.W2 = rng.normal(0, 1 / np.sqrt(n_hidden), (n_hidden, n_out))  # linear (non-saturated) range
        self.b2 = np.zeros(n_out)
        self.weight_decay = weight_decay
        self.history: dict[str, list[float]] = {}

    @property
    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.tanh(X @ self.W1 + self.b1)
        return z, softmax(z @ self.W2 + self.b2)

    def loss(self, X: np.ndarray, T: np.ndarray) -> float:
        _, p = self.forward(X)
        ce = -np.mean(np.sum(T * np.log(np.clip(p, 1e-12, None)), axis=1))
        return float(ce + 0.5 * self.weight_decay * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)))

    def gradients(self, X: np.ndarray, T: np.ndarray) -> list[np.ndarray]:
        """Error backpropagation: the exact gradient of loss() for every weight, in one backward pass."""
        z, p = self.forward(X)
        d2 = (p - T) / len(X)                       # dE/da2
        d1 = (d2 @ self.W2.T) * (1 - z ** 2)        # dE/da1
        return [X.T @ d1 + self.weight_decay * self.W1, d1.sum(axis=0),
                z.T @ d2 + self.weight_decay * self.W2, d2.sum(axis=0)]

    def fit(self, X, y, X_valid=None, y_valid=None, epochs: int = 60, batch_size: int = 128, lr: float = 3e-3,
            seed: int = config.SEED) -> "MLP":
        k = self.b2.size
        T = one_hot(y, k)
        rng = np.random.default_rng(seed)
        m = [np.zeros_like(p) for p in self.params]          # Adam: running mean of the gradient
        v = [np.zeros_like(p) for p in self.params]          # Adam: running mean of the squared gradient
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        hist = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                step += 1
                for p, g, mi, vi in zip(self.params, self.gradients(X[idx], T[idx]), m, v):
                    mi *= b1; mi += (1 - b1) * g
                    vi *= b2; vi += (1 - b2) * g * g
                    p -= lr * (mi / (1 - b1 ** step)) / (np.sqrt(vi / (1 - b2 ** step)) + eps)
            hist["train_loss"].append(self.loss(X, T))
            hist["train_acc"].append(100 * float(np.mean(self.predict(X) == y)))
            if X_valid is not None:
                hist["valid_loss"].append(self.loss(X_valid, one_hot(y_valid, k)))
                hist["valid_acc"].append(100 * float(np.mean(self.predict(X_valid) == y_valid)))
        self.history = hist
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def gradient_check(model: MLP, X: np.ndarray, T: np.ndarray, n_checks: int = 30, eps: float = 1e-6,
                   seed: int = config.SEED) -> float:
    """Largest relative difference between backprop and central finite differences (E(w+e) - E(w-e)) / 2e.

    Below about 1e-5 means the backpropagation code is correct (a bug gives errors near 1).
    """
    rng = np.random.default_rng(seed)
    grads = model.gradients(X, T)
    worst = 0.0
    for _ in range(n_checks):
        i = int(rng.integers(len(model.params)))
        p, g = model.params[i], grads[i]
        j = tuple(int(rng.integers(s)) for s in p.shape)
        old = p[j]
        p[j] = old + eps; up = model.loss(X, T)
        p[j] = old - eps; down = model.loss(X, T)
        p[j] = old
        num = (up - down) / (2 * eps)
        worst = max(worst, abs(num - g[j]) / max(1e-8, abs(num) + abs(g[j])))
    return worst


def part_nn(data: dict, reports_dir: Path) -> dict:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    from src.evaluate import save_table, text_table

    heading("UNIT 5 - PART 1: NEURAL NETWORKS  (multilayer perceptron, backpropagation)")
    X, Xv = standardise(data)
    y, yv = data["y_train"], data["y_valid"]
    k = len(data["class_names"])
    say(f"Network: {X.shape[1]} inputs (leaf features) -> 32 tanh hidden units -> {k} softmax outputs (diseases)")
    say(f"Weights to learn: {X.shape[1] * 32 + 32 + 32 * k + k:,}   training leaves: {len(y):,}   "
        f"held-out leaves: {len(yv):,}")

    # 1. is backpropagation correct?  compare with numerical derivatives
    probe = MLP(X.shape[1], 32, k, seed=config.SEED)
    idx = stratified_idx(y, 20)
    err = gradient_check(probe, X[idx], one_hot(y[idx], k))
    say(f"\n[1] Gradient check: backprop vs finite differences, worst relative error = {err:.1e} "
        f"({'correct' if err < 1e-5 else 'WRONG'})")

    # 2. train our MLP, and sklearn's with the same architecture
    t = time.time()
    net = MLP(X.shape[1], 32, k).fit(X, y, Xv, yv, epochs=60)
    t_ours = time.time() - t
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk = MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-4 * len(X) / 128,
                           batch_size=128, learning_rate_init=3e-3, max_iter=60, random_state=config.SEED).fit(X, y)
    t_sk = time.time() - t
    t = time.time()
    lr = LogisticRegression(C=100, max_iter=5000).fit(X, y)
    t_lr = time.time() - t
    acc = lambda m: 100 * float(np.mean(m.predict(Xv) == yv))
    rows = [{"Model": "MLP from scratch (NumPy, 32 tanh units)", "Valid accuracy (%)": acc(net),
             "Train accuracy (%)": net.history["train_acc"][-1], "Fit time (s)": t_ours},
            {"Model": "sklearn MLPClassifier (same architecture)", "Valid accuracy (%)": acc(sk),
             "Train accuracy (%)": 100 * float(np.mean(sk.predict(X) == y)), "Fit time (s)": t_sk},
            {"Model": "Logistic regression (no hidden layer)", "Valid accuracy (%)": acc(lr),
             "Train accuracy (%)": 100 * float(np.mean(lr.predict(X) == y)), "Fit time (s)": t_lr}]
    table = pd.DataFrame(rows)
    say("\n[2] Training 60 epochs of mini-batch Adam:\n" + text_table(table))
    say("    The hidden layer learns its own non-linear features of the 55 inputs, so the network beats the")
    say("    purely linear logistic regression.")

    # 3. how many hidden units?
    sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    sweep = []
    for h in sizes:
        m = MLP(X.shape[1], h, k).fit(X, y, Xv, yv, epochs=40)
        sweep.append({"Hidden units": h, "Train accuracy (%)": m.history["train_acc"][-1],
                      "Valid accuracy (%)": m.history["valid_acc"][-1]})
    sweep = pd.DataFrame(sweep)
    say("\n[3] Model complexity - number of hidden units (40 epochs each):\n" + text_table(sweep))

    save_table(table, "unit5_nn_results", reports_dir, "Unit 5 - neural network vs sklearn vs logistic regression")
    save_table(sweep, "unit5_nn_hidden_units", reports_dir, "Unit 5 - accuracy vs number of hidden units")
    graphs = nn_graphs(net, sweep, X, y, data["class_names"], reports_dir)
    return {"grad_check": err, "table": table, "sweep": sweep, "net": net, "graphs": graphs}


def nn_graphs(net: MLP, sweep, X, y, class_names, reports_dir) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS
    out = []
    h = net.history
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], color=C["blue"], label="training leaves")
    ax[0].plot(ep, h["valid_loss"], color=C["orange"], label="held-out leaves")
    ax[0].set(xlabel="epoch (one pass over the training leaves)", ylabel="cross-entropy error E(w)",
              title="Backpropagation + Adam drive the error down")
    ax[0].legend()
    ax[1].plot(ep, h["train_acc"], color=C["blue"], label="training leaves")
    ax[1].plot(ep, h["valid_acc"], color=C["orange"], label="held-out leaves")
    ax[1].set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy while training")
    ax[1].legend()
    fig.tight_layout()
    out.append(save_fig(fig, "nn", "1_training_curves", reports_dir))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep["Hidden units"], sweep["Train accuracy (%)"], "o-", color=C["blue"], label="training leaves")
    ax.plot(sweep["Hidden units"], sweep["Valid accuracy (%)"], "s-", color=C["orange"], label="held-out leaves")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sweep["Hidden units"], [str(v) for v in sweep["Hidden units"]])
    ax.set(xlabel="hidden units H (log scale)", ylabel="accuracy (%)",
           title="More hidden units = a more flexible model (gap = over-fitting)")
    ax.legend()
    out.append(save_fig(fig, "nn", "2_accuracy_vs_hidden_units", reports_dir))

    # 2-D picture: logistic regression (straight boundaries) vs the MLP (curved boundaries) on 2 PCA components
    Z = PCA(2, random_state=config.SEED).fit_transform(X)
    idx = stratified_idx(y, 300)
    Z, yy = Z[idx], y[idx]
    k = len(class_names)
    lin = LogisticRegression(max_iter=5000).fit(Z, yy)
    mlp = MLP(2, 32, k, weight_decay=1e-4).fit(Z, yy, epochs=300, batch_size=64, lr=1e-2)
    gx = np.linspace(np.percentile(Z[:, 0], 1) - 1, np.percentile(Z[:, 0], 99) + 1, 300)
    gy = np.linspace(np.percentile(Z[:, 1], 1) - 1, np.percentile(Z[:, 1], 99) + 1, 300)
    GX, GY = np.meshgrid(gx, gy)
    G = np.column_stack([GX.ravel(), GY.ravel()])
    palette = ["#eb6834", "#2a78d6", "#a35ac9", "#d4a017", "#1baf7a"][:k]
    light = ListedColormap([_lighten(c) for c in palette])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, model, title in [(axes[0], lin, "Logistic regression: straight-line boundaries"),
                             (axes[1], mlp, "Neural network (32 tanh units): curved boundaries")]:
        ax.contourf(GX, GY, model.predict(G).reshape(GX.shape), levels=np.arange(k + 1) - 0.5, cmap=light)
        for c in range(k):
            ax.scatter(Z[yy == c, 0], Z[yy == c, 1], s=7, color=palette[c], label=short(class_names[c]))
        acc = 100 * np.mean(model.predict(Z) == yy)
        ax.set(title=f"{title}\n(accuracy on these leaves {acc:.1f}%)", xlabel="PC1", xlim=gx[[0, -1]],
               ylim=gy[[0, -1]])
        ax.grid(False)
    axes[0].set_ylabel("PC2")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=k, fontsize=9, markerscale=2.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    out.append(save_fig(fig, "nn", "3_decision_boundaries_2d", reports_dir))
    return out


def _lighten(hex_colour: str, amount: float = 0.72) -> tuple[float, float, float]:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# =============================================================================================================
# Part 2 - Combining models (Bishop chapter 14)
# =============================================================================================================
class AdaBoostSAMME:
    """AdaBoost written from scratch (Bishop 14.3), in its multi-class form SAMME (Zhu et al., 2009).

    start      every training leaf has weight w_n = 1/N
    round m    fit a weak learner y_m (a decision stump) to the WEIGHTED leaves
               eps_m   = sum of the weights of the leaves it gets wrong
               alpha_m = ln((1 - eps_m) / eps_m) + ln(K - 1)        (K = 2 gives Bishop's AdaBoost exactly)
               w_n    <- w_n * exp(alpha_m * [y_m(x_n) != t_n]), then normalise  -> mistakes get heavier
    predict    the class with the largest  sum_m alpha_m [y_m(x) = class]
    """

    def __init__(self, n_rounds: int = 300, max_depth: int = 1, seed: int = config.SEED):
        self.n_rounds, self.max_depth, self.seed = n_rounds, max_depth, seed
        self.learners, self.alphas, self.history = [], [], {}

    def fit(self, X, y, X_valid=None, y_valid=None) -> "AdaBoostSAMME":
        from sklearn.tree import DecisionTreeClassifier
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        w = np.full(len(y), 1 / len(y))
        F = np.zeros((len(y), k))
        Fv = None if X_valid is None else np.zeros((len(X_valid), k))
        hist = {"weighted_error": [], "alpha": [], "train_error": [], "valid_error": []}
        for _ in range(self.n_rounds):
            stump = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.seed)
            stump.fit(X, y, sample_weight=w)
            pred = stump.predict(X)
            miss = pred != y
            eps = float(np.sum(w[miss]) / np.sum(w))
            if eps >= 1 - 1 / k:                      # no better than guessing: stop
                break
            alpha = np.log((1 - eps) / max(eps, 1e-12)) + np.log(k - 1)
            w = w * np.exp(alpha * miss)
            w /= w.sum()
            self.learners.append(stump)
            self.alphas.append(alpha)
            F[np.arange(len(y)), np.searchsorted(self.classes_, pred)] += alpha
            hist["weighted_error"].append(eps)
            hist["alpha"].append(alpha)
            hist["train_error"].append(100 * float(np.mean(self.classes_[F.argmax(1)] != y)))
            if Fv is not None:
                Fv[np.arange(len(X_valid)), np.searchsorted(self.classes_, stump.predict(X_valid))] += alpha
                hist["valid_error"].append(100 * float(np.mean(self.classes_[Fv.argmax(1)] != y_valid)))
            if eps <= 1e-12:                          # perfect weak learner: nothing left to fix
                break
        self.history = hist
        return self

    def decision_function(self, X) -> np.ndarray:
        F = np.zeros((len(X), len(self.classes_)))
        for stump, alpha in zip(self.learners, self.alphas):
            F[np.arange(len(X)), np.searchsorted(self.classes_, stump.predict(X))] += alpha
        return F

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def committee_members():
    """Five different, quick classifiers: the members of the committee."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    return {"Logistic Regression": LogisticRegression(C=100, max_iter=5000),
            "LDA": LinearDiscriminantAnalysis(),
            "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-6),
            "KNN (k = 7)": KNeighborsClassifier(7),
            "Decision Tree (depth 10)": DecisionTreeClassifier(max_depth=10, random_state=config.SEED)}


def part_combining(data: dict, reports_dir: Path) -> dict:
    import pandas as pd
    from sklearn.base import clone
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    from src.evaluate import save_table, text_table

    heading("UNIT 5 - PART 2: COMBINING MODELS  (boosting, committees, stacking)")
    X, Xv = standardise(data)
    y, yv = data["y_train"], data["y_valid"]

    # 1. AdaBoost from scratch vs sklearn
    rounds = 300
    t = time.time()
    ada = AdaBoostSAMME(n_rounds=rounds).fit(X, y, Xv, yv)
    t_ours = time.time() - t
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=config.SEED), n_estimators=rounds,
                            learning_rate=1.0, random_state=config.SEED).fit(X, y)
    stump = DecisionTreeClassifier(max_depth=1, random_state=config.SEED).fit(X, y)
    acc = lambda pred: 100 * float(np.mean(pred == yv))
    agree = 100 * float(np.mean(ada.predict(Xv) == sk.predict(Xv)))
    say(f"[1] AdaBoost (SAMME) from scratch, {len(ada.learners)} decision stumps, {t_ours:.1f} s")
    say(f"    one stump alone          : {acc(stump.predict(Xv)):5.1f}% held-out accuracy")
    say(f"    AdaBoost, ours           : {acc(ada.predict(Xv)):5.1f}%")
    say(f"    AdaBoost, sklearn        : {acc(sk.predict(Xv)):5.1f}%   (same predictions on {agree:.1f}% of leaves)")
    say("    Each stump asks ONE yes/no question about ONE feature. Re-weighting the leaves the previous")
    say("    stumps got wrong forces each new stump to fix those mistakes; their weighted vote is strong.")

    # 2. committees: hard vote, soft vote, stacking
    members = committee_members()
    probs, preds, rows = {}, {}, []
    for name, model in members.items():
        m = clone(model).fit(X, y)
        probs[name], preds[name] = m.predict_proba(Xv), m.predict(Xv)
        rows.append({"Model": name, "Kind": "member", "Valid accuracy (%)": acc(preds[name])})
    P = np.stack(list(probs.values()))                              # (members, leaves, classes)
    k = P.shape[2]
    votes = np.stack(list(preds.values()))
    hard = np.array([np.bincount(col, minlength=k).argmax() for col in votes.T])
    soft = P.mean(axis=0).argmax(axis=1)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.SEED)
    oof = np.hstack([cross_val_predict(clone(m), X, y, cv=cv, method="predict_proba") for m in members.values()])
    meta = LogisticRegression(max_iter=5000).fit(oof, y)
    stack = meta.predict(np.hstack(list(probs.values())))
    member_err = np.mean([100 - r["Valid accuracy (%)"] for r in rows])
    rows += [{"Model": "Committee - hard (majority) vote", "Kind": "committee", "Valid accuracy (%)": acc(hard)},
             {"Model": "Committee - soft vote (average probability)", "Kind": "committee",
              "Valid accuracy (%)": acc(soft)},
             {"Model": "Stacking (logistic regression on members' outputs)", "Kind": "committee",
              "Valid accuracy (%)": acc(stack)},
             {"Model": "AdaBoost from scratch (300 stumps)", "Kind": "boosting",
              "Valid accuracy (%)": acc(ada.predict(Xv))}]
    table = pd.DataFrame(rows)
    say("\n[2] Committees of 5 different models:\n" + text_table(table))
    say(f"    Average error of the members: {member_err:.1f}%   soft-vote committee error: {100 - acc(soft):.1f}%")
    say("    Bishop 14.2: averaging cancels the members' independent mistakes, so E_committee <= E_average.")
    say("    A plain vote is pulled down by weak members (Naive Bayes); stacking LEARNS how much to trust each")
    say("    member, so it beats every single member.")

    save_table(table, "unit5_combining", reports_dir, "Unit 5 - boosting, committees and stacking")
    ada_hist = pd.DataFrame({"Round": np.arange(1, len(ada.alphas) + 1), **ada.history})
    save_table(ada_hist, "unit5_adaboost_rounds", reports_dir, "Unit 5 - AdaBoost from scratch, round by round")
    graphs = combining_graphs(ada, sk, Xv, yv, table, member_err, reports_dir)
    return {"ada": ada, "table": table, "agree": agree, "member_error": member_err, "graphs": graphs}


def combining_graphs(ada, sk, Xv, yv, table, member_err, reports_dir) -> list[Path]:
    import matplotlib.pyplot as plt

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS
    out = []
    h = ada.history
    r = np.arange(1, len(h["alpha"]) + 1)
    sk_err = [100 * np.mean(p != yv) for p in sk.staged_predict(Xv)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(r, h["train_error"], color=C["blue"], label="training leaves (ours)")
    ax[0].plot(r, h["valid_error"], color=C["orange"], label="held-out leaves (ours)")
    ax[0].plot(np.arange(1, len(sk_err) + 1), sk_err, ":", color=C["ink"], lw=1.4, label="held-out (sklearn)")
    ax[0].set(xlabel="boosting round m (number of stumps)", ylabel="error (%)",
              title="AdaBoost: every new stump fixes earlier mistakes")
    ax[0].legend()
    l1 = ax[1].plot(r, 100 * np.array(h["weighted_error"]), color=C["muted"], lw=1.2,
                    label="eps_m  weighted error (%), left axis")
    ax2 = ax[1].twinx()
    l2 = ax2.plot(r, h["alpha"], color=C["aqua"], lw=1.2, label="alpha_m  vote weight, right axis")
    ax2.set_ylabel("alpha_m", color=C["aqua"])
    ax2.grid(False)
    ax[1].set(xlabel="boosting round m", ylabel="eps_m (%)",
              title="Later stumps face harder leaves:\nhigher error eps_m, smaller vote alpha_m")
    ax[1].legend(l1 + l2, [ln.get_label() for ln in l1 + l2], loc="center right", fontsize=8)
    fig.tight_layout()
    out.append(save_fig(fig, "combining", "1_adaboost_rounds", reports_dir))

    t = table.iloc[::-1]
    colours = {"member": C["muted"], "committee": C["blue"], "boosting": C["orange"]}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(t["Model"], t["Valid accuracy (%)"], color=[colours[kd] for kd in t["Kind"]])
    for yy, v in enumerate(t["Valid accuracy (%)"]):
        ax.text(v + 0.3, yy, f"{v:.1f}%", va="center", fontsize=9)
    ax.axvline(100 - member_err, ls="--", color=C["ink"], lw=1)
    ax.text(100 - member_err, len(t) - 0.4, "  average member", fontsize=8, color=C["ink"])
    lo = max(0, float(t["Valid accuracy (%)"].min()) - 8)
    ax.set(xlim=(lo, 101), xlabel="held-out accuracy (%)",
           title="Combining models: grey = single members, blue = committees, orange = boosting")
    ax.grid(axis="y", visible=False)
    out.append(save_fig(fig, "combining", "2_committee_vs_members", reports_dir))
    return out


# =============================================================================================================
# Part 3 - Kernel methods and sparse kernel machines (Bishop chapters 6 and 7)
# =============================================================================================================
def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    """k(x, x') = exp(-gamma ||x - x'||^2)."""
    d2 = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0))


class RVMRegression:
    """Relevance Vector Machine for regression, from scratch (Bishop 7.2.1, Tipping 2001).

    model      t = sum_n w_n k(x, x_n) + w_0 + noise,   noise ~ N(0, 1/beta)
    prior      w_i ~ N(0, 1/alpha_i)      ONE precision per basis function (automatic relevance determination)
    posterior  Sigma = (A + beta Phi^T Phi)^-1,   m = beta Sigma Phi^T t,   A = diag(alpha)
    evidence   gamma_i = 1 - alpha_i Sigma_ii
               alpha_i <- gamma_i / m_i^2,   1/beta <- ||t - Phi m||^2 / (N - sum gamma_i)
    sparsity   most alpha_i grow to infinity -> w_i = 0, that leaf is pruned. The leaves that survive are the
               'relevance vectors'.
    """

    def __init__(self, gamma: float, max_iter: int = 300, prune_at: float = 1e9, tol: float = 1e-4):
        self.gamma, self.max_iter, self.prune_at, self.tol = gamma, max_iter, prune_at, tol

    def _phi(self, X: np.ndarray, centres: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((len(X), 1)), rbf_kernel(X, centres, self.gamma)])

    def fit(self, X: np.ndarray, t: np.ndarray) -> "RVMRegression":
        Phi = self._phi(X, X)
        n = len(t)
        keep = np.arange(Phi.shape[1])                     # column 0 = bias, column j = leaf j - 1
        alpha = np.full(Phi.shape[1], 1.0)
        beta = 1.0 / max(np.var(t) * 0.1, 1e-12)
        self.n_iter_ = 0
        for it in range(self.max_iter):
            P = Phi[:, keep]
            Sigma = np.linalg.inv(np.diag(alpha[keep]) + beta * P.T @ P)
            m = beta * Sigma @ P.T @ t
            gam = 1 - alpha[keep] * np.diag(Sigma)
            new_alpha = gam / np.maximum(m ** 2, 1e-300)
            beta = (n - gam.sum()) / max(np.sum((t - P @ m) ** 2), 1e-12)
            change = np.max(np.abs(np.log(new_alpha + 1e-300) - np.log(alpha[keep] + 1e-300)))
            alpha[keep] = new_alpha
            self.n_iter_ = it + 1
            alive = alpha[keep] < self.prune_at
            if not alive.all():
                keep = keep[alive]
            elif change < self.tol:
                break
        P = Phi[:, keep]
        self.Sigma_ = np.linalg.inv(np.diag(alpha[keep]) + beta * P.T @ P)
        self.m_ = beta * self.Sigma_ @ P.T @ t
        self.beta_ = beta
        self.has_bias_ = bool(keep[0] == 0) if len(keep) else False
        rel = keep[keep > 0] - 1
        self.relevance_vectors_ = X[rel]
        self.relevance_idx_ = rel
        return self

    def _phi_rel(self, X: np.ndarray) -> np.ndarray:
        K = rbf_kernel(X, self.relevance_vectors_, self.gamma)
        return np.hstack([np.ones((len(X), 1)), K]) if self.has_bias_ else K

    def predict(self, X: np.ndarray, return_std: bool = False):
        Phi = self._phi_rel(X)
        mean = Phi @ self.m_
        if not return_std:
            return mean
        var = 1 / self.beta_ + np.sum(Phi @ self.Sigma_ * Phi, axis=1)
        return mean, np.sqrt(var)


def svm_decision_by_hand(svc, X: np.ndarray) -> np.ndarray:
    """Binary SVC decision y(x) = sum_n a_n t_n k(x, x_n) + b, using ONLY the support vectors (Bishop 7.13)."""
    K = rbf_kernel(X, svc.support_vectors_, svc._gamma)
    return K @ svc.dual_coef_[0] + svc.intercept_[0]


LESION_INPUTS = ("R mean", "G mean", "B mean", "R std", "G std", "B std", "GLCM contrast", "GLCM dissimilarity",
                 "GLCM homogeneity", "GLCM energy", "GLCM correlation", "GLCM ASM")


def part_kernels(data: dict, reports_dir: Path) -> dict:
    import pandas as pd
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    from src import bayes_logreg
    from src.evaluate import save_table, text_table

    heading("UNIT 5 - PART 3: KERNEL METHODS & SPARSE KERNEL MACHINES  (SVM dual, Gaussian process, RVM)")
    X, Xv = standardise(data)
    names = data["class_names"]
    yb, ybv = bayes_logreg.binary_target(data["y_train"], names), bayes_logreg.binary_target(data["y_valid"], names)

    # 1. SVM: the dual solution is sparse - only the support vectors matter
    idx = stratified_idx(data["y_train"], 300)
    Xs, ys = X[idx], yb[idx]
    rows = []
    for C in [0.01, 0.1, 1, 10, 100, 1000]:
        svc = SVC(kernel="rbf", C=C, gamma="scale").fit(Xs, ys)
        rows.append({"C": C, "Support vectors": int(svc.n_support_.sum()),
                     "Support vectors (% of leaves)": 100 * svc.n_support_.sum() / len(ys),
                     "Valid accuracy (%)": 100 * float(np.mean(svc.predict(Xv) == ybv))})
    sv_table = pd.DataFrame(rows)
    svc = SVC(kernel="rbf", C=10, gamma="scale").fit(Xs, ys)
    diff = float(np.max(np.abs(svm_decision_by_hand(svc, Xv) - svc.decision_function(Xv))))
    say(f"[1] SVM (RBF kernel), healthy vs diseased, {len(ys)} training leaves:")
    say(text_table(sv_table))
    say(f"    Decision function rebuilt by hand from the dual, y(x) = sum_n a_n t_n k(x, x_n) + b, over only the "
        f"{int(svc.n_support_.sum())} support vectors:")
    say(f"    max |ours - sklearn| = {diff:.1e}   -> the other {len(ys) - int(svc.n_support_.sum())} leaves "
        f"can be thrown away.")

    # 2 + 3. regression of disease severity (lesion % of the leaf) from colour and texture only
    fi = [data["feature_names"].index(f) for f in LESION_INPUTS]
    li = data["feature_names"].index("Lesion fraction")
    sx = StandardScaler().fit(data["X_train"][:, fi])
    tr = stratified_idx(data["y_train"], 120)
    te = stratified_idx(data["y_valid"], 80, seed=config.SEED + 1)
    Xr, Xrv = sx.transform(data["X_train"][tr][:, fi]), sx.transform(data["X_valid"][te][:, fi])
    tr_t = np.sqrt(100 * data["X_train"][tr, li])                    # sqrt(lesion %): less skewed target
    te_t = np.sqrt(100 * data["X_valid"][te, li])
    t_mean = tr_t.mean()
    gamma = 1.0 / Xr.shape[1]
    say(f"\n[2] Disease severity regression: sqrt(lesion % of leaf) from {len(fi)} colour + texture features")
    say(f"    (no brown-pixel mask used), {len(tr)} training leaves, {len(te)} held-out leaves.")

    t = time.time()
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.full(len(fi), 3.0)) + WhiteKernel(0.5)
    with warnings.catch_warnings():                                # ARD length-scales hitting their upper
        warnings.simplefilter("ignore")                            # bound is expected: see 'switched off' below
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=config.SEED).fit(Xr, tr_t)
    scales = gp.kernel_.k1.k2.length_scale
    off = [f for f, l_ in zip(LESION_INPUTS, scales) if l_ > 100]
    gp_mean, gp_std = gp.predict(Xrv, return_std=True)
    t_gp = time.time() - t
    inside = 100 * float(np.mean(np.abs(te_t - gp_mean) <= 1.96 * gp_std))
    t = time.time()
    rvm = RVMRegression(gamma=gamma).fit(Xr, tr_t - t_mean)
    rvm_mean, rvm_std = rvm.predict(Xrv, return_std=True)
    rvm_mean = rvm_mean + t_mean
    t_rvm = time.time() - t
    svr_C = float(np.std(tr_t) * 3)
    svr = SVR(kernel="rbf", gamma=gamma, C=svr_C, epsilon=0.1 * float(np.std(tr_t))).fit(Xr, tr_t)
    rmse = lambda p: float(np.sqrt(np.mean((te_t - p) ** 2)))
    reg = pd.DataFrame([
        {"Model": "Gaussian process (ARD RBF + noise, hyper-parameters by marginal likelihood)",
         "Basis vectors kept": len(tr), "R2": r2_score(te_t, gp_mean), "RMSE": rmse(gp_mean),
         "95% interval coverage (%)": inside, "Fit time (s)": t_gp},
        {"Model": "Relevance Vector Machine (from scratch)", "Basis vectors kept": len(rvm.relevance_idx_),
         "R2": r2_score(te_t, rvm_mean), "RMSE": rmse(rvm_mean),
         "95% interval coverage (%)": 100 * float(np.mean(np.abs(te_t - rvm_mean) <= 1.96 * rvm_std)),
         "Fit time (s)": t_rvm},
        {"Model": "Support Vector Regression (epsilon-insensitive)", "Basis vectors kept": int(len(svr.support_)),
         "R2": r2_score(te_t, svr.predict(Xrv)), "RMSE": rmse(svr.predict(Xrv)),
         "95% interval coverage (%)": float("nan"), "Fit time (s)": float("nan")},
    ])
    say(text_table(reg))
    say(f"    GP: {inside:.0f}% of held-out leaves fall inside its 95% interval - its error bars are honest.")
    say("    GP automatic relevance determination (one length-scale per input) switched off: "
        + (", ".join(off) if off else "none"))
    say("    (a huge length-scale means the function hardly changes along that input: it is ignored).")
    say(f"    RVM keeps {len(rvm.relevance_idx_)} relevance vectors, SVR keeps {len(svr.support_)} support "
        f"vectors: the RVM is much sparser (and also gives error bars).")

    save_table(sv_table, "unit5_svm_sparsity", reports_dir, "Unit 5 - SVM support vectors vs C")
    save_table(reg, "unit5_kernel_regression", reports_dir, "Unit 5 - GP vs RVM vs SVR for lesion severity")
    graphs = kernel_graphs(sv_table, te_t, gp_mean, gp_std, rvm_mean, rvm_std, rvm, svr, Xr, tr_t, reports_dir)
    return {"sv_table": sv_table, "dual_diff": diff, "reg": reg, "rvm": rvm, "ard_off": off, "graphs": graphs}


def kernel_graphs(sv_table, te_t, gp_mean, gp_std, rvm_mean, rvm_std, rvm, svr, Xr, tr_t, reports_dir):
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS
    out = []

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(sv_table["C"], sv_table["Support vectors (% of leaves)"], "o-", color=C["blue"],
            label="support vectors (% of training leaves)")
    ax.set_xscale("log")
    ax.set(xlabel="C (penalty for margin violations, log scale)", ylabel="support vectors (%)",
           title="SVM is a sparse kernel machine: only the support vectors are kept")
    ax2 = ax.twinx()
    ax2.plot(sv_table["C"], sv_table["Valid accuracy (%)"], "s--", color=C["orange"], label="held-out accuracy (%)")
    ax2.set_ylabel("held-out accuracy (%)", color=C["orange"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    out.append(save_fig(fig, "kernels", "1_svm_support_vectors_vs_C", reports_dir))

    # GP: 1-D picture on the most useful single input, then the multi-input predictions with error bars
    j = int(np.argmax([abs(np.corrcoef(Xr[:, i], tr_t)[0, 1]) for i in range(Xr.shape[1])]))
    x1 = Xr[:, [j]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp1 = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.5), normalize_y=True,
                                       random_state=config.SEED).fit(x1, tr_t)
    gx = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 300)[:, None]
    m1, s1 = gp1.predict(gx, return_std=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].fill_between(gx[:, 0], m1 - 1.96 * s1, m1 + 1.96 * s1, color=C["blue"], alpha=0.18,
                       label="95% predictive interval")
    ax[0].plot(gx[:, 0], m1, color=C["blue"], label="GP predictive mean")
    ax[0].scatter(x1[:, 0], tr_t, s=9, color=C["ink"], alpha=0.5, label="training leaves")
    ax[0].set(xlabel=f"{LESION_INPUTS[j]} (standardised)", ylabel="sqrt(lesion % of leaf)",
              title=f"Gaussian process on one input: a distribution over functions")
    ax[0].legend(fontsize=8)
    order = np.argsort(gp_mean)
    xs = np.arange(len(order))
    ax[1].errorbar(xs, gp_mean[order], yerr=1.96 * gp_std[order], fmt="none", ecolor=C["blue"], alpha=0.35,
                   label="GP 95% interval")
    ax[1].plot(xs, gp_mean[order], color=C["blue"], lw=1.5, label="GP prediction")
    ax[1].scatter(xs, te_t[order], s=10, color=C["orange"], zorder=3, label="true value (held-out leaf)")
    ax[1].set(xlabel="held-out leaves, sorted by prediction", ylabel="sqrt(lesion % of leaf)",
              title="GP with all 12 inputs: every prediction comes with an error bar")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out.append(save_fig(fig, "kernels", "2_gaussian_process_regression", reports_dir))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    lim = [min(te_t.min(), rvm_mean.min()) - 0.3, max(te_t.max(), rvm_mean.max()) + 0.3]
    ax[0].errorbar(te_t, rvm_mean, yerr=1.96 * rvm_std, fmt="o", ms=3, color=C["aqua"], ecolor=C["aqua"],
                   alpha=0.5, label="RVM prediction +- 95%")
    ax[0].plot(lim, lim, "--", color=C["muted"], lw=1, label="perfect prediction")
    ax[0].set(xlim=lim, ylim=lim, xlabel="true sqrt(lesion %)", ylabel="predicted",
              title="Relevance Vector Machine (from scratch): predictions")
    ax[0].legend(fontsize=8)
    counts = [len(tr_t), len(svr.support_), len(rvm.relevance_idx_)]
    labels = ["all training leaves\n(Gaussian process)", "support vectors\n(SVR)", "relevance vectors\n(RVM)"]
    ax[1].bar(labels, counts, color=[C["blue"], C["orange"], C["aqua"]], width=0.55)
    for i, v in enumerate(counts):
        ax[1].text(i, v + max(counts) * 0.01, str(v), ha="center", fontsize=10)
    ax[1].set(ylabel="leaves kept in the model", title="Sparsity: how many training leaves each model needs")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    out.append(save_fig(fig, "kernels", "3_rvm_vs_svr_sparsity", reports_dir))
    return out


# =============================================================================================================
# Part 4 - Sampling methods (Bishop chapter 11) and graphical models (Bishop chapter 8)
# =============================================================================================================
def log_posterior_logreg(w: np.ndarray, Phi: np.ndarray, t: np.ndarray, alpha: float) -> float:
    """ln p(w | D) + const for logistic regression; Gaussian prior on the weights, flat prior on the bias w[0]."""
    a = Phi @ w
    return float(np.sum(t * a - np.logaddexp(0, a)) - 0.5 * alpha * np.sum(w[1:] ** 2))


def metropolis_hastings(log_p, w0: np.ndarray, n_samples: int, step: float, seed: int = config.SEED):
    """Random-walk Metropolis (Bishop 11.2.2).

    propose   w* = w + step * N(0, I)          (symmetric proposal, so Hastings' correction cancels)
    accept    with probability  min(1, p(w*) / p(w))      - only RATIOS of p are needed, never its normaliser
    """
    rng = np.random.default_rng(seed)
    w, lp = w0.copy(), log_p(w0)
    out = np.empty((n_samples, len(w0)))
    accepted = 0
    for i in range(n_samples):
        cand = w + step * rng.standard_normal(len(w))
        lc = log_p(cand)
        if np.log(rng.random()) < lc - lp:
            w, lp = cand, lc
            accepted += 1
        out[i] = w
    return out, accepted / n_samples


def effective_sample_size(x: np.ndarray, max_lag: int = 500) -> float:
    """N / (1 + 2 sum of autocorrelations): how many independent samples the correlated chain is worth."""
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    s = 0.0
    for lag in range(1, min(max_lag, len(x) - 1)):
        r = np.dot(x[:-lag], x[lag:]) / (len(x) * var)
        if r < 0.05:
            break
        s += r
    return len(x) / (1 + 2 * s)


def ising_gibbs(y: np.ndarray, region: np.ndarray, beta: float, eta: float, sweeps: int = 30, burn_in: int = 10,
                seed: int = config.SEED):
    """Gibbs sampling in an Ising Markov random field (Bishop 8.3.3, 11.3).

    hidden x_i in {-1,+1} (+1 = lesion) on the pixel grid, observed noisy labels y_i in {-1,+1}
    energy   E(x, y) = -beta sum_{i~j} x_i x_j  -  eta sum_i x_i y_i     (neighbours like to agree)
    Gibbs    resample one pixel from p(x_i | all others) = sigma(2 (beta sum_{j~i} x_j + eta y_i)).
             Pixels of one colour of a checkerboard do not touch, so half the grid is updated at once.
    returns  the posterior P(x_i = +1 | y) estimated from the samples after burn-in, and the energy per sweep
    """
    rng = np.random.default_rng(seed)
    x = np.where(region, y, -1).astype(float)
    ii, jj = np.indices(y.shape)
    boards = [((ii + jj) % 2 == c) & region for c in (0, 1)]
    count = np.zeros(y.shape)
    energy = []
    for s in range(sweeps):
        for board in boards:
            nb = np.zeros_like(x)
            nb[1:] += x[:-1]; nb[:-1] += x[1:]; nb[:, 1:] += x[:, :-1]; nb[:, :-1] += x[:, 1:]
            p_plus = expit(2 * (beta * nb + eta * y))
            x[board] = np.where(rng.random(board.sum()) < p_plus[board], 1.0, -1.0)
        energy.append(float(-beta * (np.sum(x[1:] * x[:-1]) + np.sum(x[:, 1:] * x[:, :-1])) - eta * np.sum(x * y)))
        if s >= burn_in:
            count += x > 0
    return count / max(1, sweeps - burn_in), np.array(energy)


class DiscreteBayesNet:
    """A small Bayesian network learned from the leaves (Bishop 8.1):

        Disease D  ->  Lesion level L       (low / medium / high share of brown pixels)
        Disease D  ->  Yellowing Y          (low / medium / high share of yellow-hue pixels)
        Disease D  ->  Texture T  <-  L     (low / medium / high GLCM contrast; spots make the leaf rough)

    joint     p(D, L, Y, T) = p(D) p(L | D) p(Y | D) p(T | D, L)
    learning  every table by counting the training leaves (+1 Laplace smoothing)
    """

    def __init__(self, n_d: int, n_bins: int = 3):
        self.n_d, self.n = n_d, n_bins

    def fit(self, D, L, Y, T) -> "DiscreteBayesNet":
        n_d, n = self.n_d, self.n
        cnt = lambda *a: np.ones(a)                                   # +1 smoothing
        pD, pL, pY, pT = cnt(n_d), cnt(n_d, n), cnt(n_d, n), cnt(n_d, n, n)
        np.add.at(pD, D, 1); np.add.at(pL, (D, L), 1); np.add.at(pY, (D, Y), 1); np.add.at(pT, (D, L, T), 1)
        self.pD = pD / pD.sum()
        self.pL = pL / pL.sum(1, keepdims=True)
        self.pY = pY / pY.sum(1, keepdims=True)
        self.pT = pT / pT.sum(2, keepdims=True)
        return self

    def joint(self) -> np.ndarray:
        """The full table p(D, L, Y, T) - small enough here to enumerate exactly."""
        return (self.pD[:, None, None, None] * self.pL[:, :, None, None] * self.pY[:, None, :, None]
                * self.pT[:, :, None, :])

    def exact_posterior(self, evidence: dict) -> np.ndarray:
        """p(D | evidence) by enumeration: sum the joint over every unobserved variable, then normalise."""
        J = self.joint()
        sl = [slice(None)] * 4
        for v, val in evidence.items():
            sl["DLYT".index(v)] = slice(val, val + 1)
        p = J[tuple(sl)].sum(axis=(1, 2, 3))
        return p / p.sum()

    def ancestral_sample(self, n: int, rng) -> np.ndarray:
        """Sample parents before children: D, then L and Y given D, then T given D and L."""
        D = rng.choice(self.n_d, n, p=self.pD)
        L = _sample_rows(self.pL[D], rng)
        Y = _sample_rows(self.pY[D], rng)
        T = _sample_rows(self.pT[D, L], rng)
        return np.column_stack([D, L, Y, T])

    def rejection(self, evidence: dict, n: int, rng) -> np.ndarray:
        S = self.ancestral_sample(n, rng)
        ok = np.all([S[:, "DLYT".index(v)] == val for v, val in evidence.items()], axis=0)
        return np.bincount(S[ok, 0], minlength=self.n_d) / max(1, ok.sum())

    def likelihood_weighting(self, t_value: int, n: int, rng) -> np.ndarray:
        """Evidence T = t_value: sample the rest, weight each sample by p(T = t_value | D, L)."""
        D = rng.choice(self.n_d, n, p=self.pD)
        L = _sample_rows(self.pL[D], rng)
        w = self.pT[D, L, t_value]
        return np.bincount(D, weights=w, minlength=self.n_d) / w.sum()

    def gibbs(self, t_value: int, n: int, rng, burn_in: int = 200) -> np.ndarray:
        """Evidence T = t_value. Alternately resample D | L, T and L | D, T from their exact conditionals."""
        d, l = 0, 0
        counts = np.zeros(self.n_d)
        for i in range(n + burn_in):
            pd_ = self.pD * self.pL[:, l] * self.pT[:, l, t_value]
            d = rng.choice(self.n_d, p=pd_ / pd_.sum())
            pl = self.pL[d] * self.pT[d, :, t_value]
            l = rng.choice(self.n, p=pl / pl.sum())
            if i >= burn_in:
                counts[d] += 1
        return counts / counts.sum()

    def predict(self, L, Y, T) -> np.ndarray:
        log_p = (np.log(self.pD)[None, :] + np.log(self.pL[:, L]).T + np.log(self.pY[:, Y]).T
                 + np.log(self.pT[:, L, T]).T)
        return log_p.argmax(axis=1)


def _sample_rows(P: np.ndarray, rng) -> np.ndarray:
    """One categorical sample from every row of P."""
    return (rng.random(len(P))[:, None] > np.cumsum(P, axis=1)).sum(axis=1).clip(0, P.shape[1] - 1)


def bayes_net_variables(data: dict, split: str, edges: dict | None = None):
    """Lesion level, yellowing and texture of every leaf, cut into 3 levels at the training tertiles."""
    fn = data["feature_names"]
    X = data[f"X_{split}"]
    raw = {"L": X[:, fn.index("Lesion fraction")],
           "Y": X[:, [fn.index(f"HSV h1 s{s} v{v}") for s in (0, 1) for v in (0, 1)]].sum(axis=1),
           "T": X[:, fn.index("GLCM contrast")]}
    if edges is None:
        edges = {k: np.quantile(v, [1 / 3, 2 / 3]) for k, v in raw.items()}
    return {k: np.searchsorted(edges[k], v) for k, v in raw.items()}, edges


def part_sampling(data: dict, reports_dir: Path) -> dict:
    import pandas as pd

    from src import bayes_logreg
    from src import features as F
    from src.evaluate import save_table, text_table

    heading("UNIT 5 - PART 4: SAMPLING METHODS & GRAPHICAL MODELS  (MCMC, Gibbs, MRF, Bayesian network)")
    X, Xv = standardise(data)
    names = data["class_names"]
    yb, ybv = bayes_logreg.binary_target(data["y_train"], names), bayes_logreg.binary_target(data["y_valid"], names)
    fn = data["feature_names"]

    # 1. Metropolis-Hastings vs the Laplace approximation (Bayesian logistic regression, 2 features + bias)
    feats = ["Lesion fraction", "GLCM contrast"]
    cols = [fn.index(f) for f in feats]
    idx = stratified_idx(yb, 40)                                   # few leaves -> a wide, visible posterior
    X2, t2 = X[idx][:, cols], yb[idx]
    Phi = np.hstack([np.ones((len(X2), 1)), X2])
    alpha = 1.0
    lap = bayes_logreg.BayesianLogisticRegression(alpha=alpha).fit(X2, t2)
    order = [len(lap.w_map_) - 1] + list(range(len(lap.w_map_) - 1))   # our MCMC puts the bias first
    w_map, S = lap.w_map_[order], lap.cov_[np.ix_(order, order)]
    log_p = lambda w: log_posterior_logreg(w, Phi, t2, alpha)
    t = time.time()
    samples, acc_rate = metropolis_hastings(log_p, w_map, n_samples=30000, step=0.45)
    t_mh = time.time() - t
    burn = 3000
    post = samples[burn:]
    ess = [effective_sample_size(post[:, i]) for i in range(post.shape[1])]
    Pv = np.hstack([np.ones((len(Xv), 1)), Xv[:, cols]])
    p_mcmc = expit(Pv @ post[::10].T).mean(axis=1)                  # average the sigmoid over posterior samples
    p_lap = lap.predict_proba(Xv[:, cols])[:, 1]
    ll = lambda p: float(-np.mean(ybv * np.log(np.clip(p, 1e-12, 1)) + (1 - ybv) * np.log(np.clip(1 - p, 1e-12, 1))))
    mh = pd.DataFrame({"Parameter": ["bias"] + [f"w ({f})" for f in feats],
                       "MCMC mean": post.mean(0), "MCMC std": post.std(0),
                       "Laplace mean (w_MAP)": w_map, "Laplace std": np.sqrt(np.diag(S)),
                       "Effective samples": np.round(ess).astype(int)})
    say(f"[1] Metropolis-Hastings MCMC: Bayesian logistic regression (healthy vs diseased) on {len(t2)} leaves,")
    say(f"    features {feats}.  {len(samples):,} samples in {t_mh:.1f} s, acceptance rate {100 * acc_rate:.0f}%, "
        f"first {burn:,} discarded (burn-in).")
    say(text_table(mh))
    say(f"    Held-out log-loss:  MCMC predictive {ll(p_mcmc):.4f}   Laplace + probit {ll(p_lap):.4f}")
    say("    MCMC needs no Gaussian assumption: its samples come from the true posterior. Where the two agree,")
    say("    the Laplace approximation used in src/bayes_logreg.py is justified.")

    # 2. Gibbs sampling in an Ising MRF: clean a noisy lesion mask of a real leaf
    sample = config.SAMPLES_DIR / "tomato_early_blight.jpg"
    img = F.preprocess(F.read_image(sample))
    seg = F.segment_leaf(img)
    region = seg.mask > 0
    truth = np.where(seg.brown > 0, 1.0, -1.0)
    rng = np.random.default_rng(config.SEED)
    flip_p = 0.15
    flips = (rng.random(truth.shape) < flip_p) & region
    noisy = np.where(flips, -truth, truth)
    noisy[~region] = -1
    eta = 0.5 * np.log((1 - flip_p) / flip_p)                     # exact log-odds of the noise model
    beta = 1.0
    post_lesion, energy = ising_gibbs(noisy, region, beta=beta, eta=eta, sweeps=40, burn_in=10)
    clean = np.where(post_lesion > 0.5, 1.0, -1.0)
    clean[~region] = -1
    err_before = 100 * float(np.mean(noisy[region] != truth[region]))
    err_after = 100 * float(np.mean(clean[region] != truth[region]))
    lesion_true = 100 * float(np.mean(truth[region] > 0))
    say(f"\n[2] Gibbs sampling in a Markov random field (Ising model) on {sample.name}:")
    say(f"    {flip_p:.0%} of the leaf's pixel labels were flipped at random (noise). The MRF prior says")
    say(f"    'neighbouring pixels usually agree' (beta = {beta}), the data term trusts each label (eta = {eta:.2f}).")
    say(f"    pixels wrong before: {err_before:.1f}%   after 40 Gibbs sweeps: {err_after:.2f}%")
    say(f"    true lesion area {lesion_true:.1f}%  noisy {100 * np.mean(noisy[region] > 0):.1f}%  "
        f"cleaned {100 * np.mean(clean[region] > 0):.1f}%")

    # 3. Bayesian network: exact inference vs sampling
    v_tr, edges = bayes_net_variables(data, "train")
    v_va, _ = bayes_net_variables(data, "valid", edges)
    bn = DiscreteBayesNet(len(names)).fit(data["y_train"], v_tr["L"], v_tr["Y"], v_tr["T"])
    bn_acc = 100 * float(np.mean(bn.predict(v_va["L"], v_va["Y"], v_va["T"]) == data["y_valid"]))
    healthy = next(i for i, c in enumerate(names) if config.is_healthy(c))
    high = 2
    exact = bn.exact_posterior({"T": high})
    rng = np.random.default_rng(config.SEED)
    ns = [100, 300, 1000, 3000, 10000, 30000]
    conv = []
    for n in ns:
        for method, f in [("Rejection sampling", lambda n: bn.rejection({"T": high}, n, rng)),
                          ("Likelihood weighting", lambda n: bn.likelihood_weighting(high, n, rng)),
                          ("Gibbs sampling", lambda n: bn.gibbs(high, n, rng))]:
            errs = [np.abs(f(n) - exact).max() for _ in range(5)]
            conv.append({"Samples": n, "Method": method, "Max error": float(np.mean(errs))})
    conv = pd.DataFrame(conv)
    est = {"Exact (enumeration)": exact, "Rejection sampling": bn.rejection({"T": high}, 30000, rng),
           "Likelihood weighting": bn.likelihood_weighting(high, 30000, rng),
           "Gibbs sampling": bn.gibbs(high, 30000, rng)}
    post_tab = pd.DataFrame({"Disease": [short(c) for c in names], **{k: 100 * v for k, v in est.items()}})
    say("\n[3] Bayesian network  D -> L, D -> Y, D -> T, L -> T  (3 levels each), tables learned by counting.")
    say(f"    Prior: P(healthy) = {100 * bn.pD[healthy]:.1f}%.  Query: P(Disease | texture contrast = high) (%):")
    say(text_table(post_tab))
    say(f"    Rough texture raises P(healthy) to {100 * exact[healthy]:.1f}% - healthy tomato leaves have strong "
        f"vein texture.")
    q = {"P(healthy | lesion low)": bn.exact_posterior({"L": 0})[healthy],
         "P(healthy | lesion low, texture high)": bn.exact_posterior({"L": 0, "T": high})[healthy],
         "P(healthy | lesion high)": bn.exact_posterior({"L": 2})[healthy]}
    for k_, v in q.items():
        say(f"    {k_:<40} = {100 * v:5.1f}%")
    say(f"    As a classifier with only these 3 coarse variables: {bn_acc:.1f}% held-out accuracy "
        f"(chance = {100 / len(names):.0f}%).")

    save_table(mh, "unit5_mcmc_vs_laplace", reports_dir, "Unit 5 - Metropolis-Hastings vs Laplace approximation")
    save_table(post_tab, "unit5_bayes_net_posterior", reports_dir,
               "Unit 5 - Bayesian network: P(Disease | texture high), exact vs sampling (%)")
    save_table(conv, "unit5_bayes_net_convergence", reports_dir, "Unit 5 - sampling error vs number of samples")
    graphs = sampling_graphs(samples, burn, w_map, S, feats, img, noisy, post_lesion, clean, truth, region, energy,
                             err_before, err_after, bn, names, est, conv, reports_dir)
    return {"mh": mh, "acceptance": acc_rate, "ising": (err_before, err_after), "bn_acc": bn_acc, "bn": bn,
            "posterior": post_tab, "graphs": graphs}


def sampling_graphs(samples, burn, w_map, S, feats, img, noisy, post_lesion, clean, truth, region, energy,
                    err_before, err_after, bn, names, est, conv, reports_dir):
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS
    out = []

    labels = ["bias", f"w ({feats[0]})", f"w ({feats[1]})"]
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
    for i in range(3):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(samples[:, i], lw=0.4, color=C["blue"])
        ax.axvspan(0, burn, color=C["grid"], alpha=0.8)
        ax.axhline(w_map[i], color=C["orange"], lw=1, ls="--")
        ax.set_ylabel(labels[i], fontsize=8)
        if i == 0:
            ax.set_title("Metropolis-Hastings trace (grey = burn-in, dashed = Laplace / MAP)")
        if i == 2:
            ax.set_xlabel("MCMC step")
    ax = fig.add_subplot(gs[:, 1])
    post = samples[burn::5]
    ax.scatter(post[:, 1], post[:, 2], s=3, alpha=0.25, color=C["blue"], label="MCMC samples (true posterior)")
    vals, vecs = np.linalg.eigh(S[1:, 1:])
    for k_sd in (1, 2):
        ax.add_patch(Ellipse(w_map[1:], *(2 * k_sd * np.sqrt(vals[::-1])),
                             angle=np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1])), fill=False,
                             ec=C["orange"], lw=1.8, ls="-" if k_sd == 1 else "--"))
    ax.plot([], [], color=C["orange"], label="Laplace Gaussian (1 and 2 sd)")
    ax.scatter(*w_map[1:], marker="x", s=60, color=C["ink"], label="w_MAP", zorder=4)
    ax.set(xlabel=labels[1], ylabel=labels[2], title="Posterior p(w | D): samples vs Laplace")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out.append(save_fig(fig, "sampling", "1_mcmc_vs_laplace", reports_dir))

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    show = lambda m: np.where(region, (m > 0).astype(float), np.nan)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    panels = [(rgb, "leaf photo", None), (show(truth), "lesion mask (segmentation)", "cividis"),
              (show(noisy), f"+15% random flips\n{err_before:.1f}% pixels wrong", "cividis"),
              (np.where(region, post_lesion, np.nan), "Gibbs: P(lesion | noisy labels)", "magma"),
              (show(clean), f"MRF result\n{err_after:.2f}% pixels wrong", "cividis")]
    for ax, (im, title, cmap) in zip(axes, panels):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1) if cmap else ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Gibbs sampling in an Ising Markov random field cleans a noisy lesion mask "
                 "(yellow = lesion pixel)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out.append(save_fig(fig, "sampling", "2_gibbs_mrf_lesion_denoising", reports_dir))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})
    a = ax[0]
    a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")
    nodes = {"D": (5, 8.3, "Disease\nD"), "L": (1.8, 4.6, "Lesion\nlevel L"), "Y": (8.2, 4.6, "Yellowing\nY"),
             "T": (3.6, 1.2, "Texture\nT")}
    for k_, (x, yy, txt) in nodes.items():
        a.add_patch(FancyBboxPatch((x - 1.3, yy - 0.8), 2.6, 1.6, boxstyle="round,pad=0.1",
                                   fc="#eef4fb" if k_ != "T" else "#fdeee6", ec=C["ink"]))
        a.text(x, yy, txt, ha="center", va="center", fontsize=9)
    for s_, d_ in [("D", "L"), ("D", "Y"), ("D", "T"), ("L", "T")]:
        (x1, y1, _), (x2, y2, _) = nodes[s_], nodes[d_]
        a.add_patch(FancyArrowPatch((x1, y1 - 0.9), (x2, y2 + 0.9), arrowstyle="-|>", mutation_scale=14,
                                    color=C["ink"], lw=1.2))
    a.set_title("Bayesian network (orange = observed: T = high)")
    a.text(5, -0.5, "p(D,L,Y,T) = p(D) p(L|D) p(Y|D) p(T|D,L)", ha="center", fontsize=8.5)
    width = 0.2
    xs = np.arange(len(names))
    cols = [C["ink"], C["muted"], C["blue"], C["orange"]]
    for i, (k_, v) in enumerate(est.items()):
        ax[1].bar(xs + (i - 1.5) * width, 100 * v, width, color=cols[i], label=k_)
    ax[1].bar(xs, 100 * bn.pD, 0.85, fill=False, ec=C["aqua"], lw=1.4, ls="--", label="prior P(D)")
    ax[1].set_xticks(xs, [short(c).replace(" ", "\n", 1) for c in names], fontsize=8)
    ax[1].set(ylabel="probability (%)", title="P(Disease | texture = high): exact vs 3 samplers")
    ax[1].legend(fontsize=7)
    ax[1].grid(axis="x", visible=False)
    for method, col in zip(["Rejection sampling", "Likelihood weighting", "Gibbs sampling"], cols[1:]):
        c = conv[conv["Method"] == method]
        ax[2].plot(c["Samples"], 100 * c["Max error"], "o-", color=col, label=method)
    n = np.array(sorted(conv["Samples"].unique()))
    ref = 100 * conv[conv["Samples"] == n[0]]["Max error"].mean() * np.sqrt(n[0] / n)
    ax[2].plot(n, ref, ":", color=C["ink"], lw=1, label="1 / sqrt(N) rate")
    ax[2].set(xscale="log", yscale="log", xlabel="number of samples N",
              ylabel="largest error vs exact (percentage points)", title="Sampling error shrinks like 1/sqrt(N)")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    out.append(save_fig(fig, "sampling", "3_bayes_net_inference", reports_dir))
    return out


# =============================================================================================================
def run(data: dict, parts=PARTS, reports_dir: Path = config.REPORTS_DIR) -> dict:
    """Run the chosen Unit 5 parts; every part prints its results and saves its tables and graphs."""
    import pandas as pd

    from src.evaluate import save_table
    config.set_seeds()
    reports_dir = Path(reports_dir)
    fns = {"nn": part_nn, "combining": part_combining, "kernels": part_kernels, "sampling": part_sampling}
    results = {}
    for p in parts:
        t = time.time()
        results[p] = fns[p](data, reports_dir)
        say(f"\n  ({p} finished in {time.time() - t:.1f} s; graphs:)")
        for g in results[p]["graphs"]:
            try:
                shown = g.relative_to(config.PROJECT_ROOT)
            except ValueError:
                shown = g
            say(f"    {shown}")
    if set(parts) == set(PARTS):
        save_table(pd.DataFrame(SYLLABUS), "unit5_syllabus_mapping", reports_dir,
                   "Unit 5 - where every topic is in the project")
    return results


SYLLABUS = [
    {"Topic": "Feed-forward network / multilayer perceptron", "Where": "src/unit5.py MLP",
     "Graph": "15_neural_networks/1, 3"},
    {"Topic": "Error backpropagation + gradient check", "Where": "MLP.gradients, gradient_check",
     "Graph": "15_neural_networks/1"},
    {"Topic": "Network complexity / regularisation (weight decay)", "Where": "part_nn hidden-unit sweep",
     "Graph": "15_neural_networks/2"},
    {"Topic": "Committees (voting), stacking", "Where": "part_combining", "Graph": "16_combining_models/2"},
    {"Topic": "Boosting - AdaBoost from scratch", "Where": "AdaBoostSAMME", "Graph": "16_combining_models/1"},
    {"Topic": "Tree-based models, bagging, random forest, gradient boosting", "Where": "src/classify.py, src/figures.py",
     "Graph": "05_decision_trees, 06_bagging_random_forest, 07_boosting"},
    {"Topic": "Kernel trick, dual representation, support vectors", "Where": "svm_decision_by_hand",
     "Graph": "17_kernel_methods/1"},
    {"Topic": "Gaussian processes (regression)", "Where": "part_kernels", "Graph": "17_kernel_methods/2"},
    {"Topic": "Relevance vector machine (sparse Bayesian kernel model)", "Where": "RVMRegression",
     "Graph": "17_kernel_methods/3"},
    {"Topic": "Markov chain Monte Carlo - Metropolis-Hastings", "Where": "metropolis_hastings",
     "Graph": "18_sampling_graphical_models/1"},
    {"Topic": "Markov random field + Gibbs sampling", "Where": "ising_gibbs",
     "Graph": "18_sampling_graphical_models/2"},
    {"Topic": "Bayesian network: exact inference, rejection sampling, likelihood weighting, Gibbs",
     "Where": "DiscreteBayesNet", "Graph": "18_sampling_graphical_models/3"},
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unit 5: neural networks, combining models, kernels, sampling.")
    ap.add_argument("--part", choices=PARTS, action="append", help="run only this part (can repeat)")
    ap.add_argument("--limit", type=int, default=None, help="use the --limit N feature cache")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    from src import dataset
    try:
        data = dataset.load_features(args.limit)
    except FileNotFoundError as e:
        print(e)
        return 1
    run(data, parts=args.part or PARTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
