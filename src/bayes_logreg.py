"""
src/bayes_logreg.py - Phase 3: Bayesian logistic regression from scratch (NumPy only).

Task: healthy (0) vs diseased (1) from the 55 standardised leaf features.

Model (Bishop, Pattern Recognition and Machine Learning, sections 4.3-4.5)
  likelihood   p(y | x, w) = sigma(w^T x)^y (1 - sigma(w^T x))^(1-y)
  prior        w ~ N(0, alpha^-1 I)          (the bias gets a flat prior, like sklearn's unpenalised intercept)
  MAP          minimise  E(w) = sum_n [log(1 + e^a_n) - y_n a_n] + alpha/2 ||w||^2      (a_n = w^T x_n)
               with Newton's method:  w <- w - H^-1 g
                   g = X^T (p - y) + alpha w
                   H = X^T R X + alpha I,   R = diag(p_n (1 - p_n))
  Laplace      posterior  p(w | D) ~ N(w_MAP, S_N),   S_N = H^-1 evaluated at w_MAP
  predictive   a = w^T x is Gaussian with mean mu = w_MAP^T x and variance s^2 = x^T S_N x, and
               p(y=1 | x, D) ~ sigma(kappa(s^2) mu),   kappa(s^2) = (1 + pi s^2 / 8)^(-1/2)   (probit approximation)
  evidence     ln p(D | alpha) ~ -E(w_MAP) - 1/2 ln|H| + M/2 ln alpha + const     (used to choose alpha)

Equivalence with sklearn: LogisticRegression(C) minimises C * sum(loss) + 1/2 ||w||^2, i.e. alpha = 1/C.
So the MAP weights must equal sklearn's weights with C = 1/alpha (checked in tests/test_bayes_logreg.py).

Usage (from the plant-disease-ml folder):
    python -m src.bayes_logreg               # uses cache/features.npz
    python -m src.bayes_logreg --limit 100   # uses the --limit 100 cache
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/bayes_logreg.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from scipy.stats import norm

import config


class BayesianLogisticRegression:
    """Binary Bayesian logistic regression: Gaussian prior, MAP by Newton's method, Laplace posterior."""

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True, max_iter: int = 100, tol: float = 1e-10):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    # ------------------------------------------------------------------ helpers
    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.hstack([X, np.ones((len(X), 1))]) if self.fit_intercept else X

    def _prior_precision(self, d: int) -> np.ndarray:
        """alpha on every weight; 0 on the bias (flat prior, not penalised)."""
        a = np.full(d, self.alpha, dtype=np.float64)
        if self.fit_intercept:
            a[-1] = 0.0
        return a

    @staticmethod
    def _neg_log_posterior(w, Phi, y, a_prior) -> float:
        act = Phi @ w
        return float(np.sum(np.logaddexp(0.0, act) - y * act) + 0.5 * np.sum(a_prior * w * w))

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLogisticRegression":
        Phi = self._design(X)
        y = np.asarray(y, dtype=np.float64)
        if not set(np.unique(y)) <= {0.0, 1.0}:
            raise ValueError("y must contain only 0 and 1")
        n, d = Phi.shape
        a_prior = self._prior_precision(d)
        w = np.zeros(d)
        f = self._neg_log_posterior(w, Phi, y, a_prior)
        self.converged_ = False
        for it in range(1, self.max_iter + 1):
            p = expit(Phi @ w)
            grad = Phi.T @ (p - y) + a_prior * w
            H = (Phi * (p * (1 - p))[:, None]).T @ Phi + np.diag(a_prior)
            step = cho_solve(cho_factor(H), grad)
            # Newton step with backtracking: never let the objective go up
            t = 1.0
            while True:
                w_new = w - t * step
                f_new = self._neg_log_posterior(w_new, Phi, y, a_prior)
                if f_new <= f + 1e-12 or t < 1e-10:
                    break
                t *= 0.5
            change = np.max(np.abs(w_new - w))
            w, f = w_new, f_new
            if change < self.tol * (1 + np.max(np.abs(w))):
                self.converged_ = True
                break
        self.n_iter_ = it

        # Laplace approximation: Hessian at the MAP is the posterior precision
        p = expit(Phi @ w)
        H = (Phi * (p * (1 - p))[:, None]).T @ Phi + np.diag(a_prior)
        chol = cho_factor(H)
        self.w_map_ = w
        self.hessian_ = H
        self.cov_ = cho_solve(chol, np.eye(d))
        self.cov_ = 0.5 * (self.cov_ + self.cov_.T)                 # remove round-off asymmetry
        self.neg_log_posterior_ = f
        n_penalised = int(np.count_nonzero(a_prior))
        logdet_H = 2 * np.sum(np.log(np.diag(chol[0])))
        self.log_evidence_ = -f - 0.5 * logdet_H + 0.5 * n_penalised * np.log(self.alpha)
        self.coef_ = w[:-1] if self.fit_intercept else w
        self.intercept_ = w[-1] if self.fit_intercept else 0.0
        return self

    # ------------------------------------------------------------------ prediction
    def activation_mean_var(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean mu = w_MAP^T x and variance s^2 = x^T S_N x of the activation a = w^T x."""
        Phi = self._design(X)
        mu = Phi @ self.w_map_
        var = np.einsum("ij,jk,ik->i", Phi, self.cov_, Phi)
        return mu, np.maximum(var, 0.0)

    def predict_proba_map(self, X: np.ndarray) -> np.ndarray:
        """Plug-in probability sigma(w_MAP^T x): ignores weight uncertainty (what plain LR does)."""
        p = expit(self._design(X) @ self.w_map_)
        return np.column_stack([1 - p, p])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Bayesian predictive probability with the probit approximation (moderated towards 0.5)."""
        mu, var = self.activation_mean_var(X)
        p = expit(mu / np.sqrt(1.0 + np.pi * var / 8.0))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        # kappa > 0, so predictive and MAP probabilities are on the same side of 0.5: same decisions
        return (self.decision_function(X) >= 0).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._design(X) @ self.w_map_

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Standard deviation of the activation, s = sqrt(x^T S_N x): how unsure the weights make us."""
        return np.sqrt(self.activation_mean_var(X)[1])

    def credible_intervals(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """(lower, upper) of the Laplace posterior for every coefficient (bias excluded)."""
        z = norm.ppf(0.5 + level / 2)
        sd = np.sqrt(np.diag(self.cov_))[: len(self.coef_)]
        return self.coef_ - z * sd, self.coef_ + z * sd

    def sample_weights(self, n: int, seed: int = config.SEED) -> np.ndarray:
        """Draws from the Laplace posterior N(w_MAP, S_N) (used to check the probit approximation)."""
        return np.random.default_rng(seed).multivariate_normal(self.w_map_, self.cov_, size=n)


def select_alpha(X: np.ndarray, y: np.ndarray, alphas=config.BAYES_ALPHAS) -> tuple[float, list[float]]:
    """Pick the prior precision with the highest Laplace evidence ln p(D | alpha) (no validation set needed)."""
    evidence = [BayesianLogisticRegression(alpha=a).fit(X, y).log_evidence_ for a in alphas]
    return float(alphas[int(np.argmax(evidence))]), evidence


# ----------------------------------------------------------------------------- metrics
def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Average |accuracy - confidence| over equal-width confidence bins, weighted by bin size."""
    conf = np.maximum(p, 1 - p)
    correct = (p >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
    pred = (p >= 0.5).astype(int)
    acc = 100 * float((pred == y).mean())
    return {"Accuracy (%)": acc,
            "Balanced acc (%)": 100 * balanced_accuracy_score(y, pred) if len(np.unique(y)) == 2 else acc,
            "Log-loss": log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]),
            "Brier": brier_score_loss(y, p),
            "ECE": expected_calibration_error(y, p)}


# ----------------------------------------------------------------------------- experiment
def binary_target(y: np.ndarray, class_names: list[str]) -> np.ndarray:
    """1 = diseased, 0 = healthy."""
    healthy = np.array([config.is_healthy(c) for c in class_names])
    return (~healthy[y]).astype(int)


def _stratified_subset(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n indices with both classes in their original proportion (at least 2 of each)."""
    idx = []
    for c in (0, 1):
        pool = np.flatnonzero(y == c)
        k = max(2, int(round(n * len(pool) / len(y))))
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.concatenate(idx)


def unseen_disease_test(data: dict):
    """Hide one disease from training, then predict leaves of that disease (all truly 'diseased').

    This is the out-of-distribution case where weight uncertainty matters: the leaves do not look like
    anything the model was trained on.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    names = data["class_names"]
    yb = binary_target(data["y_train"], names)
    rows = []
    for k, cls in enumerate(names):
        if config.is_healthy(cls):
            continue
        keep = data["y_train"] != k
        if len(np.unique(yb[keep])) < 2:          # hiding it would leave no diseased leaves to learn from
            continue
        Xo = data["X_valid"][data["y_valid"] == k]
        if len(Xo) == 0:
            continue
        scaler = StandardScaler().fit(data["X_train"][keep])
        X, y = scaler.transform(data["X_train"][keep]), yb[keep]
        alpha, _ = select_alpha(X, y)
        m = BayesianLogisticRegression(alpha=alpha).fit(X, y)
        Xo = scaler.transform(Xo)
        yo = np.ones(len(Xo), dtype=int)
        p_map, p_bayes = m.predict_proba_map(Xo)[:, 1], m.predict_proba(Xo)[:, 1]
        wrong = p_map < 0.5
        conf = lambda p: float(np.maximum(p, 1 - p).mean())
        rows.append({"Hidden disease": config.pretty_class(cls), "Leaves": len(Xo),
                     "Detected as diseased (%)": 100 * float((~wrong).mean()),
                     "MAP log-loss": binary_metrics(yo, p_map)["Log-loss"],
                     "Bayes log-loss": binary_metrics(yo, p_bayes)["Log-loss"],
                     "MAP confidence when wrong": conf(p_map[wrong]) if wrong.any() else np.nan,
                     "Bayes confidence when wrong": conf(p_bayes[wrong]) if wrong.any() else np.nan})
    return pd.DataFrame(rows)


def run(data: dict, reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR,
        small_sizes=(20, 50, 100, 200, 500, 1000), repeats: int = 10, small_alpha: float = 1.0) -> dict:
    """The whole of Phase 3: fit, compare with sklearn, measure uncertainty, save tables, graphs and model."""
    import joblib
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src import features as F
    from src.evaluate import save_table, text_table

    config.set_seeds()
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    names = data["class_names"]
    if not any(config.is_healthy(c) for c in names) or all(config.is_healthy(c) for c in names):
        raise ValueError("Healthy vs diseased needs at least one healthy and one diseased class.")

    scaler = StandardScaler().fit(data["X_train"])
    X, Xv = scaler.transform(data["X_train"]), scaler.transform(data["X_valid"])
    y, yv = binary_target(data["y_train"], names), binary_target(data["y_valid"], names)
    print(f"Healthy vs diseased: {len(y)} train ({int((y == 0).sum())} healthy), "
          f"{len(yv)} valid ({int((yv == 0).sum())} healthy), {X.shape[1]} standardised features.")

    # 1. choose alpha by the evidence, fit, compare with sklearn at C = 1/alpha
    alpha, evidence = select_alpha(X, y)
    blr = BayesianLogisticRegression(alpha=alpha).fit(X, y)
    sk = LogisticRegression(C=1 / alpha, tol=1e-10, max_iter=20000).fit(X, y)
    max_diff = float(np.max(np.abs(np.append(sk.coef_[0], sk.intercept_) - blr.w_map_)))
    print(f"alpha chosen by evidence: {alpha:g}  (sklearn C = {1 / alpha:g}).  Newton converged in "
          f"{blr.n_iter_} iterations.  Max |w_MAP - w_sklearn| = {max_diff:.2e}")

    # 2. metrics on the held-out valid set
    p_sk = sk.predict_proba(Xv)[:, 1]
    p_map = blr.predict_proba_map(Xv)[:, 1]
    p_bayes = blr.predict_proba(Xv)[:, 1]
    rows = [{"Model": "sklearn LogisticRegression (C = 1/alpha)", **binary_metrics(yv, p_sk)},
            {"Model": "Bayesian LR - MAP plug-in sigma(w^T x)", **binary_metrics(yv, p_map)},
            {"Model": "Bayesian LR - predictive (probit approx.)", **binary_metrics(yv, p_bayes)}]
    table = pd.DataFrame(rows)
    print("\nHeld-out valid set (all training leaves):\n" + text_table(table))

    # 3. small training sets. The evidence is unreliable when there are fewer leaves than features
    #    (it picks a very weak prior), so a fixed, sensible prior alpha = small_alpha is used here.
    rng = np.random.default_rng(config.SEED)
    small = []
    for n in small_sizes:
        if n >= len(y):
            continue
        for r in range(repeats):
            idx = _stratified_subset(y, n, rng)
            a_n = small_alpha
            m = BayesianLogisticRegression(alpha=a_n).fit(X[idx], y[idx])
            mm = binary_metrics(yv, m.predict_proba_map(Xv)[:, 1])
            mb = binary_metrics(yv, m.predict_proba(Xv)[:, 1])
            small.append({"Train size": len(idx), "Repeat": r, "alpha": a_n,
                          "MAP log-loss": mm["Log-loss"], "Bayes log-loss": mb["Log-loss"],
                          "MAP ECE": mm["ECE"], "Bayes ECE": mb["ECE"], "Accuracy (%)": mb["Accuracy (%)"]})
    small = pd.DataFrame(small)
    small_summary = (small.groupby("Train size")
                     .agg(**{"MAP log-loss": ("MAP log-loss", "mean"), "Bayes log-loss": ("Bayes log-loss", "mean"),
                             "MAP ECE": ("MAP ECE", "mean"), "Bayes ECE": ("Bayes ECE", "mean"),
                             "Accuracy (%)": ("Accuracy (%)", "mean")})
                     .reset_index())
    if len(small_summary):
        print(f"\nSmall training sets, alpha = {small_alpha:g} (mean of {repeats} random draws):\n"
              + text_table(small_summary))

    # 4. unseen disease (out-of-distribution) test
    unseen = unseen_disease_test(data)
    if len(unseen):
        print("\nUnseen disease test (one disease hidden from training, its valid leaves predicted):\n"
              + text_table(unseen))

    # 5. confidence vs mistakes on the valid set. Rank by the predictive confidence max(p, 1-p): it combines
    #    the mean activation and its uncertainty s. (s alone also grows far out on the 'obviously right' side.)
    s = blr.uncertainty(Xv)
    conf = np.maximum(p_bayes, 1 - p_bayes)
    wrong = (p_bayes >= 0.5).astype(int) != yv
    order = np.argsort(-conf)                                 # most confident first
    keep = np.arange(1, len(conf) + 1) / len(conf)
    acc_kept = np.cumsum(~wrong[order]) / np.arange(1, len(conf) + 1)
    unc = {"mean_conf_correct": float(conf[~wrong].mean()),
           "mean_conf_wrong": float(conf[wrong].mean()) if wrong.any() else None,
           "accuracy_all": float(1 - wrong.mean()),
           "accuracy_most_confident_80": float(acc_kept[int(0.8 * len(conf)) - 1])}
    if wrong.any():
        print(f"\nPredictive confidence: correct {unc['mean_conf_correct']:.3f}, wrong {unc['mean_conf_wrong']:.3f}. "
              f"Accuracy {100 * unc['accuracy_all']:.2f}% on all leaves, "
              f"{100 * unc['accuracy_most_confident_80']:.2f}% on the 80% most confident.")

    # 6. tables, model, graphs
    save_table(table, "bayes_results", reports_dir,
               f"Bayesian logistic regression, healthy vs diseased (alpha = {alpha:g} chosen by evidence)")
    if len(small_summary):
        save_table(small_summary, "bayes_small_data", reports_dir,
                   f"Small training sets (alpha = {small_alpha:g}) - mean over {repeats} random draws, valid set")
    if len(unseen):
        save_table(unseen, "bayes_unseen_disease", reports_dir,
                   "Unseen disease: one disease hidden from training, its valid leaves predicted")
    lo, hi = blr.credible_intervals()
    weights = pd.DataFrame({"Feature": F.FEATURE_NAMES, "w_MAP": blr.coef_, "CI 2.5%": lo, "CI 97.5%": hi,
                            "w_sklearn": sk.coef_[0]})
    weights = weights.reindex(weights["w_MAP"].abs().sort_values(ascending=False).index)
    save_table(weights.round(4), "bayes_weights", reports_dir, "MAP weights with 95% Laplace credible intervals")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": blr, "alpha": alpha, "feature_names": list(F.FEATURE_NAMES),
                 "feature_version": config.FEATURE_VERSION, "positive_class": "diseased",
                 "uncertainty_quantiles": np.quantile(s, [0.5, 0.9]).tolist()},
                models_dir / config.BAYES_MODEL_PATH.name, compress=3)

    make_graphs(data, X, y, Xv, yv, blr, sk, alpha, evidence, p_sk, p_bayes, conf, wrong, keep, acc_kept,
                small, unseen, reports_dir)
    print(f"\nTables and graphs written to {reports_dir}; model saved to {models_dir / config.BAYES_MODEL_PATH.name}")
    return {"alpha": alpha, "max_weight_diff": max_diff, "table": table, "small": small_summary,
            "unseen": unseen, "uncertainty": unc}


# ----------------------------------------------------------------------------- graphs
def make_graphs(data, X, y, Xv, yv, blr, sk, alpha, evidence, p_sk, p_bayes, conf, wrong, keep, acc_kept,
                small, unseen, reports_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from src import features as F
    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS

    # -- 1. evidence vs alpha
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(config.BAYES_ALPHAS, evidence, color=C["bayes"], marker="o", ms=6)
    ax.axvline(alpha, color=C["muted"], ls=":", lw=1)
    ax.annotate(f"best alpha = {alpha:g}", (alpha, max(evidence)), xytext=(8, -14), textcoords="offset points",
                fontsize=9, color=C["ink"])
    ax.set_xscale("log"); ax.set_xlabel("prior precision alpha   (larger = stronger pull of weights towards 0)")
    ax.set_ylabel("log evidence  ln p(D | alpha)")
    ax.set_title("Choosing the prior strength by the Laplace evidence")
    save_fig(fig, "bayes", "1_evidence", reports_dir)

    # -- 2. 2-feature picture: probability and uncertainty, trained on few leaves so uncertainty is visible
    fx, fy = F.FEATURE_NAMES.index("Lesion fraction"), F.FEATURE_NAMES.index("GLCM contrast")
    rng = np.random.default_rng(config.SEED)
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), size=min(20, int((y == c).sum())), replace=False)
                          for c in (0, 1)])                 # 20 healthy + 20 diseased, so both are visible
    X2, y2 = X[idx][:, [fx, fy]], y[idx]
    m2 = BayesianLogisticRegression(alpha=1.0).fit(X2, y2)
    pad = 1.0
    gx = np.linspace(X2[:, 0].min() - pad, X2[:, 0].max() + 3 * pad, 220)
    gy = np.linspace(X2[:, 1].min() - pad, X2[:, 1].max() + 3 * pad, 220)
    GX, GY = np.meshgrid(gx, gy)
    G = np.column_stack([GX.ravel(), GY.ravel()])
    P = m2.predict_proba(G)[:, 1].reshape(GX.shape)
    S = m2.uncertainty(G).reshape(GX.shape)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    seq = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#c9dcf3", C["blue"], "#173f73"])
    div = LinearSegmentedColormap.from_list("div", [C["healthy"], "#f2f2ef", C["diseased"]])
    for ax, Z, cmap, title, lab in [
            (axes[0], P, div, "Predictive P(diseased | x) - Bayesian", "P(diseased)"),
            (axes[1], S, seq, "Uncertainty s = sqrt(x^T S_N x)", "s (std of activation)")]:
        levels = np.linspace(0, 1, 21) if Z is P else 14
        im = ax.contourf(GX, GY, Z, levels=levels, cmap=cmap)
        ax.contour(GX, GY, P, levels=[0.5], colors=C["ink"], linewidths=1.2)
        for cls, col, name, mk in [(0, C["healthy"], "healthy", "o"), (1, C["diseased"], "diseased", "s")]:
            ax.scatter(X2[y2 == cls, 0], X2[y2 == cls, 1], c=col, marker=mk, s=42, edgecolors="white",
                       linewidths=1.2, label=f"{name} leaf (training)", zorder=3)
        ax.set_title(title); ax.set_xlabel("lesion fraction (standardised)"); ax.grid(False)
        fig.colorbar(im, ax=ax, label=lab, shrink=0.85)
    axes[0].set_ylabel("GLCM contrast (standardised)")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].text(0.98, 0.04, "black line = decision boundary P = 0.5\nuncertainty grows away from the data",
                 transform=axes[1].transAxes, ha="right", fontsize=8, color=C["ink"])
    fig.suptitle(f"Bayesian logistic regression on 2 features, {len(y2)} training leaves",
                 x=0.01, ha="left", fontsize=12)
    save_fig(fig, "bayes", "2_uncertainty_2d", reports_dir)

    # -- 3. small training sets: log-loss of MAP vs Bayesian predictive
    if len(small):
        g = small.groupby("Train size")
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, metric, ylab in [(axes[0], "log-loss", "log-loss on valid set (lower = better)"),
                                 (axes[1], "ECE", "calibration error ECE (lower = better)")]:
            for key, col, lab, mk in [("MAP", C["plain"], "MAP plug-in (plain LR)", "o"),
                                      ("Bayes", C["bayes"], "Bayesian predictive", "s")]:
                mean, sd = g[f"{key} {metric}"].mean(), g[f"{key} {metric}"].std()
                ax.plot(mean.index, mean.values, color=col, marker=mk, ms=6, label=lab)
                ax.fill_between(mean.index, mean - sd, mean + sd, color=col, alpha=0.12, lw=0)
            ax.set_xscale("log"); ax.set_xlabel("number of training leaves"); ax.set_ylabel(ylab)
        axes[0].legend(fontsize=9)
        axes[0].set_title("Familiar leaves: the plug-in MAP is already well calibrated")
        axes[1].set_title("Calibration: is 90% confidence right 90% of the time?")
        save_fig(fig, "bayes", "3_small_data", reports_dir)

    # -- 4. confidence vs mistakes
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    bins = np.linspace(0.5, 1.0, 26)
    axes[0].hist(conf[~wrong], bins=bins, density=True, color=C["muted"], alpha=0.6, label="correct predictions")
    if wrong.any():
        axes[0].hist(conf[wrong], bins=bins, density=True, color=C["orange"], alpha=0.75, label="wrong predictions")
    axes[0].set_xlabel("Bayesian predictive confidence max(p, 1-p)"); axes[0].set_ylabel("density")
    axes[0].legend(fontsize=9, loc="upper left"); axes[0].set_title("Mistakes come with low confidence")
    axes[1].plot(100 * keep, 100 * acc_kept, color=C["bayes"])
    axes[1].axhline(100 * (1 - wrong.mean()), color=C["muted"], ls=":", lw=1)
    axes[1].text(2, 100 * (1 - wrong.mean()), "accuracy on all leaves", va="bottom", fontsize=8, color=C["muted"])
    axes[1].set_xlabel("% of leaves kept (most confident first; the rest go to an expert)")
    axes[1].set_ylabel("accuracy on kept leaves (%)"); axes[1].set_xlim(0, 100)
    axes[1].set_title("Refer unsure leaves -> higher accuracy on the rest")
    save_fig(fig, "bayes", "4_confidence_vs_error", reports_dir)

    # -- 4b. unseen disease
    if len(unseen):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
        yy = np.arange(len(unseen))
        for ax, key, xlab in [(axes[0], "log-loss", "log-loss on the hidden disease (lower = better)"),
                              (axes[1], "confidence when wrong", "mean confidence on missed leaves (lower = more honest)")]:
            ax.barh(yy + 0.2, unseen[f"MAP {key}"].fillna(0), 0.38, color=C["plain"], label="MAP plug-in (plain LR)")
            ax.barh(yy - 0.2, unseen[f"Bayes {key}"].fillna(0), 0.38, color=C["bayes"], label="Bayesian predictive")
            ax.set_xlabel(xlab); ax.grid(axis="y", visible=False)
        axes[0].set_yticks(yy, unseen["Hidden disease"])
        axes[1].set_xlim(0.5, 1.0)
        for i, v in enumerate(unseen["MAP confidence when wrong"]):
            if np.isnan(v):
                axes[1].text(0.51, i, "no missed leaves", va="center", fontsize=8, color=C["muted"])
        axes[0].legend(fontsize=8, loc="lower right")
        axes[0].set_title("A disease the model never saw")
        save_fig(fig, "bayes", "7_unseen_disease", reports_dir)

    # -- 5. weights with 95% credible intervals
    lo, hi = blr.credible_intervals()
    top = np.argsort(-np.abs(blr.coef_))[:15][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ypos = np.arange(len(top))
    ax.errorbar(blr.coef_[top], ypos, xerr=[blr.coef_[top] - lo[top], hi[top] - blr.coef_[top]], fmt="none",
                ecolor=C["muted"], elinewidth=1.4, capsize=3)
    ax.scatter(blr.coef_[top], ypos, s=46, zorder=3, edgecolors="white", linewidths=1.2,
               c=[C["diseased"] if v > 0 else C["healthy"] for v in blr.coef_[top]])
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(ypos, [F.FEATURE_NAMES[j] for j in top]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("weight on standardised feature   (+ pushes towards diseased, - towards healthy)")
    ax.set_title("15 largest weights with 95% credible intervals")
    save_fig(fig, "bayes", "5_weights", reports_dir)

    # -- 6. reliability diagram
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], color=C["muted"], ls=":", lw=1, label="perfect calibration")
    edges = np.linspace(0, 1, 11)
    for p, col, lab, mk in [(p_sk, C["plain"], "plain logistic regression", "o"),
                            (p_bayes, C["bayes"], "Bayesian predictive", "s")]:
        b = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
        ys = [yv[b == k].mean() for k in range(10) if (b == k).sum() >= 5]
        ax.plot(xs, ys, color=col, marker=mk, ms=6, label=lab)
    ax.set_xlabel("predicted P(diseased)"); ax.set_ylabel("observed fraction diseased")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Reliability diagram (valid set)")
    save_fig(fig, "bayes", "6_reliability", reports_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bayesian logistic regression (healthy vs diseased) from scratch.")
    ap.add_argument("--limit", type=int, default=None, help="use the --limit N feature cache")
    args = ap.parse_args(argv)
    from src import dataset
    try:
        data = dataset.load_features(args.limit)
    except FileNotFoundError as e:
        print(e)
        return 1
    # call run() through the imported module, so the saved model pickles as src.bayes_logreg.BayesianLogisticRegression
    # (not __main__.BayesianLogisticRegression, which other modules could not load)
    from src import bayes_logreg
    bayes_logreg.run(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
