"""Unit 5 tests: neural network, AdaBoost, SVM dual, RVM, MCMC, Gibbs/Ising and the Bayesian network."""
import numpy as np
import pytest

from src import unit5 as U
from tests.conftest import make_leaf


def blobs(n=300, k=3, d=4, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (k, d))
    y = np.repeat(np.arange(k), n // k)
    return centres[y] + rng.normal(0, 1, (len(y), d)), y


# ----------------------------------------------------------------------------- neural network
def test_backprop_matches_finite_differences():
    X, y = blobs(60)
    net = U.MLP(X.shape[1], 6, 3)
    assert U.gradient_check(net, X, U.one_hot(y, 3), n_checks=50) < 1e-5


def test_mlp_learns_a_non_linear_problem():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (400, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)            # XOR-like: no straight line separates it
    net = U.MLP(2, 16, 2).fit(X, y, epochs=200, batch_size=32, lr=1e-2)
    assert np.mean(net.predict(X) == y) > 0.9
    assert net.history["train_loss"][-1] < net.history["train_loss"][0]
    assert np.allclose(net.predict_proba(X).sum(axis=1), 1)


# ----------------------------------------------------------------------------- combining models
def test_adaboost_from_scratch_matches_sklearn():
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    X, y = blobs(300, k=3, seed=2)
    ours = U.AdaBoostSAMME(n_rounds=30).fit(X, y)
    sk = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=30,
                            random_state=42).fit(X, y)
    assert np.mean(ours.predict(X) == sk.predict(X)) > 0.98
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert np.mean(ours.predict(X) == y) > np.mean(stump.predict(X) == y)


# ----------------------------------------------------------------------------- kernels
def test_svm_decision_rebuilt_from_support_vectors():
    from sklearn.svm import SVC
    X, y = blobs(200, k=2, seed=3)
    svc = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X, y)
    assert np.allclose(U.svm_decision_by_hand(svc, X), svc.decision_function(X), atol=1e-8)


def test_rvm_is_sparse_and_fits_a_sine():
    rng = np.random.default_rng(4)
    X = np.sort(rng.uniform(-3, 3, 100))[:, None]
    t = np.sin(2 * X[:, 0]) + rng.normal(0, 0.1, 100)
    rvm = U.RVMRegression(gamma=1.0).fit(X, t)
    mean, std = rvm.predict(X, return_std=True)
    assert len(rvm.relevance_idx_) < 30                   # far fewer than the 100 training points
    assert np.sqrt(np.mean((mean - np.sin(2 * X[:, 0])) ** 2)) < 0.1
    assert np.all(std > 0)


# ----------------------------------------------------------------------------- sampling & graphical models
def test_metropolis_hastings_agrees_with_laplace_on_lots_of_data():
    from src.bayes_logreg import BayesianLogisticRegression
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, (2000, 1))
    t = (rng.random(2000) < 1 / (1 + np.exp(-(0.5 + 2 * X[:, 0])))).astype(int)
    lap = BayesianLogisticRegression(alpha=1.0).fit(X, t)
    Phi = np.hstack([np.ones((len(X), 1)), X])
    w0 = lap.w_map_[[1, 0]]
    samples, rate = U.metropolis_hastings(lambda w: U.log_posterior_logreg(w, Phi, t, 1.0), w0, 6000, step=0.08)
    assert 0.1 < rate < 0.9
    assert np.allclose(samples[1000:].mean(axis=0), w0, atol=0.05)


def test_gibbs_ising_removes_noise_from_a_lesion_mask():
    img, leaf, lesion = make_leaf()
    region = leaf > 0
    truth = np.where(lesion > 0, 1.0, -1.0)
    rng = np.random.default_rng(6)
    noisy = np.where((rng.random(truth.shape) < 0.15) & region, -truth, truth)
    post, energy = U.ising_gibbs(noisy, region, beta=1.0, eta=0.87, sweeps=20, burn_in=5)
    clean = np.where(post > 0.5, 1.0, -1.0)
    before = np.mean(noisy[region] != truth[region])
    after = np.mean(clean[region] != truth[region])
    assert after < before / 3
    assert energy[-1] < energy[0]


def test_bayes_net_exact_and_sampled_inference():
    rng = np.random.default_rng(7)
    n = 3000
    D = rng.integers(0, 3, n)
    L = np.clip(D + rng.integers(-1, 2, n), 0, 2)
    Y = rng.integers(0, 3, n)
    T = np.clip(L + rng.integers(-1, 2, n), 0, 2)
    bn = U.DiscreteBayesNet(3).fit(D, L, Y, T)
    J = bn.joint()
    assert np.isclose(J.sum(), 1)
    exact = bn.exact_posterior({"T": 2})
    brute = J[:, :, :, 2].sum(axis=(1, 2))
    assert np.allclose(exact, brute / brute.sum())
    for est in (bn.rejection({"T": 2}, 40000, rng), bn.likelihood_weighting(2, 40000, rng), bn.gibbs(2, 20000, rng)):
        assert np.abs(est - exact).max() < 0.03
    assert np.mean(bn.predict(L, Y, T) == D) > 0.4


@pytest.mark.parametrize("name,expected", [("Tomato___Early_blight", "Early blight"),
                                           ("Tomato___healthy", "Healthy"),
                                           ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Yellow leaf curl virus")])
def test_short_class_names(name, expected):
    assert U.short(name) == expected


def test_run_on_synthetic_features(tmp_path):
    """The whole Unit 5 pipeline runs end to end on a small synthetic feature set and writes its graphs."""
    from src import features as F
    rng = np.random.default_rng(8)
    names = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]
    nf = len(F.FEATURE_NAMES)

    def make(n):
        y = np.repeat(np.arange(3), n)
        X = rng.normal(0, 1, (len(y), nf)) + y[:, None] * 0.8
        X[:, F.FEATURE_NAMES.index("Lesion fraction")] = rng.uniform(0, 0.3, len(y)) * (y != 2)
        return np.abs(X), y

    Xt, yt = make(150)
    Xv, yv = make(60)
    data = {"X_train": Xt, "y_train": yt, "X_valid": Xv, "y_valid": yv, "class_names": names,
            "feature_names": list(F.FEATURE_NAMES)}
    res = U.run(data, parts=("kernels", "sampling"), reports_dir=tmp_path)
    for part in ("kernels", "sampling"):
        assert res[part]["graphs"] and all(p.exists() for p in res[part]["graphs"])
    assert (tmp_path / "tables" / "unit5_kernel_regression.csv").exists()
