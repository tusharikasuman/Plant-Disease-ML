"""Phase 3 tests: Bayesian logistic regression from scratch."""
import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.bayes_logreg import (BayesianLogisticRegression, binary_target, expected_calibration_error,
                              select_alpha)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5))
    w = np.array([1.5, -2.0, 0.5, 0.0, 1.0])
    y = (rng.random(300) < expit(X @ w + 0.3)).astype(int)
    return X, y


@pytest.mark.parametrize("alpha", [0.01, 1.0, 10.0])
def test_map_weights_match_sklearn(data, alpha):
    """MAP with prior N(0, 1/alpha) == sklearn LogisticRegression with C = 1/alpha (unpenalised intercept)."""
    X, y = data
    blr = BayesianLogisticRegression(alpha=alpha).fit(X, y)
    sk = LogisticRegression(C=1 / alpha, tol=1e-12, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(blr.coef_, sk.coef_[0], atol=1e-5)
    assert blr.intercept_ == pytest.approx(sk.intercept_[0], abs=1e-5)
    assert blr.converged_


def test_gradient_is_zero_and_hessian_matches_finite_differences(data):
    X, y = data
    m = BayesianLogisticRegression(alpha=1.0).fit(X, y)
    Phi = m._design(X)
    a = m._prior_precision(Phi.shape[1])
    grad = lambda w: Phi.T @ (expit(Phi @ w) - y) + a * w
    assert np.abs(grad(m.w_map_)).max() < 1e-8
    eps = 1e-6
    H_num = np.array([(grad(m.w_map_ + eps * e) - grad(m.w_map_ - eps * e)) / (2 * eps) for e in np.eye(len(a))])
    np.testing.assert_allclose(H_num, m.hessian_, rtol=1e-5, atol=1e-6)


def test_posterior_covariance_is_symmetric_positive_definite(data):
    m = BayesianLogisticRegression(alpha=1.0).fit(*data)
    np.testing.assert_allclose(m.cov_, m.cov_.T)
    assert np.all(np.linalg.eigvalsh(m.cov_) > 0)
    np.testing.assert_allclose(m.cov_ @ m.hessian_, np.eye(len(m.cov_)), atol=1e-8)


def test_predictive_is_moderated_but_makes_the_same_decisions(data):
    X, y = data
    m = BayesianLogisticRegression(alpha=1.0).fit(X, y)
    p_map, p_bayes = m.predict_proba_map(X)[:, 1], m.predict_proba(X)[:, 1]
    assert np.all(np.abs(p_bayes - 0.5) <= np.abs(p_map - 0.5) + 1e-12)     # pulled towards 0.5
    np.testing.assert_array_equal(p_bayes >= 0.5, p_map >= 0.5)               # same side of 0.5
    np.testing.assert_allclose(m.predict_proba(X).sum(axis=1), 1.0)


def test_probit_approximation_matches_monte_carlo(data):
    X, y = data
    m = BayesianLogisticRegression(alpha=1.0).fit(X[:40], y[:40])           # small data: wide posterior
    W = m.sample_weights(40000)
    mc = expit(m._design(X[:50]) @ W.T).mean(axis=1)
    assert np.abs(mc - m.predict_proba(X[:50])[:, 1]).max() < 0.02


def test_uncertainty_grows_away_from_the_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 1))
    y = (X[:, 0] + rng.normal(scale=0.5, size=100) > 0).astype(int)
    m = BayesianLogisticRegression(alpha=1.0).fit(X, y)
    s = m.uncertainty(np.array([[0.0], [3.0], [10.0]]))
    assert s[0] < s[1] < s[2]


def test_separable_data_stays_finite():
    """Plain MLE would send the weights to infinity; the Gaussian prior keeps them finite."""
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    m = BayesianLogisticRegression(alpha=0.1).fit(X, y)
    assert m.converged_ and np.all(np.isfinite(m.w_map_)) and np.all(np.isfinite(m.cov_))


def test_credible_intervals_contain_map(data):
    m = BayesianLogisticRegression(alpha=1.0).fit(*data)
    lo, hi = m.credible_intervals(0.95)
    assert np.all(lo < m.coef_) and np.all(m.coef_ < hi)
    lo90, hi90 = m.credible_intervals(0.90)
    assert np.all(hi90 - lo90 < hi - lo)


def test_evidence_prefers_a_reasonable_prior(data):
    alpha, evidence = select_alpha(*data, alphas=[1e-4, 1e-2, 1.0, 100.0, 1e4])
    assert alpha in (1e-2, 1.0)                     # neither "no prior" nor "crush every weight to 0"
    assert len(evidence) == 5


def test_binary_target_and_ece():
    names = ["Tomato___Early_blight", "Tomato___healthy"]
    np.testing.assert_array_equal(binary_target(np.array([0, 1, 1, 0]), names), [1, 0, 0, 1])
    y = np.array([1, 1, 0, 0])
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(1.0)


def test_rejects_non_binary_labels(data):
    with pytest.raises(ValueError):
        BayesianLogisticRegression().fit(data[0], np.arange(len(data[0])) % 3)
