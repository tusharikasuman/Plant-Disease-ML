"""HMM tests: forward-backward and Viterbi checked against brute force over every possible state path."""
import itertools

import numpy as np
import pytest

from src.hmm import HMM, N_SYMBOLS, quantise


@pytest.fixture
def tiny():
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.2, 0.8]])
    B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    obs = np.array([[0, 2, 1, 2, 0], [2, 2, 2, 0, 1]])
    return HMM(pi, A, B), obs


def brute_force(hmm, x):
    """Probability of every path z: p(x, z) = pi[z0] B[z0, x0] prod A[z_t-1, z_t] B[z_t, x_t]."""
    paths, probs = [], []
    for z in itertools.product(range(len(hmm.pi)), repeat=len(x)):
        p = hmm.pi[z[0]] * hmm.B[z[0], x[0]]
        for t in range(1, len(x)):
            p *= hmm.A[z[t - 1], z[t]] * hmm.B[z[t], x[t]]
        paths.append(z)
        probs.append(p)
    return np.array(paths), np.array(probs)


def test_forward_log_likelihood_matches_brute_force(tiny):
    hmm, obs = tiny
    _, ll = hmm.posteriors(obs)
    for r, x in enumerate(obs):
        _, probs = brute_force(hmm, x)
        assert ll[r] == pytest.approx(np.log(probs.sum()), abs=1e-10)


def test_forward_backward_posteriors_match_brute_force(tiny):
    hmm, obs = tiny
    gamma, _ = hmm.posteriors(obs)
    for r, x in enumerate(obs):
        paths, probs = brute_force(hmm, x)
        for t in range(len(x)):
            for k in range(2):
                exact = probs[paths[:, t] == k].sum() / probs.sum()
                assert gamma[r, t, k] == pytest.approx(exact, abs=1e-10)
    np.testing.assert_allclose(gamma.sum(-1), 1.0)


def test_viterbi_finds_the_most_probable_path(tiny):
    hmm, obs = tiny
    path, logp = hmm.viterbi(obs)
    for r, x in enumerate(obs):
        paths, probs = brute_force(hmm, x)
        np.testing.assert_array_equal(path[r], paths[probs.argmax()])
        assert logp[r] == pytest.approx(np.log(probs.max()), abs=1e-10)


def test_supervised_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    true = HMM(np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([[0.8, 0.2], [0.1, 0.9]]))
    R, T = 400, 60
    z = np.empty((R, T), dtype=int); x = np.empty((R, T), dtype=int)
    z[:, 0] = rng.random(R) < true.pi[1]
    for t in range(1, T):
        z[:, t] = rng.random(R) < true.A[z[:, t - 1], 1]
    x[:] = rng.random((R, T)) < true.B[z, 1]
    fit = HMM.fit_supervised(x, z, 2, 2, smoothing=0)
    np.testing.assert_allclose(fit.A, true.A, atol=0.01)
    np.testing.assert_allclose(fit.B, true.B, atol=0.01)


def test_long_rows_do_not_underflow():
    hmm = HMM(np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.9, 0.1], [0.1, 0.9]]))
    obs = np.random.default_rng(0).integers(0, 2, size=(3, 5000))
    gamma, ll = hmm.posteriors(obs)
    assert np.all(np.isfinite(gamma)) and np.all(np.isfinite(ll)) and np.all(ll < 0)


def test_quantise_range():
    img = np.random.default_rng(0).integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    q = quantise(img)
    assert q.min() >= 0 and q.max() < N_SYMBOLS
