"""Unit 3 tests: PCA, k-means, Gaussian mixture (EM) and the cluster scores."""
import numpy as np
import pytest

import config

from src import features as F
from src import unsupervised as U


def test_purity_and_scores():
    y = np.array([0, 0, 1, 1, 2, 2])
    assert U.purity(y, np.array([5, 5, 7, 7, 9, 9])) == 1.0          # cluster ids do not need to match labels
    assert U.purity(y, np.zeros(6, dtype=int)) == pytest.approx(2 / 6)
    s = U.cluster_scores(y, np.array([1, 1, 0, 0, 2, 2]))
    assert s["ARI"] == pytest.approx(1.0) and s["NMI"] == pytest.approx(1.0)


def test_em_log_likelihood_never_decreases():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(loc=m, size=(100, 2)) for m in (-4, 0, 4)])
    trace = U.em_trace(Z, 3)
    assert np.all(np.diff(trace) >= -1e-9)


def test_run_on_separable_data(tmp_path):
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 60)
    X = rng.normal(size=(len(y), F.N_FEATURES))
    for k in range(3):                                                # three well separated groups:
        X[y == k, 15 * k:15 * k + 15] += 4.0                          # each shifted along its own 15 features
    data = {"X_train": X, "y_train": y,
            "class_names": ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]}
    out = U.run(data, reports_dir=tmp_path, n_jobs=1)
    clus = out["clustering"].set_index("Method")
    assert clus.loc["k-means", "ARI"] > 0.9 and clus.loc["GMM (EM)", "ARI"] > 0.9
    assert clus.loc["Random assignment (reference)", "ARI"] < 0.1
    for topic, name in [("pca", "1_pca_scatter_scree.png"), ("pca", "2_accuracy_vs_components.png"),
                        ("kmeans", "1_elbow_silhouette.png"), ("kmeans", "2_clusters_vs_classes.png"),
                        ("gmm", "1_bic_and_em.png"), ("gmm", "2_clusters_vs_classes.png")]:
        assert (config.figure_dir(topic, tmp_path) / name).exists(), name
    for name in ["unsup_pca.csv", "unsup_clustering.csv", "unsup_bic.csv", "unsup_kmeans_elbow.csv"]:
        assert (config.tables_dir(tmp_path) / name).exists(), name
