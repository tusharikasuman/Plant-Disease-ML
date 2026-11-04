"""Phase 2 tests: classification pipeline, saved models, ablation, report tables."""
import joblib
import numpy as np
import pytest

from src import classify as C
from src import features as F

CLASSES = ["Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"]


def fake_features(n_train=30, n_valid=10, seed=0) -> dict:
    """Separable synthetic data shaped like the real feature cache (3 classes x 55 features)."""
    rng = np.random.default_rng(seed)

    def draw(n):
        y = np.repeat(np.arange(len(CLASSES)), n)
        X = rng.normal(size=(len(y), F.N_FEATURES))
        for k in range(len(CLASSES)):                   # each class is shifted along its own 3 features
            X[y == k, 3 * k:3 * k + 3] += 3.0
        return X, y

    X, y = draw(n_train)
    Xv, yv = draw(n_valid)
    return {"X_train": X, "y_train": y, "X_valid": Xv, "y_valid": yv, "class_names": CLASSES,
            "feature_names": list(F.FEATURE_NAMES), "valid_source": "test", "meta": {}}


def test_every_model_maps_to_a_syllabus_unit():
    specs = C.model_specs()
    assert [s.key for s in specs] == ["lr", "lda", "gnb", "svm_linear", "svm_rbf", "knn", "tree", "rf"]
    assert {s.unit for s in specs} == {"Unit 1", "Unit 2", "Baseline (Paper 1)"}
    assert all(s.grid for s in specs)                     # every model is tuned


def test_balanced_sample_caps_each_class():
    y = np.array([0] * 50 + [1] * 5 + [2] * 20)
    idx = C.balanced_sample(y, 10)
    assert np.bincount(y[idx]).tolist() == [10, 5, 10]
    np.testing.assert_array_equal(idx, C.balanced_sample(y, 10))      # reproducible
    assert len(C.balanced_sample(y, None)) == len(y)


def test_full_run_writes_reports_and_models(tmp_path):
    data = fake_features()
    results = C.run(data, cv_per_class=None, n_jobs=1,
                    reports_dir=tmp_path / "reports", models_dir=tmp_path / "models")
    assert len(results) == 8
    assert (results["Valid accuracy (%)"] > 80).all()       # easy data: every model should do well
    assert (results["Best"] == "*").sum() == 1
    for name in ["classification_results.csv", "classification_results.md", "ablation.csv", "ablation.md",
                 "syllabus_mapping.md", "classification_report_best.txt", "classification_scores.json"]:
        assert (tmp_path / "reports" / name).exists(), name

    for model_file in ["best_model.joblib", "logistic_regression.joblib"]:
        bundle = joblib.load(tmp_path / "models" / model_file)
        assert bundle["class_names"] == CLASSES and bundle["feature_names"] == F.FEATURE_NAMES
        proba = bundle["pipeline"].predict_proba(data["X_valid"])
        assert proba.shape == (len(data["y_valid"]), len(CLASSES))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_ablation_adds_feature_groups_in_order(tmp_path):
    data = fake_features()
    abl = C.run_ablation(data, {"lr": {"C": 1.0}}, cv_per_class=None, n_jobs=1)
    assert abl["# features"].tolist() == [6, 42, 48, 55]
    assert "Logistic Regression CV acc (%)" in abl.columns


def test_svm_best_model_still_gives_probabilities(tmp_path):
    """SVC has no predict_proba; the saved best model must be wrapped in CalibratedClassifierCV."""
    data = fake_features()
    C.run(data, model_keys=["svm_rbf"], cv_per_class=None, ablation=False, n_jobs=1,
          reports_dir=tmp_path, models_dir=tmp_path)
    bundle = joblib.load(tmp_path / "best_model.joblib")
    assert bundle["model_key"] == "svm_rbf"
    assert bundle["pipeline"].predict_proba(data["X_valid"]).shape == (30, 3)
    assert not (tmp_path / "logistic_regression.joblib").exists()   # LR was not trained


def test_results_are_reproducible(tmp_path):
    data = fake_features()
    a = C.run_models(data, ["lr", "rf"], cv_per_class=None, n_jobs=1)[0]
    b = C.run_models(data, ["lr", "rf"], cv_per_class=None, n_jobs=1)[0]
    np.testing.assert_array_equal(a["CV accuracy (%)"], b["CV accuracy (%)"])
    np.testing.assert_array_equal(a["Valid accuracy (%)"], b["Valid accuracy (%)"])


def test_cli_rejects_unknown_model():
    with pytest.raises(SystemExit):
        C.main(["--models", "cnn"])
