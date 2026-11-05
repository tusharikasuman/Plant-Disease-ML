"""Tests for src/predict.py and the Streamlit app."""
import numpy as np
import pytest

import config
from src import classify as C
from src import features as F
from src import predict as P


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory, ):
    """Train LR + RF on real features of the sample leaves (plus jittered copies) into a temp folder."""
    rng = np.random.default_rng(0)
    paths = sorted(config.SAMPLES_DIR.glob("*.jpg"))
    names = [p.stem.replace("tomato_", "Tomato___") for p in paths]   # e.g. Tomato___healthy
    base = np.array([F.extract_features(p)[0] for p in paths])
    X = np.vstack([base * rng.normal(1, 0.03, size=base.shape) for _ in range(12)])
    y = np.tile(np.arange(len(paths)), 12)
    data = {"X_train": X, "y_train": y, "X_valid": X[:9], "y_valid": y[:9], "class_names": names,
            "feature_names": list(F.FEATURE_NAMES), "valid_source": "test", "meta": {}}
    out = tmp_path_factory.mktemp("models")
    C.run(data, model_keys=["lr", "rf"], cv_per_class=None, ablation=False, n_jobs=1,
          reports_dir=out, models_dir=out)

    import joblib
    from sklearn.preprocessing import StandardScaler
    from src.bayes_logreg import BayesianLogisticRegression, binary_target
    scaler = StandardScaler().fit(X)
    blr = BayesianLogisticRegression(alpha=1.0).fit(scaler.transform(X), binary_target(y, names))
    joblib.dump({"scaler": scaler, "model": blr}, out / config.BAYES_MODEL_PATH.name)
    return P.load_models(out), paths, names


def test_predict_sample_images(tiny_models):
    models, paths, names = tiny_models
    for path, name in zip(paths, names):
        p = P.predict(path, models)
        assert p.label == name                               # trained on these leaves -> must be right
        assert len(p.top3) == 3 and p.top3[0][0] == p.label
        assert abs(sum(prob for _, prob in p.top3) - 1) < 1e-6   # 3 classes -> top-3 covers everything
        assert p.is_healthy == name.endswith("healthy")
        assert 0 <= p.lesion_percent <= 100 and -1 <= p.greenness_index <= 1
        assert p.spad is None                                # no chlorophyll model -> never a SPAD value
        assert p.explanation and set(p.group_explanation) == set(P.GROUP_LABELS.values())
        assert (p.bayes_p_diseased >= 0.5) == (not p.is_healthy)     # Bayesian second opinion agrees
        assert abs(p.bayes_p_diseased - 0.5) <= abs(p.bayes_p_diseased_map - 0.5) + 1e-12


def test_group_explanation_sums_single_contributions(tiny_models):
    models, paths, _ = tiny_models
    p = P.predict(paths[0], models)
    top, groups = P.explain(models["logreg"], p.features, p.label, top_k=F.N_FEATURES)
    assert sum(groups.values()) == pytest.approx(sum(e["contribution"] for e in top))


def test_predict_accepts_bytes_and_arrays(tiny_models):
    models, paths, _ = tiny_models
    a = P.predict(paths[0].read_bytes(), models)
    b = P.predict(F.read_image(paths[0]), models)
    assert a.top3 == b.top3


def test_greenness_is_labelled_not_chlorophyll(tiny_models):
    models, paths, _ = tiny_models
    text = P.format_prediction(P.predict(paths[0], models))
    assert "NOT a chlorophyll" in text and "SPAD (calibrated" not in text


def test_missing_models_give_instructions(tmp_path):
    with pytest.raises(P.ModelsNotTrainedError, match="python -m src.classify"):
        P.load_models(tmp_path)


def test_cli(tiny_models, monkeypatch, capsys):
    models, paths, _ = tiny_models
    monkeypatch.setattr(P, "load_models", lambda: models)
    assert P.main([str(paths[0])]) == 0
    assert "Top-3 predictions" in capsys.readouterr().out
    assert P.main(["does_not_exist.jpg"]) == 1


def test_streamlit_app_runs_every_page(tiny_models, monkeypatch):
    from streamlit.testing.v1 import AppTest
    models, _, _ = tiny_models
    monkeypatch.setattr(P, "load_models", lambda: models)
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.selectbox[0].select(at.selectbox[0].options[1]).run()      # first sample leaf
    assert not at.exception
    assert {m.label for m in at.metric} >= {"Diagnosis", "Confidence", "Lesion area", "Greenness index (GLI)"}
    assert "P(diseased), Bayesian" in {m.label for m in at.metric}
    for page in ["Model results", "All graphs", "Bayesian LR", "Chlorophyll", "How it works"]:
        at.radio[0].set_value(page).run()
        assert not at.exception
