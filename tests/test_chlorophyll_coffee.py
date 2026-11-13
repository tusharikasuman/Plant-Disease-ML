"""Coffee chlorophyll (SPAD) module: reading the ArabicaLens layout, training, held-out test, one-photo estimate.

The synthetic leaves and SPAD values exist only inside pytest's temporary folder, to test the code path; they are
never written to the project's data/, cache/ or models/ folders.
"""
import cv2
import numpy as np
import pandas as pd
import pytest

from src import chlorophyll_coffee as CC
from src import features as F
from tests.conftest import make_leaf


@pytest.fixture
def fake_arabicalens(tmp_path):
    """60 synthetic leaves: the greener the leaf, the higher its 'SPAD' (5 noisy readings each)."""
    rng = np.random.default_rng(0)
    images = tmp_path / "Images"
    images.mkdir()
    rows = []
    for i in range(60):
        g = rng.uniform(70, 220)
        img, leaf, _ = make_leaf(spots=False)
        img[leaf > 0] = (40, int(g), 60)
        cv2.imwrite(str(images / f"cof_{i + 1}.jpg"), img)
        spad = 5 + 0.3 * (g - 70)
        rows.append({"id": f"cof_{i + 1}", **{f"SPAD_{k}": spad + rng.normal(0, 1) for k in range(1, 6)},
                     "Healthy/Unhealthy": 1})
    df = pd.DataFrame(rows)
    df["SPAD_Avg"] = df[[f"SPAD_{k}" for k in range(1, 6)]].mean(axis=1)
    df.to_excel(tmp_path / "coffee.xlsx", index=False)
    return tmp_path


def test_missing_data_gives_download_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="zenodo"):
        CC.load_readings(tmp_path / "coffee.xlsx", tmp_path / "Images")


def test_readings_are_averaged_per_leaf(fake_arabicalens):
    r = CC.load_readings(fake_arabicalens / "coffee.xlsx", fake_arabicalens / "Images")
    assert len(r) == 60 and r["spad_sd"].gt(0).all()
    sheet = pd.read_excel(fake_arabicalens / "coffee.xlsx")
    assert np.allclose(r["spad"], sheet["SPAD_Avg"])


def test_split_is_stratified_and_disjoint():
    y = np.linspace(5, 70, 200)
    tr, te = CC.split(y)
    assert len(te) == 40 and not set(tr) & set(te)
    assert y[te].min() < 20 and y[te].max() > 55          # the test set covers pale to dark leaves


def test_train_test_and_predict(fake_arabicalens, tmp_path):
    r = CC.load_readings(fake_arabicalens / "coffee.xlsx", fake_arabicalens / "Images")
    X = np.array([F.extract_features(p)[0] for p in r["image_path"]])
    out = CC.run(reports_dir=tmp_path / "reports", models_dir=tmp_path / "models", readings=r, X=X)
    assert out["test_metrics"]["R2"] > 0.8                 # greener = more SPAD is easy to learn
    assert len(out["graphs"]) == 3 and all(g.exists() for g in out["graphs"])
    model = tmp_path / "models" / CC.MODEL_PATH.name
    e = CC.predict(r["image_path"].iloc[0], model_path=model)
    assert abs(e["spad"] - r["spad"].iloc[0]) < 10 and e["gp_sd"] > 0 and e["leaf_found"]


def test_predict_without_model_explains(tmp_path):
    with pytest.raises(FileNotFoundError, match="chlorophyll_coffee"):
        CC.predict("x.jpg", model_path=tmp_path / "none.joblib")
