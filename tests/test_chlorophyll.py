"""Phase 4 tests: SPAD regression - no fake data in the project, validation, pipeline on synthetic leaves.

The synthetic SPAD values below exist only inside pytest's temporary folders to test the code path;
they are never written to the project's spad_readings.csv or models/ folder.
"""
import cv2
import joblib
import numpy as np
import pandas as pd
import pytest

import config
from src import chlorophyll as CH
from src import features as F
from tests.conftest import make_leaf


def write_leaves(folder, greens):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, g in enumerate(greens):
        img, leaf, _ = make_leaf(spots=False)
        img[leaf > 0] = (40, int(g), 60)
        p = folder / f"leaf_{i:02d}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


def test_template_is_header_only(tmp_path):
    path = CH.write_template(tmp_path / "spad_readings.csv")
    assert path.read_text().strip() == ",".join(CH.TEMPLATE_COLUMNS)    # no example numbers in it
    assert CH.load_readings(path).empty


def test_project_ships_no_spad_data():
    """A chlorophyll model may only exist in the project once enough real SPAD readings exist."""
    n_real = len(CH.load_readings(config.SPAD_CSV)) if config.SPAD_CSV.exists() else 0
    if n_real < config.MIN_SPAD_LEAVES:
        assert not config.CHLOROPHYLL_MODEL_PATH.exists()


def test_no_readings_means_no_model(tmp_path, capsys):
    out = CH.run(tmp_path / "spad_readings.csv", reports_dir=tmp_path, models_dir=tmp_path)
    assert out is None
    assert not (tmp_path / config.CHLOROPHYLL_MODEL_PATH.name).exists()
    assert "UNCALIBRATED greenness index" in capsys.readouterr().out


def test_too_few_readings_refused(tmp_path):
    paths = write_leaves(tmp_path / "photos", np.linspace(90, 200, 5))
    pd.DataFrame({"image_path": [str(p) for p in paths], "spad_value": [30, 35, 40, 45, 50]}).to_csv(
        tmp_path / "s.csv", index=False)
    assert CH.run(tmp_path / "s.csv", reports_dir=tmp_path, models_dir=tmp_path) is None
    assert not (tmp_path / config.CHLOROPHYLL_MODEL_PATH.name).exists()


def test_validation_rejects_bad_rows(tmp_path):
    paths = write_leaves(tmp_path / "photos", [120])
    csv = tmp_path / "s.csv"
    pd.DataFrame({"image_path": [str(paths[0])], "spad_value": [140]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="between"):
        CH.load_readings(csv)
    pd.DataFrame({"image_path": ["photos/missing.jpg"], "spad_value": [40]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="not found"):
        CH.load_readings(csv)
    pd.DataFrame({"photo": ["x.jpg"], "value": [40]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="columns"):
        CH.load_readings(csv)


def test_repeated_readings_are_averaged_and_relative_paths_resolved(tmp_path):
    write_leaves(tmp_path / "photos", [120, 150])
    csv = tmp_path / "s.csv"
    pd.DataFrame({"image_path": ["photos/leaf_00.jpg"] * 3 + ["photos/leaf_01.jpg"],
                  "spad_value": [40, 42, 44, 30], "crop": "tomato"}).to_csv(csv, index=False)
    r = CH.load_readings(csv).set_index("image_path")
    first = str((tmp_path / "photos" / "leaf_00.jpg").resolve())
    assert r.loc[first, "spad_value"] == pytest.approx(42.0) and r.loc[first, "n_readings"] == 3
    assert len(r) == 2


def test_vif():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=500), rng.normal(size=500)
    independent = CH.variance_inflation_factors(np.column_stack([a, b, rng.normal(size=500)]))
    assert np.all(independent < 1.1)
    collinear = CH.variance_inflation_factors(np.column_stack([a, b, a + b + 0.01 * rng.normal(size=500)]))
    assert np.all(collinear > 100)


def test_training_pipeline_on_synthetic_leaves(tmp_path):
    """SPAD made to depend on leaf greenness: every model must find it, and the best one must be saved."""
    rng = np.random.default_rng(0)
    paths = write_leaves(tmp_path / "photos", np.linspace(80, 210, 24))
    g = np.array([F.extract_features(p)[0][F.FEATURE_NAMES.index("G mean")] for p in paths])
    spad = 10 + 0.2 * g + rng.normal(scale=0.5, size=len(g))
    pd.DataFrame({"image_path": [p.name for p in paths], "spad_value": spad.round(2), "crop": "test"}).to_csv(
        tmp_path / "photos" / "s.csv", index=False)

    results = CH.run(tmp_path / "photos" / "s.csv", reports_dir=tmp_path, models_dir=tmp_path)
    assert list(results.columns[:2]) == ["Model", "Syllabus"] and len(results) == 5
    assert results.iloc[0]["R2"] > 0.8                                    # leave-one-out R^2 of the best model
    bundle = joblib.load(tmp_path / config.CHLOROPHYLL_MODEL_PATH.name)
    assert bundle["n_leaves"] == 24 and bundle["cv"] == "leave-one-out"
    x = F.extract_features(paths[0])[0]
    assert abs(bundle["pipeline"].predict(x[None])[0] - spad[0]) < 3      # takes the full 55-feature vector
    assert (config.tables_dir(tmp_path) / "chlorophyll_results.csv").exists()
    for name in ["2_pred_vs_actual.png", "3_coefficient_stability.png"]:
        assert (config.figure_dir("chlorophyll", tmp_path) / name).exists(), name


def test_collinearity_report_runs_on_dataset_like_features(tmp_path):
    rng = np.random.default_rng(0)
    data = {"X_train": rng.normal(size=(60, F.N_FEATURES)), "X_valid": rng.normal(size=(20, F.N_FEATURES))}
    CH.run(tmp_path / "s.csv", reports_dir=tmp_path, models_dir=tmp_path, dataset_features=data)
    assert (config.figure_dir("chlorophyll", tmp_path) / "1_collinearity.png").exists()
    assert (config.tables_dir(tmp_path) / "chlorophyll_vif.csv").exists()
