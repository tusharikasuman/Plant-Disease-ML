"""Phase 1 tests: dataset auto-detection, --limit sampling, parallel extraction and the feature cache."""
import cv2
import numpy as np
import pytest

from src import dataset as D
from src import features as F
from tests.conftest import make_leaf

CLASSES = ["Tomato___Early_blight", "Tomato___healthy"]


def _write_split(folder, n_per_class):
    for cls in CLASSES:
        (folder / cls).mkdir(parents=True)
        for i in range(n_per_class):
            img, _, _ = make_leaf(spots=cls != "Tomato___healthy", size=96)
            img = np.roll(img, i, axis=1)                    # make every image slightly different
            cv2.imwrite(str(folder / cls / f"img{i:03d}.jpg"), img)


@pytest.fixture
def kaggle_like(tmp_path):
    """tmp/New Plant Diseases Dataset(Augmented)/{train,valid}/<class>/*.jpg, plus a 1-class decoy folder."""
    top = tmp_path / "New Plant Diseases Dataset(Augmented)"
    _write_split(top / "train", 30)
    _write_split(top / "valid", 4)
    (top / "train" / "Tomato___healthy" / "corrupt.jpg").write_bytes(b"broken")
    decoy = tmp_path / "old_subset"
    (decoy / "Tomato___healthy").mkdir(parents=True)
    cv2.imwrite(str(decoy / "Tomato___healthy" / "a.jpg"), make_leaf()[0])
    return tmp_path


def test_find_dataset_prefers_train_with_valid(kaggle_like):
    loc = D.find_dataset(kaggle_like, verbose=False)
    assert loc.train_dir.name == "train" and loc.valid_dir.name == "valid"
    assert loc.classes == CLASSES


def test_missing_dataset_prints_instructions(tmp_path, capsys):
    assert D.find_dataset(tmp_path, verbose=True) is None
    out = capsys.readouterr().out
    assert "Dataset not found" in out and "kaggle datasets download" in out
    assert D.main(["--data", str(tmp_path)]) == 1                   # clean exit code, no traceback


def test_limit_is_per_class_and_reproducible(kaggle_like):
    loc = D.find_dataset(kaggle_like, verbose=False)
    paths, labels = D.list_images(loc.train_dir, loc.classes, limit=5)
    assert np.bincount(labels).tolist() == [5, 5]
    assert D.list_images(loc.train_dir, loc.classes, limit=5)[0] == paths


def test_parallel_matches_sequential(kaggle_like):
    loc = D.find_dataset(kaggle_like, verbose=False)
    paths, _ = D.list_images(loc.train_dir, loc.classes, limit=None)
    X1, ok1, _ = D.extract_many(paths, n_jobs=1)
    X2, ok2, _ = D.extract_many(paths, n_jobs=2)
    np.testing.assert_array_equal(ok1, ok2)
    np.testing.assert_array_equal(X1, X2)
    assert ok1.sum() == len(paths) - 1                               # only the corrupt file is skipped


def test_build_and_load_cache(kaggle_like, tmp_path, capsys):
    loc = D.find_dataset(kaggle_like, verbose=False)
    cache = tmp_path / "cache"
    D.build_feature_cache(loc, limit=None, n_jobs=1, cache_dir=cache)
    d = D.load_features(cache_dir=cache)
    assert d["X_train"].shape == (60, F.N_FEATURES)                  # 61 files - 1 corrupt
    assert d["X_valid"].shape == (8, F.N_FEATURES)
    assert len(d["y_train"]) == len(d["paths_train"]) == len(d["leaf_found_train"]) == 60
    assert d["class_names"] == CLASSES and d["feature_names"] == F.FEATURE_NAMES
    assert np.all(np.isfinite(d["X_train"]))
    assert d["valid_source"] == "valid folder"

    capsys.readouterr()
    D.build_feature_cache(loc, limit=None, n_jobs=1, cache_dir=cache)
    assert "up to date" in capsys.readouterr().out                   # second run uses the cache


def test_split_when_no_valid_folder(tmp_path):
    _write_split(tmp_path / "color", 10)
    loc = D.find_dataset(tmp_path, verbose=False)
    assert loc.valid_dir is None
    D.build_feature_cache(loc, limit=None, n_jobs=1, cache_dir=tmp_path / "cache")
    d = D.load_features(cache_dir=tmp_path / "cache")
    assert len(d["y_train"]) == 16 and len(d["y_valid"]) == 4
    assert np.bincount(d["y_valid"]).tolist() == [2, 2]              # stratified
    assert not set(d["paths_train"]) & set(d["paths_valid"])         # no overlap


def test_load_missing_cache_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.dataset --limit 7"):
        D.load_features(limit=7, cache_dir=tmp_path)
