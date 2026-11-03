"""Phase 1 tests: preprocessing, leaf mask and the 55 hand-made features."""
import cv2
import numpy as np
import pytest

import config
from src import features as F
from tests.conftest import make_leaf


def test_feature_names_and_groups():
    assert F.N_FEATURES == len(F.FEATURE_NAMES) == 55
    assert len(set(F.FEATURE_NAMES)) == F.N_FEATURES                 # names are unique
    cols = sorted(i for idx in F.FEATURE_GROUPS.values() for i in idx)
    assert cols == list(range(F.N_FEATURES))                          # groups cover every column once
    assert {k: len(v) for k, v in F.FEATURE_GROUPS.items()} == \
        {"rgb": 6, "indices": 4, "hsv": 32, "texture": 6, "shape": 7}


def test_real_leaves_give_finite_vectors_and_one_leaf_mask(sample_images):
    for path in sample_images:
        vec, names = F.extract_features(path)
        assert vec.shape == (55,) and names == F.FEATURE_NAMES
        assert np.all(np.isfinite(vec)), path.name
        seg = F.segment_leaf(F.preprocess(F.read_image(path)))
        assert seg.leaf_found, path.name
        assert np.count_nonzero(seg.mask) > 0.1 * seg.mask.size      # mask is non-empty
        n_parts, _ = cv2.connectedComponents(seg.mask)
        assert n_parts == 2                                           # background + one leaf
        hist = vec[F.FEATURE_GROUPS["hsv"]]
        assert hist.sum() == pytest.approx(1.0, abs=1e-6)


def test_mask_matches_synthetic_leaf(synthetic_leaf):
    img, leaf, _ = synthetic_leaf
    seg = F.segment_leaf(F.preprocess(img))
    iou = np.logical_and(seg.mask, leaf).sum() / np.logical_or(seg.mask, leaf).sum()
    assert iou > 0.95


def test_lesion_fraction_matches_drawn_spots(synthetic_leaf):
    img, leaf, lesion = synthetic_leaf
    vec, _ = F.extract_features(img)
    expected = np.count_nonzero(lesion) / np.count_nonzero(leaf)
    assert vec[F.LESION_INDEX] == pytest.approx(expected, abs=0.01)


def test_healthy_leaf_has_no_lesions_and_high_gli():
    img, _, _ = make_leaf(spots=False)
    vec, _ = F.extract_features(img)
    assert vec[F.LESION_INDEX] < 0.005
    assert vec[F.GLI_INDEX] > 0.3                    # (2G-R-B)/(2G+R+B) = 0.53 for this green
    assert vec[F.FEATURE_NAMES.index("G/R ratio")] > 2.5


def test_lesions_inside_leaf_stay_in_mask():
    """A dark spot (neither green nor brown) inside the leaf must not become a hole."""
    img, leaf, _ = make_leaf(spots=False)
    cv2.circle(img, (128, 128), 15, (20, 20, 20), -1)                # near-black lesion
    seg = F.segment_leaf(F.preprocess(img))
    assert seg.mask[128, 128] == 255


def test_dark_shadow_beside_leaf_is_not_a_lesion():
    """A dark brownish shadow touching the leaf (outside its outline) must not count as lesion area."""
    img, leaf, _ = make_leaf(spots=False)
    img[:] = (150, 165, 180)                                           # beige paper
    img[leaf > 0] = (40, 150, 50)
    shadow = np.roll(leaf, 14, axis=1) & ~leaf
    img[shadow > 0] = (20, 35, 50)                                     # dark warm shadow: H ~ 7, S ~ 153, V 50
    vec, _ = F.extract_features(img)
    shadow_fraction = np.count_nonzero(shadow) / np.count_nonzero(leaf)
    # only blur-blended pixels along the leaf edge may remain, not the shadow itself
    assert vec[F.LESION_INDEX] < 0.2 * shadow_fraction


def test_background_only_image_falls_back_without_crashing():
    blank = np.full((300, 200, 3), 128, np.uint8)
    seg = F.segment_leaf(F.preprocess(blank))
    assert not seg.leaf_found and seg.mask.min() == 255              # whole image used
    vec, _ = F.extract_features(blank)
    assert np.all(np.isfinite(vec))


def test_black_image_has_no_divide_by_zero():
    with np.errstate(all="raise"):                                   # any 0/0 would raise here
        vec, _ = F.extract_features(np.zeros((64, 64, 3), np.uint8))
    assert np.all(np.isfinite(vec))


@pytest.mark.parametrize("shape", [(100, 150), (100, 150, 4), (40, 30, 3)])
def test_odd_image_shapes(shape, synthetic_leaf):
    img = cv2.resize(synthetic_leaf[0], (shape[1], shape[0]))
    if len(shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    vec, _ = F.extract_features(img)
    assert vec.shape == (55,) and np.all(np.isfinite(vec))


def test_deterministic(sample_images):
    a, _ = F.extract_features(sample_images[0])
    b, _ = F.extract_features(sample_images[0])
    np.testing.assert_array_equal(a, b)


def test_path_and_array_give_same_features(sample_images):
    from_path, _ = F.extract_features(sample_images[0])
    from_array, _ = F.extract_features(F.read_image(sample_images[0]))
    np.testing.assert_array_equal(from_path, from_array)


def test_unreadable_file(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        F.extract_features(bad)
    assert F.extract_for_dataset(str(bad)) == (None, False)


def test_segmented_views(sample_images):
    v = F.segmented_views(sample_images[0])
    size = config.IMG_SIZE
    assert v["image"].shape == v["leaf"].shape == v["lesions"].shape == (size, size, 3)
    assert v["mask"].shape == (size, size)
    assert (v["leaf"][v["mask"] == 0] == 255).all()                   # background painted white
