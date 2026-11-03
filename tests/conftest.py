import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def make_leaf(spots: bool = True, size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic BGR leaf: green ellipse on a grey background, with optional brown lesion spots.

    Returns (image, true leaf mask, true lesion mask).
    """
    img = np.full((size, size, 3), 128, np.uint8)                      # neutral grey (saturation 0)
    leaf = np.zeros((size, size), np.uint8)
    cv2.ellipse(leaf, (size // 2, size // 2), (size // 3, size // 4), 30, 0, 360, 255, -1)
    img[leaf > 0] = (40, 150, 50)                                       # green (OpenCV BGR)
    lesion = np.zeros_like(leaf)
    if spots:
        for cx, cy, r in [(110, 120, 12), (150, 140, 9), (130, 100, 7)]:
            cv2.circle(lesion, (cx, cy), r, 255, -1)
        lesion &= leaf
        img[lesion > 0] = (30, 70, 140)                                 # brown, HSV hue ~11
    return img, leaf, lesion


@pytest.fixture
def synthetic_leaf():
    return make_leaf()


@pytest.fixture
def sample_images():
    paths = sorted(config.SAMPLES_DIR.glob("*.jpg"))
    assert paths, "samples/ folder has no .jpg leaf photos"
    return paths
