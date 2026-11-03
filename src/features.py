"""
src/features.py - Phase 1: preprocessing + hand-made feature extraction.

Adapted from the tested extractor in legacy/ml_unit1_unit2_proof.py (feats / leaf_mask).

Pipeline for one image
  1. resize to 256 x 256
  2. Gaussian blur 3 x 3 (removes JPEG noise)
  3. HSV threshold: green pixels (healthy tissue) OR brown pixels (lesions)  -> raw mask
  4. morphological closing (fills small gaps)
  5. keep the largest connected component            -> leaf mask
Every feature below is then computed from LEAF PIXELS ONLY, and every feature is one explainable number:

  group     count  features
  rgb         6    mean and std of R, G, B
  indices     4    GLI, G/R ratio, ExG (normalised), lesion fraction (brown / leaf pixels)
  hsv        32    HSV colour histogram, 8 hue x 2 saturation x 2 value bins, sums to 1
  texture     6    Haralick / GLCM: contrast, dissimilarity, homogeneity, energy, correlation, ASM
  shape       7    Hu moments of the leaf mask (log-scaled)
  total      55
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

import config

# ----------------------------------------------------------------------------- feature names
RGB_NAMES = ["R mean", "G mean", "B mean", "R std", "G std", "B std"]
INDEX_NAMES = ["GLI", "G/R ratio", "ExG (normalised)", "Lesion fraction"]
HSV_NAMES = [f"HSV h{h} s{s} v{v}"
             for h in range(config.HSV_BINS[0]) for s in range(config.HSV_BINS[1]) for v in range(config.HSV_BINS[2])]
TEXTURE_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
TEXTURE_NAMES = [f"GLCM {p}" for p in TEXTURE_PROPS]
SHAPE_NAMES = [f"Hu moment {i}" for i in range(1, 8)]

FEATURE_NAMES = RGB_NAMES + INDEX_NAMES + HSV_NAMES + TEXTURE_NAMES + SHAPE_NAMES
N_FEATURES = len(FEATURE_NAMES)


def _group_slices() -> dict[str, list[int]]:
    groups, start = {}, 0
    for name, names in [("rgb", RGB_NAMES), ("indices", INDEX_NAMES), ("hsv", HSV_NAMES),
                        ("texture", TEXTURE_NAMES), ("shape", SHAPE_NAMES)]:
        groups[name] = list(range(start, start + len(names)))
        start += len(names)
    return groups


# column indices of each feature group (used by the Phase 2 ablation study)
FEATURE_GROUPS = _group_slices()
LESION_INDEX = FEATURE_NAMES.index("Lesion fraction")
GLI_INDEX = FEATURE_NAMES.index("GLI")


# ----------------------------------------------------------------------------- image loading
def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image as BGR uint8. Works with non-ASCII Windows paths. Returns None if unreadable."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode image bytes (e.g. a Streamlit upload) to BGR uint8."""
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZE x IMG_SIZE and apply the 3x3 Gaussian blur."""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    img = cv2.resize(img_bgr, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, config.BLUR_KSIZE, 0)


# ----------------------------------------------------------------------------- segmentation
@dataclass
class Segmentation:
    mask: np.ndarray     # uint8, 255 = leaf pixel
    brown: np.ndarray    # uint8, 255 = brown (lesion) pixel inside the leaf
    leaf_found: bool     # False -> no leaf detected, the whole image was used instead


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected white region of a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:                                   # only background
        return np.zeros_like(mask)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions completely enclosed by the leaf (dark or grey lesions inside the leaf)."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 255)          # everything reachable from outside the leaf
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def convex_hull(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros_like(mask)
    if contours:
        cv2.drawContours(hull, [cv2.convexHull(np.vstack(contours))], -1, 255, -1)
    return hull


def segment_leaf(img: np.ndarray) -> Segmentation:
    """HSV green + brown threshold -> closing -> largest connected component -> fill enclosed holes.

    Shadow rejection: the leaf's shadow on a beige background also looks "brown". Brown pixels are
    therefore kept only if they lie inside the convex hull of the green leaf, or are bright enough
    (V >= SHADOW_V_MAX) to be dead leaf tissue rather than shadow.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, config.GREEN_LO, config.GREEN_HI)
    brown = cv2.inRange(hsv, config.BROWN_LO, config.BROWN_HI)
    kernel = np.ones((config.CLOSE_KERNEL, config.CLOSE_KERNEL), np.uint8)
    close = lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    green_leaf = fill_holes(largest_component(close(green)))
    if np.count_nonzero(green_leaf) >= config.MIN_LEAF_FRACTION * green_leaf.size:
        bright = np.where(hsv[..., 2] >= config.SHADOW_V_MAX, 255, 0).astype(np.uint8)
        brown = cv2.bitwise_and(brown, cv2.bitwise_or(convex_hull(green_leaf), bright))
        mask = fill_holes(largest_component(close(cv2.bitwise_or(green_leaf, brown))))
    else:                                        # (almost) no green: e.g. a fully dead brown leaf
        mask = fill_holes(largest_component(close(cv2.bitwise_or(green, brown))))

    leaf_found = np.count_nonzero(mask) >= config.MIN_LEAF_FRACTION * mask.size
    if not leaf_found:                           # nothing leaf-like: use the whole image, never crash
        mask = np.full_like(mask, 255)
    return Segmentation(mask=mask, brown=cv2.bitwise_and(brown, mask), leaf_found=leaf_found)


# ----------------------------------------------------------------------------- feature groups
def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b, with 0 wherever b == 0 (no divide-by-zero warnings, no inf)."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def colour_features(img: np.ndarray, seg: Segmentation) -> list[float]:
    """RGB mean/std (6) + GLI, G/R ratio, normalised ExG, lesion fraction (4)."""
    px = img[seg.mask > 0].astype(np.float64)          # OpenCV order: B, G, R
    B, G, R = px[:, 0], px[:, 1], px[:, 2]
    rgb = [R.mean(), G.mean(), B.mean(), R.std(), G.std(), B.std()]

    # Green Leaf Index, per pixel, range -1..1 (1 = pure green)
    gli = _safe_divide(2 * G - R - B, 2 * G + R + B).mean()
    # G/R ratio of the channel means (a per-pixel ratio explodes when R is 0)
    g_over_r = G.mean() / max(R.mean(), 1.0)
    # Excess Green on chromatic coordinates r = R/(R+G+B) ..., so it ignores brightness; range -1..2
    total = R + G + B
    exg = (2 * _safe_divide(G, total) - _safe_divide(R, total) - _safe_divide(B, total)).mean()
    lesion = np.count_nonzero(seg.brown) / max(np.count_nonzero(seg.mask), 1)
    return rgb + [gli, g_over_r, exg, lesion]


def hsv_histogram(img: np.ndarray, seg: Segmentation) -> list[float]:
    """8x2x2 HSV histogram of leaf pixels, normalised to sum to 1 (32 numbers)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], seg.mask, list(config.HSV_BINS), [0, 180, 0, 256, 0, 256]).flatten()
    return list(hist / max(hist.sum(), 1e-9))


def texture_features(img: np.ndarray, seg: Segmentation) -> list[float]:
    """Haralick (GLCM) texture, leaf pixels only, averaged over distances 1, 2 and 4 angles (6 numbers).

    Leaf grey levels are mapped to 1..32 and the background to 0. After counting pixel pairs we drop
    row/column 0, so pairs that touch the background (the leaf edge) do not count as texture.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    q = (gray.astype(np.uint16) * config.GLCM_LEVELS // 256 + 1).astype(np.uint8)   # 1..32
    q[seg.mask == 0] = 0
    glcm = graycomatrix(q, distances=list(config.GLCM_DISTANCES), angles=list(config.GLCM_ANGLES),
                        levels=config.GLCM_LEVELS + 1, symmetric=True, normed=False)
    glcm = glcm[1:, 1:].astype(np.float64)
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = _safe_divide(glcm, np.broadcast_to(sums, glcm.shape))
    return [float(np.nanmean(graycoprops(glcm, p))) for p in TEXTURE_PROPS]


def shape_features(seg: Segmentation) -> list[float]:
    """7 Hu moments of the leaf mask, log-scaled: -sign(h) * log10|h| (0 stays 0)."""
    hu = cv2.HuMoments(cv2.moments(seg.mask, binaryImage=True)).flatten()
    scaled = np.zeros_like(hu)
    nonzero = hu != 0
    scaled[nonzero] = -np.sign(hu[nonzero]) * np.log10(np.abs(hu[nonzero]))
    return list(scaled)


# ----------------------------------------------------------------------------- public API
def features_from_array(img_bgr: np.ndarray) -> tuple[np.ndarray, Segmentation]:
    """Full pipeline on a BGR image (any size). Returns (55 features, segmentation)."""
    img = preprocess(img_bgr)
    seg = segment_leaf(img)
    vec = (colour_features(img, seg) + hsv_histogram(img, seg)
           + texture_features(img, seg) + shape_features(seg))
    vec = np.nan_to_num(np.asarray(vec, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    assert vec.shape == (N_FEATURES,)
    return vec, seg


def extract_features(image: str | Path | np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Features for one image (a file path or a BGR array). Returns (feature vector, feature names)."""
    img = read_image(image) if isinstance(image, (str, Path)) else image
    if img is None:
        raise ValueError(f"Could not read image: {image}")
    vec, _ = features_from_array(img)
    return vec, list(FEATURE_NAMES)


def extract_for_dataset(path: str) -> tuple[np.ndarray | None, bool]:
    """Worker used by src/dataset.py: (features, leaf_found), or (None, False) if the file is unreadable."""
    try:
        img = read_image(path)
        if img is None:
            return None, False
        vec, seg = features_from_array(img)
        return vec, seg.leaf_found
    except Exception:          # one corrupt file must never stop a 90k-image run
        return None, False


def segmented_views(image: str | Path | np.ndarray) -> dict[str, np.ndarray]:
    """RGB views for display (Streamlit app / notebook): preprocessed image, mask, segmented leaf, lesions."""
    img = read_image(image) if isinstance(image, (str, Path)) else image
    img = preprocess(img)
    seg = segment_leaf(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    leaf = rgb.copy()
    leaf[seg.mask == 0] = 255                    # white background
    lesions = leaf.copy()
    lesions[seg.brown > 0] = (230, 159, 0)       # highlight brown pixels in orange (colour-blind safe)
    return {"image": rgb, "mask": seg.mask, "leaf": leaf, "lesions": lesions, "leaf_found": seg.leaf_found}
