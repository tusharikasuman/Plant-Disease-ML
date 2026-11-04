"""
config.py - every path, seed and setting used by the project lives here.

Change values here (not inside the src/ files) so that all phases stay consistent.
"""
import os
import random
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"          # put the unzipped Kaggle dataset here (or anywhere: it is auto-detected)
CACHE_DIR = PROJECT_ROOT / "cache"        # features.npz lives here
MODELS_DIR = PROJECT_ROOT / "models"      # trained models (joblib)
REPORTS_DIR = PROJECT_ROOT / "reports"    # graphs, CSV and Markdown tables
SAMPLES_DIR = PROJECT_ROOT / "samples"    # a few leaf photos for tests and the demo
SPAD_CSV = PROJECT_ROOT / "spad_readings.csv"   # real SPAD meter readings (Phase 4)

# ----------------------------------------------------------------------------- reproducibility
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Fix every random number generator we use, so results are reproducible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ----------------------------------------------------------------------------- dataset
KAGGLE_URL = "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset"
KAGGLE_SLUG = "vipoooool/new-plant-diseases-dataset"
DATA_ENV_VAR = "PLANT_DATA_DIR"           # optional: set this to the dataset folder to skip the search
MARKER_CLASS = "Tomato___healthy"         # a folder that tells us "this directory holds class folders"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Folders searched (in this order) when no dataset path is given. Depth-limited, so it stays quick.
_HOME = Path.home()
SEARCH_ROOTS = [
    DATA_DIR,
    PROJECT_ROOT,
    PROJECT_ROOT.parent,
    _HOME / "Downloads",
    _HOME / "Desktop",
    _HOME / "OneDrive" / "Desktop",
    _HOME / "Documents",
    Path("/content"),                     # Google Colab
    Path("/content/drive/MyDrive"),       # Google Drive mounted in Colab
    Path("/kaggle/input"),                # Kaggle notebooks
]
SEARCH_MAX_DEPTH = 6

# The 38 classes of the Kaggle "New Plant Diseases Dataset (Augmented)" (folder names, sorted).
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def pretty_class(name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato: Early blight' (for tables and graphs)."""
    plant, _, disease = name.partition("___")
    plant = plant.replace("_", " ").replace(",", "")
    disease = disease.replace("_", " ").strip()
    return f"{plant}: {disease}" if disease else plant


def is_healthy(name: str) -> bool:
    return name.lower().endswith("healthy")


# ----------------------------------------------------------------------------- preprocessing
IMG_SIZE = 256                  # every image is resized to IMG_SIZE x IMG_SIZE
BLUR_KSIZE = (3, 3)             # Gaussian blur kernel
CLOSE_KERNEL = 7                # morphological closing kernel size (pixels)
# OpenCV HSV ranges: H in 0..179, S and V in 0..255
GREEN_LO, GREEN_HI = (25, 30, 30), (95, 255, 255)   # healthy leaf tissue (pale leaf edges have S ~ 30)
BROWN_LO, BROWN_HI = (5, 70, 20), (24, 255, 255)    # lesions / dead tissue. S >= 70 so beige/grey backgrounds
                                                    # (S ~ 30-50) are not brown; H stops at 24 so no pixel is both
SHADOW_V_MAX = 60               # brown pixels OUTSIDE the green leaf outline darker than this are shadows
MIN_LEAF_FRACTION = 0.01       # if the leaf mask covers < 1 % of the image, fall back to the whole image

# ----------------------------------------------------------------------------- features
HSV_BINS = (8, 2, 2)            # hue x saturation x value histogram = 32 numbers
GLCM_LEVELS = 32                # grey levels for the co-occurrence matrix
GLCM_DISTANCES = (1, 2)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
FEATURE_VERSION = "v2"          # bump this when features.py changes, so old caches are rebuilt

# ----------------------------------------------------------------------------- classification (Phase 2)
CV_FOLDS = 5                    # stratified k-fold cross-validation
CV_PER_CLASS = 500              # leaves per class used for grid search + CV (keeps 38 classes fast)
SVM_MAX_PER_CLASS = 1000        # kernel SVM training cost grows ~n^2: cap the final fit per class
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
LOGREG_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"   # interpretable model for explanations
BAYES_MODEL_PATH = MODELS_DIR / "bayes_logreg.joblib"            # healthy vs diseased, with uncertainty
BAYES_REFER_BELOW = 0.90       # Bayesian healthy/diseased confidence below this -> "refer to an expert"
BAYES_ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]   # prior precisions tried (evidence)
CHLOROPHYLL_MODEL_PATH = MODELS_DIR / "chlorophyll_model.joblib"  # only exists after real SPAD calibration

# ----------------------------------------------------------------------------- chlorophyll (Phase 4)
# Paper 2 fits b = A X from leaf colour X to the SPAD meter value b
CHLORO_FEATURE_SETS = {
    "rgb": ["R mean", "G mean", "B mean"],                                          # Paper 2's linear model
    "colour": ["R mean", "G mean", "B mean", "GLI", "G/R ratio", "ExG (normalised)"],
}
CHLORO_DEFAULT_FEATURES = "rgb"
MIN_SPAD_LEAVES = 10            # refuse to train on fewer real readings than this
LOOCV_MAX_LEAVES = 60           # leave-one-out CV up to this many leaves, 10-fold above
SPAD_RANGE = (0.0, 100.0)       # plausible SPAD meter values; anything outside is a typing error

LOW_CONFIDENCE = 0.6            # below this the app warns that the prediction is uncertain

# ----------------------------------------------------------------------------- parallelism
# leave one core free; cap at 8 workers so 8 GB laptops are fine
N_JOBS = max(1, min(8, (os.cpu_count() or 2) - 1))
