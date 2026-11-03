"""
src/dataset.py - Phase 1: find the dataset, extract features for every image, cache them.

Usage (from the plant-disease-ml folder):
    python -m src.dataset --limit 50          # quick test: 50 images per class
    python -m src.dataset                     # every image (~88k, a few minutes with multiprocessing)
    python -m src.dataset --data "D:/datasets/New Plant Diseases Dataset(Augmented)"
    python -m src.dataset --find              # only show which dataset folder was detected

Result: cache/features.npz (all images) or cache/features_limit50.npz (with --limit 50), containing
X_train, y_train, paths_train, X_valid, y_valid, paths_valid, class_names, feature_names and metadata.
Extraction only runs once; later runs load the cache unless --force is given or settings changed.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/dataset.py` as well as `python -m src.dataset`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from tqdm.auto import tqdm

import config
from src import features

SKIP_DIRS = {"node_modules", "__pycache__", "site-packages", "AppData", "venv", ".venv", "env",
             "Library", "Program Files", "Windows", "$Recycle.Bin"}


class DatasetNotFoundError(FileNotFoundError):
    pass


DOWNLOAD_INSTRUCTIONS = f"""
==========================================================================================
 Dataset not found.

 This project uses the Kaggle "New Plant Diseases Dataset (Augmented)" (38 classes, ~88k images):
     {config.KAGGLE_URL}

 Option A - browser (Windows):
   1. Sign in to Kaggle, open the link above and click "Download" (about 2.7 GB zip).
   2. Unzip it into:  {config.DATA_DIR}
      You should end up with ...\\New Plant Diseases Dataset(Augmented)\\train\\Tomato___healthy\\...
   3. Run again:      python -m src.dataset --limit 50

 Option B - Kaggle command line (Windows or Colab):
   pip install kaggle
   (put your kaggle.json API token in ~/.kaggle/ - Kaggle > Settings > API > Create New Token)
   kaggle datasets download -d {config.KAGGLE_SLUG} -p data --unzip

 Option C - Google Colab:
   from google.colab import files; files.upload()   # upload kaggle.json
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d {config.KAGGLE_SLUG} -p data --unzip

 Already downloaded somewhere else? Point to it directly:
   python -m src.dataset --data "D:\\path\\to\\New Plant Diseases Dataset(Augmented)"
   or set the environment variable {config.DATA_ENV_VAR} to that folder.
==========================================================================================
"""


# ----------------------------------------------------------------------------- finding the dataset
@dataclass
class DatasetLocation:
    train_dir: Path              # folder whose sub-folders are the classes
    valid_dir: Path | None       # matching valid/ folder, or None (then we make our own split)
    classes: list[str]           # class folder names, sorted (label i = classes[i])

    def describe(self) -> str:
        n_known = len(set(self.classes) & set(config.CLASS_NAMES))
        lines = [f"Dataset (train): {self.train_dir}",
                 f"Dataset (valid): {self.valid_dir or 'none - a stratified 80/20 split of train will be used'}",
                 f"Classes found:   {len(self.classes)} ({n_known} of the expected {len(config.CLASS_NAMES)})"]
        if len(self.classes) < len(config.CLASS_NAMES):
            lines.append("NOTE: this looks like a subset of the full 38-class dataset.")
        return "\n".join(lines)


def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in config.IMAGE_EXTENSIONS


def _class_folders(root: Path) -> list[str]:
    """Sub-folders of root that contain at least one image, sorted."""
    out = []
    for entry in os.scandir(root):
        if entry.is_dir() and any(_is_image(e.name) for e in os.scandir(entry.path) if e.is_file()):
            out.append(entry.name)
    return sorted(out)


def _search(roots: list[Path], max_depth: int) -> list[Path]:
    """Every folder (up to max_depth below a root) that contains a MARKER_CLASS sub-folder."""
    found, seen = [], set()
    for root in roots:
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            folder, depth = stack.pop()
            try:
                key = folder.resolve()
            except OSError:
                continue
            if key in seen:
                continue
            seen.add(key)
            if (folder / config.MARKER_CLASS).is_dir():
                found.append(key)
                continue                         # never walk into the class folders themselves
            if depth >= max_depth:
                continue
            try:
                subdirs = [Path(e.path) for e in os.scandir(folder)
                           if e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIRS
                           and not e.name.startswith(".")]
            except OSError:
                continue
            stack.extend((d, depth + 1) for d in subdirs)
    return found


def _sibling(folder: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        candidate = folder.parent / n
        if (candidate / config.MARKER_CLASS).is_dir():
            return candidate
    return None


def find_dataset(data_dir: str | Path | None = None, verbose: bool = True) -> DatasetLocation | None:
    """Locate the dataset. Priority: data_dir argument > PLANT_DATA_DIR env var > search config.SEARCH_ROOTS.

    Among all candidates we prefer a train/ folder with a matching valid/ folder, then the most classes.
    """
    explicit = data_dir or os.environ.get(config.DATA_ENV_VAR)
    roots = [Path(explicit)] if explicit else config.SEARCH_ROOTS
    candidates = _search(roots, max_depth=config.SEARCH_MAX_DEPTH)

    best, best_score = None, None
    for folder in candidates:
        if folder.name.lower() in ("valid", "val", "validation", "test") and \
                _sibling(folder, ("train", "Train")) is not None:
            continue                             # handled through its train/ sibling
        valid = _sibling(folder, ("valid", "val", "validation")) if folder.name.lower() == "train" else None
        classes = _class_folders(folder)
        score = (valid is not None, len(classes))
        if best_score is None or score > best_score:
            best, best_score = DatasetLocation(folder, valid, classes), score

    if verbose:
        print(best.describe() if best else DOWNLOAD_INSTRUCTIONS)
    return best


# ----------------------------------------------------------------------------- listing images
def list_images(root: Path, classes: list[str], limit: int | None, seed: int = config.SEED):
    """(paths, labels) for every class folder; with limit, a reproducible random sample per class."""
    paths, labels = [], []
    for label, cls in enumerate(classes):
        folder = root / cls
        if not folder.is_dir():
            continue
        files = sorted(e.path for e in os.scandir(folder) if e.is_file() and _is_image(e.name))
        if limit is not None and len(files) > limit:
            files = sorted(random.Random(f"{seed}-{cls}").sample(files, limit))
        paths += files
        labels += [label] * len(files)
    return paths, np.array(labels, dtype=np.int64)


def _stratified_split(paths: list[str], labels: np.ndarray, valid_fraction: float = 0.2, seed: int = config.SEED):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(paths))
    tr, va = train_test_split(idx, test_size=valid_fraction, stratify=labels, random_state=seed)
    pick = lambda ids: ([paths[i] for i in ids], labels[ids])
    return pick(np.sort(tr)), pick(np.sort(va))


# ----------------------------------------------------------------------------- extraction
def _init_worker() -> None:
    import cv2
    cv2.setNumThreads(1)                         # one OpenCV thread per process: no oversubscription


def extract_many(paths: list[str], n_jobs: int = config.N_JOBS, desc: str = "features"):
    """Extract features for many images in parallel (with a tqdm progress bar).

    Returns (X, ok, leaf_found): X has one row per readable image; ok marks which input paths were readable.
    """
    if n_jobs <= 1 or len(paths) < 50:
        _init_worker()
        results = [features.extract_for_dataset(p) for p in tqdm(paths, desc=desc, unit="img")]
    else:
        chunk = max(1, min(64, len(paths) // (n_jobs * 8)))
        with mp.get_context().Pool(n_jobs, initializer=_init_worker) as pool:
            results = list(tqdm(pool.imap(features.extract_for_dataset, paths, chunksize=chunk),
                                total=len(paths), desc=desc, unit="img"))
    ok = np.array([r[0] is not None for r in results], dtype=bool)
    X = np.array([r[0] for r in results if r[0] is not None], dtype=np.float64).reshape(-1, features.N_FEATURES)
    leaf_found = np.array([r[1] for r in results if r[0] is not None], dtype=bool)
    return X, ok, leaf_found


def cache_path(limit: int | None, cache_dir: Path = config.CACHE_DIR) -> Path:
    return Path(cache_dir) / ("features.npz" if limit is None else f"features_limit{limit}.npz")


def _metadata(loc: DatasetLocation, limit: int | None) -> dict:
    return {"feature_version": config.FEATURE_VERSION, "limit": limit, "seed": config.SEED,
            "train_dir": str(loc.train_dir), "valid_dir": str(loc.valid_dir) if loc.valid_dir else None,
            "classes": loc.classes}


def build_feature_cache(loc: DatasetLocation, limit: int | None = None, n_jobs: int = config.N_JOBS,
                        force: bool = False, cache_dir: Path = config.CACHE_DIR) -> Path:
    """Extract features for train + valid images and save them to an .npz cache (skipped if up to date)."""
    out = cache_path(limit, cache_dir)
    meta = _metadata(loc, limit)
    if out.exists() and not force:
        with np.load(out, allow_pickle=False) as d:
            if json.loads(str(d["meta"])) == meta:
                print(f"Cache is up to date: {out}  (use --force to rebuild)")
                return out
        print("Cache settings changed - rebuilding.")

    paths_tr, y_tr = list_images(loc.train_dir, loc.classes, limit)
    if loc.valid_dir is not None:
        paths_va, y_va = list_images(loc.valid_dir, loc.classes, limit)
        valid_source = "valid folder"
    else:
        (paths_tr, y_tr), (paths_va, y_va) = _stratified_split(paths_tr, y_tr)
        valid_source = "stratified 80/20 split of train"
    print(f"Images: {len(paths_tr)} train + {len(paths_va)} valid ({valid_source}), "
          f"{len(loc.classes)} classes, {n_jobs} worker(s)")

    # one pass over train + valid, so the worker pool only starts once (slow on Windows)
    t0 = time.perf_counter()
    X_all, ok_all, leaf_all = extract_many(paths_tr + paths_va, n_jobs, desc="extracting")
    seconds = time.perf_counter() - t0
    ok_tr, ok_va = ok_all[:len(paths_tr)], ok_all[len(paths_tr):]
    n_tr = int(ok_tr.sum())
    X_tr, X_va = X_all[:n_tr], X_all[n_tr:]
    leaf_tr, leaf_va = leaf_all[:n_tr], leaf_all[n_tr:]

    bad = [p for p, ok in zip(paths_tr + paths_va, ok_all) if not ok]
    if bad:
        print(f"WARNING: skipped {len(bad)} unreadable image(s), e.g. {bad[0]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_train=X_tr, y_train=y_tr[ok_tr], paths_train=np.array(paths_tr)[ok_tr], leaf_found_train=leaf_tr,
        X_valid=X_va, y_valid=y_va[ok_va], paths_valid=np.array(paths_va)[ok_va], leaf_found_valid=leaf_va,
        class_names=np.array(loc.classes), feature_names=np.array(features.FEATURE_NAMES),
        valid_source=np.array(valid_source), meta=np.array(json.dumps(meta)),
    )
    n = len(X_tr) + len(X_va)
    print(f"Extracted {n} images in {seconds:.1f} s ({1000 * seconds / max(n, 1):.1f} ms/image). "
          f"No leaf found (whole image used) in {int((~leaf_tr).sum() + (~leaf_va).sum())} image(s).")
    print(f"Saved {out}")
    return out


def load_features(limit: int | None = None, cache_dir: Path = config.CACHE_DIR) -> dict:
    """Load a cache built by build_feature_cache as a plain dict of arrays / lists."""
    path = cache_path(limit, cache_dir)
    if not path.exists():
        hint = "" if limit is None else f" --limit {limit}"
        raise FileNotFoundError(f"{path} not found. Build it first:  python -m src.dataset{hint}")
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["class_names"] = [str(c) for c in data["class_names"]]
    data["feature_names"] = [str(f) for f in data["feature_names"]]
    data["valid_source"] = str(data["valid_source"])
    data["meta"] = json.loads(str(data["meta"]))
    return data


# ----------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find the dataset and cache hand-made features for every image.")
    ap.add_argument("--data", help="dataset folder (default: auto-detect)")
    ap.add_argument("--limit", type=int, default=None, help="images per class (quick test), default: all")
    ap.add_argument("--jobs", type=int, default=config.N_JOBS, help=f"worker processes (default {config.N_JOBS})")
    ap.add_argument("--force", action="store_true", help="re-extract even if the cache is up to date")
    ap.add_argument("--find", action="store_true", help="only report which dataset folder is found")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be at least 1")

    config.set_seeds()
    loc = find_dataset(args.data)
    if loc is None:
        return 1
    if args.find:
        return 0
    build_feature_cache(loc, limit=args.limit, n_jobs=args.jobs, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
