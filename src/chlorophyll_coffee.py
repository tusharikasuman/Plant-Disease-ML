"""
src/chlorophyll_coffee.py - chlorophyll (SPAD) estimation from a leaf photo, trained on REAL meter readings.

Data: ArabicaLens (Radhakrishnan, Joseph & Kumar, 2026, Zenodo, doi:10.5281/zenodo.19447467, CC BY 4.0):
425 photos of Coffea arabica leaves on a white background, each measured 5 times with a SPAD chlorophyll meter
(the average is the target, SPAD 6 - 70). Put it in data/ArabicaLens/ (coffee.xlsx + Images/cof_<n>.jpg).

The tomato dataset has no chlorophyll measurements, so the tomato app keeps showing the uncalibrated greenness
index. This model is for COFFEE leaves only: the colour -> SPAD relation is different for every crop
(Paper 2, Anderson et al. 2026, calibrates per crop).

Method (same discipline as the disease models)
  1. every photo -> the same 55 features (src/features.py: leaf segmentation, colour, indices, HSV histogram,
     texture, shape)
  2. 80 / 20 split, stratified by SPAD level; the 20 % test leaves are used ONCE, at the end
  3. on the 80 %: 10-fold cross-validation of every (feature set, model) pair
       feature sets  rgb (Paper 2: b = A X), colour (+ greenness indices), colour + HSV histogram, all 55
       models        least squares, Ridge (lambda by CV), Bayesian ridge, polynomial (degree 2, small sets only),
                     random forest, Gaussian process (Unit 5: kernel method, gives error bars)
  4. the pair with the lowest CV RMSE is refitted on all training leaves and tested on the held-out leaves

Usage (from the plant-disease-ml folder):
    python -m src.chlorophyll_coffee                          # train + evaluate, graphs, save the model
    python -m src.chlorophyll_coffee --predict data/ArabicaLens/Images/cof_26.jpg   # SPAD of one coffee leaf
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/chlorophyll_coffee.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src import features as F

DATA_DIR = config.DATA_DIR / "ArabicaLens"
SHEET = DATA_DIR / "coffee.xlsx"
IMAGES = DATA_DIR / "Images"
CACHE = config.CACHE_DIR / "chlorophyll_coffee.npz"
MODEL_PATH = config.MODELS_DIR / "chlorophyll_coffee.joblib"
SOURCE = ("ArabicaLens - Radhakrishnan, Joseph & Kumar (2026), Zenodo, doi:10.5281/zenodo.19447467, CC BY 4.0")
TEST_FRACTION = 0.2

_RGB = ["R mean", "G mean", "B mean"]
_COLOUR = _RGB + ["GLI", "G/R ratio", "ExG (normalised)"]
_HSV = [n for n in F.FEATURE_NAMES if n.startswith("HSV ")]
FEATURE_SETS = {
    "rgb (Paper 2)": _RGB,
    "colour": _COLOUR,
    "colour + HSV histogram": _COLOUR + _HSV,
    "all 55 features": list(F.FEATURE_NAMES),
}


def say(text: str = "") -> None:
    print(text, flush=True)


# ----------------------------------------------------------------------------- data
def load_readings(sheet: Path = SHEET, images: Path = IMAGES) -> pd.DataFrame:
    """One row per leaf: id, image path, the 5 SPAD readings, their mean (the target) and their spread."""
    if not Path(sheet).exists():
        raise FileNotFoundError(
            f"{sheet} not found. Download ArabicaLens from https://zenodo.org/records/19447467 "
            f"(coffee.xlsx + Images.zip) into {Path(sheet).parent}/ and unzip Images.zip there.")
    df = pd.read_excel(sheet)
    spad_cols = [c for c in df.columns if c.startswith("SPAD_") and c != "SPAD_Avg"]
    df["image_path"] = [str(Path(images) / f"{i}.jpg") for i in df["id"]]
    missing = [p for p in df["image_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} photos listed in {Path(sheet).name} are missing, e.g. {missing[0]}")
    readings = df[spad_cols].to_numpy(float)
    bad = ~np.isfinite(readings) | (readings < config.SPAD_RANGE[0]) | (readings > config.SPAD_RANGE[1])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} SPAD readings are outside {config.SPAD_RANGE}")
    out = pd.DataFrame({"id": df["id"], "image_path": df["image_path"],
                        "spad": readings.mean(axis=1), "spad_sd": readings.std(axis=1, ddof=1),
                        "healthy": df.get("Healthy/Unhealthy", pd.Series(np.nan, index=df.index))})
    return out


def leaf_features(paths: list[str], cache: Path = CACHE) -> np.ndarray:
    """55 features per photo, cached (the 4000 x 3000 photos are slow to decode)."""
    key = json.dumps({"paths": [Path(p).name for p in paths], "version": config.FEATURE_VERSION})
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            if str(d["key"]) == key:
                return d["X"]
    from src import dataset
    t = time.time()
    X, ok, _ = dataset.extract_many(list(paths), desc="coffee leaves")
    if not ok.all():
        raise ValueError(f"{int((~ok).sum())} photos could not be read")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, key=np.array(key))
    say(f"  features of {len(X)} photos extracted in {time.time() - t:.0f} s -> {cache.name}")
    return X


def split(y: np.ndarray, seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """80 / 20 split stratified by SPAD level (5 quantile bins), so the test set covers pale to dark leaves."""
    from sklearn.model_selection import train_test_split
    bins = np.digitize(y, np.quantile(y, [0.2, 0.4, 0.6, 0.8]))
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=TEST_FRACTION, stratify=bins, random_state=seed)


# ----------------------------------------------------------------------------- models
def selector(names: list[str]):
    from sklearn.compose import ColumnTransformer
    return ColumnTransformer([("cols", "passthrough", [F.FEATURE_NAMES.index(n) for n in names])])


def model_specs(names: list[str]) -> dict:
    """name -> (pipeline taking the full 55-feature vector, syllabus note)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    alphas = np.logspace(-3, 3, 25)
    std = lambda: StandardScaler()
    specs = {
        "Least squares (OLS)": (Pipeline([("sel", selector(names)), ("sc", std()), ("reg", LinearRegression())]),
                                "Unit 2: b = A X (Paper 2)"),
        "Ridge (lambda by CV)": (Pipeline([("sel", selector(names)), ("sc", std()),
                                           ("reg", RidgeCV(alphas=alphas))]),
                                 "Unit 2: L2 penalty against collinear colours"),
        "Bayesian ridge": (Pipeline([("sel", selector(names)), ("sc", std()), ("reg", BayesianRidge())]),
                           "Unit 2: prior strength from the evidence"),
        "Random forest": (Pipeline([("sel", selector(names)),
                                    ("reg", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                  random_state=config.SEED, n_jobs=1))]),
                          "Combining models: bagged trees"),
        "Gaussian process": (Pipeline([("sel", selector(names)), ("sc", std()),
                                       ("reg", GaussianProcessRegressor(
                                           ConstantKernel(1.0) * RBF(3.0) + WhiteKernel(0.1),
                                           normalize_y=True, random_state=config.SEED))]),
                             "Unit 5: kernel method with error bars"),
    }
    if len(names) <= 10:                 # degree-2 terms of 38 or 55 features would be hundreds of columns
        specs["Polynomial (degree 2) + ridge"] = (
            Pipeline([("sel", selector(names)), ("sc", std()),
                      ("poly", PolynomialFeatures(2, include_bias=False)), ("sc2", std()),
                      ("reg", RidgeCV(alphas=alphas))]),
            "Unit 2: linear model on non-linear basis functions")
    return specs


def metrics(y, p) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score
    return {"R2": float(r2_score(y, p)), "RMSE (SPAD)": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAE (SPAD)": float(mean_absolute_error(y, p))}


# ----------------------------------------------------------------------------- run
def run(reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR, readings=None, X=None) -> dict:
    import joblib
    from sklearn.base import clone
    from sklearn.model_selection import KFold, cross_val_predict

    from src.evaluate import save_table, text_table
    config.set_seeds()
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    if readings is None:
        readings = load_readings()
    if X is None:
        X = leaf_features(readings["image_path"].tolist())
    y = readings["spad"].to_numpy(float)
    tr, te = split(y)
    say(f"Coffee leaves with real SPAD readings: {len(y)} (SPAD {y.min():.1f} - {y.max():.1f}, mean {y.mean():.1f}).")
    say(f"Meter repeatability: the 5 readings of one leaf differ by SD {readings['spad_sd'].mean():.1f} SPAD on "
        f"average - no model can be more exact than that.")
    say(f"Training {len(tr)} leaves (10-fold CV chooses the model), {len(te)} held-out test leaves (used once).\n")

    # 1. 10-fold CV of every (feature set, model) on the training leaves
    cv = KFold(10, shuffle=True, random_state=config.SEED)
    rows, cv_preds = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fs, names in FEATURE_SETS.items():
            for name, (pipe, note) in model_specs(names).items():
                t = time.time()
                p = cross_val_predict(clone(pipe), X[tr], y[tr], cv=cv)
                cv_preds[(fs, name)] = p
                rows.append({"Features": fs, "# features": len(names), "Model": name, "Syllabus": note,
                             **{f"CV {k}": v for k, v in metrics(y[tr], p).items()}, "Time (s)": time.time() - t})
    cv_table = pd.DataFrame(rows).sort_values("CV RMSE (SPAD)").reset_index(drop=True)
    say("10-fold cross-validation on the training leaves (best first):")
    say(text_table(cv_table.drop(columns=["Syllabus", "Time (s)"]).head(12)))

    # 2. refit the best pair on all training leaves, test ONCE on the held-out leaves
    best = cv_table.iloc[0]
    best_fs, best_model = best["Features"], best["Model"]
    names = FEATURE_SETS[best_fs]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = clone(model_specs(names)[best_model][0]).fit(X[tr], y[tr])
        paper2 = clone(model_specs(FEATURE_SETS["rgb (Paper 2)"])["Least squares (OLS)"][0]).fit(X[tr], y[tr])
    p_test = final.predict(X[te])
    p_paper2 = paper2.predict(X[te])
    gli = X[:, F.FEATURE_NAMES.index("GLI")]
    test = pd.DataFrame([
        {"Model": f"{best_model} on {best_fs} (chosen by CV)", **metrics(y[te], p_test)},
        {"Model": "Paper 2 baseline: least squares on mean R, G, B", **metrics(y[te], p_paper2)},
        {"Model": "Always predict the training mean (no model)",
         **metrics(y[te], np.full(len(te), y[tr].mean()))},
    ])
    r_gli = float(np.corrcoef(gli, y)[0, 1])
    say(f"\nHeld-out test ({len(te)} leaves never used for training or choosing):")
    say(text_table(test))
    say(f"Correlation of the uncalibrated greenness index (GLI) with SPAD: r = {r_gli:.2f} - greenness tracks "
        "chlorophyll, but only a calibrated model turns it into SPAD units.")

    # 3. GP error bars on the test leaves (Unit 5): how honest are they?
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp = clone(model_specs(names)["Gaussian process"][0]).fit(X[tr], y[tr])
    gp_mean, gp_std = _gp_predict(gp, X[te])
    coverage = 100 * float(np.mean(np.abs(y[te] - gp_mean) <= 1.96 * gp_std))
    say(f"Gaussian process error bars: {coverage:.0f}% of held-out leaves fall inside the 95% interval "
        f"(mean +- {1.96 * gp_std.mean():.1f} SPAD).")
    lo_q, hi_q = np.quantile(y[tr], [0.1, 0.9])
    low, high = y[te] < lo_q, y[te] > hi_q
    if low.any() and high.any():
        say(f"Extremes are pulled towards the middle: leaves below SPAD {lo_q:.0f} are over-estimated by "
            f"{float(np.mean(p_test[low] - y[te][low])):.1f}, leaves above {hi_q:.0f} under-estimated by "
            f"{float(np.mean(y[te][high] - p_test[high])):.1f} SPAD on average (few training leaves are that "
            "pale/dark).")

    # 4. save the model, tables and graphs
    models_dir.mkdir(parents=True, exist_ok=True)
    tm = metrics(y[te], p_test)
    joblib.dump({"pipeline": final, "gp": gp, "model_name": best_model, "feature_set": best_fs,
                 "features": names, "crop": "coffee (Coffea arabica)", "n_train": len(tr),
                 "spad_range": (float(y.min()), float(y.max())), "test_metrics": tm, "source": SOURCE,
                 "gp_sd_typical": float(np.median(gp_std)),
                 "feature_version": config.FEATURE_VERSION}, models_dir / MODEL_PATH.name, compress=3)
    save_table(cv_table.round(4), "chlorophyll_coffee_cv", reports_dir,
               f"Coffee SPAD - 10-fold CV on {len(tr)} training leaves ({SOURCE})")
    save_table(test.round(4), "chlorophyll_coffee_test", reports_dir,
               f"Coffee SPAD - held-out test on {len(te)} leaves")
    graphs = make_graphs(y, tr, te, p_test, gp_mean, gp_std, cv_table, gli, readings, best_model, best_fs,
                         reports_dir)
    say(f"\nModel saved: {models_dir / MODEL_PATH.name}  ({best_model}, {best_fs}; coffee leaves only)")
    return {"cv": cv_table, "test": test, "best": (best_fs, best_model), "test_metrics": tm,
            "coverage": coverage, "r_gli": r_gli, "graphs": graphs, "n": len(y), "n_test": len(te),
            "meter_sd": float(readings["spad_sd"].mean())}


def _gp_predict(pipe, X):
    """Predictive mean and standard deviation from a fitted GP pipeline."""
    Z = X
    for _, step in pipe.steps[:-1]:
        Z = step.transform(Z)
    return pipe.steps[-1][1].predict(Z, return_std=True)


# ----------------------------------------------------------------------------- graphs
def make_graphs(y, tr, te, p_test, gp_mean, gp_std, cv_table, gli, readings, best_model, best_fs, reports_dir):
    import matplotlib.pyplot as plt

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    C = COLORS
    out = []

    # 1. predicted vs measured on the held-out leaves + residuals
    m = metrics(y[te], p_test)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lo, hi = 0, max(y.max(), p_test.max()) + 3
    a1.plot([lo, hi], [lo, hi], ls=":", color=C["muted"], lw=1, label="perfect prediction")
    a1.errorbar(y[te], p_test, xerr=readings["spad_sd"].to_numpy()[te], fmt="o", ms=5, color=C["aqua"],
                ecolor=C["grid"], elinewidth=1, mec="white", mew=0.8, label="held-out leaf (bar = meter SD)")
    a1.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="measured SPAD (meter, mean of 5)", ylabel="predicted SPAD (photo)",
           title=f"{best_model} on {best_fs}: held-out leaves")
    a1.set_aspect("equal")
    a1.text(0.04, 0.96, f"R² = {m['R2']:.2f}\nRMSE = {m['RMSE (SPAD)']:.1f} SPAD\nMAE = {m['MAE (SPAD)']:.1f} SPAD",
            transform=a1.transAxes, va="top", fontsize=10)
    a1.legend(loc="lower right", fontsize=8)
    a2.scatter(p_test, y[te] - p_test, s=26, color=C["aqua"], edgecolors="white", linewidths=0.8)
    a2.axhline(0, color=C["ink"], lw=0.8)
    a2.set(xlabel="predicted SPAD", ylabel="measured − predicted", title="Residuals: no pattern = no systematic error")
    fig.tight_layout()
    out.append(save_fig(fig, "spad", "1_predicted_vs_measured", reports_dir))

    # 2. model comparison: best CV RMSE of every model, per feature set
    piv = cv_table.pivot_table(index="Model", columns="Features", values="CV RMSE (SPAD)")
    piv = piv.loc[piv.min(axis=1).sort_values(ascending=False).index, list(FEATURE_SETS)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    h = 0.8 / piv.shape[1]
    colours = [C["muted"], C["blue"], C["aqua"], C["orange"]]
    for j, (fs, col) in enumerate(zip(piv.columns, colours)):
        ax.barh(np.arange(len(piv)) + (j - (piv.shape[1] - 1) / 2) * h, piv[fs].to_numpy(), h, color=col, label=fs)
    ax.set_yticks(np.arange(len(piv)), piv.index)
    ax.set(xlabel="10-fold CV RMSE (SPAD units, lower is better)",
           title="Which model and which features predict chlorophyll best?")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)
    out.append(save_fig(fig, "spad", "2_model_comparison", reports_dir))

    # 3. why calibration is needed: GLI vs SPAD, and GP error bars on the test leaves
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1.scatter(gli, y, s=14, color=C["blue"], alpha=0.6, edgecolors="none")
    a1.set(xlabel="greenness index GLI (uncalibrated)", ylabel="measured SPAD",
           title=f"Greenness tracks chlorophyll (r = {np.corrcoef(gli, y)[0, 1]:.2f}) but is not in SPAD units")
    order = np.argsort(gp_mean)
    xs = np.arange(len(order))
    a2.fill_between(xs, (gp_mean - 1.96 * gp_std)[order], (gp_mean + 1.96 * gp_std)[order], color=C["blue"],
                    alpha=0.18, label="GP 95% interval")
    a2.plot(xs, gp_mean[order], color=C["blue"], label="GP prediction")
    a2.scatter(xs, y[te][order], s=14, color=C["orange"], zorder=3, label="measured SPAD")
    a2.set(xlabel="held-out leaves, sorted by prediction", ylabel="SPAD",
           title="Gaussian process: every estimate with an error bar")
    a2.legend(fontsize=8)
    fig.tight_layout()
    out.append(save_fig(fig, "spad", "3_greenness_vs_spad_and_error_bars", reports_dir))
    return out


# ----------------------------------------------------------------------------- one photo
def predict(image, model_path: Path = MODEL_PATH) -> dict:
    """SPAD estimate (with GP error bar) for one COFFEE leaf photo."""
    import joblib
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} not found - run  python -m src.chlorophyll_coffee  first")
    b = joblib.load(model_path)
    x, seg = F.features_from_array(F.read_image(image) if isinstance(image, (str, Path)) else image)
    spad = float(b["pipeline"].predict(x[None])[0])
    mean, sd = _gp_predict(b["gp"], x[None])
    lo, hi = b["spad_range"]
    # a GP is unsure far from its training data: much wider than usual = this photo is unlike the coffee leaves
    unfamiliar = float(sd[0]) > 2 * b.get("gp_sd_typical", np.inf)
    return {"spad": spad, "gp_mean": float(mean[0]), "gp_sd": float(sd[0]), "leaf_found": seg.leaf_found,
            "unfamiliar": unfamiliar,
            "outside_range": not (lo <= spad <= hi), "model": b["model_name"], "features": b["feature_set"],
            "crop": b["crop"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chlorophyll (SPAD) from a coffee leaf photo - ArabicaLens readings.")
    ap.add_argument("--predict", metavar="PHOTO", help="estimate SPAD for one coffee leaf photo")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        if args.predict:
            r = predict(args.predict)
            say(f"{Path(args.predict).name}: chlorophyll = {r['spad']:.1f} SPAD  ({r['model']}, {r['features']})")
            say(f"Gaussian process: {r['gp_mean']:.1f} ± {1.96 * r['gp_sd']:.1f} SPAD (95% interval)")
            if r["outside_range"]:
                say("WARNING: outside the range of the training leaves - treat with care.")
            if r["unfamiliar"]:
                say("WARNING: the Gaussian process is far less sure than usual - this photo does not look like the "
                    "coffee leaves the model learned from (another crop, background or lighting?). "
                    "Do not trust this value.")
            if not r["leaf_found"]:
                say("WARNING: no leaf found in the photo - the estimate is unreliable.")
            say(f"Calibrated for {r['crop']} only.")
        else:
            run()
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
