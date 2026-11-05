"""
src/chlorophyll.py - Phase 4: chlorophyll (SPAD) estimation from leaf colour, following Paper 2
(Anderson et al., 2026): a linear model  b = A X  from colour features X to the SPAD meter value b.

IMPORTANT - no fake data.
  A model is trained ONLY from real SPAD meter readings in spad_readings.csv
  (columns: image_path, spad_value). Until that file holds at least MIN_SPAD_LEAVES real readings,
  no chlorophyll model exists, and predict.py / the app show only the uncalibrated greenness index.
  See docs/spad_collection.md for how to collect the readings.

What runs without SPAD data
  The RGB correlation matrix and the variance inflation factors (VIF), computed on the real leaf features
  of the disease dataset. They show that R, G and B are strongly collinear, which is why plain least squares
  (OLS) gives unstable coefficients and Ridge regression is needed (Unit 1 covariance -> Unit 2 ridge).

What runs once spad_readings.csv has data
  OLS, Ridge (lambda by cross-validation), BayesianRidge, polynomial regression (degree 2) and Random Forest,
  evaluated by leave-one-out CV (<= LOOCV_MAX_LEAVES leaves) or 10-fold CV: R^2, RMSE, MAE, MAPE.
  The best model (lowest CV RMSE) is saved and predict.py / the app then show a calibrated SPAD value.

Usage (from the plant-disease-ml folder):
    python -m src.chlorophyll                       # uses spad_readings.csv (creates the template if missing)
    python -m src.chlorophyll --csv my_readings.csv --features colour
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):          # allow `python src/chlorophyll.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config
from src import features as F

TEMPLATE_COLUMNS = ["image_path", "spad_value", "crop", "leaf_id", "date", "notes"]


# ----------------------------------------------------------------------------- reading the CSV
def write_template(path: Path = config.SPAD_CSV) -> Path:
    """Create an EMPTY spad_readings.csv with the right columns (header only - no example values)."""
    path = Path(path)
    if not path.exists():
        path.write_text(",".join(TEMPLATE_COLUMNS) + "\n", encoding="utf-8")
    return path


def load_readings(path: Path = config.SPAD_CSV) -> pd.DataFrame:
    """Read and validate real SPAD readings. Several readings of the same photo are averaged.

    Returns a DataFrame with image_path (absolute), spad_value (mean), n_readings, crop. Empty if no data.
    Raises ValueError with a clear message for impossible values or missing images.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["image_path", "spad_value", "n_readings", "crop"])
    df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    missing = {"image_path", "spad_value"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} must have the columns image_path and spad_value (missing: {missing})")
    df = df.dropna(subset=["image_path", "spad_value"])
    if df.empty:
        return pd.DataFrame(columns=["image_path", "spad_value", "n_readings", "crop"])

    df["spad_value"] = pd.to_numeric(df["spad_value"], errors="coerce")
    bad = df[df["spad_value"].isna() | ~df["spad_value"].between(*config.SPAD_RANGE)]
    if len(bad):
        raise ValueError(f"{len(bad)} row(s) in {path.name} have a spad_value that is not a number between "
                         f"{config.SPAD_RANGE[0]:g} and {config.SPAD_RANGE[1]:g}, e.g. row: {bad.iloc[0].to_dict()}")

    base = path.parent
    df["image_path"] = [str(p if Path(p).is_absolute() else (base / p).resolve()) for p in df["image_path"]]
    not_found = [p for p in df["image_path"].unique() if not Path(p).exists()]
    if not_found:
        raise ValueError(f"{len(not_found)} image(s) listed in {path.name} were not found, e.g. {not_found[0]}")

    if "crop" not in df.columns:
        df["crop"] = "unknown"
    df["crop"] = df["crop"].fillna("unknown").astype(str)
    return (df.groupby("image_path", as_index=False)
              .agg(spad_value=("spad_value", "mean"), n_readings=("spad_value", "size"), crop=("crop", "first")))


def feature_matrix(image_paths) -> np.ndarray:
    """All 55 features for each image (the chlorophyll pipeline selects the colour columns itself)."""
    return np.array([F.extract_features(p)[0] for p in image_paths])


# ----------------------------------------------------------------------------- collinearity (no SPAD needed)
def variance_inflation_factors(X: np.ndarray) -> np.ndarray:
    """VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing column j on all the other columns.

    VIF = 1: independent; > 5-10: strong multicollinearity (OLS coefficients become unstable).
    """
    from sklearn.linear_model import LinearRegression
    vif = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(others, X[:, j]).score(others, X[:, j])
        vif.append(np.inf if r2 >= 1 else 1.0 / (1.0 - r2))
    return np.array(vif)


def collinearity_report(X: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr = pd.DataFrame(np.corrcoef(X.T), index=names, columns=names)
    vif = pd.DataFrame({"Feature": names, "VIF": variance_inflation_factors(X)})
    return corr, vif


# ----------------------------------------------------------------------------- models
def column_selector(feature_set: str):
    """Pipeline step that picks the colour columns out of the full 55-feature vector."""
    from sklearn.compose import ColumnTransformer
    idx = [F.FEATURE_NAMES.index(n) for n in config.CHLORO_FEATURE_SETS[feature_set]]
    return ColumnTransformer([("colour", "passthrough", idx)])


def model_specs(feature_set: str) -> dict:
    """name -> (unpretrained pipeline, syllabus note). Every pipeline takes the full 55-feature vector."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import BayesianRidge, LinearRegression, RidgeCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    sel = lambda: column_selector(feature_set)
    alphas = np.logspace(-3, 3, 25)
    return {
        "OLS (least squares)": (Pipeline([("sel", sel()), ("scale", StandardScaler()), ("reg", LinearRegression())]),
                                "Unit 2: least squares, b = A X (Paper 2)"),
        "Ridge (lambda by CV)": (Pipeline([("sel", sel()), ("scale", StandardScaler()), ("reg", RidgeCV(alphas=alphas))]),
                                 "Unit 2: L2 regularisation against multicollinearity"),
        "Bayesian Ridge": (Pipeline([("sel", sel()), ("scale", StandardScaler()), ("reg", BayesianRidge())]),
                           "Unit 2: Bayesian linear regression (prior learned from the evidence)"),
        "Polynomial (degree 2)": (Pipeline([("sel", sel()), ("scale", StandardScaler()),
                                            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                                            ("reg", LinearRegression())]),
                                  "Unit 2: linear model on non-linear basis functions"),
        "Random Forest": (Pipeline([("sel", sel()), ("reg", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                                                    random_state=config.SEED))]),
                          "Baseline: non-linear ensemble"),
    }


def cv_strategy(n: int):
    from sklearn.model_selection import KFold, LeaveOneOut
    if n <= config.LOOCV_MAX_LEAVES:
        return LeaveOneOut(), "leave-one-out"
    return KFold(n_splits=10, shuffle=True, random_state=config.SEED), "10-fold"


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"R2": r2_score(y, pred),
            "RMSE (SPAD)": float(np.sqrt(mean_squared_error(y, pred))),
            "MAE (SPAD)": mean_absolute_error(y, pred),
            "MAPE (%)": 100 * float(np.mean(np.abs((y - pred) / np.maximum(np.abs(y), 1e-9))))}


def coefficient_stability(X: np.ndarray, y: np.ndarray, feature_set: str, n_boot: int = 300,
                          subsample: float = 0.7) -> dict[str, np.ndarray]:
    """Refit OLS and Ridge on many random subsets; the spread of each coefficient = its instability."""
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.preprocessing import StandardScaler
    idx = [F.FEATURE_NAMES.index(n) for n in config.CHLORO_FEATURE_SETS[feature_set]]
    Z = StandardScaler().fit_transform(X[:, idx])
    alpha = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Z, y).alpha_
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(config.SEED)
    m = max(len(idx) + 2, int(subsample * len(y)))
    ols, ridge = [], []
    for _ in range(n_boot):
        s = rng.choice(len(y), size=m, replace=False)
        ols.append(LinearRegression().fit(Z[s], y[s]).coef_)
        ridge.append(Ridge(alpha=alpha).fit(Z[s], y[s]).coef_)
    return {"ols": np.array(ols), "ridge": np.array(ridge), "alpha": float(alpha)}


def train(readings: pd.DataFrame, X: np.ndarray, feature_set: str = config.CHLORO_DEFAULT_FEATURES,
          reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR) -> pd.DataFrame:
    """Cross-validate every model on REAL readings, save the best one, write tables and graphs."""
    import joblib
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    from src.evaluate import save_table, text_table

    y = readings["spad_value"].to_numpy(dtype=float)
    n = len(y)
    if n < config.MIN_SPAD_LEAVES:
        raise ValueError(f"Only {n} leaves with SPAD readings - collect at least {config.MIN_SPAD_LEAVES} "
                         "(30-50 is much better) before training a chlorophyll model.")
    cv, cv_name = cv_strategy(n)
    crops = sorted(readings["crop"].unique())
    print(f"{n} leaves, SPAD {y.min():.1f}-{y.max():.1f} (mean {y.mean():.1f}), crops: {', '.join(crops)}. "
          f"Features: {feature_set} ({', '.join(config.CHLORO_FEATURE_SETS[feature_set])}). CV: {cv_name}.")
    if len(crops) > 1:
        print("NOTE: several crops in one model - the colour->SPAD relation differs per crop; "
              "consider training one model per crop.")

    rows, preds = [], {}
    for name, (pipe, note) in model_specs(feature_set).items():
        p = cross_val_predict(clone(pipe), X, y, cv=cv)
        preds[name] = p
        rows.append({"Model": name, "Syllabus": note, **regression_metrics(y, p)})
    results = pd.DataFrame(rows).sort_values("RMSE (SPAD)").reset_index(drop=True)
    print("\n" + text_table(results.drop(columns=["Syllabus"])))

    best = results.iloc[0]["Model"]
    pipe = clone(model_specs(feature_set)[best][0]).fit(X, y)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "model_name": best, "feature_set": feature_set,
                 "features": config.CHLORO_FEATURE_SETS[feature_set], "n_leaves": n, "crops": crops,
                 "spad_range": (float(y.min()), float(y.max())), "cv": cv_name,
                 "metrics": results.iloc[0].drop(["Model", "Syllabus"]).to_dict(),
                 "feature_version": config.FEATURE_VERSION},
                models_dir / config.CHLOROPHYLL_MODEL_PATH.name, compress=3)
    save_table(results, "chlorophyll_results", reports_dir,
               f"SPAD regression - {cv_name} CV on {n} real leaves ({', '.join(crops)})")
    stab = coefficient_stability(X, y, feature_set)
    make_spad_graphs(y, preds, best, stab, feature_set, reports_dir)
    print(f"\nBest model: {best}  ->  saved to {models_dir / config.CHLOROPHYLL_MODEL_PATH.name}")
    return results


# ----------------------------------------------------------------------------- graphs
def make_collinearity_graph(corr: pd.DataFrame, vif: pd.DataFrame, n_leaves: int, reports_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    div = LinearSegmentedColormap.from_list("div", [COLORS["blue"], "#f2f2ef", COLORS["orange"]])
    im = a1.imshow(corr.values, cmap=div, vmin=-1, vmax=1)
    labels = list(corr.columns)
    a1.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8.5)
    a1.set_yticks(range(len(labels)), labels, fontsize=8.5); a1.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            a1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.75 else COLORS["ink"])
    fig.colorbar(im, ax=a1, shrink=0.8, label="Pearson correlation r")
    a1.set_title(f"Correlation of leaf colour features ({n_leaves:,} leaves)")

    v = vif.set_index("Feature")["VIF"].clip(upper=1000)
    bars = a2.barh(v.index[::-1], v.values[::-1], color=[COLORS["orange"] if x > 10 else COLORS["blue"]
                                                         for x in v.values[::-1]])
    a2.bar_label(bars, labels=[f"{x:.0f}" if x < 1000 else ">1000" for x in v.values[::-1]], padding=3, fontsize=8.5)
    a2.axvline(10, color=COLORS["muted"], ls=":", lw=1)
    a2.text(10, -0.9, " VIF = 10: strong collinearity", fontsize=8, color=COLORS["muted"])
    a2.set_xscale("log"); a2.set_xlabel("variance inflation factor (log scale)"); a2.grid(axis="y", visible=False)
    a2.set_title("VIF: why least squares is unstable -> Ridge")
    fig.tight_layout()
    return save_fig(fig, "chlorophyll", "1_collinearity", reports_dir)


def make_spad_graphs(y, preds, best, stab, feature_set, reports_dir) -> None:
    import matplotlib.pyplot as plt

    from src.evaluate import COLORS, apply_style, save_fig
    apply_style()
    p = preds[best]
    m = regression_metrics(y, p)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    lo, hi = min(y.min(), p.min()) - 2, max(y.max(), p.max()) + 2
    a1.plot([lo, hi], [lo, hi], color=COLORS["muted"], ls=":", lw=1, label="perfect prediction")
    a1.scatter(y, p, s=40, color=COLORS["blue"], edgecolors="white", linewidths=1, zorder=3)
    a1.set_xlim(lo, hi); a1.set_ylim(lo, hi); a1.set_aspect("equal")
    a1.set_xlabel("measured SPAD (meter)"); a1.set_ylabel("predicted SPAD (cross-validated)")
    a1.text(0.03, 0.97, f"R² = {m['R2']:.2f}\nRMSE = {m['RMSE (SPAD)']:.2f}\nMAPE = {m['MAPE (%)']:.1f}%",
            transform=a1.transAxes, va="top", fontsize=9)
    a1.set_title(f"{best}: predicted vs measured"); a1.legend(loc="lower right", fontsize=8)
    a2.scatter(p, y - p, s=40, color=COLORS["blue"], edgecolors="white", linewidths=1, zorder=3)
    a2.axhline(0, color=COLORS["ink"], lw=0.8)
    a2.set_xlabel("predicted SPAD"); a2.set_ylabel("residual = measured − predicted")
    a2.set_title("Residuals (no pattern = good)")
    save_fig(fig, "chlorophyll", "2_pred_vs_actual", reports_dir)

    names = config.CHLORO_FEATURE_SETS[feature_set]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    pos = np.arange(len(names))
    for arr, off, col, lab in [(stab["ols"], -0.18, COLORS["plain"], "least squares (OLS)"),
                               (stab["ridge"], 0.18, COLORS["bayes"], f"Ridge (lambda = {stab['alpha']:.3g})")]:
        bp = ax.boxplot([arr[:, j] for j in range(len(names))], positions=pos + off, widths=0.3, patch_artist=True,
                        showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_alpha(0.75); b.set_edgecolor(COLORS["ink"])
        for k in ("whiskers", "caps", "medians"):
            for line in bp[k]:
                line.set_color(COLORS["ink"])
        ax.plot([], [], color=col, lw=8, alpha=0.75, label=lab)
    ax.axhline(0, color=COLORS["muted"], lw=0.8)
    ax.set_xticks(pos, names); ax.set_ylabel("coefficient (standardised features)")
    ax.legend(fontsize=8.5); ax.grid(axis="x", visible=False)
    ax.set_title("Coefficient stability over 300 random refits (narrow = stable)")
    save_fig(fig, "chlorophyll", "3_coefficient_stability", reports_dir)


# ----------------------------------------------------------------------------- main
def run(csv: Path = config.SPAD_CSV, feature_set: str = config.CHLORO_DEFAULT_FEATURES,
        reports_dir: Path = config.REPORTS_DIR, models_dir: Path = config.MODELS_DIR,
        dataset_features: dict | None = None) -> pd.DataFrame | None:
    """Collinearity analysis (always) + training (only with real readings). Returns results or None."""
    from src.evaluate import save_table, text_table
    reports_dir, models_dir = Path(reports_dir), Path(models_dir)
    config.set_seeds()

    # ---- 1. collinearity on real leaf features (no SPAD needed)
    names = config.CHLORO_FEATURE_SETS["colour"]
    if dataset_features is not None:
        cols = [F.FEATURE_NAMES.index(n) for n in names]
        Xc = np.vstack([dataset_features["X_train"], dataset_features["X_valid"]])[:, cols]
        corr, vif = collinearity_report(Xc, names)
        print(f"Collinearity of leaf colour features ({len(Xc):,} leaves of the disease dataset):")
        print(text_table(vif.round(2)))
        save_table(corr.round(3).reset_index().rename(columns={"index": "Feature"}), "chlorophyll_rgb_correlation",
                   reports_dir, "Correlation of leaf colour features (disease dataset)")
        save_table(vif.round(2), "chlorophyll_vif", reports_dir, "Variance inflation factors of leaf colour features")
        make_collinearity_graph(corr, vif, len(Xc), reports_dir)

    # ---- 2. real SPAD readings?
    csv = Path(csv)
    if not csv.exists():
        write_template(csv)
        print(f"\nCreated an empty template: {csv}")
    readings = load_readings(csv)
    if len(readings) < config.MIN_SPAD_LEAVES:
        print(f"\n{len(readings)} leaf/leaves with SPAD readings in {csv.name} (need >= {config.MIN_SPAD_LEAVES}). "
              "No chlorophyll model is trained - the app keeps showing the UNCALIBRATED greenness index.\n"
              "How to collect readings: see docs/spad_collection.md")
        return None
    X = feature_matrix(readings["image_path"])
    return train(readings, X, feature_set, reports_dir, models_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chlorophyll (SPAD) regression from leaf colour - real readings only.")
    ap.add_argument("--csv", default=str(config.SPAD_CSV), help="CSV with columns image_path, spad_value")
    ap.add_argument("--features", choices=sorted(config.CHLORO_FEATURE_SETS), default=config.CHLORO_DEFAULT_FEATURES)
    ap.add_argument("--limit", type=int, default=None, help="feature cache used for the collinearity analysis")
    args = ap.parse_args(argv)
    from src import dataset
    try:
        data = dataset.load_features(args.limit)
    except FileNotFoundError:
        data = None
        print("(No disease feature cache - skipping the collinearity analysis.)")
    try:
        run(Path(args.csv), args.features, dataset_features=data)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
