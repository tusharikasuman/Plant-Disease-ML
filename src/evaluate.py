"""
src/evaluate.py - metrics, report tables and the shared graph style (graphs are added in Phase 5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import config

# ----------------------------------------------------------------------------- graph style
# Colour-blind safe palette (checked with a CVD validator: every pair below passes in light mode).
# Colour follows the entity: healthy is always aqua, diseased always orange, and so on.
COLORS = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "healthy": "#1baf7a", "diseased": "#eb6834",
    "plain": "#2a78d6", "bayes": "#eb6834",           # plain (MAP) logistic regression vs Bayesian
    "ink": "#1f2a22", "muted": "#6b7266", "grid": "#e6e3da",
}


def apply_style() -> None:
    """White background, recessive axes and grid, readable labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left", "axes.labelcolor": COLORS["ink"],
        "axes.edgecolor": "#bdb6a4", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"], "text.color": COLORS["ink"],
        "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def save_fig(fig, name: str, reports_dir: Path = config.REPORTS_DIR) -> Path:
    """Save reports/<name>.png at 160 dpi and close the figure."""
    import matplotlib.pyplot as plt
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{name}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Accuracy, balanced accuracy (mean recall per class) and macro F1, as percentages."""
    return {
        "accuracy": 100 * accuracy_score(y_true, y_pred),
        "balanced_accuracy": 100 * balanced_accuracy_score(y_true, y_pred),
        "macro_f1": 100 * f1_score(y_true, y_pred, average="macro"),
    }


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> str:
    """DataFrame -> GitHub Markdown table (no extra dependency needed)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def text_table(df: pd.DataFrame) -> str:
    """DataFrame -> aligned plain-text table for the console."""
    cells = [[str(c) for c in df.columns]] + [[_fmt(v) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
    out = []
    for n, r in enumerate(cells):
        out.append("  ".join(v.ljust(w) for v, w in zip(r, widths)))
        if n == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def save_table(df: pd.DataFrame, name: str, reports_dir: Path = config.REPORTS_DIR, title: str = "") -> None:
    """Write reports/<name>.csv and reports/<name>.md."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / f"{name}.csv", index=False)
    md = (f"## {title}\n\n" if title else "") + markdown_table(df) + "\n"
    (reports_dir / f"{name}.md").write_text(md, encoding="utf-8")


# ----------------------------------------------------------------------------- graphs (Phase 5)
def plot_segmentation(image_paths, reports_dir: Path = config.REPORTS_DIR) -> Path:
    """One row per leaf: original | leaf mask | segmented leaf with lesion pixels highlighted."""
    import matplotlib.pyplot as plt
    from src import features as F
    apply_style()
    rows = len(image_paths)
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3.1 * rows), squeeze=False)
    for r, path in enumerate(image_paths):
        v = F.segmented_views(path)
        x, _ = F.extract_features(path)
        name = Path(path).stem.replace("tomato_", "").replace("_", " ")
        for c, (img, title) in enumerate([(v["image"], f"{name}: original"),
                                          (v["mask"], "leaf mask (white = leaf)"),
                                          (v["lesions"], f"lesions in orange: {100 * x[F.LESION_INDEX]:.1f}% of leaf")]):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            ax.set_title(title, fontsize=9.5); ax.axis("off")
    fig.suptitle("Segmentation: HSV threshold -> closing -> largest component -> fill holes",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return save_fig(fig, "segmentation_examples", reports_dir)


def plot_model_comparison(results: pd.DataFrame, reports_dir: Path = config.REPORTS_DIR) -> Path:
    """Horizontal bars: 5-fold CV accuracy (+- std) and held-out balanced accuracy for every model."""
    import matplotlib.pyplot as plt
    apply_style()
    order = results.sort_values("CV accuracy (%)").reset_index(drop=True)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(order) + 1.4))
    b1 = ax.barh(y + 0.2, order["CV accuracy (%)"], 0.38, xerr=order["CV std"], color=COLORS["blue"],
                 error_kw={"ecolor": COLORS["muted"], "capsize": 2, "lw": 1}, label="5-fold CV accuracy (± std)")
    b2 = ax.barh(y - 0.2, order["Valid balanced acc (%)"], 0.38, color=COLORS["orange"],
                 label="held-out balanced accuracy")
    ax.bar_label(b1, fmt="%.1f", padding=10, fontsize=8, color=COLORS["ink"])
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8, color=COLORS["ink"])
    ax.set_yticks(y, [f"{m}  ·  {u}" for m, u in zip(order["Model"], order["Syllabus"])])
    ax.set_xlim(max(0, order[["CV accuracy (%)", "Valid balanced acc (%)"]].min().min() - 10), 103)
    ax.set_xlabel("accuracy (%)"); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Disease classification: 8 classical models, same 55 features")
    return save_fig(fig, "model_comparison", reports_dir)


def plot_ablation(abl: pd.DataFrame, reports_dir: Path = config.REPORTS_DIR) -> Path:
    """Grouped bars: CV accuracy as feature groups are added."""
    import matplotlib.pyplot as plt
    apply_style()
    cols = [c for c in abl.columns if c.endswith("CV acc (%)")]
    x = np.arange(len(abl))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    width = 0.8 / max(len(cols), 1)
    for i, (col, color) in enumerate(zip(cols, [COLORS["blue"], COLORS["orange"]])):
        bars = ax.bar(x + (i - (len(cols) - 1) / 2) * width, abl[col], width * 0.95, color=color,
                      label=col.replace(" CV acc (%)", ""))
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2, color=COLORS["ink"])
    ax.set_xticks(x, [f"{s}\n({n} features)" for s, n in zip(abl["Feature set"], abl["# features"])], fontsize=8.5)
    ax.set_ylim(max(0, abl[cols].min().min() - 12), 100); ax.set_ylabel("5-fold CV accuracy (%)")
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Feature ablation: what each feature group adds")
    return save_fig(fig, "ablation", reports_dir)


def main() -> int:
    """Regenerate every graph that can be made from the saved results."""
    made = []
    samples = sorted(config.SAMPLES_DIR.glob("*.jpg"))
    if samples:
        made.append(plot_segmentation(samples))
    res = config.REPORTS_DIR / "classification_results.csv"
    if res.exists():
        made.append(plot_model_comparison(pd.read_csv(res)))
    abl = config.REPORTS_DIR / "ablation.csv"
    if abl.exists():
        made.append(plot_ablation(pd.read_csv(abl)))
    if not res.exists():
        print("No classification results yet - run  python -m src.classify  first.")
    for p in made:
        print(f"Saved {p}")
    return 0


if __name__ == "__main__":
    import sys
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
