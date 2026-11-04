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
