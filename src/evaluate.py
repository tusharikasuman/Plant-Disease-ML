"""
src/evaluate.py - metrics and report tables (graphs are added in Phase 5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import config


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
