"""
src/tree_report.py - print the whole decision tree and how it classifies every held-out leaf.

The tree uses the settings grid search chose in src/classify.py (max_depth=10, min_samples_leaf=1, seed 42).
It is fitted on the raw features, so thresholds read in real units (e.g. "Lesion fraction <= 0.051" = 5.1 %).
A tree's splits only depend on the order of values, so this gives the same predictions as the
StandardScaler -> tree pipeline in classify.py.

Usage (from the plant-disease-ml folder):
    python -m src.tree_report              # writes reports/decision_tree/, prints a summary
    python -m src.tree_report --depth 3    # also print the top 3 levels of the tree on screen

Outputs in reports/decision_tree/:
    tree_rules.txt          every split and every leaf of the tree, with class counts
    leaf_rules.csv          one row per leaf of the tree: its full rule, prediction, size and purity
    test_predictions.csv    every held-out leaf photo: true class, predicted class, confidence, and the rule used
    classification_report.txt   precision / recall / F1 per class + confusion matrix
    feature_importance.csv  how much each feature reduces Gini impurity in the tree
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

import config
from src import dataset

PARAMS = {"max_depth": 10, "min_samples_leaf": 1}      # chosen by grid search, reports/tables/classification_results.md


def short(name: str) -> str:
    s = config.pretty_class(name).replace("Tomato: ", "").replace("Tomato ", "")
    return s[:1].upper() + s[1:].lower()          # "Yellow leaf curl virus", "Healthy"


def leaf_rules(tree: DecisionTreeClassifier, feature_names: list[str], classes: list[str]) -> dict[int, dict]:
    """Walk the tree once: the full if-then rule, prediction, size and purity of every leaf node."""
    t = tree.tree_
    out: dict[int, dict] = {}

    def walk(node: int, conds: list[str]) -> None:
        if t.children_left[node] == -1:
            counts = t.value[node][0] * t.n_node_samples[node] if t.value[node][0].sum() <= 1.0001 else t.value[node][0]
            counts = np.rint(counts).astype(int)
            k = int(np.argmax(counts))
            out[node] = {"leaf_id": node, "predicts": classes[k], "training_leaves": int(t.n_node_samples[node]),
                         "purity_%": round(100 * counts[k] / counts.sum(), 1),
                         **{f"n_{c}": int(n) for c, n in zip(classes, counts)},
                         "depth": len(conds), "rule": " AND ".join(conds) or "(root)"}
            return
        f, thr = feature_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.4g}"])

    walk(0, [])
    return out


TREE_MODEL_PATH = config.MODELS_DIR / "decision_tree.joblib"


def node_table(tree: DecisionTreeClassifier, feature_names: list[str], classes: list[str], Xv, yv) -> pd.DataFrame:
    """One row per node: parent, yes/no branch, depth, question, training-class counts, held-out leaves reaching it."""
    t = tree.tree_
    passed = tree.decision_path(Xv).toarray().astype(bool)          # (test leaves, nodes)
    pred = tree.predict(Xv)
    parent, branch, depth = {0: -1}, {0: ""}, {0: 0}
    for i in range(t.node_count):
        for child, b in ((t.children_left[i], "yes"), (t.children_right[i], "no")):
            if child != -1:
                parent[child], branch[child], depth[child] = i, b, depth[i] + 1
    rows = []
    for i in range(t.node_count):
        v = t.value[i][0]
        cnt = np.rint(v * t.n_node_samples[i] if v.sum() <= 1.0001 else v).astype(int)
        leaf = t.children_left[i] == -1
        p = passed[:, i]
        rows.append({"node": i, "parent": parent[i], "branch": branch[i], "depth": depth[i], "is_leaf": int(leaf),
                     "feature": "" if leaf else feature_names[t.feature[i]],
                     "threshold": np.nan if leaf else round(float(t.threshold[i]), 5),
                     "train_leaves": int(t.n_node_samples[i]), "predicts": classes[int(np.argmax(cnt))],
                     "purity_pct": round(100 * cnt.max() / cnt.sum(), 1), "test_leaves": int(p.sum()),
                     "test_correct": int((pred[p] == yv[p]).sum()) if leaf else np.nan,
                     **{f"n_{c}": int(n) for c, n in zip(classes, cnt)}})
    return pd.DataFrame(rows)


def run(data: dict, out_dir: Path, show_depth: int = 0) -> dict:
    config.set_seeds()
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, Xv, yv = data["X_train"], data["y_train"], data["X_valid"], data["y_valid"]
    names = list(data["feature_names"])
    classes = [short(c) for c in data["class_names"]]

    tree = DecisionTreeClassifier(random_state=config.SEED, **PARAMS).fit(X, y)
    pred, proba = tree.predict(Xv), tree.predict_proba(Xv)
    acc = 100 * float(np.mean(pred == yv))

    # 1. the whole tree as text
    text = export_text(tree, feature_names=names, class_names=classes, show_weights=True, decimals=4, max_depth=PARAMS["max_depth"])
    header = (f"Decision tree (CART, Gini impurity), max_depth={PARAMS['max_depth']}, min_samples_leaf={PARAMS['min_samples_leaf']}\n"
              f"Trained on {len(y):,} leaves, {tree.tree_.node_count:,} nodes, {tree.get_n_leaves():,} leaves (end nodes), depth {tree.get_depth()}\n"
              f"Held-out accuracy: {acc:.2f}% on {len(yv):,} leaves\n"
              f"weights = training leaves of each class reaching that node, in the order {classes}\n\n")
    (out_dir / "tree_rules.txt").write_text(header + text, encoding="utf-8")

    # 2. one row per end node
    rules = leaf_rules(tree, names, classes)
    leaf_df = pd.DataFrame(rules.values()).sort_values("training_leaves", ascending=False)
    leaf_df.to_csv(out_dir / "leaf_rules.csv", index=False)

    # 3. every held-out photo
    node = tree.apply(Xv)
    paths = [Path(str(p)) for p in data["paths_valid"]]
    test = pd.DataFrame({
        "photo": [p.name for p in paths], "true_class": [classes[i] for i in yv], "predicted_class": [classes[i] for i in pred],
        "correct": pred == yv, "confidence_%": np.round(100 * proba.max(1), 1), "end_node": node,
        "end_node_training_leaves": [rules[n]["training_leaves"] for n in node],
        "rule_used": [rules[n]["rule"] for n in node]})
    test.to_csv(out_dir / "test_predictions.csv", index=False)

    # 4. per-class report + confusion matrix
    cm = confusion_matrix(yv, pred)
    cm_df = pd.DataFrame(cm, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes])
    rep = classification_report(yv, pred, target_names=classes, digits=3)
    (out_dir / "classification_report.txt").write_text(f"{header}{rep}\nConfusion matrix (rows = true class):\n{cm_df.to_string()}\n", encoding="utf-8")

    # 4b. every node (for the app's tree drawing) + the fitted tree itself (to trace any photo through it)
    node_table(tree, names, classes, Xv, yv).to_csv(out_dir / "tree_nodes.csv", index=False)
    import joblib
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tree": tree, "feature_names": names, "classes": classes, "feature_version": config.FEATURE_VERSION,
                 "accuracy": acc}, TREE_MODEL_PATH, compress=3)

    # 5. feature importance
    imp = pd.DataFrame({"feature": names, "importance_%": np.round(100 * tree.feature_importances_, 2)}).sort_values("importance_%", ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", index=False)

    # ---- screen summary
    print(header.rstrip())
    print(f"\nPer-class results on the held-out leaves:\n{rep}")
    print("Confusion matrix (rows = true class, columns = predicted):")
    print(cm_df.to_string())
    print("\nMost used features (share of the total Gini decrease):")
    print(imp.head(10).to_string(index=False))
    wrong = test[~test["correct"]]
    print(f"\nMistakes: {len(wrong)} of {len(test)}. Most common confusions:")
    print(wrong.groupby(["true_class", "predicted_class"]).size().sort_values(ascending=False).head(8).to_string())
    if show_depth:
        print(f"\nTop {show_depth} levels of the tree:\n")
        print(export_text(tree, feature_names=names, class_names=classes, max_depth=show_depth, decimals=4))
    print(f"\nFull outputs in {out_dir}:")
    for f in ["tree_rules.txt", "leaf_rules.csv", "test_predictions.csv", "classification_report.txt", "feature_importance.csv"]:
        print(f"  {f}")
    return {"tree": tree, "accuracy": acc, "test": test, "leaves": leaf_df, "importance": imp, "confusion": cm}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the whole decision tree and how it classifies every held-out leaf.")
    ap.add_argument("--depth", type=int, default=0, help="also print the top N levels of the tree on screen")
    ap.add_argument("--limit", type=int, default=None, help="use the --limit N feature cache")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        data = dataset.load_features(args.limit)
    except FileNotFoundError as e:
        print(e)
        return 1
    run(data, config.REPORTS_DIR / "decision_tree", args.depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
