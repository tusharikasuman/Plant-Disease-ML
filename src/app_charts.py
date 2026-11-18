"""
src/app_charts.py - interactive charts for the Streamlit app (Altair: hover for exact values, scroll to zoom).

    leaf_section(...)   graphs for ONE uploaded leaf: every model's probability for every disease, where the leaf
                        sits among the training leaves, its key features vs each disease, its path through the
                        decision tree, and Units 3-4 (k-means, GMM, HMM) run on its own pixels
    page_units_1_4()    Units 1-4 from reports/tables + the feature cache
    page_unit5()        the 11 Unit 5 graphs + 3 chlorophyll graphs from reports/graph_data/unit5.json
    page_tree()         the decision tree: drawing, every split, every held-out classification

Every chart is drawn from data the project's own code produced; nothing here trains a new kind of model.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import features as F

# Okabe-Ito colour-blind safe palette, one fixed colour per disease
DISEASES = ["Early blight", "Late blight", "Septoria leaf spot", "Yellow leaf curl virus", "Healthy"]
DCOL = ["#E69F00", "#0072B2", "#CC79A7", "#D55E00", "#009E73"]
DSCALE = alt.Scale(domain=DISEASES, range=DCOL)
TRAIN, HELD = "#0072B2", "#E69F00"                 # training leaves blue, held-out leaves orange (as in the PNGs)
SET_SCALE = alt.Scale(domain=["training leaves", "held-out leaves"], range=[TRAIN, HELD])


def short(name: str) -> str:
    s = config.pretty_class(name).replace("Tomato: ", "").replace("Tomato ", "")
    return s[:1].upper() + s[1:].lower()


def show(chart, height: int | None = None) -> None:
    if height:
        chart = chart.properties(height=height)
    st.altair_chart(chart, width="stretch")


def long(df: pd.DataFrame, x: str, cols: dict[str, str]) -> pd.DataFrame:
    """Wide -> long for multi-line charts: cols = {column: series label}."""
    out = df[[x, *cols]].melt(x, var_name="series", value_name="value")
    out["series"] = out["series"].map(cols)
    return out


def lines(df, x, cols, x_title, y_title, scale=SET_SCALE, x_log=False, y_log=False, points=False, y_zero=False, fmt=".2f", dashed=()):
    d = long(df, x, cols)
    lo, hi = float(df[x].min()), float(df[x].max())
    xs = alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(type="log", domain=[lo, hi], nice=False) if x_log else alt.Scale(zero=False, domain=[lo, hi], nice=False))
    ys = alt.Y("value:Q", title=y_title, scale=alt.Scale(type="log") if y_log else alt.Scale(zero=y_zero))
    labels = list(cols.values())
    return alt.Chart(d).mark_line(point=points, strokeWidth=2).encode(
        x=xs, y=ys, color=alt.Color("series:N", scale=scale, title=None, legend=alt.Legend(orient="top")),
        strokeDash=alt.StrokeDash("series:N", legend=None, scale=alt.Scale(domain=labels, range=[[5, 4] if lab in dashed else [1, 0] for lab in labels])),
        tooltip=[alt.Tooltip(f"{x}:Q", title=x_title), alt.Tooltip("series:N", title="series"),
                 alt.Tooltip("value:Q", title=y_title, format=fmt)]).interactive()


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unit5_data() -> dict | None:
    p = config.REPORTS_DIR / "graph_data" / "unit5.json"
    return _json(str(p), p.stat().st_mtime) if p.exists() else None


def table(name: str) -> pd.DataFrame | None:
    p = config.tables_dir() / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner="Loading the leaf features...")
def features():
    from src import dataset
    try:
        return dataset.load_features()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner=False)
def pca_space():
    """StandardScaler + 2-component PCA fitted on the training leaves, and a sample of them for plotting."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    d = features()
    if d is None:
        return None
    sc = StandardScaler().fit(d["X_train"])
    pca = PCA(2, random_state=config.SEED).fit(sc.transform(d["X_train"]))
    rng = np.random.default_rng(config.SEED)
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(d["y_train"] == c), min(300, int((d["y_train"] == c).sum())), replace=False)
                                  for c in np.unique(d["y_train"])]))
    Z = pca.transform(sc.transform(d["X_train"][idx]))
    pts = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "disease": [short(d["class_names"][c]) for c in d["y_train"][idx]]})
    return sc, pca, pts


@st.cache_resource(show_spinner=False)
def tree_model():
    import joblib
    from src.tree_report import TREE_MODEL_PATH
    if not TREE_MODEL_PATH.exists():
        return None
    b = joblib.load(TREE_MODEL_PATH)
    return b if b.get("feature_version") == config.FEATURE_VERSION else None


# =============================================================================================================
# Graphs for ONE uploaded leaf
# =============================================================================================================
KEY_FEATURES = ["Lesion fraction", "GLI", "ExG (normalised)", "G/R ratio", "GLCM contrast", "R std"]


def leaf_section(img, x: np.ndarray, probs: pd.DataFrame, logreg: dict | None) -> None:
    """probs: long table Model / Disease / Probability (%) for this leaf."""
    st.subheader("3 · Graphs for this leaf")
    tabs = st.tabs(["Every model × every disease", "Where this leaf sits", "Its features vs each disease",
                    "Why (logistic regression)", "Path through the decision tree", "Units 3–4 on its pixels"])

    with tabs[0]:
        st.caption("Each row is a model, each column a disease; the number is the model's probability for that disease. "
                   "A dark cell in one column means the model is sure. Point at a cell for the exact value.")
        order = probs.groupby("Model", sort=False).size().index.tolist()
        base = alt.Chart(probs).encode(x=alt.X("Disease:N", sort=DISEASES, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
                                       y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)))
        heat = base.mark_rect().encode(color=alt.Color("Probability (%):Q", scale=alt.Scale(scheme="greens", domain=[0, 100]), legend=None),
                                       tooltip=["Model", "Disease", alt.Tooltip("Probability (%):Q", format=".1f")])
        text = base.mark_text(fontSize=11).encode(text=alt.Text("Probability (%):Q", format=".0f"),
                                                  color=alt.condition("datum['Probability (%)'] > 60", alt.value("white"), alt.value("#333")))
        show(heat + text, height=34 * len(order))

    with tabs[1]:
        space = pca_space()
        if space is None:
            st.info("Needs the feature cache (python -m src.dataset).")
        else:
            sc, pca, pts = space
            z = pca.transform(sc.transform(x[None]))[0]
            me = pd.DataFrame({"PC1": [z[0]], "PC2": [z[1]], "disease": ["THIS LEAF"]})
            st.caption("1,500 training leaves (300 per disease) squeezed onto 2 dimensions with PCA (Unit 3). "
                       "The black star is your leaf: the disease cloud it falls into is what the models lean towards. "
                       "Scroll to zoom, drag to pan.")
            dots = alt.Chart(pts).mark_circle(size=22, opacity=0.6).encode(
                x=alt.X("PC1:Q", scale=alt.Scale(zero=False)), y=alt.Y("PC2:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("disease:N", scale=DSCALE, title=None, legend=alt.Legend(orient="top")),
                tooltip=["disease", alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")])
            star = alt.Chart(me).mark_point(shape="diamond", size=420, filled=True, color="black", stroke="white", strokeWidth=2).encode(
                x="PC1:Q", y="PC2:Q", tooltip=[alt.Tooltip("disease:N", title="point"), alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")])
            show((dots + star).interactive(), height=430)

    with tabs[2]:
        d = features()
        if d is None:
            st.info("Needs the feature cache (python -m src.dataset).")
        else:
            names = list(d["feature_names"])
            rows = []
            for f in KEY_FEATURES:
                j = names.index(f)
                for c in np.unique(d["y_train"]):
                    v = d["X_train"][d["y_train"] == c, j]
                    rows.append({"feature": f, "who": short(d["class_names"][c]), "value": float(np.median(v)),
                                 "low": float(np.percentile(v, 25)), "high": float(np.percentile(v, 75))})
                rows.append({"feature": f, "who": "THIS LEAF", "value": float(x[j]), "low": float(x[j]), "high": float(x[j])})
            df = pd.DataFrame(rows)
            if "Lesion fraction" in KEY_FEATURES:
                m = df["feature"] == "Lesion fraction"
                df.loc[m, ["value", "low", "high"]] *= 100
                df.loc[m, "feature"] = "Lesion area (% of leaf)"
            st.caption("For six key features: the middle 50% of training leaves of each disease (bar) and their median (dot), "
                       "with your leaf as the black diamond. Your leaf looks like the diseases whose bars it falls in.")
            scale = alt.Scale(domain=DISEASES + ["THIS LEAF"], range=DCOL + ["black"])
            band = alt.Chart().mark_rule(strokeWidth=5, opacity=0.5).encode(
                x=alt.X("low:Q", title=None, scale=alt.Scale(zero=False)), x2="high:Q", y=alt.Y("who:N", sort=DISEASES + ["THIS LEAF"], title=None),
                color=alt.Color("who:N", scale=scale, legend=None))
            dot = alt.Chart().mark_point(filled=True, size=90).encode(
                x="value:Q", y=alt.Y("who:N", sort=DISEASES + ["THIS LEAF"]), color=alt.Color("who:N", scale=scale, legend=None),
                shape=alt.condition("datum.who == 'THIS LEAF'", alt.value("diamond"), alt.value("circle")),
                tooltip=["feature", "who", alt.Tooltip("value:Q", format=".3f", title="median / this leaf"),
                         alt.Tooltip("low:Q", format=".3f", title="25th percentile"), alt.Tooltip("high:Q", format=".3f", title="75th percentile")])
            chart = alt.layer(band, dot, data=df).properties(width=300, height=150).facet(
                facet=alt.Facet("feature:N", title=None, header=alt.Header(labelFontWeight="bold", labelFontSize=12)), columns=2
            ).resolve_scale(x="independent")
            st.altair_chart(chart)

    with tabs[3]:
        from src import predict as P
        names = [c for c in (logreg or {}).get("class_names", [])]
        if not names:
            st.info("Needs models/logistic_regression.joblib (python -m src.classify).")
        else:
            probs_lr = logreg["pipeline"].predict_proba(x[None])[0]
            label = names[int(np.argmax(probs_lr))]
            top, groups = P.explain(logreg, x, label, top_k=10)
            st.caption(f"Logistic regression predicts **{short(label)}**. For a linear model, weight × (standardised feature value) "
                       "is exactly how much each feature pushed the score towards that disease (right) or away from it (left).")
            g = pd.DataFrame([{"feature group": k, "push": v} for k, v in groups.items()])
            f = pd.DataFrame(top)
            c1, c2 = st.columns(2)
            with c1:
                show(alt.Chart(g).mark_bar(cornerRadius=3).encode(
                    x=alt.X("push:Q", title="push towards the prediction"), y=alt.Y("feature group:N", sort="-x", title=None),
                    color=alt.condition("datum.push > 0", alt.value("#009E73"), alt.value("#D55E00")),
                    tooltip=["feature group", alt.Tooltip("push:Q", format="+.2f")]), height=220)
            with c2:
                show(alt.Chart(f).mark_bar(cornerRadius=3).encode(
                    x=alt.X("contribution:Q", title="push (single feature)"), y=alt.Y("feature:N", sort="-x", title=None),
                    color=alt.condition("datum.contribution > 0", alt.value("#009E73"), alt.value("#D55E00")),
                    tooltip=["feature", "group", alt.Tooltip("value:Q", format=".4f"), alt.Tooltip("z_score:Q", format=".2f", title="standardised"),
                             alt.Tooltip("weight:Q", format=".2f"), alt.Tooltip("contribution:Q", format="+.2f")]), height=300)

    with tabs[4]:
        b = tree_model()
        if b is None:
            st.info("Needs the decision tree: run  python -m src.tree_report  once.")
        else:
            tree, fn = b["tree"], b["feature_names"]
            path = tree.decision_path(x[None]).indices
            t = tree.tree_
            steps = []
            for node in path:
                if t.children_left[node] == -1:
                    v = t.value[node][0]
                    steps.append({"step": len(steps) + 1, "question": "END LEAF", "this leaf's value": "",
                                  "answer": f"→ {b['classes'][int(np.argmax(v))]} ({100 * v.max() / v.sum():.0f}% of {t.n_node_samples[node]} training leaves here)"})
                else:
                    f, thr = fn[t.feature[node]], t.threshold[node]
                    steps.append({"step": len(steps) + 1, "question": f"{f} ≤ {thr:.4g} ?", "this leaf's value": f"{x[t.feature[node]]:.4g}",
                                  "answer": "yes" if x[t.feature[node]] <= thr else "no"})
            st.caption("The exact questions the decision tree (Unit 5, CART) asks about this leaf, top to bottom, and its answers.")
            st.graphviz_chart(tree_dot(b, highlight=set(path.tolist()), max_depth=None, path_only=True))
            st.dataframe(pd.DataFrame(steps), hide_index=True, width="stretch")

    with tabs[5]:
        st.caption("Unit 3 and Unit 4 run on this photo's own pixels (no labels): k-means and a Gaussian mixture (EM) "
                   "group the pixels by colour into background / healthy tissue / lesion; then an HMM reads each row of "
                   "pixels left to right and Viterbi picks the most likely tissue sequence, which removes speckle.")
        res = pixel_units(img)
        cols = st.columns(4)
        cols[0].image(res["photo"], caption="Photo (128 × 128)", width="stretch")
        for col, key, title in [(cols[1], "kmeans", "k-means, K = 3 (Unit 3)"), (cols[2], "gmm", "Gaussian mixture + EM (Unit 3)"),
                                (cols[3], "viterbi", "HMM + Viterbi (Unit 4)")]:
            col.image(res[key]["image"], caption=f"{title}: lesion {res[key]['lesion']:.1f}% of leaf", width="stretch")
        st.caption("Dark = background, green = healthy tissue, orange = lesion. "
                   f"Viterbi changed {res['relabelled']:.1f}% of pixels; state changes along the leafiest row: "
                   f"k-means {res['switch_km']}, Viterbi {res['switch_vit']}.")
        show(lines(pd.DataFrame({"x": np.arange(len(res["row_km"])), "k-means": res["row_km"], "Viterbi": res["row_vit"]}),
                   "x", {"k-means": "k-means", "Viterbi": "Viterbi (HMM)"}, "pixel position along the row (left → right)",
                   "state (0 bg, 1 healthy, 2 lesion)", scale=alt.Scale(domain=["k-means", "Viterbi (HMM)"], range=["#9E9E9E", "#D55E00"]), fmt=".0f"),
             height=200)


@st.cache_data(show_spinner="Running k-means, GMM and the HMM on the pixels...")
def pixel_units(img: np.ndarray, size: int = 128) -> dict:
    import cv2
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    from src import leaf_unsupervised as LU
    small = cv2.resize(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (size, size), interpolation=cv2.INTER_AREA)
    if small.shape[2] == 4:
        small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
    rgb, hsv = cv2.cvtColor(small, cv2.COLOR_BGR2RGB), cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    raw = LU.pixel_features(rgb, hsv)
    xs = StandardScaler().fit_transform(raw)
    colours = np.array([[40, 40, 40], [0, 158, 115], [230, 159, 0]], dtype=np.uint8)

    def summary(states):
        leaf = np.isin(states, [1, 2]).sum()
        return {"image": colours[states], "lesion": 100 * (states == 2).sum() / max(leaf, 1)}

    km = KMeans(3, n_init=10, random_state=config.SEED).fit(xs)
    km_states = LU.name_clusters(raw, km.labels_, 3)[km.labels_].reshape(size, size)
    gmm = GaussianMixture(3, covariance_type="full", means_init=km.cluster_centers_, random_state=config.SEED).fit(xs)
    gl = gmm.predict(xs)
    gmm_states = LU.name_clusters(raw, gl, 3)[gl].reshape(size, size) if len(np.unique(gl)) == 3 else km_states
    codebook = KMeans(8, n_init=5, random_state=config.SEED).fit(xs)
    obs = codebook.labels_.reshape(size, size)
    pi = np.bincount(km_states[:, 0], minlength=3) + 1.0; pi /= pi.sum()
    A = np.ones((3, 3)); np.add.at(A, (km_states[:, :-1].ravel(), km_states[:, 1:].ravel()), 1); A /= A.sum(1, keepdims=True)
    B = np.ones((3, 8)); np.add.at(B, (km_states.ravel(), obs.ravel()), 1); B /= B.sum(1, keepdims=True)
    path, _ = LU.viterbi(obs, pi, A, B)
    row = int(np.argmax(np.isin(km_states, [1, 2]).sum(1)))
    return {"photo": rgb, "kmeans": summary(km_states), "gmm": summary(gmm_states), "viterbi": summary(path),
            "relabelled": 100 * float((path != km_states).mean()), "row_km": km_states[row].tolist(), "row_vit": path[row].tolist(),
            "switch_km": int((km_states[row, 1:] != km_states[row, :-1]).sum()), "switch_vit": int((path[row, 1:] != path[row, :-1]).sum())}


# =============================================================================================================
# Decision tree drawing (Graphviz DOT, rendered by Streamlit in the browser)
# =============================================================================================================
def tree_dot(b: dict, highlight: set | None = None, max_depth: int | None = 3, path_only: bool = False) -> str:
    t, fn, classes = b["tree"].tree_, b["feature_names"], b["classes"]
    colour = dict(zip(DISEASES, DCOL))
    highlight = highlight or set()
    out = ['digraph T {', 'graph [rankdir=TB, nodesep=0.15, ranksep=0.35, bgcolor="transparent"];',
           'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, penwidth=1.2];',
           'edge [fontname="Helvetica", fontsize=9, color="#888888"];']

    def visit(i: int, depth: int):
        if path_only and i not in highlight:
            return
        v = t.value[i][0]
        k = int(np.argmax(v)); n = int(t.n_node_samples[i]); pur = 100 * v.max() / v.sum()
        hl = i in highlight and not path_only
        pen = 'penwidth=3, color="black"' if hl else f'color="{colour[classes[k]]}"'
        if t.children_left[i] == -1:
            out.append(f'n{i} [label="{classes[k]}\\n{pur:.0f}% of {n} leaves", fillcolor="{colour[classes[k]]}55", {pen}];')
            return
        if max_depth is not None and depth >= max_depth:
            out.append(f'n{i} [label="… {n} leaves\\nmostly {classes[k]} ({pur:.0f}%)", fillcolor="#f2f2f2", style="rounded,filled,dashed", {pen}];')
            return
        out.append(f'n{i} [label="{fn[t.feature[i]]} ≤ {t.threshold[i]:.4g} ?\\n{n} leaves · mostly {classes[k]}", fillcolor="{colour[classes[k]]}22", {pen}];')
        for child, lab in ((t.children_left[i], "yes"), (t.children_right[i], "no")):
            if path_only and child not in highlight:
                continue
            out.append(f'n{i} -> n{child} [label="{lab}"{", penwidth=2.5, color=black" if child in highlight and not path_only else ""}];')
            visit(child, depth + 1)

    visit(0, 0)
    out.append("}")
    return "\n".join(out)


def page_tree() -> None:
    st.title("🌳 Decision tree: every split, every classification")
    b = tree_model()
    nodes = None
    p = config.REPORTS_DIR / "decision_tree" / "tree_nodes.csv"
    if p.exists():
        nodes = pd.read_csv(p)
    if b is None or nodes is None:
        st.error("Run  `python -m src.tree_report`  once in the terminal, then reload this page.")
        return
    st.markdown("A CART decision tree (Unit 5: tree-based models), Gini impurity, depth 10, trained on 9,302 leaves. "
                "Each box asks one yes/no question about one feature; thresholds are in real units "
                "(lesion fraction 0.05 = 5% of the leaf). Random forests and gradient boosting are many trees like this one.")
    m = st.columns(4)
    m[0].metric("Held-out accuracy", f"{b['accuracy']:.2f}%")
    m[1].metric("Questions (split nodes)", int((nodes.is_leaf == 0).sum()))
    m[2].metric("End leaves (answers)", int(nodes.is_leaf.sum()))
    m[3].metric("Depth", int(nodes.depth.max()))

    tabs = st.tabs(["Draw the tree", "Trace a test photo", "Every classification", "Every end leaf (rule)", "Confusion matrix", "Features it uses"])
    folder = config.REPORTS_DIR / "decision_tree"
    with tabs[0]:
        depth = st.slider("Levels to draw", 1, 6, 3, help="The full tree has 10 levels and 681 boxes; deeper than ~5 gets very wide.")
        st.caption("Box colour = the disease most training leaves in that box have. Dashed grey boxes are cut off below the chosen level.")
        st.graphviz_chart(tree_dot(b, max_depth=depth), width="stretch")
    with tabs[1]:
        tp = pd.read_csv(folder / "test_predictions.csv")
        c1, c2 = st.columns(2)
        cls = c1.selectbox("True disease", DISEASES, key="tree_trace_cls")
        kind = c2.radio("Show", ["wrong answers", "right answers"], horizontal=True, key="tree_trace_kind")
        sub = tp[(tp.true_class == cls) & (tp.correct == (kind == "right answers"))]
        if sub.empty:
            st.info("No such photos.")
        else:
            photo = st.selectbox(f"Photo ({len(sub)} found)", sub.photo.tolist(), key="tree_trace_photo")
            r = sub[sub.photo == photo].iloc[0]
            d = features()
            img_path = next((Path(str(q)) for q in (d["paths_valid"] if d is not None else []) if Path(str(q)).name == photo), None)
            c1, c2 = st.columns([1, 2])
            if img_path is not None and img_path.exists():
                c1.image(F.segmented_views(str(img_path))["image"], caption=photo, width="stretch")
            c2.markdown(f"**True:** {r.true_class}  \n**Tree says:** {r.predicted_class} "
                        f"({'right' if r.correct else 'WRONG'}, {r['confidence_%']:.0f}% of the training leaves in its end leaf)")
            c2.caption("The path below is highlighted in black in the tree; the rule it followed:")
            c2.code(r.rule_used.replace(" AND ", "\n"), language=None)
            if d is not None:
                j = [Path(str(q)).name for q in d["paths_valid"]].index(photo)
                path = set(b["tree"].decision_path(d["X_valid"][j:j + 1]).indices.tolist())
                st.graphviz_chart(tree_dot(b, highlight=path, path_only=True))
    with tabs[2]:
        tp = pd.read_csv(folder / "test_predictions.csv")
        st.caption(f"How the tree classified all {len(tp):,} held-out photos. Filter, sort by any column, or search with the magnifier.")
        c1, c2 = st.columns(2)
        f_true = c1.multiselect("True disease", DISEASES, key="tree_f_true")
        f_ok = c2.selectbox("Answer", ["all", "right only", "wrong only"], key="tree_f_ok")
        v = tp if not f_true else tp[tp.true_class.isin(f_true)]
        v = v if f_ok == "all" else v[v.correct == (f_ok == "right only")]
        st.dataframe(v, hide_index=True, width="stretch", height=420)
        st.caption(f"{len(v):,} photos shown · {int(v.correct.sum()):,} right, {int((~v.correct).sum()):,} wrong")
    with tabs[3]:
        lr = pd.read_csv(folder / "leaf_rules.csv")
        st.caption("Every end leaf of the tree with the full if-then rule that leads to it, sorted by how many training leaves end there.")
        pick = st.selectbox("Show end leaves that predict", ["any disease"] + DISEASES, key="tree_rule_cls")
        st.dataframe(lr if pick == "any disease" else lr[lr.predicts == pick], hide_index=True, width="stretch", height=420)
    with tabs[4]:
        tp = pd.read_csv(folder / "test_predictions.csv")
        cm = (tp.groupby(["true_class", "predicted_class"]).size().unstack(fill_value=0)
              .reindex(index=DISEASES, columns=DISEASES, fill_value=0).stack().rename("leaves").reset_index())
        cm.columns = ["true_class", "predicted_class", "leaves"]
        cm["row %"] = 100 * cm["leaves"] / cm.groupby("true_class")["leaves"].transform("sum")
        st.caption("Rows: the true disease. Columns: what the tree said. The diagonal is correct.")
        confusion(cm, "true_class", "predicted_class")
    with tabs[5]:
        imp = pd.read_csv(folder / "feature_importance.csv").head(15)
        show(alt.Chart(imp).mark_bar(cornerRadius=3, color="#0072B2").encode(
            x=alt.X("importance_%:Q", title="share of the tree's total Gini decrease (%)"), y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("importance_%:Q", format=".2f")]), height=380)


def confusion(cm: pd.DataFrame, true: str, pred: str) -> None:
    base = alt.Chart(cm).encode(x=alt.X(f"{pred}:N", sort=DISEASES, title="predicted", axis=alt.Axis(labelAngle=0, labelLimit=130)),
                                y=alt.Y(f"{true}:N", sort=DISEASES, title="true"))
    heat = base.mark_rect().encode(color=alt.Color("row %:Q", scale=alt.Scale(scheme="blues", domain=[0, 100]), legend=None),
                                   tooltip=[true, pred, "leaves", alt.Tooltip("row %:Q", format=".1f")])
    text = base.mark_text(fontSize=12).encode(text="leaves:Q", color=alt.condition("datum['row %'] > 55", alt.value("white"), alt.value("#222")))
    show(heat + text, height=300)


# =============================================================================================================
# Units 1-4
# =============================================================================================================
@st.cache_resource(show_spinner="Scoring the best model on the held-out leaves...")
def best_model_eval():
    from src import predict as P
    d = features()
    try:
        m = P.load_models()
    except Exception:
        return None
    if d is None:
        return None
    pred = m["best"]["pipeline"].predict(d["X_valid"])
    names = [short(c) for c in d["class_names"]]
    cm = pd.crosstab(pd.Series([names[i] for i in d["y_valid"]], name="true"), pd.Series([names[i] for i in pred], name="predicted"))
    cm = cm.stack().rename("leaves").reset_index()
    cm["row %"] = 100 * cm["leaves"] / cm.groupby("true")["leaves"].transform("sum")
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(d["y_valid"], pred)
    prf = pd.DataFrame({"disease": names, "precision": 100 * p, "recall": 100 * r, "F1": 100 * f})
    return m["best"]["model_name"], cm, prf


@st.cache_resource(show_spinner="Clustering the training leaves (k-means, GMM)...")
def clusters_vs_classes():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    d = features()
    if d is None:
        return None
    Z = PCA(30, random_state=config.SEED).fit_transform(StandardScaler().fit_transform(d["X_train"]))
    names = [short(c) for c in d["class_names"]]
    out = {}
    for label, model in [("k-means", KMeans(5, n_init=10, random_state=config.SEED)),
                         ("Gaussian mixture (EM)", GaussianMixture(5, covariance_type="full", random_state=config.SEED))]:
        c = model.fit_predict(Z)
        ct = pd.crosstab(pd.Series([names[i] for i in d["y_train"]], name="disease"), pd.Series([f"cluster {k + 1}" for k in c], name="cluster"))
        ct = ct.stack().rename("leaves").reset_index()
        ct["% of disease"] = 100 * ct["leaves"] / ct.groupby("disease")["leaves"].transform("sum")
        out[label] = ct
    return out


def page_units_1_4() -> None:
    st.title("📊 Units 1–4: interactive graphs")
    st.caption("Every chart is drawn from the project's saved results (reports/tables) or the feature cache. "
               "Point at any mark for its value; scroll to zoom and drag to pan on line and scatter charts.")
    t12, t3, t4 = st.tabs(["Units 1 & 2: classifiers", "Unit 3: PCA & clustering", "Unit 4: HMM"])

    with t12:
        res = table("classification_results.csv")
        if res is not None:
            st.subheader("Every classifier: cross-validation vs held-out test")
            d = res[["Model", "Syllabus", "CV accuracy (%)", "Valid accuracy (%)", "Valid macro F1 (%)"]].rename(
                columns={"CV accuracy (%)": "5-fold CV accuracy", "Valid accuracy (%)": "held-out accuracy", "Valid macro F1 (%)": "held-out macro F1"})
            dl = d.melt(["Model", "Syllabus"], var_name="measure", value_name="%")
            show(alt.Chart(dl).mark_bar(cornerRadius=2).encode(
                y=alt.Y("Model:N", sort=d.sort_values("held-out accuracy", ascending=False).Model.tolist(), title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("%:Q", scale=alt.Scale(domain=[50, 100]), title="%"), x2=alt.datum(50), yOffset="measure:N",
                color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="top"),
                                scale=alt.Scale(range=["#0072B2", "#E69F00", "#009E73"])),
                tooltip=["Model", "Syllabus", "measure", alt.Tooltip("%:Q", format=".2f")]), height=420)
            st.caption("Naive Bayes (Unit 1) is last because its independence assumption is false; the RBF kernel beats the "
                       "linear SVM; tree ensembles (Unit 5) are best.")
        ev = best_model_eval()
        if ev is not None:
            name, cm, prf = ev
            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"Confusion matrix: {name}")
                confusion(cm, "true", "predicted")
            with c2:
                st.subheader("Precision, recall, F1 per disease")
                pl = prf.melt("disease", var_name="measure", value_name="%")
                show(alt.Chart(pl).mark_bar(cornerRadius=2).encode(
                    y=alt.Y("disease:N", sort=DISEASES, title=None), x=alt.X("%:Q", scale=alt.Scale(domain=[70, 100])), x2=alt.datum(70), yOffset="measure:N",
                    color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(range=["#0072B2", "#E69F00", "#009E73"])),
                    tooltip=["disease", "measure", alt.Tooltip("%:Q", format=".1f")]), height=300)
        ab = table("ablation.csv")
        if ab is not None:
            st.subheader("Feature ablation: what each feature group adds")
            a = ab.rename(columns={"SVM (RBF kernel) CV acc (%)": "SVM (RBF)", "Logistic Regression CV acc (%)": "Logistic regression"})
            show(lines(a, "# features", {"SVM (RBF)": "SVM (RBF)", "Logistic regression": "Logistic regression"},
                       "number of features (RGB → + indices & HSV → + texture → + shape)", "5-fold CV accuracy (%)",
                       scale=alt.Scale(domain=["SVM (RBF)", "Logistic regression"], range=["#0072B2", "#E69F00"]), points=True), height=280)
        lo = table("ablation_leave_one_group_out.csv")
        if lo is not None:
            show(alt.Chart(lo).mark_bar(cornerRadius=3).encode(
                x=alt.X("Drop (points):Q", title="accuracy lost when this group is removed (points)"), y=alt.Y("Removed group:N", sort="-x", title=None),
                color=alt.condition("datum['Drop (points)'] > 0", alt.value("#D55E00"), alt.value("#009E73")),
                tooltip=["Removed group", "# features removed", alt.Tooltip("CV acc (%):Q", format=".2f"), alt.Tooltip("Drop (points):Q", format=".2f")]),
                height=200)
        corr = table("chlorophyll_rgb_correlation.csv")
        vif = table("chlorophyll_vif.csv")
        if corr is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Unit 1: correlation of colour features")
                cl = corr.melt("Feature", var_name="with", value_name="r")
                base = alt.Chart(cl).encode(x=alt.X("with:N", title=None, sort=list(corr.Feature), axis=alt.Axis(labelAngle=-30)),
                                            y=alt.Y("Feature:N", title=None, sort=list(corr.Feature)))
                show(base.mark_rect().encode(color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True), legend=None),
                                             tooltip=["Feature", "with", alt.Tooltip("r:Q", format=".2f")])
                     + base.mark_text(fontSize=10).encode(text=alt.Text("r:Q", format=".2f")), height=280)
                st.caption("R and G are 91% correlated, GLI and ExG almost 100%: this breaks Naive Bayes' independence assumption.")
            if vif is not None:
                with c2:
                    st.subheader("Variance inflation factor")
                    v = vif.assign(one=1.0)
                    lsc = alt.Scale(type="log", domain=[1, 1000])
                    yv = alt.Y("Feature:N", sort=v.sort_values("VIF", ascending=False).Feature.tolist(), title=None)
                    show(alt.Chart(v).mark_rule(strokeWidth=3, color="#CC79A7").encode(x=alt.X("one:Q", scale=lsc, title="VIF (log scale; dashed line = 10, above it least squares is unstable)"), x2="VIF:Q", y=yv)
                         + alt.Chart(v).mark_point(filled=True, size=90, color="#CC79A7").encode(x=alt.X("VIF:Q", scale=lsc), y=yv, tooltip=["Feature", alt.Tooltip("VIF:Q", format=".1f")])
                         + alt.Chart(pd.DataFrame({"t": [10.0]})).mark_rule(strokeDash=[4, 3], color="#555").encode(x=alt.X("t:Q", scale=lsc)), height=280)
        st.subheader("Unit 2: Bayesian logistic regression (healthy vs diseased)")
        w = table("bayes_weights.csv")
        un = table("bayes_unseen_disease.csv")
        sm = table("bayes_small_data.csv")
        c1, c2 = st.columns(2)
        if w is not None:
            with c1:
                top = w.reindex(w["w_MAP"].abs().sort_values(ascending=False).index).head(15).rename(columns={"CI 2.5%": "ci_low", "CI 97.5%": "ci_high"})
                base = alt.Chart(top).encode(y=alt.Y("Feature:N", sort=alt.EncodingSortField("w_MAP", order="descending"), title=None))
                show(base.mark_rule(strokeWidth=2, color="#999").encode(x=alt.X("ci_low:Q", title="weight with 95% credible interval"), x2="ci_high:Q")
                     + base.mark_point(filled=True, size=70, color="#0072B2").encode(
                         x="w_MAP:Q", tooltip=["Feature", alt.Tooltip("w_MAP:Q", format=".2f"), alt.Tooltip("ci_low:Q", format=".2f", title="2.5%"), alt.Tooltip("ci_high:Q", format=".2f", title="97.5%")]),
                     height=380)
                st.caption("The 15 largest weights. An interval that crosses 0 means the data cannot tell the sign of that feature's effect.")
        if sm is not None:
            with c2:
                show(lines(sm, "Train size", {"MAP log-loss": "plain (MAP)", "Bayes log-loss": "Bayesian predictive"}, "training leaves (log scale)",
                           "log-loss on new leaves (lower = better)", scale=alt.Scale(domain=["plain (MAP)", "Bayesian predictive"], range=["#E69F00", "#0072B2"]),
                           x_log=True, points=True, fmt=".3f"), height=180)
            if un is not None:
                with c2:
                    u = un.assign(disease=un["Hidden disease"].str.replace("Tomato: ", "").str.replace("Tomato ", "").str.capitalize())
                    ul = u.melt("disease", ["MAP log-loss", "Bayes log-loss"], var_name="model", value_name="log-loss")
                    show(alt.Chart(ul).mark_bar(cornerRadius=2).encode(
                        y=alt.Y("disease:N", title="disease never seen in training"), x=alt.X("log-loss:Q"), yOffset="model:N",
                        color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(range=["#0072B2", "#E69F00"])),
                        tooltip=["disease", "model", alt.Tooltip("log-loss:Q", format=".3f")]), height=180)
                    st.caption("On a disease it never saw, the Bayesian model is less over-confident (lower log-loss for late blight and Septoria).")

    with t3:
        pc = table("unsup_pca.csv")
        if pc is not None:
            st.subheader("PCA: how many dimensions do we need?")
            show(lines(pc, "PCA components", {"Variance kept (%)": "variance kept", "LR CV accuracy (%)": "logistic regression accuracy"},
                       "principal components kept", "%", scale=alt.Scale(domain=["variance kept", "logistic regression accuracy"], range=["#0072B2", "#E69F00"]),
                       points=True, y_zero=True, fmt=".1f"), height=280)
            st.caption("30 of 55 components keep 95% of the variance, but accuracy keeps rising to all 55: low-variance directions still carry disease information.")
        space = pca_space()
        if space is not None:
            st.subheader("Training leaves on the first two principal components")
            show(alt.Chart(space[2]).mark_circle(size=22, opacity=0.65).encode(
                x=alt.X("PC1:Q", scale=alt.Scale(zero=False)), y=alt.Y("PC2:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("disease:N", scale=DSCALE, title=None, legend=alt.Legend(orient="top")),
                tooltip=["disease", alt.Tooltip("PC1:Q", format=".2f"), alt.Tooltip("PC2:Q", format=".2f")]).interactive(), height=420)
            st.caption("300 leaves per disease. Healthy leaves form two groups (two photo backgrounds); the brown-spot diseases overlap.")
        el, bic, cl = table("unsup_kmeans_elbow.csv"), table("unsup_bic.csv"), table("unsup_clustering.csv")
        c1, c2 = st.columns(2)
        if el is not None:
            with c1:
                st.subheader("k-means: elbow")
                show(alt.Chart(el).mark_line(point=True, color="#0072B2").encode(x="K:Q", y=alt.Y("Inertia:Q", scale=alt.Scale(zero=False)),
                                                                                tooltip=["K", alt.Tooltip("Inertia:Q", format=",.0f"), alt.Tooltip("Silhouette:Q", format=".3f")]), height=220)
            with c2:
                st.subheader("k-means: silhouette")
                show(alt.Chart(el).mark_line(point=True, color="#E69F00").encode(x="K:Q", y=alt.Y("Silhouette:Q", scale=alt.Scale(zero=False)),
                                                                                tooltip=["K", alt.Tooltip("Silhouette:Q", format=".3f")]), height=220)
        if bic is not None:
            with c1:
                st.subheader("Gaussian mixture: BIC (lower = better)")
                show(alt.Chart(bic).mark_line(point=True, color="#CC79A7").encode(x=alt.X("GMM components:Q"), y=alt.Y("BIC:Q", scale=alt.Scale(zero=False)),
                                                                                 tooltip=["GMM components", alt.Tooltip("BIC:Q", format=",.0f")]), height=220)
        if cl is not None:
            with c2:
                st.subheader("Clusters vs diseases (labels used only to score)")
                cll = cl.melt("Method", ["ARI", "NMI", "Purity"], var_name="score", value_name="value")
                show(alt.Chart(cll).mark_bar(cornerRadius=2).encode(
                    y=alt.Y("Method:N", title=None), x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1])), yOffset="score:N",
                    color=alt.Color("score:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(range=["#0072B2", "#E69F00", "#009E73"])),
                    tooltip=["Method", "score", alt.Tooltip("value:Q", format=".3f")]), height=220)
        cv = clusters_vs_classes()
        if cv is not None:
            st.subheader("Which disease ends up in which cluster")
            which = st.radio("Method", list(cv), horizontal=True, key="u3_cluster_method")
            ct = cv[which]
            base = alt.Chart(ct).encode(x=alt.X("cluster:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("disease:N", sort=DISEASES, title=None))
            show(base.mark_rect().encode(color=alt.Color("% of disease:Q", scale=alt.Scale(scheme="oranges", domain=[0, 100]), legend=None),
                                         tooltip=["disease", "cluster", "leaves", alt.Tooltip("% of disease:Q", format=".1f")])
                 + base.mark_text().encode(text=alt.Text("% of disease:Q", format=".0f"),
                                           color=alt.condition("datum['% of disease'] > 55", alt.value("white"), alt.value("#222"))), height=260)
            st.caption("Each row adds up to 100%. Healthy leaves gather in one cluster; the diseases are spread over several.")

    with t4:
        tr, dec = table("hmm_transitions.csv"), table("hmm_decoding.csv")
        if tr is None:
            st.info("Run  python -m src.hmm  to create the HMM results.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Transition matrix A")
                tl = tr.melt("from \\ to", var_name="to", value_name="p").rename(columns={"from \\ to": "from"})
                order = ["background", "healthy leaf", "lesion"]
                base = alt.Chart(tl).encode(x=alt.X("to:N", sort=order, title="next pixel", axis=alt.Axis(labelAngle=0)), y=alt.Y("from:N", sort=order, title="this pixel"))
                show(base.mark_rect().encode(color=alt.Color("p:Q", scale=alt.Scale(scheme="purples", domain=[0, 1]), legend=None),
                                             tooltip=["from", "to", alt.Tooltip("p:Q", format=".3f")])
                     + base.mark_text(fontSize=13).encode(text=alt.Text("p:Q", format=".2f"), color=alt.condition("datum.p > 0.5", alt.value("white"), alt.value("#222"))),
                     height=230)
                st.caption("Neighbouring pixels are usually the same tissue. Expected run length 1/(1−a): ~100 px background, ~33 healthy, ~5 lesion.")
            with c2:
                st.subheader("Decoding 40 held-out leaves")
                dl = dec[dec.Decoder != "Threshold segmentation (reference labels)"].melt(
                    "Decoder", ["Pixel agreement (%)", "Lesion recall (%)", "Lesion precision (%)"], var_name="measure", value_name="%")
                show(alt.Chart(dl).mark_bar(cornerRadius=2).encode(
                    y=alt.Y("Decoder:N", title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("%:Q", scale=alt.Scale(domain=[0, 100])), yOffset="measure:N",
                    color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(range=["#0072B2", "#E69F00", "#009E73"])),
                    tooltip=["Decoder", "measure", alt.Tooltip("%:Q", format=".2f")]), height=230)
            sw = dec[["Decoder", "State changes per row"]]
            show(alt.Chart(sw).mark_bar(cornerRadius=3, color="#CC79A7").encode(
                x=alt.X("State changes per row:Q", title="tissue changes per row (lower = less speckle)"), y=alt.Y("Decoder:N", sort="x", title=None, axis=alt.Axis(labelLimit=300)),
                tooltip=["Decoder", alt.Tooltip("State changes per row:Q", format=".2f")]), height=170)
            st.caption("The HMM cuts noisy jumps from 3.97 to about 2.5 per row (the reference has 2.85). Upload a leaf on "
                       "'Compare all models' to watch k-means, GMM and Viterbi run on its own pixels.")


# =============================================================================================================
# Unit 5 (+ chlorophyll)
# =============================================================================================================
def page_unit5() -> None:
    st.title("🧠 Unit 5: interactive graphs")
    D = unit5_data()
    if D is None:
        st.error("Run  `python -m src.graph_data`  once in the terminal (about 2 minutes), then reload this page.")
        return
    df = {k: pd.DataFrame(v) for k, v in D.items()}
    st.caption("The data behind each graph, captured from the project's own src/unit5.py run (seed 42). "
               "Blue = training leaves, orange = held-out leaves; the gap between them is over-fitting. Point to read, scroll to zoom.")
    a, b, c, d, e = st.tabs(["Neural networks", "Combining models", "Kernel methods", "Sampling & graphical models", "Chlorophyll"])

    with a:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1 · Error falls while training")
            show(lines(df["nn_curves"], "epoch", {"train_loss": "training leaves", "heldout_loss": "held-out leaves"}, "epoch", "cross-entropy error", fmt=".3f"), 280)
        with c2:
            st.subheader("Accuracy while training")
            show(lines(df["nn_curves"], "epoch", {"train_acc": "training leaves", "heldout_acc": "held-out leaves"}, "epoch", "accuracy (%)", fmt=".1f"), 280)
        st.caption("Both errors drop fast; after ~25 epochs the held-out error stops falling: the network starts memorising.")
        st.subheader("2 · How many hidden units?")
        show(lines(df["nn_sweep"], "hidden_units", {"train_acc": "training leaves", "heldout_acc": "held-out leaves"}, "hidden units (log scale)",
                   "accuracy (%)", x_log=True, points=True, fmt=".1f"), 280)
        st.caption("Up to 32 units both lines rise; after that only training accuracy climbs: over-fitting. That is why we use 32.")
        st.subheader("3 · Straight vs curved decision boundaries")
        model = st.radio("Model", ["Logistic regression", "Neural network"], horizontal=True, key="u5_boundary")
        g = df["nn_grid"][df["nn_grid"].model == model]
        acc = df["nn_acc2d"].set_index("model").accuracy[model]
        dx, dy = float(np.diff(np.sort(g.pc1.unique()))[0]) / 2, float(np.diff(np.sort(g.pc2.unique()))[0]) / 2
        g = g.assign(x1=g.pc1 - dx, x2=g.pc1 + dx, y1=g.pc2 - dy, y2=g.pc2 + dy)
        xsc = alt.Scale(domain=[float(g.x1.min()), float(g.x2.max())], nice=False)
        ysc = alt.Scale(domain=[float(g.y1.min()), float(g.y2.max())], nice=False)
        region = alt.Chart(g).mark_rect(opacity=0.25).encode(
            x=alt.X("x1:Q", title="PC1", scale=xsc), x2="x2:Q", y=alt.Y("y1:Q", title="PC2", scale=ysc), y2="y2:Q",
            color=alt.Color("region:N", scale=DSCALE, legend=None), tooltip=[alt.Tooltip("region:N", title=f"{model} predicts")])
        pts = alt.Chart(df["nn_points"]).mark_circle(size=16, opacity=0.85, clip=True).encode(
            x=alt.X("pc1:Q", scale=xsc), y=alt.Y("pc2:Q", scale=ysc), color=alt.Color("disease:N", scale=DSCALE, title=None, legend=alt.Legend(orient="top")),
            tooltip=["disease", alt.Tooltip("pc1:Q", format=".2f"), alt.Tooltip("pc2:Q", format=".2f")])
        show(region + pts, 420)
        st.caption(f"{model}: {acc:.1f}% on these 1,500 leaves (2 dimensions only). Shaded area = what the model predicts there. "
                   "Logistic regression can only cut with straight lines; the hidden layer bends them. With all 55 features: 93.2% vs 89.3%.")

    with b:
        st.subheader("4 · AdaBoost: error per round")
        r = df["ada_rounds"]
        show(lines(r, "round", {"train_error": "training (ours)", "heldout_error": "held-out (ours)", "sklearn_heldout_error": "held-out (scikit-learn)"},
                   "boosting round (number of stumps)", "error (%)",
                   scale=alt.Scale(domain=["training (ours)", "held-out (ours)", "held-out (scikit-learn)"], range=[TRAIN, HELD, "#333333"]), fmt=".1f",
                   dashed={"held-out (scikit-learn)"}), 300)
        st.caption(f"One stump: {r.heldout_error.iloc[0]:.1f}% error; 300 stumps: {r.heldout_error.iloc[-1]:.1f}%. The scikit-learn line lies on ours: same predictions.")
        c1, c2 = st.columns(2)
        with c1:
            show(lines(r, "round", {"weighted_error": "weighted error ε (%)"}, "boosting round", "ε (%)",
                       scale=alt.Scale(domain=["weighted error ε (%)"], range=["#666666"]), fmt=".1f"), 220)
        with c2:
            show(lines(r, "round", {"alpha": "vote weight α"}, "boosting round", "α", scale=alt.Scale(domain=["vote weight α"], range=["#009E73"]), fmt=".3f"), 220)
        st.caption("Later stumps face harder leaves: their error ε rises and their vote α shrinks (two separate charts, one scale each).")
        st.subheader("5 · Committees vs their members")
        cm = df["committee"]
        avg = cm[cm.kind == "member"].accuracy.mean()
        bars = alt.Chart(cm).mark_bar(cornerRadius=3).encode(
            x=alt.X("accuracy:Q", scale=alt.Scale(domain=[60, 100]), title="held-out accuracy (%)"), x2=alt.datum(60), y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["member", "committee", "boosting"], range=["#9E9E9E", "#0072B2", "#E69F00"]), title=None, legend=alt.Legend(orient="top")),
            tooltip=["model", "kind", alt.Tooltip("accuracy:Q", format=".2f")])
        rule = alt.Chart(pd.DataFrame({"avg": [avg]})).mark_rule(strokeDash=[4, 3], color="black").encode(x="avg:Q", tooltip=[alt.Tooltip("avg:Q", title="average member", format=".1f")])
        show(bars + rule, 320)
        st.caption(f"Dashed line = average member ({avg:.1f}%). Every committee beats it; stacking beats every single member.")

    with c:
        sv = df["svm_c"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("6 · Support vectors kept")
            show(alt.Chart(sv).mark_line(point=alt.OverlayMarkDef(size=70), color=TRAIN).encode(
                x=alt.X("C:Q", scale=alt.Scale(type="log", domain=[0.01, 1000], nice=False), title="C (log scale)"), y=alt.Y("support_vector_pct:Q", title="support vectors (% of 1,500 leaves)"),
                tooltip=["C", "support_vectors", alt.Tooltip("support_vector_pct:Q", format=".1f"), alt.Tooltip("accuracy:Q", format=".2f")]), 240)
        with c2:
            st.subheader("Held-out accuracy")
            show(alt.Chart(sv).mark_line(point=alt.OverlayMarkDef(size=70), color=HELD).encode(
                x=alt.X("C:Q", scale=alt.Scale(type="log", domain=[0.01, 1000], nice=False), title="C (log scale)"), y=alt.Y("accuracy:Q", scale=alt.Scale(zero=False), title="held-out accuracy (%)"),
                tooltip=["C", "support_vectors", alt.Tooltip("accuracy:Q", format=".2f")]), 240)
        st.caption("At C = 10 the SVM keeps only 162 of 1,500 leaves; its decision rebuilt from them matches scikit-learn to 10⁻¹³.")
        st.subheader("7 · Gaussian process: every prediction has an error bar")
        name = df["gp_input"].input.iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            cur, trn = df["gp_curve"], df["gp_train"]
            show(alt.Chart(cur).mark_area(opacity=0.2, color=TRAIN).encode(x=alt.X("x:Q", title=f"{name} (standardised)"), y=alt.Y("lower:Q", title="√ lesion %"), y2="upper:Q")
                 + alt.Chart(trn).mark_circle(size=12, color="#777", opacity=0.45).encode(x="x:Q", y="y:Q", tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("y:Q", format=".2f", title="√ lesion %")])
                 + alt.Chart(cur).mark_line(color=TRAIN, strokeWidth=2.5).encode(x="x:Q", y="mean:Q", tooltip=[alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("mean:Q", format=".2f", title="GP mean"),
                                                                                                       alt.Tooltip("lower:Q", format=".2f"), alt.Tooltip("upper:Q", format=".2f")]), 300)
            st.caption("One input. The band widens where training leaves are few.")
        with c2:
            t = df["gp_test"]
            show(alt.Chart(t).mark_rule(color=TRAIN, opacity=0.3).encode(x=alt.X("rank:Q", title="400 held-out leaves, sorted by prediction"), y=alt.Y("lower:Q", title="√ lesion %"), y2="upper:Q")
                 + alt.Chart(t).mark_line(color=TRAIN).encode(x="rank:Q", y="predicted:Q")
                 + alt.Chart(t).mark_circle(size=14, color=HELD).encode(x="rank:Q", y="true:Q", tooltip=["rank", alt.Tooltip("true:Q", format=".2f"), alt.Tooltip("predicted:Q", format=".2f"),
                                                                                                      alt.Tooltip("lower:Q", format=".2f"), alt.Tooltip("upper:Q", format=".2f"), "inside"]), 300)
            st.caption(f"{100 * t.inside.mean():.1f}% of true values (orange) fall inside their 95% bar: honest error bars.")
        st.subheader("8 · Relevance Vector Machine (from scratch) and sparsity")
        c1, c2 = st.columns(2)
        with c1:
            rv = df["rvm_test"]; lo, hi = float(min(rv.true.min(), rv.lower.min())), float(max(rv.true.max(), rv.upper.max()))
            show(alt.Chart(rv).mark_rule(color="#009E73", opacity=0.2).encode(x=alt.X("true:Q", title="true √ lesion %", scale=alt.Scale(domain=[lo, hi])), y=alt.Y("lower:Q", title="RVM prediction", scale=alt.Scale(domain=[lo, hi])), y2="upper:Q")
                 + alt.Chart(rv).mark_circle(size=18, color="#009E73").encode(x="true:Q", y="predicted:Q", tooltip=[alt.Tooltip("true:Q", format=".2f"), alt.Tooltip("predicted:Q", format=".2f"), alt.Tooltip("lower:Q", format=".2f"), alt.Tooltip("upper:Q", format=".2f")])
                 + alt.Chart(pd.DataFrame({"a": [lo, hi]})).mark_line(strokeDash=[4, 3], color="#777").encode(x="a:Q", y="a:Q"), 300)
        with c2:
            sp = df["sparsity"]
            show(alt.Chart(sp).mark_bar(cornerRadius=3).encode(x=alt.X("leaves_kept:Q", title="training leaves kept in the model"), y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                                                               color=alt.Color("model:N", legend=None, scale=alt.Scale(range=[TRAIN, HELD, "#009E73"])), tooltip=["model", "leaves_kept"])
                 + alt.Chart(sp).mark_text(align="left", dx=4).encode(x="leaves_kept:Q", y=alt.Y("model:N", sort=None), text="leaves_kept:Q"), 200)
            st.caption("The RVM keeps 35 leaves (SVR 447, GP 600) and still gives error bars.")

    with d:
        st.subheader("9 · Metropolis–Hastings MCMC vs the Laplace approximation")
        p = st.radio("Weight", ["w_lesion", "w_contrast", "bias"], horizontal=True, key="u5_mcmc",
                     format_func={"w_lesion": "lesion fraction", "w_contrast": "GLCM contrast", "bias": "bias"}.get)
        tr, lap = df["mcmc_trace"], df["laplace"].set_index("parameter")
        burn = tr[tr.phase == "burn-in"].step.max()
        c1, c2 = st.columns([3, 2])
        with c1:
            show(alt.Chart(pd.DataFrame({"a": [0], "b": [burn]})).mark_rect(color="#ddd", opacity=0.7).encode(x=alt.X("a:Q", title="MCMC step"), x2="b:Q")
                 + alt.Chart(tr).mark_line(strokeWidth=0.8, color=TRAIN).encode(x="step:Q", y=alt.Y(f"{p}:Q", title=p, scale=alt.Scale(zero=False)),
                                                                                 tooltip=["step", "phase", alt.Tooltip(f"{p}:Q", format=".3f")])
                 + alt.Chart(pd.DataFrame({"m": [lap.loc[p, "w_map"]]})).mark_rule(strokeDash=[5, 4], color="black").encode(y="m:Q", tooltip=[alt.Tooltip("m:Q", title="Laplace / MAP", format=".3f")]), 300)
            st.caption("Grey = burn-in (discarded); dashed = Laplace/MAP value. The chain wanders around it without drifting: it has mixed.")
        with c2:
            s = df["mcmc_samples"]
            a_, b_ = lap.loc["w_lesion"], lap.loc["w_contrast"]
            va, vb, cv = a_["var"], b_["var"], a_["cov"]
            tr_, det = va + vb, va * vb - cv * cv
            l1, l2 = tr_ / 2 + np.sqrt(tr_ ** 2 / 4 - det), tr_ / 2 - np.sqrt(tr_ ** 2 / 4 - det)
            ang = np.arctan2(l1 - va, cv) if abs(cv) > 1e-12 else 0.0
            tt = np.linspace(0, 2 * np.pi, 120)
            ell = pd.concat([pd.DataFrame({"x": a_["w_map"] + k * np.sqrt(l1) * np.cos(tt) * np.cos(ang) - k * np.sqrt(l2) * np.sin(tt) * np.sin(ang),
                                           "y": b_["w_map"] + k * np.sqrt(l1) * np.cos(tt) * np.sin(ang) + k * np.sqrt(l2) * np.sin(tt) * np.cos(ang),
                                           "o": np.arange(120), "sd": f"Laplace {k} sd"}) for k in (1, 2)])
            show(alt.Chart(s).mark_circle(size=8, opacity=0.3, color=TRAIN).encode(x=alt.X("w_lesion:Q", title="w (lesion fraction)", scale=alt.Scale(zero=False)),
                                                                                   y=alt.Y("w_contrast:Q", title="w (GLCM contrast)", scale=alt.Scale(zero=False)),
                                                                                   tooltip=[alt.Tooltip("w_lesion:Q", format=".3f"), alt.Tooltip("w_contrast:Q", format=".3f")])
                 + alt.Chart(ell).mark_line(strokeWidth=2).encode(x="x:Q", y="y:Q", order="o:Q", detail="sd:N",
                                                                  color=alt.Color("sd:N", scale=alt.Scale(domain=["Laplace 1 sd", "Laplace 2 sd"], range=[HELD, "#555"]), title=None, legend=alt.Legend(orient="top"))), 300)
            st.caption("MCMC samples fill the Laplace ellipses: the Unit 2 approximation was justified.")
        st.subheader("10 · Gibbs sampling in an Ising random field cleans a noisy lesion mask")
        cells = df["gibbs_cells"]; n = int(max(cells.row.max(), cells.col.max())) + 1
        tissue, lesion = np.array([200, 200, 200]), np.array([213, 94, 0])
        imgs = []
        for key in ["true_lesion", "noisy_lesion", "p_lesion", "cleaned_lesion"]:
            im = np.full((n, n, 3), 255, dtype=np.uint8)
            v = cells[key].to_numpy(dtype=float)[:, None]
            im[cells.row, cells.col] = (tissue + (lesion - tissue) * v).astype(np.uint8)
            imgs.append(np.kron(im, np.ones((4, 4, 1), dtype=np.uint8)))
        cols = st.columns(5)
        if "gibbs_photo" in df:
            cols[0].image(df["gibbs_photo"].data_url.iloc[0], caption="Leaf photo", width="stretch")
        errs = df["gibbs_errors"].wrong_pixels_pct.tolist()
        for col_, im, cap in zip(cols[1:], imgs, ["Lesion mask (segmentation)", f"15% flipped: {errs[0]:.1f}% wrong", "Gibbs: P(lesion)", f"Cleaned: {errs[1]:.2f}% wrong"]):
            col_.image(im, caption=cap, width="stretch")
        st.caption("Orange = lesion. The prior 'neighbouring pixels agree' removes the noise but keeps the real spots: the 2-D version of the Unit 4 HMM.")
        st.subheader("11 · Bayesian network: P(disease | texture = high)")
        c1, c2 = st.columns(2)
        with c1:
            bp = df["bn_posterior"]
            bl = bp.melt("disease", ["exact", "rejection", "likelihood_weighting", "gibbs"], var_name="method", value_name="%")
            show(alt.Chart(bl).mark_bar(cornerRadius=2).encode(
                x=alt.X("disease:N", sort=DISEASES, title=None, axis=alt.Axis(labelAngle=0, labelLimit=90)), y=alt.Y("%:Q", title="probability (%)"), xOffset="method:N",
                color=alt.Color("method:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(domain=["exact", "rejection", "likelihood_weighting", "gibbs"], range=["#222", "#9E9E9E", TRAIN, HELD])),
                tooltip=["disease", "method", alt.Tooltip("%:Q", format=".2f")])
                 + alt.Chart(bp).mark_tick(thickness=2, size=40, color="#009E73").encode(x=alt.X("disease:N", sort=DISEASES), y="prior:Q", tooltip=["disease", alt.Tooltip("prior:Q", format=".1f", title="prior %")]), 300)
            st.caption("Green ticks = prior. Rough texture lowers yellow leaf curl and raises healthy; all samplers land on the exact answer.")
        with c2:
            bc = df["bn_convergence"]
            show(alt.Chart(bc).mark_line(point=True).encode(x=alt.X("samples:Q", scale=alt.Scale(type="log", domain=[100, 30000], nice=False), title="samples N (log)"),
                                                            y=alt.Y("max_error_pts:Q", scale=alt.Scale(type="log"), title="largest error (points, log)"),
                                                            color=alt.Color("method:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(range=["#9E9E9E", TRAIN, HELD])),
                                                            tooltip=["method", "samples", alt.Tooltip("max_error_pts:Q", format=".2f")]).interactive(), 300)
            st.caption("Error shrinks like 1/√N: 100× more samples gives about 10× less error.")

    with e:
        if "spad_test" not in df:
            st.info("Chlorophyll data not available (ArabicaLens photos not downloaded).")
        else:
            sp = df["spad_test"]
            m_ = sp.measured.mean(); r2 = 1 - ((sp.measured - sp.predicted) ** 2).sum() / ((sp.measured - m_) ** 2).sum()
            rmse = float(np.sqrt(((sp.measured - sp.predicted) ** 2).mean()))
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Predicted vs measured (85 held-out coffee leaves)")
                show(alt.Chart(sp.assign(lo=sp.measured - sp.meter_sd, hi=sp.measured + sp.meter_sd)).mark_rule(color="#bbb").encode(x=alt.X("lo:Q", title="measured SPAD (meter)"), x2="hi:Q", y=alt.Y("predicted:Q", title="predicted SPAD (photo)"))
                     + alt.Chart(sp).mark_circle(size=50, color="#009E73").encode(x="measured:Q", y="predicted:Q", tooltip=[alt.Tooltip("measured:Q", format=".1f"), alt.Tooltip("predicted:Q", format=".1f"), alt.Tooltip("residual:Q", format="+.1f"), alt.Tooltip("meter_sd:Q", format=".1f")])
                     + alt.Chart(pd.DataFrame({"a": [0, 72]})).mark_line(strokeDash=[4, 3], color="#777").encode(x="a:Q", y="a:Q"), 320)
                st.caption(f"R² = {r2:.2f}, typical error {rmse:.1f} SPAD (the meter itself varies by ~3.2). Grey bars = meter spread.")
            with c2:
                st.subheader("Residuals")
                show(alt.Chart(sp).mark_circle(size=50, color="#009E73").encode(x=alt.X("predicted:Q", title="predicted SPAD", scale=alt.Scale(zero=False)), y=alt.Y("residual:Q", title="measured − predicted"),
                                                                                tooltip=[alt.Tooltip("predicted:Q", format=".1f"), alt.Tooltip("residual:Q", format="+.1f")])
                     + alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color="#777").encode(y="z:Q"), 320)
                st.caption("No trend: no systematic error, apart from very pale leaves being over-estimated.")
            st.subheader("Which model and features predict chlorophyll best?")
            show(alt.Chart(df["spad_cv"]).mark_point(filled=True, size=110).encode(
                x=alt.X("rmse:Q", title="10-fold CV error, RMSE in SPAD (lower = better)"), y=alt.Y("model:N", sort=alt.EncodingSortField("rmse", op="min"), title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color("features:N", title=None, legend=alt.Legend(orient="top"), scale=alt.Scale(domain=["rgb (Paper 2)", "colour", "colour + HSV histogram", "all 55 features"], range=["#9E9E9E", TRAIN, "#009E73", HELD])),
                tooltip=["model", "features", alt.Tooltip("rmse:Q", format=".2f"), alt.Tooltip("r2:Q", format=".2f")]), 260)
