"""
demo.py - the whole project, explained and run step by step in the terminal.

For every step the demo
  1. explains WHAT the step does and WHY (the idea and the maths in plain words),
  2. shows the real CODE that does it (taken live from the src/ files, with file and line number),
  3. RUNS it on the real leaf data and prints the results,
  4. draws the key result as a chart IN THE TERMINAL, and
  5. lists the saved GRAPHS (PNG, for slides) with one line on how to read each.

Usage (from the plant-disease-ml folder):
    python demo.py                 full guided tour, press Enter between steps   (about 6-8 minutes of compute)
    python demo.py --no-pause      run straight through
    python demo.py --quick         skip the slower live re-runs (Unit 3 clustering, single-leaf HMM) and show
                                   their saved results instead
    python demo.py --list          list the steps
    python demo.py --step 9        run only step 9  (several:  --step 2 --step 9)
    python demo.py --open          also open every graph in the image viewer as it is explained
    python demo.py --no-code       hide the code listings
    python demo.py --retrain       also re-run the full grid search of the 9 classifiers (slow, ~10+ min)

Everything printed is also saved to reports/demo_log.txt.
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

import config  # noqa: E402

# =============================================================================================================
# terminal output helpers
# =============================================================================================================
ARGS = argparse.Namespace(pause=False, code=True, open=False, quick=False, retrain=False, ascii=False)
USE_COLOUR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
if os.name == "nt":
    os.system("")                         # switches on ANSI colour codes in the classic Windows console
ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Tee:
    """Write to the terminal and (without colour codes) to reports/demo_log.txt."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(ANSI.sub("", text))
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def c(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m" if USE_COLOUR else text


def bold(t): return c(t, "1")
def dim(t): return c(t, "2")
def green(t): return c(t, "32")
def yellow(t): return c(t, "33")
def blue(t): return c(t, "34")
def cyan(t): return c(t, "36")
def magenta(t): return c(t, "35")


WIDTH = 100


def rule(ch="─"):
    print(dim((ch if not ARGS.ascii else "-") * WIDTH))


def banner(num: int, title: str, unit: str):
    print()
    bar = "═" if not ARGS.ascii else "="
    print(green(bar * WIDTH))
    print(green(bold(f" STEP {num}  ·  {title}")) + (dim(f"      [{unit}]") if unit else ""))
    print(green(bar * WIDTH))


def section(name: str, colour=cyan):
    print()
    print(colour(bold(f"▶ {name}" if not ARGS.ascii else f"> {name}")))


def para(text: str, indent: int = 2):
    """Print text re-flowed to the terminal width. Lines are joined into paragraphs; a line starting with
    '- ' starts a new bullet; an empty line starts a new paragraph."""
    items: list[str] = []
    new = True
    for line in textwrap.dedent(text).strip("\n").split("\n"):
        s = line.strip()
        if not s:
            items.append("")
            new = True
        elif new or s.startswith(("- ", "* ")):
            items.append(s)
            new = False
        else:
            items[-1] += " " + s
    for it in items:
        if not it:
            print()
            continue
        hang = 2 if it.startswith(("- ", "* ")) else 0
        print(textwrap.fill(it, WIDTH - indent, initial_indent=" " * indent,
                            subsequent_indent=" " * (indent + hang)))


class indented:
    """Indent everything printed inside the block (used when a project module prints its own report)."""

    def __init__(self, pad: str = "    "):
        self.pad = pad

    def write(self, text):
        out = text.replace("\n", "\n" + self.pad)
        if self.at_start and text:
            out = self.pad + out
        self.at_start = text.endswith("\n")
        if self.at_start:
            out = out[: -len(self.pad)]
        return self.old.write(out)

    def flush(self):
        self.old.flush()

    def __getattr__(self, name):
        return getattr(self.old, name)

    def __enter__(self):
        self.old, self.at_start = sys.stdout, True
        sys.stdout = self
        return self

    def __exit__(self, *a):
        sys.stdout = self.old


def formula(text: str):
    for line in textwrap.dedent(text).strip("\n").split("\n"):
        print("      " + yellow(line))


def show_code(obj, max_lines: int = 45, title: str | None = None):
    """Print the real source of a function / class from the project, with its file and line number."""
    if not ARGS.code:
        return
    try:
        lines, start = inspect.getsourcelines(obj)
        file = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT)
    except (OSError, TypeError, ValueError):
        return
    name = title or getattr(obj, "__qualname__", str(obj))
    print()
    print(magenta(f"  ┌─ code: {file}:{start}  ({name})" if not ARGS.ascii else f"  +- code: {file}:{start} ({name})"))
    for i, line in enumerate(lines[:max_lines]):
        text = line.rstrip("\n")
        stripped = text.lstrip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            text = dim(text)
        elif re.match(r"\s*(def|class) ", text):
            text = bold(text)
        print(magenta("  │ ") + dim(f"{start + i:4d} ") + text)
    if len(lines) > max_lines:
        print(magenta("  │ ") + dim(f"     ... {len(lines) - max_lines} more lines in {file}"))
    print(magenta("  └" + "─" * 60))


def bars(labels, values, unit="%", vmax=None, colour=green, digits=1, highlight=None):
    """Horizontal bar chart drawn with block characters."""
    vmax = vmax or max(values) or 1
    w = max(len(str(lab)) for lab in labels)
    full = 50
    block = "█" if not ARGS.ascii else "#"
    for lab, v in zip(labels, values):
        n = int(round(full * max(v, 0) / vmax))
        col = yellow if highlight is not None and lab == highlight else colour
        print(f"    {str(lab):<{w}} │{col(block * n)}{' ' * (full - n)} {v:.{digits}f}{unit}")


def line_chart(series: dict, title: str, xlabel: str = "", ylabel: str = "", xs=None, logx=False):
    """A line chart in the terminal (plotext). Falls back to a list of numbers if plotext is missing."""
    try:
        import plotext as plt
    except ImportError:
        print(dim("    (pip install plotext for terminal line charts)"))
        for name, ys in series.items():
            print(f"    {name}: " + ", ".join(f"{v:.1f}" for v in ys[:: max(1, len(ys) // 12)]))
        return
    plt.clf()
    plt.plotsize(WIDTH - 6, 20)
    plt.theme("clear")
    colours = ["blue", "orange", "green", "magenta"]
    # without colour the series are told apart by their marker instead
    markers = (["braille"] * 4 if USE_COLOUR else ["braille", "+", "o", "x"]) if not ARGS.ascii else \
        ["*", "+", "o", "x"]
    for (name, ys), col, mk in zip(series.items(), colours, markers):
        x = list(xs) if xs is not None else list(range(1, len(ys) + 1))
        plt.plot(x, list(ys), label=name, color=col, marker=mk)
    if logx:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    chart = plt.build()
    if not USE_COLOUR:
        chart = plt.uncolorize(chart)
    print(textwrap.indent(chart, "    "))


GRAPH_NOTES: dict[str, str] = {}


def graphs(topic: str, notes: dict[str, str] | None = None):
    """List the saved PNG graphs of one topic folder, each with how to read it."""
    folder = config.figure_dir(topic)
    files = sorted(folder.glob("*.png"))
    if notes:                                   # show only the graphs this step talks about
        files = [f for f in files if f.stem in notes]
    if not files:
        print(dim(f"    (no graphs yet in {folder.relative_to(ROOT)})"))
        return
    for f in files:
        note = (notes or {}).get(f.stem, "")
        print(f"    {blue('■' if not ARGS.ascii else '*')} {f.relative_to(ROOT)}")
        if note:
            para(note, indent=8)
        if ARGS.open:
            open_file(f)


def open_file(path: Path):
    try:
        if os.name == "nt":
            os.startfile(str(path))                         # noqa: S606 - opens in the default image viewer
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


def table_file(name: str, max_rows: int = 30, columns: list[str] | None = None):
    """Print a saved result table (reports/tables/<name>.csv) as an aligned text table."""
    import pandas as pd

    from src.evaluate import text_table
    path = config.tables_dir() / f"{name}.csv"
    if not path.exists():
        print(dim(f"    ({path.relative_to(ROOT)} not found)"))
        return
    df = pd.read_csv(path)
    if columns:
        df = df[[c_ for c_ in columns if c_ in df.columns]]
    print(textwrap.indent(text_table(df.head(max_rows)), "    "))


def pause():
    if ARGS.pause:
        try:
            ans = input(dim("\n  Press Enter for the next step (q + Enter to quit) ... "))
        except EOFError:
            return
        if ans.strip().lower() == "q":
            raise KeyboardInterrupt


def run_module(args: list[str]):
    """Run `python -m <module> ...` and stream its output (so the module explains itself)."""
    print(dim(f"    $ python -m {' '.join(args)}"))
    proc = subprocess.Popen([sys.executable, "-u", "-m", *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        print("    " + line.rstrip())
    return proc.wait()


def timed(label: str):
    class _T:
        def __enter__(self):
            self.t = time.time()
            print(dim(f"    running: {label} ..."))
            return self

        def __exit__(self, *a):
            print(dim(f"    done in {time.time() - self.t:.1f} s"))
    return _T()


# =============================================================================================================
# shared data
# =============================================================================================================
_DATA = {}


def data():
    if "d" not in _DATA:
        from src import dataset
        _DATA["d"] = dataset.load_features()
    return _DATA["d"]


def short(name: str) -> str:
    from src.unit5 import short as s
    return s(name)


SAMPLE = config.SAMPLES_DIR / "tomato_early_blight.jpg"


# =============================================================================================================
# the steps
# =============================================================================================================
@dataclass
class Step:
    title: str
    unit: str
    fn: object


STEPS: list[Step] = []


def step(title: str, unit: str = ""):
    def deco(fn):
        STEPS.append(Step(title, unit, fn))
        return fn
    return deco


# ----------------------------------------------------------------------------------------------------------
@step("Welcome: the problem, the pipeline and what is ready", "overview")
def s_welcome():
    section("What this project does")
    para("""
    We take ONE photo of a plant leaf and answer three questions: Is the leaf diseased? Which disease is it?
    How much of the leaf is damaged? We use CLASSICAL machine learning only (no deep learning): every photo is
    turned into 55 numbers a person can understand (colour, texture, shape), and models from the syllabus learn
    from those numbers. Because the features are meaningful, every prediction can be explained.
    """)
    section("The pipeline (5 stages)")
    print(textwrap.indent(textwrap.dedent("""\
        leaf photo
          │  ① PREPROCESS   resize to 256×256, Gaussian blur 3×3 (removes sensor noise)
          ▼
          │  ② SEGMENT      HSV colour thresholds (green leaf + brown lesions) → morphological closing
          │                 → largest connected region → fill holes → drop shadows   = "leaf mask"
          ▼
          │  ③ FEATURES     55 numbers from leaf pixels only: colour 6 · indices 4 · HSV histogram 32
          │                 · GLCM texture 6 · Hu shape moments 7
          ▼
          │  ④ MODELS       Unit 1 Naive Bayes · Unit 2 linear models, SVM, Bayesian LR, regression
          │                 · Unit 3 PCA, k-means, GMM/EM, HMM · trees, forests, boosting, KNN
          │                 · Unit 5 neural network, AdaBoost, committees, GP, RVM, MCMC, Gibbs, Bayes nets
          ▼
          ⑤ APP            Streamlit web app + this terminal demo"""), "    "))
    section("What is ready on this computer")
    ok, bad = green("✔" if not ARGS.ascii else "OK"), yellow("✘" if not ARGS.ascii else "--")
    print(f"    {ok} Python {sys.version.split()[0]}")
    try:
        d = data()
        n_tr, n_va = len(d["y_train"]), len(d["y_valid"])
        print(f"    {ok} feature cache: {n_tr + n_va:,} leaf photos already turned into 55 features "
              f"({n_tr:,} for training, {n_va:,} held out for testing)")
        print(f"         classes: " + ", ".join(short(c) for c in d["class_names"]))
        counts = np.bincount(d["y_train"], minlength=len(d["class_names"]))
        print()
        bars([short(c) for c in d["class_names"]], counts.tolist(), unit=" leaves", digits=0)
    except FileNotFoundError:
        print(f"    {bad} no feature cache - download the Kaggle dataset into data/ and run  python -m src.dataset")
    for name, path in [("best classifier", config.BEST_MODEL_PATH), ("Bayesian LR", config.BAYES_MODEL_PATH),
                       ("logistic regression (explanations)", config.LOGREG_MODEL_PATH)]:
        print(f"    {ok if path.exists() else bad} trained model: {name}  ({path.relative_to(ROOT)})")
    n_png = len(list((config.REPORTS_DIR / "figures").rglob("*.png")))
    print(f"    {ok} {n_png} graphs in reports/figures/")
    print(f"    {ok} sample leaves: " + ", ".join(p.name for p in sorted(config.SAMPLES_DIR.glob("*.jpg"))))
    if not any(config.DATA_DIR.iterdir()) if config.DATA_DIR.exists() else True:
        print(dim("    (the raw photos are not in data/ - not needed: everything runs from the feature cache)"))


# ----------------------------------------------------------------------------------------------------------
@step("Image processing: from a photo to 55 features", "image processing + feature representation")
def s_features():
    import cv2

    from src import features as F
    section("What and why")
    para("""
    A model cannot use a photo directly, so we measure the leaf. First we must know WHICH pixels are leaf,
    otherwise the background would be measured too. In HSV colour space the hue (H) separates "green" and
    "brown" no matter how bright the photo is, so two colour thresholds find healthy tissue and lesions.
    Morphological closing fills small gaps, the largest connected region removes specks, hole filling keeps
    dark spots inside the leaf, and dark brown pixels outside the leaf outline are dropped as shadows.
    """)
    show_code(F.segment_leaf, 35)
    section("Run it on a real leaf: " + SAMPLE.name, green)
    img = F.preprocess(F.read_image(SAMPLE))
    seg = F.segment_leaf(img)
    leaf_px = int(np.count_nonzero(seg.mask))
    brown_px = int(np.count_nonzero(seg.brown))
    print(f"    image {img.shape[1]}×{img.shape[0]} pixels, leaf found: {seg.leaf_found}")
    print(f"    leaf pixels   {leaf_px:6,}  ({100 * leaf_px / seg.mask.size:.1f}% of the image)")
    print(f"    lesion pixels {brown_px:6,}  → lesion fraction = {100 * brown_px / max(leaf_px, 1):.1f}% of the leaf")
    x, names = F.extract_features(SAMPLE)
    section("The 55 features of this leaf, by group")
    labels = {"rgb": "Colour averages (RGB)", "indices": "Greenness & lesion indices",
              "hsv": "Colour histogram (HSV)", "texture": "Texture (GLCM)", "shape": "Shape (Hu moments)"}
    for group, idx in F.FEATURE_GROUPS.items():
        vals = ", ".join(f"{names[i]} = {x[i]:.3g}" for i in idx[:4])
        more = f"  (+{len(idx) - 4} more)" if len(idx) > 4 else ""
        print(f"    {bold(labels.get(group, group))}  ({len(idx)} features)")
        print(f"        {vals}{more}")
    para("""
    - Colour averages (RGB mean, std): diseased leaves are less green and more blotchy (higher std).
    - Indices: GLI, G/R ratio and Excess Green measure greenness; lesion fraction = brown ÷ leaf pixels.
    - HSV histogram: the share of leaf pixels in 8 hues × 2 saturations × 2 brightness levels
      (yellowing and browning move pixels into other bins).
    - GLCM texture (Haralick): how grey levels of neighbouring pixels relate - spots make the leaf "rough".
    - Hu moments: 7 shape numbers of the outline that do not change when the leaf is moved, turned or resized.
    """)
    show_code(F.texture_features, 18)
    section("Does one feature already separate the diseases?  Mean lesion fraction per class (training leaves)", green)
    d = data()
    li = d["feature_names"].index("Lesion fraction")
    means = [100 * d["X_train"][d["y_train"] == k, li].mean() for k in range(len(d["class_names"]))]
    order = np.argsort(means)
    bars([short(d["class_names"][k]) for k in order], [means[k] for k in order])
    para("Healthy leaves have almost no brown pixels; the blights have the most. One feature is not enough to "
         "tell all 5 apart, so the models combine all 55.")
    section("Graphs")
    graphs("features", {
        "1_segmentation_examples": "Photo → leaf mask → lesion mask for leaves of every class. Check that the "
                                   "mask follows the leaf edge and the lesions are marked.",
        "2_key_features_by_class": "Distribution of the most useful features per class: where the boxes do "
                                   "not overlap, that feature separates those diseases."})
    del cv2


# ----------------------------------------------------------------------------------------------------------
@step("Unit 1: probability, Bayes' rule and Naive Bayes", "Unit 1")
def s_unit1():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    from src import features as F
    section("What and why")
    para("""
    Bayes' rule turns "how do leaves of each disease look" into "which disease is this leaf":
    """)
    formula("""
    P(class | x) = P(x | class) · P(class) / P(x)
    Naive Bayes assumes the 55 features are independent given the class:
    P(x | class) = Π_j N(x_j | μ_class,j , σ²_class,j)      (one 1-D Gaussian per feature and class)
    """)
    para("""
    Learning = counting: P(class) is the share of training leaves of that class, μ and σ² are the mean and
    variance of each feature inside each class. It is very fast but the independence assumption is wrong here
    (e.g. R, G and B means are strongly correlated), which is why it is the weakest model.
    """)
    section("Run it: Naive Bayes written out by hand, checked against sklearn", green)
    d = data()
    sc = StandardScaler().fit(d["X_train"])
    X, Xv, y, yv = sc.transform(d["X_train"]), sc.transform(d["X_valid"]), d["y_train"], d["y_valid"]
    k = len(d["class_names"])
    prior = np.bincount(y) / len(y)
    mu = np.array([X[y == c].mean(0) for c in range(k)])
    var = np.array([X[y == c].var(0) for c in range(k)]) + 1e-6 * X.var(0).max()
    print("    # our Naive Bayes, 4 lines of NumPy:")
    print(dim("    prior = class counts / N;  mu, var = per-class mean and variance of every feature"))
    print(dim("    log P(c|x) = log prior_c + Σ_j [-½ log(2π var_cj) - (x_j - mu_cj)² / (2 var_cj)]  + const"))

    def log_post(Z):
        ll = -0.5 * (np.log(2 * np.pi * var)[None] + (Z[:, None, :] - mu[None]) ** 2 / var[None]).sum(-1)
        return ll + np.log(prior)[None]
    ours = log_post(Xv).argmax(1)
    gnb = GaussianNB(var_smoothing=1e-6).fit(X, y)
    print(f"    held-out accuracy, ours: {100 * np.mean(ours == yv):.2f}%   sklearn GaussianNB: "
          f"{100 * np.mean(gnb.predict(Xv) == yv):.2f}%   same predictions on "
          f"{100 * np.mean(ours == gnb.predict(Xv)):.1f}% of leaves")
    xs = sc.transform(F.extract_features(SAMPLE)[0][None])
    lp = log_post(xs)[0]
    post = np.exp(lp - lp.max())
    post /= post.sum()
    print(f"\n    Posterior P(class | x) for {SAMPLE.name}:")
    bars([short(n) for n in d["class_names"]], (100 * post).tolist(), vmax=100,
         highlight=short(d["class_names"][int(post.argmax())]))
    guess = d["class_names"][int(post.argmax())]
    if "early_blight" in guess.lower():
        para("Naive Bayes picks the right disease for this leaf.")
    else:
        para(f"Naive Bayes gets this leaf WRONG (it is early blight; it says {short(guess)}). Treating 55 "
             "correlated features as independent counts the same evidence many times over and makes it "
             "over-confident. The later models fix this - the best one gets this leaf right (step 10).")
    section("Unit 1 also: covariance and correlation")
    fn = d["feature_names"]
    rgb = d["X_train"][:, [fn.index("R mean"), fn.index("G mean"), fn.index("B mean")]]
    C = np.corrcoef(rgb.T)
    print(f"    correlation of the leaf's mean R, G, B:  r(R,G) = {C[0, 1]:.2f}   r(R,B) = {C[0, 2]:.2f}   "
          f"r(G,B) = {C[1, 2]:.2f}")
    para("Strongly correlated inputs break Naive Bayes' independence assumption and make ordinary least squares "
         "unstable - that is why the chlorophyll regression (step 3) needs ridge regularisation.")


# ----------------------------------------------------------------------------------------------------------
@step("Unit 2: linear models - logistic regression, LDA, SVM, Bayesian LR, regression", "Unit 2")
def s_unit2():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from src import bayes_logreg, chlorophyll
    section("What and why")
    para("""
    - Logistic regression (discriminative): learns P(disease | x) = softmax(W x + b) directly, by maximising
      the likelihood (MLE) with an L2 penalty λ = 1/C against over-fitting.
    - LDA (generative): models every class as a Gaussian with a SHARED covariance, then uses Bayes' rule; the
      boundaries between classes come out as straight lines (linear discriminant functions).
    - SVM: the boundary with the widest margin to the nearest leaves; the RBF kernel bends it (Unit 5 shows the
      dual / support vectors).
    """)
    formula("""
    logistic regression:  minimise  -Σ_n log p(t_n | x_n, W)  +  λ/2 ||W||²
    LDA:                  p(x | c) = N(x | μ_c, Σ)  →  log P(c | x) is LINEAR in x
    """)
    section("Run it live on all training leaves", green)
    d = data()
    sc = StandardScaler().fit(d["X_train"])
    X, Xv, y, yv = sc.transform(d["X_train"]), sc.transform(d["X_valid"]), d["y_train"], d["y_valid"]
    res = {}
    for name, m in [("Logistic Regression (C=100)", LogisticRegression(C=100, max_iter=5000)),
                    ("LDA", LinearDiscriminantAnalysis())]:
        with timed(name):
            m.fit(X, y)
        res[name] = 100 * np.mean(m.predict(Xv) == yv)
        print(f"    {name:<30} held-out accuracy {res[name]:.2f}%")
    section("Saved grid search + 5-fold cross-validation of all 9 models  (src/classify.py)")
    table_file("classification_results", 12, ["Model", "Syllabus", "Best params", "CV accuracy (%)", "CV std",
                                              "Valid accuracy (%)", "Valid macro F1 (%)"])
    para("CV accuracy = mean over 5 folds of the training set (used to CHOOSE the model); held-out accuracy = "
         "measured once on the 2,326 leaves never used for training or choosing.")

    section("Bayesian logistic regression - written from scratch in NumPy (src/bayes_logreg.py)")
    para("""
    Plain logistic regression gives ONE weight vector. The Bayesian version keeps a whole distribution over
    weights, so it can say how sure it is. Steps: Gaussian prior → MAP weights by Newton's method → Laplace
    approximation (posterior ≈ Gaussian, covariance = inverse Hessian) → predictive probability that averages
    over the weights (probit approximation). The prior strength α is chosen by the evidence, no validation set.
    """)
    formula("""
    prior  w ~ N(0, α⁻¹ I)        MAP  w ← w - H⁻¹ g     g = Φᵀ(p - t) + α w,   H = Φᵀ R Φ + α I
    Laplace  p(w | D) ≈ N(w_MAP, S_N = H⁻¹)        predictive  p(diseased | x) ≈ σ(κ(s²) μ),  κ = (1 + π s²/8)^-½
    """)
    show_code(bayes_logreg.BayesianLogisticRegression.fit, 40)
    with timed("Bayesian logistic regression (fit, evidence, uncertainty tests)"), indented():
        bayes_logreg.run(d)
    section("Graphs")
    graphs("bayes", {
        "1_uncertainty_2d": "Left: P(diseased); right: uncertainty. It is low near the training leaves and grows "
                            "away from them - the model knows what it has not seen.",
        "2_confidence_vs_error": "Wrong answers have low confidence: sending the least-confident leaves to an "
                                 "expert raises the accuracy on the rest.",
        "3_unseen_disease": "One disease hidden from training: the Bayesian predictive is less over-confident "
                            "on it.",
        "4_weights": "Every weight with a 95% credible interval; intervals that cross 0 = not pinned down."})

    section("Linear regression: chlorophyll (SPAD) from colour  (src/chlorophyll.py, Paper 2)")
    para("""
    Paper 2 fits b = A X: SPAD meter value from the leaf's mean colour. Because R, G and B are strongly
    correlated (step 2), ordinary least squares has unstable weights (high VIF); ridge regression (L2) and
    Bayesian ridge fix that. A SPAD model is trained ONLY on real meter readings (spad/spad_readings.csv) -
    the project never invents chlorophyll numbers; until then it reports an uncalibrated greenness index.
    """)
    with timed("chlorophyll analysis"), indented():
        chlorophyll.run(dataset_features=d)
    graphs("chlorophyll", {"1_collinearity": "Correlation of R, G, B and their variance inflation factors: "
                                             "why ridge is needed."})
    graphs("cv", {"2_svm_grid_search": "SVM accuracy for every (C, gamma): the grid search picks the brightest "
                                       "cell by 5-fold CV."})


# ----------------------------------------------------------------------------------------------------------
@step("Unit 3: PCA, k-means, Gaussian mixtures with EM, and HMMs", "Unit 3")
def s_unit3():
    from src import leaf_unsupervised, unsupervised
    section("What and why")
    para("""
    Unsupervised learning finds structure WITHOUT using the disease labels.
    - PCA: rotate the 55 features onto the directions of largest variance; a few components keep most of the
      information (the 55 features are highly redundant).
    - k-means: put leaves into k groups so each leaf is close to its group's centre.
    - Gaussian mixture + EM: soft clusters. E-step = responsibility of each Gaussian for each leaf; M-step =
      re-estimate means, covariances and weights. The log-likelihood never goes down.
    - Hidden Markov Model: a hidden state (background / healthy / lesion) moves along each pixel row; the
      forward-backward algorithm gives P(state) per pixel, Viterbi the single best state sequence.
    """)
    formula("""
    EM  E-step  γ_nk = π_k N(x_n | μ_k, Σ_k) / Σ_j π_j N(x_n | μ_j, Σ_j)
        M-step  N_k = Σ_n γ_nk,  μ_k = Σ_n γ_nk x_n / N_k,  π_k = N_k / N
    HMM Viterbi  ω(z_n) = ln p(x_n | z_n) + max_{z_n-1} [ ln p(z_n | z_n-1) + ω(z_n-1) ]
    """)
    show_code(unsupervised.em_trace, 20)
    section("Run it on all leaves (PCA, k-means, GMM)", green)
    if ARGS.quick:
        print(dim("    --quick: showing the saved results of  python -m src.unsupervised"))
        table_file("unsup_pca", 10)
        table_file("unsup_clustering", 10)
    else:
        with timed("PCA, k-means and GMM/EM on the 55 features"), indented():
            unsupervised.run(data())
    section("Graphs")
    graphs("pca", {"1_pca_scatter_scree": "Leaves on the first 2 principal components (colour = true disease, "
                                          "NOT used by PCA) and the variance kept by each component.",
                   "2_accuracy_vs_components": "Classifier accuracy using only the first m components."})
    graphs("kmeans", {"1_elbow_silhouette": "Choosing k: the elbow of the within-cluster distance and the "
                                            "silhouette score.",
                      "2_clusters_vs_classes": "Which diseases end up in which cluster."})
    graphs("gmm", {"1_bic_and_em": "BIC picks the number of Gaussians; EM's log-likelihood rises every "
                                   "iteration.",
                   "2_clusters_vs_classes": "GMM clusters vs the true diseases."})

    section("Unit 3 on ONE photo: PCA, k-means, GMM and an HMM on the pixels of " + SAMPLE.name, green)
    show_code(leaf_unsupervised.viterbi, 20)
    out = config.REPORTS_DIR / "single_leaf" / SAMPLE.stem
    if ARGS.quick and (out / "results.txt").exists():
        print(dim("    --quick: showing the saved results.txt"))
        for ln in (out / "results.txt").read_text(encoding="utf-8").splitlines():
            print("    " + ln)
    else:
        run_module(["src.leaf_unsupervised", str(SAMPLE.relative_to(ROOT))])
    for f in sorted(out.glob("*.png")):
        print(f"    {blue('■' if not ARGS.ascii else '*')} {f.relative_to(ROOT)}")
        if ARGS.open:
            open_file(f)
    section("HMM on the dataset (forward-backward + Viterbi along image rows, src/hmm.py) - saved results")
    table_file("hmm_decoding", 8)
    graphs("hmm", {"1_viterbi_and_forward_backward_leaves": "Per-pixel states of whole leaves: emission-only vs "
                                                            "forward-backward vs Viterbi (smoother).",
                   "2_one_row_forward_backward_viterbi": "One pixel row: posterior probability of each state "
                                                         "and the Viterbi path.",
                   "3_learned_parameters": "Learned transition matrix A and emission probabilities B."})


# ----------------------------------------------------------------------------------------------------------
@step("Disease classifiers: trees, random forest, boosting, KNN, CV, metrics, ablation", "model selection")
def s_classify():
    import json

    from src import classify
    section("What and why")
    para("""
    Nine classifiers are compared fairly: each is a pipeline StandardScaler → model (scaling learned inside each
    CV fold: no data leakage); its settings are chosen by grid search with 5-fold stratified cross-validation;
    the model with the best CV score is then tested ONCE on held-out leaves.
    - Decision tree: yes/no questions about one feature at a time, chosen by Gini impurity.
    - Random forest: bagging - many trees, each on a bootstrap sample and random feature subsets; averaging
      them lowers the variance.
    - Gradient boosting: trees added one by one, each fitting the errors of the ensemble so far.
    - KNN: majority vote of the k nearest training leaves.
    """)
    show_code(classify.model_specs if hasattr(classify, "model_specs") else classify.ModelSpec, 40)
    if ARGS.retrain:
        section("--retrain: re-running the full grid search (slow)", green)
        run_module(["src.classify"])
        run_module(["src.evaluate"])
    section("Results (5-fold CV and held-out), from src/classify.py", green)
    import pandas as pd
    path = config.tables_dir() / "classification_results.csv"
    if path.exists():
        r = pd.read_csv(path).sort_values("Valid accuracy (%)")
        bars(r["Model"].tolist(), r["Valid accuracy (%)"].tolist(), vmax=100,
             highlight=r.iloc[-1]["Model"])
        scores = json.loads((config.tables_dir() / "classification_scores.json").read_text())
        best = scores["models"][scores["best_model"]]
        print(f"\n    best model by CV: {bold(best['name'])}  ({best['params']})")
        print("    its 5 CV fold accuracies: " + ", ".join(f"{100 * s:.1f}%" if s <= 1 else f"{s:.1f}%"
                                                         for s in best["cv_fold_scores"]))
    section("Per-class precision / recall / F1 of the best model on held-out leaves")
    rep = config.tables_dir() / "classification_report_best.txt"
    if rep.exists():
        for ln in rep.read_text(encoding="utf-8").splitlines():
            print("    " + ln)
    para("Precision = of the leaves we CALLED this disease, how many really had it. Recall = of the leaves that "
         "really had it, how many we found. F1 = their harmonic mean.")
    section("Ablation: which feature groups matter?")
    table_file("ablation_leave_one_group_out", 8)
    section("Graphs")
    graphs("tree", {"1_depth_vs_overfitting": "Deeper trees fit training leaves perfectly but held-out "
                                              "accuracy stops improving: over-fitting.",
                    "2_tree_depth3": "The first 3 levels of a tree: readable if/else rules."})
    graphs("rf", {"1_accuracy_vs_number_of_trees": "More trees = lower variance; accuracy levels off.",
                  "2_feature_importance": "Which features the forest uses most."})
    graphs("boosting", {"1_accuracy_per_round": "Gradient boosting: accuracy after every added tree.",
                        "2_bagging_vs_boosting": "Bagging (parallel, lowers variance) vs boosting "
                                                 "(sequential, lowers bias)."})
    graphs("knn", {"1_accuracy_vs_k": "Small k = jumpy boundary (over-fit), large k = too smooth."})
    graphs("cv", {"1_model_comparison": "All 9 models: CV accuracy with its spread, and held-out accuracy.",
                  "3_learning_curve_best_model": "Accuracy vs number of training leaves: would more data help?"})
    graphs("metrics", {"1_confusion_matrix": "Rows = true disease, columns = predicted: off-diagonal cells are "
                                             "the confusions.",
                       "2_precision_recall_f1": "Per-class precision, recall and F1.",
                       "3_roc_curves": "One-vs-rest ROC curves; area under the curve near 1 = excellent."})
    graphs("ablation", {"1_cumulative_ablation": "Accuracy as feature groups are added one at a time.",
                        "2_leave_one_group_out": "Accuracy lost when one group is removed."})


# ----------------------------------------------------------------------------------------------------------
@step("Unit 5 (1/4): neural networks - multilayer perceptron and backpropagation", "Unit 5")
def s_nn():
    from src import unit5 as U
    section("What and why")
    para("""
    A feed-forward neural network stacks two layers of weights with a non-linear function in between. The hidden
    tanh units learn their OWN features from the 55 inputs, so the network can draw curved class boundaries -
    something logistic regression (a network with no hidden layer) cannot do. Training = gradient descent on
    the cross-entropy error; error BACKPROPAGATION computes all the gradients in one backward pass using the
    chain rule. We check backprop against numerical derivatives ("gradient check").
    """)
    formula("""
    forward    z = tanh(x W1 + b1)          p = softmax(z W2 + b2)
    error      E = -(1/N) Σ_n Σ_k t_nk ln p_nk + λ/2 (||W1||² + ||W2||²)
    backprop   δ2 = (p - t)/N     δ1 = (δ2 W2ᵀ) ⊙ (1 - z²)     ∂E/∂W2 = zᵀ δ2 + λW2    ∂E/∂W1 = xᵀ δ1 + λW1
    """)
    show_code(U.MLP.gradients, 12)
    show_code(U.MLP.fit, 30)
    section("Run it on all training leaves", green)
    with indented():
        r = U.part_nn(data(), config.REPORTS_DIR)
    h = r["net"].history
    section("Terminal chart: accuracy while training (60 epochs)")
    line_chart({"training leaves": h["train_acc"], "held-out leaves": h["valid_acc"]},
               "Accuracy (%) per epoch", "epoch", "accuracy %")
    section("How to read it")
    para("""
    Both curves rise as backprop lowers the error. The gap between them is over-fitting: the network keeps
    improving on leaves it has seen, while held-out accuracy levels off. Weight decay (λ) and fewer hidden
    units reduce that gap; the hidden-unit sweep shows the trade-off: 1-2 units are too simple, beyond ~64
    the held-out accuracy no longer improves.
    """)
    section("Graphs")
    graphs("nn", {"1_training_curves": "Cross-entropy error and accuracy per epoch for training and held-out "
                                       "leaves.",
                  "2_accuracy_vs_hidden_units": "Model complexity: train vs held-out accuracy as the hidden "
                                                "layer grows.",
                  "3_decision_boundaries_2d": "Same 2 inputs (PCA): logistic regression can only draw straight "
                                              "boundaries, the network draws curved ones."})


@step("Unit 5 (2/4): combining models - AdaBoost, committees, stacking", "Unit 5")
def s_combining():
    from src import unit5 as U
    section("What and why")
    para("""
    Several models together are usually better than one. A COMMITTEE averages the members' votes or
    probabilities; their independent mistakes partly cancel. STACKING trains a second model on the members'
    outputs, so it learns which member to trust. BOOSTING builds the committee one member at a time: every new
    weak learner (here a one-question decision stump) is trained on re-weighted leaves where the mistakes of
    the previous ones count more. (Bagging, random forests and gradient boosting are in step 5.)
    """)
    formula("""
    AdaBoost (SAMME)  ε_m = Σ_{n wrong} w_n / Σ_n w_n        α_m = ln((1 - ε_m)/ε_m) + ln(K - 1)
                      w_n ← w_n · exp(α_m · [y_m(x_n) ≠ t_n])       predict: argmax_c Σ_m α_m [y_m(x) = c]
    """)
    show_code(U.AdaBoostSAMME.fit, 42)
    section("Run it on all training leaves", green)
    with indented():
        r = U.part_combining(data(), config.REPORTS_DIR)
    section("Terminal chart: AdaBoost held-out error as stumps are added")
    line_chart({"training": r["ada"].history["train_error"], "held-out": r["ada"].history["valid_error"]},
               "AdaBoost error (%) vs boosting round", "round m", "error %")
    section("Terminal chart: single members vs committees (held-out accuracy)")
    t = r["table"]
    bars(t["Model"].tolist(), t["Valid accuracy (%)"].tolist(), vmax=100,
         highlight=t.loc[t["Valid accuracy (%)"].idxmax(), "Model"])
    section("Graphs")
    graphs("combining", {"1_adaboost_rounds": "Left: error falls as stumps are added (ours and sklearn overlap "
                                              "exactly). Right: later stumps face harder leaves, so their error "
                                              "ε_m is higher and their vote α_m smaller.",
                         "2_committee_vs_members": "Grey = single models, blue = committees, orange = boosting; "
                                                   "dashed line = average member."})


@step("Unit 5 (3/4): kernel methods - SVM dual, Gaussian processes, Relevance Vector Machine", "Unit 5")
def s_kernels():
    from src import unit5 as U
    section("What and why")
    para("""
    A KERNEL k(x, x') measures how similar two leaves are; models written in terms of kernels can draw very
    flexible boundaries without ever computing features in the high-dimensional space (kernel trick).
    - SVM (dual form): the prediction is a weighted sum of kernels to the SUPPORT VECTORS only - a sparse
      kernel machine. We rebuild the decision function by hand from the dual coefficients.
    - Gaussian process: a probability distribution over functions; every prediction comes with an error bar.
      Its kernel settings (one length-scale per input = automatic relevance determination) are learned by
      maximising the marginal likelihood.
    - Relevance Vector Machine: a Bayesian kernel model with one prior precision per training leaf; most
      precisions go to infinity, so very few "relevance vectors" are kept - sparser than the SVM.

    Task for GP / RVM / SVR: estimate disease SEVERITY (lesion % of the leaf) from colour + texture only.
    """)
    formula("""
    SVM dual     y(x) = Σ_{n ∈ SV} a_n t_n k(x, x_n) + b         RBF  k(x, x') = exp(-γ ||x - x'||²)
    GP           mean = k*ᵀ (K + σ²I)⁻¹ t        variance = k(x,x) - k*ᵀ (K + σ²I)⁻¹ k*
    RVM          Σ = (A + β ΦᵀΦ)⁻¹,  m = β Σ Φᵀ t,   γ_i = 1 - α_i Σ_ii,   α_i ← γ_i / m_i²
    """)
    show_code(U.svm_decision_by_hand, 6)
    show_code(U.RVMRegression.fit, 34)
    section("Run it", green)
    with indented():
        r = U.part_kernels(data(), config.REPORTS_DIR)
    section("Terminal chart: how many training leaves each model keeps")
    reg = r["reg"]
    bars([m.split(" (")[0] for m in reg["Model"]], reg["Basis vectors kept"].tolist(), unit=" leaves", digits=0)
    section("Terminal chart: SVM support vectors (%) vs C")
    sv = r["sv_table"]
    bars([f"C = {v:g}" for v in sv["C"]], sv["Support vectors (% of leaves)"].tolist())
    section("Graphs")
    graphs("kernels", {"1_svm_support_vectors_vs_C": "Larger C = fewer margin violations allowed = fewer "
                                                     "support vectors; held-out accuracy on the other axis.",
                       "2_gaussian_process_regression": "Left: a GP on one input with its 95% band. Right: all "
                                                        "held-out leaves with their error bars; ~95% of true "
                                                        "values fall inside.",
                       "3_rvm_vs_svr_sparsity": "RVM predictions vs truth, and how many training leaves GP, SVR "
                                                "and RVM keep."})


@step("Unit 5 (4/4): sampling methods and graphical models - MCMC, Gibbs, MRF, Bayesian network", "Unit 5")
def s_sampling():
    from src import unit5 as U
    section("What and why")
    para("""
    When a posterior cannot be written in closed form we SAMPLE from it.
    - Metropolis-Hastings MCMC: propose a small random step, accept it with probability min(1, p(new)/p(old)).
      Only ratios are needed, never the normalising constant. We sample the posterior of Bayesian logistic
      regression and compare with the Laplace (Gaussian) approximation of Unit 2.
    - Markov random field + Gibbs sampling: an undirected graphical model on the pixel grid (Ising model) says
      neighbouring pixels usually have the same label. Gibbs sampling resamples one pixel at a time from its
      conditional given its 4 neighbours. We use it to clean a noisy lesion mask of a real leaf.
    - Bayesian network: a directed graphical model Disease → lesion level, yellowing, texture, with
      lesion → texture. Its tables are learned by counting. We answer queries EXACTLY by summing the joint
      distribution, then approximately with rejection sampling, likelihood weighting and Gibbs sampling.
    """)
    formula("""
    MH accept      min(1, p(w*) / p(w))                   Ising  E(x) = -β Σ_{i~j} x_i x_j - η Σ_i x_i y_i
    Gibbs (pixel)  p(x_i = +1 | rest) = σ(2(β Σ_{j~i} x_j + η y_i))
    Bayes net      p(D,L,Y,T) = p(D) p(L|D) p(Y|D) p(T|D,L)      p(D | T) ∝ Σ_L Σ_Y p(D,L,Y,T)
    """)
    show_code(U.metropolis_hastings, 20)
    show_code(U.ising_gibbs, 28)
    show_code(U.DiscreteBayesNet.exact_posterior, 12)
    section("Run it", green)
    with indented():
        r = U.part_sampling(data(), config.REPORTS_DIR)
    section("Terminal chart: P(Disease | texture = high) from the Bayesian network")
    p = r["posterior"]
    bars(p["Disease"].tolist(), p["Exact (enumeration)"].tolist(), vmax=100)
    eb, ea = r["ising"]
    section("Terminal chart: wrong pixels in the lesion mask")
    bars(["noisy labels", "after Gibbs / MRF"], [eb, ea], colour=yellow, digits=2)
    section("Graphs")
    graphs("sampling", {"1_mcmc_vs_laplace": "Left: MCMC traces (after the grey burn-in they wander around the "
                                             "posterior). Right: posterior samples vs the Laplace Gaussian "
                                             "ellipses - they agree, so Laplace was a good approximation.",
                        "2_gibbs_mrf_lesion_denoising": "Photo → true lesion mask → 15% noise → Gibbs posterior "
                                                        "P(lesion) → cleaned mask.",
                        "3_bayes_net_inference": "The network, exact vs sampled answers to one query, and the "
                                                 "sampling error falling like 1/√N."})


# ----------------------------------------------------------------------------------------------------------
@step("Diagnose the sample leaves and ask the assistant", "the finished product")
def s_predict():
    from src import assistant as A
    from src import predict as P
    section("What happens when a farmer uploads a photo")
    para("""
    The photo goes through the same pipeline: segment → 55 features → the best classifier gives the top-3
    diseases; the Bayesian LR adds P(diseased) with uncertainty ("refer to an expert" if unsure); logistic
    regression explains which features pushed the decision. Chlorophyll is only reported as SPAD once real
    meter readings exist; until then an uncalibrated greenness index is shown.
    """)
    show_code(P.predict, 45)
    try:
        models = P.load_models()
    except P.ModelsNotTrainedError as e:
        print(yellow(f"    {e}"))
        return
    for path in sorted(config.SAMPLES_DIR.glob("*.jpg")):
        section(f"Leaf photo: {path.name}", green)
        pred = P.predict(str(path), models)
        print(textwrap.indent(P.format_prediction(pred), "    "))
    section("Ask Leaf Doctor (TF-IDF + nearest-neighbour question matching, answers from the real results)")
    pred = P.predict(str(SAMPLE), models)
    ctx = A.Context(prediction=pred, models=models, source=SAMPLE.name)
    for q in ["What disease is this?", "How confident are you?", "Which model is best?"]:
        ans = A.ask(q, ctx)
        print(f"    {bold('Q:')} {q}   {dim(f'(matched intent: {ans.intent}, similarity {ans.similarity:.2f})')}")
        para(ans.text.replace("**", ""), indent=7)


@step("Summary: every graph, and how to run each part yourself", "wrap-up")
def s_summary():
    section("Every graph in the project (reports/figures/)")
    total = 0
    for topic, folder in config.FIG.items():
        files = sorted((config.REPORTS_DIR / "figures" / folder).glob("*.png"))
        total += len(files)
        if files:
            print(f"    {folder:<36} {len(files)} graph{'s' if len(files) != 1 else ''}")
    print(f"    {'':<36} {bold(str(total))} graphs in total  (+ single-leaf graphs in reports/single_leaf/)")
    section("Syllabus → where it is")
    for unit, items in [
        ("Unit 1", "Bayes' rule, Naive Bayes, covariance / correlation           → step 2"),
        ("Unit 2", "logistic regression, LDA, SVM, Bayesian LR, ridge regression → step 3"),
        ("Unit 3", "PCA, k-means, GMM + EM, HMM forward-backward + Viterbi        → step 4"),
        ("Models", "trees, bagging, random forest, boosting, KNN, CV, metrics     → step 5"),
        ("Unit 5", "neural networks, combining models, kernel methods + sparse    → steps 6-9\n"
                   "                kernel machines, sampling methods + graphical models"),
    ]:
        print(f"    {bold(unit)}  {items}")
    section("Run any part yourself")
    for cmd, what in [
        ("streamlit run app.py", "web app: diagnose a photo, browse every result and graph"),
        ("python -m src.predict samples/tomato_early_blight.jpg", "diagnose one photo"),
        ("python -m src.unit5", "all of Unit 5 (or --part nn | combining | kernels | sampling)"),
        ("python -m src.evaluate --all", "regenerate every graph"),
        ("python -m src.assistant --image samples/tomato_healthy.jpg", "chat with Leaf Doctor in the terminal"),
        ("python -m pytest -q", "run all the tests"),
    ]:
        print(f"    {cyan(cmd):<70} {dim(what)}")
    print()
    print(dim(f"    The full text of this demo is saved in {(config.REPORTS_DIR / 'demo_log.txt').relative_to(ROOT)}"))


# =============================================================================================================
def main(argv=None) -> int:
    global USE_COLOUR, WIDTH
    ap = argparse.ArgumentParser(description="Guided terminal demo of the whole project.")
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--step", type=int, action="append", help="run only this step number (can repeat)")
    ap.add_argument("--no-pause", action="store_true", help="do not wait for Enter between steps")
    ap.add_argument("--quick", action="store_true", help="show saved results for the slower Unit 3 runs")
    ap.add_argument("--retrain", action="store_true", help="also re-run the full classifier grid search (slow)")
    ap.add_argument("--open", action="store_true", help="open every graph in the image viewer")
    ap.add_argument("--no-code", action="store_true", help="hide the code listings")
    ap.add_argument("--no-colour", "--no-color", action="store_true", help="plain text, no colours")
    ap.add_argument("--ascii", action="store_true", help="ASCII only (for terminals that cannot show ▶ █ ─)")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    try:
        import shutil
        WIDTH = max(80, min(110, shutil.get_terminal_size((100, 30)).columns - 2))
    except OSError:
        pass
    if a.no_colour:
        USE_COLOUR = False
    ARGS.pause = not a.no_pause and sys.stdin.isatty() and not a.list
    ARGS.code, ARGS.open, ARGS.quick, ARGS.retrain, ARGS.ascii = not a.no_code, a.open, a.quick, a.retrain, a.ascii

    if a.list:
        for i, s in enumerate(STEPS):
            print(f"  {i:2d}  {s.title}" + (f"   [{s.unit}]" if s.unit else ""))
        return 0
    chosen = a.step if a.step else list(range(len(STEPS)))
    bad = [n for n in chosen if not 0 <= n < len(STEPS)]
    if bad:
        print(f"No step {bad}; use --list to see the steps (0-{len(STEPS) - 1}).")
        return 2

    sys.stdout = Tee(sys.stdout, config.REPORTS_DIR / "demo_log.txt")
    config.set_seeds()
    import warnings
    warnings.filterwarnings("ignore")
    t0 = time.time()
    try:
        for n in chosen:
            s = STEPS[n]
            banner(n, s.title, s.unit)
            t = time.time()
            try:
                s.fn()
            except FileNotFoundError as e:
                print(yellow(f"\n    Skipped - a required file is missing: {e}"))
            print(dim(f"\n    (step {n} took {time.time() - t:.1f} s)"))
            if n != chosen[-1]:
                pause()
    except KeyboardInterrupt:
        print("\n  Demo stopped.")
    print(dim(f"\n  Total time {time.time() - t0:.0f} s."))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
