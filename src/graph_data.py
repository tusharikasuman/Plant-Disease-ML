"""
src/graph_data.py - save the data behind every Unit 5 graph (and the chlorophyll graphs) as one JSON file,
so the Streamlit app can draw them as interactive charts (hover for values, zoom) instead of pictures.

It runs the real src/unit5.py and src/chlorophyll_coffee.py code (seed 42). Their graph functions are
replaced by functions that capture the data they would plot, so nothing is drawn, and their tables and
models are written to a temporary folder (the project's own reports/tables and models are not touched).

Usage (from the plant-disease-ml folder, about 2 minutes):
    python -m src.graph_data            -> reports/graph_data/unit5.json
"""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from src import chlorophyll_coffee as CC
from src import dataset, unit5

OUT = config.REPORTS_DIR / "graph_data" / "unit5.json"
G: dict = {}
r = lambda a, d=3: np.round(np.asarray(a, dtype=float), d).tolist()

r = lambda a, d=3: np.round(np.asarray(a, dtype=float), d).tolist()


def nn_capture(net, sweep, X, y, class_names, reports_dir):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    h = net.history
    G["nn_curves"] = {k: r(v, 4) for k, v in h.items()}
    G["nn_sweep"] = {"hidden": sweep["Hidden units"].tolist(), "train": r(sweep["Train accuracy (%)"], 2),
                     "valid": r(sweep["Valid accuracy (%)"], 2)}
    Z = PCA(2, random_state=config.SEED).fit_transform(X)
    idx = unit5.stratified_idx(y, 300)
    Z, yy = Z[idx], y[idx]
    k = len(class_names)
    lin = LogisticRegression(max_iter=5000).fit(Z, yy)
    mlp = unit5.MLP(2, 32, k, weight_decay=1e-4).fit(Z, yy, epochs=300, batch_size=64, lr=1e-2)
    gx = np.linspace(np.percentile(Z[:, 0], 1) - 1, np.percentile(Z[:, 0], 99) + 1, 90)
    gy = np.linspace(np.percentile(Z[:, 1], 1) - 1, np.percentile(Z[:, 1], 99) + 1, 90)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    G["nn_boundary"] = {
        "classes": [unit5.short(c) for c in class_names], "gx": r(gx), "gy": r(gy),
        "lr_grid": lin.predict(grid).astype(int).tolist(), "nn_grid": mlp.predict(grid).astype(int).tolist(),
        "lr_acc": round(100 * float(np.mean(lin.predict(Z) == yy)), 1),
        "nn_acc": round(100 * float(np.mean(mlp.predict(Z) == yy)), 1),
        "points": [[round(float(a), 3), round(float(b), 3), int(c)] for (a, b), c in zip(Z, yy)]}
    return []


def combining_capture(ada, sk, Xv, yv, table, member_err, reports_dir):
    h = ada.history
    G["ada"] = {"train": r(h["train_error"], 2), "valid": r(h["valid_error"], 2),
                "sklearn": r([100 * np.mean(p != yv) for p in sk.staged_predict(Xv)], 2),
                "eps": r(100 * np.array(h["weighted_error"]), 2), "alpha": r(h["alpha"], 3)}
    G["committee"] = {"rows": [{"model": m, "kind": kd, "acc": round(float(a), 2)} for m, kd, a in
                               zip(table["Model"], table["Kind"], table["Valid accuracy (%)"])],
                      "avg_member": round(100 - float(member_err), 2)}
    return []


def kernel_capture(sv_table, te_t, gp_mean, gp_std, rvm_mean, rvm_std, rvm, svr, Xr, tr_t, reports_dir):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    G["svm"] = {"C": sv_table["C"].tolist(), "sv_pct": r(sv_table["Support vectors (% of leaves)"], 2),
                "sv": sv_table["Support vectors"].tolist(), "acc": r(sv_table["Valid accuracy (%)"], 2)}
    j = int(np.argmax([abs(np.corrcoef(Xr[:, i], tr_t)[0, 1]) for i in range(Xr.shape[1])]))
    x1 = Xr[:, [j]]
    gp1 = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.5), normalize_y=True,
                                   random_state=config.SEED).fit(x1, tr_t)
    gx = np.linspace(x1.min() - 0.5, x1.max() + 0.5, 120)[:, None]
    m1, s1 = gp1.predict(gx, return_std=True)
    o = np.argsort(gp_mean)
    G["gp"] = {"input": unit5.LESION_INPUTS[j], "gx": r(gx[:, 0]), "mean": r(m1), "sd": r(s1),
               "train_x": r(x1[:, 0]), "train_y": r(tr_t),
               "test_mean": r(gp_mean[o]), "test_sd": r(gp_std[o]), "test_true": r(te_t[o]),
               "coverage": round(100 * float(np.mean(np.abs(te_t - gp_mean) <= 1.96 * gp_std)), 1)}
    G["rvm"] = {"true": r(te_t), "pred": r(rvm_mean), "sd": r(rvm_std),
                "counts": {"Gaussian process (all training leaves)": int(len(tr_t)),
                           "SVR (support vectors)": int(len(svr.support_)),
                           "RVM (relevance vectors)": int(len(rvm.relevance_idx_))}}
    return []


def sampling_capture(samples, burn, w_map, S, feats, img, noisy, post_lesion, clean, truth, region, energy,
                     err_before, err_after, bn, names, est, conv, reports_dir):
    step = 30
    G["mcmc"] = {"labels": ["bias", f"w ({feats[0]})", f"w ({feats[1]})"], "burn": int(burn), "step": step,
                 "trace": [r(samples[::step, i]) for i in range(3)], "w_map": r(w_map),
                 "post": r(samples[burn::12, 1:]), "cov": r(S[1:, 1:], 5)}
    half = lambda a: a[::2, ::2]
    reg = half(region)
    enc = lambda m: "".join("." if not g else ("1" if v > 0 else "0") for v, g in zip(half(m).ravel(), reg.ravel()))
    rgb = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    jpg = base64.b64encode(cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
    G["gibbs"] = {"size": int(reg.shape[0]), "photo": "data:image/jpeg;base64," + jpg,
                  "truth": enc(truth), "noisy": enc(noisy), "clean": enc(clean),
                  "post": [None if not g else round(float(v), 2) for v, g in zip(half(post_lesion).ravel(), reg.ravel())],
                  "err_before": round(float(err_before), 2), "err_after": round(float(err_after), 2)}
    G["bayesnet"] = {"diseases": [unit5.short(c) for c in names], "prior": r(100 * bn.pD, 2),
                     "estimates": {k: r(100 * v, 2) for k, v in est.items()},
                     "conv": {m: {"n": conv[conv["Method"] == m]["Samples"].tolist(),
                                  "err": r(100 * conv[conv["Method"] == m]["Max error"], 3)}
                              for m in conv["Method"].unique()}}
    return []


def spad_capture(y, tr, te, p_test, gp_mean, gp_std, cv_table, gli, readings, best_model, best_fs, reports_dir):
    sd = readings["spad_sd"].to_numpy()[te]
    G["spad"] = {"measured": r(y[te], 2), "predicted": r(p_test, 2), "meter_sd": r(sd, 2),
                 "gp_mean": r(gp_mean, 2), "gp_sd": r(gp_std, 2), "gli": r(gli, 4), "spad_all": r(y, 2),
                 "best": f"{best_model} on {best_fs}",
                 "cv": [{"features": f, "model": m, "rmse": round(float(v), 3), "r2": round(float(q), 3)}
                        for f, m, v, q in zip(cv_table["Features"], cv_table["Model"],
                                              cv_table["CV RMSE (SPAD)"], cv_table["CV R2"])]}
    return []



def to_rows(G: dict) -> dict:
    """Reshape the captured data into one table (list of row dicts) per chart."""
    P = {}

    c = G["nn_curves"]
    P["nn_curves"] = [{"epoch": i + 1, "train_loss": c["train_loss"][i], "heldout_loss": c["valid_loss"][i],
                       "train_acc": c["train_acc"][i], "heldout_acc": c["valid_acc"][i]} for i in range(len(c["train_loss"]))]
    s = G["nn_sweep"]
    P["nn_sweep"] = [{"hidden_units": h, "train_acc": a, "heldout_acc": b} for h, a, b in zip(s["hidden"], s["train"], s["valid"])]
    b = G["nn_boundary"]
    names = b["classes"]
    gx, gy, n = b["gx"], b["gy"], len(b["gx"])
    grid = []
    for model, key in [("Logistic regression", "lr_grid"), ("Neural network", "nn_grid")]:
        for j in range(0, n, 2):
            for i in range(0, n, 2):
                grid.append({"model": model, "pc1": gx[i], "pc2": gy[j], "region": names[b[key][j * n + i]]})
    P["nn_grid"] = grid
    P["nn_points"] = [{"pc1": x, "pc2": y, "disease": names[k]} for x, y, k in b["points"]]
    P["nn_acc2d"] = [{"model": "Logistic regression", "accuracy": b["lr_acc"]}, {"model": "Neural network", "accuracy": b["nn_acc"]}]

    a = G["ada"]
    P["ada_rounds"] = [{"round": i + 1, "train_error": a["train"][i], "heldout_error": a["valid"][i],
                        "sklearn_heldout_error": a["sklearn"][i], "weighted_error": a["eps"][i], "alpha": a["alpha"][i]}
                       for i in range(len(a["train"]))]
    P["committee"] = [{"model": r["model"], "kind": r["kind"], "accuracy": r["acc"]} for r in G["committee"]["rows"]]

    v = G["svm"]
    P["svm_c"] = [{"C": C, "support_vectors": n_, "support_vector_pct": p, "accuracy": q}
                  for C, n_, p, q in zip(v["C"], v["sv"], v["sv_pct"], v["acc"])]
    g = G["gp"]
    P["gp_curve"] = [{"x": x, "mean": m, "lower": round(m - 1.96 * sd, 3), "upper": round(m + 1.96 * sd, 3)}
                     for x, m, sd in zip(g["gx"], g["mean"], g["sd"])]
    P["gp_train"] = [{"x": x, "y": y} for x, y in zip(g["train_x"], g["train_y"])]
    P["gp_test"] = [{"rank": i + 1, "predicted": m, "lower": round(m - 1.96 * sd, 3), "upper": round(m + 1.96 * sd, 3),
                     "true": t, "inside": int(abs(t - m) <= 1.96 * sd)}
                    for i, (m, sd, t) in enumerate(zip(g["test_mean"], g["test_sd"], g["test_true"]))]
    P["gp_input"] = [{"input": g["input"]}]
    rv = G["rvm"]
    P["rvm_test"] = [{"true": t, "predicted": p, "lower": round(p - 1.96 * sd, 3), "upper": round(p + 1.96 * sd, 3)}
                     for t, p, sd in zip(rv["true"], rv["pred"], rv["sd"])]
    P["sparsity"] = [{"model": k, "leaves_kept": v_} for k, v_ in rv["counts"].items()]

    m = G["mcmc"]
    tr = m["trace"]
    P["mcmc_trace"] = [{"step": i * m["step"], "phase": "burn-in" if i * m["step"] < m["burn"] else "kept",
                        "bias": tr[0][i], "w_lesion": tr[1][i], "w_contrast": tr[2][i]} for i in range(len(tr[0]))]
    P["mcmc_samples"] = [{"w_lesion": p[0], "w_contrast": p[1]} for p in m["post"]]
    cv = m["cov"]
    P["laplace"] = [{"parameter": "bias", "w_map": m["w_map"][0]},
                    {"parameter": "w_lesion", "w_map": m["w_map"][1], "var": cv[0][0], "cov": cv[0][1]},
                    {"parameter": "w_contrast", "w_map": m["w_map"][2], "var": cv[1][1], "cov": cv[0][1]}]

    gb = G["gibbs"]
    N = gb["size"]
    cells = []
    for r_ in range(0, N, 2):
        for c_ in range(0, N, 2):
            k = r_ * N + c_
            if gb["truth"][k] == ".":
                continue
            cells.append({"row": r_ // 2, "col": c_ // 2, "true_lesion": int(gb["truth"][k]), "noisy_lesion": int(gb["noisy"][k]),
                          "p_lesion": gb["post"][k], "cleaned_lesion": int(gb["clean"][k])})
    P["gibbs_cells"] = cells
    P["gibbs_errors"] = [{"stage": "Noisy mask (15% of labels flipped)", "wrong_pixels_pct": gb["err_before"]},
                         {"stage": "After Gibbs sampling", "wrong_pixels_pct": gb["err_after"]}]

    bn = G["bayesnet"]
    e = bn["estimates"]
    P["bn_posterior"] = [{"disease": d, "prior": bn["prior"][i], "exact": e["Exact (enumeration)"][i],
                          "rejection": e["Rejection sampling"][i], "likelihood_weighting": e["Likelihood weighting"][i],
                          "gibbs": e["Gibbs sampling"][i]} for i, d in enumerate(bn["diseases"])]
    P["bn_convergence"] = [{"samples": nn, "method": meth, "max_error_pts": err}
                           for meth, d in bn["conv"].items() for nn, err in zip(d["n"], d["err"])]

    sp = G["spad"]
    P["spad_test"] = [{"measured": a_, "predicted": p, "meter_sd": sd, "residual": round(a_ - p, 2)}
                      for a_, p, sd in zip(sp["measured"], sp["predicted"], sp["meter_sd"])]
    P["spad_cv"] = sp["cv"]

    P["gibbs_photo"] = [{"data_url": G["gibbs"]["photo"]}]
    return P


def main() -> int:
    warnings.filterwarnings("ignore")
    unit5.nn_graphs, unit5.combining_graphs = nn_capture, combining_capture
    unit5.kernel_graphs, unit5.sampling_graphs = kernel_capture, sampling_capture
    CC.make_graphs = spad_capture
    data = dataset.load_features()
    tmp = Path(tempfile.mkdtemp())
    unit5.run(data, reports_dir=tmp)
    try:
        CC.run(reports_dir=tmp, models_dir=tmp)
    except FileNotFoundError as e:                       # ArabicaLens photos not downloaded
        print(f"Chlorophyll skipped: {e}")
    if "spad" not in G:
        G["spad"] = None
    P = to_rows(G) if G["spad"] is not None else None
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(P, separators=(",", ":")), encoding="utf-8")
    print(f"Saved the data of every Unit 5 graph to {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
