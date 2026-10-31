"""Figures for the sixth review: calibration with bootstrap bands (Fig. 2), withheld-block forest plot (Fig. 3),
relation heatmap ordered by community (Online Resource Fig. S1)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis2 import dev, hold, psu_boot
from graph2 import CONCEPTS, relation_matrices
import rq3x as X
import rq6lib as L
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
OFF = L.OFF
COL = {"product": "#1b4f9c", "hamacher": "#c0392b", "min": "#7f7f7f", "product_recal": "#1b4f9c", "hamacher_recal": "#c0392b"}
which = sys.argv[1]

if which == "calib":
    Ed, Eh = relation_matrices(dev)[2], relation_matrices(hold)[2]
    def curves(E, H):
        out = {}
        for t in ["product", "hamacher", "min"]:
            p = X.paths(E, X.T_FIXED[t])[OFF]
            out[t] = (p, H[OFF])
            for kind in ["linear"]:
                if t != "min":
                    out[t + "_recal"] = (L.recalibrate(p, E[OFF], p, kind), H[OFF])
        return out
    def bins(p, h, nb=10):
        o = np.argsort(p, kind="stable"); b = np.array_split(o, nb)
        return np.array([p[i].mean() for i in b]), np.array([h[i].mean() for i in b])
    pt = {k: bins(*v) for k, v in curves(Ed, Eh).items()}
    rng = np.random.default_rng(7)
    boots = {k: [] for k in pt}
    for b in range(200):
        Eb, Hb = relation_matrices(psu_boot(dev, rng))[2], relation_matrices(psu_boot(hold, rng))[2]
        for k, v in curves(Eb, Hb).items():
            boots[k].append(bins(*v)[1])
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.2), dpi=600, sharey=True)
    sty = {"product": ("o", "-"), "hamacher": ("s", "--"), "min": ("^", ":"), "product_recal": ("o", "-"), "hamacher_recal": ("s", "--")}
    lab = {"product": "Product", "hamacher": "Hamacher", "min": "Minimum", "product_recal": "Product, recalibrated", "hamacher_recal": "Hamacher, recalibrated"}
    for ax, keys, title in [(axes[0], ["product", "hamacher", "min"], "a  As composed"), (axes[1], ["product_recal", "hamacher_recal"], "b  Linear recalibration")]:
        for k in keys:
            x, y = pt[k]; B = np.array(boots[k])
            lo, hi = np.quantile(B, .025, 0), np.quantile(B, .975, 0)
            mk, ls = sty[k]
            ax.fill_between(x, lo, hi, color=COL[k], alpha=0.15, linewidth=0)
            ax.plot(x, y, marker=mk, linestyle=ls, color=COL[k], markersize=3.5, linewidth=1,
                    markerfacecolor="white" if "recal" in k or k == "hamacher" else COL[k], label=lab[k])
        lim = 0.5
        ax.plot([0, lim], [0, lim], color="0.6", linewidth=0.7, zorder=0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Mean predicted value (tenth of pairs)")
        ax.text(0.01, 0.98, title, transform=ax.transAxes, va="top", fontsize=8, fontweight="bold")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    axes[0].set_ylabel("Mean held-out relation")
    fig.tight_layout()
    fig.savefig("Fig2_calibration.tiff", pil_kwargs={"compression": "tiff_lzw"}); fig.savefig("Fig2_calibration.png")
    json.dump({k: {"x": pt[k][0].tolist(), "y": pt[k][1].tolist()} for k in pt}, open("results2/fig2_points.json", "w"))

if which == "forest":
    b3 = json.load(open("results2/revision3_blocks.json"))["blocks"]
    b6 = json.load(open("results2/revision6_blocks.json"))["blocks"]
    names = list(b6.keys()); old = list(b3.keys())
    fig, ax = plt.subplots(figsize=(6.85, 4.0), dpi=600)
    meths = [("product", "Product", "o", "#1b4f9c"), ("hamacher", "Hamacher", "s", "#c0392b"), ("rank3_svd", "Rank-3 completion", "D", "#555555"),
             ("softimpute_cv", "Soft-impute (λ by CV)", "v", "#2e8b57"), ("prev_type_reg", "Prevalence and type regression", "^", "#b8860b"),
             ("learned_paths", "Learned path model", "P", "#6a3d9a")]
    y0 = np.arange(len(names))[::-1]
    for m, (key, lab, mk, col) in enumerate(meths):
        off = (m - (len(meths) - 1) / 2) * 0.12
        xs = [b6[n][key]["rho"] for n in names]
        ax.scatter(xs, y0 + off, marker=mk, s=14, color=col, label=lab, zorder=3)
        if key in ("product", "rank3_svd"):
            kk = {"product": "product", "rank3_svd": "rank3"}[key]
            for n, o, yy in zip(names, old, y0):
                lo, hi = b3[o]["ci_shifted"][kk]
                ax.plot([max(lo, -1), min(hi, 1)], [yy + off] * 2, color=col, linewidth=0.8, zorder=2)
    ax.set_yticks(y0)
    ax.set_yticklabels([f"{n} ({b6[n]['n_pairs']}){' *' if b6[n]['n_pairs'] < 30 else ''}" for n in names])
    ax.axvline(0, color="0.7", linewidth=0.6)
    ax.set_xlim(-1, 1); ax.set_xlabel("Spearman ρ with the held-out relation, block withheld")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig("Fig3_withheld_blocks.tiff", pil_kwargs={"compression": "tiff_lzw"}); fig.savefig("Fig3_withheld_blocks.png")

if which == "heatmap":
    Ed = relation_matrices(dev)[2]
    order = ["anhedonia", "depressed_mood", "sleep_problems", "fatigue", "appetite", "worthlessness", "concentration", "psychomotor",
             "self_harm_thoughts", "inactivity", "obesity", "poor_self_rated_health", "hypertension", "diabetes", "cardiovascular",
             "cancer", "arthritis", "low_income", "low_education", "no_partner", "uninsured", "smoking", "heavy_drinking", "short_sleep"]
    ix = [CONCEPTS.index(c) for c in order]
    M = Ed[np.ix_(ix, ix)]
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=600)
    im = ax.imshow(np.where(np.eye(24, dtype=bool), np.nan, M), cmap="Greys", vmin=0, vmax=0.4)
    lbl = [c.replace("_", " ") for c in order]
    ax.set_xticks(range(24)); ax.set_yticks(range(24)); ax.set_xticklabels(lbl, rotation=90, fontsize=6); ax.set_yticklabels(lbl, fontsize=6)
    for b in (8.5, 16.5):
        ax.axhline(b, color="k", linewidth=0.8); ax.axvline(b, color="k", linewidth=0.8)
    ax.set_xlabel("Target concept j"); ax.set_ylabel("Source concept i")
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("μ(i → j), development (capped at 0.4)")
    fig.tight_layout()
    fig.savefig("FigS1_heatmap.tiff", pil_kwargs={"compression": "tiff_lzw"}); fig.savefig("FigS1_heatmap.png")
