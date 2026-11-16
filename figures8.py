"""Summary figure: path rules against completion and Katz as more of the relation matrix is missing (a) and
when the development graph is thinned to its strongest edges (b). Reads results2/gen_*.json and
results2/ext_*.json; writes figures/fig_missingness.pdf."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = lambda p, k: json.load(open(f"results2/{p}.json"))[k]
miss, cold, blk, sp = J("gen_missing", "missing"), J("gen_coldstart", "coldstart"), J("gen_blockjrr", "blockjrr"), J("gen_sparse", "sparse")
holm = J("gen_holm", "holm")["jrr"]
desi = J("ext_designs", "designs")

COL = {"product": "#2a78d6", "hamacher": "#eb6834", "min": "#1baf7a", "rank3": "#eda100", "katz": "#e87ba4", "si": "#008300"}
LAB = {"product": "Product", "hamacher": "Hamacher", "min": "Minimum", "rank3": "Rank-3 completion",
       "katz": "Katz", "si": "Soft-impute (CV)"}
MK = {"product": "o", "hamacher": "s", "min": "v", "rank3": "D", "katz": "^", "si": "P"}

xs = ["1 dyad", "10%", "25%", "50%", "75%", "$m$=6", "$m$=3", "$m$=1", "all pairs"]
val = {k: [] for k in COL}
val["product"].append(holm["product"]["point"]); val["hamacher"].append(holm["hamacher"]["point"])
val["min"].append(holm["min"]["point"]); val["rank3"].append(holm["rank3"]["point"])
val["katz"].append(holm["katz"]["point"]); val["si"].append(desi["temporal"]["rho"])
for f in ["0.1", "0.25", "0.5", "0.75"]:
    for k, kk in [("product", "product"), ("hamacher", "hamacher"), ("min", "min"), ("rank3", "rank3"), ("katz", "katz"), ("si", "softimpute_cv")]:
        val[k].append(miss[f][kk]["mean"])
for m in ["6", "3", "1"]:
    for k, kk in [("product", "product"), ("hamacher", "hamacher"), ("min", "min"), ("rank3", "rank3"), ("katz", "katz"), ("si", "softimpute_cv")]:
        val[k].append(cold[m][kk])
bj = blk["jrr"]
for k, kk in [("product", "product"), ("hamacher", "hamacher"), ("rank3", "rank3"), ("katz", "katz"), ("si", "softimpute_cv")]:
    val[k].append(bj[f"weighted_mean_{kk}"]["point"])
val["min"].append(np.nan)

plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [2.15, 1]})
x = np.arange(len(xs))
for k in COL:
    for g, (lo, hi) in enumerate([(0, 5), (5, 8), (8, 9)]):
        a.plot(x[lo:hi], val[k][lo:hi], color=COL[k], marker=MK[k], ms=4.2, lw=1.6, label=LAB[k] if g == 0 else None,
               mec="white", mew=0.6)
for sep in [4.5, 7.5]:
    a.axvline(sep, color="#c3c2b7", lw=0.6, zorder=0)
for xx, t in [(2, "dyads withheld"), (6, "cold start"), (8, "type block")]:
    a.text(xx, 0.255, t, ha="center", fontsize=6.6, color="#52514e")
a.set_xticks(x); a.set_xticklabels(xs, fontsize=6.8)
a.set_ylim(0.2, 0.95); a.set_ylabel(r"Spearman $\rho$ on withheld pairs")
a.grid(axis="y", color="#e6e5e0", lw=0.5); a.set_axisbelow(True)
a.set_title("(a) Increasing and structured missingness", fontsize=7.8, loc="left")

lv = ["top50", "top25", "mu_ge_0.25"]
lx = ["Top 50%", "Top 25%", r"$\mu\geq0.25$"]
xb = np.arange(3)
for k, kk in [("product", "product"), ("hamacher", "hamacher"), ("min", "min"), ("katz", "katz"), ("rank3", "rank3_absent_zero")]:
    b.plot(xb, [sp[l]["rho_unrecorded"][kk] for l in lv], color=COL[k], marker=MK[k], ms=4.2, lw=1.6, mec="white", mew=0.6)
b.plot(xb, [sp[l]["rho_unrecorded"]["rank3_absent_missing"] for l in lv], color=COL["rank3"], marker=MK["rank3"], ms=4.2,
       lw=1.2, ls="--", mec="white", mew=0.6, label="Rank-3, absent = missing")
b.axhline(0, color="#8a8984", lw=0.6)
b.set_xticks(xb); b.set_xticklabels(lx, fontsize=6.8); b.set_xlim(-0.3, 2.3)
b.set_ylabel(r"$\rho$ on unrecorded pairs"); b.set_ylim(-0.3, 0.7)
b.grid(axis="y", color="#e6e5e0", lw=0.5); b.set_axisbelow(True)
b.set_title("(b) Thinned graph", fontsize=7.8, loc="left")
h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=7, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.02),
           handlelength=1.6, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("figures/fig_missingness.pdf"); fig.savefig("figures/fig_missingness.png", dpi=200)
print({k: [round(v, 3) for v in vv] for k, vv in val.items()})
