"""Summary figure: path rules against completion and Katz under four patterns of missing relations.
(a) single dyads and random dyads withheld; (b) cold start, median over concepts; (c) whole relation types and
simulated sources; (d) graphs thinned by strength or at random. Reads results2/gen_*.json, mech_*.json and
ext_designs.json; writes figures/fig_missingness.pdf."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = lambda p, k: json.load(open(f"results2/{p}.json"))[k]
miss, cold, blk = J("gen_missing", "missing"), J("gen_coldstart", "coldstart"), J("gen_blockjrr", "blockjrr")["jrr"]
holm, sc, bx, rt = J("gen_holm", "holm")["jrr"], J("mech_scale", "scale"), J("mech_blockextra", "blockextra"), J("mech_randthin", "randthin")

COL = {"product": "#2a78d6", "hamacher": "#eb6834", "min": "#1baf7a", "rank3": "#eda100", "katz": "#e87ba4", "si": "#008300"}
LAB = {"product": "Product", "hamacher": "Hamacher", "min": "Minimum", "rank3": "Rank-3 completion", "katz": "Katz",
       "si": "Soft-impute (logit, CV)"}
MK = {"product": "o", "hamacher": "s", "min": "v", "rank3": "D", "katz": "^", "si": "P"}
nan = np.nan

xa = ["1 dyad", "10%", "25%", "50%", "75%"]
A = {k: [holm[k]["point"]] for k in ["product", "hamacher", "min", "rank3", "katz"]}
A["si"] = [J("ext_designs", "designs")["temporal"]["rho"]]
for f in ["0.1", "0.25", "0.5", "0.75"]:
    for k in ["product", "hamacher", "min", "rank3", "katz"]:
        A[k].append(miss[f][k]["mean"])
    A["si"].append(sc[f"random_{f}"]["logit"])
xb = ["$m$=6", "$m$=3", "$m$=1"]
B = {k: [cold[m][f"median_per_concept_{k}"] for m in ["6", "3", "1"]] for k in ["product", "hamacher", "min", "rank3", "katz"]}
B["si"] = [sc[f"coldstart_m{m}_median_per_concept"]["logit"] for m in ["6", "3", "1"]]
xc = ["Type\nblocks", "Sources\nA", "Sources\nB"]
sa, sb = rt["sources_SymSES_LifeHlth_SymHlth"], rt["sources_SymLife_SESHlth_SymSES"]
C = {"product": [blk["weighted_mean_product"]["point"], sa["product"], sb["product"]],
     "hamacher": [blk["weighted_mean_hamacher"]["point"], sa["hamacher"], sb["hamacher"]],
     "min": [bx["min"]["weighted_mean"], sa["min"], sb["min"]],
     "rank3": [blk["weighted_mean_rank3"]["point"], sa["rank3_absent_missing"], sb["rank3_absent_missing"]],
     "katz": [blk["weighted_mean_katz"]["point"], sa["katz"], sb["katz"]],
     "si": [sc["blocks_weighted"]["logit"], sa["softimpute_absent_missing"], sb["softimpute_absent_missing"]]}
xd = ["Strength\n50%", "Strength\n25%", "Random\n50%", "Random\n25%"]
s50, s25, r50, r25 = rt["strength_top50"], rt["strength_top25"], rt["random_keep_0.5"], rt["random_keep_0.25"]
D = {"product": [s50["product"], s25["product"], r50["product"], r25["product"]],
     "hamacher": [s50["hamacher"], s25["hamacher"], r50["hamacher"], r25["hamacher"]],
     "min": [s50["min"], s25["min"], r50["min"], r25["min"]],
     "rank3": [s50["rank3_absent_zero"], s25["rank3_absent_zero"], r50["rank3_absent_missing"], r25["rank3_absent_missing"]],
     "katz": [s50["katz"], s25["katz"], r50["katz"], r25["katz"]],
     "si": [nan, nan, r50["softimpute_absent_missing"], r25["softimpute_absent_missing"]]}

# uncertainty: JRR 95% CI (single dyad, type blocks) and +/-1 SD over repetitions (random dyads)
EA = {k: [nan] * 5 for k in COL}; EAlo = {k: [nan] * 5 for k in COL}
for k in ["product", "hamacher", "min", "rank3", "katz"]:
    lo, hi = holm[k]["ci95"]; EA[k][0] = (hi - lo) / 2
    for q, f in enumerate(["0.1", "0.25", "0.5", "0.75"]):
        EA[k][q + 1] = miss[f][k]["sd"]
lo, hi = J("ext_designs", "designs")["temporal"]["jrr_ci"]; EA["si"][0] = (hi - lo) / 2
EC = {k: [nan] * 3 for k in COL}
for k, kk in [("product", "product"), ("hamacher", "hamacher"), ("rank3", "rank3"), ("katz", "katz")]:
    lo, hi = blk[f"weighted_mean_{kk}"]["ci95"]; EC[k][0] = (hi - lo) / 2
ERR = {"a": EA, "c": EC}
plt.rcParams.update({"font.size": 7.3, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 4, figsize=(7.16, 2.45), gridspec_kw={"width_ratios": [5, 3, 3, 4]}, sharey=True)
panels = [(axs[0], xa, A, "(a) Dyads withheld", [(0, 1), (1, 5)], "a"), (axs[1], xb, B, "(b) Cold start, median", [(0, 3)], None),
          (axs[2], xc, C, "(c) Types, sources", None, "c"), (axs[3], xd, D, "(d) Thinned graph", [(0, 2), (2, 4)], None)]
for ax, xs, V, title, segs, ek in panels:
    x = np.arange(len(xs))
    for q, k in enumerate(COL):
        if ek:
            off = (q - 2.5) * 0.06
            ax.errorbar(x + off, V[k], yerr=ERR[ek][k], fmt="none", ecolor=COL[k], elinewidth=0.7, capsize=0, alpha=0.8)
        if segs is None:
            ax.plot(x, V[k], color=COL[k], marker=MK[k], ms=4.4, lw=0, mec="white", mew=0.6)
        else:
            for g, (lo, hi) in enumerate(segs):
                ax.plot(x[lo:hi], V[k][lo:hi], color=COL[k], marker=MK[k], ms=4.2, lw=1.5, mec="white", mew=0.6,
                        label=LAB[k] if (g == 0 and ax is axs[1]) else None)
    ax.set_xticks(x); ax.set_xticklabels(xs, fontsize=6.4)
    ax.set_xlim(-0.4, len(xs) - 0.6)
    ax.grid(axis="y", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=7.3, loc="left")
axs[0].set_ylabel(r"Spearman $\rho$")
axs[0].set_ylim(0.0, 0.95)
h, l = axs[1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.02), handlelength=1.6)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.5)
fig.savefig("figures/fig_missingness.pdf"); fig.savefig("figures/fig_missingness.png", dpi=200)
