"""Supplementary Fig. S1: the 90 JRR replicate values of product - rank-k completion (pair-weighted mean of
within-block rho) when whole relation types are withheld. Reads results2/repl_blockreps.json and writes
figures/figS1_block_replicates.pdf."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results2/repl_blockreps.json"))["blockreps"]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3), sharey=True)
panels = [("rank3", "(a) Rank fixed at 3"), ("rankcv", "(b) Rank re-chosen on the other blocks")]
bins = np.arange(0.10, 0.32, 0.005)
for ax, (k, title) in zip(axes, panels):
    o = d[f"product_minus_{k}"]
    v = np.array(o["replicate_values"])
    dev, hold = v[:45], v[45:]
    ax.hist([dev, hold], bins=bins, stacked=True, color=["#3b6ea8", "#b8b8b8"],
            label=["2013–2018 replicates", "2007–2012 replicates"], edgecolor="white", linewidth=0.3)
    ax.axvline(o["point"], color="black", lw=0.8, ls="--")
    ax.set_title(title, fontsize=8, loc="left")
    ax.set_xlabel("Product $-$ rank-$k$ completion (weighted $\\rho$)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.10, 0.32)
axes[0].set_ylabel("Replicates", fontsize=7)
axes[0].legend(fontsize=6, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("figures/figS1_block_replicates.pdf")
print("written figures/figS1_block_replicates.pdf")
