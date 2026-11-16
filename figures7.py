"""Network drawing of the development graph (J-BHI Fig. 2): nodes placed by Leiden community and coloured by
concept type, edges with mu >= the upper quartile drawn with width proportional to mu.
Writes figures/Fig2_network.pdf and .png."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from graph2 import TYPE

E = pd.read_csv("results2/edges_dev.csv")
part = json.load(open("results2/best_partition.json"))
C, M = part["concepts"], part["membership"]
LABEL = {"anhedonia": "Anhedonia", "depressed_mood": "Depressed mood", "sleep_problems": "Sleep problems",
         "fatigue": "Fatigue", "appetite": "Appetite change", "worthlessness": "Worthlessness",
         "concentration": "Poor concentration", "psychomotor": "Psychomotor change", "self_harm_thoughts": "Thoughts of self-harm",
         "low_income": "Low income", "low_education": "Low education", "no_partner": "No partner", "uninsured": "Uninsured",
         "smoking": "Smoking", "heavy_drinking": "Heavy drinking", "inactivity": "Inactivity", "short_sleep": "Short sleep",
         "obesity": "Obesity", "poor_self_rated_health": "Poor self-rated health", "hypertension": "Hypertension",
         "diabetes": "Diabetes", "cardiovascular": "Cardiovascular disease", "cancer": "Cancer", "arthritis": "Arthritis"}
TCOL = {"Symptom": "#3b6fb6", "Socioeconomic": "#d08a2e", "Lifestyle": "#5aa469", "Health": "#b5485d"}
# community sectors: 0 symptoms (left), 1 somatic (lower right), 2 social (upper right)
centre = {0: (-1.35, 0.0), 1: (1.15, -0.95), 2: (1.15, 0.95)}
pos = {}
for k in sorted(set(M)):
    members = [c for c, m in zip(C, M) if m == k]
    members.sort(key=lambda c: (TYPE[c], c))
    n = len(members)
    rad = 0.55 + 0.04 * n
    for t, c in enumerate(members):
        a = 2 * np.pi * t / n + (np.pi / 2 if k == 0 else 0)
        pos[c] = (centre[k][0] + rad * np.cos(a), centre[k][1] + rad * np.sin(a))
thr = float(np.quantile(E.mu, 0.75))
Es = E[E.mu >= thr].sort_values("mu")
fig, ax = plt.subplots(figsize=(7.2, 4.6))
memb = dict(zip(C, M))
for r in Es.itertuples():
    same = memb[r.src] == memb[r.dst]
    ar = FancyArrowPatch(pos[r.src], pos[r.dst], arrowstyle="-|>", mutation_scale=5, lw=0.3 + 3.2 * r.mu,
                         color="#555555" if same else "#c0392b", alpha=0.35 + 0.5 * r.mu,
                         connectionstyle="arc3,rad=0.12", shrinkA=7, shrinkB=7, zorder=1)
    ax.add_patch(ar)
for c, (x, y) in pos.items():
    lw, ls, ec = 0.6, "-", "white"
    if c in ("sleep_problems", "fatigue"):
        lw, ec = 2.0, "black"
    if c == "poor_self_rated_health":
        from matplotlib.patches import Circle
        ax.add_patch(Circle((x, y), 0.11, fill=False, ls="--", lw=1.2, ec="black", zorder=4))
    ax.scatter([x], [y], s=150, color=TCOL[TYPE[c]], edgecolors=ec, linewidths=lw, linestyle=ls, zorder=3)
    dx = 0.09 if x >= centre[memb[c]][0] else -0.09
    ax.text(x + dx, y, LABEL[c], fontsize=7.2, ha="left" if dx > 0 else "right", va="center", zorder=4)
for k, (x, y) in centre.items():
    name = {0: "Depressive symptoms", 1: "Somatic health", 2: "Social disadvantage and risk behavior"}[k]
    ax.text(x, y + (1.08 if k != 1 else -1.12) * (0.55 + 0.04 * sum(1 for m in M if m == k)) / 0.9 * (1 if k != 1 else 1),
            name, fontsize=7.5, ha="center", weight="bold")
h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=v, markersize=7, label=k) for k, v in TCOL.items()]
h += [Line2D([0], [0], color="#555555", lw=1.5, label="within community"),
      Line2D([0], [0], color="#c0392b", lw=1.5, label="between communities"),
      Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey", markeredgecolor="black", markeredgewidth=1.6,
             markersize=7, label="moves when grades are discarded"),
      Line2D([0], [0], color="black", lw=1.0, ls="--", label="least stable placement (dashed ring)")]
ax.legend(handles=h, fontsize=7.5, loc="lower left", frameon=False, ncol=2, bbox_to_anchor=(-0.02, -0.08))
ax.set_xlim(-2.9, 2.9); ax.set_ylim(-2.25, 2.15); ax.axis("off")
fig.tight_layout()
fig.savefig("figures/Fig2_network.pdf"); fig.savefig("figures/Fig2_network.png", dpi=300)
json.dump({"edge_threshold_upper_quartile": thr, "n_edges_drawn": int(len(Es))}, open("results2/fig2_network.json", "w"))
print(thr, len(Es))
