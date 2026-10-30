"""Vignettes from held-out NHANES and pathway rankings (system outputs only;
no clinician data exist yet, so no agreement statistics are computed here)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from graph import *

SEED = 20261029
K = 12          # number of vignettes: provisional, to be fixed by the power calculation in 5.4
rng = np.random.default_rng(SEED)
hold = pd.read_pickle("data_hold.pkl")
E = pd.read_csv("results/edges_v0_dev.csv")   # graph built on DEVELOPMENT cycles only

ITEMS = SYM  # activations a = item/3 (item 9 handled by the safety layer)
elig = hold[(hold.safety_fail == 0) & (hold.phq9_total >= 5)]
flow = {"heldout_total": int(len(hold)), "safety_blocked": int(hold.safety_fail.sum()),
        "blocked_item9": int((hold.self_harm_thoughts > 0).sum()),
        "blocked_total_ge20_only": int(((hold.phq9_total >= 20) & (hold.self_harm_thoughts == 0)).sum()),
        "eligible_phq5_19": int(len(elig))}


def kmedoids(X, k, rng, iters=100):
    n = len(X)
    D = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    med = list(rng.choice(n, k, replace=False))
    for _ in range(iters):
        lab = D[:, med].argmin(1)
        new = []
        for j in range(k):
            idx = np.where(lab == j)[0]
            new.append(int(idx[D[np.ix_(idx, idx)].sum(1).argmin()]) if len(idx) else med[j])
        if new == med:
            break
        med = new
    lab = D[:, med].argmin(1)
    return med, lab, D[np.arange(n), np.array(med)[lab]].sum()


sample = elig.sample(n=min(3000, len(elig)), random_state=SEED)
X = sample[ITEMS].to_numpy() * 3
best = None
for r in range(10):
    m, lab, cost = kmedoids(X, K, rng)
    if best is None or cost < best[2]:
        best = (m, lab, cost)
med, lab, _ = best
vign = sample.iloc[med].copy()
vign["cluster_size_in_sample"] = np.bincount(lab, minlength=K)
vign = vign.sort_values("phq9_total").reset_index(drop=True)
vign["vignette"] = [f"V{i+1:02d}" for i in range(K)]

methods = ["product", "hamacher", "lukasiewicz", "yager2", "maxmin"]
rows, ranks = [], {}
for _, v in vign.iterrows():
    act = {s: float(v[s]) for s in SYM}
    items = {s: int(round(v[s] * 3)) for s in SYM}
    rk = {m: rank(component_scores(E, act, m)) for m in methods}
    rk["binary_G0.50"] = rank(binary_scores(E, items, 0.5))
    rk["curriculum"] = list(COMP)
    ranks[v.vignette] = rk
    sc = component_scores(E, act, "product")
    rows.append({"vignette": v.vignette, "phq9_total": int(v.phq9_total),
                 **{f"item_{s}": items[s] for s in SYM},
                 "top3_product": rk["product"][:3], "top3_binary": rk["binary_G0.50"][:3],
                 "top_score_product": max(sc.values()),
                 "n_binary_reachable": int(sum(1 for c, x in binary_scores(E, items, 0.5).items() if x > -1e9)),
                 "cluster_size_in_sample": int(v.cluster_size_in_sample)})


def tau(a, b):
    pa = [a.index(c) for c in COMP]
    pb = [b.index(c) for c in COMP]
    return kendalltau(pa, pb).statistic


cmp = {}
for m in ["hamacher", "lukasiewicz", "yager2", "maxmin", "binary_G0.50", "curriculum"]:
    t = [tau(ranks[v]["product"], ranks[v][m]) for v in ranks]
    top1 = [ranks[v]["product"][0] == ranks[v][m][0] for v in ranks]
    top3 = [len(set(ranks[v]["product"][:3]) & set(ranks[v][m][:3])) / 3 for v in ranks]
    cmp[m] = {"kendall_tau_median": float(np.median(t)), "kendall_tau_min": float(np.min(t)),
              "kendall_tau_max": float(np.max(t)), "top1_same_share": float(np.mean(top1)),
              "top3_overlap_mean": float(np.mean(top3))}
distinct_top3_product = len({tuple(ranks[v]["product"][:3]) for v in ranks})
distinct_top3_binary = len({tuple(ranks[v]["binary_G0.50"][:3]) for v in ranks})

pd.DataFrame(rows).to_csv("results/vignettes.csv", index=False)
json.dump({"flow": flow, "k": K, "comparison_vs_product": cmp,
           "distinct_top3_product": distinct_top3_product, "distinct_top3_binary": distinct_top3_binary,
           "ranks": ranks}, open("results/pathways.json", "w"), indent=2, default=float)
print(json.dumps({"flow": flow, "cmp": cmp, "distinct": [distinct_top3_product, distinct_top3_binary]}, indent=2, default=float))
print(pd.DataFrame(rows)[["vignette", "phq9_total", "top3_product", "top3_binary", "n_binary_reachable"]].to_string())
