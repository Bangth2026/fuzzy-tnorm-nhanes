"""Descriptive numbers quoted in Results 6.1-6.2 (edge distribution, strongest targets, low-threshold stability, N1 rewiring share)."""
import json, collections
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
from analysis2 import graph, dev, thresholds, OFF
from graph2 import CONCEPTS, adj, leiden, qf, n1_rewire

Ed, E = graph(dev)
_, S = graph(dev, which="S")
q = np.quantile(Ed.mu, [.25, .5, .75])
top = Ed.sort_values("mu", ascending=False)
def strongest(M):
    M = M.copy(); np.fill_diagonal(M, -1)
    return collections.Counter(CONCEPTS[j] for j in M.argmax(1))
out = {"n_edges": int(len(Ed)), "mu_quartiles": q.round(3).tolist(), "median_mu": round(float(q[1]), 3),
       "top5": [[r.src, r.dst, round(float(r.mu), 3)] for r in top.head(5).itertuples()],
       "cvd_to_hypertension": round(float(E[CONCEPTS.index("cardiovascular"), CONCEPTS.index("hypertension")]), 3),
       "strongest_target_raw": dict(strongest(S).most_common()), "strongest_target_corrected": dict(strongest(E).most_common())}
A = adj(Ed); thr = thresholds(Ed)
B = (A >= thr["q25"]).astype(float)
sols = collections.Counter()
for s in range(100):
    m = leiden(B, s); sols[(len(set(m)), round(qf(B, m), 4))] += 1
out["q25_restart_solutions"] = [[k[0], k[1], v] for k, v in sorted(sols.items(), key=lambda x: -x[1])]
rng = np.random.default_rng(7); moved = []
for _ in range(50):
    En = n1_rewire(Ed, rng)
    a = set(zip(Ed.src, Ed.dst)); b = set(zip(En.src, En.dst))
    moved.append(1 - len(a & b) / len(a))
out["n1_share_edges_moved_mean"] = round(float(np.mean(moved)), 4)
# NMI of sensitivity partitions with the main partition
R1 = json.load(open("results2/rq1.json"))
lab = lambda comm: np.array([next(int(k) for k, v in comm.items() if c in v) for c in CONCEPTS])
main = lab(R1["rq1_main"]["communities"])
out["sens_NMI"] = {k: round(float(NMI(main, lab(v["communities"]))), 2) for k, v in R1["sensitivity"].items()}
out["sens_communities"] = {k: v["communities"] for k, v in R1["sensitivity"].items()}
json.dump(out, open("results2/describe.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:2500])
