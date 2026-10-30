"""Implementation checks for the current pipeline (results2/sanity2.json)."""
import json
import numpy as np
import igraph as ig, leidenalg as la
from analysis2 import graph, dev
from graph2 import adj, qf, n1_rewire, CONCEPTS, relation_matrices
out = {}
Ed, _ = graph(dev); A = adj(Ed)
src, dst = np.nonzero(A)
g = ig.Graph(n=A.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=True); g.es["weight"] = A[src, dst].tolist()
p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight", seed=1)
out["leiden_quality_minus_qf"] = float(p.quality() / A.sum() - qf(A, np.array(p.membership)))
# S0 equals weighted mean activation of j when i is crisp
d = dev.copy(); d["anhedonia"] = (d["anhedonia"] >= 0.5).astype(float).where(d["anhedonia"].notna())
S, S0, _ = relation_matrices(d)
i = CONCEPTS.index("anhedonia"); errs = []
for j, c in enumerate(CONCEPTS):
    if j == i: continue
    m = d["anhedonia"].notna() & d[c].notna() & (d["anhedonia"] == 1)
    errs.append(abs(S0[i, j] - float(np.average(d[c][d["anhedonia"].notna() & d[c].notna()], weights=d.w[d["anhedonia"].notna() & d[c].notna()]))))
out["S0_crisp_max_abs_error"] = float(max(errs))
rng = np.random.default_rng(3); ok = True
for _ in range(20):
    En = n1_rewire(Ed, rng)
    for rel, gg in Ed.groupby("rel"):
        hh = En[En.rel == rel]
        ok &= gg.src.value_counts().sort_index().equals(hh.src.value_counts().sort_index()) and gg.dst.value_counts().sort_index().equals(hh.dst.value_counts().sort_index())
out["n1_preserves_within_type_degrees"] = bool(ok)
json.dump(out, open("results2/sanity2.json", "w"), indent=1); print(out)
