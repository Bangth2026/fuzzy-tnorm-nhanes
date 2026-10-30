"""Power/size calibration. N2: full delta grid (50 reps x 100 nulls x 5 restarts).
N1: size only (delta = 0; 20 reps x 50 nulls) because typed rewiring is ~2 s per null graph."""
import json, sys, numpy as np
from graph2 import *
from analysis2 import graph, dev, null_test, tic
rng = np.random.default_rng(20261030)
Ed, _ = graph(dev)
bp = json.load(open("results2/best_partition.json")); comm = dict(zip(bp["concepts"], bp["membership"]))
kind = sys.argv[1]
grid = [0.0, 0.1, 0.25, 0.5] if kind == "N2" else [0.0]
reps, nn = (50, 100) if kind == "N2" else (20, 50)
out = {}
for delta in grid:
    tic(f"{kind} {delta}")
    rej = []
    for _ in range(reps):
        Ep = n2_permute(Ed, rng)
        inside = np.array([comm[s] == comm[d] for s, d in zip(Ep.src, Ep.dst)])
        Ep.loc[inside, "mu"] = np.minimum(1.0, Ep.loc[inside, "mu"] * (1 + delta))
        rej.append(null_test(Ep, kind, nn, 5, rng)["p"] <= 0.05)
    out[f"delta_{delta}"] = float(np.mean(rej)); out[f"reps_{delta}"] = reps; out["nulls"] = nn
    json.dump(out, open(f"results2/power_{kind}.json", "w"), indent=1)
tic("done")
