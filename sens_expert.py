"""Reviewer-requested sensitivity variants: swap one membership function at a time."""
import json
from sklearn.metrics import normalized_mutual_info_score as NMI
from analysis2 import graph, rq1, dev, tic
from graph2 import CONCEPTS

bp = json.load(open("results2/best_partition.json"))["membership"]
Ed, _ = graph(dev)
main, _ = rq1(Ed)
out = {"main_Q": main["Q_mean"]}
for c in ["low_education", "smoking", "poor_self_rated_health", "diabetes", "cancer"]:
    tic(c)
    d = dev.copy()
    d[c] = d[c + "_alt"]
    o, best = rq1(graph(d)[0])
    out[c] = {"Q_mean": o["Q_mean"], "n_comm": o["n_comm"], "NMI_vs_main": float(NMI(bp, best)),
              "cost_q50": o["binarised"]["q50"]["cost"], "cost_0+": o["binarised"]["0+"]["cost"],
              "communities": o["communities"]}
json.dump(out, open("results2/sens_expert.json", "w"), indent=1)
tic("done")
