"""Agreement of the tuned parametric t-norms with the product (Spearman over 552 development predictions)."""
import json
from analysis2 import graph, dev, OFF
import rq3x as X
R = json.load(open("results2/revision2_r3.json"))["families"]
_, Ed = graph(dev)
P0 = X.paths(Ed, X.t_product)[OFF]
ctor = {"hamacher_lambda": X.t_hamacher_lam, "frank_s": X.t_frank, "schweizer_sklar_p": X.t_ss}
out = {k: X.rho(X.paths(Ed, ctor[k](R[k]["chosen"]))[OFF], P0) for k in ctor}
json.dump({"tuned_vs_product_spearman": out}, open("results2/revision2_extra.json", "w"), indent=1)
print(out)
