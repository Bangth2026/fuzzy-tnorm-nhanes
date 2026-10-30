"""RQ1 and RQ2 on the MFKG v0 built from NHANES development cycles."""
import json
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from graph import *

SEED = 20261029
RESTARTS = 100
ALPHAS = [0.25, 0.50, 0.75]
rng = np.random.default_rng(SEED)
dev, hold = pd.read_pickle("data_dev.pkl"), pd.read_pickle("data_hold.pkl")
R = {}


def build(df, eps=0.05, layers=("data", "lit")):
    e = []
    if "data" in layers:
        e += data_edges(df, eps)
    if "lit" in layers:
        e += literature_edges()
    return edges_df(e)


def nodes_of(E):
    return [n for n in NODES if n in set(E.src) | set(E.dst)]


def binarise(A, alpha):
    return (A >= alpha).astype(float) if alpha > 0 else (A > 0).astype(float)


def rq1(E, restarts=RESTARTS, gamma=1.0, seed0=0):
    nodes = nodes_of(E)
    A = adj(E, nodes)
    seeds = range(seed0, seed0 + restarts)
    ms = [leiden(A, s, gamma) for s in seeds]
    qs = np.array([qf(A, m, gamma) for m in ms])
    best = ms[int(qs.argmax())]
    out = {"n_nodes": len(nodes), "n_edges": int((A > 0).sum()), "Q_mean": qs.mean(), "Q_best": qs.max(),
           "n_comm": int(len(set(best))),
           "communities": {int(k): [n for n, mm in zip(nodes, best) if mm == k] for k in sorted(set(best))},
           "binarised": {}}
    for a in ALPHAS + [0.0]:
        B = binarise(A, a)
        mb = [leiden(B, s, gamma) if B.sum() > 0 else np.arange(len(nodes)) for s in seeds]
        qb = np.array([qf(A, m, gamma) for m in mb])
        bb = mb[int(np.array([qf(B, m, gamma) if B.sum() > 0 else 0 for m in mb]).argmax())]
        out["binarised"]["0+" if a == 0 else a] = {
            "edges_kept": int(B.sum()), "Q_on_GF_mean": qb.mean(), "cost": qs.mean() - qb.mean(),
            "NMI": NMI(best, bb), "ARI": ARI(best, bb), "n_comm": int(len(set(bb))),
            "communities": {int(k): [n for n, mm in zip(nodes, bb) if mm == k] for k in sorted(set(bb))}}
    return out, best, nodes


def psu_boot(df, rng):
    parts = []
    for st, g in df.groupby("strata"):
        psus = g.psu.unique()
        for p in rng.choice(psus, len(psus), replace=True):
            parts.append(g[g.psu == p])
    return pd.concat(parts)


def null_test(E, kind, n_null, restarts, rng, gamma=1.0):
    nodes = nodes_of(E)
    A = adj(E, nodes)
    q_obs = best_q(A, range(restarts), gamma)[0]
    qn = []
    for _ in range(n_null):
        En = n2_permute(E, rng) if kind == "N2" else n1_rewire(E, rng)
        qn.append(best_q(adj(En, nodes), range(restarts), gamma)[0])
    qn = np.array(qn)
    return {"Q_obs": q_obs, "null_mean": qn.mean(), "null_sd": qn.std(ddof=1), "dQ": q_obs - qn.mean(),
            "z": (q_obs - qn.mean()) / qn.std(ddof=1), "p": (1 + (qn >= q_obs).sum()) / (n_null + 1),
            "n_null": n_null, "restarts": restarts, "null_q05": float(np.quantile(qn, 0.05)),
            "null_q95": float(np.quantile(qn, 0.95))}, qn


def tic(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "main"

    if part == "main":
        E = build(dev)
        E.to_csv("results/edges_v0_dev.csv", index=False)
        tic("RQ1 main")
        R["rq1_main"], best, nodes = rq1(E)
        np.save("results/best_partition.npy", best)
        json.dump({"nodes": nodes, "membership": best.tolist()}, open("results/best_partition.json", "w"))

        tic("bootstrap CI (B=200, 10 restarts)")
        boots = {str(a): [] for a in ALPHAS + ["0+"]}
        wboot = []
        for b in range(200):
            Eb = build(psu_boot(dev, rng))
            o, _, _ = rq1(Eb, restarts=10, seed0=1000 + b)
            for k, v in o["binarised"].items():
                boots[str(k)].append(v["cost"])
            wboot.append(Eb.set_index(["src", "dst"]).mu)
        R["rq1_boot_ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boots.items()}
        W = pd.concat(wboot, axis=1)
        R["edge_weight_boot_sd_median"] = float(W.std(axis=1).median())
        R["edge_weight_boot_sd_max"] = float(W.std(axis=1).max())

        tic("held-out replication")
        Eh = build(hold)
        Eh.to_csv("results/edges_v0_heldout.csv", index=False)
        oh, bh, nh = rq1(Eh)
        R["heldout_rq1"] = oh
        R["heldout_NMI_with_dev"] = NMI(best, bh) if nh == nodes else None
        R["heldout_ARI_with_dev"] = ARI(best, bh) if nh == nodes else None
        m = E.merge(Eh, on=["src", "dst", "rel"], suffixes=("_dev", "_hold"))
        d = m[m.prov_dev == "NHANES"]
        R["heldout_weight_corr"] = {"pearson": float(np.corrcoef(d.mu_dev, d.mu_hold)[0, 1]),
                                    "max_abs_diff": float((d.mu_dev - d.mu_hold).abs().max()),
                                    "n_edges": int(len(d))}

        tic("sensitivity")
        sens = {}
        for eps in [0.02, 0.10]:
            o, _, _ = rq1(build(dev, eps=eps))
            sens[f"eps_{eps}"] = o
        for gam in [0.5, 1.5]:
            o, _, _ = rq1(E, gamma=gam)
            sens[f"gamma_{gam}"] = o
        for lay in [("data",), ("lit",)]:
            o, _, _ = rq1(build(dev, layers=lay))
            sens["layer_" + "+".join(lay)] = o
        R["sensitivity"] = sens
        json.dump(R, open("results/rq1.json", "w"), indent=2, default=float)
        tic("done main")

    if part == "nulls":
        E = build(dev)
        out, qns = {}, {}
        for kind in ["N1", "N2"]:
            tic(kind)
            out[kind], qns[kind] = null_test(E, kind, 1000, 10, rng)
        for lay in [("data",), ("lit",)]:
            El = build(dev, layers=lay)
            for kind in ["N1", "N2"]:
                tic(f"{kind} {lay}")
                out[f"{kind}_layer_{'+'.join(lay)}"], _ = null_test(El, kind, 1000, 10, rng)
        Eh = build(hold)
        for kind in ["N1", "N2"]:
            tic(f"{kind} heldout")
            out[f"{kind}_heldout"], _ = null_test(Eh, kind, 1000, 10, rng)
        np.save("results/null_N1.npy", qns["N1"])
        np.save("results/null_N2.npy", qns["N2"])
        json.dump(out, open("results/rq2.json", "w"), indent=2, default=float)
        tic("done nulls")

    if part == "power":
        # planted structure: weights first scrambled within type (destroys real structure),
        # then edges inside the observed best partition are multiplied by (1 + delta), capped at 1
        E = build(dev)
        bp = json.load(open("results/best_partition.json"))
        comm = dict(zip(bp["nodes"], bp["membership"]))
        out = {}
        for kind in ["N2", "N1"]:
            for delta in [0.0, 0.25, 0.5, 1.0]:
                tic(f"power {kind} delta={delta}")
                rej = []
                for rep in range(50):
                    Ep = n2_permute(E, rng)
                    inside = np.array([comm[s] == comm[d] for s, d in zip(Ep.src, Ep.dst)])
                    Ep.loc[inside, "mu"] = np.minimum(1.0, Ep.loc[inside, "mu"] * (1 + delta))
                    res, _ = null_test(Ep, kind, 100, 5, rng)
                    rej.append(res["p"] <= 0.05)
                out[f"{kind}_delta_{delta}"] = float(np.mean(rej))
        json.dump(out, open("results/power.json", "w"), indent=2)
        tic("done power")
