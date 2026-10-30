"""Data-driven fuzzy KG: RQ1 (binarisation cost), RQ2 (structure vs nulls), RQ3 (t-norm inference)."""
import json
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
from graph2 import *

SEED = 20261029
rng = np.random.default_rng(SEED)
dev, hold = pd.read_pickle("data2_dev.pkl"), pd.read_pickle("data2_hold.pkl")
EPS = 0.01
OFF = ~np.eye(P, dtype=bool)


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def graph(df, eps=EPS, which="E"):
    S, S0, E = relation_matrices(df)
    M = {"E": E, "S": S}[which]
    return edges_from(M, eps), M


def thresholds(Ed):
    q = np.quantile(Ed.mu, [0.25, 0.50, 0.75])
    return {"q25": q[0], "q50": q[1], "q75": q[2], "abs0.25": 0.25, "abs0.50": 0.50}


def rq1(Ed, restarts=100, gamma=1.0, seed0=0, thr=None):
    A = adj(Ed)
    seeds = range(seed0, seed0 + restarts)
    ms = [leiden(A, s, gamma) for s in seeds]
    qs = np.array([qf(A, m, gamma) for m in ms])
    best = ms[int(qs.argmax())]
    out = {"n_edges": int(len(Ed)), "Q_mean": float(qs.mean()), "Q_best": float(qs.max()), "n_comm": int(len(set(best))),
           "communities": {int(k): [c for c, mm in zip(CONCEPTS, best) if mm == k] for k in sorted(set(best))}, "binarised": {}}
    thr = thr or thresholds(Ed)
    for name, a in list(thr.items()) + [("0+", 0.0)]:
        B = (A >= a).astype(float) if a > 0 else (A > 0).astype(float)
        mb = [leiden(B, s, gamma) for s in seeds]
        qb = np.array([qf(A, m, gamma) for m in mb])
        qbb = [qf(B, m, gamma) for m in mb]
        bb = mb[int(np.argmax(qbb))]
        out["binarised"][name] = {"alpha": float(a), "edges_kept": int(B.sum()), "Q_on_GF_mean": float(qb.mean()),
                                  "cost": float(qs.mean() - qb.mean()), "NMI": float(NMI(best, bb)), "ARI": float(ARI(best, bb)),
                                  "n_comm": int(len(set(bb))),
                                  "communities": {int(k): [c for c, mm in zip(CONCEPTS, bb) if mm == k] for k in sorted(set(bb))}}
    return out, best


def psu_boot(df, rng):
    """Rao-Wu rescaling bootstrap: in each stratum draw n_h - 1 PSUs with replacement and
    multiply weights by n_h / (n_h - 1) times the number of draws (Rao & Wu 1988)."""
    mult = {}
    for _, g in df.groupby("strata"):
        psus = g.psu.unique()
        n = len(psus)
        draws = rng.choice(psus, n - 1, replace=True)
        for p in psus:
            mult[p] = n / (n - 1) * float((draws == p).sum())
    out = df.copy()
    out["w"] = out["w"] * out["psu"].map(mult).to_numpy()
    return out[out["w"] > 0]


def null_test(Ed, kind, n_null, restarts, rng, df=None):
    A = adj(Ed)
    q_obs = best_q(A, range(restarts))[0]
    qn = []
    for _ in range(n_null):
        if kind == "N1":
            En = n1_rewire(Ed, rng)
        elif kind == "N2":
            En = n2_permute(Ed, rng)
        else:
            En, _ = graph(respondent_permute(df, rng))
        qn.append(best_q(adj(En), range(restarts))[0])
    qn = np.array(qn)
    sd = qn.std(ddof=1)
    return {"Q_obs": float(q_obs), "null_mean": float(qn.mean()), "null_sd": float(sd), "dQ": float(q_obs - qn.mean()),
            "z": float((q_obs - qn.mean()) / sd) if sd > 0 else None, "p": float((1 + (qn >= q_obs).sum()) / (n_null + 1)),
            "n_null": n_null, "restarts": restarts, "null_max": float(qn.max())}


if __name__ == "__main__":
    part = sys.argv[1]

    if part == "main":
        R = {}
        Ed, Edev = graph(dev)
        Ed.to_csv("results2/edges_dev.csv", index=False)
        np.save("results2/E_dev.npy", Edev)
        thr = thresholds(Ed)
        R["thresholds"] = thr
        tic("RQ1 main")
        R["rq1_main"], best = rq1(Ed, thr=thr)
        json.dump({"concepts": CONCEPTS, "membership": best.tolist()}, open("results2/best_partition.json", "w"))
        tic("bootstrap")
        boots = {k: [] for k in list(thr) + ["0+"]}
        for b in range(200):
            Eb, _ = graph(psu_boot(dev, rng))
            o, _ = rq1(Eb, restarts=10, seed0=1000 + b, thr=thr)
            for k, v in o["binarised"].items():
                boots[k].append(v["cost"])
        R["rq1_boot_ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in boots.items()}
        tic("cost under N2 nulls")
        nc = {k: [] for k in list(thr) + ["0+"]}
        for b in range(200):
            o, _ = rq1(n2_permute(Ed, rng), restarts=10, seed0=5000 + b, thr=thr)
            for k, v in o["binarised"].items():
                nc[k].append(v["cost"])
        R["cost_null_N2"] = {k: {"mean": float(np.mean(v)), "q025": float(np.quantile(v, .025)), "q975": float(np.quantile(v, .975)),
                                 "p_ge": float((1 + np.sum(np.array(v) >= R["rq1_main"]["binarised"][k]["cost"])) / 201)} for k, v in nc.items()}
        tic("held-out")
        Eh, Ehold = graph(hold)
        Eh.to_csv("results2/edges_hold.csv", index=False)
        np.save("results2/E_hold.npy", Ehold)
        oh, bh = rq1(Eh, thr=thr)
        R["heldout"] = oh
        R["heldout_vs_dev"] = {"NMI": float(NMI(best, bh)), "ARI": float(ARI(best, bh)),
                               "edge_spearman": float(spearmanr(Edev[OFF], Ehold[OFF]).statistic),
                               "edge_pearson": float(np.corrcoef(Edev[OFF], Ehold[OFF])[0, 1]),
                               "max_abs_diff": float(np.abs(Edev[OFF] - Ehold[OFF]).max())}
        tic("sensitivity")
        sens = {}
        for eps in [0.0, 0.05]:
            sens[f"eps_{eps}"], _ = rq1(graph(dev, eps=eps)[0])
        for gam in [0.5, 1.5]:
            sens[f"gamma_{gam}"], _ = rq1(Ed, gamma=gam, thr=thr)
        Sd, _ = graph(dev, which="S")
        sens["raw_subsethood"], _ = rq1(Sd)
        crisp = dev.copy()
        for c in CONCEPTS:
            crisp[c] = (crisp[c] >= 0.5).astype(float).where(crisp[c].notna())
        sens["crisp_concepts"], _ = rq1(graph(crisp)[0])
        R["sensitivity"] = sens
        json.dump(R, open("results2/rq1.json", "w"), indent=1)
        tic("done main")

    if part == "nulls":
        Ed, _ = graph(dev)
        Eh, _ = graph(hold)
        out = {}
        for kind, n in [("N2", 1000), ("N1", 1000), ("Nresp", 300)]:
            tic(kind)
            out[kind] = null_test(Ed, kind, n, 10, rng, df=dev)
        for kind, n in [("N2", 1000), ("N1", 1000)]:
            tic(kind + " heldout")
            out[kind + "_heldout"] = null_test(Eh, kind, n, 10, rng, df=hold)
        json.dump(out, open("results2/rq2.json", "w"), indent=1)
        tic("done nulls")

    if part == "power":
        Ed, _ = graph(dev)
        bp = json.load(open("results2/best_partition.json"))
        comm = dict(zip(bp["concepts"], bp["membership"]))
        out = {}
        for kind in ["N2", "N1"]:
            for delta in [0.0, 0.1, 0.25, 0.5]:
                tic(f"power {kind} {delta}")
                rej = []
                for _ in range(50):
                    Ep = n2_permute(Ed, rng)
                    inside = np.array([comm[s] == comm[d] for s, d in zip(Ep.src, Ep.dst)])
                    Ep.loc[inside, "mu"] = np.minimum(1.0, Ep.loc[inside, "mu"] * (1 + delta))
                    rej.append(null_test(Ep, kind, 100, 5, rng)["p"] <= 0.05)
                out[f"{kind}_delta_{delta}"] = float(np.mean(rej))
        json.dump(out, open("results2/power.json", "w"), indent=1)
        tic("done power")

    if part == "rq3":
        _, Edev = graph(dev)
        _, Ehold = graph(hold)
        tnorms = ["product", "hamacher", "yager2", "lukasiewicz", "min"]
        tic("indirect inference on development graph")
        IND = {T: indirect(Edev, T, 3) for T in tnorms}
        for T, M in IND.items():
            np.save(f"results2/indirect_{T}.npy", M)
        Eded, _ = graph(dev)
        thr = thresholds(Eded)
        HOPS = {k: binary_hops(Edev, thr[k]) for k in ["q50", "abs0.25"]}

        def metrics(target, Ed_=None, IND_=None, HOPS_=None):
            Ed_ = Edev if Ed_ is None else Ed_
            IND_ = IND if IND_ is None else IND_
            HOPS_ = HOPS if HOPS_ is None else HOPS_
            res = {}
            y = target[OFF]
            res["direct_dev_edge"] = {"spearman": float(spearmanr(Ed_[OFF], y).statistic),
                                      "mae": float(np.abs(Ed_[OFF] - y).mean())}
            for T, M in IND_.items():
                x = M[OFF]
                res[T] = {"spearman": float(spearmanr(x, y).statistic), "mae": float(np.abs(x - y).mean()),
                          "bias": float((x - y).mean()), "share_zero": float((x == 0).mean())}
            for k, H in HOPS_.items():
                x = -np.where(np.isinf(H), 4, H)[OFF]
                res[f"binary_{k}"] = {"spearman": float(spearmanr(x, y).statistic),
                                      "coverage": float(np.isfinite(H[OFF]).mean()),
                                      "n_ties_levels": int(len(np.unique(x)))}
            return res

        tic("metrics on held-out")
        R = {"thresholds": thr, "held_out": metrics(Ehold)}
        tic("T-transitivity of the held-out relation")
        trans = {}
        for T in tnorms:
            C = compose(Ehold, T)
            v = (C - Ehold)[OFF]
            trans[T] = {"violation_share": float((v > 0.01).mean()), "mean_excess_when_violated": float(v[v > 0.01].mean()) if (v > 0.01).any() else 0.0}
        R["transitivity_heldout"] = trans
        tic("Rao-Wu bootstrap of BOTH periods (B=200): development graph and held-out target rebuilt")
        keys = ["direct_dev_edge"] + tnorms + [f"binary_{k}" for k in HOPS]
        bs = {k: [] for k in keys}
        for b in range(200):
            _, Edb = graph(psu_boot(dev, rng))
            _, Eb = graph(psu_boot(hold, rng))
            INDb = {T: indirect(Edb, T, 3) for T in tnorms}
            HOPSb = {k: binary_hops(Edb, thr[k]) for k in ["q50", "abs0.25"]}
            m = metrics(Eb, Edb, INDb, HOPSb)
            for k in keys:
                bs[k].append(m[k]["spearman"])
        R["bootstrap"] = "Rao-Wu, both periods resampled"
        R["spearman_ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in bs.items()}
        R["spearman_diff_vs_product_ci"] = {k: [float(np.quantile(np.array(v) - np.array(bs["product"]), .025)),
                                                float(np.quantile(np.array(v) - np.array(bs["product"]), .975))]
                                            for k, v in bs.items() if k != "product"}
        json.dump(R, open("results2/rq3.json", "w"), indent=1)
        tic("done rq3")
