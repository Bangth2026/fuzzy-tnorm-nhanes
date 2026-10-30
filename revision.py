"""Analyses requested in the pre-submission review (M1-M9, minor 4, 5, 7, 8, 9).
Writes results2/revision.json. Run after analysis2.py main (needs results2/rq1.json, best_partition.json)."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
import igraph as ig
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.covariance import graphical_lasso
from analysis2 import graph, rq1, dev, hold, psu_boot, thresholds, OFF, null_test
from graph2 import (CONCEPTS, TYPE, P, SYM, SOC, LIFE, HEALTH, relation_matrices, edges_from, adj, leiden, qf,
                    best_q, n2_permute, respondent_permute, TNORM, compose, indirect)

rng = np.random.default_rng(20261101)
T5 = ["product", "hamacher", "yager2", "lukasiewicz", "min"]
OUT = {}
part = sys.argv[1] if len(sys.argv) > 1 else "all"


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/revision_{part}.json", "w"), indent=1)


Ed, Edev = graph(dev)
Eh, Ehold = graph(hold)
S_dev, S0_dev, _ = relation_matrices(dev)
A = adj(Ed)
main = np.array(json.load(open("results2/best_partition.json"))["membership"])
R1 = json.load(open("results2/rq1.json"))
TYP = np.array([TYPE[c] for c in CONCEPTS])
BLOCK = np.array([[f"{TYP[i]}->{TYP[j]}" for j in range(P)] for i in range(P)])
ix = {c: k for k, c in enumerate(CONCEPTS)}


def permute_full(M, rng):
    """Permute a full relation matrix within relation-type blocks (off-diagonal)."""
    out = M.copy()
    for b in np.unique(BLOCK[OFF]):
        m = (BLOCK == b) & OFF
        out[m] = rng.permutation(M[m])
    return out


def holm(ps):
    k = sorted(ps, key=ps.get)
    adj_, run = {}, 0.0
    for r, name in enumerate(k):
        run = max(run, min(1.0, (len(ps) - r) * ps[name]))
        adj_[name] = run
    return adj_


# ------------------------------------------------------------------ M1
if part in ("all", "m1"):
    tic("M1: noise floor from respondent-permutation graphs")
    vals, maxs = [], []
    for b in range(100):
        _, En = graph(respondent_permute(dev, rng))
        vals.append(En[OFF]); maxs.append(En[OFF].max())
    vals = np.concatenate(vals)
    floors = {"p95": float(np.quantile(vals, .95)), "p99": float(np.quantile(vals, .99)), "mean_max": float(np.mean(maxs))}
    OUT["m1_noise_floor"] = {k: round(v, 4) for k, v in floors.items()}
    OUT["m1_edges_above"] = {k: int((Ed.mu > v).sum()) for k, v in floors.items()}
    OUT["m1_holm"] = {k: round(v, 3) for k, v in holm({k: v["p_ge"] for k, v in R1["cost_null_N2"].items()}).items()}
    sub = {}
    for k in ["p95", "p99", "mean_max"]:
        tic(f"M1: RQ1 on edges above the {k} noise floor")
        Es = Ed[Ed.mu > floors[k]].reset_index(drop=True)
        o, best = rq1(Es)
        nc = {kk: [] for kk in o["binarised"]}
        thr_s = thresholds(Es)
        for b in range(200):
            on, _ = rq1(n2_permute(Es, rng), restarts=10, seed0=7000 + b, thr=thr_s)
            for kk, v in on["binarised"].items():
                nc[kk].append(v["cost"])
        sub[k] = {"edges": int(len(Es)), "Q": o["Q_mean"], "n_comm": o["n_comm"], "NMI_vs_main": float(NMI(main, best)),
                  "communities": o["communities"],
                  "binarised": {kk: {"cost": v["cost"], "n_comm": v["n_comm"], "NMI": v["NMI"],
                                     "null_mean": float(np.mean(nc[kk])),
                                     "p": float((1 + np.sum(np.array(nc[kk]) >= v["cost"])) / 201)}
                                for kk, v in o["binarised"].items()}}
    OUT["m1_subgraph"] = sub
    save()

# ------------------------------------------------------------------ M2, minor 4, 5
def baselines(E_d):
    so = E_d.sum(1)
    si = E_d.sum(0)
    strength = np.zeros((P, P)); block = np.zeros((P, P))
    for i, j in itertools.permutations(range(P), 2):
        strength[i, j] = (so[i] - E_d[i, j]) * (si[j] - E_d[i, j])
        m = (BLOCK == BLOCK[i, j]) & OFF
        m[i, j] = False
        block[i, j] = E_d[m].mean() if m.any() else 0.0
    return {"strength_product": strength, "block_mean": block}


def rank_partial(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    def res(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    return float(np.corrcoef(res(rx, rz), res(ry, rz))[0, 1])


def best23(R, T):
    b2, b3 = np.zeros_like(R), np.zeros_like(R)
    for i, j in itertools.permutations(range(R.shape[0]), 2):
        mids = [k for k in range(R.shape[0]) if k not in (i, j)]
        b2[i, j] = max(float(TNORM[T](R[i, k], R[k, j])) for k in mids)
        b3[i, j] = max(float(TNORM[T](TNORM[T](R[i, k], R[k, l]), R[l, j])) for k, l in itertools.permutations(mids, 2))
    return b2, b3


if part in ("all", "m2"):
    tic("M2: baselines and permutation of the development relation")
    y = Ehold[OFF]
    IND = {T: np.load(f"results2/indirect_{T}.npy") for T in T5}
    BL = baselines(Edev)
    res = {k: float(spearmanr(v[OFF], y).statistic) for k, v in BL.items()}
    res.update({T: float(spearmanr(IND[T][OFF], y).statistic) for T in T5})
    OUT["m2_rho"] = res
    OUT["m2_partial_product_given_strength"] = rank_partial(IND["product"][OFF], y, BL["strength_product"][OFF])
    OUT["m2_partial_product_given_block"] = rank_partial(IND["product"][OFF], y, BL["block_mean"][OFF])
    OUT["m2_within_dev"] = {T: float(spearmanr(IND[T][OFF], Edev[OFF]).statistic) for T in T5}
    OUT["m2_within_dev"].update({k: float(spearmanr(v[OFF], Edev[OFF]).statistic) for k, v in BL.items()})
    tic("M2: t-norm composition on 200 within-type permutations of the development relation")
    perm = {T: [] for T in T5}
    for b in range(200):
        Ep = permute_full(Edev, rng)
        for T in T5:
            perm[T].append(float(spearmanr(indirect(Ep, T, 3)[OFF], y).statistic))
    OUT["m2_perm"] = {T: {"mean": float(np.mean(v)), "q025": float(np.quantile(v, .025)), "q975": float(np.quantile(v, .975)),
                          "p": float((1 + np.sum(np.array(v) >= res[T])) / 201)} for T, v in perm.items()}
    tic("M2: two-period Rao-Wu bootstrap of product minus baselines")
    diffs = {"strength_product": [], "block_mean": []}
    for b in range(200):
        _, Edb = graph(psu_boot(dev, rng)); _, Ehb = graph(psu_boot(hold, rng))
        yb = Ehb[OFF]
        rp = spearmanr(indirect(Edb, "product", 3)[OFF], yb).statistic
        for k, v in baselines(Edb).items():
            diffs[k].append(float(rp - spearmanr(v[OFF], yb).statistic))
    OUT["m2_diff_ci"] = {k: [float(np.quantile(v, .025)), float(np.quantile(v, .975))] for k, v in diffs.items()}
    tic("minor 4-5: path length and tie-robust metrics")
    pl, kt = {}, {}
    for T in T5:
        b2, b3 = best23(Edev, T)
        both = np.maximum(b2, b3)
        pl[T] = {"rho_two_hop": float(spearmanr(b2[OFF], y).statistic),
                 "share_best_is_three_hop": float(((b3 > b2) & (both > 0))[OFF].sum() / max(1, (both > 0)[OFF].sum()))}
        x = IND[T][OFF]
        nz = x > 0
        kt[T] = {"kendall_tau_b": float(kendalltau(x, y).statistic),
                 "spearman_nonzero": float(spearmanr(x[nz], y[nz]).statistic) if nz.sum() > 10 else None,
                 "n_nonzero": int(nz.sum())}
    OUT["minor4_path_length"] = pl
    OUT["minor5_ties"] = kt
    save()

# ------------------------------------------------------------------ M3
if part in ("all", "m3"):
    tic("M3: transitivity against within-type permutation of the held-out relation")
    def viol(R, T, tol):
        v = (compose(R, T) - R)[OFF]
        return float((v > tol).mean())
    def excess(R, T):
        return float(np.maximum(0, (compose(R, T) - R)[OFF]).mean())
    obs = {T: {"tol0.005": viol(Ehold, T, .005), "tol0.01": viol(Ehold, T, .01), "tol0.02": viol(Ehold, T, .02),
               "mean_excess": excess(Ehold, T)} for T in T5}
    nul = {T: {"tol0.01": [], "mean_excess": []} for T in T5}
    for b in range(200):
        Rp = permute_full(Ehold, rng)
        for T in T5:
            nul[T]["tol0.01"].append(viol(Rp, T, .01)); nul[T]["mean_excess"].append(excess(Rp, T))
    OUT["m3"] = {T: {**obs[T], "null_tol0.01_mean": float(np.mean(nul[T]["tol0.01"])),
                     "null_tol0.01_range": [float(np.quantile(nul[T]["tol0.01"], .025)), float(np.quantile(nul[T]["tol0.01"], .975))],
                     "p_lower_tol0.01": float((1 + np.sum(np.array(nul[T]["tol0.01"]) <= obs[T]["tol0.01"])) / 201),
                     "null_excess_mean": float(np.mean(nul[T]["mean_excess"])),
                     "p_lower_excess": float((1 + np.sum(np.array(nul[T]["mean_excess"]) <= obs[T]["mean_excess"])) / 201)}
                 for T in T5}
    save()

# ------------------------------------------------------------------ M4
if part in ("all", "m4"):
    tic("M4: a-priori type partitions")
    typ4 = np.array([["Symptom", "Socioeconomic", "Lifestyle", "Health"].index(TYPE[c]) for c in CONCEPTS])
    out = {"Q_type4": qf(A, typ4), "NMI_main_type4": float(NMI(main, typ4))}
    life = [ix[c] for c in LIFE]
    best, qs = None, []
    for assign in itertools.product([0, 1, 3], repeat=len(life)):
        m = typ4.copy()
        for k, a in zip(life, assign):
            m[k] = a
        q = qf(A, m); qs.append(q)
        if best is None or q > best[0]:
            best = (q, m.copy())
    out["Q_type3_best"] = best[0]
    out["Q_type3_range"] = [float(min(qs)), float(max(qs))]
    out["type3_best_equals_main"] = bool(NMI(best[1], main) > 0.999)
    out["Q_type3_symptom_only_rule"] = None
    out["Q_main"] = qf(A, main)
    tic("M4: bootstrap co-assignment (200 Rao-Wu replicates, best of 10 restarts)")
    co = np.zeros((P, P))
    for b in range(200):
        Eb, _ = graph(psu_boot(dev, rng))
        _, m, _ = best_q(adj(Eb), range(10))
        m = np.asarray(m)
        co += (m[:, None] == m[None, :])
    co /= 200
    stab = {}
    for c in CONCEPTS:
        i = ix[c]
        same = [j for j in range(P) if main[j] == main[i] and j != i]
        stab[c] = float(co[i, same].mean())
    out["coassign_stability"] = dict(sorted(stab.items(), key=lambda kv: kv[1]))
    OUT["m4"] = out
    save()

# ------------------------------------------------------------------ M5
if part in ("all", "m5"):
    tic("M5: negative excess and asymmetry")
    raw = np.where(S0_dev < 1, (S_dev - S0_dev) / (1 - S0_dev), 0)
    neg = [(CONCEPTS[i], CONCEPTS[j], float(raw[i, j])) for i, j in itertools.permutations(range(P), 2) if raw[i, j] < 0]
    neg.sort(key=lambda t: t[2])
    OUT["m5_negative"] = {"n_negative": len(neg), "n_below_minus0.01": int(sum(v < -0.01 for *_, v in neg)),
                          "most_negative": [[a, b, round(v, 3)] for a, b, v in neg[:6]]}
    act = {c: float(np.average(dev[c].dropna(), weights=dev.w[dev[c].notna()])) for c in CONCEPTS}
    logit = lambda p: np.log(p / (1 - p))
    x, yv = [], []
    for i, j in itertools.combinations(range(P), 2):
        if Edev[i, j] > 0.01 and Edev[j, i] > 0.01:
            yv.append(np.log(Edev[i, j] / Edev[j, i]))
            x.append(logit(act[CONCEPTS[j]]) - logit(act[CONCEPTS[i]]))
    x, yv = np.array(x), np.array(yv)
    r = np.corrcoef(x, yv)[0, 1]
    OUT["m5_asymmetry"] = {"n_pairs": int(len(x)), "pearson_r": float(r), "r2": float(r ** 2),
                           "spearman": float(spearmanr(x, yv).statistic)}
    save()

# ------------------------------------------------------------------ M6: psychometric network (EGA-style)
if part in ("all", "m6"):
    tic("M6: regularised partial-correlation network with EBIC and walktrap")
    X = dev[CONCEPTS].to_numpy(float); w = dev.w.to_numpy(float)
    C = np.eye(P); nmin = len(X)
    for i, j in itertools.combinations(range(P), 2):
        m = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
        nmin = min(nmin, int(m.sum()))
        C[i, j] = C[j, i] = np.cov(X[m, i], X[m, j], aweights=w[m])[0, 1] / np.sqrt(
            np.cov(X[m, i], aweights=w[m]) * np.cov(X[m, j], aweights=w[m]))
    ev, V = np.linalg.eigh(C); C = V @ np.diag(np.clip(ev, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    best = None
    for lam in np.geomspace(0.001, 0.5, 60):
        cov, prec = graphical_lasso(C, alpha=lam, max_iter=500)
        k = int((np.abs(prec[np.triu_indices(P, 1)]) > 1e-8).sum())
        ll = nmin / 2 * (np.log(np.linalg.det(prec)) - np.trace(C @ prec))
        ebic = -2 * ll + k * np.log(nmin) + 4 * 0.5 * k * np.log(P)
        if best is None or ebic < best[0]:
            best = (ebic, lam, prec)
    _, lam, prec = best
    pc = -prec / np.sqrt(np.outer(np.diag(prec), np.diag(prec))); np.fill_diagonal(pc, 0)
    W = np.abs(pc)
    src, dst = np.nonzero(np.triu(W, 1))
    g = ig.Graph(n=P, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = W[src, dst].tolist()
    wt = np.array(g.community_walktrap(weights="weight").as_clustering().membership)
    ld = np.array(g.community_multilevel(weights="weight").membership)
    lab = lambda m: {int(k): [CONCEPTS[i] for i in range(P) if m[i] == k] for k in sorted(set(m))}
    OUT["m6"] = {"lambda": float(lam), "n_edges": int(len(src)), "n_min_pairwise": nmin,
                 "walktrap": lab(wt), "NMI_walktrap_main": float(NMI(main, wt)),
                 "louvain": lab(ld), "NMI_louvain_main": float(NMI(main, ld)),
                 "n_negative_partial": int((pc[np.triu_indices(P, 1)] < -1e-8).sum())}
    save()

# ------------------------------------------------------------------ minor 7, 8, 9 and node-drop sensitivity
if part in ("all", "minor"):
    tic("minor 7: planted delta expressed as Delta Q")
    comm = dict(zip(CONCEPTS, main))
    dq = {}
    for delta in [0.1, 0.25, 0.5]:
        vals = []
        for _ in range(30):
            Ep = n2_permute(Ed, rng)
            inside = np.array([comm[s] == comm[d] for s, d in zip(Ep.src, Ep.dst)])
            Ep.loc[inside, "mu"] = np.minimum(1.0, Ep.loc[inside, "mu"] * (1 + delta))
            q = best_q(adj(Ep), range(5))[0]
            qn = np.mean([best_q(adj(n2_permute(Ep, rng)), range(5))[0] for _ in range(30)])
            vals.append(q - qn)
        dq[str(delta)] = float(np.mean(vals))
    OUT["minor7_deltaQ"] = dq
    tic("minor 8: effective sample size per source concept")
    neff = {}
    for c in CONCEPTS:
        m = dev[c].notna()
        z = (dev.w[m] * dev[c][m]).to_numpy()
        neff[c] = float(z.sum() ** 2 / (z ** 2).sum())
    OUT["minor8_neff"] = dict(sorted(neff.items(), key=lambda kv: kv[1]))
    tic("minor 8: effective sample size per edge (Kish, weights w*min(a_i,a_j) over pairs observed on both)")
    ne_edge = {}
    for r in Ed.itertuples():
        m = dev[r.src].notna() & dev[r.dst].notna()
        z = (dev.w[m] * np.minimum(dev[r.src][m], dev[r.dst][m])).to_numpy()
        ne_edge[f"{r.src}->{r.dst}"] = float(z.sum() ** 2 / (z ** 2).sum()) if (z ** 2).sum() > 0 else 0.0
    v = np.array(list(ne_edge.values()))
    sh = [x for k, x in ne_edge.items() if k.startswith("self_harm_thoughts->")]
    OUT["minor8_neff_edge"] = {"min": float(v.min()), "median": float(np.median(v)), "q10": float(np.quantile(v, .1)),
                              "self_harm_source_median": float(np.median(sh)), "self_harm_source_min": float(min(sh)),
                              "n_edges_below_100": int((v < 100).sum())}
    tic("minor 9 / clinical review: drop one node")
    drop = {}
    yfull = Ehold
    for c in ["short_sleep", "poor_self_rated_health", "self_harm_thoughts"]:
        k = ix[c]; keep = [i for i in range(P) if i != k]
        As = A[np.ix_(keep, keep)]
        q, m, _ = best_q(As, range(100))
        qs = [qf(As, leiden(As, s)) for s in range(100)]
        Ds, Hs = Edev[np.ix_(keep, keep)], yfull[np.ix_(keep, keep)]
        off = ~np.eye(P - 1, dtype=bool)
        drop[c] = {"Q_mean": float(np.mean(qs)), "n_comm": int(len(set(m))), "NMI_vs_main": float(NMI(main[keep], np.asarray(m))),
                   "rho_product": float(spearmanr(indirect(Ds, "product", 3)[off], Hs[off]).statistic),
                   "rho_hamacher": float(spearmanr(indirect(Ds, "hamacher", 3)[off], Hs[off]).statistic),
                   "rho_min": float(spearmanr(indirect(Ds, "min", 3)[off], Hs[off]).statistic)}
    OUT["drop_node"] = drop
    save()

# ------------------------------------------------------------------ M9: exclusions
if part in ("all", "m9"):
    tic("M9: excluded versus included respondents")
    from clean2 import rd, ITEMS
    rows = []
    for s in "EFGHIJ":
        dpq, demo = rd("DPQ", s), rd("DEMO", s)
        d = demo[["RIDAGEYR", "RIAGENDR", "INDFMPIR", "WTMEC2YR"]].join(dpq[ITEMS], how="inner")
        ok_items = d[ITEMS].isin([0, 1, 2, 3]).all(axis=1)
        any_refused = d[ITEMS].isin([7, 9]).any(axis=1)
        reason = np.where(ok_items & (d.WTMEC2YR > 0), "included",
                          np.where(ok_items, "zero_weight", np.where(any_refused, "refused_or_dont_know", "item_missing")))
        d["reason"] = reason; d["period"] = "dev" if s in "HIJ" else "hold"
        rows.append(d)
    d = pd.concat(rows)
    out = {}
    for p, g in d.groupby("period"):
        out[p] = {"by_reason": g.reason.value_counts().to_dict()}
        for grp, gg in [("included", g[g.reason == "included"]), ("excluded", g[g.reason != "included"])]:
            out[p][grp] = {"n": int(len(gg)), "age_mean": float(gg.RIDAGEYR.mean()), "female_share": float((gg.RIAGENDR == 2).mean()),
                           "pir_mean": float(gg.INDFMPIR.mean()), "pir_missing_share": float(gg.INDFMPIR.isna().mean())}
    OUT["m9_exclusions"] = out
    save()
tic("done " + part)
