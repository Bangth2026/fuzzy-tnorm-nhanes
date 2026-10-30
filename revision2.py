"""Analyses requested in the second review (M1-M10 and minor points).
Parts: r3 (RQ3 extensions), boot (two-period Rao-Wu bootstrap with bias-corrected intervals),
sim (simulation with known generating rules), null (strength-preserving null, RQ1 power, Holm families),
misc (symmetric relation for RQ2, marginal-correlation network, complete cases).
Writes results2/revision2_<part>.json."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
import igraph as ig
from scipy.stats import spearmanr, norm
from sklearn.metrics import normalized_mutual_info_score as NMI
from analysis2 import graph, rq1, dev, hold, psu_boot, thresholds, OFF
from graph2 import (CONCEPTS, TYPE, P, relation_matrices, edges_from, adj, leiden, qf, best_q, n2_permute, wmean)
import rq3x as X

part = sys.argv[1]
rng = np.random.default_rng(20261030)
OUT = {}
T5 = ["product", "hamacher", "yager2", "lukasiewicz", "min"]
TYP = np.array([TYPE[c] for c in CONCEPTS])
BLOCK = np.array([[f"{TYP[i]}->{TYP[j]}" for j in range(P)] for i in range(P)])
main = np.array(json.load(open("results2/best_partition.json"))["membership"])


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/revision2_{part}.json", "w"), indent=1)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def signed_E(df):
    S, S0, _ = relation_matrices(df)
    E = np.where(S0 < 1, (S - S0) / (1 - S0), 0.0)
    np.fill_diagonal(E, 0)
    return E


def predictors(Ed, prev, which="all"):
    """All predictors of the held-out relation built from the development relation only."""
    out = {T: X.paths(Ed, X.T_FIXED[T]) for T in T5}
    out["strength"] = X.strength_baseline(Ed)
    out["block"] = X.block_baseline(Ed, BLOCK)
    out["reverse_odds"], _ = X.reverse_baseline(Ed, prev)
    out["dcsbm_type"] = X.dcsbm_baseline(Ed, TYP, dyad=True)
    for k in ([1, 2, 3] if which == "all" else [3]):
        out[f"rank{k}"] = X.lowrank_baseline(Ed, k, dyad=True)
    out["direct"] = Ed.copy()
    return out


_, Edev = graph(dev)
_, Ehold = graph(hold)
prev_dev = prevalence(dev)
y = Ehold[OFF]

# ------------------------------------------------------------------ RQ3 extensions
if part == "r3":
    tic("point estimates of all predictors")
    PR = predictors(Edev, prev_dev)
    OUT["rho"] = {k: X.rho(v[OFF], y) for k, v in PR.items()}
    OUT["mae"] = {k: float(np.abs(v[OFF] - y).mean()) for k, v in PR.items() if k not in ("strength",)}
    OUT["bias"] = {k: float((v[OFF] - y).mean()) for k, v in PR.items() if k not in ("strength",)}
    _, info = X.reverse_baseline(Edev, prev_dev)
    OUT["reverse_fit"] = info
    OUT["reverse_zero_share"] = float((PR["reverse_odds"][OFF] == 0).mean())
    # cell-hidden (not dyad) versions for completeness
    OUT["rho_cell_hidden"] = {f"rank{k}": X.rho(X.lowrank_baseline(Edev, k, dyad=False)[OFF], y) for k in [1, 2, 3]}
    OUT["rho_cell_hidden"]["dcsbm_type"] = X.rho(X.dcsbm_baseline(Edev, TYP, dyad=False)[OFF], y)
    # partial rank correlations of product given each strong non-path predictor
    def rank_partial(a, b, c):
        ra, rb, rc = (pd.Series(v).rank().to_numpy() for v in (a, b, c))
        def res(u, z):
            z1 = np.c_[np.ones_like(z), z]
            return u - z1 @ np.linalg.lstsq(z1, u, rcond=None)[0]
        return float(np.corrcoef(res(ra, rc), res(rb, rc))[0, 1])
    OUT["partial_product_given"] = {k: rank_partial(PR["product"][OFF], y, PR[k][OFF])
                                    for k in ["rank3", "reverse_odds", "dcsbm_type", "strength"]}
    OUT["partial_given_product"] = {k: rank_partial(PR[k][OFF], y, PR["product"][OFF])
                                    for k in ["rank3", "reverse_odds", "dcsbm_type"]}
    # within-development consistency of the non-path predictors
    OUT["within_dev"] = {k: X.rho(v[OFF], Edev[OFF]) for k, v in PR.items() if k != "direct"}
    # leave-dyad-out: paths never use i->j or j->i (k, l not in {i, j}); verify numerically
    tic("leave-dyad-out check")
    chk = 0.0
    for i, j in list(itertools.permutations(range(P), 2))[::23]:
        Em = Edev.copy(); Em[i, j] = 0; Em[j, i] = 0
        chk = max(chk, abs(X.paths(Em, X.t_product)[i, j] - PR["product"][i, j]))
    OUT["leave_dyad_paths_max_abs_change"] = float(chk)

    tic("M3: scale invariance under mu -> mu^a")
    inv = {}
    for a in [0.5, 2.0]:
        Ea, Ha = Edev ** a, Ehold ** a
        inv[str(a)] = {T: X.rho(X.paths(Ea, X.T_FIXED[T])[OFF], Ha[OFF]) for T in T5}
    OUT["power_transform"] = inv
    tic("M3: normalised relations")
    norm_ = {}
    mx = Edev[OFF].max()
    Er = np.zeros_like(Edev); Er[OFF] = pd.Series(Edev[OFF]).rank(method="average").to_numpy() / OFF.sum()
    Er[OFF] = np.where(Edev[OFF] > 0, Er[OFF], 0.0)
    for name, Em in [("divide_by_max", Edev / mx), ("rank_to_unit", Er)]:
        norm_[name] = {T: X.rho(X.paths(Em, X.T_FIXED[T])[OFF], y) for T in T5}
        norm_[name]["share_zero"] = {T: float((X.paths(Em, X.T_FIXED[T])[OFF] == 0).mean()) for T in T5}
    OUT["normalised"] = norm_
    tic("M3: noisy-OR aggregation over all paths")
    OUT["probsum"] = {T: X.rho(X.paths(Edev, X.T_FIXED[T], agg="probsum")[OFF], y) for T in T5}
    tic("M3: parametric families, parameter chosen within development")
    fam = {"hamacher_lambda": (X.t_hamacher_lam, [0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30, 100]),
           "frank_s": (X.t_frank, list(np.geomspace(1e-4, 1e4, 33))),
           "schweizer_sklar_p": (X.t_ss, [-10, -5, -3, -2, -1.5, -1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1])}
    F = {}
    for name, (ctor, grid) in fam.items():
        dv, ho = [], []
        for prm in grid:
            M = X.paths(Edev, ctor(float(prm)))
            dv.append(X.rho(M[OFF], Edev[OFF])); ho.append(X.rho(M[OFF], y))
        k = int(np.nanargmax(dv))
        F[name] = {"grid": [float(g) for g in grid], "rho_within_dev": dv, "rho_heldout": ho,
                   "chosen": float(grid[k]), "rho_heldout_at_chosen": ho[k],
                   "best_heldout": float(grid[int(np.nanargmax(ho))]), "rho_heldout_best": float(np.nanmax(ho))}
    OUT["families"] = F
    tic("M9: signed target and symmetric relation")
    Es_hold = signed_E(hold)
    OUT["signed_target"] = {k: X.rho(v[OFF], Es_hold[OFF]) for k, v in PR.items() if k in T5 + ["rank3", "strength", "direct"]}
    OUT["n_negative_target"] = int((Es_hold[OFF] < 0).sum())
    Gd, Gh = np.sqrt(Edev * Edev.T), np.sqrt(Ehold * Ehold.T)
    up = np.triu(np.ones((P, P), bool), 1)
    sym = {T: X.rho(X.paths(Gd, X.T_FIXED[T])[up], Gh[up]) for T in T5}
    sym["strength"] = X.rho(X.strength_baseline(Gd)[up], Gh[up])
    sym["rank3"] = X.rho(X.lowrank_baseline(Gd, 3, dyad=True)[up], Gh[up])
    sym["direct"] = X.rho(Gd[up], Gh[up])
    OUT["symmetric_geomean"] = sym
    tic("M5: jackknife over the 24 concepts")
    jk = []
    for d in range(P):
        keep = [i for i in range(P) if i != d]
        Ed_, Eh_ = Edev[np.ix_(keep, keep)], Ehold[np.ix_(keep, keep)]
        off = ~np.eye(P - 1, dtype=bool)
        r = {T: X.rho(X.paths(Ed_, X.T_FIXED[T])[off], Eh_[off]) for T in T5}
        r["strength"] = X.rho(X.strength_baseline(Ed_)[off], Eh_[off])
        r["rank3"] = X.rho(X.lowrank_baseline(Ed_, 3)[off], Eh_[off])
        r["reverse_odds"] = X.rho(X.reverse_baseline(Ed_, prev_dev[keep])[0][off], Eh_[off])
        r["dropped"] = CONCEPTS[d]
        jk.append(r)
    OUT["jackknife"] = jk
    J = pd.DataFrame(jk)
    OUT["jackknife_summary"] = {
        "product_first_among_tnorms": int((J[T5].idxmax(1) == "product").sum()),
        "product_above_strength": int((J["product"] > J["strength"]).sum()),
        "rank3_above_product": int((J["rank3"] > J["product"]).sum()),
        "reverse_above_product": int((J["reverse_odds"] > J["product"]).sum()),
        "range": {k: [float(J[k].min()), float(J[k].max())] for k in T5 + ["strength", "rank3", "reverse_odds"]},
        "jk_se": {k: float(np.sqrt((P - 1) / P * ((J[k] - J[k].mean()) ** 2).sum())) for k in T5 + ["strength", "rank3", "reverse_odds"]},
        "jk_se_diff_product_hamacher": float(np.sqrt((P - 1) / P * (((J["product"] - J["hamacher"]) - (J["product"] - J["hamacher"]).mean()) ** 2).sum()))}
    tic("M6c: whole relation blocks unmeasured in development")
    types = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
    hb = {}
    for a, b in itertools.combinations_with_replacement(types, 2):
        H = np.zeros((P, P), bool)
        for i, j in itertools.permutations(range(P), 2):
            if {TYP[i], TYP[j]} == {a, b} and (a != b or TYP[i] == a):
                H[i, j] = True
        Em = np.where(H, 0.0, Edev)
        mask = OFF & ~H
        r = {T: X.rho(X.paths(Em, X.T_FIXED[T])[H], Ehold[H]) for T in T5}
        r["strength"] = X.rho(X.strength_baseline(Em)[H], Ehold[H])
        for k in [1, 2, 3]:
            r[f"rank{k}"] = X.rho(X.soft_impute(Edev, mask, k)[H], Ehold[H])
        r["n_pairs"] = int(H.sum())
        hb[f"{a}<->{b}"] = r
    OUT["hidden_blocks"] = hb
    HB = pd.DataFrame(hb).T
    OUT["hidden_blocks_summary"] = {"n_blocks": len(HB), "product_first_among_tnorms": int((HB[T5].astype(float).idxmax(1) == "product").sum()),
                                    "product_above_rank3": int((HB["product"].astype(float) > HB["rank3"].astype(float)).sum()),
                                    "product_above_strength": int((HB["product"].astype(float) > HB["strength"].astype(float)).sum()),
                                    "median": {k: float(HB[k].astype(float).median()) for k in T5 + ["strength", "rank1", "rank2", "rank3"]}}
    tic("M2: product transitivity of low-rank fits")
    off = OFF
    lr = {}
    for k in [1, 2, 3]:
        L = X.soft_impute(Edev, OFF, k)
        np.fill_diagonal(L, 0)
        C = X.paths(L, X.t_product, max_len=2)
        lr[f"rank{k}"] = {"violation_share": float(((C - L)[off] > 0.01).mean()), "fit_spearman_dev": X.rho(L[off], Edev[off])}
    OUT["lowrank_transitivity"] = lr
    save()
    tic("done r3")

# ------------------------------------------------------------------ bootstrap
if part == "boot":
    tic("point estimates")
    PR = predictors(Edev, prev_dev, which="boot")
    keys = list(PR)
    point = {k: X.rho(v[OFF], y) for k, v in PR.items()}
    B = 200
    draws = {k: [] for k in keys}
    for b in range(B):
        if b % 20 == 0:
            tic(f"replicate {b}")
        dv, hv = psu_boot(dev, rng), psu_boot(hold, rng)
        _, Edb = graph(dv)
        _, Ehb = graph(hv)
        PRb = predictors(Edb, prevalence(dv), which="boot")
        for k in keys:
            draws[k].append(X.rho(PRb[k][OFF], Ehb[OFF]))
    D = {k: np.array(v) for k, v in draws.items()}
    OUT["point"] = point
    OUT["bc_ci"] = {k: X.bc_ci(D[k], point[k]) for k in keys}
    OUT["percentile_ci"] = {k: [float(np.quantile(D[k], .025)), float(np.quantile(D[k], .975))] for k in keys}
    OUT["boot_mean"] = {k: float(D[k].mean()) for k in keys}
    OUT["diff_vs_product"] = {k: {"point": point[k] - point["product"],
                                  "bc_ci": X.bc_ci(D[k] - D["product"], point[k] - point["product"]),
                                  "share_above_product": float((D[k] > D["product"]).mean())} for k in keys if k != "product"}
    np.savez("results2/revision2_boot_draws.npz", **D)
    save()
    tic("done boot")

# ------------------------------------------------------------------ simulation with known generating rules
if part == "sim":
    sd_dev = np.std(np.log(Edev[OFF & (Edev > 0)]))

    def closure(Bw, T, iters=30):
        R = Bw.copy()
        for _ in range(iters):
            n = R.shape[0]
            C = np.zeros_like(R)
            for k in range(n):
                v = T(R[:, k][:, None], R[k, :][None, :]); v[k, :] = 0; v[:, k] = 0
                C = np.maximum(C, v)
            np.fill_diagonal(C, 0)
            Rn = np.maximum(R, C)
            if np.allclose(Rn, R):
                break
            R = Rn
        return R

    def noisy(R, s, rng_):
        Z = R * np.exp(rng_.normal(0, s, R.shape))
        Z = np.clip(Z, 0, 1); Z[Z < 0.01] = 0; np.fill_diagonal(Z, 0)
        return Z

    gens = {}
    n = P
    off = ~np.eye(n, dtype=bool)
    for g in ["min_closure", "product_closure", "hamacher_closure", "rank1_factor", "rank3_blocks"]:
        tic(f"simulate {g}")
        res = []
        for rep in range(200):
            if g.endswith("closure"):
                T = {"min_closure": X.t_min, "product_closure": X.t_product, "hamacher_closure": X.t_hamacher_lam(0.0)}[g]
                Bw = np.where(rng.random((n, n)) < 0.12, rng.beta(2, 5, (n, n)), 0.0); np.fill_diagonal(Bw, 0)
                Rt = closure(Bw, T)
            elif g == "rank1_factor":
                lam = rng.beta(2, 4, n); Rt = np.outer(lam, lam); np.fill_diagonal(Rt, 0)
            else:
                grp = rng.integers(0, 3, n); lam = rng.beta(2, 4, n)
                W = np.where(grp[:, None] == grp[None, :], 1.0, 0.25)
                Rt = np.outer(lam, lam) * W; np.fill_diagonal(Rt, 0)
            s = 0.35
            Rd, Rh = noisy(Rt, s, rng), noisy(Rt, s, rng)
            r = {T_: X.rho(X.paths(Rd, X.T_FIXED[T_])[off], Rh[off]) for T_ in T5}
            r["direct"] = X.rho(Rd[off], Rh[off])
            res.append(r)
        Rr = pd.DataFrame(res)
        gens[g] = {"mean_rho": {k: float(Rr[k].mean()) for k in T5 + ["direct"]},
                   "share_first": {T_: float((Rr[T5].idxmax(1) == T_).mean()) for T_ in T5}}
    OUT["simulation"] = gens
    OUT["simulation_settings"] = {"replicates": 200, "noise_sd_log": 0.35, "base_density": 0.12, "base_weights": "Beta(2,5)",
                                  "factor_loadings": "Beta(2,4)", "between_block_factor": 0.25, "observed_log_sd_dev": float(sd_dev)}
    save()
    tic("done sim")

# ------------------------------------------------------------------ nulls, RQ1 power, Holm families
if part == "null":
    Ed, _ = graph(dev)
    A = adj(Ed)

    def strength_null(E, rng_, steps_per_edge=60, T0=0.02):
        """Ansmann-Lehnertz style constrained randomisation: weights are swapped within relation-type blocks by simulated
        annealing so that every node's in- and out-strength stays close to the observed values."""
        Ep = n2_permute(E, rng_).reset_index(drop=True)
        src = Ep.src.map({c: k for k, c in enumerate(CONCEPTS)}).to_numpy()
        dst = Ep.dst.map({c: k for k, c in enumerate(CONCEPTS)}).to_numpy()
        w = Ep.mu.to_numpy().copy()
        so_t = np.bincount(E.src.map({c: k for k, c in enumerate(CONCEPTS)}), E.mu, P)
        si_t = np.bincount(E.dst.map({c: k for k, c in enumerate(CONCEPTS)}), E.mu, P)
        so, si = np.bincount(src, w, P), np.bincount(dst, w, P)
        rel = Ep.rel.to_numpy()
        groups = {r: np.nonzero(rel == r)[0] for r in np.unique(rel)}
        glist = [g for g in groups.values() if len(g) > 1]
        cost = np.abs(so - so_t).sum() + np.abs(si - si_t).sum()
        nsteps = steps_per_edge * len(w)
        for t in range(nsteps):
            Tt = T0 * (1 - t / nsteps) + 1e-6
            g = glist[rng_.integers(len(glist))]
            a, b = g[rng_.integers(len(g))], g[rng_.integers(len(g))]
            if a == b or w[a] == w[b]:
                continue
            d = w[b] - w[a]
            so_new = so.copy(); si_new = si.copy()
            so_new[src[a]] += d; so_new[src[b]] -= d; si_new[dst[a]] += d; si_new[dst[b]] -= d
            idx_o = list({src[a], src[b]}); idx_i = list({dst[a], dst[b]})
            dc = (np.abs(so_new[idx_o] - so_t[idx_o]).sum() - np.abs(so[idx_o] - so_t[idx_o]).sum()
                  + np.abs(si_new[idx_i] - si_t[idx_i]).sum() - np.abs(si[idx_i] - si_t[idx_i]).sum())
            if dc <= 0 or rng_.random() < np.exp(-dc / Tt):
                w[a], w[b] = w[b], w[a]; so, si = so_new, si_new; cost += dc
        Ep["mu"] = w
        return Ep, float(cost / (so_t.sum() + si_t.sum()))

    tic("M7: strength-preserving null (200)")
    q_obs = rq1(Ed)[0]["Q_mean"]
    qs, rel_err = [], []
    for b in range(200):
        Ep, e = strength_null(Ed, rng)
        qs.append(best_q(adj(Ep), range(10))[0]); rel_err.append(e)
    qs = np.array(qs)
    OUT["strength_null"] = {"Q_obs": q_obs, "mean": float(qs.mean()), "sd": float(qs.std(ddof=1)), "max": float(qs.max()),
                            "z": float((q_obs - qs.mean()) / qs.std(ddof=1)), "p": float((1 + (qs >= q_obs).sum()) / 201),
                            "strength_rel_error_mean": float(np.mean(rel_err)), "n": 200}
    save()
    tic("M7: RQ1 power - null and bootstrap distributions of the binarisation cost")
    thr = thresholds(Ed)
    obs = rq1(Ed, thr=thr)[0]["binarised"]
    nulls = {k: [] for k in obs}
    boots = {k: [] for k in obs}
    for b in range(200):
        on, _ = rq1(n2_permute(Ed, rng), restarts=10, seed0=9000 + b, thr=thr)
        for k, v in on["binarised"].items():
            nulls[k].append(v["cost"])
        Eb, _ = graph(psu_boot(dev, rng))
        ob, _ = rq1(Eb, restarts=10, seed0=9500 + b, thr=thresholds(Eb))
        for k, v in ob["binarised"].items():
            boots[k].append(v["cost"])
    mdc = {}
    for k in obs:
        nb, bb = np.array(nulls[k]), np.array(boots[k])
        c95 = float(np.quantile(nb, .95))
        dev20 = float(np.quantile(bb - bb.mean(), .20))
        mdc[k] = {"observed": obs[k]["cost"], "null_mean": float(nb.mean()), "null_q95": c95,
                  "boot_sd": float(bb.std(ddof=1)), "mdc_power80": c95 - dev20,
                  "mdc_excess_over_null_mean": c95 - dev20 - float(nb.mean())}
    OUT["rq1_power"] = mdc
    tic("Holm over 18 noise-floor tests")
    m1 = json.load(open("results2/revision_m1.json"))["m1_subgraph"]
    ps = {f"{fl}:{k}": v["p"] for fl, d in m1.items() for k, v in d["binarised"].items()}
    order = sorted(ps, key=ps.get)
    m = len(order); adjp, run = {}, 0
    for r, k in enumerate(order):
        run = max(run, min(1, (m - r) * ps[k])); adjp[k] = run
    OUT["holm_noise_floor_18"] = {k: round(v, 3) for k, v in adjp.items()}
    within = {}
    for fl, d in m1.items():
        p6 = {k: v["p"] for k, v in d["binarised"].items()}
        o6 = sorted(p6, key=p6.get); run = 0; a6 = {}
        for r, k in enumerate(o6):
            run = max(run, min(1, (6 - r) * p6[k])); a6[k] = run
        within[fl] = {k: round(v, 3) for k, v in a6.items()}
    OUT["holm_noise_floor_within6"] = within
    save()
    tic("done null")

# ------------------------------------------------------------------ misc: symmetric RQ2, marginal network, complete cases
if part == "misc":
    tic("M9: RQ2 on the symmetric relation (geometric mean), undirected modularity with N2 null")
    Gd = np.sqrt(Edev * Edev.T)
    rows = [(CONCEPTS[i], CONCEPTS[j], float(Gd[i, j]), f"{'-'.join(sorted([TYP[i], TYP[j]]))}")
            for i, j in itertools.permutations(range(P), 2) if Gd[i, j] >= 0.01]
    Es = pd.DataFrame(rows, columns=["src", "dst", "mu", "rel"])
    o, bs = rq1(Es)
    def sym_perm(E, rng_):
        und = E[E.src < E.dst].copy()
        und["mu"] = und.groupby("rel")["mu"].transform(lambda s: rng_.permutation(s.to_numpy()))
        back = und.rename(columns={"src": "dst", "dst": "src"})
        return pd.concat([und, back], ignore_index=True)
    qn = np.array([best_q(adj(sym_perm(Es, rng)), range(10))[0] for _ in range(200)])
    OUT["symmetric_rq2"] = {"edges_directed_copies": int(len(Es)), "Q": o["Q_mean"], "n_comm": o["n_comm"],
                            "NMI_vs_main": float(NMI(main, bs)), "communities": o["communities"],
                            "null_mean": float(qn.mean()), "null_sd": float(qn.std(ddof=1)),
                            "z": float((o["Q_mean"] - qn.mean()) / qn.std(ddof=1)), "p": float((1 + (qn >= o["Q_mean"]).sum()) / 201)}
    save()
    tic("M10: survey-weighted marginal correlation network")
    Xd = dev[CONCEPTS].to_numpy(float); w = dev.w.to_numpy(float)
    C = np.eye(P)
    for i, j in itertools.combinations(range(P), 2):
        m = ~np.isnan(Xd[:, i]) & ~np.isnan(Xd[:, j])
        C[i, j] = C[j, i] = np.cov(Xd[m, i], Xd[m, j], aweights=w[m])[0, 1] / np.sqrt(
            np.cov(Xd[m, i], aweights=w[m]) * np.cov(Xd[m, j], aweights=w[m]))
    W = np.clip(C, 0, None); np.fill_diagonal(W, 0)
    src, dst = np.nonzero(np.triu(W, 1))
    g = ig.Graph(n=P, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.es["weight"] = W[src, dst].tolist()
    wt = np.array(g.community_walktrap(weights="weight").as_clustering().membership)
    ld = np.array(g.community_multilevel(weights="weight").membership)
    lab = lambda mm: {int(k): [CONCEPTS[i] for i in range(P) if mm[i] == k] for k in sorted(set(mm))}
    OUT["marginal_network"] = {"n_positive": int(len(src)), "n_negative": int((C[np.triu_indices(P, 1)] < 0).sum()),
                               "walktrap": lab(wt), "NMI_walktrap_main": float(NMI(main, wt)),
                               "louvain": lab(ld), "NMI_louvain_main": float(NMI(main, ld))}
    save()
    tic("minor: complete cases (no pairwise deletion)")
    cc_d, cc_h = dev.dropna(subset=CONCEPTS), hold.dropna(subset=CONCEPTS)
    Ecd, Mcd = graph(cc_d)
    _, Mch = graph(cc_h)
    oc, bc = rq1(Ecd)
    OUT["complete_cases"] = {"n_dev": int(len(cc_d)), "n_hold": int(len(cc_h)), "Q": oc["Q_mean"], "n_comm": oc["n_comm"],
                             "NMI_vs_main": float(NMI(main, bc)),
                             "edge_spearman_vs_pairwise": X.rho(Mcd[OFF], Edev[OFF]),
                             "rho": {T: X.rho(X.paths(Mcd, X.T_FIXED[T])[OFF], Mch[OFF]) for T in T5}}
    save()
    tic("done misc")
