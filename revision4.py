"""Analyses requested in the fourth review.
Parts:
  sens  - RQ3 under the Table 6 settings that lacked it (edge threshold, crisp concepts, raw subsethood,
          membership variants) and with PHQ-9 items scored as binary (item >= 2, "more than half the days").
  swap  - periods swapped: relation built from 2007-2012, predicted relation measured in 2013-2018
          (point estimates and 200 two-period Rao-Wu replicates).
  split - within-period evaluation: 2013-2018 split by primary sampling unit within each stratum,
          relation built from one half and evaluated on the other (20 random splits).
Writes results2/revision4_<part>.json."""
import json, sys, time
import numpy as np
import pandas as pd
from analysis2 import graph, dev, hold, psu_boot, OFF
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
from clean2 import SYM
import rq3x as X

part = sys.argv[1]
rng = np.random.default_rng(20261101)
OUT = {}
T5 = ["product", "hamacher", "yager2", "lukasiewicz", "min"]
KEYS = T5 + ["reverse_odds", "rank3"]


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/revision4_{part}.json", "w"), indent=1)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def predict(Ein, prev):
    out = {T: X.paths(Ein, X.T_FIXED[T]) for T in T5}
    out["reverse_odds"], _ = X.reverse_baseline(Ein, prev)
    out["rank3"] = X.lowrank_baseline(Ein, 3, dyad=True)
    return out


def evaluate(Ein, Etarget, prev):
    pr = predict(Ein, prev)
    off = ~np.eye(len(Ein), dtype=bool)
    r = {k: X.rho(v[off], Etarget[off]) for k, v in pr.items()}
    r["best_tnorm"] = max(T5, key=lambda t: r[t] if np.isfinite(r[t]) else -9)
    return r


def rel(df, which="E"):
    S, S0, E = relation_matrices(df)
    return {"E": E, "S": S}[which]


if part == "sens":
    res = {}
    Ed, Eh = rel(dev), rel(hold)
    pd_ = prevalence(dev)
    for eps in [0.0, 0.01, 0.05]:
        tic(f"edge threshold {eps}")
        res[f"eps_{eps}"] = evaluate(np.where(Ed >= eps, Ed, 0.0) if eps > 0 else Ed, Eh, pd_)
    tic("crisp concepts")
    cd, ch = dev.copy(), hold.copy()
    for c in CONCEPTS:
        cd[c] = (cd[c] >= 0.5).astype(float).where(cd[c].notna())
        ch[c] = (ch[c] >= 0.5).astype(float).where(ch[c].notna())
    res["crisp_concepts"] = evaluate(rel(cd), rel(ch), prevalence(cd))
    tic("PHQ-9 items binary at >= 2")
    bd, bh = dev.copy(), hold.copy()
    for c in SYM:
        bd[c] = (bd[c] >= 2 / 3 - 1e-9).astype(float).where(bd[c].notna())
        bh[c] = (bh[c] >= 2 / 3 - 1e-9).astype(float).where(bh[c].notna())
    res["phq_binary"] = evaluate(rel(bd), rel(bh), prevalence(bd))
    tic("raw subsethood S")
    Sd, Sh = rel(dev, "S"), rel(hold, "S")
    res["raw_S"] = evaluate(Sd, Sh, pd_)
    for c in ["low_education", "smoking", "poor_self_rated_health", "diabetes", "cancer"]:
        tic(f"membership variant {c}")
        vd, vh = dev.copy(), hold.copy()
        vd[c] = vd[c + "_alt"]
        if c + "_alt" in vh.columns:
            vh[c] = vh[c + "_alt"]
        res[f"variant_{c}"] = evaluate(rel(vd), rel(vh), prevalence(vd))
    for c in ["short_sleep", "poor_self_rated_health", "self_harm_thoughts"]:
        tic(f"without {c}")
        k = CONCEPTS.index(c); keep = [i for i in range(P) if i != k]
        res[f"drop_{c}"] = evaluate(Ed[np.ix_(keep, keep)], Eh[np.ix_(keep, keep)], pd_[keep])
    tic("complete cases")
    cc_d, cc_h = dev.dropna(subset=CONCEPTS), hold.dropna(subset=CONCEPTS)
    res["complete_cases"] = evaluate(rel(cc_d), rel(cc_h), prevalence(cc_d))
    OUT["sens"] = res
    save(); tic("done sens")

if part == "swap":
    Ed, Eh = rel(dev), rel(hold)
    point = evaluate(Eh, Ed, prevalence(hold))
    tic("swap bootstrap (200)")
    draws = {k: [] for k in KEYS}
    best = []
    for b in range(200):
        if b % 20 == 0:
            tic(f"replicate {b}")
        db, hb = psu_boot(dev, rng), psu_boot(hold, rng)
        r = evaluate(rel(hb), rel(db), prevalence(hb))
        for k in KEYS:
            draws[k].append(r[k])
        best.append(r["best_tnorm"])
    def sh(k):
        d = np.array(draws[k], float); d = d[np.isfinite(d)]; bias = d.mean() - point[k]
        return [float(np.quantile(d, .025) - bias), float(np.quantile(d, .975) - bias)]
    diff = {}
    for k in ["hamacher", "min", "reverse_odds", "rank3"]:
        dd = np.array(draws[k]) - np.array(draws["product"])
        diff[k] = {"point": point[k] - point["product"], "ci": X.bc_ci(dd, point[k] - point["product"])}
    OUT["swap"] = {"point": point, "ci_shifted": {k: sh(k) for k in KEYS}, "diff_vs_product": diff,
                   "share_product_best_tnorm": float(np.mean([b_ == "product" for b_ in best]))}
    save(); tic("done swap")

if part == "split":
    rows = []
    for s in range(20):
        tic(f"split {s}")
        a_psu = set()
        for (cy, st), g in dev.groupby(["cycle", "strata"]):
            ps = list(g.psu.unique()); rng.shuffle(ps)
            for p in ps[: max(1, len(ps) // 2)]:
                a_psu.add((cy, st, p))
        key = list(zip(dev.cycle, dev.strata, dev.psu))
        inA = np.array([k in a_psu for k in key])
        A, B = dev[inA], dev[~inA]
        r = evaluate(rel(A), rel(B), prevalence(A))
        r["n_A"], r["n_B"] = int(len(A)), int(len(B))
        rows.append(r)
    D = pd.DataFrame(rows)
    OUT["split"] = {"n_splits": 20, "mean": {k: float(D[k].mean()) for k in KEYS},
                    "min": {k: float(D[k].min()) for k in KEYS}, "max": {k: float(D[k].max()) for k in KEYS},
                    "share_product_best_tnorm": float((D["best_tnorm"] == "product").mean()),
                    "share_reverse_above_product": float((D["reverse_odds"] > D["product"]).mean()),
                    "share_rank3_above_product": float((D["rank3"] > D["product"]).mean()),
                    "n_A_mean": float(D.n_A.mean()), "n_B_mean": float(D.n_B.mean()),
                    "rows": rows}
    save(); tic("done split")

if part == "simdesign":
    # realised properties of the matched simulation (settings from revision3_sim2.json)
    st = json.load(open("results2/revision3_sim2.json"))["settings"]

    def closure(Bw, T, iters=30):
        R = Bw.copy()
        for _ in range(iters):
            Rn = np.maximum(R, X.paths(R, T, max_len=2))
            if np.allclose(Rn, R):
                break
            R = Rn
        np.fill_diagonal(R, 0)
        return R

    def truth(g, scale, r_):
        if g.endswith("closure"):
            T = {"min_closure": X.t_min, "product_closure": X.t_product, "hamacher_closure": X.t_hamacher_lam(0.0)}[g]
            Bw = np.where(r_.random((P, P)) < 0.12, scale * r_.beta(2, 5, (P, P)), 0.0); np.fill_diagonal(Bw, 0)
            return closure(Bw, T)
        lam = np.sqrt(scale) * r_.beta(2, 4, P)
        Rt = np.outer(lam, lam)
        if g == "rank3_blocks":
            grp = r_.integers(0, 3, P); Rt = Rt * np.where(grp[:, None] == grp[None, :], 1.0, 0.25)
        np.fill_diagonal(Rt, 0)
        return Rt

    def noisy(R, s, r_):
        Z = np.clip(R * np.exp(r_.normal(0, s, R.shape)), 0, 1); Z[Z < 0.01] = 0; np.fill_diagonal(Z, 0)
        return Z

    res = {}
    for g, v in st.items():
        dens_t, dens_n = [], []
        for _ in range(200):
            Rt = truth(g, v["scale"], rng); Rn = noisy(Rt, v["noise_sd"], rng)
            dens_t.append(float((Rt[OFF] >= 0.01).mean())); dens_n.append(float((Rn[OFF] > 0).mean()))
        res[g] = {"density_true": float(np.mean(dens_t)), "density_noisy": float(np.mean(dens_n))}
    OUT["simdesign"] = {"realised": res, "observed_density": float((graph(dev)[1][OFF] >= 0.01).mean())}
    save(); tic("done simdesign")
