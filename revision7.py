"""Additional analyses for the J-BHI version (v1.4.0).
Parts:
  ident     - prevalence-odds identity of edge direction: crisp-crisp pairs vs pairs with graded concepts
  rank1gap  - why product composition outranks global rank-1 completion: block-wise rank-1 completion
  phqmap    - non-linear PHQ-9 item mappings (activations rebuilt, RQ3 rerun)
  shrink    - empirical-Bayes shrinkage of low-information edges towards the type-block mean; exclusion of edges
              with Kish effective sample < 100
  jrr       - jackknife repeated replication (JRR, one PSU deleted per stratum) standard errors for RQ3,
              and the attenuation of rho under the Rao-Wu bootstrap
  coverage  - coverage of JRR and Rao-Wu percentile intervals in a clustered respondent-level simulation
  blocksum  - withheld-block summary against soft-impute (CV), Katz and prevalence-type regression with sign tests
  fastcheck - FastCompose (fast_compose.py) reproduces the enumerated predictions on the NHANES relation
Writes results2/revision7_<part>.json."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest, rankdata
from analysis2 import graph, dev, hold, psu_boot
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
import rq3x as X
import rq6lib as L

part = sys.argv[1]
rng = np.random.default_rng(20261115)
OFF = ~np.eye(P, dtype=bool)
OUT = {}
TYP = np.array([TYPE[c] for c in CONCEPTS])
SYMI = [i for i, c in enumerate(CONCEPTS) if TYPE[c] == "Symptom"]


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/revision7_{part}.json", "w"), indent=1)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def rel(df):
    return relation_matrices(df)[2]


def r(a, b):
    return float(spearmanr(a, b).statistic)


def rq3_core(Ed, Eh, prev, rank3=True, reverse=True):
    o = {T: r(X.paths(Ed, X.T_FIXED[T])[OFF], Eh[OFF]) for T in ["product", "hamacher", "min", "yager2", "lukasiewicz"]}
    if rank3:
        o["rank3"] = r(X.lowrank_baseline(Ed, 3, dyad=True)[OFF], Eh[OFF])
    if reverse:
        o["reverse_odds"] = r(X.reverse_baseline(Ed, prev)[0][OFF], Eh[OFF])
    o["product_best_tnorm"] = bool(o["product"] >= max(o[T] for T in ["hamacher", "min", "yager2", "lukasiewicz"]))
    return o


if part == "ident":
    Ed = rel(dev)
    prev = prevalence(dev)
    lg = np.log(prev / (1 - prev))
    vals = {c: set(np.unique(dev[c].dropna().round(6))) for c in CONCEPTS}
    crisp = np.array([v <= {0.0, 1.0} for v in vals.values()])
    rows = []
    for i, j in itertools.permutations(range(P), 2):
        if i < j and Ed[i, j] >= 0.01 and Ed[j, i] >= 0.01:
            g = "crisp-crisp" if crisp[i] and crisp[j] else ("involves PHQ-9 item" if (i in SYMI or j in SYMI) else "other graded")
            rows.append((g, np.log(Ed[i, j] / Ed[j, i]), lg[j] - lg[i]))
    df = pd.DataFrame(rows, columns=["group", "y", "x"])
    res = {"crisp_concepts": [c for c, k in zip(CONCEPTS, crisp) if k]}
    for g, d in [("all", df)] + list(df.groupby("group")):
        b, a = np.polyfit(d.x, d.y, 1)
        resid = d.y - (a + b * d.x)
        res[g] = {"n_pairs": int(len(d)), "r": float(np.corrcoef(d.x, d.y)[0, 1]), "slope": float(b),
                  "resid_sd": float(resid.std(ddof=1)), "max_abs_dev_from_identity": float(np.abs(d.y - d.x).max())}
    OUT["ident"] = res
    save()

if part == "rank1gap":
    Ed, Eh = rel(dev), rel(hold)
    prod = X.paths(Ed, X.t_product)
    r1 = X.lowrank_baseline(Ed, 1, dyad=True)
    # block-wise rank-1: within each ordered relation-type block, a rank-1 fit on the block's other cells
    TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
    br1 = np.zeros_like(Ed)
    for a in TYPES:
        for b in TYPES:
            I = np.where(TYP == a)[0]; J = np.where(TYP == b)[0]
            B = Ed[np.ix_(I, J)]
            for ii, i in enumerate(I):
                for jj, j in enumerate(J):
                    if i == j:
                        continue
                    m = np.ones_like(B, bool)
                    m[ii, jj] = False
                    if a == b:
                        m[np.arange(len(I)), np.arange(len(I))] = False
                        if j in I and i in J:
                            m[list(I).index(j), list(J).index(i)] = False
                    br1[i, j] = X.soft_impute(B, m, 1)[ii, jj] if m.sum() >= 3 else 0.0
    # rank-1 fitted separately on rows/cols of the product prediction: how close is composition to a factorization?
    OUT["rank1gap"] = {"rho_product": r(prod[OFF], Eh[OFF]), "rho_rank1_global": r(r1[OFF], Eh[OFF]),
                       "rho_rank1_blockwise": r(br1[OFF], Eh[OFF]),
                       "corr_product_vs_rank1_global": r(prod[OFF], r1[OFF]),
                       "corr_product_vs_rank1_blockwise": r(prod[OFF], br1[OFF]),
                       "within_block_rho_product": L.within_block(prod, Eh),
                       "within_block_rho_rank1_global": L.within_block(r1, Eh),
                       "within_block_rho_rank1_blockwise": L.within_block(br1, Eh)}
    save()

if part == "phqmap":
    Eh0 = rel(hold)
    maps = {"concave_0_.5_.8_1": {0: 0.0, 1: 0.5, 2: 0.8, 3: 1.0}, "convex_0_.2_.5_1": {0: 0.0, 1: 0.2, 2: 0.5, 3: 1.0},
            "linear_main": {0: 0.0, 1: 1 / 3, 2: 2 / 3, 3: 1.0}}
    res = {}
    for name, mp in maps.items():
        tic(name)
        dd, hh = dev.copy(), hold.copy()
        for c in [CONCEPTS[i] for i in SYMI]:
            for d in (dd, hh):
                k = np.round(d[c] * 3)
                d[c] = k.map(mp)
        Ed, Eh = rel(dd), rel(hh)
        o = rq3_core(Ed, Eh, prevalence(dd))
        o["rho_dev_vs_main_relation"] = r(Ed[OFF], rel(dev)[OFF]) if name != "linear_main" else 1.0
        res[name] = o
    OUT["phqmap"] = res
    save()

if part == "shrink":
    Ed, Eh = rel(dev), rel(hold)
    prev = prevalence(dev)
    neff = np.zeros((P, P))
    for i, j in itertools.permutations(range(P), 2):
        m = dev[CONCEPTS[i]].notna() & dev[CONCEPTS[j]].notna()
        z = (dev.w[m] * np.minimum(dev[CONCEPTS[i]][m], dev[CONCEPTS[j]][m])).to_numpy()
        neff[i, j] = z.sum() ** 2 / (z ** 2).sum() if (z ** 2).sum() > 0 else 0.0
    blk = np.zeros((P, P))
    for i, j in itertools.permutations(range(P), 2):
        m = (TYP[:, None] == TYP[i]) & (TYP[None, :] == TYP[j]) & OFF
        m[i, j] = False
        blk[i, j] = Ed[m].mean()
    res = {"neff_median": float(np.median(neff[OFF])), "n_pairs_neff_below_100": int((neff[OFF] < 100).sum())}
    for kappa in [50, 100, 300]:
        Es = (neff * Ed + kappa * blk) / (neff + kappa)
        np.fill_diagonal(Es, 0)
        res[f"eb_kappa{kappa}"] = rq3_core(Es, Eh, prev)
    low = (neff < 100) & OFF
    Ex = np.where(low, 0.0, Ed)
    keep = OFF & ~low & ~low.T
    o = {}
    for T in ["product", "hamacher", "min"]:
        o[T] = r(X.paths(Ex, X.T_FIXED[T])[keep], Eh[keep])
    o["rank3"] = r(X.lowrank_baseline(Ex, 3, dyad=True)[keep], Eh[keep])
    o["n_pairs_evaluated"] = int(keep.sum())
    res["exclude_neff_below_100"] = o
    OUT["shrink"] = res
    save()


def jrr_replicates(df):
    """JK2 replicates: in each stratum delete one PSU and double the weight of the other."""
    reps = []
    for h, g in df.groupby("strata"):
        psus = sorted(g.psu.unique())
        if len(psus) < 2:
            continue
        d = df.copy()
        d.loc[d.psu == psus[0], "w"] = 0.0
        d.loc[d.psu == psus[1], "w"] *= 2.0
        reps.append(d[d.w > 0])
    return reps


def rq3_stats(Ed, Eh, prev):
    p = X.paths(Ed, X.t_product)[OFF]; hm = X.paths(Ed, X.T_FIXED["hamacher"])[OFF]
    r3 = X.lowrank_baseline(Ed, 3, dyad=True)[OFF]
    y = Eh[OFF]
    rp, rh, rr = r(p, y), r(hm, y), r(r3, y)
    return {"product": rp, "hamacher": rh, "rank3": rr, "product_minus_hamacher": rp - rh, "rank3_minus_product": rr - rp}


if part == "jrr":
    Ed, Eh = rel(dev), rel(hold)
    prev = prevalence(dev)
    full = rq3_stats(Ed, Eh, prev)
    reps = []
    tic("development replicates")
    for d in jrr_replicates(dev):
        reps.append(rq3_stats(rel(d), Eh, prev))
    tic("held-out replicates")
    for h in jrr_replicates(hold):
        reps.append(rq3_stats(Ed, rel(h), prev))
    res = {"n_replicates": len(reps)}
    for k in full:
        v = np.array([x[k] for x in reps])
        se = float(np.sqrt(((v - full[k]) ** 2).sum()))
        res[k] = {"point": full[k], "se": se, "ci95": [full[k] - 1.96 * se, full[k] + 1.96 * se],
                  "replicate_mean": float(v.mean())}
    # Rao-Wu attenuation: one PSU of two drawn per stratum roughly halves the effective sample of every edge
    tic("Rao-Wu replicates for attenuation")
    rw = []
    for b in range(40):
        rw.append(rq3_stats(rel(psu_boot(dev, rng)), rel(psu_boot(hold, rng)), prev))
    res["raowu_replicate_mean"] = {k: float(np.mean([x[k] for x in rw])) for k in full}
    # how noisy is a relation built from one PSU per stratum?
    res["edge_corr_full_vs_raowu_dev"] = float(np.mean([r(rel(psu_boot(dev, rng))[OFF], Ed[OFF]) for _ in range(10)]))
    res["edge_corr_full_vs_jrr_dev"] = float(np.mean([r(rel(d)[OFF], Ed[OFF]) for d in jrr_replicates(dev)[:10]]))
    OUT["jrr"] = res
    save()

if part == "coverage":
    # clustered respondent-level simulation: 45 strata x 2 PSUs, PSU random effects on the latent factors,
    # two independent samples (development and held-out) from the same population; truth from a large sample.
    from rq6lib import path_features
    NS, NPSU, NPER = 45, 2, 170
    kinds = ["ordinal"] * 9 + ["binary"] * 15
    def make_pop(seed):
        g = np.random.default_rng(seed)
        k = 3
        Lm = np.zeros((P, k))
        for c in range(P):
            Lm[c, c % k] = g.uniform(0.4, 0.8)
            Lm[c] += g.uniform(0, 0.25, k)
        # cut-points fixed from a large reference sample
        Zr = g.standard_normal((200000, k)) @ Lm.T + g.standard_normal((200000, P)) * 0.8
        cuts = [np.quantile(Zr[:, c], [0.70, 0.88, 0.96]) if kinds[c] == "ordinal" else np.quantile(Zr[:, c], [0.75]) for c in range(P)]
        def measure(Z):
            A = np.empty_like(Z)
            for c in range(P):
                A[:, c] = np.digitize(Z[:, c], cuts[c]) / (3 if kinds[c] == "ordinal" else 1)
            return A
        def sample(gg, n_per=NPER, icc=0.05):
            rows, strata, psu = [], [], []
            for h in range(NS):
                for p in range(NPSU):
                    u = gg.standard_normal(k) * np.sqrt(icc)
                    F = gg.standard_normal((n_per, k)) + u
                    rows.append(F @ Lm.T + gg.standard_normal((n_per, P)) * 0.8)
                    strata += [f"s{h}"] * n_per; psu += [f"s{h}_{p}"] * n_per
            df = pd.DataFrame(measure(np.vstack(rows)), columns=CONCEPTS)
            df["w"] = 1.0; df["strata"] = strata; df["psu"] = psu
            return df
        return sample
    reps_out = []
    NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for rep in range(START, START + NREP):
        sample = make_pop(5000 + rep)
        g = np.random.default_rng(9000 + rep)
        # truth: large unclustered samples of the same population
        big1, big2 = sample(g, n_per=1500, icc=0.05), sample(g, n_per=1500, icc=0.05)
        truth = rq3_stats(rel(big1), rel(big2), None)
        d1, d2 = sample(g), sample(g)
        E1, E2 = rel(d1), rel(d2)
        est = rq3_stats(E1, E2, None)
        jr = [rq3_stats(rel(d), E2, None) for d in jrr_replicates(d1)] + [rq3_stats(E1, rel(d), None) for d in jrr_replicates(d2)]
        rw = [rq3_stats(rel(psu_boot(d1, g)), rel(psu_boot(d2, g)), None) for _ in range(20)]
        row = {}
        for k in ["product", "product_minus_hamacher", "rank3_minus_product"]:
            se = np.sqrt(sum((x[k] - est[k]) ** 2 for x in jr))
            v = np.array([x[k] for x in rw]); bias = v.mean() - est[k]
            lo, hi = np.quantile(v, [0.025, 0.975]) - bias
            row[k] = {"truth": truth[k], "est": est[k], "jrr_cover": bool(abs(est[k] - truth[k]) <= 1.96 * se),
                      "rw_shift_cover": bool(lo <= truth[k] <= hi), "jrr_se": float(se), "rw_width": float(hi - lo)}
        reps_out.append(row)
        tic(f"rep {rep}")
    res = {}
    for k in reps_out[0]:
        res[k] = {"jrr_coverage": float(np.mean([x[k]["jrr_cover"] for x in reps_out])),
                  "raowu_shifted_coverage": float(np.mean([x[k]["rw_shift_cover"] for x in reps_out])),
                  "mean_bias": float(np.mean([x[k]["est"] - x[k]["truth"] for x in reps_out])),
                  "mean_jrr_se": float(np.mean([x[k]["jrr_se"] for x in reps_out])),
                  "mean_raowu_width": float(np.mean([x[k]["rw_width"] for x in reps_out]))}
    res["n_rep"] = len(reps_out)
    res["rows"] = reps_out
    res["design"] = {"strata": NS, "psu_per_stratum": NPSU, "respondents_per_psu": NPER, "icc_latent": 0.05,
                     "generator": "three-factor latent model, PHQ-like ordinal and binary measurement", "rao_wu_reps": 20}
    OUT["coverage"] = res
    json.dump(OUT, open(f"results2/revision7_coverage_{START}.json", "w"), indent=1)

if part == "blocksum":
    B = json.load(open("results2/revision6_blocks.json"))["blocks"]
    res = {"per_block": {}}
    comps = ["softimpute_cv", "katz", "prev_type_reg", "rank3_svd", "learned_paths"]
    for blk, d in B.items():
        if not isinstance(d, dict) or "product" not in d:
            continue
        res["per_block"][blk] = {m: d[m]["rho"] for m in ["product", "hamacher"] + comps if m in d}
        res["per_block"][blk]["n_pairs"] = d.get("n_pairs")
    for m in comps:
        wins = [v["product"] > v[m] for v in res["per_block"].values() if m in v]
        res[f"product_vs_{m}"] = {"product_better": int(sum(wins)), "n_blocks": len(wins),
                                 "sign_test_p": float(binomtest(sum(wins), len(wins)).pvalue),
                                 "median_rho_other": float(np.median([v[m] for v in res["per_block"].values() if m in v]))}
    res["median_rho_product"] = float(np.median([v["product"] for v in res["per_block"].values()]))
    OUT["blocksum"] = res
    save()

if part == "fastcheck":
    from fast_compose import fast, TN
    Ed = rel(dev)
    mp = {"product": "product", "hamacher": "hamacher", "minimum": "min", "yager": "yager2", "lukasiewicz": "lukasiewicz"}
    res = {}
    for k, T in TN.items():
        a = fast(Ed, T)
        b = X.paths(Ed, X.T_FIXED[mp[k]])
        res[k] = float(np.abs(a - b)[OFF].max())
    OUT["fastcheck"] = {"max_abs_diff_vs_enumeration": res}
    save()

if part == "covsum":
    rows = []
    for s in [0, 12]:
        rows += json.load(open(f"results2/revision7_coverage_{s}.json"))["coverage"]["rows"]
    out = {"n_rep": len(rows)}
    for k in rows[0]:
        out[k] = {"jrr_cov": float(np.mean([x[k]["jrr_cover"] for x in rows])),
                  "rw_cov": float(np.mean([x[k]["rw_shift_cover"] for x in rows])),
                  "bias": float(np.mean([x[k]["est"] - x[k]["truth"] for x in rows])),
                  "jrr_se": float(np.mean([x[k]["jrr_se"] for x in rows])),
                  "rw_w": float(np.mean([x[k]["rw_width"] for x in rows])),
                  "emp_sd": float(np.std([x[k]["est"] - x[k]["truth"] for x in rows], ddof=1)),
                  "truth": float(np.mean([x[k]["truth"] for x in rows]))}
    json.dump(out, open("results2/revision7_coverage_summary.json", "w"), indent=1)
