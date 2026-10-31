"""Analyses requested in the sixth review (JHIR R1).
Parts:
  main    - RQ3 with stronger baselines (regularised soft-impute, logit soft-impute, NMF, prevalence-and-block
            regression, Katz, weighted common neighbours, Adamic-Adar, resource allocation, personalised PageRank,
            learned path aggregator, path length 4), within-block and per-source evaluation, concept jackknife
  blocks  - withheld relation blocks with the new predictors, MAE, and calibration fitted on measured blocks
  mask    - random masking of 25% of dyads (200 repetitions) and cold-start (all relations of one concept)
  strat   - age-sex stratified relation (direct standardisation of within-stratum relations)
  signed  - signed relation as input with signed product composition
  sim     - respondent-level simulation separating chained (structural) from latent-factor generation
  misc    - Table 9 completion, grid error, bootstrap edge filter, sleep 2015-2018, reliability, Q interval,
            age-sex description of the two periods
Writes results2/revision6_<part>.json."""
import json, sys, time, itertools
import numpy as np
import pandas as pd
from analysis2 import graph, dev, hold, psu_boot
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean, adj, best_q, leiden, qf
import rq3x as X
import rq6lib as L

part = sys.argv[1]
rng = np.random.default_rng(20261031)
OFF = L.OFF
OUT = {}


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/revision6_{part}.json", "w"), indent=1)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def rel(df):
    return relation_matrices(df)[2]


Ed, Eh = rel(dev), rel(hold)
pdev = prevalence(dev)
LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]


def predictors(E, prev, heavy=True):
    """All predictors of every ordered pair from relation E (never using the dyad itself)."""
    f = L.path_features(E)
    out = {"product": f["prod_max"], "hamacher": f["ham_max"], "min": f["min_max"],
           "product_len4": L.best_path_product(E, 4),
           "cn_weighted": f["cn_sum"], "adamic_adar": f["adamic_adar"], "resource_alloc": f["resource_alloc"],
           "strength": X.strength_baseline(E)}
    out["reverse_odds"], _ = X.reverse_baseline(E, prev)
    out["prev_block_reg"] = L.prevalence_regression(E, prev)
    if heavy:
        out["katz"] = L.katz(E)
        out["ppr"] = L.ppr(E)
        out["rank3_svd"] = X.lowrank_baseline(E, 3, dyad=True)
        lam, _ = L.choose_lambda(E, LAM_GRID)
        out["softimpute_cv"] = L.softimpute_dyad(E, lam)
        laml, _ = L.choose_lambda(E, LAM_GRID, transform="logit")
        out["softimpute_logit_cv"] = L.softimpute_dyad(E, laml, transform="logit")
        out["nmf3"] = L.nmf_dyad(E, 3)
        # learned aggregator: features from E, trained on E's own pairs (dyad never used in its features)
        out["learned_paths"] = L.learned_aggregator(E, E, L.OFF, E)
    return out


INFO = {"product": "paths", "hamacher": "paths", "min": "paths", "product_len4": "paths",
        "cn_weighted": "paths (sum over intermediates)", "adamic_adar": "paths (degree-weighted)",
        "resource_alloc": "paths (degree-weighted)", "katz": "all walks, dyad removed",
        "ppr": "random walk from source, dyad removed", "learned_paths": "path features, trained within development",
        "strength": "node strengths", "reverse_odds": "reverse edge + prevalences",
        "prev_block_reg": "prevalences + type block", "rank3_svd": "rest of matrix (rank 3)",
        "softimpute_cv": "rest of matrix (soft-impute)", "softimpute_logit_cv": "rest of matrix (soft-impute, logit)",
        "nmf3": "rest of matrix (NMF, rank 3)"}

if part == "main":
    tic("predictors")
    pr = predictors(Ed, pdev)
    res = {k: L.evaluate(v, Eh) for k, v in pr.items()}
    for k in res:
        res[k]["information"] = INFO[k]
    OUT["main"] = res
    tic("concept jackknife")
    jk = {k: [] for k in pr}
    for c in range(P):
        keep = [i for i in range(P) if i != c]
        sub = np.ix_(keep, keep)
        Edc, Ehc = Ed[sub], Eh[sub]
        # predictors on the reduced graph (prevalence vector reduced as well)
        Lc = L
        Pn = P - 1
        offc = ~np.eye(Pn, dtype=bool)
        saveP, saveOFF, saveB, saveTI = Lc.P, Lc.OFF, Lc.BLOCK, Lc.TI
        Lc.P, Lc.OFF, Lc.BLOCK, Lc.TI = Pn, offc, saveB[sub], saveTI[keep]
        try:
            prc = predictors(Edc, pdev[keep])
        finally:
            Lc.P, Lc.OFF, Lc.BLOCK, Lc.TI = saveP, saveOFF, saveB, saveTI
        for k, v in prc.items():
            jk[k].append(L.rho(v[offc], Ehc[offc]))
        tic(f"  jackknife {c}")
    def jse(x):
        x = np.asarray(x); m = len(x)
        return float(np.sqrt((m - 1) / m * ((x - x.mean()) ** 2).sum()))
    OUT["jackknife"] = {k: {"values": v, "se": jse(v),
                            "diff_vs_product_se": jse(np.array(v) - np.array(jk["product"])),
                            "share_above_product": float(np.mean(np.array(v) > np.array(jk["product"])))}
                        for k, v in jk.items()}
    save(); tic("done main")

if part == "blocks":
    out = {}
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        Em = np.where(H, 0.0, Ed)
        tic(f"block {L.BLOCK_NAMES[b]}")
        pr = {}
        f = L.path_features(Em)
        pr["product"], pr["hamacher"] = f["prod_max"], f["ham_max"]
        pr["strength"] = X.strength_baseline(Em)
        pr["rank3_svd"] = X.soft_impute(Ed, OFF & ~H, 3)
        lam, _ = L.choose_lambda(Ed, LAM_GRID, base=OFF & ~H)
        pr["softimpute_cv"] = L.soft_impute_reg(Ed, OFF & ~H, lam)
        pr["prev_type_reg"] = L.prevalence_regression(Ed, pdev, withheld=H)
        pr["katz"] = L.katz(Em, dyad=False)
        pr["learned_paths"] = L.learned_aggregator(Em, Ed, OFF & ~H, Em)
        r = {k: {"rho": L.rho(v[H], Eh[H]), "mae": float(np.abs(v[H] - Eh[H]).mean())} for k, v in pr.items()}
        # calibration fitted on measured pairs only: composed values on Em vs development edges outside H
        tr = OFF & ~H
        for t in ["product", "hamacher"]:
            for kind in ["linear", "power", "isotonic"]:
                cal = L.recalibrate(pr[t][tr], Ed[tr], pr[t][H], kind)
                r[f"{t}_recal_{kind}"] = {"rho": L.rho(cal, Eh[H]), "mae": float(np.abs(cal - Eh[H]).mean())}
        r["n_pairs"] = int(H.sum())
        r["n_ties_product"] = int(H.sum() - len(np.unique(pr["product"][H])))
        out[L.BLOCK_NAMES[b]] = r
    OUT["blocks"] = out
    # pooled over the 10 blocks (each pair predicted once, when its block is withheld)
    save(); tic("done blocks")

if part == "mask":
    keys = ["product", "hamacher", "strength", "rank3_svd", "softimpute_cv", "prev_type_reg", "learned_paths"]
    lam_full, _ = L.choose_lambda(Ed, LAM_GRID)
    def run(H):
        Em = np.where(H, 0.0, Ed)
        f = L.path_features(Em)
        pr = {"product": f["prod_max"], "hamacher": f["ham_max"], "strength": X.strength_baseline(Em),
              "rank3_svd": X.soft_impute(Ed, OFF & ~H, 3), "softimpute_cv": L.soft_impute_reg(Ed, OFF & ~H, lam_full),
              "prev_type_reg": L.prevalence_regression(Ed, pdev, withheld=H),
              "learned_paths": L.learned_aggregator(Em, Ed, OFF & ~H, Em)}
        tr = OFF & ~H
        pr["product_recal_linear"] = np.zeros_like(Ed)
        pr["product_recal_linear"][H] = L.recalibrate(pr["product"][tr], Ed[tr], pr["product"][H], "linear")
        return {k: (L.rho(v[H], Eh[H]), float(np.abs(v[H] - Eh[H]).mean())) for k, v in pr.items()}
    dy = [(i, j) for i in range(P) for j in range(i + 1, P)]
    rnd = []
    tic("random masking 25%, 200 repetitions")
    for rep in range(200):
        sel = rng.choice(len(dy), int(0.25 * len(dy)), replace=False)
        H = np.zeros((P, P), bool)
        for s in sel:
            i, j = dy[s]; H[i, j] = H[j, i] = True
        rnd.append(run(H))
    def summ(rows):
        ks = rows[0].keys()
        return {k: {"rho_mean": float(np.mean([r[k][0] for r in rows])), "rho_sd": float(np.std([r[k][0] for r in rows])),
                    "mae_mean": float(np.mean([r[k][1] for r in rows])),
                    "share_rho_above_product": float(np.mean([r[k][0] > r["product"][0] for r in rows]))} for k in ks}
    OUT["random25"] = summ(rnd)
    tic("cold start")
    cs = []
    for c in range(P):
        H = np.zeros((P, P), bool); H[c, :] = True; H[:, c] = True; np.fill_diagonal(H, False)
        r = run(H); r["concept"] = CONCEPTS[c]; cs.append(r)
    OUT["cold_start"] = summ([{k: v for k, v in r.items() if k != "concept"} for r in cs])
    OUT["cold_start_per_concept"] = [{"concept": r["concept"], **{k: r[k][0] for k in r if k != "concept"}} for r in cs]
    save(); tic("done mask")

if part == "strat":
    demo = pd.concat([pd.read_sas(f"raw/DEMO_{s}.xpt", format="xport").set_index("SEQN")[["RIAGENDR"]] for s in "EFGHIJ"])
    def strata(df):
        d = df.join(demo, how="left")
        age = pd.cut(d.RIDAGEYR, [17, 39, 59, 200], labels=["18-39", "40-59", "60+"])
        return d.assign(stratum=age.astype(str) + "_" + d.RIAGENDR.map({1: "M", 2: "F"}).astype(str))
    sd, sh = strata(dev), strata(hold)
    pooled = pd.concat([sd, sh])
    ref = pooled.groupby("stratum").w.sum() / pooled.w.sum()
    def standardised(d, signed=False):
        Es, prev = np.zeros((P, P)), np.zeros(P)
        for s, pi in ref.items():
            sub = d[d.stratum == s]
            S, S0, E = relation_matrices(sub)
            if signed:
                E = np.where(S0 < 1, (S - S0) / (1 - S0), 0); np.fill_diagonal(E, 0)
            Es += pi * E
            prev += pi * prevalence(sub)
        return Es, prev
    tic("stratified relations")
    Eds, pds = standardised(sd)
    Ehs, _ = standardised(sh)
    OUT["reference_shares"] = ref.to_dict()
    OUT["rho_marginal_vs_stratified_dev"] = L.rho(Ed[OFF], Eds[OFF])
    OUT["median_mu"] = {"marginal": float(np.median(Ed[OFF])), "stratified": float(np.median(Eds[OFF]))}
    pr = predictors(Eds, pds)
    OUT["main"] = {k: L.evaluate(v, Ehs) for k, v in pr.items()}
    # violation of product transitivity in the stratified held-out relation
    C = X.paths(Ehs, X.t_product, max_len=2)
    OUT["product_violation_share_stratified"] = float(((C - Ehs)[OFF] > 0.01).mean())
    C0 = X.paths(Eh, X.t_product, max_len=2)
    OUT["product_violation_share_marginal"] = float(((C0 - Eh)[OFF] > 0.01).mean())
    tic("stratified withheld blocks")
    blk = {}
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        Em = np.where(H, 0.0, Eds)
        f = L.path_features(Em)
        lam, _ = L.choose_lambda(Eds, LAM_GRID, base=OFF & ~H)
        blk[L.BLOCK_NAMES[b]] = {"product": L.rho(f["prod_max"][H], Ehs[H]), "hamacher": L.rho(f["ham_max"][H], Ehs[H]),
                                 "rank3_svd": L.rho(X.soft_impute(Eds, OFF & ~H, 3)[H], Ehs[H]),
                                 "softimpute_cv": L.rho(L.soft_impute_reg(Eds, OFF & ~H, lam)[H], Ehs[H])}
    OUT["blocks"] = blk
    np.save("results2/E_dev_agesex.npy", Eds); np.save("results2/E_hold_agesex.npy", Ehs)
    save(); tic("done strat")

if part == "signed":
    def signed(df):
        S, S0, E = relation_matrices(df)
        Es = np.where(S0 < 1, (S - S0) / (1 - S0), 0); np.fill_diagonal(Es, 0)
        return Es
    Sd, Sh = signed(dev), signed(hold)
    n = P
    # signed product composition: among paths of length 2-3, the path whose product has the largest magnitude
    best = np.zeros((n, n))
    for i, j in itertools.permutations(range(n), 2):
        ks = [k for k in range(n) if k not in (i, j)]
        v2 = Sd[i, ks] * Sd[ks, j]
        cand = list(v2)
        for k in ks:
            for l in ks:
                if l != k:
                    cand.append(Sd[i, k] * Sd[k, l] * Sd[l, j])
        cand = np.array(cand)
        best[i, j] = cand[np.argmax(np.abs(cand))]
    clipped_prod = X.paths(Ed, X.t_product)
    OUT["signed"] = {"n_negative_dev": int((Sd[OFF] < 0).sum()), "n_negative_hold": int((Sh[OFF] < 0).sum()),
                     "signed_input_signed_target": L.rho(best[OFF], Sh[OFF]),
                     "clipped_input_signed_target": L.rho(clipped_prod[OFF], Sh[OFF]),
                     "signed_input_clipped_target": L.rho(best[OFF], Eh[OFF]),
                     "sign_agreement_on_negative_targets": float(np.mean(np.sign(best[OFF][Sh[OFF] < -0.01]) < 0))}
    save(); tic("done signed")

if part == "sim":
    # respondent-level simulation: chained structural model vs latent-factor model, both cut into the
    # same measurement scales as the data (ordinal 0-3 for 9 items, binary or 3-level for the rest)
    nresp = 15000
    kinds = ["ordinal"] * 9 + ["binary"] * 15
    def measure(Z):
        A = np.empty_like(Z)
        for c in range(P):
            if kinds[c] == "ordinal":
                cuts = np.quantile(Z[:, c], [0.70, 0.88, 0.96])
                A[:, c] = np.digitize(Z[:, c], cuts) / 3
            else:
                A[:, c] = (Z[:, c] > np.quantile(Z[:, c], 0.75)).astype(float)
        return A
    def chain_model(rng_):
        # sparse random DAG along a random order; each variable = sum of parents * w + noise
        order = rng_.permutation(P)
        B = np.zeros((P, P))
        for a in range(1, P):
            for b in range(a):
                if rng_.random() < 0.12:
                    B[order[b], order[a]] = rng_.uniform(0.3, 0.7)
        def sample(nn):
            Z = np.zeros((nn, P))
            for c in order:
                Z[:, c] = Z @ B[:, c] + rng_.standard_normal(nn)
            return Z
        return sample
    def factor_model(rng_, k):
        Lm = np.zeros((P, k))
        for c in range(P):
            Lm[c, c % k if k > 1 else 0] = rng_.uniform(0.4, 0.8)
            if k > 1:
                Lm[c] += rng_.uniform(0, 0.25, k)
        def sample(nn):
            F = rng_.standard_normal((nn, k))
            return F @ Lm.T + rng_.standard_normal((nn, P)) * 0.8
        return sample
    def one(sampler):
        dfs = []
        for _ in range(2):
            A = measure(sampler(nresp))
            df = pd.DataFrame(A, columns=CONCEPTS); df["w"] = 1.0
            dfs.append(relation_matrices(df)[2])
        A1, A2 = dfs
        f = L.path_features(A1)
        r3 = X.lowrank_baseline(A1, 3, dyad=True)
        res = {"product": L.rho(f["prod_max"][OFF], A2[OFF]), "hamacher": L.rho(f["ham_max"][OFF], A2[OFF]),
               "min": L.rho(f["min_max"][OFF], A2[OFF]), "rank3": L.rho(r3[OFF], A2[OFF])}
        # partial rank correlation of the product with the target given rank-3 completion
        from scipy.stats import rankdata
        a, b, c = rankdata(f["prod_max"][OFF]), rankdata(A2[OFF]), rankdata(r3[OFF])
        def resid(y, x):
            X_ = np.vstack([np.ones_like(x), x]).T
            return y - X_ @ np.linalg.lstsq(X_, y, rcond=None)[0]
        res["partial_product_given_rank3"] = float(np.corrcoef(resid(a, c), resid(b, c))[0, 1])
        C = X.paths(A2, X.t_product, max_len=2)
        res["product_violation"] = float(((C - A2)[OFF] > 0.01).mean())
        res["density"] = float((A1[OFF] >= 0.01).mean()); res["median_mu"] = float(np.median(A1[OFF][A1[OFF] >= 0.01]))
        return res
    sims = {}
    for name, mk in [("chain", lambda r: chain_model(r)), ("factor1", lambda r: factor_model(r, 1)),
                     ("factor3", lambda r: factor_model(r, 3))]:
        rows = []
        tic(f"sim {name}")
        for rep in range(60):
            rows.append(one(mk(np.random.default_rng(1000 * rep + len(name)))))
        sims[name] = {k: {"mean": float(np.nanmean([r[k] for r in rows])), "sd": float(np.nanstd([r[k] for r in rows]))} for k in rows[0]}
        sims[name]["share_product_best_tnorm"] = float(np.mean([r["product"] >= max(r["hamacher"], r["min"]) for r in rows]))
        sims[name]["share_rank3_above_product"] = float(np.mean([r["rank3"] > r["product"] for r in rows]))
    OUT["sim"] = sims
    # observed diagnostic
    f = L.path_features(Ed); r3 = X.lowrank_baseline(Ed, 3, dyad=True)
    from scipy.stats import rankdata
    a, b, c = rankdata(f["prod_max"][OFF]), rankdata(Eh[OFF]), rankdata(r3[OFF])
    def resid(y, x):
        X_ = np.vstack([np.ones_like(x), x]).T
        return y - X_ @ np.linalg.lstsq(X_, y, rcond=None)[0]
    OUT["observed_partial_product_given_rank3"] = float(np.corrcoef(resid(a, c), resid(b, c))[0, 1])
    save(); tic("done sim")

if part == "misc":
    res = {}
    tic("Table 9: RQ3 and edges for variants")
    t9 = {}
    for alt in ["low_education", "smoking", "poor_self_rated_health", "diabetes", "cancer"]:
        d2, h2 = dev.copy(), hold.copy()
        d2[alt], h2[alt] = d2[alt + "_alt"], h2[alt + "_alt"]
        E2, H2 = rel(d2), rel(h2)
        t9[alt] = {"edges": int((E2[OFF] >= 0.01).sum()), "product": L.rho(X.paths(E2, X.t_product)[OFF], H2[OFF])}
    for c in ["short_sleep", "poor_self_rated_health", "self_harm_thoughts"]:
        k = [i for i, x in enumerate(CONCEPTS) if x != c]
        t9["drop_" + c] = {"edges": int((Ed[np.ix_(k, k)][~np.eye(P - 1, dtype=bool)] >= 0.01).sum())}
    cc_d, cc_h = dev.dropna(subset=CONCEPTS), hold.dropna(subset=CONCEPTS)
    t9["complete_cases"] = {"edges": int((rel(cc_d)[OFF] >= 0.01).sum()), "n": [len(cc_d), len(cc_h)]}
    res["table9"] = t9
    tic("grid error of S0")
    import graph2
    g0 = graph2.GRID
    graph2.GRID = np.linspace(0, 1, 2001)
    Efine = relation_matrices(dev)[2]
    graph2.GRID = g0
    res["grid_error_max_abs_mu"] = float(np.abs(Efine - Ed)[OFF].max())
    res["grid_error_mean_abs_mu"] = float(np.abs(Efine - Ed)[OFF].mean())
    tic("bootstrap edge filter (200 Rao-Wu replicates of development)")
    B = np.stack([rel(psu_boot(dev, rng)) for _ in range(200)])
    lo = np.quantile(B, 0.025, axis=0)
    keep = lo > 0
    Ef = np.where(keep, Ed, 0.0)
    res["edge_filter"] = {"n_edges_lower_bound_positive": int(keep[OFF].sum()),
                          "product": L.rho(X.paths(Ef, X.t_product)[OFF], Eh[OFF]),
                          "hamacher": L.rho(X.paths(Ef, X.T_FIXED["hamacher"])[OFF], Eh[OFF]),
                          "min": L.rho(X.paths(Ef, X.t_min)[OFF], Eh[OFF])}
    tic("sleep: development restricted to 2015-2018")
    d15 = dev[dev.cycle.isin(["I", "J"])]
    E15 = rel(d15)
    res["dev_2015_2018"] = {"n": len(d15), **{t: L.rho(X.paths(E15, X.T_FIXED[t])[OFF], Eh[OFF]) for t in ["product", "hamacher", "min"]},
                            "rank3": L.rho(X.lowrank_baseline(E15, 3)[OFF], Eh[OFF])}
    tic("split-half reliability of each period's relation (20 splits by sampling unit)")
    def split_rel(df):
        rs = []
        for s in range(20):
            r_ = np.random.default_rng(500 + s)
            key = df.strata.astype(str) + "_" + df.psu.astype(str)
            units = key.unique()
            a = set(r_.choice(units, len(units) // 2, replace=False))
            A, Bb = df[key.isin(a)], df[~key.isin(a)]
            rs.append(L.rho(rel(A)[OFF], rel(Bb)[OFF]))
        r = float(np.mean(rs))
        return {"split_half_rho": r, "spearman_brown": 2 * r / (1 + r)}
    res["reliability"] = {"development": split_rel(dev), "heldout": split_rel(hold)}
    tic("Q interval (Rao-Wu, best of 10 restarts)")
    from analysis2 import graph as gfun
    qs = []
    for _ in range(200):
        Eb, _ = gfun(psu_boot(dev, rng))
        qs.append(best_q(adj(Eb), range(10))[0])
    res["Q_dev_boot"] = {"point": 0.215, "interval": [float(np.quantile(qs, .025)), float(np.quantile(qs, .975))]}
    tic("age and sex by period")
    demo = pd.concat([pd.read_sas(f"raw/DEMO_{s}.xpt", format="xport").set_index("SEQN")[["RIAGENDR"]] for s in "EFGHIJ"])
    desc = {}
    for nm, df in [("development", dev), ("heldout", hold)]:
        d = df.join(demo)
        w = d.w
        desc[nm] = {"mean_age": wmean(d.RIDAGEYR, w), "female": wmean((d.RIAGENDR == 2).astype(float), w),
                    "age_18_39": wmean((d.RIDAGEYR <= 39).astype(float), w), "age_40_59": wmean(((d.RIDAGEYR >= 40) & (d.RIDAGEYR <= 59)).astype(float), w),
                    "age_60_plus": wmean((d.RIDAGEYR >= 60).astype(float), w)}
    res["age_sex"] = desc
    OUT["misc"] = res
    save(); tic("done misc")
