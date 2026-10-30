"""Analyses requested in the second-round review (R1, R3, R5).
Parts: blocks (withheld blocks: k chosen by cross-validation within development, bootstrap intervals, sign test),
sim2 (simulation matched to the data's noise and scale; confusion matrices), reverse (zero reverse edges).
Writes results2/revision3_<part>.json."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from analysis2 import graph, dev, hold, psu_boot, OFF
from graph2 import CONCEPTS, TYPE, P, wmean
import rq3x as X

part = sys.argv[1]
rng = np.random.default_rng(20261031)
OUT = {}
T5 = ["product", "hamacher", "yager2", "lukasiewicz", "min"]
TYP = np.array([TYPE[c] for c in CONCEPTS])
TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
BLOCKS = list(itertools.combinations_with_replacement(TYPES, 2))
KGRID = [1, 2, 3, 4, 5]


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/revision3_{part}.json", "w"), indent=1)


def block_mask(a, b):
    H = np.zeros((P, P), bool)
    for i, j in itertools.permutations(range(P), 2):
        if {TYP[i], TYP[j]} == {a, b} and (a != b or TYP[i] == a):
            H[i, j] = True
    return H


MASKS = {f"{a}<->{b}": block_mask(a, b) for a, b in BLOCKS}


def choose_k(Ed, target):
    """Choose the completion rank by withholding each other block in turn within development."""
    H = MASKS[target]
    score = {k: [] for k in KGRID}
    for other, H2 in MASKS.items():
        if other == target:
            continue
        mask = OFF & ~H & ~H2
        for k in KGRID:
            L = X.soft_impute(Ed, mask, k)
            score[k].append(X.rho(L[H2], Ed[H2]))
    mean = {k: float(np.nanmean(v)) for k, v in score.items()}
    return max(mean, key=mean.get), mean


def block_predictions(Ed, target, k):
    H = MASKS[target]
    Em = np.where(H, 0.0, Ed)
    out = {T: X.paths(Em, X.T_FIXED[T])[H] for T in ["product", "hamacher", "min"]}
    out["strength"] = X.strength_baseline(Em)[H]
    out["rank3"] = X.soft_impute(Ed, OFF & ~H, 3)[H]
    out["rank_cv"] = X.soft_impute(Ed, OFF & ~H, k)[H]
    return out


_, Edev = graph(dev)
_, Ehold = graph(hold)

if part == "blocks":
    tic("k by cross-validation within development")
    KSEL = {}
    for t in MASKS:
        k, mean = choose_k(Edev, t)
        KSEL[t] = {"k": k, "cv_mean_rho": mean}
    point = {}
    for t, H in MASKS.items():
        pr = block_predictions(Edev, t, KSEL[t]["k"])
        point[t] = {m: X.rho(v, Ehold[H]) for m, v in pr.items()}
        point[t]["n_pairs"] = int(H.sum())
    tic("bootstrap (200), both periods rebuilt, k fixed at the development choice")
    draws = {t: {m: [] for m in ["product", "hamacher", "min", "strength", "rank3", "rank_cv"]} for t in MASKS}
    for b in range(200):
        if b % 20 == 0:
            tic(f"replicate {b}")
        _, Edb = graph(psu_boot(dev, rng))
        _, Ehb = graph(psu_boot(hold, rng))
        for t, H in MASKS.items():
            pr = block_predictions(Edb, t, KSEL[t]["k"])
            for m, v in pr.items():
                draws[t][m].append(X.rho(v, Ehb[H]))
    res = {}
    for t in MASKS:
        D = {m: np.array(v, float) for m, v in draws[t].items()}
        def sh(m):
            d = D[m][np.isfinite(D[m])]; bias = d.mean() - point[t][m]
            return [float(np.quantile(d, .025) - bias), float(np.quantile(d, .975) - bias)]
        res[t] = {"n_pairs": point[t]["n_pairs"], "k_cv": KSEL[t]["k"],
                  "rho": {m: point[t][m] for m in D}, "ci_shifted": {m: sh(m) for m in D},
                  "diff_product_minus_rank3": {"point": point[t]["product"] - point[t]["rank3"],
                                               "ci": X.bc_ci(D["product"] - D["rank3"], point[t]["product"] - point[t]["rank3"])},
                  "diff_product_minus_rank_cv": {"point": point[t]["product"] - point[t]["rank_cv"],
                                                 "ci": X.bc_ci(D["product"] - D["rank_cv"], point[t]["product"] - point[t]["rank_cv"])}}
    OUT["blocks"] = res
    wins3 = sum(res[t]["rho"]["product"] > res[t]["rho"]["rank3"] for t in res)
    winscv = sum(res[t]["rho"]["product"] > res[t]["rho"]["rank_cv"] for t in res)
    sig3 = sum(res[t]["diff_product_minus_rank3"]["ci"][0] > 0 for t in res)
    sig3n = sum(res[t]["diff_product_minus_rank3"]["ci"][1] < 0 for t in res)
    sigcv = sum(res[t]["diff_product_minus_rank_cv"]["ci"][0] > 0 for t in res)
    sigcvn = sum(res[t]["diff_product_minus_rank_cv"]["ci"][1] < 0 for t in res)
    OUT["summary"] = {"n_blocks": len(res), "product_beats_rank3": int(wins3), "sign_test_p_rank3": float(binomtest(wins3, len(res)).pvalue),
                      "product_beats_rank_cv": int(winscv), "sign_test_p_rank_cv": float(binomtest(winscv, len(res)).pvalue),
                      "product_ci_above_rank3": int(sig3), "product_ci_below_rank3": int(sig3n),
                      "product_ci_above_rank_cv": int(sigcv), "product_ci_below_rank_cv": int(sigcvn),
                      "k_cv_counts": {str(k): sum(KSEL[t]["k"] == k for t in KSEL) for k in KGRID},
                      "median": {m: float(np.median([res[t]["rho"][m] for t in res])) for m in ["product", "hamacher", "min", "strength", "rank3", "rank_cv"]}}
    save(); tic("done blocks")

if part == "sim2":
    obs_med = float(np.median(Edev[OFF & (Edev >= 0.01)]))
    obs_rho = X.rho(Edev[OFF], Ehold[OFF])

    def closure(Bw, T, iters=30):
        R = Bw.copy()
        for _ in range(iters):
            C = X.paths(R, T, max_len=2)
            Rn = np.maximum(R, C)
            if np.allclose(Rn, R):
                break
            R = Rn
        np.fill_diagonal(R, 0)
        return R

    def truth(g, scale, rng_):
        n = P
        if g.endswith("closure"):
            T = {"min_closure": X.t_min, "product_closure": X.t_product, "hamacher_closure": X.t_hamacher_lam(0.0)}[g]
            Bw = np.where(rng_.random((n, n)) < 0.12, scale * rng_.beta(2, 5, (n, n)), 0.0); np.fill_diagonal(Bw, 0)
            return closure(Bw, T)
        lam = np.sqrt(scale) * rng_.beta(2, 4, n)
        Rt = np.outer(lam, lam)
        if g == "rank3_blocks":
            grp = rng_.integers(0, 3, n); Rt = Rt * np.where(grp[:, None] == grp[None, :], 1.0, 0.25)
        np.fill_diagonal(Rt, 0)
        return Rt

    def noisy(R, s, rng_):
        Z = np.clip(R * np.exp(rng_.normal(0, s, R.shape)), 0, 1); Z[Z < 0.01] = 0; np.fill_diagonal(Z, 0)
        return Z

    gens = ["min_closure", "product_closure", "hamacher_closure", "rank1_factor", "rank3_blocks"]
    settings = {}
    for g in gens:
        tic(f"calibrate {g}")
        # scale so that the median retained relation is near the observed median
        best = None
        for scale in [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]:
            meds = []
            for _ in range(20):
                Rt = truth(g, scale, rng); v = Rt[OFF & (Rt >= 0.01)]
                meds.append(np.median(v) if len(v) else 0)
            d = abs(np.mean(meds) - obs_med)
            if best is None or d < best[0]:
                best = (d, scale, float(np.mean(meds)))
        scale = best[1]
        bests = None
        for s in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6]:
            rr = []
            for _ in range(20):
                Rt = truth(g, scale, rng); rr.append(X.rho(noisy(Rt, s, rng)[OFF], noisy(Rt, s, rng)[OFF]))
            d = abs(np.nanmean(rr) - obs_rho)
            if bests is None or d < bests[0]:
                bests = (d, s, float(np.nanmean(rr)))
        settings[g] = {"scale": scale, "median_mu": best[2], "noise_sd": bests[1], "dev_hold_rho": bests[2]}
    res = {}
    for g in gens:
        tic(f"simulate {g} (matched)")
        st = settings[g]; rows = []
        for _ in range(200):
            Rt = truth(g, st["scale"], rng)
            Rd, Rh = noisy(Rt, st["noise_sd"], rng), noisy(Rt, st["noise_sd"], rng)
            r = {T: X.rho(X.paths(Rd, X.T_FIXED[T])[OFF], Rh[OFF]) for T in T5}
            rows.append(r)
        Rr = pd.DataFrame(rows).fillna(-1)
        res[g] = {"share_first": {T: float((Rr[T5].idxmax(1) == T).mean()) for T in T5},
                  "mean_rho": {T: float(Rr[T].replace(-1, np.nan).mean()) for T in T5}}
    OUT["observed"] = {"median_mu": obs_med, "dev_hold_rho": obs_rho}
    OUT["settings"] = settings
    OUT["matched"] = res
    save(); tic("done sim2")

if part == "reverse":
    prev = np.array([wmean(dev.loc[dev[c].notna(), c], dev.loc[dev[c].notna(), "w"]) for c in CONCEPTS])
    R, _ = X.reverse_baseline(Edev, prev)
    pos = OFF & (Edev.T > 0)
    Pr = X.paths(Edev, X.t_product)
    OUT["reverse"] = {"n_pairs": int(OFF.sum()), "n_reverse_zero": int((OFF & (Edev.T == 0)).sum()),
                      "rho_all": X.rho(R[OFF], Ehold[OFF]), "rho_positive_reverse_only": X.rho(R[pos], Ehold[pos]),
                      "product_rho_positive_reverse_only": X.rho(Pr[pos], Ehold[pos]),
                      "heldout_mean_when_reverse_zero": float(Ehold[OFF & (Edev.T == 0)].mean()),
                      "heldout_mean_when_reverse_positive": float(Ehold[pos].mean())}
    save(); tic("done reverse")
