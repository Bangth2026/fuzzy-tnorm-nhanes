"""Convergence of path-consistent completion (PCC) at the selected setting, PCC without the nuclear norm, the
two-edge product path, the sample size an equivalence test of the type-block difference would need, and mean
activations by cycle (v2.2.0).

Parts (results2/pcclim_<part>.json):
  conv     the selected setting (lambda = 0.003, gamma = 0.5; per-block settings chosen without the evaluated block)
           run to 3,000 iterations in the single-dyad and type-block designs; max |dX| at the iteration cap used in
           the paper (300 for single dyads, 400 for type blocks) and at the end; rho against 2007-2012 and 2021-2023 at
           both lengths; the largest change in a prediction; the pre-specified quantities P1-P3 recomputed with the
           longer runs (point estimates)
  lam0     PCC with lambda = 0 and gamma = 0.5 (no nuclear norm) and the product path restricted to two edges, against
           the product path over two or three edges (main Eq. (5)), single dyads and type blocks, 2007-2012 and
           2021-2023 targets; JRR intervals for the two-edge minus two-or-three-edge product path
  tostn    type blocks, 2021-2023 target: rank blend minus product, its JRR standard error split into the part from
           the development replicates and the part from the target replicates, and the target sample, as a multiple
           of 2021-2023, at which a two one-sided test with margin 0.03 would have power 0.8
  means    survey-weighted mean activation of the 24 concepts in 2013-2018, 2007-2012 and 2021-2023
  s1       single dyads: rank-3 completion and the product path by pair group (symptom-symptom, symptom-correlate,
           correlate-correlate), 2007-2012 and 2021-2023 targets (exploratory; where the S1 difference grew)
"""
import json
import sys
import time
import os
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr, rankdata, norm
from analysis2 import dev, hold
from graph2 import relation_matrices, P, CONCEPTS, wmean
import rq3x as X
import rq6lib as L
import tcomp as TC
from nhanes_l import load_c21

part = sys.argv[1] if len(sys.argv) > 1 else "none"
OFF = ~np.eye(P, dtype=bool)
NB = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
DYADS = [(i, j) for i in range(P) for j in range(i + 1, P)]
NCPU = int(os.environ.get("NCPU", "2"))
LAM_SI = [0.01, 0.03, 0.1, 0.3, 0.6]
OUT = {}


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/pcclim_{part}.json", "w"), indent=1, default=float)


def rel(df):
    return relation_matrices(df)[2]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def rk(v):
    return rankdata(v) / len(v)


def pmap(f, args):
    with Pool(NCPU) as pool:
        return pool.starmap(f, args)


def wblock(vals):
    return float((np.asarray(vals) * NB).sum() / NB.sum())


def setting():
    d = json.load(open("results2/pcc2_dev.json"))["dev"]
    ch = d["choices"]["two"]
    return {"pcc": tuple(map(float, ch["pcc"].split("|"))), "w_rank": ch["blend_rank"],
            "block": [tuple(map(float, x["pcc"].split("|"))) for x in d["block_choices_two"]],
            "block_w_rank": [x["blend_rank"] for x in d["block_choices_two"]]}


def pcc_long(R, mask, lam, mu, snap, iters, tol=1e-6):
    """tcomp.tc_complete (two_sided=True, cells='missing', product) with the iterate kept at iteration `snap`.

    Returns X at the snapshot (what the paper's capped run returns), the max |dX| of the step that produced it,
    the final X, the number of iterations run and the last max |dX|."""
    off = OFF
    T, dTa, dTb = TC._tnorm_and_grads("product")
    m = R[mask].mean()
    X_ = np.where(mask, R, m)
    step = 1.0 / (1.0 + 3.0 * mu)
    snapX, snapd, d = None, None, None
    it = 0
    for it in range(1, iters + 1):
        G = np.where(mask, X_ - R, 0.0)
        if mu > 0:
            b, k = TC.best_path(X_, "product")
            phi = np.where(off & ~mask, b - X_, 0.0)
            if phi.any():
                ii, jj = np.nonzero(phi)
                kk = k[ii, jj]
                a, c = X_[ii, kk], X_[kk, jj]
                w = mu * phi[ii, jj]
                G[ii, jj] -= w
                np.add.at(G, (ii, kk), w * dTa(a, c))
                np.add.at(G, (kk, jj), w * dTb(a, c))
        U, s, Vt = np.linalg.svd(X_ - step * G - m, full_matrices=False)
        s = np.maximum(s - step * lam, 0.0)
        Xn = np.clip((U * s) @ Vt + m, 0.0, 1.0)
        d = float(np.abs(Xn - X_).max())
        X_ = Xn
        if it == snap:
            snapX, snapd = X_.copy(), d
        if d < tol:
            break
    if snapX is None:          # converged before the snapshot: the capped run returns the same matrix
        snapX, snapd = X_.copy(), d
    np.fill_diagonal(snapX, 0.0)
    Xf = X_.copy(); np.fill_diagonal(Xf, 0.0)
    return snapX, snapd, Xf, it, d


Ed, Eh = rel(dev), rel(hold)


# ================================================================= convergence
def _conv_dyad(i, j, lam, mu, iters):
    m = OFF.copy(); m[i, j] = m[j, i] = False
    Xs, ds, Xf, it, df = pcc_long(Ed, m, lam, mu, 300, iters)
    return [Xs[i, j], Xs[j, i]], [Xf[i, j], Xf[j, i]], ds, it, df


def _conv_block(b, lam, mu, iters):
    H = OFF & (L.BLOCK == b)
    Xs, ds, Xf, it, df = pcc_long(Ed, OFF & ~H, lam, mu, 400, iters)
    return Xs[H], Xf[H], ds, it, df


if part == "conv":
    S = setting()
    c21, _ = load_c21()
    E21 = rel(c21)
    ITERS = int(os.environ.get("ITERS", "3000"))
    lam, mu = S["pcc"]
    # check that the snapshot reproduces the capped run of the paper
    m0 = OFF.copy(); m0[0, 1] = m0[1, 0] = False
    ref = TC.tc_complete(Ed, m0, lam, mu, "product", 300, two_sided=True, cells="missing")
    chk = pcc_long(Ed, m0, lam, mu, 300, 300)[0]
    o = {"setting": [lam, mu], "iters_long": ITERS, "snapshot_equals_capped_run": float(np.abs(ref - chk).max())}
    tic("dyads")
    rows = pmap(_conv_dyad, [(i, j, lam, mu, ITERS) for i, j in DYADS])
    cap = np.zeros((P, P)); lon = np.zeros((P, P))
    for (i, j), (a, b_, _, _, _) in zip(DYADS, rows):
        cap[i, j], cap[j, i] = a; lon[i, j], lon[j, i] = b_
    dcap = np.array([x[2] for x in rows]); its = np.array([x[3] for x in rows]); dend = np.array([x[4] for x in rows])
    o["dyad"] = {"maxdX_at_cap_median": float(np.median(dcap)), "maxdX_at_cap_max": float(dcap.max()),
                 "maxdX_at_end_median": float(np.median(dend)), "maxdX_at_end_max": float(dend.max()),
                 "iters_median": float(np.median(its)), "share_reached_tol": float((dend < 1e-6).mean()),
                 "max_abs_pred_change": float(np.abs(lon - cap)[OFF].max()),
                 "median_abs_pred_change": float(np.median(np.abs(lon - cap)[OFF])),
                 "rho_2007_cap": r(cap[OFF], Eh[OFF]), "rho_2007_long": r(lon[OFF], Eh[OFF]),
                 "rho_2021_cap": r(cap[OFF], E21[OFF]), "rho_2021_long": r(lon[OFF], E21[OFF])}
    np.savez("results2/pcclim_conv_dyad_predictions.npz", cap=cap, long=lon)
    OUT["conv"] = o; save()
    tic(json.dumps(o["dyad"]))
    tic("blocks")
    rows = pmap(_conv_block, [(b, *S["block"][b], ITERS) for b in range(10)])
    capb = np.zeros((P, P)); lonb = np.zeros((P, P))
    for b, (a, b_, _, _, _) in enumerate(rows):
        H = OFF & (L.BLOCK == b)
        capb[H], lonb[H] = a, b_
    vals = {}
    for nm, M in [("cap", capb), ("long", lonb)]:
        for tn, Et in [("2007", Eh), ("2021", E21)]:
            vals[f"rho_{tn}_{nm}"] = wblock([r(M[OFF & (L.BLOCK == b)], Et[OFF & (L.BLOCK == b)]) for b in range(10)])
    o["block"] = {"maxdX_at_cap": [float(x[2]) for x in rows], "iters": [int(x[3]) for x in rows],
                  "maxdX_at_end": [float(x[4]) for x in rows], "max_abs_pred_change": float(np.abs(lonb - capb)[OFF].max()),
                  **vals}
    np.savez("results2/pcclim_conv_block_predictions.npz", cap=capb, long=lonb)
    # P1-P3 with the longer runs (point estimates; the other methods as in results2/pcc2_c21.json and pcc2_jrr.json)
    for tn, fn, nm in [("2021", "results2/pcc2_c21.json", "c21"), ("2007", "results2/pcc2_jrr.json", "jrr")]:
        F = json.load(open(fn))[nm]["full"]
        six = ["pcc", "product", "katz", "si_logit", "rank3", "blend_rank"]
        res = {}
        for lab in ["cap", "long"]:
            f = dict(F)
            f["dyad_pcc"] = o["dyad"][f"rho_{tn}_{lab}"]; f["block_pcc"] = o["block"][f"rho_{tn}_{lab}"]
            bd = max(f[f"dyad_{k}"] for k in six); bb = max(f[f"block_{k}"] for k in six)
            g = {k: max(bd - f[f"dyad_{k}"], bb - f[f"block_{k}"]) for k in six}
            res[lab] = {"g": g, "P1": g["product"] - g["blend_rank"], "P2": g["pcc"] - g["blend_rank"],
                        "P3": g["product"] - g["pcc"], "dyad_pcc": f["dyad_pcc"], "block_pcc": f["block_pcc"]}
        res["reported_P"] = [F["regretsix_product"] - F["regretsix_blend_rank"],
                             F["regretsix_pcc"] - F["regretsix_blend_rank"], F["regretsix_product"] - F["regretsix_pcc"]]
        o[f"P_{tn}"] = res
    OUT["conv"] = o; save()
    tic(json.dumps({k: o[k] for k in ["block", "P_2021", "P_2007"]}))


# ================================================================= lambda = 0 and the two-edge product path
def _lam0_dyad(i, j, mu, iters):
    m = OFF.copy(); m[i, j] = m[j, i] = False
    Xs, _, Xf, it, d = pcc_long(Ed, m, 0.0, mu, 300, iters)
    return [Xs[i, j], Xs[j, i]], [Xf[i, j], Xf[j, i]], it, d


def _lam0_block(b, mu, iters):
    H = OFF & (L.BLOCK == b)
    Xs, _, Xf, it, d = pcc_long(Ed, OFF & ~H, 0.0, mu, 400, iters)
    return Xs[H], Xf[H], it, d


def path2(E1):
    return TC.best_path(E1, "product")[0]


def path23(E1):
    return L.path_features(E1)["prod_max"]


def paths_stats(E1, E2):
    """Two-edge and two-or-three-edge product paths, single dyads and type blocks."""
    o = {"dyad_path2": r(path2(E1)[OFF], E2[OFF]), "dyad_path23": r(path23(E1)[OFF], E2[OFF])}
    v2, v23 = [], []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        Em = np.where(OFF & ~H, E1, 0.0)
        v2.append(r(path2(Em)[H], E2[H])); v23.append(r(X.paths(Em, X.t_product)[H], E2[H]))
    o["block_path2"], o["block_path23"] = wblock(v2), wblock(v23)
    o["dyad_path2_minus_path23"] = o["dyad_path2"] - o["dyad_path23"]
    o["block_path2_minus_path23"] = o["block_path2"] - o["block_path23"]
    return o


def jrr_reps(df):
    reps = []
    for h, g in df.groupby("strata"):
        ps = sorted(g.psu.unique())
        if len(ps) < 2:
            continue
        d = df.copy()
        d.loc[d.psu == ps[0], "w"] = 0.0
        d.loc[d.psu == ps[1], "w"] *= 2.0
        reps.append(d[d.w > 0])
    return reps


def jrr_summary(full, reps):
    out = {}
    for k, v0 in full.items():
        v = np.array([x[k] for x in reps], float)
        se = float(np.sqrt(((v - v0) ** 2).sum()))
        out[k] = {"point": float(v0), "se": se, "ci95": [v0 - 1.96 * se, v0 + 1.96 * se],
                  "tost_p_0.03": float(max(norm.cdf(-(v0 + 0.03) / se), norm.cdf(-(0.03 - v0) / se))) if se > 0 else None}
    return out


def _ps_dev(d, E2):
    return paths_stats(rel(d), E2)


def _ps_tgt(h, E1):
    return paths_stats(E1, rel(h))


if part == "lam0":
    S = setting()
    c21, _ = load_c21()
    E21 = rel(c21)
    mu = S["pcc"][1]
    ITERS = int(os.environ.get("ITERS", "3000"))
    o = {"gamma": mu}
    rows = pmap(_lam0_dyad, [(i, j, mu, ITERS) for i, j in DYADS])
    cap = np.zeros((P, P)); lon = np.zeros((P, P))
    for (i, j), (a, b_, _, _) in zip(DYADS, rows):
        cap[i, j], cap[j, i] = a; lon[i, j], lon[j, i] = b_
    P2 = path2(Ed)
    o["dyad"] = {"pcc_lam0_cap_2007": r(cap[OFF], Eh[OFF]), "pcc_lam0_long_2007": r(lon[OFF], Eh[OFF]),
                 "pcc_lam0_cap_2021": r(cap[OFF], E21[OFF]), "pcc_lam0_long_2021": r(lon[OFF], E21[OFF]),
                 "rho_pcc_lam0_vs_path2": r(lon[OFF], P2[OFF]),
                 "max_abs_pcc_lam0_minus_path2": float(np.abs(lon - P2)[OFF].max()),
                 "iters_median": float(np.median([x[2] for x in rows])),
                 "share_reached_tol": float(np.mean([x[3] < 1e-6 for x in rows]))}
    sel = np.load("results2/pcclim_conv_dyad_predictions.npz")
    o["dyad"]["rho_pcc_selected_vs_path2"] = r(sel["cap"][OFF], P2[OFF])
    o["dyad"]["rho_pcc_selected_vs_path23"] = r(sel["cap"][OFF], path23(Ed)[OFF])
    OUT["lam0"] = o; save()
    tic(json.dumps(o["dyad"]))
    rows = pmap(_lam0_block, [(b, mu, ITERS) for b in range(10)])
    vals = {"cap": [], "long": []}
    v21 = {"cap": [], "long": []}
    for b, (a, b_, _, _) in enumerate(rows):
        H = OFF & (L.BLOCK == b)
        vals["cap"].append(r(a, Eh[H])); vals["long"].append(r(b_, Eh[H]))
        v21["cap"].append(r(a, E21[H])); v21["long"].append(r(b_, E21[H]))
    o["block"] = {"pcc_lam0_cap_2007": wblock(vals["cap"]), "pcc_lam0_long_2007": wblock(vals["long"]),
                  "pcc_lam0_cap_2021": wblock(v21["cap"]), "pcc_lam0_long_2021": wblock(v21["long"])}
    OUT["lam0"] = o; save()
    for tn, d2, E2 in [("2007", hold, Eh), ("2021", c21, E21)]:
        full = paths_stats(Ed, E2)
        reps = pmap(_ps_dev, [(d, E2) for d in jrr_reps(dev)]) + pmap(_ps_tgt, [(h, Ed) for h in jrr_reps(d2)])
        o[f"paths_{tn}"] = jrr_summary(full, reps)
        o[f"paths_{tn}_n_replicates"] = len(reps)
        OUT["lam0"] = o; save()
        tic(tn + " " + json.dumps({k: round(v["point"], 4) for k, v in o[f"paths_{tn}"].items()}))


# ================================================================= sample size for an equivalence test (type blocks)
def block_blend_minus_product(E1, E2, S):
    vb, vp = [], []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        obs = OFF & ~H
        Em = np.where(obs, E1, 0.0)
        pr = X.paths(Em, X.t_product)
        ll = L.choose_lambda(E1, LAM_SI, base=obs, transform="logit")[0]
        si = L.soft_impute_reg(E1, obs, ll, transform="logit")
        w = S["block_w_rank"][b]
        vb.append(r(w * rk(pr[H]) + (1 - w) * rk(si[H]), E2[H])); vp.append(r(pr[H], E2[H]))
    return wblock(vb) - wblock(vp)


def _tn_dev(d, E2, S):
    return block_blend_minus_product(rel(d), E2, S)


def _tn_tgt(h, E1, S):
    return block_blend_minus_product(E1, rel(h), S)


if part == "tostn":
    S = setting()
    c21, _ = load_c21()
    E21 = rel(c21)
    d0 = block_blend_minus_product(Ed, E21, S)
    rd = np.array(pmap(_tn_dev, [(d, E21, S) for d in jrr_reps(dev)]))
    rt = np.array(pmap(_tn_tgt, [(h, Ed, S) for h in jrr_reps(c21)]))
    v_dev, v_tgt = float(((rd - d0) ** 2).sum()), float(((rt - d0) ** 2).sum())
    se = np.sqrt(v_dev + v_tgt)
    delta, za = 0.03, norm.ppf(0.95)
    o = {"point": d0, "se": float(se), "se_dev_part": float(np.sqrt(v_dev)), "se_target_part": float(np.sqrt(v_tgt)),
         "n_dev_reps": len(rd), "n_target_reps": len(rt)}

    def power(se_, true):
        return float(max(0.0, norm.cdf((delta - true) / se_ - za) - norm.cdf((-delta - true) / se_ + za)))
    o["power_now_true0"] = power(se, 0.0); o["power_now_true_obs"] = power(se, d0)
    for true, lab in [(0.0, "true0"), (d0, "true_obs")]:
        # target sample k times 2021-2023 (target variance / k); development unchanged; also both enlarged by k
        res = {}
        for mode in ["target_only", "both"]:
            k_ = None
            for k in np.arange(1.0, 200.01, 0.05):
                s_ = np.sqrt(v_dev + v_tgt / k) if mode == "target_only" else se / np.sqrt(k)
                if power(s_, true) >= 0.8:
                    k_ = float(round(k, 2)); break
            res[mode] = k_
        o[f"multiple_for_power08_{lab}"] = res
    o["required_se_true0"] = float(delta / (za + norm.ppf(0.9)))
    OUT["tostn"] = o; save()
    tic(json.dumps(o))


# ================================================================= mean activations
if part == "means":
    c21, _ = load_c21()
    o = {}
    for nm, df in [("2013-2018", dev), ("2007-2012", hold), ("2021-2023", c21)]:
        o[nm] = {c: float(wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"])) for c in CONCEPTS}
    o["shift_2021_minus_2013"] = {c: o["2021-2023"][c] - o["2013-2018"][c] for c in CONCEPTS}
    OUT["means"] = o; save()
    for c in CONCEPTS:
        print(f"{c:22s} {o['2013-2018'][c]:.3f} {o['2007-2012'][c]:.3f} {o['2021-2023'][c]:.3f}")


# ================================================================= where the S1 difference grew in 2021-2023
if part == "s1":
    from graph2 import TYPE
    c21, _ = load_c21()
    E21 = rel(c21)
    fp = np.load("results2/pcc2_dyad_fixed_predictions.npz")
    sym = np.array([TYPE[c] == "Symptom" for c in CONCEPTS])
    groups = {"symptom_symptom": np.outer(sym, sym) & OFF,
              "symptom_correlate": (np.outer(sym, ~sym) | np.outer(~sym, sym)) & OFF,
              "correlate_correlate": np.outer(~sym, ~sym) & OFF, "all": OFF}
    o = {}
    for g, M in groups.items():
        row = {"n_pairs": int(M.sum()), "mean_mu_2013": float(Ed[M].mean())}
        for tn, Et in [("2007", Eh), ("2021", E21)]:
            row[f"mean_mu_{tn}"] = float(Et[M].mean())
            row[f"rho_dev_vs_{tn}"] = r(Ed[M], Et[M])
            for k in ["product", "rank3"]:
                row[f"{k}_{tn}"] = r(fp[k][M], Et[M])
            row[f"rank3_minus_product_{tn}"] = row[f"rank3_{tn}"] - row[f"product_{tn}"]
        o[g] = row
    OUT["s1"] = o; save()
    print(json.dumps(o, indent=1))
