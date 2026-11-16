"""Checks added in response to the third review round (v1.9.0).
Parts:
  power     - probability that each registered decision rule is met when the true difference is 0, 0.03 or 0.05,
              from the JRR standard errors of the 2005-2006 test (normal approximation)
  blocks    - whole relation types withheld: soft-impute on the raw scale, on the logit scale, and with the scale
              chosen by dyad cross-validation on the other blocks; JRR intervals (cross-period design)
  blocks05  - the same comparison in the registered 2005-2006 design (secondary; the registered rule H4 used the
              raw scale, as run_2005.py did)
  recalib   - every t-norm composed on the symmetric relation, its output mapped monotonically onto the scale of s
              (power map fitted leave-dyad-out on development, or leave-dyad-out isotonic regression), then
              re-directed by sqrt(odds_j / odds_i); rho over 552 ordered pairs with JRR intervals
  grid      - soft-impute (logit) with the penalty grid extended to {..., 0.6, 1, 2, 5}: chosen once and inside
              every one of the 276 single-dyad masks; and in the block design
  coldstart - diagnostics for soft-impute with one anchor relation (m = 1)
  randthin  - random thinning (50%, 25% of edges, 20 repetitions) and simulated sources with soft-impute on both
              scales; means and standard deviations over repetitions
  crisp     - which crisp-crisp dyads have both directions >= 0.01
Writes results2/rev3_<part>.json."""
import json, sys, time, os
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr, norm
from sklearn.isotonic import IsotonicRegression
from analysis2 import dev, hold
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
import rq3x as X
import rq6lib as L

part = sys.argv[1]
OFF = ~np.eye(P, dtype=bool)
UP = np.triu(OFF, 1)
OUT = {}
LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]
LAM_EXT = LAM_GRID + [1.0, 2.0, 5.0]
TNORMS = ["product", "hamacher", "min", "yager2", "lukasiewicz"]
NB = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/rev3_{part}.json", "w"), indent=1, default=float)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def rel(df):
    return relation_matrices(df)[2]


def r(a, b):
    return float(spearmanr(a, b).statistic)


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
    for k in full:
        if not isinstance(full[k], (float, int, np.floating)):
            continue
        v = np.array([x[k] for x in reps], float)
        se = float(np.sqrt(((v - full[k]) ** 2).sum()))
        out[k] = {"point": float(full[k]), "se": se, "ci95": [full[k] - 1.96 * se, full[k] + 1.96 * se]}
    return out


def pmap(f, args):
    with Pool(2) as pool:
        return pool.starmap(f, args)


Ed, Eh = rel(dev), rel(hold)
pdev = prevalence(dev)

# ------------------------------------------------------------------------------------------------
if part == "power":
    c = json.load(open("results2/confirm_2005_main.json"))
    rules = {"H1": ("H1_rank3_minus_product", "above"), "H2": ("H2_sym_product_minus_hamacher", "contains0"),
             "H3": ("H3_product_minus_min", "above"), "H4": ("H4_block_product_minus_softimpute", "above"),
             "H5": ("H5_block_product_minus_katz", "contains0")}
    res = {}
    for h, (k, kind) in rules.items():
        lo, hi = c[k]["ci95"]; se = (hi - lo) / 3.92
        row = {"estimate": c[k]["point"], "se": se, "rule": kind}
        for d in [0.0, 0.03, 0.05]:
            if kind == "above":
                p = 1 - norm.cdf(1.96 - d / se)
            else:
                p = norm.cdf(1.96 - d / se) - norm.cdf(-1.96 - d / se)
            row[f"p_met_true_{d}"] = float(p)
        # true difference at which a directional rule is met with probability 0.8
        if kind == "above":
            row["diff_for_80pct"] = float((1.96 + norm.ppf(0.8)) * se)
        # TOST power for equivalence within +/-0.03 at a true difference of 0 (90% interval)
        for delta in [0.03, 0.05]:
            z = norm.ppf(0.95)
            row[f"tost_power_true0_delta{delta}"] = float(max(0.0, 2 * norm.cdf(delta / se - z) - 1))
        res[h] = row
    res["note"] = "normal approximation with the 2005-2006 JRR standard error; H6 is descriptive and has no interval"
    OUT["power"] = res
    save()

# ------------------------------------------------------------------------------------------------
def block_stats(E1, E2, grid=LAM_GRID):
    pp = {k: np.zeros_like(E1) for k in ["product", "katz", "si_raw", "si_logit", "si_scalecv"]}
    chosen = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        base = OFF & ~H
        Em = np.where(H, 0.0, E1)
        pp["product"][H] = X.paths(Em, X.t_product)[H]
        pp["katz"][H] = L.katz(Em, dyad=False)[H]
        lr, sr = L.choose_lambda(E1, grid, base=base)
        ll, sl = L.choose_lambda(E1, grid, base=base, transform="logit")
        Fr = L.soft_impute_reg(E1, base, lr)
        Fl = L.soft_impute_reg(E1, base, ll, transform="logit")
        pp["si_raw"][H] = Fr[H]; pp["si_logit"][H] = Fl[H]
        use_logit = max(sl.values()) >= max(sr.values())
        pp["si_scalecv"][H] = (Fl if use_logit else Fr)[H]
        chosen.append({"lambda_raw": lr, "lambda_logit": ll, "scale": "logit" if use_logit else "raw"})
    o = {}
    for k, v in pp.items():
        pb = np.array([r(v[OFF & (L.BLOCK == b)], E2[OFF & (L.BLOCK == b)]) for b in range(10)])
        o[f"weighted_{k}"] = float((pb * NB).sum() / NB.sum())
        o[f"pooled_{k}"] = r(v[OFF], E2[OFF])
        for b in range(10):
            o[f"block{b}_{k}"] = float(pb[b])
    for k in ["si_raw", "si_logit", "si_scalecv", "katz"]:
        o[f"weighted_product_minus_{k}"] = o["weighted_product"] - o[f"weighted_{k}"]
        o[f"pooled_product_minus_{k}"] = o["pooled_product"] - o[f"pooled_{k}"]
    o["n_blocks_logit_chosen"] = float(sum(c["scale"] == "logit" for c in chosen))
    return o, chosen


def _bs_dev(d):
    return block_stats(rel(d), Eh)[0]


def _bs_hold(h):
    return block_stats(Ed, rel(h))[0]


if part == "blocks":
    full, chosen = block_stats(Ed, Eh)
    tic("full done")
    reps = pmap(_bs_dev, [(d,) for d in jrr_reps(dev)]) + pmap(_bs_hold, [(h,) for h in jrr_reps(hold)])
    tic("JRR done")
    OUT["blocks"] = {"block_names": L.BLOCK_NAMES, "chosen": chosen, "jrr": jrr_summary(full, reps),
                     "n_replicates": len(reps)}
    full_ext, chosen_ext = block_stats(Ed, Eh, grid=LAM_EXT)
    OUT["blocks"]["extended_grid_point"] = {k: v for k, v in full_ext.items() if k.startswith("weighted")}
    OUT["blocks"]["extended_grid_chosen"] = chosen_ext
    save()

# ------------------------------------------------------------------------------------------------
if part == "blocks05":
    import clean2
    _rd = clean2.rd

    def rd(c, s):
        d = _rd(c, s)
        if c == "PAQ" and s == "D":
            v = d.PAD200.replace({3: 2}); m = d.PAD320.replace({3: 2})
            d = d.assign(PAQ605=v, PAQ650=v, PAQ620=m, PAQ665=m)
        return d
    clean2.rd = rd
    c05, _ = clean2.cycle("D")
    c05["w"] = c05.WTMEC2YR
    c05 = c05[CONCEPTS + ["w", "strata", "psu"]]
    E05 = rel(c05)

    def f_dev(d):
        return block_stats(rel(d), E05)[0]

    def f_05(h):
        return block_stats(Ed, rel(h))[0]
    full, chosen = block_stats(Ed, E05)
    reps = pmap(f_dev, [(d,) for d in jrr_reps(dev)]) + pmap(f_05, [(h,) for h in jrr_reps(c05)])
    OUT["blocks05"] = {"chosen": chosen, "jrr": jrr_summary(full, reps), "n_replicates": len(reps)}
    save()

# ------------------------------------------------------------------------------------------------
def recal_ldo(C, S, kind):
    """Leave-dyad-out monotone map of composed values C onto the scale of S over the 276 dyads (upper triangle)."""
    iu = np.where(UP)
    x, y = C[iu], S[iu]
    out = np.zeros_like(C)
    pars = []
    c = 1e-4
    for q in range(len(x)):
        keep = np.ones(len(x), bool); keep[q] = False
        if kind == "power":
            keep &= (y >= 0.01) & (x > 0)
            A = np.vstack([np.ones(keep.sum()), np.log(x[keep] + c)]).T
            a0, a1 = np.linalg.lstsq(A, np.log(y[keep] + c), rcond=None)[0]
            v = np.exp(a0) * x[q] ** a1
            pars.append(a1)
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x[keep], y[keep])
            v = float(iso.predict([x[q]])[0])
        out[iu[0][q], iu[1][q]] = v
    out = out + out.T
    return out, (float(np.median(pars)) if pars else None)


def recal_stats(E1, E2, p1):
    odds = p1 / (1 - p1)
    D = np.sqrt(odds[None, :] / odds[:, None])
    S = np.sqrt(E1 * E1.T)
    o = {}
    for T in TNORMS:
        C = X.paths(S, X.T_FIXED[T])
        o[f"{T}_redirected"] = r((C * D)[OFF], E2[OFF])
        for kind in ["power", "isotonic"]:
            Cr, a = recal_ldo(C, S, kind)
            o[f"{T}_{kind}_redirected"] = r((Cr * D)[OFF], E2[OFF])
            if a is not None:
                o[f"{T}_power_exponent"] = a
        o[f"{T}_directed"] = r(X.paths(E1, X.T_FIXED[T])[OFF], E2[OFF])
    for kind in ["", "_power", "_isotonic"]:
        for T in ["hamacher", "min"]:
            o[f"product_minus_{T}{kind}_redirected"] = o[f"product{kind}_redirected"] - o[f"{T}{kind}_redirected"]
    o["product_directed_minus_hamacher_power_redirected"] = o["product_directed"] - o["hamacher_power_redirected"]
    return o


def _rs_dev(d):
    return recal_stats(rel(d), Eh, prevalence(d))


def _rs_hold(h):
    return recal_stats(Ed, rel(h), pdev)


if part == "recalib":
    full = recal_stats(Ed, Eh, pdev)
    tic("full done")
    reps = pmap(_rs_dev, [(d,) for d in jrr_reps(dev)]) + pmap(_rs_hold, [(h,) for h in jrr_reps(hold)])
    OUT["recalib"] = {"jrr": jrr_summary(full, reps), "n_replicates": len(reps)}
    save()

# ------------------------------------------------------------------------------------------------
def _nested_mask(args):
    i, j = args
    base = OFF.copy(); base[i, j] = base[j, i] = False
    lam = L.choose_lambda(Ed, LAM_EXT, transform="logit", base=base)[0]
    F = L.soft_impute_reg(Ed, base, lam, transform="logit")
    return i, j, lam, F[i, j], F[j, i]


if part == "grid":
    lam_once, cvs = L.choose_lambda(Ed, LAM_EXT, transform="logit")
    once = L.softimpute_dyad(Ed, lam_once, transform="logit")
    orig = L.softimpute_dyad(Ed, 0.6, transform="logit")
    prod = X.paths(Ed, X.t_product)
    res = {"cv_rho_by_lambda": cvs, "lambda_once": lam_once, "rho_once": r(once[OFF], Eh[OFF]),
           "rho_original_grid_lambda0.6": r(orig[OFF], Eh[OFF]), "rho_product": r(prod[OFF], Eh[OFF])}
    tic("once done")
    with Pool(2) as pool:
        out = pool.map(_nested_mask, [(i, j) for i in range(P) for j in range(i + 1, P)])
    nest = np.zeros_like(Ed); lams = []
    for i, j, lam, a, b in out:
        nest[i, j], nest[j, i] = a, b; lams.append(lam)
    res["rho_nested"] = r(nest[OFF], Eh[OFF])
    res["nested_lambda_counts"] = {str(l): int(sum(np.isclose(lams, l))) for l in LAM_EXT}
    OUT["grid"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "coldstart":
    m = 1
    rows = []
    for rep in range(5):
        g = np.random.default_rng(900 + 31 * m + rep)
        per = {k: [] for k in ["logit_cv", "raw_cv", "logit_ext", "logit_lam5", "rank3", "product"]}
        lamc, spread, anchor_mu = [], [], []
        for c in range(P):
            others = [j for j in range(P) if j != c]
            anchors = set(g.choice(others, m, replace=False).tolist())
            H = np.zeros((P, P), bool)
            for j in others:
                if j not in anchors:
                    H[c, j] = H[j, c] = True
            obs = OFF & ~H
            ll = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep, transform="logit")[0]
            lr = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep)[0]
            le = L.choose_lambda(Ed, LAM_EXT, base=obs, seed=rep, transform="logit")[0]
            Fl = L.soft_impute_reg(Ed, obs, ll, transform="logit")
            per["logit_cv"].append(r(Fl[H], Eh[H]))
            per["raw_cv"].append(r(L.soft_impute_reg(Ed, obs, lr)[H], Eh[H]))
            per["logit_ext"].append(r(L.soft_impute_reg(Ed, obs, le, transform="logit")[H], Eh[H]))
            per["logit_lam5"].append(r(L.soft_impute_reg(Ed, obs, 5.0, transform="logit")[H], Eh[H]))
            per["rank3"].append(r(X.soft_impute(Ed, obs, 3)[H], Eh[H]))
            per["product"].append(r(X.paths(np.where(obs, Ed, 0.0), X.t_product)[H], Eh[H]))
            lamc.append(ll)
            # spread of the logit-scale predictions for the withheld row relative to the observed matrix
            Y = np.log((Fl + 0.005) / (1 - Fl + 0.005))
            spread.append(float(np.std(Y[H]) / np.std(np.log((Ed[obs] + 0.005) / (1 - Ed[obs] + 0.005)))))
            a = list(anchors)[0]
            anchor_mu.append(float((Ed[c, a] + Ed[a, c]) / 2))
        row = {f"median_{k}": float(np.nanmedian(v)) for k, v in per.items()}
        row["lambda_logit_counts"] = {str(l): int(sum(np.isclose(lamc, l))) for l in LAM_GRID}
        row["median_spread_ratio_logit"] = float(np.median(spread))
        # does logit soft-impute fail more when the single anchor is weak?
        al = np.array(anchor_mu); pl = np.array(per["logit_cv"]); pr3 = np.array(per["rank3"])
        weak = al < np.median(al)
        row["logit_median_weak_anchor"] = float(np.nanmedian(pl[weak])); row["logit_median_strong_anchor"] = float(np.nanmedian(pl[~weak]))
        row["rank3_median_weak_anchor"] = float(np.nanmedian(pr3[weak])); row["rank3_median_strong_anchor"] = float(np.nanmedian(pr3[~weak]))
        rows.append(row)
        tic(f"rep {rep}")
    keys = [k for k in rows[0] if not isinstance(rows[0][k], dict)]
    OUT["coldstart_m1"] = {"mean_over_reps": {k: float(np.mean([x[k] for x in rows])) for k in keys},
                           "reps": rows}
    save()

# ------------------------------------------------------------------------------------------------
def thin_eval(keep):
    keep = keep & OFF
    A = np.where(keep, Ed, 0.0)
    unrec = OFF & ~keep
    pr = {T: X.paths(A, X.T_FIXED[T]) for T in ["product", "hamacher", "min"]}
    pr["katz"] = L.katz(A, dyad=False)
    pr["rank3"] = X.soft_impute(Ed, keep, 3)
    lr = L.choose_lambda(Ed, LAM_GRID, base=keep)[0]
    ll = L.choose_lambda(Ed, LAM_GRID, base=keep, transform="logit")[0]
    pr["si_raw"] = L.soft_impute_reg(Ed, keep, lr)
    pr["si_logit"] = L.soft_impute_reg(Ed, keep, ll, transform="logit")
    return {k: r(v[unrec], Eh[unrec]) for k, v in pr.items()}


def _thin_rep(frac, rep):
    g = np.random.default_rng(3100 + rep + int(100 * frac))
    keep = (g.random((P, P)) < frac) & OFF
    return thin_eval(keep)


if part == "randthin":
    res = {}
    for frac in [0.5, 0.25]:
        rows = pmap(_thin_rep, [(frac, rep) for rep in range(20)])
        res[f"random_keep_{frac}"] = {k: {"mean": float(np.nanmean([x[k] for x in rows])),
                                          "sd": float(np.nanstd([x[k] for x in rows], ddof=1))} for k in rows[0]}
        res[f"random_keep_{frac}"]["product_minus_si_logit_sd"] = float(np.std([x["product"] - x["si_logit"] for x in rows], ddof=1))
        res[f"random_keep_{frac}"]["si_logit_first_count"] = int(sum(x["si_logit"] >= max(x.values()) for x in rows))
        res[f"random_keep_{frac}"]["hamacher_first_count"] = int(sum(x["hamacher"] >= max(x.values()) for x in rows))
        tic(f"random {frac}")
    TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
    ti = np.array([TYPES.index(TYPE[c]) for c in CONCEPTS])
    for name, srcs in {"sources_A": [(0, 1), (2, 3), (0, 3)], "sources_B": [(0, 2), (1, 3), (0, 1)]}.items():
        keep = np.zeros((P, P), bool)
        for a, b in srcs:
            mm = np.isin(ti, [a, b]); keep |= mm[:, None] & mm[None, :]
        res[name] = thin_eval(keep)
    OUT["randthin"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "crisp":
    crisp = ["uninsured", "hypertension", "cardiovascular", "cancer", "arthritis"]
    idx = [CONCEPTS.index(c) for c in CONCEPTS if any(c.startswith(k) for k in crisp)]
    rows = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            rows.append({"pair": [CONCEPTS[i], CONCEPTS[j]], "mu_ij": float(Ed[i, j]), "mu_ji": float(Ed[j, i]),
                         "both_ge_0.01": bool(Ed[i, j] >= 0.01 and Ed[j, i] >= 0.01)})
    OUT["crisp"] = {"concepts": [CONCEPTS[i] for i in idx], "pairs": rows}
    save()
