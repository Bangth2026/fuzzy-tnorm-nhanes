"""Path-consistent completion (PCC): experiments (v2.0.0).

PCC (tcomp.tc_complete with two_sided=True, cells="missing") completes the relation with a nuclear-norm penalty and pulls
every unobserved cell towards its best two-edge composition under a t-norm T, with weight mu:
    F(X) = 1/2 ||P_Omega(X - R)||^2 + mu/2 sum_{(i,j) not in Omega} (max_k T(X_ik, X_kj) - X_ij)^2 + lam ||X - m||_*.
mu = 0 is soft-impute on the raw scale; large mu makes the unobserved cells follow the composed paths.

Parts (results2/pcc_<part>.json):
  dyad       single dyads withheld, development -> 2007-2012; (lam, mu) by dyad cross-validation; JRR (90 replicates)
             for PCC and its differences from the product path, rank-3 completion and logit soft-impute
  block      whole relation types withheld; (lam, mu) chosen on the other nine blocks either by dyad cross-validation or
             by block cross-validation (each other block withheld in turn); JRR for differences from the product path,
             Katz and logit soft-impute (lambda by dyad CV, as in main Table III, and by block CV)
  coldstart  one concept keeps one anchor relation (5 repetitions); median rho over concepts
  tnorm      dyad design with T in {product, Hamacher, minimum} chosen by cross-validation (point estimates)
  violation  share of ordered pairs whose best two-edge product path exceeds the prediction by more than 0.01
"""
import json, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr
from analysis2 import dev, hold
from graph2 import relation_matrices, P
import rq3x as X
import rq6lib as L
import tcomp as TC

part = sys.argv[1]
OFF = ~np.eye(P, dtype=bool)
NB = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
LAM_SI = [0.01, 0.03, 0.1, 0.3, 0.6]
LAM_PCC = [0.03, 0.1, 0.3]
MU_PCC = [0.0, 0.25, 1.0, 5.0]
LAM_PCC_B = [0.03, 0.1]
MU_PCC_B = [0.0, 1.0, 5.0, 20.0]
OUT = {}


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/pcc_{part}.json", "w"), indent=1, default=float)


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
    for k, v0 in full.items():
        if not isinstance(v0, (float, int, np.floating)):
            continue
        v = np.array([x[k] for x in reps], float)
        se = float(np.sqrt(((v - v0) ** 2).sum()))
        out[k] = {"point": float(v0), "se": se, "ci95": [v0 - 1.96 * se, v0 + 1.96 * se],
                  "replicate_min": float(v.min()), "replicate_max": float(v.max())}
    return out


def pcc(R, mask, lam, mu, t="product", iters=300):
    return TC.tc_complete(R, mask, lam, mu, t, iters, two_sided=True, cells="missing")


def pcc_dyad(R, lam, mu, t="product"):
    out = np.zeros_like(R)
    for i in range(P):
        for j in range(i + 1, P):
            m = OFF.copy(); m[i, j] = m[j, i] = False
            Z = pcc(R, m, lam, mu, t)
            out[i, j], out[j, i] = Z[i, j], Z[j, i]
    return out


def pcc_cv_dyad(R, base=None, lams=LAM_PCC, mus=MU_PCC, tnorms=("product",), folds=5, seed=0):
    base = OFF if base is None else base
    g = np.random.default_rng(seed)
    dy = [(i, j) for i in range(P) for j in range(i + 1, P) if base[i, j] and base[j, i]]
    idx = g.permutation(len(dy))
    sc = {}
    for f in range(folds):
        m = base.copy()
        for q in idx[f::folds]:
            i, j = dy[q]; m[i, j] = m[j, i] = False
        H = base & ~m
        for t in tnorms:
            for lam in lams:
                for mu in mus:
                    sc.setdefault((t, lam, mu), []).append(r(pcc(R, m, lam, mu, t)[H], R[H]))
    mean = {k: float(np.mean(v)) for k, v in sc.items()}
    return max(mean, key=mean.get), {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in mean.items()}


# ------------------------------------------------------------------------------------------------
def dyad_stats(E1, E2):
    (t, lam, mu), _ = pcc_cv_dyad(E1)
    p_pcc = pcc_dyad(E1, lam, mu, t)
    p_prod = L.path_features(E1)["prod_max"]  # as in main Table II (paths never use the dyad)
    p_r3 = X.lowrank_baseline(E1, 3, dyad=True)
    ll, _ = L.choose_lambda(E1, LAM_SI, transform="logit")
    p_si = L.softimpute_dyad(E1, ll, transform="logit")
    o = {"pcc": r(p_pcc[OFF], E2[OFF]), "product": r(p_prod[OFF], E2[OFF]), "rank3": r(p_r3[OFF], E2[OFF]),
         "si_logit": r(p_si[OFF], E2[OFF]), "lam": lam, "mu": mu}
    for k in ["product", "rank3", "si_logit"]:
        o[f"pcc_minus_{k}"] = o["pcc"] - o[k]
    return o, {"pcc": p_pcc, "product": p_prod, "rank3": p_r3, "si_logit": p_si}


def _dyad_dev(d):
    return dyad_stats(rel(d), Eh)[0]


def _dyad_hold(h):
    return dyad_stats(Ed, rel(h))[0]


# ------------------------------------------------------------------------------------------------
def block_cv_pcc(E1, b, lams=LAM_PCC_B, mus=MU_PCC_B):
    """Choose (lam, mu) for block b by withholding each other block c in turn (b also withheld)."""
    Hb = OFF & (L.BLOCK == b)
    sc = {}
    for c in range(10):
        if c == b:
            continue
        Hc = OFF & (L.BLOCK == c)
        m = OFF & ~Hb & ~Hc
        for lam in lams:
            for mu in mus:
                sc.setdefault((lam, mu), []).append((r(pcc(E1, m, lam, mu, iters=400)[Hc], E1[Hc]), NB[c]))
    mean = {k: float(sum(a * w for a, w in v) / sum(w for _, w in v)) for k, v in sc.items()}
    return max(mean, key=mean.get)


def block_cv_si(E1, b):
    Hb = OFF & (L.BLOCK == b)
    sc = {}
    for c in range(10):
        if c == b:
            continue
        Hc = OFF & (L.BLOCK == c)
        m = OFF & ~Hb & ~Hc
        for lam in LAM_SI:
            sc.setdefault(lam, []).append((r(L.soft_impute_reg(E1, m, lam, transform="logit")[Hc], E1[Hc]), NB[c]))
    mean = {k: float(sum(a * w for a, w in v) / sum(w for _, w in v)) for k, v in sc.items()}
    return max(mean, key=mean.get)


def block_stats(E1, E2):
    keys = ["product", "katz", "si_logit", "si_logit_blockcv", "pcc_dyadcv", "pcc_blockcv"]
    pp = {k: np.zeros_like(E1) for k in keys}
    chosen = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        base = OFF & ~H
        Em = np.where(H, 0.0, E1)
        pp["product"][H] = X.paths(Em, X.t_product)[H]
        pp["katz"][H] = L.katz(Em, dyad=False)[H]
        ll, _ = L.choose_lambda(E1, LAM_SI, base=base, transform="logit")
        pp["si_logit"][H] = L.soft_impute_reg(E1, base, ll, transform="logit")[H]
        lb = block_cv_si(E1, b)
        pp["si_logit_blockcv"][H] = L.soft_impute_reg(E1, base, lb, transform="logit")[H]
        (t, lam_d, mu_d), _ = pcc_cv_dyad(E1, base=base, lams=LAM_PCC_B, mus=MU_PCC_B)
        pp["pcc_dyadcv"][H] = pcc(E1, base, lam_d, mu_d, iters=400)[H]
        lam_b, mu_b = block_cv_pcc(E1, b)
        pp["pcc_blockcv"][H] = pcc(E1, base, lam_b, mu_b, iters=400)[H]
        chosen.append({"si_lambda_dyadcv": ll, "si_lambda_blockcv": lb, "pcc_dyadcv": [lam_d, mu_d],
                       "pcc_blockcv": [lam_b, mu_b]})
    o = {}
    for k, v in pp.items():
        pb = np.array([r(v[OFF & (L.BLOCK == b)], E2[OFF & (L.BLOCK == b)]) for b in range(10)])
        o[f"weighted_{k}"] = float((pb * NB).sum() / NB.sum())
        o[f"pooled_{k}"] = r(v[OFF], E2[OFF])
        for b in range(10):
            o[f"block{b}_{k}"] = float(pb[b])
    for a in ["pcc_blockcv", "pcc_dyadcv"]:
        for k in ["product", "katz", "si_logit", "si_logit_blockcv"]:
            o[f"weighted_{a}_minus_{k}"] = o[f"weighted_{a}"] - o[f"weighted_{k}"]
            o[f"pooled_{a}_minus_{k}"] = o[f"pooled_{a}"] - o[f"pooled_{k}"]
    return o, chosen


def _block_dev(d):
    return block_stats(rel(d), Eh)[0]


def _block_hold(h):
    return block_stats(Ed, rel(h))[0]


def pmap(f, args):
    with Pool(2) as pool:
        return pool.starmap(f, args)


Ed, Eh = rel(dev), rel(hold)

if part == "dyad":
    full, preds = dyad_stats(Ed, Eh)
    tic(f"full {full}")
    np.savez("results2/pcc_dyad_predictions.npz", **preds)
    reps = pmap(_dyad_dev, [(d,) for d in jrr_reps(dev)]) + pmap(_dyad_hold, [(h,) for h in jrr_reps(hold)])
    OUT["dyad"] = {"full": full, "jrr": jrr_summary(full, reps), "n_replicates": len(reps),
                   "chosen_in_replicates": [[x["lam"], x["mu"]] for x in reps], "grid": {"lam": LAM_PCC, "mu": MU_PCC}}
    save()

if part == "block":
    full, chosen = block_stats(Ed, Eh)
    tic("full done")
    OUT["block"] = {"full": full, "chosen": chosen, "block_names": L.BLOCK_NAMES}
    save()
    reps = pmap(_block_dev, [(d,) for d in jrr_reps(dev)]) + pmap(_block_hold, [(h,) for h in jrr_reps(hold)])
    OUT["block"]["jrr"] = jrr_summary(full, reps)
    OUT["block"]["n_replicates"] = len(reps)
    save()

if part == "coldstart":
    rows = []
    for rep in range(5):
        g = np.random.default_rng(900 + 31 + rep)  # same anchors as scale_checks.py coldstart (m = 1)
        per = {k: [] for k in ["pcc", "si_logit", "rank3", "product"]}
        ch = []
        for c in range(P):
            others = [j for j in range(P) if j != c]
            anchors = set(g.choice(others, 1, replace=False).tolist())
            H = np.zeros((P, P), bool)
            for j in others:
                if j not in anchors:
                    H[c, j] = H[j, c] = True
            obs = OFF & ~H
            (t, lam, mu), _ = pcc_cv_dyad(Ed, base=obs, seed=rep)
            per["pcc"].append(r(pcc(Ed, obs, lam, mu, iters=400)[H], Eh[H]))
            ll = L.choose_lambda(Ed, LAM_SI, base=obs, seed=rep, transform="logit")[0]
            per["si_logit"].append(r(L.soft_impute_reg(Ed, obs, ll, transform="logit")[H], Eh[H]))
            per["rank3"].append(r(X.soft_impute(Ed, obs, 3)[H], Eh[H]))
            per["product"].append(r(X.paths(np.where(obs, Ed, 0.0), X.t_product)[H], Eh[H]))
            ch.append([lam, mu])
        rows.append({**{f"median_{k}": float(np.nanmedian(v)) for k, v in per.items()}, "chosen": ch})
        tic(f"rep {rep}")
    OUT["coldstart"] = {"reps": rows, **{f"mean_median_{k}": float(np.mean([x[f'median_{k}'] for x in rows]))
                                         for k in ["pcc", "si_logit", "rank3", "product"]}}
    save()

if part == "tnorm":
    (t, lam, mu), sc = pcc_cv_dyad(Ed, tnorms=("product", "hamacher", "min"))
    o = {"chosen": [t, lam, mu], "cv": sc}
    for tt in ["product", "hamacher", "min"]:
        best = max(((k, v) for k, v in sc.items() if k.startswith(tt + "|") and not k.endswith("|0.0")), key=lambda kv: kv[1])
        _, l2, m2 = best[0].split("|")
        o[f"{tt}_best_positive_mu"] = {"lam": float(l2), "mu": float(m2), "cv_rho": best[1],
                                       "rho_heldout": r(pcc_dyad(Ed, float(l2), float(m2), tt)[OFF], Eh[OFF])}
    OUT["tnorm"] = o
    save()

if part == "violation":
    P_ = dict(np.load("results2/pcc_dyad_predictions.npz"))
    o = {}
    for k, M in list(P_.items()) + [("development_relation", Ed), ("heldout_relation", Eh)]:
        share, excess = TC.violation(M, "product", tol=0.01)
        o[k] = {"share_pairs_violated_tol0.01": share, "mean_excess": excess}
    OUT["violation"] = o
    save()

# ------------------------------------------------------------------------------------------------
LAM_R = [0.03, 0.1, 0.3]
MU_R = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]


def regime_scores(E1, base_extra=None):
    """Cross-validation scores on E1 for every (lam, mu) under two missingness regimes:
    dyad folds (5 folds of random dyads; pooled rho) and block folds (each block withheld in turn; within-block rho,
    pair-weighted). base_extra: additional cells treated as unobserved throughout (e.g. a withheld block)."""
    base = OFF if base_extra is None else OFF & ~base_extra
    _, sd = pcc_cv_dyad(E1, base=base, lams=LAM_R, mus=MU_R)
    sb = {}
    for c in range(10):
        Hc = OFF & (L.BLOCK == c) & base
        if Hc.sum() < 5:
            continue
        m = base & ~Hc
        for lam in LAM_R:
            for mu in MU_R:
                sb.setdefault((lam, mu), []).append((r(pcc(E1, m, lam, mu, iters=400)[Hc], E1[Hc]), Hc.sum()))
    sb = {k: float(sum(a * w for a, w in v) / sum(w for _, w in v)) for k, v in sb.items()}
    sd = {(float(k.split("|")[1]), float(k.split("|")[2])): v for k, v in sd.items()}
    return sd, sb


def minimax_choice(sd, sb):
    bd, bb = max(sd.values()), max(sb.values())
    regret = {k: max(bd - sd[k], bb - sb[k]) for k in sd}
    return min(regret, key=regret.get), regret


if part == "robust":
    sd, sb = regime_scores(Ed)
    (lam, mu), regret = minimax_choice(sd, sb)
    tic(f"minimax choice lam={lam} mu={mu}")
    o = {"cv_dyad": {f"{k[0]}|{k[1]}": v for k, v in sd.items()}, "cv_block": {f"{k[0]}|{k[1]}": v for k, v in sb.items()},
         "regret": {f"{k[0]}|{k[1]}": v for k, v in regret.items()}, "chosen": [lam, mu]}
    p = pcc_dyad(Ed, lam, mu)
    o["dyad_rho"] = r(p[OFF], Eh[OFF])
    pool = np.zeros_like(Ed); vals = []; chosen_b = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        sdb, sbb = regime_scores(Ed, base_extra=H)       # selection never sees block b
        (lb, mb), _ = minimax_choice(sdb, sbb)
        chosen_b.append([lb, mb])
        Z = pcc(Ed, OFF & ~H, lb, mb, iters=400)
        pool[H] = Z[H]; vals.append(r(Z[H], Eh[H]))
    o["block_chosen"] = chosen_b
    o["block_weighted_rho"] = float((np.array(vals) * NB).sum() / NB.sum())
    o["block_pooled_rho"] = r(pool[OFF], Eh[OFF])
    o["block_rho_by_block"] = vals
    OUT["robust"] = o
    save()

# ------------------------------------------------------------------------------------------------
# Regime-robust setting: (lam, mu) chosen once on development data by minimax regret (part "robust"); in the block design
# the setting chosen without each block is used for that block. Selection is not repeated inside JRR replicates.
def robust_setting():
    d = json.load(open("results2/pcc_robust.json"))["robust"]
    return tuple(d["chosen"]), [tuple(x) for x in d["block_chosen"]]


def robust_stats(E1, E2, setting, block_setting):
    lam, mu = setting
    o = {}
    pd_ = pcc_dyad(E1, lam, mu)
    pr = L.path_features(E1)["prod_max"]
    r3 = X.lowrank_baseline(E1, 3, dyad=True)
    ll, _ = L.choose_lambda(E1, LAM_SI, transform="logit")
    si = L.softimpute_dyad(E1, ll, transform="logit")
    kz = L.katz(E1)
    dy = {"pcc": pd_, "product": pr, "rank3": r3, "si_logit": si, "katz": kz}
    for k, v in dy.items():
        o[f"dyad_{k}"] = r(v[OFF], E2[OFF])
    pb = {k: np.zeros_like(E1) for k in ["pcc", "product", "katz", "si_logit", "rank3"]}
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        base = OFF & ~H
        Em = np.where(H, 0.0, E1)
        lb, mb = block_setting[b]
        pb["pcc"][H] = pcc(E1, base, lb, mb, iters=400)[H]
        pb["product"][H] = X.paths(Em, X.t_product)[H]
        pb["katz"][H] = L.katz(Em, dyad=False)[H]
        l2, _ = L.choose_lambda(E1, LAM_SI, base=base, transform="logit")
        pb["si_logit"][H] = L.soft_impute_reg(E1, base, l2, transform="logit")[H]
        pb["rank3"][H] = X.soft_impute(E1, base, 3)[H]
    for k, v in pb.items():
        vals = np.array([r(v[OFF & (L.BLOCK == b)], E2[OFF & (L.BLOCK == b)]) for b in range(10)])
        o[f"block_{k}"] = float((vals * NB).sum() / NB.sum())
        o[f"blockpooled_{k}"] = r(v[OFF], E2[OFF])
    meths = ["pcc", "product", "katz", "si_logit", "rank3"]
    bd = max(o[f"dyad_{k}"] for k in meths); bb = max(o[f"block_{k}"] for k in meths)
    bp = max(o[f"blockpooled_{k}"] for k in meths)
    for k in meths:
        o[f"regret_{k}"] = max(bd - o[f"dyad_{k}"], bb - o[f"block_{k}"])
        o[f"regretpooled_{k}"] = max(bd - o[f"dyad_{k}"], bp - o[f"blockpooled_{k}"])
    for k in ["product", "katz", "si_logit", "rank3"]:
        o[f"dyad_pcc_minus_{k}"] = o["dyad_pcc"] - o[f"dyad_{k}"]
        o[f"block_pcc_minus_{k}"] = o["block_pcc"] - o[f"block_{k}"]
        o[f"blockpooled_pcc_minus_{k}"] = o["blockpooled_pcc"] - o[f"blockpooled_{k}"]
        o[f"regret_{k}_minus_pcc"] = o[f"regret_{k}"] - o["regret_pcc"]
        o[f"regretpooled_{k}_minus_pcc"] = o[f"regretpooled_{k}"] - o["regretpooled_pcc"]
    return o, dy


def _rob_dev(d):
    s, bs = robust_setting()
    return robust_stats(rel(d), Eh, s, bs)[0]


def _rob_hold(h):
    s, bs = robust_setting()
    return robust_stats(Ed, rel(h), s, bs)[0]


if part == "robust_jrr":
    s, bs = robust_setting()
    full, dy = robust_stats(Ed, Eh, s, bs)
    tic(f"full {json.dumps({k: round(v, 4) for k, v in full.items()})}")
    np.savez("results2/pcc_robust_dyad_predictions.npz", **dy)
    OUT["robust_jrr"] = {"setting": s, "block_setting": bs, "full": full}
    save()
    reps = pmap(_rob_dev, [(d,) for d in jrr_reps(dev)]) + pmap(_rob_hold, [(h,) for h in jrr_reps(hold)])
    OUT["robust_jrr"]["jrr"] = jrr_summary(full, reps)
    OUT["robust_jrr"]["n_replicates"] = len(reps)
    save()

if part == "robust_extra":
    (lam, mu), _ = robust_setting()
    o = {}
    # cold start (m = 1), same anchors as scale_checks.py
    rows = []
    for rep in range(5):
        g = np.random.default_rng(900 + 31 + rep)
        per = {k: [] for k in ["pcc", "si_logit", "rank3", "product", "katz"]}
        for c in range(P):
            others = [j for j in range(P) if j != c]
            anchors = set(g.choice(others, 1, replace=False).tolist())
            H = np.zeros((P, P), bool)
            for j in others:
                if j not in anchors:
                    H[c, j] = H[j, c] = True
            obs = OFF & ~H
            per["pcc"].append(r(pcc(Ed, obs, lam, mu, iters=400)[H], Eh[H]))
            ll = L.choose_lambda(Ed, LAM_SI, base=obs, seed=rep, transform="logit")[0]
            per["si_logit"].append(r(L.soft_impute_reg(Ed, obs, ll, transform="logit")[H], Eh[H]))
            per["rank3"].append(r(X.soft_impute(Ed, obs, 3)[H], Eh[H]))
            Em = np.where(obs, Ed, 0.0)
            per["product"].append(r(X.paths(Em, X.t_product)[H], Eh[H]))
            per["katz"].append(r(L.katz(Em, dyad=False)[H], Eh[H]))
        rows.append({k: float(np.nanmedian(v)) for k, v in per.items()})
        tic(f"coldstart rep {rep}")
    o["coldstart_m1"] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
    o["coldstart_m1_reps"] = rows
    # random 25% / 50% dyad masking (20 repetitions)
    for frac in [0.25, 0.5]:
        acc = {k: [] for k in ["pcc", "si_logit", "rank3", "product"]}
        dyl = [(i, j) for i in range(P) for j in range(i + 1, P)]
        for rep in range(20):
            g = np.random.default_rng(5000 + rep + int(frac * 100))
            sel = g.choice(len(dyl), int(round(frac * len(dyl))), replace=False)
            H = np.zeros((P, P), bool)
            for q in sel:
                i, j = dyl[q]; H[i, j] = H[j, i] = True
            obs = OFF & ~H
            acc["pcc"].append(r(pcc(Ed, obs, lam, mu, iters=400)[H], Eh[H]))
            ll = L.choose_lambda(Ed, LAM_SI, base=obs, seed=rep, transform="logit")[0]
            acc["si_logit"].append(r(L.soft_impute_reg(Ed, obs, ll, transform="logit")[H], Eh[H]))
            acc["rank3"].append(r(X.soft_impute(Ed, obs, 3)[H], Eh[H]))
            acc["product"].append(r(X.paths(np.where(obs, Ed, 0.0), X.t_product)[H], Eh[H]))
        o[f"random_{frac}"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))} for k, v in acc.items()}
        tic(f"random {frac}")
    # transitivity violations of the dyad-design predictions
    Pr = dict(np.load("results2/pcc_robust_dyad_predictions.npz"))
    o["violation"] = {k: dict(zip(["share_tol0.01", "mean_excess"], TC.violation(M, "product", tol=0.01)))
                      for k, M in list(Pr.items()) + [("heldout_relation", Eh), ("development_relation", Ed)]}
    # t-norm inside PCC (dyad design, fixed lam, mu)
    o["tnorm_dyad"] = {t: r(pcc_dyad(Ed, lam, mu, t)[OFF], Eh[OFF]) for t in ["product", "hamacher", "min"]}
    OUT["robust_extra"] = o
    save()

def _c05_dev(d):
    s, bs = robust_setting()
    return robust_stats(rel(d), E05, s, bs)[0]


def _c05_05(h):
    s, bs = robust_setting()
    return robust_stats(Ed, rel(h), s, bs)[0]


if part == "c05":
    # NHANES 2005-2006 as an additional target (post hoc for PCC: the setting was fixed on 2013-2018 only)
    import clean2
    from graph2 import CONCEPTS
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
    s, bs = robust_setting()
    full, _ = robust_stats(Ed, E05, s, bs)
    tic("c05 full done")
    reps = pmap(_c05_dev, [(d,) for d in jrr_reps(dev)]) + pmap(_c05_05, [(h,) for h in jrr_reps(c05)])
    OUT["c05"] = {"full": full, "jrr": jrr_summary(full, reps), "n_replicates": len(reps)}
    save()
