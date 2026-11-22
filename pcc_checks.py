"""Checks of path-consistent completion (PCC) across masking regimes (v2.1.0).

Parts (results2/pcc2_<part>.json):
  grid      penalty grid for the nuclear norm extended to lambda in {0.003, 0.01}; minimax-regret choice over the two
            selection regimes (random-dyad folds, type-block folds) on 2013-2018, for the full matrix and with each
            type block removed before selection (lambda and gamma are both re-chosen per block; the inner block folds are
            computed with the evaluated block and the fold block withheld together)
  dev       cross-validation on 2013-2018 alone under five regimes (random-dyad folds, type-block folds, random 25% and
            50% of dyads, one concept with one anchor relation) for every PCC setting and for the fixed methods and
            blends; minimax choices over the two selection regimes, over all five, and leaving each regime out
  target    2007-2012 as target: every method and every PCC setting under every regime run in the paper (single
            dyads, type blocks, random 25%/50%, cold start m = 1, 3, 6, two simulated-source designs, random and strength
            thinning)
  jrr       JRR intervals (90 replicates) in the single-dyad and type-block designs for PCC, the product path, logit
            soft-impute and the two blends, with worst-case regret and two one-sided tests
  diag      PCC diagnostics: movement of observed cells, agreement with the product path at large gamma (two-edge and
            two-or-three-edge paths), iterations, objective monotonicity, starting points, ties in the argmax
  variants  the two variants set aside during development, re-run on the 2013-2018 cross-validation and on the target
  tnormsym  the t-norm inside PCC on the symmetric relation (276 dyads)
  runtime   time per PCC fit for n = 24 and simulated relations with n = 100, 200, 400
  target21  NHANES August 2021-August 2023 as a further target (plan: confirmatory/PLAN_2021.md); parts c21 and c21jrr

Blends (comparators for the path term): blend_rank(w) = w * rank(product) + (1 - w) * rank(logit soft-impute), ranks
taken over the cells withheld by one mask (over all 552 cells in the single-dyad design); blend_mu(w) the same on the
relation scale; blend_raw(w) = w * rank(product) + (1 - w) * rank(raw-scale soft-impute at the PCC lambda, i.e. PCC with
gamma = 0). w is chosen on 2013-2018 by the minimax-regret rule used for PCC.
"""
import json, sys, time, os
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr, rankdata, norm
from analysis2 import dev, hold
from graph2 import relation_matrices, P
import rq3x as X
import rq6lib as L
import tcomp as TC

part = sys.argv[1] if len(sys.argv) > 1 else "none"
OFF = ~np.eye(P, dtype=bool)
NB = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
LAM_SI = [0.01, 0.03, 0.1, 0.3, 0.6]
LAM_G = [0.003, 0.01, 0.03, 0.1, 0.3]
MU_G = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
THETAS = [(l, m) for l in LAM_G for m in MU_G]
W_G = [round(x, 2) for x in np.linspace(0, 1, 11)]
DYADS = [(i, j) for i in range(P) for j in range(i + 1, P)]
NCPU = int(os.environ.get("NCPU", "2"))
OUT = {}


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save(name=None):
    json.dump(OUT, open(f"results2/pcc2_{name or part}.json", "w"), indent=1, default=float)


def rel(df):
    return relation_matrices(df)[2]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def key(t):
    return f"{t[0]}|{t[1]}"


def pcc(R, mask, lam, mu, t="product", iters=400, X0=None):
    return TC.tc_complete(R, mask, lam, mu, t, iters, X0=X0, two_sided=True, cells="missing")


def pmap(f, args, n=None):
    with Pool(n or NCPU) as pool:
        return pool.starmap(f, args)


def rk(v):
    return rankdata(v) / len(v)


def within_block(pred, tgt, cells=None):
    vals = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        vals.append(r(pred[H], tgt[H]))
    vals = np.array(vals)
    return float((vals * NB).sum() / NB.sum()), vals


# ---------------------------------------------------------------- masks
def mask_random(frac, rep, seed0):
    g = np.random.default_rng(seed0 + rep + int(frac * 100))
    sel = g.choice(len(DYADS), int(round(frac * len(DYADS))), replace=False)
    H = np.zeros((P, P), bool)
    for q in sel:
        i, j = DYADS[q]; H[i, j] = H[j, i] = True
    return H


def masks_cold(m, rep, seed0=900):
    g = np.random.default_rng(seed0 + 31 * m + rep)
    out = []
    for c in range(P):
        others = [j for j in range(P) if j != c]
        anchors = set(g.choice(others, m, replace=False).tolist())
        H = np.zeros((P, P), bool)
        for j in others:
            if j not in anchors:
                H[c, j] = H[j, c] = True
        out.append(H)
    return out


def masks_dyadfolds(seed=0, base=None, folds=5):
    base = OFF if base is None else base
    g = np.random.default_rng(seed)
    dy = [(i, j) for (i, j) in DYADS if base[i, j] and base[j, i]]
    idx = g.permutation(len(dy))
    out = []
    for f in range(folds):
        H = np.zeros((P, P), bool)
        for q in idx[f::folds]:
            i, j = dy[q]; H[i, j] = H[j, i] = True
        out.append(H)
    return out


TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]


def keep_sources(srcs):
    from graph2 import CONCEPTS, TYPE
    ti = np.array([TYPES.index(TYPE[c]) for c in CONCEPTS])
    keep = np.zeros((P, P), bool)
    for a, b in srcs:
        m = np.isin(ti, [a, b])
        keep |= m[:, None] & m[None, :]
    return keep & OFF


SOURCES = {"sources_A": [(0, 1), (2, 3), (0, 3)], "sources_B": [(0, 2), (1, 3), (0, 1)]}


# ---------------------------------------------------------------- fixed methods on one mask
def fixed_preds(E1, H, si_seed=0, lam_raw=0.03):
    """Predictions of the fixed methods when the cells H are unobserved (H symmetric or not)."""
    obs = OFF & ~H
    Em = np.where(obs, E1, 0.0)
    out = {"product": X.paths(Em, X.t_product), "hamacher": X.paths(Em, X.T_FIXED["hamacher"]),
           "katz": L.katz(Em, dyad=False), "rank3": X.soft_impute(E1, obs, 3)}
    ll = L.choose_lambda(E1, LAM_SI, base=obs, seed=si_seed, transform="logit")[0]
    out["si_logit"] = L.soft_impute_reg(E1, obs, ll, transform="logit")
    out["si_raw_pcclam"] = pcc(E1, obs, lam_raw, 0.0)
    return out


def blends(pred, cells, w):
    """Rank and relation-scale blends of the product with logit soft-impute and with raw soft-impute (PCC gamma = 0)."""
    a, b, c = pred["product"][cells], pred["si_logit"][cells], pred["si_raw_pcclam"][cells]
    return {"blend_rank": w * rk(a) + (1 - w) * rk(b), "blend_mu": w * a + (1 - w) * b,
            "blend_raw": w * rk(a) + (1 - w) * rk(c)}


# ---------------------------------------------------------------- development regimes
DEV_REGIMES = ["dyad", "block", "random_0.25", "random_0.5", "cold_1"]


def dev_masks(regime, base_extra=None):
    """List of (H, group) masks on 2013-2018 for a regime. Groups are pooled into one score per regime."""
    base = OFF if base_extra is None else OFF & ~base_extra
    if regime == "dyad":
        return [(H, f) for f, H in enumerate(masks_dyadfolds(0, base))]
    if regime == "block":
        out = []
        for c in range(10):
            Hc = OFF & (L.BLOCK == c) & base
            if Hc.sum() >= 5:
                out.append((Hc, c))
        return out
    if regime.startswith("random"):
        frac = float(regime.split("_")[1])
        return [(mask_random(frac, rep, 8000) & base, rep) for rep in range(5)]
    if regime == "cold_1":
        out = []
        for rep in range(2):
            for c, H in enumerate(masks_cold(1, rep, seed0=8100)):
                out.append((H & base, (rep, c)))
        return out
    raise ValueError(regime)


def regime_score(regime, scores):
    """scores: list of (rho, n_cells, group). Aggregation matches the target designs."""
    if regime == "dyad":
        return float(np.mean([s for s, _, _ in scores]))
    if regime == "block":
        return float(sum(s * n for s, n, _ in scores) / sum(n for _, n, _ in scores))
    if regime.startswith("random"):
        return float(np.nanmean([s for s, _, _ in scores]))
    if regime == "cold_1":
        reps = sorted({g[0] for _, _, g in scores})
        return float(np.mean([np.nanmedian([s for s, _, g in scores if g[0] == k]) for k in reps]))
    raise ValueError(regime)


def _dev_one(E1, H, base_extra):
    """All PCC settings and fixed methods on one development mask; returns dicts of rho on H."""
    base = OFF if base_extra is None else OFF & ~base_extra
    obs = base & ~H
    sc = {}
    for t in THETAS:
        sc[key(t)] = r(pcc(E1, obs, *t)[H], E1[H])
    # fixed methods see the cells withheld by base_extra as unobserved too
    fp = fixed_preds(E1, ~obs & OFF)
    for k, v in fp.items():
        sc[k] = r(v[H], E1[H])
    for w in W_G:
        for k, v in blends(fp, H, w).items():
            sc[f"{k}|{w}"] = r(v, E1[H])
    return sc, int(H.sum())


def dev_scores(E1, base_extra=None, regimes=DEV_REGIMES):
    jobs, tags = [], []
    for reg in regimes:
        for H, g in dev_masks(reg, base_extra):
            jobs.append((E1, H, base_extra)); tags.append((reg, g))
    res = pmap(_dev_one, jobs)
    out = {}
    for reg in regimes:
        rows = [(sc, n, g) for (sc, n), (rg, g) in zip(res, tags) if rg == reg]
        out[reg] = {k: regime_score(reg, [(sc[k], n, g) for sc, n, g in rows]) for k in rows[0][0]}
    return out


def minimax(scores, regimes, cands):
    best = {reg: max(scores[reg][c] for c in cands) for reg in regimes}
    reg_ = {c: max(best[reg] - scores[reg][c] for reg in regimes) for c in cands}
    return min(reg_, key=reg_.get), reg_


PCC_KEYS = [key(t) for t in THETAS]


def choices(scores):
    """Minimax choices of the PCC setting and of the blend weights over several regime sets."""
    sets = {"two": ["dyad", "block"], "all5": DEV_REGIMES}
    for reg in DEV_REGIMES:
        sets[f"without_{reg}"] = [x for x in DEV_REGIMES if x != reg]
    out = {}
    for nm, regs in sets.items():
        c, rg = minimax(scores, regs, PCC_KEYS)
        o = {"pcc": c, "pcc_regret_dev": rg[c]}
        for b in ["blend_rank", "blend_mu", "blend_raw"]:
            cw, rw = minimax(scores, regs, [f"{b}|{w}" for w in W_G])
            o[b] = float(cw.split("|")[1]); o[f"{b}_regret_dev"] = rw[cw]
        out[nm] = o
    return out


Ed, Eh = rel(dev), rel(hold)

# ================================================================= grid (extended lambda)
def regime_scores_two(E1, base_extra=None):
    """Exactly the selection of pcc_experiments.py (robust) on the extended lambda grid."""
    sc = dev_scores_pcc_only(E1, base_extra)
    return sc


def _pcc_only(E1, H, base_extra):
    base = OFF if base_extra is None else OFF & ~base_extra
    obs = base & ~H
    return {key(t): r(pcc(E1, obs, *t)[H], E1[H]) for t in THETAS}, int(H.sum())


def dev_scores_pcc_only(E1, base_extra=None, regimes=("dyad", "block")):
    jobs, tags = [], []
    for reg in regimes:
        for H, g in dev_masks(reg, base_extra):
            jobs.append((E1, H, base_extra)); tags.append((reg, g))
    res = pmap(_pcc_only, jobs)
    out = {}
    for reg in regimes:
        rows = [(sc, n, g) for (sc, n), (rg, g) in zip(res, tags) if rg == reg]
        out[reg] = {k: regime_score(reg, [(sc[k], n, g) for sc, n, g in rows]) for k in PCC_KEYS}
    return out


def _pcc_dyad_theta(E1, lam, mu):
    out = np.zeros_like(E1)
    for i, j in DYADS:
        m = OFF.copy(); m[i, j] = m[j, i] = False
        Z = pcc(E1, m, lam, mu, iters=300)
        out[i, j], out[j, i] = Z[i, j], Z[j, i]
    return out


if part == "grid":
    sc = dev_scores_pcc_only(Ed)
    c, rg = minimax(sc, ["dyad", "block"], PCC_KEYS)
    tic(f"extended grid choice {c} regret {rg[c]:.4f}")
    c_old, rg_old = minimax(sc, ["dyad", "block"], [k for k in PCC_KEYS if float(k.split('|')[0]) >= 0.03])
    o = {"cv_dyad": sc["dyad"], "cv_block": sc["block"], "regret": rg, "chosen": c, "chosen_original_grid": c_old,
         "grid": {"lam": LAM_G, "gamma": MU_G}}
    bl = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        sb = dev_scores_pcc_only(Ed, base_extra=H)
        cb, _ = minimax(sb, ["dyad", "block"], PCC_KEYS)
        cbo, _ = minimax(sb, ["dyad", "block"], [k for k in PCC_KEYS if float(k.split('|')[0]) >= 0.03])
        bl.append({"block": b, "chosen": cb, "chosen_original_grid": cbo})
        tic(f"block {b}: {cb} (original grid {cbo})")
    o["block_chosen"] = bl
    lam, mu = map(float, c.split("|"))
    o["dyad_rho_heldout"] = r(_pcc_dyad_theta(Ed, lam, mu)[OFF], Eh[OFF])
    pool = np.zeros_like(Ed)
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        lb, mb = map(float, bl[b]["chosen"].split("|"))
        pool[H] = pcc(Ed, OFF & ~H, lb, mb)[H]
    o["block_weighted_rho_heldout"], vals = within_block(pool, Eh)
    o["block_rho_by_block"] = list(vals)
    OUT["grid"] = o
    save()

# ================================================================= dev regimes
if part == "dev":
    sc = dev_scores(Ed)
    OUT["dev"] = {"scores": sc, "choices": choices(sc)}
    tic(json.dumps(OUT["dev"]["choices"]))
    save()
    # per-block choices without the evaluated block (for the type-block target design)
    blk = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        sb = dev_scores(Ed, base_extra=H, regimes=["dyad", "block"])
        cc = {}
        c, _ = minimax(sb, ["dyad", "block"], PCC_KEYS); cc["pcc"] = c
        for bb in ["blend_rank", "blend_mu", "blend_raw"]:
            cw, _ = minimax(sb, ["dyad", "block"], [f"{bb}|{w}" for w in W_G]); cc[bb] = float(cw.split("|")[1])
        blk.append(cc)
        tic(f"block {b} {cc}")
    OUT["dev"]["block_choices_two"] = blk
    save()

# ================================================================= target regimes
def target_masks(regime):
    """Masks of the target designs, with the seeds of the paper's analyses."""
    if regime.startswith("random"):
        frac = float(regime.split("_")[1])
        return [mask_random(frac, rep, 5000) for rep in range(20)]
    if regime.startswith("cold"):
        m = int(regime.split("_")[1])
        return [masks_cold(m, rep) for rep in range(5)]          # list of reps, each a list of 24 masks
    if regime in SOURCES:
        return [OFF & ~keep_sources(SOURCES[regime])]
    if regime.startswith("thin_random"):
        frac = float(regime.split("_")[2])
        out = []
        for rep in range(20):
            g = np.random.default_rng(3100 + rep + int(100 * frac))
            out.append(OFF & ~((g.random((P, P)) < frac) & OFF))
        return out
    if regime.startswith("thin_strength"):
        q = {"thin_strength_top50": 0.5, "thin_strength_top25": 0.75}[regime]
        return [OFF & ~(Ed >= np.quantile(Ed[OFF], q))]
    raise ValueError(regime)


TARGET_REGIMES = ["random_0.25", "random_0.5", "cold_1", "cold_3", "cold_6", "sources_A", "sources_B",
                  "thin_random_0.5", "thin_random_0.25", "thin_strength_top50", "thin_strength_top25"]


def _target_one(E1, E2, H, wsets, si_seed=0):
    obs = OFF & ~H
    sc = {key(t): r(pcc(E1, obs, *t)[H], E2[H]) for t in THETAS}
    fp = fixed_preds(E1, H, si_seed)
    for k, v in fp.items():
        sc[k] = r(v[H], E2[H])
    for w in wsets:
        for k, v in blends(fp, H, w).items():
            sc[f"{k}|{w}"] = r(v, E2[H])
    return sc


def target_regime(E1, E2, regime):
    ms = target_masks(regime)
    if regime.startswith("cold"):
        jobs = [(E1, E2, H, W_G, rep) for rep, reps in enumerate(ms) for H in reps]
        res = pmap(_target_one, jobs)
        keys = res[0].keys()
        per_rep = []
        for rep in range(len(ms)):
            rows = res[rep * P:(rep + 1) * P]
            per_rep.append({k: float(np.nanmedian([x[k] for x in rows])) for k in keys})
        return {k: float(np.mean([x[k] for x in per_rep])) for k in keys}, {k: float(np.std([x[k] for x in per_rep])) for k in keys}
    jobs = [(E1, E2, H, W_G, rep) for rep, H in enumerate(ms)]
    res = pmap(_target_one, jobs)
    keys = res[0].keys()
    return ({k: float(np.nanmean([x[k] for x in res])) for k in keys},
            {k: float(np.nanstd([x[k] for x in res])) for k in keys})


def _dyad_theta_job(E1, t):
    return key(t), _pcc_dyad_theta(E1, *t)


def single_dyad_all(E1, E2):
    """Single-dyad design: every PCC setting and the fixed methods, leave-dyad-out."""
    res = dict(pmap(_dyad_theta_job, [(E1, t) for t in THETAS]))
    out = {k: r(v[OFF], E2[OFF]) for k, v in res.items()}
    fp = {"product": L.path_features(E1)["prod_max"], "hamacher": X.paths(E1, X.T_FIXED["hamacher"]),
          "katz": L.katz(E1), "rank3": X.lowrank_baseline(E1, 3, dyad=True)}
    ll, _ = L.choose_lambda(E1, LAM_SI, transform="logit")
    fp["si_logit"] = L.softimpute_dyad(E1, ll, transform="logit")
    fp["si_raw_pcclam"] = res["0.03|0.0"]
    for k, v in fp.items():
        out[k] = r(v[OFF], E2[OFF])
    for w in W_G:
        for k, v in blends(fp, OFF, w).items():
            out[f"{k}|{w}"] = r(v, E2[OFF])
    return out, fp, res


def _block_job(E1, b):
    H = OFF & (L.BLOCK == b)
    obs = OFF & ~H
    preds = {key(t): pcc(E1, obs, *t) for t in THETAS}
    preds.update(fixed_preds(E1, H))
    return b, {k: v[H] for k, v in preds.items()}


def block_all(E1, E2, block_w=None):
    """Type-block design: predictions for every block; within-block blends; pair-weighted within-block rho."""
    res = dict(pmap(_block_job, [(E1, b) for b in range(10)]))
    keys = list(res[0].keys())
    out = {}
    for k in keys:
        vals = np.array([r(res[b][k], E2[OFF & (L.BLOCK == b)]) for b in range(10)])
        out[k] = float((vals * NB).sum() / NB.sum())
    for w in W_G:
        for bk in ["blend_rank", "blend_mu", "blend_raw"]:
            vals = []
            for b in range(10):
                H = OFF & (L.BLOCK == b)
                pr = {k: np.zeros((P, P)) for k in ["product", "si_logit", "si_raw_pcclam"]}
                for k in pr:
                    pr[k][H] = res[b][k]
                vals.append(r(blends(pr, H, w)[bk], E2[H]))
            out[f"{bk}|{w}"] = float((np.array(vals) * NB).sum() / NB.sum())
    return out, res


if part == "target":
    o = {}
    tic("single dyad")
    o["dyad"], fp, _ = single_dyad_all(Ed, Eh)
    np.savez("results2/pcc2_dyad_fixed_predictions.npz", **fp)
    OUT["target"] = {"mean": o, "sd": {}}
    save()
    tic("block")
    o["block"], _ = block_all(Ed, Eh)
    save()
    for reg in TARGET_REGIMES:
        tic(reg)
        m, s = target_regime(Ed, Eh, reg)
        o[reg] = m; OUT["target"]["sd"][reg] = s
        save()

# ================================================================= JRR for blends and PCC
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
                  "ci90": [v0 - 1.645 * se, v0 + 1.645 * se],
                  "tost_p_0.03": float(max(norm.cdf(-(v0 + 0.03) / se), norm.cdf(-(0.03 - v0) / se))) if se > 0 else None,
                  "tost_p_0.05": float(max(norm.cdf(-(v0 + 0.05) / se), norm.cdf(-(0.05 - v0) / se))) if se > 0 else None,
                  "replicate_min": float(v.min()), "replicate_max": float(v.max())}
    return out


def load_setting():
    d = json.load(open("results2/pcc2_dev.json"))["dev"]
    ch = d["choices"]["two"]
    lam, mu = map(float, ch["pcc"].split("|"))
    blk = d["block_choices_two"]
    return {"pcc": (lam, mu), "w": {b: ch[b] for b in ["blend_rank", "blend_mu", "blend_raw"]},
            "block": [tuple(map(float, x["pcc"].split("|"))) for x in blk],
            "block_w": [{b: x[b] for b in ["blend_rank", "blend_mu", "blend_raw"]} for x in blk]}


def jrr_stats(E1, E2, S):
    lam, mu = S["pcc"]
    o = {}
    # single dyads
    dy = {"pcc": _pcc_dyad_theta(E1, lam, mu), "product": L.path_features(E1)["prod_max"]}
    ll, _ = L.choose_lambda(E1, LAM_SI, transform="logit")
    dy["si_logit"] = L.softimpute_dyad(E1, ll, transform="logit")
    dy["si_raw_pcclam"] = _pcc_dyad_theta(E1, lam, 0.0)
    dy["katz"] = L.katz(E1)
    dy["rank3"] = X.lowrank_baseline(E1, 3, dyad=True)
    for k in ["pcc", "product", "si_logit", "katz", "rank3"]:
        o[f"dyad_{k}"] = r(dy[k][OFF], E2[OFF])
    for b, w in S["w"].items():
        o[f"dyad_{b}"] = r(blends(dy, OFF, w)[b], E2[OFF])
    # secondary outcomes of the audit (RQ1, RQ2): directed Hamacher and minimum; symmetric product and Hamacher
    o["dyad_hamacher"] = r(X.paths(E1, X.T_FIXED["hamacher"])[OFF], E2[OFF])
    o["dyad_min"] = r(X.paths(E1, X.T_FIXED["min"])[OFF], E2[OFF])
    S1, S2 = np.sqrt(E1 * E1.T), np.sqrt(E2 * E2.T)
    iu = np.triu_indices(P, 1)
    o["sym_product"] = r(X.paths(S1, X.t_product)[iu], S2[iu])
    o["sym_hamacher"] = r(X.paths(S1, X.T_FIXED["hamacher"])[iu], S2[iu])
    o["dyad_rank3_minus_product"] = o["dyad_rank3"] - o["dyad_product"]
    o["dyad_product_minus_min"] = o["dyad_product"] - o["dyad_min"]
    o["dyad_product_minus_hamacher"] = o["dyad_product"] - o["dyad_hamacher"]
    o["sym_product_minus_hamacher"] = o["sym_product"] - o["sym_hamacher"]
    # type blocks
    pb = {k: [] for k in ["pcc", "product", "si_logit", "katz", "rank3", "blend_rank", "blend_mu", "blend_raw"]}
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        obs = OFF & ~H
        Em = np.where(obs, E1, 0.0)
        lb, mb = S["block"][b]
        pr = {"pcc": pcc(E1, obs, lb, mb), "product": X.paths(Em, X.t_product), "katz": L.katz(Em, dyad=False),
              "rank3": X.soft_impute(E1, obs, 3)}
        l2 = L.choose_lambda(E1, LAM_SI, base=obs, transform="logit")[0]
        pr["si_logit"] = L.soft_impute_reg(E1, obs, l2, transform="logit")
        pr["si_raw_pcclam"] = pcc(E1, obs, lb, 0.0)
        for k in ["pcc", "product", "si_logit", "katz", "rank3"]:
            pb[k].append(r(pr[k][H], E2[H]))
        for bk, w in S["block_w"][b].items():
            pb[bk].append(r(blends(pr, H, w)[bk], E2[H]))
    for k, v in pb.items():
        o[f"block_{k}"] = float((np.array(v) * NB).sum() / NB.sum())
    meths = ["pcc", "product", "katz", "si_logit", "rank3"]
    allm = meths + ["blend_rank", "blend_mu", "blend_raw"]
    for ref, ms in [("five", meths), ("six", meths + ["blend_rank"]), ("all", allm)]:
        bd = max(o[f"dyad_{k}"] for k in ms); bb = max(o[f"block_{k}"] for k in ms)
        for k in allm:
            o[f"regret{ref}_{k}"] = max(bd - o[f"dyad_{k}"], bb - o[f"block_{k}"])
    for k in allm[1:]:
        o[f"dyad_pcc_minus_{k}"] = o["dyad_pcc"] - o[f"dyad_{k}"]
        o[f"block_pcc_minus_{k}"] = o["block_pcc"] - o[f"block_{k}"]
        o[f"regretfive_{k}_minus_pcc"] = o[f"regretfive_{k}"] - o["regretfive_pcc"]
        o[f"regretall_{k}_minus_pcc"] = o[f"regretall_{k}"] - o["regretall_pcc"]
        o[f"regretsix_{k}_minus_pcc"] = o[f"regretsix_{k}"] - o["regretsix_pcc"]
    for k in ["product", "pcc", "katz", "si_logit", "rank3"]:
        o[f"regretsix_{k}_minus_blend_rank"] = o[f"regretsix_{k}"] - o["regretsix_blend_rank"]
        o[f"dyad_blend_rank_minus_{k}"] = o["dyad_blend_rank"] - o[f"dyad_{k}"]
        o[f"block_blend_rank_minus_{k}"] = o["block_blend_rank"] - o[f"block_{k}"]
    return o


def _jrr_dev(d, E2):
    return jrr_stats(rel(d), E2, load_setting())


def _jrr_tgt(h, E1):
    return jrr_stats(E1, rel(h), load_setting())


def run_jrr(E1, E2, d1, d2, name):
    S = load_setting()
    full = jrr_stats(E1, E2, S)
    tic(f"{name} full " + json.dumps({k: round(v, 4) for k, v in full.items()}))
    OUT[name] = {"setting": S, "full": full}
    save()
    reps = pmap(_jrr_dev, [(d, E2) for d in jrr_reps(d1)]) + pmap(_jrr_tgt, [(h, E1) for h in jrr_reps(d2)])
    OUT[name]["jrr"] = jrr_summary(full, reps)
    OUT[name]["n_replicates"] = len(reps)
    save()


if part == "jrr":
    run_jrr(Ed, Eh, dev, hold, "jrr")

if part == "jrr05":
    sys.path.insert(0, "confirmatory")
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
    run_jrr(Ed, rel(c05), dev, c05, "jrr05")

# ================================================================= diagnostics (R5)
def objective(X_, R, mask, lam, mu, t="product"):
    m = R[mask].mean()
    b, _ = TC.best_path(X_, t)
    sel = OFF & ~mask
    s = np.linalg.svd(X_ - m, compute_uv=False)
    return float(0.5 * ((X_ - R)[mask] ** 2).sum() + lam * s.sum() + 0.5 * mu * ((b - X_)[sel] ** 2).sum())


def tc_trace(R, mask, lam, mu, t="product", iters=400, tol=1e-6, X0=None):
    """tcomp.tc_complete with the objective and the argmax ties recorded at every iteration."""
    n = R.shape[0]
    T, dTa, dTb = TC._tnorm_and_grads(t)
    m = R[mask].mean()
    X_ = np.where(mask, R, m) if X0 is None else X0.copy()
    step = 1.0 / (1.0 + 3.0 * mu)
    obj, ties = [objective(X_, R, mask, lam, mu, t)], []
    it_done = iters
    for it in range(iters):
        G = np.where(mask, X_ - R, 0.0)
        if mu > 0:
            b, k = TC.best_path(X_, t)
            sel = OFF & ~mask
            phi = np.where(sel, b - X_, 0.0)
            V = np.where(TC._triple_mask(n), T(X_[:, :, None], X_[None, :, :]), -1.0)
            top2 = np.sort(V, axis=1)[:, -2:, :]
            ties.append(float(((top2[:, 1, :] - top2[:, 0, :]) < 1e-9)[sel].mean()))
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
        done = np.abs(Xn - X_).max() < tol
        X_ = Xn
        obj.append(objective(X_, R, mask, lam, mu, t))
        if done:
            it_done = it + 1
            break
    X_ = X_.copy(); np.fill_diagonal(X_, 0.0)
    return X_, it_done, obj, ties


def _diag_dyad(i, j, lam, mu, seeds):
    m = OFF.copy(); m[i, j] = m[j, i] = False
    Xa, it, obj, ties = tc_trace(Ed, m, lam, mu, iters=300)
    dif = np.diff(obj)
    o = {"iters": it, "cap": it >= 300, "monotone": bool((dif <= 1e-9 * max(1, abs(obj[0]))).all()),
         "max_increase": float(max(0.0, dif.max())), "ties_mean": float(np.mean(ties)) if ties else 0.0,
         "obs_shift_median": float(np.median(np.abs(Xa - Ed)[m])), "obs_shift_max": float(np.abs(Xa - Ed)[m].max()),
         "pred": [float(Xa[i, j]), float(Xa[j, i])], "obj": obj[-1]}
    alt = []
    for s in seeds:
        g = np.random.default_rng(s)
        X0 = np.where(m, Ed, g.uniform(0, Ed[m].max(), (P, P))); np.fill_diagonal(X0, 0.0)
        Xb = TC.tc_complete(Ed, m, lam, mu, "product", 300, X0=X0, two_sided=True, cells="missing")
        alt.append([float(Xb[i, j]), float(Xb[j, i])])
    o["pred_random_starts"] = alt
    return o


if part == "diag":
    S = load_setting()
    lam, mu = S["pcc"]
    o = {"setting": [lam, mu]}
    # (1) observed cells move: single-dyad design at the chosen setting and at gamma = 0
    rows = pmap(_diag_dyad, [(i, j, lam, mu, [11, 12, 13]) for i, j in DYADS])
    o["dyad"] = {"iters_median": float(np.median([x["iters"] for x in rows])),
                 "iters_q90": float(np.quantile([x["iters"] for x in rows], 0.9)),
                 "share_cap": float(np.mean([x["cap"] for x in rows])),
                 "share_monotone": float(np.mean([x["monotone"] for x in rows])),
                 "max_increase": float(max(x["max_increase"] for x in rows)),
                 "ties_share_mean": float(np.mean([x["ties_mean"] for x in rows])),
                 "obs_shift_median": float(np.median([x["obs_shift_median"] for x in rows])),
                 "obs_shift_max": float(max(x["obs_shift_max"] for x in rows))}
    base = np.array([x["pred"] for x in rows]).ravel()
    alt = np.array([x["pred_random_starts"] for x in rows])     # (276, 3, 2)
    o["dyad"]["random_start_max_abs_diff"] = float(np.abs(alt.reshape(len(rows), 3, 2) - base.reshape(len(rows), 1, 2)).max())
    o["dyad"]["random_start_rho_with_fixed_start"] = [r(alt[:, s, :].ravel(), base) for s in range(3)]
    tgt = np.array([[Eh[i, j], Eh[j, i]] for i, j in DYADS]).ravel()
    o["dyad"]["random_start_heldout_rho"] = [r(alt[:, s, :].ravel(), tgt) for s in range(3)]
    o["dyad"]["fixed_start_heldout_rho"] = r(base, tgt)
    m0 = OFF.copy(); m0[0, 1] = m0[1, 0] = False
    X0g = TC.tc_complete(Ed, m0, lam, 0.0, "product", 300, two_sided=True, cells="missing")
    o["dyad"]["obs_shift_median_gamma0_example"] = float(np.median(np.abs(X0g - Ed)[m0]))
    save()
    tic("dyad diagnostics done")
    # (2) type blocks: gamma = 20 against the product path; observed shift; convergence
    blk = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        obs = OFF & ~H
        Em = np.where(obs, Ed, 0.0)
        row = {"block": b}
        for g_ in [S["block"][b][1], 20.0, 100.0]:
            Xb, it, obj, ties = tc_trace(Ed, obs, S["block"][b][0], g_, iters=2000 if g_ >= 100 else 400)
            two, _ = TC.best_path(Em, "product")
            p23 = X.paths(Em, X.t_product)
            row[f"g{g_}"] = {"iters": it, "monotone": bool((np.diff(obj) <= 1e-9 * max(1, abs(obj[0]))).all()),
                            "rho_pcc_vs_path2": r(Xb[H], two[H]), "rho_pcc_vs_path23": r(Xb[H], p23[H]),
                            "rho_heldout": r(Xb[H], Eh[H]), "obs_shift_median": float(np.median(np.abs(Xb - Ed)[obs])),
                            "share_best_path_3_edges": float((p23[H] > two[H] + 1e-12).mean())}
        row["rho_path2_heldout"] = r(TC.best_path(Em, "product")[0][H], Eh[H])
        row["rho_path23_heldout"] = r(X.paths(Em, X.t_product)[H], Eh[H])
        blk.append(row)
    o["block"] = blk
    agg = {}
    for lab in ["g20.0", "g100.0"]:
        for k in ["rho_pcc_vs_path2", "rho_pcc_vs_path23", "rho_heldout", "obs_shift_median"]:
            v = np.array([x[lab][k] for x in blk])
            agg[f"{lab}_{k}_weighted"] = float((v * NB).sum() / NB.sum())
    agg["path2_heldout_weighted"] = float((np.array([x["rho_path2_heldout"] for x in blk]) * NB).sum() / NB.sum())
    agg["path23_heldout_weighted"] = float((np.array([x["rho_path23_heldout"] for x in blk]) * NB).sum() / NB.sum())
    agg["chosen_obs_shift_median"] = float(np.median([x[f"g{S['block'][b][1]}"]["obs_shift_median"] for b, x in enumerate(blk)]))
    o["block_summary"] = agg
    OUT["diag"] = o
    save()

# ================================================================= rejected variants
if part == "variants":
    def var_fit(R, mask, lam, mu, v):
        if v == "pcc":
            return pcc(R, mask, lam, mu)
        if v == "onesided_all":
            return TC.tc_complete(R, mask, lam, mu, "product", 400, two_sided=False, cells="all")
        if v == "logit_lowrank":
            return TC.pcc_logit(R, mask, lam, mu)
    o = {}
    for v in ["pcc", "onesided_all", "logit_lowrank"]:
        lams = LAM_G if v != "logit_lowrank" else LAM_SI
        sc = {}
        for reg in ["dyad", "block"]:
            rows = []
            for H, g in dev_masks(reg):
                obs = OFF & ~H
                rows.append(({f"{l}|{m}": r(var_fit(Ed, obs, l, m, v)[H], Ed[H]) for l in lams for m in MU_G},
                             int(H.sum()), g))
            sc[reg] = {k: regime_score(reg, [(x[k], n, g) for x, n, g in rows]) for k in rows[0][0]}
        c, rg = minimax(sc, ["dyad", "block"], list(sc["dyad"].keys()))
        lam, mu = map(float, c.split("|"))
        tgt_block = np.zeros_like(Ed)
        for b in range(10):
            H = OFF & (L.BLOCK == b)
            tgt_block[H] = var_fit(Ed, OFF & ~H, lam, mu, v)[H]
        o[v] = {"cv": sc, "minimax": c, "minimax_regret_dev": rg[c],
                "best_dev_block": max(sc["block"].values()), "best_dev_dyad": max(sc["dyad"].values()),
                "target_block_weighted": within_block(tgt_block, Eh)[0]}
        tic(f"{v}: {c} dev-regret {rg[c]:.3f} best block {o[v]['best_dev_block']:.3f}")
        OUT["variants"] = o
        save()

# ================================================================= t-norm inside PCC on the symmetric relation
if part == "tnormsym":
    S = load_setting()
    lam, mu = S["pcc"]

    def sym(E):
        return np.sqrt(E * E.T)
    Sd, Sh = sym(Ed), sym(Eh)
    iu = np.triu_indices(P, 1)
    o = {}
    for t in ["product", "hamacher", "min"]:
        out = np.zeros_like(Sd)
        for i, j in DYADS:
            m = OFF.copy(); m[i, j] = m[j, i] = False
            Z = pcc(Sd, m, lam, mu, t, iters=300)
            out[i, j] = out[j, i] = 0.5 * (Z[i, j] + Z[j, i])
        o[t] = r(out[iu], Sh[iu])
        tic(f"{t} {o[t]:.4f}")
    o["path_product_sym"] = r(X.paths(Sd, X.t_product)[iu], Sh[iu])
    o["path_hamacher_sym"] = r(X.paths(Sd, X.T_FIXED["hamacher"])[iu], Sh[iu])
    OUT["tnormsym"] = o
    save()

# ================================================================= run time
if part == "runtime":
    o = {}
    for n in [24, 100, 200, 400]:
        g = np.random.default_rng(n)
        F = g.uniform(0, 1, (n, 3))
        R = np.clip(F @ F.T / 3 + g.normal(0, 0.03, (n, n)), 0, 1); np.fill_diagonal(R, 0)
        mask = ~np.eye(n, dtype=bool)
        H = g.random((n, n)) < 0.1
        H = H | H.T
        mask &= ~H
        if n == 24:
            R, mask = Ed, OFF.copy(); mask[0, 1] = mask[1, 0] = False
        TC._MASKS.clear()
        t0 = time.time(); TC.tc_complete(R, mask, 0.03, 2.0, "product", 50, tol=0.0, two_sided=True, cells="missing")
        o[str(n)] = {"seconds_per_iteration": (time.time() - t0) / 50}
        tic(f"n={n}: {o[str(n)]}")
    OUT["runtime"] = o
    save()

# ================================================================= 2021-2023 target
from nhanes_l import load_c21


if part == "c21":
    c21, flow = load_c21()
    E21 = rel(c21)
    S = load_setting()
    o = {"flow": flow, "n": int(len(c21)), "n_strata": int(c21.strata.nunique()),
         "corr_dev": r(Ed[OFF], E21[OFF]), "corr_2007": r(Eh[OFF], E21[OFF])}
    full = jrr_stats(Ed, E21, S)
    o["full"] = full
    OUT["c21"] = o
    save()
    tic("c21 full " + json.dumps({k: round(v, 4) for k, v in full.items()}))
    reps = pmap(_jrr_dev, [(d, E21) for d in jrr_reps(dev)]) + pmap(_jrr_tgt, [(h, Ed) for h in jrr_reps(c21)])
    o["jrr"] = jrr_summary(full, reps)
    o["n_replicates"] = len(reps)
    save()

if part == "c21regimes":
    c21, _ = load_c21()
    E21 = rel(c21)
    o = {}
    for reg in ["random_0.25", "random_0.5", "cold_1", "cold_3", "cold_6", "sources_A", "sources_B"]:
        m, s = target_regime(Ed, E21, reg)
        o[reg] = m
        tic(reg)
        OUT["c21regimes"] = o
        save()

# ================================================================= lower boundary of the lambda grid
if part == "grid0":
    lams0 = [0.0, 0.0003, 0.001]
    jobs, tags = [], []
    for reg in ["dyad", "block"]:
        for H, g in dev_masks(reg):
            jobs.append((Ed, H)); tags.append((reg, g))

    def _one(E1, H):
        obs = OFF & ~H
        return {f"{l}|{m}": r(pcc(E1, obs, l, m)[H], E1[H]) for l in lams0 for m in MU_G}, int(H.sum())
    res = pmap(_one, jobs)
    sc = {}
    for reg in ["dyad", "block"]:
        rows = [(s, n, g) for (s, n), (rg, g) in zip(res, tags) if rg == reg]
        sc[reg] = {k: regime_score(reg, [(s[k], n, g) for s, n, g in rows]) for k in rows[0][0]}
    prev = json.load(open("results2/pcc2_grid.json"))["grid"]
    for reg, kk in [("dyad", "cv_dyad"), ("block", "cv_block")]:
        sc[reg].update(prev[kk])
    c, rg = minimax(sc, ["dyad", "block"], list(sc["dyad"].keys()))
    OUT["grid0"] = {"cv_dyad": sc["dyad"], "cv_block": sc["block"], "chosen": c, "regret": rg}
    tic(f"grid with lambda down to 0: {c} {rg[c]:.4f}")
    save()
