"""Generalization and inference checks for the composition-versus-completion audit (v1.6.0).
Parts:
  sym        - symmetric (276-dyad) evaluation in temporal, split-half and swapped designs with JRR intervals;
               prevalence-partialled Spearman on the directed relation; directed predictions symmetrized
  holm       - JRR intervals and Holm-adjusted p-values for every predictor in Table II against the product;
               precision at K = 20 and 50; rho on positive target pairs; development matrix as its own target
  measci     - JRR intervals for the differences on the alternative association measures
  sparse     - sparsified development graphs (top 50%, top 25%, mu >= 0.25): t-norm ranking, paths against
               completion, best-path lengths and path counts
  missing    - random withholding of 10%, 25%, 50% and 75% of dyads
  coldstart  - one concept at a time with only m anchor dyads known (m = 1, 3, 6)
  blockjrr   - withheld relation types: JRR intervals per block and pooled; ranks 2 and 4
  query      - clinical query: JRR, tie handling, direct-edge ceiling, effective sample sizes
  subgroup   - temporal design within age and sex groups
  partner    - never-married respondents scored 0 or 1 on 'no partner'; split halves within 2015-2018
Writes results2/gen_<part>.json."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, norm
from analysis2 import dev, hold
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
import rq3x as X
import rq6lib as L

part = sys.argv[1]
rng = np.random.default_rng(20261116)
OFF = ~np.eye(P, dtype=bool)
UP = np.triu(OFF, 1)
OUT = {}
LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]
TNORMS = ["product", "hamacher", "min", "yager2", "lukasiewicz"]
MODIFIABLE = ["smoking", "heavy_drinking", "inactivity", "short_sleep", "obesity"]
SYMPTOMS = [c for c in CONCEPTS if TYPE[c] == "Symptom"]


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/gen_{part}.json", "w"), indent=1, default=float)


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


def split_halves(df, n, seed):
    g = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = []
        for h, gg in df.groupby("strata"):
            ps = gg.psu.unique()
            a += list(g.permutation(ps)[: max(1, len(ps) // 2)])
        A = df[df.psu.isin(a)].copy(); B = df[~df.psu.isin(a)].copy()
        A["w"] *= df.w.sum() / A.w.sum(); B["w"] *= df.w.sum() / B.w.sum()
        out.append((A, B))
    return out


def jrr_summary(full, reps):
    """full: dict of statistics; reps: list of dicts. JK2 variance = sum of squared deviations."""
    out = {}
    for k in full:
        v = np.array([x[k] for x in reps], float)
        se = float(np.sqrt(((v - full[k]) ** 2).sum()))
        out[k] = {"point": float(full[k]), "se": se, "ci95": [full[k] - 1.96 * se, full[k] + 1.96 * se]}
    return out


def prec_at(pred, target, mask, K=None, q=0.75):
    y, s = target[mask], pred[mask]
    pos = y >= np.quantile(y, q)
    K = int(pos.sum()) if K is None else K
    top = np.argsort(-s, kind="stable")[:K]
    return float(pos[top].mean())


def soft_mask_dyad(R, k, base):
    """Rank-k completion, leave-dyad-out, when only `base` cells are observed."""
    out = np.zeros_like(R)
    cache = {}
    for i, j in itertools.permutations(range(P), 2):
        key = (min(i, j), max(i, j))
        if key not in cache:
            m = base.copy(); m[i, j] = m[j, i] = False
            cache[key] = X.soft_impute(R, m, k)
        out[i, j] = cache[key][i, j]
    return out


def si_mask_dyad(R, lam, base, transform=None):
    out = np.zeros_like(R)
    for i in range(P):
        for j in range(i + 1, P):
            m = base.copy(); m[i, j] = m[j, i] = False
            F = L.soft_impute_reg(R, m, lam, transform=transform)
            out[i, j], out[j, i] = F[i, j], F[j, i]
    return out


Ed, Eh = rel(dev), rel(hold)
pdev, phold = prevalence(dev), prevalence(hold)
LAM_LOGIT = L.choose_lambda(Ed, LAM_GRID, transform="logit")[0]


# ------------------------------------------------------------------------------------------------
if part == "sym":
    def sym(E):
        return np.sqrt(E * E.T)

    def sym_stats(E1, E2, lam):
        S1, S2 = sym(E1), sym(E2)
        o = {T: r(X.paths(S1, X.T_FIXED[T])[UP], S2[UP]) for T in TNORMS}
        o["rank3"] = r(X.lowrank_baseline(S1, 3, dyad=True)[UP], S2[UP])
        o["softimpute"] = r(L.softimpute_dyad(S1, lam)[UP], S2[UP])
        o["product_minus_hamacher"] = o["product"] - o["hamacher"]
        o["rank3_minus_product"] = o["rank3"] - o["product"]
        o["softimpute_minus_product"] = o["softimpute"] - o["product"]
        return o

    lam = L.choose_lambda(sym(Ed), LAM_GRID)[0]
    res = {"lambda": lam}
    full = sym_stats(Ed, Eh, lam)
    res["temporal"] = full
    res["swapped"] = sym_stats(Eh, Ed, lam)
    tic("split halves")
    sh = [sym_stats(rel(A), rel(B), lam) for A, B in split_halves(dev, 20, 4)]
    res["split_half_mean"] = {k: float(np.mean([x[k] for x in sh])) for k in sh[0]}
    res["split_half_product_first"] = int(sum(x["product"] >= max(x[T] for T in TNORMS) for x in sh))
    res["split_half_hamacher_first"] = int(sum(x["hamacher"] >= max(x[T] for T in TNORMS) for x in sh))
    res["split_half_product_minus_hamacher_positive"] = int(sum(x["product_minus_hamacher"] > 0 for x in sh))
    tic("JRR")
    reps = [sym_stats(rel(d), Eh, lam) for d in jrr_reps(dev)] + [sym_stats(Ed, rel(h), lam) for h in jrr_reps(hold)]
    res["temporal_jrr"] = jrr_summary(full, reps)
    # directed predictions, symmetrized by geometric mean, against the symmetric target
    Sh = sym(Eh)
    dpred = {T: X.paths(Ed, X.T_FIXED[T]) for T in TNORMS}
    dpred["rank3"] = X.lowrank_baseline(Ed, 3, dyad=True)
    res["directed_symmetrized"] = {k: r(np.sqrt(v * v.T)[UP], Sh[UP]) for k, v in dpred.items()}
    # prevalence-partialled Spearman on the directed relation
    lg = np.log(pdev / (1 - pdev))
    Z = np.vstack([rankdata(np.broadcast_to(lg[:, None], (P, P))[OFF]), rankdata(np.broadcast_to(lg[None, :], (P, P))[OFF])]).T
    Z = np.column_stack([np.ones(len(Z)), Z])

    def resid(y):
        return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    yt = resid(rankdata(Eh[OFF]))
    res["directed_partial_prevalence"] = {k: float(np.corrcoef(resid(rankdata(v[OFF])), yt)[0, 1]) for k, v in dpred.items()}
    res["target_rank_r2_on_prevalence"] = float(1 - (yt ** 2).sum() / ((rankdata(Eh[OFF]) - rankdata(Eh[OFF]).mean()) ** 2).sum())
    OUT["sym"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "holm":
    def preds(E):
        f = L.path_features(E)
        o = {"product": f["prod_max"], "hamacher": f["ham_max"], "min": f["min_max"],
             "yager2": X.paths(E, X.T_FIXED["yager2"]), "lukasiewicz": X.paths(E, X.T_FIXED["lukasiewicz"]),
             "resource_alloc": f["resource_alloc"], "adamic_adar": f["adamic_adar"], "katz": L.katz(E),
             "rank3": X.lowrank_baseline(E, 3, dyad=True),
             "softimpute_logit": L.softimpute_dyad(E, LAM_LOGIT, transform="logit")}
        return o

    def stats(E1, E2):
        pr = preds(E1)
        o = {k: r(v[OFF], E2[OFF]) for k, v in pr.items()}
        for k in list(pr):
            if k != "product":
                o[f"{k}_minus_product"] = o[k] - o["product"]
        return o, pr

    full, pr = stats(Ed, Eh)
    res = {"lambda_logit": LAM_LOGIT}
    tic("JRR")
    reps = [stats(rel(d), Eh)[0] for d in jrr_reps(dev)] + [stats(Ed, rel(h))[0] for h in jrr_reps(hold)]
    js = jrr_summary(full, reps)
    diffs = [k for k in full if k.endswith("_minus_product")]
    pv = {k: float(2 * norm.sf(abs(js[k]["point"]) / js[k]["se"])) for k in diffs}
    order = sorted(diffs, key=lambda k: pv[k])
    holm, run = {}, 0.0
    for q, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - q) * pv[k]))
        holm[k] = run
    res["jrr"] = js
    res["p_raw"] = pv
    res["p_holm"] = holm
    res["prec_at"] = {k: {"K20": prec_at(v, Eh, OFF, 20), "K50": prec_at(v, Eh, OFF, 50), "K138": prec_at(v, Eh, OFF)}
                      for k, v in pr.items()}
    pos = OFF & (Eh >= 0.01)
    res["rho_target_positive"] = {"n_pairs": int(pos.sum()), **{k: r(v[pos], Eh[pos]) for k, v in pr.items()}}
    res["n_target_zero"] = int((Eh[OFF] <= 0).sum())
    res["internal_target_development"] = {k: r(v[OFF], Ed[OFF]) for k, v in pr.items()}
    OUT["holm"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "measci":
    from audit_extensions import wcorr, yule_q, glasso_partial

    def build(df, kind):
        if kind == "cf_sym":
            E = rel(df); return np.sqrt(E * E.T)
        if kind == "pearson":
            return np.clip(wcorr(df, CONCEPTS), 0, 1) * OFF
        if kind == "yule":
            return np.clip(yule_q(df, CONCEPTS), 0, 1) * OFF
        return np.clip(glasso_partial(df, CONCEPTS)[0], 0, 1) * OFF

    def stats(M1, M2, lam, with_si):
        o = {T: r(X.paths(M1, X.T_FIXED[T])[UP], M2[UP]) for T in ["product", "hamacher", "min"]}
        o["rank3"] = r(X.lowrank_baseline(M1, 3, dyad=True)[UP], M2[UP])
        o["product_minus_hamacher"] = o["product"] - o["hamacher"]
        o["rank3_minus_product"] = o["rank3"] - o["product"]
        if with_si:
            o["softimpute"] = r(L.softimpute_dyad(M1, lam)[UP], M2[UP])
            o["softimpute_minus_product"] = o["softimpute"] - o["product"]
        return o

    res = {}
    for kind in ["cf_sym", "pearson", "yule", "partial"]:
        tic(kind)
        Md, Mh = build(dev, kind), build(hold, kind)
        lam = L.choose_lambda(Md, LAM_GRID)[0]
        si = kind == "partial"
        full = stats(Md, Mh, lam, si)
        reps = [stats(build(d, kind), Mh, lam, si) for d in jrr_reps(dev)] + \
               [stats(Md, build(h, kind), lam, si) for h in jrr_reps(hold)]
        res[kind] = jrr_summary(full, reps)
        OUT["measci"] = res
        save()

# ------------------------------------------------------------------------------------------------
if part == "sparse":
    q50, q75 = np.quantile(Ed[OFF], [0.5, 0.75])
    levels = {"full": Ed > 0, "top50": Ed >= q50, "top25": Ed >= q75, "mu_ge_0.25": Ed >= 0.25}
    res = {}
    I = np.arange(P)
    for name, keep in levels.items():
        tic(name)
        keep = keep & OFF
        A = np.where(keep, Ed, 0.0)
        o = {"n_edges": int(keep.sum()), "share_pairs_with_edge": float(keep[OFF].mean())}
        pr = {T: X.paths(A, X.T_FIXED[T]) for T in TNORMS}
        pr["katz"] = L.katz(A)
        pr["product_len4"] = L.best_path_product(A, max_len=4)
        # completion: absent edges treated as unobserved, or as observed zeros
        pr["rank3_absent_missing"] = soft_mask_dyad(Ed, 3, keep)
        pr["rank3_absent_zero"] = X.lowrank_baseline(A, 3, dyad=True)
        try:
            lam = L.choose_lambda(Ed, LAM_GRID, base=keep)[0]
            pr["softimpute_absent_missing"] = si_mask_dyad(Ed, lam, keep)
        except Exception as e:
            o["softimpute_error"] = str(e)
        unrec = OFF & ~keep
        o["rho_all"] = {k: r(v[OFF], Eh[OFF]) for k, v in pr.items()}
        o["rho_unrecorded"] = {k: r(v[unrec], Eh[unrec]) for k, v in pr.items()} if unrec.sum() > 10 else None
        o["prec_all"] = {k: prec_at(v, Eh, OFF) for k, v in pr.items()}
        o["n_unrecorded"] = int(unrec.sum())
        # path structure under the product
        V2 = np.where(X._MASK[P][0], A[:, :, None] * A[None, :, :], 0.0)
        P2 = A[:, :, None] * A[None, :, :]
        V3 = np.where(X._MASK[P][1], P2[:, :, :, None] * A[None, None, :, :], 0.0)
        b2, b3 = V2.max(1), V3.max((1, 2))
        n2, n3 = (V2 > 0).sum(1), (V3 > 0).sum((1, 2))
        has = (np.maximum(b2, b3) > 0) & OFF
        o["share_pairs_no_path"] = float(1 - has[OFF].mean())
        o["share_best_path_3_edges"] = float((b3[has] > b2[has]).mean())
        o["share_best_path_4_edges"] = float((pr["product_len4"][has] > np.maximum(b2, b3)[has] + 1e-15).mean())
        o["median_n_paths_2"] = float(np.median(n2[OFF])); o["median_n_paths_3"] = float(np.median(n3[OFF]))
        o["share_pairs_no_2edge_path"] = float((n2[OFF] == 0).mean())
        o["product_first_tnorm"] = bool(o["rho_all"]["product"] >= max(o["rho_all"][T] for T in TNORMS))
        res[name] = o
        OUT["sparse"] = res
        save()

# ------------------------------------------------------------------------------------------------
if part == "missing":
    dyads = [(i, j) for i in range(P) for j in range(i + 1, P)]
    res = {}
    for frac in [0.10, 0.25, 0.50, 0.75]:
        tic(f"missing {frac}")
        rows = []
        for rep in range(100):
            g = np.random.default_rng(7000 + rep + int(frac * 1000))
            pick = g.choice(len(dyads), int(round(frac * len(dyads))), replace=False)
            H = np.zeros((P, P), bool)
            for q in pick:
                i, j = dyads[q]; H[i, j] = H[j, i] = True
            obs = OFF & ~H
            A = np.where(obs, Ed, 0.0)
            pr = {T: X.paths(A, X.T_FIXED[T]) for T in ["product", "hamacher", "min"]}
            pr["katz"] = L.katz(A, dyad=False)
            pr["rank3"] = X.soft_impute(Ed, obs, 3)
            lam = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep)[0]
            pr["softimpute_cv"] = L.soft_impute_reg(Ed, obs, lam)
            rows.append({**{k: r(v[H], Eh[H]) for k, v in pr.items()},
                         **{f"prec_{k}": prec_at(v, Eh, H) for k, v in pr.items()}})
        res[str(frac)] = {k: {"mean": float(np.nanmean([x[k] for x in rows])), "sd": float(np.nanstd([x[k] for x in rows]))}
                          for k in rows[0]}
        res[str(frac)]["share_rank3_above_product"] = float(np.mean([x["rank3"] > x["product"] for x in rows]))
        res[str(frac)]["share_softimpute_above_product"] = float(np.mean([x["softimpute_cv"] > x["product"] for x in rows]))
        OUT["missing"] = res
        save()

# ------------------------------------------------------------------------------------------------
if part == "coldstart":
    res = {}
    for m in [1, 3, 6]:
        tic(f"anchors {m}")
        reps = []
        for rep in range(10):
            g = np.random.default_rng(900 + 31 * m + rep)
            preds = {k: [] for k in ["product", "hamacher", "min", "katz", "rank3", "softimpute_cv"]}
            tgt, per = [], {k: [] for k in preds}
            for c in range(P):
                others = [j for j in range(P) if j != c]
                anchors = set(g.choice(others, m, replace=False).tolist())
                H = np.zeros((P, P), bool)
                for j in others:
                    if j not in anchors:
                        H[c, j] = H[j, c] = True
                obs = OFF & ~H
                A = np.where(obs, Ed, 0.0)
                pr = {T: X.paths(A, X.T_FIXED[T]) for T in ["product", "hamacher", "min"]}
                pr["katz"] = L.katz(A, dyad=False)
                pr["rank3"] = X.soft_impute(Ed, obs, 3)
                lam = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep)[0]
                pr["softimpute_cv"] = L.soft_impute_reg(Ed, obs, lam)
                for k in preds:
                    preds[k] += list(pr[k][H]); per[k].append(r(pr[k][H], Eh[H]))
                tgt += list(Eh[H])
            row = {k: r(np.array(v), np.array(tgt)) for k, v in preds.items()}
            row.update({f"median_per_concept_{k}": float(np.nanmedian(v)) for k, v in per.items()})
            row.update({f"product_better_concepts_vs_{k}": int(sum(a > b for a, b in zip(per["product"], per[k])))
                        for k in ["rank3", "softimpute_cv", "katz", "hamacher", "min"]})
            reps.append(row)
        res[str(m)] = {k: float(np.nanmean([x[k] for x in reps])) for k in reps[0]}
        OUT["coldstart"] = res
        save()

# ------------------------------------------------------------------------------------------------
if part == "blockjrr":
    def block_preds(E1, ks=(3,), with_si=True):
        pooled = {k: np.zeros_like(E1) for k in ["product", "hamacher", "katz"] + [f"rank{k}" for k in ks] + (["softimpute_cv"] if with_si else [])}
        for b in range(10):
            H = OFF & (L.BLOCK == b)
            Em = np.where(H, 0.0, E1)
            pooled["product"][H] = X.paths(Em, X.t_product)[H]
            pooled["hamacher"][H] = X.paths(Em, X.T_FIXED["hamacher"])[H]
            pooled["katz"][H] = L.katz(Em, dyad=False)[H]
            for k in ks:
                pooled[f"rank{k}"][H] = X.soft_impute(E1, OFF & ~H, k)[H]
            if with_si:
                lb = L.choose_lambda(E1, LAM_GRID, base=OFF & ~H)[0]
                pooled["softimpute_cv"][H] = L.soft_impute_reg(E1, OFF & ~H, lb)[H]
        return pooled

    def stats(E1, E2):
        pp = block_preds(E1)
        o = {}
        nb = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
        for k, v in pp.items():
            pb = np.array([r(v[OFF & (L.BLOCK == b)], E2[OFF & (L.BLOCK == b)]) for b in range(10)])
            o[f"pooled_{k}"] = r(v[OFF], E2[OFF])
            o[f"weighted_mean_{k}"] = float((pb * nb).sum() / nb.sum())
            for b in range(10):
                o[f"block{b}_{k}"] = float(pb[b])
        for k in ["rank3", "softimpute_cv", "katz", "hamacher"]:
            o[f"pooled_product_minus_{k}"] = o["pooled_product"] - o[f"pooled_{k}"]
            o[f"weighted_product_minus_{k}"] = o["weighted_mean_product"] - o[f"weighted_mean_{k}"]
            for b in range(10):
                o[f"block{b}_product_minus_{k}"] = o[f"block{b}_product"] - o[f"block{b}_{k}"]
        return o

    full = stats(Ed, Eh)
    tic("JRR")
    reps = [stats(rel(d), Eh) for d in jrr_reps(dev)] + [stats(Ed, rel(h)) for h in jrr_reps(hold)]
    res = {"block_names": L.BLOCK_NAMES, "jrr": jrr_summary(full, reps)}
    pr = block_preds(Ed, ks=(1, 2, 3, 4, 5), with_si=False)
    res["ranks"] = {f"rank{k}": {"median_block_rho": float(np.median([r(pr[f"rank{k}"][OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) for b in range(10)])),
                                 "pooled_rho": r(pr[f"rank{k}"][OFF], Eh[OFF]),
                                 "product_better_blocks": int(sum(r(pr["product"][OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) >
                                                                  r(pr[f"rank{k}"][OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) for b in range(10))),
                                 "min_block_rho": float(np.min([r(pr[f"rank{k}"][OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) for b in range(10)]))}
                    for k in (1, 2, 3, 4, 5)}
    OUT["blockjrr"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "query":
    ci = {c: CONCEPTS.index(c) for c in CONCEPTS}
    mod = [ci[c] for c in MODIFIABLE]

    def credit(p, t):
        """Top-1 credit with ties in the prediction shared: 1/m if the target's top factor is among m tied maxima."""
        top = np.where(np.isclose(p, p.max(), rtol=0, atol=1e-12))[0]
        return (1.0 / len(top)) if int(np.argmax(t)) in top else 0.0

    def stable_first(p, t):
        return float(int(np.argsort(-p, kind="stable")[0]) == int(np.argmax(t)))

    def qstats(E1, E2):
        pr = {"product": X.paths(E1, X.t_product), "hamacher": X.paths(E1, X.T_FIXED["hamacher"]),
              "min": X.paths(E1, X.t_min), "rank3": X.lowrank_baseline(E1, 3, dyad=True), "direct_edge": E1}
        o = {}
        for k, M in pr.items():
            cr, st, nt = 0.0, 0.0, 0
            for s in SYMPTOMS:
                j = ci[s]
                p = np.array([M[i, j] for i in mod]); t = np.array([E2[i, j] for i in mod])
                cr += credit(p, t); st += stable_first(p, t)
                nt += int(np.isclose(p, p.max(), rtol=0, atol=1e-12).sum() > 1)
            o[f"{k}_tieshared"] = cr; o[f"{k}_stable"] = st; o[f"{k}_n_symptoms_tied_top"] = nt
        o["product_minus_min_tieshared"] = o["product_tieshared"] - o["min_tieshared"]
        return o

    full = qstats(Ed, Eh)
    tic("JRR")
    reps = [qstats(rel(d), Eh) for d in jrr_reps(dev)] + [qstats(Ed, rel(h)) for h in jrr_reps(hold)]
    res = {"full": full, "jrr": jrr_summary(full, reps),
           "replicate_means": {k: float(np.mean([x[k] for x in reps])) for k in full},
           "share_replicates_product_above_min": float(np.mean([x["product_tieshared"] > x["min_tieshared"] for x in reps]))}
    # Kish effective sample sizes of the queried pairs
    neff = {}
    for s in SYMPTOMS:
        for c in MODIFIABLE:
            m = dev[c].notna() & dev[s].notna()
            z = (dev.w[m] * np.minimum(dev[c][m], dev[s][m])).to_numpy()
            neff[f"{c}->{s}"] = float(z.sum() ** 2 / (z ** 2).sum()) if (z ** 2).sum() > 0 else 0.0
    res["neff"] = neff
    res["neff_self_harm"] = {k: v for k, v in neff.items() if k.endswith("self_harm_thoughts")}
    res["neff_range_all"] = [min(neff.values()), float(np.median(list(neff.values()))), max(neff.values())]
    OUT["query"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "subgroup":
    demo = pd.concat([pd.read_sas(f"raw/DEMO_{s}.xpt", format="xport").set_index("SEQN")[["RIAGENDR"]] for s in "EFGHIJ"])
    dd, hh = dev.join(demo, how="left"), hold.join(demo, how="left")
    groups = {"age_18_44": lambda d: d.RIDAGEYR < 45, "age_45_plus": lambda d: d.RIDAGEYR >= 45,
              "men": lambda d: d.RIAGENDR == 1, "women": lambda d: d.RIAGENDR == 2}
    res = {}
    for g, f in groups.items():
        tic(g)
        a, b = dd[f(dd)].drop(columns="RIAGENDR"), hh[f(hh)].drop(columns="RIAGENDR")
        E1, E2 = rel(a), rel(b)
        o = {"n_dev": int(len(a)), "n_hold": int(len(b)), "between_period_rho": r(E1[OFF], E2[OFF]),
             "corr_with_full_dev": r(E1[OFF], Ed[OFF])}
        o.update({T: r(X.paths(E1, X.T_FIXED[T])[OFF], E2[OFF]) for T in TNORMS})
        o["rank3"] = r(X.lowrank_baseline(E1, 3, dyad=True)[OFF], E2[OFF])
        o["softimpute_logit"] = r(L.softimpute_dyad(E1, LAM_LOGIT, transform="logit")[OFF], E2[OFF])
        o["product_first_tnorm"] = bool(o["product"] >= max(o[T] for T in TNORMS))
        S1, S2 = np.sqrt(E1 * E1.T), np.sqrt(E2 * E2.T)
        o["sym_product"] = r(X.paths(S1, X.t_product)[UP], S2[UP]); o["sym_hamacher"] = r(X.paths(S1, X.T_FIXED["hamacher"])[UP], S2[UP])
        res[g] = o
        OUT["subgroup"] = res
        save()

# ------------------------------------------------------------------------------------------------
if part == "partner":
    demo = pd.concat([pd.read_sas(f"raw/DEMO_{s}.xpt", format="xport").set_index("SEQN")[["DMDMARTL"]] for s in "EFGHIJ"])
    res = {}
    for val in [0.0, 1.0]:
        tic(f"never married = {val}")
        d2, h2 = dev.join(demo, how="left"), hold.join(demo, how="left")
        for d in (d2, h2):
            d.loc[d.DMDMARTL == 5, "no_partner"] = val
        E1, E2 = rel(d2.drop(columns="DMDMARTL")), rel(h2.drop(columns="DMDMARTL"))
        o = {T: r(X.paths(E1, X.T_FIXED[T])[OFF], E2[OFF]) for T in ["product", "hamacher", "min"]}
        o["rank3"] = r(X.lowrank_baseline(E1, 3, dyad=True)[OFF], E2[OFF])
        o["corr_with_main_dev"] = r(E1[OFF], Ed[OFF])
        res[f"never_married_{int(val)}"] = o
    tic("split halves within 2015-2018")
    d1518 = dev[dev.cycle.isin(["I", "J"])].copy()
    d1518["w"] *= dev.w.sum() / d1518.w.sum()
    sh = []
    for A, B in split_halves(d1518, 20, 11):
        EA, EB = rel(A), rel(B)
        o = {T: r(X.paths(EA, X.T_FIXED[T])[OFF], EB[OFF]) for T in ["product", "hamacher", "min"]}
        o["rank3"] = r(X.lowrank_baseline(EA, 3, dyad=True)[OFF], EB[OFF])
        sh.append(o)
    res["split_half_2015_2018"] = {k: float(np.mean([x[k] for x in sh])) for k in sh[0]}
    res["split_half_2015_2018_product_first"] = int(sum(x["product"] >= max(x["hamacher"], x["min"]) for x in sh))
    res["split_half_2015_2018_rank3_above_product"] = int(sum(x["rank3"] > x["product"] for x in sh))
    OUT["partner"] = res
    save()
