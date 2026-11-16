"""Mechanism and robustness checks for the composition-versus-completion audit (v1.7.0).
Parts:
  redirect   - compose every t-norm on the symmetric relation, then re-direct by sqrt(odds_j / odds_i) estimated from
               development prevalences; rho over the 552 ordered pairs, with JRR intervals for the differences
  holm2      - JRR for every predictor in Table II including NMF and ComplEx, with the completion rank re-chosen by
               cross-validation in every replicate; Holm adjustment over 11 comparisons against the product
  splitq     - clinical query in the 20 PSU split halves of 2013-2018 (half A predicts half B and the 2007-2012 relation)
  randthin   - development graph thinned at random (keep 50% or 25% of edges, 20 repetitions) and by simulated sources
               that each measured two concept types; zero-prediction share and AUC for strength-based thinning
  scale      - soft-impute on the raw and the logit scale in every design (single dyad, random dyads, cold start, blocks)
  blockextra - withheld relation types: minimum and ComplEx per block, pooled and pair-weighted; rank re-chosen by CV
               on the other blocks
  simblock   - respondent-level simulation (chained, one-factor, three-factor; ordinal/binary or continuous
               measurement): t-norm order with single dyads withheld and paths against completion with whole relation
               types withheld
Writes results2/mech_<part>.json."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm, binomtest
from sklearn.metrics import roc_auc_score
from analysis2 import dev, hold
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
import rq3x as X
import rq6lib as L

part = sys.argv[1]
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
    json.dump(OUT, open(f"results2/mech_{part}.json", "w"), indent=1, default=float)


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
    out = {}
    for k in full:
        v = np.array([x[k] for x in reps], float)
        se = float(np.sqrt(((v - full[k]) ** 2).sum()))
        out[k] = {"point": float(full[k]), "se": se, "ci95": [full[k] - 1.96 * se, full[k] + 1.96 * se]}
    return out


Ed, Eh = rel(dev), rel(hold)
pdev = prevalence(dev)
LAM_LOGIT = L.choose_lambda(Ed, LAM_GRID, transform="logit")[0]
LAM_RAW = L.choose_lambda(Ed, LAM_GRID)[0]

# ------------------------------------------------------------------------------------------------
if part == "redirect":
    def redirect_stats(E1, E2, p1):
        odds = p1 / (1 - p1)
        D = np.sqrt(odds[None, :] / odds[:, None])          # sqrt(odds_j / odds_i)
        S = np.sqrt(E1 * E1.T)
        o = {}
        for T in TNORMS:
            o[f"{T}_directed"] = r(X.paths(E1, X.T_FIXED[T])[OFF], E2[OFF])
            o[f"{T}_sym_redirected"] = r((X.paths(S, X.T_FIXED[T]) * D)[OFF], E2[OFF])
        o["rank3_directed"] = r(X.lowrank_baseline(E1, 3, dyad=True)[OFF], E2[OFF])
        o["rank3_sym_redirected"] = r((X.lowrank_baseline(S, 3, dyad=True) * D)[OFF], E2[OFF])
        o["direction_factor_only"] = r(D[OFF], E2[OFF])
        for k in ["hamacher", "min"]:
            o[f"product_minus_{k}_directed"] = o["product_directed"] - o[f"{k}_directed"]
            o[f"product_minus_{k}_sym_redirected"] = o["product_sym_redirected"] - o[f"{k}_sym_redirected"]
        o["rank3_minus_product_sym_redirected"] = o["rank3_sym_redirected"] - o["product_sym_redirected"]
        return o

    full = redirect_stats(Ed, Eh, pdev)
    tic("JRR")
    reps = [redirect_stats(rel(d), Eh, prevalence(d)) for d in jrr_reps(dev)] + \
           [redirect_stats(Ed, rel(h), pdev) for h in jrr_reps(hold)]
    tic("split halves")
    sh = []
    for A, B in split_halves(dev, 20, 4):
        sh.append(redirect_stats(rel(A), rel(B), prevalence(A)))
    OUT["redirect"] = {"temporal": jrr_summary(full, reps),
                       "split_half_mean": {k: float(np.mean([x[k] for x in sh])) for k in sh[0]},
                       "split_half_product_first_redirected": int(sum(x["product_sym_redirected"] >= max(x[f"{T}_sym_redirected"] for T in TNORMS) for x in sh)),
                       "split_half_hamacher_first_redirected": int(sum(x["hamacher_sym_redirected"] >= max(x[f"{T}_sym_redirected"] for T in TNORMS) for x in sh))}
    save()

# ------------------------------------------------------------------------------------------------
if part == "holm2":
    from audit_extensions import fit_embed, rank_cv

    def complex_dyad(E):
        out = np.zeros_like(E)
        for i in range(P):
            for j in range(i + 1, P):
                Wm = OFF.astype(float); Wm[i, j] = Wm[j, i] = 0
                F = fit_embed(E, Wm, "complex", 4, 1e-2)
                out[i, j], out[j, i] = F[i, j], F[j, i]
        return out

    def stats(E1, E2):
        f = L.path_features(E1)
        k, _ = rank_cv(E1)
        pr = {"product": f["prod_max"], "hamacher": f["ham_max"], "min": f["min_max"],
              "yager2": X.paths(E1, X.T_FIXED["yager2"]), "lukasiewicz": X.paths(E1, X.T_FIXED["lukasiewicz"]),
              "resource_alloc": f["resource_alloc"], "adamic_adar": f["adamic_adar"], "katz": L.katz(E1),
              "rank_cv": X.lowrank_baseline(E1, k, dyad=True),
              "softimpute_logit_cv": L.softimpute_dyad(E1, L.choose_lambda(E1, LAM_GRID, transform="logit")[0], transform="logit"),
              "nmf3": L.nmf_dyad(E1), "complex": complex_dyad(E1)}
        o = {kk: r(v[OFF], E2[OFF]) for kk, v in pr.items()}
        o["chosen_rank"] = float(k)
        for kk in pr:
            if kk != "product":
                o[f"{kk}_minus_product"] = o[kk] - o["product"]
        return o

    full = stats(Ed, Eh)
    tic("JRR")
    reps = []
    for d in jrr_reps(dev):
        reps.append(stats(rel(d), Eh))
    for h in jrr_reps(hold):
        reps.append(stats(Ed, rel(h)))
    js = jrr_summary(full, reps)
    diffs = [k for k in full if k.endswith("_minus_product")]
    pv = {k: float(2 * norm.sf(abs(js[k]["point"]) / js[k]["se"])) for k in diffs}
    holm, run = {}, 0.0
    for q, k in enumerate(sorted(diffs, key=lambda k: pv[k])):
        run = max(run, min(1.0, (len(diffs) - q) * pv[k])); holm[k] = run
    OUT["holm2"] = {"jrr": js, "p_raw": pv, "p_holm": holm,
                    "chosen_rank_distribution": {str(int(v)): int(sum(x["chosen_rank"] == v for x in reps)) for v in sorted(set(x["chosen_rank"] for x in reps))}}
    save()

# ------------------------------------------------------------------------------------------------
if part == "splitq":
    ci = {c: CONCEPTS.index(c) for c in CONCEPTS}
    mod = [ci[c] for c in MODIFIABLE]

    def credit(p, t):
        top = np.where(np.isclose(p, p.max(), rtol=0, atol=1e-12))[0]
        return (1.0 / len(top)) if int(np.argmax(t)) in top else 0.0

    def count(E1, E2):
        pr = {"product": X.paths(E1, X.t_product), "hamacher": X.paths(E1, X.T_FIXED["hamacher"]),
              "min": X.paths(E1, X.t_min), "rank3": X.lowrank_baseline(E1, 3, dyad=True), "direct_edge": E1}
        o = {}
        for k, M in pr.items():
            o[k] = sum(credit(np.array([M[i, ci[s]] for i in mod]), np.array([E2[i, ci[s]] for i in mod])) for s in SYMPTOMS)
            o[f"{k}_ties"] = int(sum(np.isclose(np.array([M[i, ci[s]] for i in mod]), max(M[i, ci[s]] for i in mod), rtol=0, atol=1e-12).sum() > 1 for s in SYMPTOMS))
        return o

    res = {"half_to_half": [], "half_to_heldout": []}
    for A, B in split_halves(dev, 20, 4):
        EA, EB = rel(A), rel(B)
        res["half_to_half"].append(count(EA, EB)); res["half_to_heldout"].append(count(EA, Eh))
    summ = {}
    for d in res:
        summ[d] = {k: {"mean": float(np.mean([x[k] for x in res[d]])), "min": float(np.min([x[k] for x in res[d]])),
                       "max": float(np.max([x[k] for x in res[d]]))} for k in res[d][0]}
        summ[d]["product_above_min"] = int(sum(x["product"] > x["min"] for x in res[d]))
        summ[d]["product_at_least_direct"] = int(sum(x["product"] >= x["direct_edge"] for x in res[d]))
    OUT["splitq"] = summ
    save()

# ------------------------------------------------------------------------------------------------
if part == "randthin":
    def evaluate(keep, label, reps_store):
        keep = keep & OFF
        A = np.where(keep, Ed, 0.0)
        unrec = OFF & ~keep
        pr = {T: X.paths(A, X.T_FIXED[T]) for T in ["product", "hamacher", "min"]}
        pr["katz"] = L.katz(A, dyad=False)
        pr["rank3_absent_missing"] = X.soft_impute(Ed, keep, 3)
        pr["rank3_absent_zero"] = X.lowrank_baseline(A, 3, dyad=True)
        try:
            lam = L.choose_lambda(Ed, LAM_GRID, base=keep)[0]
            pr["softimpute_absent_missing"] = L.soft_impute_reg(Ed, keep, lam)
        except Exception:
            pass
        o = {k: r(v[unrec], Eh[unrec]) for k, v in pr.items()}
        o.update({f"zero_share_{k}": float((v[unrec] <= 1e-12).mean()) for k, v in pr.items()})
        pos = Eh[unrec] >= 0.01
        if 0 < pos.sum() < len(pos):
            o.update({f"auc_pos_{k}": float(roc_auc_score(pos, v[unrec])) for k, v in pr.items()})
        reps_store.append(o)

    res = {}
    for frac in [0.5, 0.25]:
        rows = []
        for rep in range(20):
            g = np.random.default_rng(3100 + rep + int(100 * frac))
            keep = (g.random((P, P)) < frac) & OFF
            evaluate(keep, f"random{frac}", rows)
        res[f"random_keep_{frac}"] = {k: float(np.nanmean([x.get(k, np.nan) for x in rows])) for k in rows[0]}
        tic(f"random {frac}")
    # simulated sources: each measured relations among two concept types
    TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
    ti = np.array([TYPES.index(TYPE[c]) for c in CONCEPTS])
    designs = {"sources_SymSES_LifeHlth_SymHlth": [(0, 1), (2, 3), (0, 3)],
               "sources_SymLife_SESHlth_SymSES": [(0, 2), (1, 3), (0, 1)]}
    for name, srcs in designs.items():
        keep = np.zeros((P, P), bool)
        for a, b in srcs:
            m = np.isin(ti, [a, b])
            keep |= m[:, None] & m[None, :]
        rows = []
        evaluate(keep, name, rows)
        res[name] = rows[0]; res[name]["n_unrecorded"] = int((OFF & ~keep).sum())
    # strength-based thinning: zero-prediction share and AUC for positive versus zero targets
    q50, q75 = np.quantile(Ed[OFF], [0.5, 0.75])
    for name, keep in {"strength_top50": Ed >= q50, "strength_top25": Ed >= q75, "strength_mu_ge_0.25": Ed >= 0.25}.items():
        rows = []
        evaluate(keep, name, rows)
        res[name] = rows[0]
    OUT["randthin"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "scale":
    res = {}
    tic("single dyad")
    res["single_dyad"] = {"raw": r(L.softimpute_dyad(Ed, LAM_RAW)[OFF], Eh[OFF]),
                          "logit": r(L.softimpute_dyad(Ed, LAM_LOGIT, transform="logit")[OFF], Eh[OFF]),
                          "lambda_raw": LAM_RAW, "lambda_logit": LAM_LOGIT}
    # choose the scale by within-development dyad CV
    cvraw = max(L.choose_lambda(Ed, LAM_GRID)[1].values()); cvlog = max(L.choose_lambda(Ed, LAM_GRID, transform="logit")[1].values())
    res["cv_rho_raw_vs_logit"] = [cvraw, cvlog]
    dyads = [(i, j) for i in range(P) for j in range(i + 1, P)]
    for frac in [0.10, 0.25, 0.50, 0.75]:
        rows = []
        for rep in range(50):
            g = np.random.default_rng(7000 + rep + int(frac * 1000))
            pick = g.choice(len(dyads), int(round(frac * len(dyads))), replace=False)
            H = np.zeros((P, P), bool)
            for q in pick:
                i, j = dyads[q]; H[i, j] = H[j, i] = True
            obs = OFF & ~H
            lr = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep)[0]
            ll = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep, transform="logit")[0]
            rows.append({"raw": r(L.soft_impute_reg(Ed, obs, lr)[H], Eh[H]),
                         "logit": r(L.soft_impute_reg(Ed, obs, ll, transform="logit")[H], Eh[H]),
                         "product": r(X.paths(np.where(obs, Ed, 0.0), X.t_product)[H], Eh[H])})
        res[f"random_{frac}"] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
        tic(f"random {frac}")
    for m in [1, 3, 6]:
        rows = []
        for rep in range(5):
            g = np.random.default_rng(900 + 31 * m + rep)
            per = {"raw": [], "logit": []}
            for c in range(P):
                others = [j for j in range(P) if j != c]
                anchors = set(g.choice(others, m, replace=False).tolist())
                H = np.zeros((P, P), bool)
                for j in others:
                    if j not in anchors:
                        H[c, j] = H[j, c] = True
                obs = OFF & ~H
                lr = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep)[0]
                ll = L.choose_lambda(Ed, LAM_GRID, base=obs, seed=rep, transform="logit")[0]
                per["raw"].append(r(L.soft_impute_reg(Ed, obs, lr)[H], Eh[H]))
                per["logit"].append(r(L.soft_impute_reg(Ed, obs, ll, transform="logit")[H], Eh[H]))
            rows.append({k: float(np.nanmedian(v)) for k, v in per.items()})
        res[f"coldstart_m{m}_median_per_concept"] = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
        tic(f"cold {m}")
    blk = {"raw": [], "logit": []}
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        lr = L.choose_lambda(Ed, LAM_GRID, base=OFF & ~H)[0]
        ll = L.choose_lambda(Ed, LAM_GRID, base=OFF & ~H, transform="logit")[0]
        blk["raw"].append(r(L.soft_impute_reg(Ed, OFF & ~H, lr)[H], Eh[H]))
        blk["logit"].append(r(L.soft_impute_reg(Ed, OFF & ~H, ll, transform="logit")[H], Eh[H]))
    nb = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
    res["blocks_weighted"] = {k: float((np.array(v) * nb).sum() / nb.sum()) for k, v in blk.items()}
    OUT["scale"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "blockextra":
    from audit_extensions import fit_embed, cv_embed, EMB_GRID, rank_cv
    nb = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
    pooled = {k: np.zeros_like(Ed) for k in ["min", "complex", "rank_cv_otherblocks", "product", "katz"]}
    ranks = []
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        Em = np.where(H, 0.0, Ed)
        pooled["min"][H] = X.paths(Em, X.t_min)[H]
        pooled["product"][H] = X.paths(Em, X.t_product)[H]
        pooled["katz"][H] = L.katz(Em, dyad=False)[H]
        cfg, _ = cv_embed(Ed, "complex", EMB_GRID, base=OFF & ~H)
        pooled["complex"][H] = fit_embed(Ed, (OFF & ~H).astype(float), "complex", *cfg)[H]
        k, _ = rank_cv(Ed, base=OFF & ~H)
        ranks.append(int(k))
        pooled["rank_cv_otherblocks"][H] = X.soft_impute(Ed, OFF & ~H, k)[H]
        tic(f"block {b}")
    res = {"ranks_chosen_on_other_blocks": ranks}
    for k, v in pooled.items():
        pb = np.array([r(v[OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) for b in range(10)])
        res[k] = {"per_block": pb.tolist(), "weighted_mean": float((pb * nb).sum() / nb.sum()),
                  "pooled": r(v[OFF], Eh[OFF]), "product_better_blocks": None}
    for k in res:
        if isinstance(res[k], dict) and k != "product":
            res[k]["product_better_blocks"] = int(sum(a > c for a, c in zip(res["product"]["per_block"], res[k]["per_block"])))
    OUT["blockextra"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "simblock":
    nresp = 15000
    kinds = ["ordinal"] * 9 + ["binary"] * 15

    def measure(Z, continuous):
        A = np.empty_like(Z)
        for c in range(P):
            z = (Z[:, c] - Z[:, c].mean()) / Z[:, c].std()
            if continuous:
                A[:, c] = norm.cdf(z - (0.9 if kinds[c] == "ordinal" else 0.7))
            elif kinds[c] == "ordinal":
                A[:, c] = np.digitize(z, np.quantile(z, [0.70, 0.88, 0.96])) / 3
            else:
                A[:, c] = (z > np.quantile(z, 0.75)).astype(float)
        return A

    def chain_model(g):
        order = g.permutation(P)
        B = np.zeros((P, P))
        for a in range(1, P):
            for b in range(a):
                if g.random() < 0.12:
                    B[order[b], order[a]] = g.uniform(0.3, 0.7)
        def sample(nn):
            Z = np.zeros((nn, P))
            for c in order:
                Z[:, c] = Z @ B[:, c] + g.standard_normal(nn)
            return Z
        return sample

    def factor_model(g, k):
        Lm = np.zeros((P, k))
        for c in range(P):
            Lm[c, c % k if k > 1 else 0] = g.uniform(0.4, 0.8)
            if k > 1:
                Lm[c] += g.uniform(0, 0.25, k)
        def sample(nn):
            return g.standard_normal((nn, k)) @ Lm.T + g.standard_normal((nn, P)) * 0.8
        return sample

    def one(sampler, continuous):
        Ms = []
        for _ in range(2):
            df = pd.DataFrame(measure(sampler(nresp), continuous), columns=CONCEPTS); df["w"] = 1.0
            Ms.append(relation_matrices(df)[2])
        A1, A2 = Ms
        o = {T: r(X.paths(A1, X.T_FIXED[T])[OFF], A2[OFF]) for T in ["product", "hamacher", "min"]}
        o["rank3"] = r(X.lowrank_baseline(A1, 3, dyad=True)[OFF], A2[OFF])
        nb = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)
        pb = {k: [] for k in ["product", "hamacher", "rank3", "softimpute_cv", "katz"]}
        for b in range(10):
            H = OFF & (L.BLOCK == b)
            Em = np.where(H, 0.0, A1)
            pb["product"].append(r(X.paths(Em, X.t_product)[H], A2[H]))
            pb["hamacher"].append(r(X.paths(Em, X.T_FIXED["hamacher"])[H], A2[H]))
            pb["katz"].append(r(L.katz(Em, dyad=False)[H], A2[H]))
            pb["rank3"].append(r(X.soft_impute(A1, OFF & ~H, 3)[H], A2[H]))
            lb = L.choose_lambda(A1, LAM_GRID, base=OFF & ~H)[0]
            pb["softimpute_cv"].append(r(L.soft_impute_reg(A1, OFF & ~H, lb)[H], A2[H]))
        for k, v in pb.items():
            o[f"block_{k}"] = float(np.nansum(np.array(v) * nb) / nb[~np.isnan(v)].sum())
        return o

    res = {}
    for cont in [False, True]:
        for name, mk in [("chain", chain_model), ("factor1", lambda g: factor_model(g, 1)), ("factor3", lambda g: factor_model(g, 3))]:
            rows = []
            for rep in range(30):
                g = np.random.default_rng(5000 + 97 * rep + len(name))
                rows.append(one(mk(g), cont))
            key = f"{name}_{'continuous' if cont else 'ordinal_binary'}"
            s = {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]}
            s["share_product_best_tnorm"] = float(np.mean([x["product"] >= max(x["hamacher"], x["min"]) for x in rows]))
            s["share_hamacher_above_product"] = float(np.mean([x["hamacher"] > x["product"] for x in rows]))
            s["share_rank3_above_product"] = float(np.mean([x["rank3"] > x["product"] for x in rows]))
            for k in ["rank3", "softimpute_cv", "katz", "hamacher"]:
                s[f"block_share_product_above_{k}"] = float(np.mean([x["block_product"] > x[f"block_{k}"] for x in rows]))
            res[key] = s
            tic(key)
            OUT["simblock"] = res
            save()
