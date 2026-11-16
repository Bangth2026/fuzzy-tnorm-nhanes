"""Precision checks for the composition-versus-completion audit (v1.8.0).
Parts:
  direction   - exact empirical decomposition mu(i->j) = s_ij sqrt(r_ij) with r_ij = mu(i->j)/mu(j->i); error of the
                path-cancellation condition r_ik r_kj = r_ij; error of the prevalence-odds approximation
                mu(i->j) ~ s_ij sqrt(R_ij); agreement of directed and re-directed best paths (values, argmax); pairs with
                one zero direction
  nested      - single-dyad design with the completion rank and the soft-impute penalty chosen by inner dyad
                cross-validation inside every outer mask (both directions of the outer dyad removed before tuning)
  querytab    - clinical query per symptom: measured top correlate, predicted tops, and the share of the 20 split
                halves in which each rule recovers the measured top
  runtime     - wall-clock time of every step of the audit, and of one JRR replicate
  equivalence - post-registration two one-sided tests (90% JRR intervals) for the differences reported as not
                detected, with margins 0.03 and 0.05 in Spearman rho
  counts      - clipped negative excess and positive edges by period
Writes results2/rig_<part>.json."""
import itertools, json, sys, time
import numpy as np
from scipy.stats import spearmanr, norm
from analysis2 import dev, hold
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
import rq3x as X
import rq6lib as L

part = sys.argv[1]
OFF = ~np.eye(P, dtype=bool)
OUT = {}
LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]
MODIFIABLE = ["smoking", "heavy_drinking", "inactivity", "short_sleep", "obesity"]
SYMPTOMS = [c for c in CONCEPTS if TYPE[c] == "Symptom"]


def save():
    json.dump(OUT, open(f"results2/rig_{part}.json", "w"), indent=1, default=float)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def rel(df):
    return relation_matrices(df)[2]


def r(a, b):
    return float(spearmanr(a, b).statistic)


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


Ed, Eh = rel(dev), rel(hold)
pdev = prevalence(dev)

if part == "direction":
    pos = (Ed > 0) & (Ed.T > 0) & OFF
    S = np.sqrt(Ed * Ed.T)
    odds = pdev / (1 - pdev)
    lR = np.log(odds[None, :] / odds[:, None])
    lr = np.where(pos, np.log(np.where(pos, Ed, 1) / np.where(pos, Ed.T, 1)), np.nan)
    # path-cancellation error for empirical ratios over all triples with all four edges positive
    errs = []
    for i, k, j in itertools.permutations(range(P), 3):
        if pos[i, k] and pos[k, j] and pos[i, j]:
            errs.append(lr[i, k] + lr[k, j] - lr[i, j])
    errs = np.abs(np.array(errs))
    # prevalence-odds approximation error
    approx = np.abs(lr - lR)[pos]
    rel_err = np.abs(Ed - S * np.sqrt(np.exp(lR)))[pos] / Ed[pos]
    D = np.sqrt(np.exp(lR))
    dirp = X.paths(Ed, X.t_product); redir = X.paths(S, X.t_product) * D
    # argmax agreement for two-edge best paths
    agree, tot = 0, 0
    for i, j in itertools.permutations(range(P), 2):
        mids = [k for k in range(P) if k not in (i, j)]
        a = max(mids, key=lambda k: Ed[i, k] * Ed[k, j]); b = max(mids, key=lambda k: S[i, k] * S[k, j])
        agree += int(a == b); tot += 1
    OUT["direction"] = {
        "n_pairs_both_positive": int(pos.sum()), "n_pairs_one_direction_zero": int(((Ed > 0) & (Ed.T <= 0) & OFF).sum()),
        "n_pairs_both_zero": int(((Ed <= 0) & (Ed.T <= 0) & OFF).sum()),
        "cancellation_abs_log_error": {"n_triples": int(len(errs)), "median": float(np.median(errs)),
                                        "p90": float(np.quantile(errs, 0.9)), "max": float(errs.max())},
        "odds_approx_abs_log_error": {"median": float(np.median(approx)), "p90": float(np.quantile(approx, 0.9)), "max": float(approx.max())},
        "odds_approx_relative_error_mu": {"median": float(np.median(rel_err)), "p90": float(np.quantile(rel_err, 0.9))},
        "directed_vs_redirected_product": {"spearman": r(dirp[OFF], redir[OFF]), "median_abs_diff": float(np.median(np.abs(dirp - redir)[OFF])),
                                            "max_abs_diff": float(np.abs(dirp - redir)[OFF].max()),
                                            "rho_directed": r(dirp[OFF], Eh[OFF]), "rho_redirected": r(redir[OFF], Eh[OFF])},
        "same_best_two_edge_intermediate_share": agree / tot}
    save()

if part == "nested":
    from audit_extensions import rank_cv
    rk = np.zeros_like(Ed); si = np.zeros_like(Ed); ks, lams = [], []
    for i in range(P):
        for j in range(i + 1, P):
            base = OFF.copy(); base[i, j] = base[j, i] = False
            k, _ = rank_cv(Ed, base=base, seed=i * P + j)
            lam = L.choose_lambda(Ed, LAM_GRID, transform="logit", base=base, seed=i * P + j)[0]
            F = X.soft_impute(Ed, base, k); G = L.soft_impute_reg(Ed, base, lam, transform="logit")
            rk[i, j], rk[j, i] = F[i, j], F[j, i]; si[i, j], si[j, i] = G[i, j], G[j, i]
            ks.append(k); lams.append(lam)
    prod = X.paths(Ed, X.t_product)
    OUT["nested"] = {"rho_rank_nested": r(rk[OFF], Eh[OFF]), "rho_softimpute_nested": r(si[OFF], Eh[OFF]),
                     "rho_product": r(prod[OFF], Eh[OFF]),
                     "rank_counts": {str(k): ks.count(k) for k in sorted(set(ks))},
                     "lambda_counts": {str(l): lams.count(l) for l in sorted(set(lams))}}
    save()

if part == "querytab":
    ci = {c: CONCEPTS.index(c) for c in CONCEPTS}; mod = [ci[c] for c in MODIFIABLE]
    def tops(M, s):
        v = np.array([M[i, ci[s]] for i in mod]); top = np.where(np.isclose(v, v.max(), rtol=0, atol=1e-12))[0]
        return [MODIFIABLE[t] for t in top]
    rules = lambda E: {"product": X.paths(E, X.t_product), "min": X.paths(E, X.t_min), "hamacher": X.paths(E, X.T_FIXED["hamacher"]),
                       "rank3": X.lowrank_baseline(E, 3, dyad=True), "direct_edge": E}
    full = rules(Ed)
    res = {s: {"measured_top_2007_2012": tops(Eh, s)[0], **{k: tops(M, s) for k, M in full.items()}} for s in SYMPTOMS}
    hits = {s: {k: 0.0 for k in full} for s in SYMPTOMS}
    halves = split_halves(dev, 20, 4)
    for A, B in halves:
        EA, EB = rel(A), rel(B); pr = rules(EA)
        for s in SYMPTOMS:
            truth = tops(EB, s)[0]
            for k, M in pr.items():
                t = tops(M, s); hits[s][k] += (1.0 / len(t) if truth in t else 0.0) / len(halves)
    for s in SYMPTOMS:
        res[s]["split_half_hit_share"] = hits[s]
    OUT["querytab"] = res
    save()

if part == "runtime":
    from audit_extensions import rank_cv, fit_embed
    t = {}
    def tm(name, f):
        a = time.time(); f(); t[name] = time.time() - a
    tm("build_relation_one_period", lambda: rel(dev))
    for T in ["product", "hamacher", "min"]:
        tm(f"paths_{T}_all_552", lambda T=T: X.paths(Ed, X.T_FIXED[T]))
    tm("rank_cv_5fold", lambda: rank_cv(Ed))
    tm("rank3_leave_dyad_out_all", lambda: X.lowrank_baseline(Ed, 3, dyad=True))
    tm("softimpute_lambda_cv", lambda: L.choose_lambda(Ed, LAM_GRID, transform="logit"))
    tm("softimpute_leave_dyad_out_all", lambda: L.softimpute_dyad(Ed, 0.6, transform="logit"))
    tm("katz_leave_dyad_out_all", lambda: L.katz(Ed))
    tm("complex_leave_dyad_out_all", lambda: [fit_embed(Ed, OFF.astype(float), "complex", 4, 1e-2) for _ in range(5)])
    t["complex_leave_dyad_out_all"] *= 276 / 5
    tm("block_design_product_rank3_softimpute_katz", lambda: [(X.paths(np.where(OFF & (L.BLOCK == b), 0, Ed), X.t_product),
        X.soft_impute(Ed, OFF & ~(L.BLOCK == b), 3), L.choose_lambda(Ed, LAM_GRID, base=OFF & ~(L.BLOCK == b)), L.katz(np.where(OFF & (L.BLOCK == b), 0, Ed), dyad=False)) for b in range(10)])
    one = t["build_relation_one_period"] + t["paths_product_all_552"] + t["paths_hamacher_all_552"] + t["paths_min_all_552"] + \
          t["rank_cv_5fold"] + t["rank3_leave_dyad_out_all"] + t["softimpute_lambda_cv"] + t["softimpute_leave_dyad_out_all"] + t["katz_leave_dyad_out_all"]
    t["one_jrr_replicate_without_embeddings"] = one
    t["jrr_90_replicates_without_embeddings_estimated"] = one * 90
    t["jrr_90_replicates_with_complex_estimated"] = (one + t["complex_leave_dyad_out_all"]) * 90
    OUT["runtime"] = t
    save()

if part == "equivalence":
    from scipy.stats import norm
    g = json.load(open("results2/gen_sym.json"))["sym"]["temporal_jrr"]["product_minus_hamacher"]
    b = json.load(open("results2/gen_blockjrr.json"))["blockjrr"]["jrr"]["weighted_product_minus_katz"]
    c = json.load(open("results2/confirm_2005_main.json"))
    rows = {"exploratory_symmetric_product_minus_hamacher": (g["point"], g["se"]),
            "exploratory_blocks_product_minus_katz_weighted": (b["point"], b["se"]),
            "confirmatory_H2": (c["H2_sym_product_minus_hamacher"]["point"], (c["H2_sym_product_minus_hamacher"]["ci95"][1] - c["H2_sym_product_minus_hamacher"]["point"]) / 1.96),
            "confirmatory_H5": (c["H5_block_product_minus_katz"]["point"], (c["H5_block_product_minus_katz"]["ci95"][1] - c["H5_block_product_minus_katz"]["point"]) / 1.96),
            "confirmatory_H1": (c["H1_rank3_minus_product"]["point"], (c["H1_rank3_minus_product"]["ci95"][1] - c["H1_rank3_minus_product"]["point"]) / 1.96)}
    out = {}
    for k, (d, se) in rows.items():
        lo, hi = d - 1.645 * se, d + 1.645 * se
        out[k] = {"diff": d, "se": se, "ci90": [lo, hi],
                  "equivalent_at_0.03": bool(lo > -0.03 and hi < 0.03), "equivalent_at_0.05": bool(lo > -0.05 and hi < 0.05),
                  "p_tost_0.03": float(max(norm.sf((d + 0.03) / se), norm.sf((0.03 - d) / se))),
                  "p_tost_0.05": float(max(norm.sf((d + 0.05) / se), norm.sf((0.05 - d) / se)))}
    OUT["equivalence"] = out
    save()

if part == "counts":
    o = {}
    for name, df in [("development", dev), ("held_out", hold)]:
        S, S0, E = relation_matrices(df)
        raw = np.where(S0 < 1, (S - S0) / (1 - S0), 0); np.fill_diagonal(raw, 0)
        o[name] = {"ordered_pairs_negative_excess": int((raw[OFF] < 0).sum()), "ordered_pairs_positive": int((E[OFF] > 0).sum()),
                   "ordered_pairs_ge_0.01": int((E[OFF] >= 0.01).sum()),
                   "dyads_with_a_negative_direction": int(sum((raw[i, j] < 0) or (raw[j, i] < 0) for i in range(P) for j in range(i + 1, P)))}
    OUT["counts"] = o
    save()
