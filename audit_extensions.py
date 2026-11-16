"""Further checks of the composition-versus-completion audit (v1.5.0).
Parts:
  mask       - forward-only masking sanity check for completion; learned path model trained leave-dyad-out
  rankcv     - rank of completion chosen by dyad cross-validation within development
  embed      - knowledge-graph embedding baselines (DistMult, ComplEx) fitted as regressions on mu, leave-dyad-out
  strong     - strong-tail metrics: precision@K and AUC for target pairs in the upper quartile, single-dyad and
               withheld-block designs
  designs    - soft-impute (logit, CV) in the split-half and swapped designs, and its JRR interval
  measures   - alternative association measures: positive Pearson correlation, positive Yule's Q, positive
               regularised partial correlation (graphical lasso, EBIC)
  signedcomp - completion on the signed (unclipped) relation
  query      - clinical query: modifiable correlates linked to each PHQ-9 symptom (top-k agreement)
  obesity    - obesity reassigned to the Health type: withheld blocks and type-partition modularity
  initial    - path rules and completion under the membership functions used before the clinical review
  uninsured  - concept 'uninsured' removed (coverage change between periods)
  misc       - range of effective sample sizes, MAE of tuned families, diagnosis of the Life-Health block
Writes results2/ext_<part>.json."""
import itertools, json, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest
from sklearn.metrics import roc_auc_score
from analysis2 import dev, hold, psu_boot
import graph2
from graph2 import CONCEPTS, TYPE, P, relation_matrices, wmean
import rq3x as X
import rq6lib as L

part = sys.argv[1]
rng = np.random.default_rng(20261116)
OFF = ~np.eye(P, dtype=bool)
OUT = {}
LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]
MODIFIABLE = ["smoking", "heavy_drinking", "inactivity", "short_sleep", "obesity"]
SYMPTOMS = [c for c in CONCEPTS if TYPE[c] == "Symptom"]


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/ext_{part}.json", "w"), indent=1, default=float)


def prevalence(df):
    return np.array([wmean(df.loc[df[c].notna(), c], df.loc[df[c].notna(), "w"]) for c in CONCEPTS])


def rel(df):
    return relation_matrices(df)[2]


def r(a, b):
    return float(spearmanr(a, b).statistic)


Ed, Eh = rel(dev), rel(hold)
pdev = prevalence(dev)


# ---------- knowledge-graph embeddings (single relation 'associated with'), squared loss on sigmoid(score) ----------
def fit_embed(R, W, kind="complex", dim=4, lam=1e-3, iters=1500, lr=0.05, seed=0):
    g = np.random.default_rng(seed)
    n = R.shape[0]
    A = g.normal(0, 0.3, (n, dim)); B = g.normal(0, 0.3, (n, dim)) if kind == "complex" else np.zeros((n, dim))
    c = np.ones(dim); d = g.normal(0, 0.1, dim) if kind == "complex" else np.zeros(dim)
    b0 = np.array([-2.0])
    params = [A, B, c, d, b0]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    for t in range(1, iters + 1):
        S = (A * c) @ A.T + (B * c) @ B.T + (A * d) @ B.T - (B * d) @ A.T + b0[0]
        p = 1 / (1 + np.exp(-S))
        G = 2 * W * (p - R) * p * (1 - p)
        gA = G @ (A * c) + G.T @ (A * c) + G @ (B * d) - G.T @ (B * d) + 2 * lam * A
        gB = G @ (B * c) + G.T @ (B * c) + G.T @ (A * d) - G @ (A * d) + 2 * lam * B
        gc = np.einsum("ij,id,jd->d", G, A, A) + np.einsum("ij,id,jd->d", G, B, B)
        gd = np.einsum("ij,id,jd->d", G, A, B) - np.einsum("ij,id,jd->d", G, B, A)
        gb = np.array([G.sum()])
        grads = [gA, gB, gc, gd, gb]
        if kind != "complex":
            grads[1] = np.zeros_like(B); grads[3] = np.zeros_like(d)
        for k, (pp, gg) in enumerate(zip(params, grads)):
            m[k] = 0.9 * m[k] + 0.1 * gg; v[k] = 0.999 * v[k] + 0.001 * gg ** 2
            pp -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    S = (A * c) @ A.T + (B * c) @ B.T + (A * d) @ B.T - (B * d) @ A.T + b0[0]
    return 1 / (1 + np.exp(-S))


def cv_embed(R, kind, grid, folds=5, base=None, seed=0):
    g = np.random.default_rng(seed)
    base = OFF if base is None else base
    dy = [(i, j) for i in range(P) for j in range(i + 1, P) if base[i, j] and base[j, i]]
    idx = g.permutation(len(dy))
    sc = {}
    for cfg in grid:
        s = []
        for f in range(folds):
            Wm = base.copy()
            for k in idx[f::folds]:
                i, j = dy[k]; Wm[i, j] = Wm[j, i] = False
            H = base & ~Wm
            s.append(r(fit_embed(R, Wm.astype(float), kind, *cfg)[H], R[H]))
        sc[cfg] = float(np.mean(s))
    best = max(sc, key=sc.get)
    return best, {str(k): v for k, v in sc.items()}


def embed_dyad(R, kind, cfg):
    out = np.zeros_like(R)
    for i in range(P):
        for j in range(i + 1, P):
            Wm = OFF.astype(float); Wm[i, j] = Wm[j, i] = 0
            F = fit_embed(R, Wm, kind, *cfg)
            out[i, j], out[j, i] = F[i, j], F[j, i]
    return out


EMB_GRID = [(2, 1e-3), (4, 1e-3), (8, 1e-3), (4, 1e-2), (8, 1e-2)]


def rank_cv(R, ks=(1, 2, 3, 4, 5), folds=5, base=None, seed=0):
    g = np.random.default_rng(seed)
    base = OFF if base is None else base
    dy = [(i, j) for i in range(P) for j in range(i + 1, P) if base[i, j] and base[j, i]]
    idx = g.permutation(len(dy))
    sc = {k: [] for k in ks}
    for f in range(folds):
        m = base.copy()
        for q in idx[f::folds]:
            i, j = dy[q]; m[i, j] = m[j, i] = False
        H = base & ~m
        for k in ks:
            sc[k].append(r(X.soft_impute(R, m, k)[H], R[H]))
    mean = {k: float(np.mean(v)) for k, v in sc.items()}
    return max(mean, key=mean.get), mean


def strong_metrics(pred, target, mask):
    y = target[mask]; s = pred[mask]
    q = np.quantile(y, 0.75)
    pos = y >= q
    K = int(pos.sum())
    top = np.argsort(-s, kind="stable")[:K]
    return {"precision_at_K": float(pos[top].mean()), "K": K, "auc_top_quartile": float(roc_auc_score(pos, s)),
            "rho": r(s, y)}


if part == "mask":
    res = {}
    r3_dyad = X.lowrank_baseline(Ed, 3, dyad=True)
    r3_fwd = X.lowrank_baseline(Ed, 3, dyad=False)
    res["rank3_both_masked"] = r(r3_dyad[OFF], Eh[OFF])
    res["rank3_forward_only_masked"] = r(r3_fwd[OFF], Eh[OFF])
    lam, _ = L.choose_lambda(Ed, LAM_GRID, transform="logit")
    si_fwd = np.zeros_like(Ed)
    for i, j in itertools.permutations(range(P), 2):
        m = OFF.copy(); m[i, j] = False
        si_fwd[i, j] = L.soft_impute_reg(Ed, m, lam, transform="logit")[i, j]
    res["softimpute_logit_forward_only_masked"] = r(si_fwd[OFF], Eh[OFF])
    res["softimpute_logit_lambda"] = lam
    # learned path model trained strictly leave-dyad-out
    F = L.path_features(Ed)
    Xall = np.stack([F[k] for k in L.FEATS], -1)
    from sklearn.linear_model import Ridge
    pred = np.zeros_like(Ed)
    for i in range(P):
        for j in range(i + 1, P):
            tr = OFF.copy(); tr[i, j] = tr[j, i] = False
            Xt = Xall[tr]; mu_, sd_ = Xt.mean(0), Xt.std(0) + 1e-12
            mdl = Ridge(alpha=1.0).fit((Xt - mu_) / sd_, Ed[tr])
            for a, b in [(i, j), (j, i)]:
                pred[a, b] = np.clip(mdl.predict(((Xall[a, b] - mu_) / sd_)[None])[0], 0, 1)
    res["learned_paths_strict_ldo"] = {"rho": r(pred[OFF], Eh[OFF]), "mae": float(np.abs(pred[OFF] - Eh[OFF]).mean())}
    res["learned_paths_features"] = L.FEATS
    OUT["mask"] = res
    save()

if part == "rankcv":
    k, sc = rank_cv(Ed)
    pr = X.lowrank_baseline(Ed, k, dyad=True)
    OUT["rankcv"] = {"chosen_rank": int(k), "cv_scores": {str(a): b for a, b in sc.items()},
                     "rho_heldout_at_chosen": r(pr[OFF], Eh[OFF]), "mae": float(np.abs(pr[OFF] - Eh[OFF]).mean())}
    save()

if part == "embed":
    res = {}
    for kind in ["distmult", "complex"]:
        tic(kind)
        cfg, sc = cv_embed(Ed, kind, EMB_GRID)
        pr = embed_dyad(Ed, kind, cfg)
        np.save(f"results2/ext_embed_{kind}.npy", pr)
        res[kind] = {"chosen_dim_l2": list(cfg), "cv_scores": sc, "rho": r(pr[OFF], Eh[OFF]),
                     "mae": float(np.abs(pr[OFF] - Eh[OFF]).mean())}
    OUT["embed"] = res
    save()

if part == "strong":
    res = {"single_dyad": {}, "blocks_pooled": {}}
    f = L.path_features(Ed)
    lam, _ = L.choose_lambda(Ed, LAM_GRID, transform="logit")
    preds = {"product": f["prod_max"], "hamacher": f["ham_max"], "min": f["min_max"],
             "resource_alloc": f["resource_alloc"], "katz": L.katz(Ed),
             "rank3": X.lowrank_baseline(Ed, 3, dyad=True),
             "softimpute_logit_cv": L.softimpute_dyad(Ed, lam, transform="logit"),
             "reverse_odds": X.reverse_baseline(Ed, pdev)[0]}
    try:
        preds["complex"] = np.load("results2/ext_embed_complex.npy")
    except FileNotFoundError:
        pass
    for k, v in preds.items():
        res["single_dyad"][k] = strong_metrics(v, Eh, OFF)
    # withheld blocks, pooled: every pair predicted once, when its block is withheld
    pooled = {k: np.zeros_like(Ed) for k in ["product", "hamacher", "rank3", "softimpute_cv", "katz", "complex"]}
    for b in range(10):
        H = OFF & (L.BLOCK == b)
        Em = np.where(H, 0.0, Ed)
        fb = L.path_features(Em)
        pooled["product"][H] = fb["prod_max"][H]; pooled["hamacher"][H] = fb["ham_max"][H]
        pooled["rank3"][H] = X.soft_impute(Ed, OFF & ~H, 3)[H]
        lb, _ = L.choose_lambda(Ed, LAM_GRID, base=OFF & ~H)
        pooled["softimpute_cv"][H] = L.soft_impute_reg(Ed, OFF & ~H, lb)[H]
        pooled["katz"][H] = L.katz(Em, dyad=False)[H]
        cfg, _ = cv_embed(Ed, "complex", EMB_GRID, base=OFF & ~H)
        pooled["complex"][H] = fit_embed(Ed, (OFF & ~H).astype(float), "complex", *cfg)[H]
        tic(f"block {b}")
    for k, v in pooled.items():
        res["blocks_pooled"][k] = strong_metrics(v, Eh, OFF)
    # per-block rho for complex embedding
    res["complex_per_block"] = {L.BLOCK_NAMES[b]: r(pooled["complex"][OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) for b in range(10)}
    pb = {k: [r(pooled[k][OFF & (L.BLOCK == b)], Eh[OFF & (L.BLOCK == b)]) for b in range(10)] for k in pooled}
    res["product_vs_complex_blocks"] = {"product_better": int(sum(a > c for a, c in zip(pb["product"], pb["complex"]))),
                                        "sign_test_p": float(binomtest(int(sum(a > c for a, c in zip(pb["product"], pb["complex"]))), 10).pvalue),
                                        "median_complex": float(np.median(pb["complex"]))}
    OUT["strong"] = res
    save()


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
            pick = g.permutation(ps)[: max(1, len(ps) // 2)]
            a += list(pick)
        A = df[df.psu.isin(a)].copy(); B = df[~df.psu.isin(a)].copy()
        A["w"] *= df.w.sum() / A.w.sum(); B["w"] *= df.w.sum() / B.w.sum()
        out.append((A, B))
    return out


if part == "designs":
    lam, _ = L.choose_lambda(Ed, LAM_GRID, transform="logit")
    si = lambda E: L.softimpute_dyad(E, lam, transform="logit")
    res = {"lambda_fixed": lam}
    tic("swapped")
    res["swapped"] = r(si(Eh)[OFF], Ed[OFF])
    tic("split halves")
    sh = []
    for A, B in split_halves(dev, 20, 4):
        EA, EB = rel(A), rel(B)
        sh.append({"si_logit": r(si(EA)[OFF], EB[OFF]), "product": r(X.paths(EA, X.t_product)[OFF], EB[OFF])})
    res["split_half_mean"] = {k: float(np.mean([x[k] for x in sh])) for k in sh[0]}
    res["split_half_share_si_above_product"] = float(np.mean([x["si_logit"] > x["product"] for x in sh]))
    tic("JRR")
    full = r(si(Ed)[OFF], Eh[OFF]); prod = r(X.paths(Ed, X.t_product)[OFF], Eh[OFF])
    reps = [(rel(d), Eh) for d in jrr_reps(dev)] + [(Ed, rel(h)) for h in jrr_reps(hold)]
    v = []; dvs = []
    for E1, E2 in reps:
        a = r(si(E1)[OFF], E2[OFF]); b = r(X.paths(E1, X.t_product)[OFF], E2[OFF])
        v.append(a); dvs.append(a - b)
    se = float(np.sqrt(((np.array(v) - full) ** 2).sum())); sed = float(np.sqrt(((np.array(dvs) - (full - prod)) ** 2).sum()))
    res["temporal"] = {"rho": full, "jrr_ci": [full - 1.96 * se, full + 1.96 * se],
                       "minus_product": full - prod, "minus_product_jrr_ci": [full - prod - 1.96 * sed, full - prod + 1.96 * sed]}
    OUT["designs"] = res
    save()


def wcorr(df, cols):
    """Survey-weighted pairwise Pearson correlation."""
    n = len(cols); C = np.eye(n)
    for a, b in itertools.combinations(range(n), 2):
        m = df[cols[a]].notna() & df[cols[b]].notna()
        x, y, w = df[cols[a]][m].to_numpy(), df[cols[b]][m].to_numpy(), df.w[m].to_numpy()
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        cv = np.average((x - mx) * (y - my), weights=w)
        C[a, b] = C[b, a] = cv / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
    return C


def yule_q(df, cols, cut=0.5):
    n = len(cols); Q = np.zeros((n, n))
    for a, b in itertools.combinations(range(n), 2):
        m = df[cols[a]].notna() & df[cols[b]].notna()
        x, y, w = (df[cols[a]][m] >= cut).to_numpy(), (df[cols[b]][m] >= cut).to_numpy(), df.w[m].to_numpy()
        n11 = w[x & y].sum(); n00 = w[~x & ~y].sum(); n10 = w[x & ~y].sum(); n01 = w[~x & y].sum()
        Q[a, b] = Q[b, a] = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01) if (n11 * n00 + n10 * n01) > 0 else 0.0
    return Q


def glasso_partial(df, cols):
    from sklearn.covariance import graphical_lasso
    cc = df.dropna(subset=cols)
    w = cc.w.to_numpy() / cc.w.sum()
    Z = cc[cols].to_numpy()
    mu = w @ Z
    S = ((Z - mu) * w[:, None]).T @ (Z - mu)
    sd = np.sqrt(np.diag(S)); S = S / np.outer(sd, sd)
    nn = len(cc)
    best = None
    for alpha in np.geomspace(0.005, 0.3, 20):
        cov, prec = graphical_lasso(S, alpha=alpha, max_iter=500)
        k = int((np.abs(prec[np.triu_indices(len(cols), 1)]) > 1e-8).sum())
        ll = nn / 2 * (np.log(np.linalg.det(prec)) - np.trace(S @ prec))
        ebic = -2 * ll + k * np.log(nn) + 4 * 0.5 * k * np.log(len(cols))
        if best is None or ebic < best[0]:
            best = (ebic, alpha, prec)
    prec = best[2]
    d = np.sqrt(np.diag(prec))
    pc = -prec / np.outer(d, d); np.fill_diagonal(pc, 0)
    return pc, float(best[1]), int(len(cc))


def sym_audit(Md, Mh):
    """Path rules and completion on a symmetric relation, evaluated over the 276 dyads."""
    up = np.triu(OFF, 1)
    o = {T: r(X.paths(Md, X.T_FIXED[T])[up], Mh[up]) for T in ["product", "hamacher", "min", "yager2", "lukasiewicz"]}
    o["rank3"] = r(X.lowrank_baseline(Md, 3, dyad=True)[up], Mh[up])
    lam, _ = L.choose_lambda(Md, LAM_GRID)
    o["softimpute_cv"] = r(L.softimpute_dyad(Md, lam)[up], Mh[up])
    o["between_period_rho"] = r(Md[up], Mh[up])
    o["share_zero_dev"] = float((Md[up] <= 0).mean())
    o["median_dev"] = float(np.median(Md[up]))
    return o


if part == "measures":
    res = {}
    tic("pearson")
    Cd, Ch = wcorr(dev, CONCEPTS), wcorr(hold, CONCEPTS)
    res["pearson_positive"] = sym_audit(np.clip(Cd, 0, 1) * OFF, np.clip(Ch, 0, 1) * OFF)
    tic("yule")
    Qd, Qh = yule_q(dev, CONCEPTS), yule_q(hold, CONCEPTS)
    res["yule_q_positive"] = sym_audit(np.clip(Qd, 0, 1), np.clip(Qh, 0, 1))
    tic("glasso")
    Pd, ad, nd = glasso_partial(dev, CONCEPTS); Ph, ah, nh = glasso_partial(hold, CONCEPTS)
    res["partial_positive"] = sym_audit(np.clip(Pd, 0, 1), np.clip(Ph, 0, 1))
    res["partial_positive"].update({"alpha_dev": ad, "alpha_hold": ah, "n_complete_dev": nd, "n_complete_hold": nh,
                                    "n_nonzero_dev": int((np.abs(Pd[np.triu(OFF, 1)]) > 1e-8).sum()),
                                    "n_positive_dev": int((Pd[np.triu(OFF, 1)] > 1e-8).sum())})
    Gd = np.sqrt(Ed * Ed.T); Gh = np.sqrt(Eh * Eh.T)
    res["certainty_factor_symmetric"] = sym_audit(Gd, Gh)
    OUT["measures"] = res
    save()

if part == "signedcomp":
    def signed(df):
        S, S0, E = relation_matrices(df)
        Es = np.where(S0 < 1, (S - S0) / (1 - S0), 0); np.fill_diagonal(Es, 0)
        return Es
    Sd, Sh = signed(dev), signed(hold)
    lam, _ = L.choose_lambda(Sd, LAM_GRID)
    OUT["signedcomp"] = {"rank3_signed": r(X.lowrank_baseline(Sd, 3, dyad=True)[OFF], Sh[OFF]),
                         "softimpute_cv_signed": r(L.softimpute_dyad(Sd, lam)[OFF], Sh[OFF]),
                         "lambda": lam}
    save()

if part == "query":
    lam, _ = L.choose_lambda(Ed, LAM_GRID, transform="logit")
    f = L.path_features(Ed)
    preds = {"product": f["prod_max"], "hamacher": f["ham_max"], "min": f["min_max"],
             "rank3": X.lowrank_baseline(Ed, 3, dyad=True), "softimpute_logit_cv": L.softimpute_dyad(Ed, lam, transform="logit"),
             "direct_dev_edge": Ed}
    ci = {c: CONCEPTS.index(c) for c in CONCEPTS}
    mod = [ci[c] for c in MODIFIABLE]
    res = {"modifiable": MODIFIABLE, "per_symptom": {}, "summary": {}}
    for k in preds:
        res["summary"][k] = {"top1_agree": 0, "top2_overlap_mean": 0.0, "rho_mean": 0.0}
    for s in SYMPTOMS:
        j = ci[s]
        tgt = np.array([Eh[i, j] for i in mod])
        order_t = [MODIFIABLE[q] for q in np.argsort(-tgt)]
        row = {"heldout_order": order_t, "heldout_mu": {MODIFIABLE[q]: float(tgt[q]) for q in range(5)}}
        for k, M in preds.items():
            p = np.array([M[i, j] for i in mod])
            order_p = [MODIFIABLE[q] for q in np.argsort(-p, kind="stable")]
            row[k] = {"order": order_p, "values": {MODIFIABLE[q]: float(p[q]) for q in range(5)}}
            res["summary"][k]["top1_agree"] += int(order_p[0] == order_t[0])
            res["summary"][k]["top2_overlap_mean"] += len(set(order_p[:2]) & set(order_t[:2])) / 2 / len(SYMPTOMS)
            res["summary"][k]["rho_mean"] += r(p, tgt) / len(SYMPTOMS)
        res["per_symptom"][s] = row
    OUT["query"] = res
    save()

if part == "obesity":
    TYPE_ALT = dict(TYPE); TYPE_ALT["obesity"] = "Health"
    TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
    ti = np.array([TYPES.index(TYPE_ALT[c]) for c in CONCEPTS])
    blk = np.zeros((P, P), int); names = []
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    for q, (a, b) in enumerate(pairs):
        names.append(f"{TYPES[a]}-{TYPES[b]}")
        m = ((ti[:, None] == a) & (ti[None, :] == b)) | ((ti[:, None] == b) & (ti[None, :] == a))
        blk[m] = q
    per = {}
    for q in range(10):
        H = OFF & (blk == q)
        if H.sum() == 0:
            continue
        Em = np.where(H, 0.0, Ed)
        fb = L.path_features(Em)
        lb, _ = L.choose_lambda(Ed, LAM_GRID, base=OFF & ~H)
        per[names[q]] = {"n": int(H.sum()), "product": r(fb["prod_max"][H], Eh[H]), "hamacher": r(fb["ham_max"][H], Eh[H]),
                         "rank3": r(X.soft_impute(Ed, OFF & ~H, 3)[H], Eh[H]),
                         "softimpute_cv": r(L.soft_impute_reg(Ed, OFF & ~H, lb)[H], Eh[H]),
                         "katz": r(L.katz(Em, dyad=False)[H], Eh[H])}
    res = {"per_block": per}
    for comp in ["rank3", "softimpute_cv", "katz"]:
        w = [v["product"] > v[comp] for v in per.values()]
        res[f"product_vs_{comp}"] = {"better": int(sum(w)), "n": len(w), "p": float(binomtest(int(sum(w)), len(w)).pvalue)}
    res["median_product"] = float(np.median([v["product"] for v in per.values()]))
    res["median_rank3"] = float(np.median([v["rank3"] for v in per.values()]))
    A = np.where(Ed >= 0.01, Ed, 0.0); np.fill_diagonal(A, 0)
    W = A.sum(); so, si_ = A.sum(1), A.sum(0)
    def qf(memb):
        memb = np.asarray(memb)
        same = memb[:, None] == memb[None, :]
        return float(((A - np.outer(so, si_) / W) * same).sum() / W)
    res["type_partition_Q_alt"] = qf([TYPES.index(TYPE_ALT[c]) for c in CONCEPTS])
    res["type_partition_Q_main"] = qf([TYPES.index(TYPE[c]) for c in CONCEPTS])
    bp = json.load(open("results2/best_partition.json"))
    res["leiden_Q_check"] = qf([bp["membership"][bp["concepts"].index(c)] for c in CONCEPTS])
    OUT["obesity"] = res
    save()

if part == "initial":
    E0d = np.load("results2_pre_expert_review/E_dev.npy"); E0h = np.load("results2_pre_expert_review/E_hold.npy")
    o = {T: r(X.paths(E0d, X.T_FIXED[T])[OFF], E0h[OFF]) for T in ["product", "hamacher", "min", "yager2", "lukasiewicz"]}
    o["rank3"] = r(X.lowrank_baseline(E0d, 3, dyad=True)[OFF], E0h[OFF])
    lam, _ = L.choose_lambda(E0d, LAM_GRID, transform="logit")
    o["softimpute_logit_cv"] = r(L.softimpute_dyad(E0d, lam, transform="logit")[OFF], E0h[OFF])
    OUT["initial"] = o
    save()

if part == "uninsured":
    keep = [i for i, c in enumerate(CONCEPTS) if c != "uninsured"]
    sub = np.ix_(keep, keep)
    Edc, Ehc = Ed[sub], Eh[sub]
    Pn = P - 1; offc = ~np.eye(Pn, dtype=bool)
    saveP, saveOFF, saveB, saveTI = L.P, L.OFF, L.BLOCK, L.TI
    L.P, L.OFF, L.BLOCK, L.TI = Pn, offc, saveB[sub], saveTI[keep]
    try:
        o = {T: r(X.paths(Edc, X.T_FIXED[T])[offc], Ehc[offc]) for T in ["product", "hamacher", "min"]}
        o["rank3"] = r(X.lowrank_baseline(Edc, 3, dyad=True)[offc], Ehc[offc])
    finally:
        L.P, L.OFF, L.BLOCK, L.TI = saveP, saveOFF, saveB, saveTI
    o["mean_uninsured_dev"] = float(pdev[CONCEPTS.index("uninsured")])
    OUT["uninsured"] = o
    save()

if part == "misc":
    res = {}
    neff = []
    for i, j in itertools.permutations(range(P), 2):
        m = dev[CONCEPTS[i]].notna() & dev[CONCEPTS[j]].notna()
        z = (dev.w[m] * np.minimum(dev[CONCEPTS[i]][m], dev[CONCEPTS[j]][m])).to_numpy()
        neff.append(z.sum() ** 2 / (z ** 2).sum() if (z ** 2).sum() > 0 else 0.0)
    res["neff"] = {"min": float(np.min(neff)), "q25": float(np.quantile(neff, 0.25)), "median": float(np.median(neff)),
                   "q75": float(np.quantile(neff, 0.75)), "max": float(np.max(neff))}
    nobs = []
    for i, j in itertools.combinations(range(P), 2):
        nobs.append(int((dev[CONCEPTS[i]].notna() & dev[CONCEPTS[j]].notna()).sum()))
    res["respondents_per_pair_dev"] = {"min": int(min(nobs)), "max": int(max(nobs))}
    # MAE of tuned families
    fam = {"hamacher_0.35": X.t_hamacher_lam(0.35), "frank_0.1": X.t_frank(0.1), "ss_-0.25": X.t_ss(-0.25)}
    res["families"] = {k: {"rho": r(X.paths(Ed, T)[OFF], Eh[OFF]), "mae": float(np.abs(X.paths(Ed, T)[OFF] - Eh[OFF]).mean())} for k, T in fam.items()}
    st = X.strength_baseline(Ed)
    res["strength_scale"] = {"median_pred": float(np.median(st[OFF])), "median_target": float(np.median(Eh[OFF]))}
    # Life-Health block, rank-3 completion with the whole block withheld
    b = L.BLOCK_NAMES.index("Lifestyle–health")
    if b is not None:
        H = OFF & (L.BLOCK == b)
        out = {}
        for k in [1, 2, 3]:
            pr = X.soft_impute(Ed, OFF & ~H, k)
            out[f"rank{k}"] = {"rho": r(pr[H], Eh[H]), "share_negative": float((pr[H] < 0).mean()),
                               "pred_min": float(pr[H].min()), "pred_max": float(pr[H].max())}
        lb, _ = L.choose_lambda(Ed, LAM_GRID, base=OFF & ~H)
        pr = L.soft_impute_reg(Ed, OFF & ~H, lb)
        out["softimpute_cv"] = {"rho": r(pr[H], Eh[H]), "lambda": lb}
        out["target_range"] = [float(Eh[H].min()), float(Eh[H].max())]
        res["life_health_block"] = out
    res["block_names"] = list(L.BLOCK_NAMES)
    OUT["misc"] = res
    save()
