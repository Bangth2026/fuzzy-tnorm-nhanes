"""Replicate-level and query-level checks (v1.10.0).
Parts:
  blockreps  - whole relation types withheld: the 90 JRR replicate values of product - rank-3 completion
               (rank fixed at 3, as in main Table III) and of product - rank-k completion with k re-chosen by dyad
               cross-validation on the other blocks in every replicate; per-block contributions to the JRR
               variance; product - Katz and product - Hamacher for comparison
  querygap   - illustrative query: for each PHQ-9 symptom, the development values of mu(c -> s) for the five
               modifiable correlates, the gap between the first and second, and, in the 20 split halves used by
               rigor_checks.py querytab, the top correlate and the gap in each half
Writes results2/repl_<part>.json."""
import json, sys, time
import numpy as np
from scipy.stats import spearmanr
from analysis2 import dev, hold
from graph2 import CONCEPTS, TYPE, P, relation_matrices
import rq3x as X
import rq6lib as L

part = sys.argv[1]
OFF = ~np.eye(P, dtype=bool)
OUT = {}
MODIFIABLE = ["smoking", "heavy_drinking", "inactivity", "short_sleep", "obesity"]
SYMPTOMS = [c for c in CONCEPTS if TYPE[c] == "Symptom"]


def tic(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save():
    json.dump(OUT, open(f"results2/repl_{part}.json", "w"), indent=1, default=float)


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


def rank_cv(R, ks=(1, 2, 3, 4, 5), folds=5, base=None, seed=0):
    """Same procedure as audit_extensions.rank_cv: five-fold dyad cross-validation on the observed cells."""
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


Ed, Eh = rel(dev), rel(hold)

# ------------------------------------------------------------------------------------------------
if part == "blockreps":
    nb = np.array([(OFF & (L.BLOCK == b)).sum() for b in range(10)], float)

    def stats(E1, E2):
        pp = {k: np.zeros_like(E1) for k in ["product", "hamacher", "katz", "rank3", "rankcv"]}
        ranks = []
        for b in range(10):
            H = OFF & (L.BLOCK == b)
            Em = np.where(H, 0.0, E1)
            pp["product"][H] = X.paths(Em, X.t_product)[H]
            pp["hamacher"][H] = X.paths(Em, X.T_FIXED["hamacher"])[H]
            pp["katz"][H] = L.katz(Em, dyad=False)[H]
            pp["rank3"][H] = X.soft_impute(E1, OFF & ~H, 3)[H]
            k, _ = rank_cv(E1, base=OFF & ~H)
            ranks.append(int(k))
            pp["rankcv"][H] = X.soft_impute(E1, OFF & ~H, k)[H]
        o = {"ranks_chosen": ranks}
        for k, v in pp.items():
            o[k] = [r(v[OFF & (L.BLOCK == b)], E2[OFF & (L.BLOCK == b)]) for b in range(10)]
            o[f"weighted_{k}"] = float((np.array(o[k]) * nb).sum() / nb.sum())
        return o

    tic("full sample")
    full = stats(Ed, Eh)
    reps = []
    dr, hr = jrr_reps(dev), jrr_reps(hold)
    for q, d in enumerate(dr):
        reps.append(stats(rel(d), Eh)); tic(f"dev replicate {q + 1}/{len(dr)}")
    for q, h in enumerate(hr):
        reps.append(stats(Ed, rel(h))); tic(f"held-out replicate {q + 1}/{len(hr)}")

    res = {"block_names": L.BLOCK_NAMES, "n_pairs": nb.tolist(), "full": full, "replicates": reps}
    for k in ["rank3", "rankcv", "katz", "hamacher"]:
        d0 = full["weighted_product"] - full[f"weighted_{k}"]
        dv = np.array([x["weighted_product"] - x[f"weighted_{k}"] for x in reps])
        se = float(np.sqrt(((dv - d0) ** 2).sum()))
        # contribution of each block to the JRR variance of the weighted difference
        contrib = []
        for b in range(10):
            db0 = (full["product"][b] - full[k][b]) * nb[b] / nb.sum()
            dbv = np.array([(x["product"][b] - x[k][b]) * nb[b] / nb.sum() for x in reps])
            contrib.append(float(((dbv - db0) ** 2).sum()))
        res[f"product_minus_{k}"] = {
            "point": d0, "se": se, "ci95": [d0 - 1.96 * se, d0 + 1.96 * se],
            "replicate_values": dv.tolist(),
            "replicate_quantiles": {str(p): float(np.quantile(dv, p)) for p in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
            "n_replicates_below_zero": int((dv < 0).sum()),
            "max_abs_deviation": float(np.abs(dv - d0).max()),
            "largest_deviations": sorted([(float(abs(v - d0)), int(i)) for i, v in enumerate(dv)], reverse=True)[:5],
            "block_contribution_to_sum_of_squares": contrib,
            "block_share_of_sum_of_squares": (np.array(contrib) / max(sum(contrib), 1e-300)).tolist()}
    # per-block rank-3 rho across replicates: range and sign changes
    res["rank3_block_rho_range"] = {L.BLOCK_NAMES[b]: [float(min(x["rank3"][b] for x in reps)), float(max(x["rank3"][b] for x in reps))]
                                    for b in range(10)}
    res["ranks_chosen_full"] = full["ranks_chosen"]
    res["ranks_chosen_replicates_counts"] = {str(k): int(sum(x["ranks_chosen"].count(k) for x in reps)) for k in range(1, 6)}
    OUT["blockreps"] = res
    save()

# ------------------------------------------------------------------------------------------------
if part == "querygap":
    ci = {c: CONCEPTS.index(c) for c in CONCEPTS}
    mod = [ci[c] for c in MODIFIABLE]

    def vals(E, s):
        return np.array([E[i, ci[s]] for i in mod])

    def top2(v):
        o = np.argsort(-v, kind="stable")
        return MODIFIABLE[o[0]], MODIFIABLE[o[1]], float(v[o[0]] - v[o[1]])

    res = {}
    halves = split_halves(dev, 20, 4)
    EH = [(rel(A), rel(B)) for A, B in halves]
    for s in SYMPTOMS:
        vd, vh = vals(Ed, s), vals(Eh, s)
        t1, t2, gap = top2(vd)
        h1, h2, hgap = top2(vh)
        per = []
        for EA, EB in EH:
            a1, a2, ag = top2(vals(EA, s)); b1, b2, bg = top2(vals(EB, s))
            per.append({"top_A": a1, "gap_A": ag, "top_B": b1, "gap_B": bg, "same_top": a1 == b1})
        # deviations of the halves from the full sample in the full-sample top-1 minus top-2 difference
        i1, i2 = MODIFIABLE.index(t1), MODIFIABLE.index(t2)
        dA = np.array([vals(EA, s)[i1] - vals(EA, s)[i2] - gap for EA, _ in EH])
        dB = np.array([vals(EB, s)[i1] - vals(EB, s)[i2] - gap for _, EB in EH])
        res[s] = {"dev_values": dict(zip(MODIFIABLE, vd.tolist())), "dev_top1": t1, "dev_top2": t2, "dev_gap": gap,
                  "hold_top1": h1, "hold_top2": h2, "hold_gap": hgap,
                  "halves_same_top_share": float(np.mean([p["same_top"] for p in per])),
                  "halves_median_gap": float(np.median([p["gap_A"] for p in per] + [p["gap_B"] for p in per])),
                  "halves_corr_of_gap_deviations": float(np.corrcoef(dA, dB)[0, 1]),
                  "halves": per}
    OUT["querygap"] = res
    save()
