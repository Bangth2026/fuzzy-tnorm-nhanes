"""Audit on larger simulated graded graphs (v1.7.0).
Respondent-level samples with n = 48 or 96 concepts in 8 types; chained (sparse directed acyclic) or four-factor
latent structure; PHQ-like ordinal items for the first type and binary concepts otherwise. Relations use the same
survey-weighted certainty factor as the paper (equal weights). Designs: a random 25% of dyads withheld, and every
relation between two concept types withheld in turn (36 blocks). Paths use the exact cubic composition.
Writes results2/scale_sim.json."""
import json, time
import numpy as np
from scipy.stats import spearmanr
from graph2 import GRID
import rq3x as X
import rq6lib as L
from fast_compose import fast, T_prod, T_min, T_ham

LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]


def r(a, b):
    return float(spearmanr(a, b).statistic)


def relation(A):
    n = A.shape[1]
    G = np.array([(np.minimum(GRID[:, None], A[None, :, j])).mean(1) for j in range(n)])
    den = A.sum(0)
    S = np.array([np.minimum(A[:, i:i + 1], A).sum(0) / den[i] for i in range(n)])
    S0 = np.zeros((n, n))
    for j in range(n):
        S0[:, j] = np.interp(A, GRID, G[j]).sum(0) / den
    E = np.where(S0 < 1, np.clip((S - S0) / (1 - S0), 0, 1), 0)
    np.fill_diagonal(E, 0)
    return E


def measure(Z, ordinal):
    A = np.empty_like(Z)
    for c in range(Z.shape[1]):
        z = (Z[:, c] - Z[:, c].mean()) / Z[:, c].std()
        A[:, c] = np.digitize(z, np.quantile(z, [0.70, 0.88, 0.96])) / 3 if ordinal[c] else (z > np.quantile(z, 0.75)).astype(float)
    return A


def chain(g, n, p_edge):
    order = g.permutation(n); B = np.zeros((n, n))
    for a in range(1, n):
        for b in range(a):
            if g.random() < p_edge:
                B[order[b], order[a]] = g.uniform(0.3, 0.7)
    def sample(m):
        Z = np.zeros((m, n))
        for c in order:
            Z[:, c] = Z @ B[:, c] + g.standard_normal(m)
        return Z
    return sample


def factors(g, n, k):
    Lm = np.zeros((n, k))
    for c in range(n):
        Lm[c, g.integers(k)] = g.uniform(0.4, 0.8); Lm[c] += g.uniform(0, 0.25, k)
    return lambda m: g.standard_normal((m, k)) @ Lm.T + g.standard_normal((m, n)) * 0.8


def katz(E):
    n = len(E)
    lam = np.max(np.abs(np.linalg.eigvals(E))).real
    return np.linalg.inv(np.eye(n) - (0.5 / max(lam, 1e-9)) * E) - np.eye(n)


def one(sample, n, ntypes, g):
    ordinal = np.zeros(n, bool); ordinal[: n // ntypes] = True
    E1 = relation(measure(sample(15000), ordinal)); E2 = relation(measure(sample(15000), ordinal))
    off = ~np.eye(n, dtype=bool)
    typ = np.repeat(np.arange(ntypes), n // ntypes)
    out = {}
    # random 25% of dyads withheld
    dy = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pick = g.choice(len(dy), len(dy) // 4, replace=False)
    H = np.zeros((n, n), bool)
    for q in pick:
        i, j = dy[q]; H[i, j] = H[j, i] = True
    A = np.where(H, 0.0, E1)
    pr = {"product": fast(A, T_prod), "hamacher": fast(A, T_ham), "min": fast(A, T_min), "katz": katz(A),
          "rank3": X.soft_impute(E1, off & ~H, 3), "rank_types": X.soft_impute(E1, off & ~H, ntypes)}
    out.update({f"random25_{k}": r(v[H], E2[H]) for k, v in pr.items()})
    # whole relation types withheld
    pb = {k: [] for k in ["product", "hamacher", "min", "katz", "rank3", "rank_types"]}; nb = []
    for a in range(ntypes):
        for b in range(a, ntypes):
            H = off & (((typ[:, None] == a) & (typ[None, :] == b)) | ((typ[:, None] == b) & (typ[None, :] == a)))
            A = np.where(H, 0.0, E1)
            pr = {"product": fast(A, T_prod), "hamacher": fast(A, T_ham), "min": fast(A, T_min), "katz": katz(A),
                  "rank3": X.soft_impute(E1, off & ~H, 3), "rank_types": X.soft_impute(E1, off & ~H, ntypes)}
            for k, v in pr.items():
                pb[k].append(r(v[H], E2[H]))
            nb.append(H.sum())
    nb = np.array(nb, float)
    for k, v in pb.items():
        v = np.array(v); ok = ~np.isnan(v)
        out[f"block_{k}"] = float((v[ok] * nb[ok]).sum() / nb[ok].sum())
    out["density"] = float((E1[off] >= 0.01).mean())
    return out


res = {}
for n in [48, 96]:
    for name in ["chain", "factor4"]:
        rows = []
        for rep in range(8):
            g = np.random.default_rng(800 + rep + n)
            s = chain(g, n, 6.0 / n) if name == "chain" else factors(g, n, 4)
            rows.append(one(s, n, 8, g))
        key = f"n{n}_{name}"
        res[key] = {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]}
        for d in ["random25", "block"]:
            res[key][f"{d}_share_product_above_rank3"] = float(np.mean([x[f"{d}_product"] > x[f"{d}_rank3"] for x in rows]))
            res[key][f"{d}_share_product_best_tnorm"] = float(np.mean([x[f"{d}_product"] >= max(x[f"{d}_hamacher"], x[f"{d}_min"]) for x in rows]))
        print(time.strftime("%H:%M:%S"), key, {k: round(v, 3) for k, v in res[key].items()}, flush=True)
        json.dump(res, open("results2/scale_sim.json", "w"), indent=1)
