"""Vectorised multi-hop inference and non-path baselines for the second review (revision2.py)."""
import itertools
import numpy as np
from scipy.stats import spearmanr

EPS_T = 1e-12


# ---------------- t-norms (fixed and parametric) ----------------
def t_product(x, y):
    return x * y


def t_min(x, y):
    return np.minimum(x, y)


def t_luk(x, y):
    return np.maximum(0.0, x + y - 1.0)


def t_yager2(x, y):
    return np.maximum(0.0, 1 - np.sqrt((1 - x) ** 2 + (1 - y) ** 2))


def t_hamacher_lam(lam):
    """Hamacher family T_lam(x, y) = xy / (lam + (1 - lam)(x + y - xy)), lam >= 0; lam = 0 Hamacher product, lam = 1 product."""
    def f(x, y):
        den = lam + (1 - lam) * (x + y - x * y)
        return np.where(den > 0, x * y / np.maximum(den, EPS_T), 0.0)
    return f


def t_frank(s):
    """Frank family; s -> 0 minimum, s = 1 product, s -> inf Lukasiewicz."""
    if abs(s - 1) < 1e-9:
        return t_product
    ls = np.log(s)

    def f(x, y):
        a = np.expm1(x * ls) * np.expm1(y * ls) / np.expm1(ls)
        return np.clip(np.log1p(a) / ls, 0, 1)
    return f


def t_ss(p):
    """Schweizer-Sklar family; p -> 0 product, p = 1 Lukasiewicz, p -> -inf minimum."""
    if abs(p) < 1e-9:
        return t_product

    def f(x, y):
        if p < 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                xx, yy = np.where(x > 0, x, np.nan) ** p, np.where(y > 0, y, np.nan) ** p
                v = (xx + yy - 1) ** (1 / p)
            return np.where((x > 0) & (y > 0), np.nan_to_num(v), 0.0)
        v = np.maximum(0.0, x ** p + y ** p - 1)
        return v ** (1 / p)
    return f


T_FIXED = {"product": t_product, "hamacher": t_hamacher_lam(0.0), "yager2": t_yager2,
           "lukasiewicz": t_luk, "min": t_min}


# ---------------- path inference ----------------
def _masks(n):
    I = np.arange(n)
    # L2 tensor index (i, k, j): k not in {i, j}, i != j
    m2 = (I[:, None, None] != I[None, :, None]) & (I[None, :, None] != I[None, None, :]) & (I[:, None, None] != I[None, None, :])
    # L3 tensor index (i, k, l, j): k, l distinct, not in {i, j}, i != j
    i_, k_, l_, j_ = I[:, None, None, None], I[None, :, None, None], I[None, None, :, None], I[None, None, None, :]
    m3 = (k_ != i_) & (k_ != j_) & (l_ != i_) & (l_ != j_) & (k_ != l_) & (i_ != j_)
    return m2, m3


_MASK = {}


def paths(R, T, agg="max", max_len=3):
    """Strength of indirect paths of length 2..max_len for every ordered pair, never using the direct edge.
    agg='max' takes the best path (Goedel t-conorm); agg='probsum' combines all paths by noisy-OR."""
    n = R.shape[0]
    if n not in _MASK:
        _MASK[n] = _masks(n)
    m2, m3 = _MASK[n]
    V2 = np.where(m2, T(R[:, :, None], R[None, :, :]), 0.0)                    # (i, k, j)
    if max_len >= 3:
        P2 = T(R[:, :, None], R[None, :, :])                                   # (i, k, l)
        V3 = np.where(m3, T(P2[:, :, :, None], R[None, None, :, :]), 0.0)      # (i, k, l, j)
    if agg == "max":
        out = V2.max(1)
        if max_len >= 3:
            out = np.maximum(out, V3.max((1, 2)))
        best3_share = None
    else:
        lg = np.log1p(-np.clip(V2, 0, 1 - 1e-12)).sum(1)
        if max_len >= 3:
            lg = lg + np.log1p(-np.clip(V3, 0, 1 - 1e-12)).sum((1, 2))
        out = -np.expm1(lg)
    np.fill_diagonal(out, 0)
    return out


def best_path_length_share(R, T):
    n = R.shape[0]
    if n not in _MASK:
        _MASK[n] = _masks(n)
    m2, m3 = _MASK[n]
    b2 = np.where(m2, T(R[:, :, None], R[None, :, :]), 0.0).max(1)
    P2 = T(R[:, :, None], R[None, :, :])
    b3 = np.where(m3, T(P2[:, :, :, None], R[None, None, :, :]), 0.0).max((1, 2))
    off = ~np.eye(n, dtype=bool)
    return float((b3[off] > b2[off]).mean())


# ---------------- non-path baselines ----------------
def strength_baseline(R):
    so, si = R.sum(1), R.sum(0)
    out = (so[:, None] - R) * (si[None, :] - R)
    np.fill_diagonal(out, 0)
    return out


def block_baseline(R, block):
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    out = np.zeros_like(R)
    for i, j in itertools.permutations(range(n), 2):
        m = (block == block[i, j]) & off
        m[i, j] = False
        out[i, j] = R[m].mean() if m.any() else 0.0
    return out


def dcsbm_baseline(R, groups, dyad=False):
    """Degree-corrected block model (Karrer-Newman form) by concept type, refitted without the held-out cell(s)."""
    n = R.shape[0]
    g = np.asarray(groups)
    G = np.unique(g)
    out = np.zeros_like(R)
    for i, j in itertools.permutations(range(n), 2):
        Rm = R.copy()
        Rm[i, j] = 0
        if dyad:
            Rm[j, i] = 0
        so, si = Rm.sum(1), Rm.sum(0)
        gi, gj = g[i], g[j]
        Sout = so[g == gi].sum()
        Sin = si[g == gj].sum()
        mgh = Rm[np.ix_(g == gi, g == gj)].sum()
        out[i, j] = (so[i] / Sout if Sout > 0 else 0) * (si[j] / Sin if Sin > 0 else 0) * mgh
    return out


def reverse_baseline(R, prev, dyad_fit=True):
    """Reverse edge corrected by the prevalence odds: log mu(i->j) = log mu(j->i) + b * (logit p_j - logit p_i) + c,
    with b and c estimated from the other dyads with both directions positive (leave-dyad-out)."""
    n = R.shape[0]
    lg = np.log(prev / (1 - prev))
    pairs = [(i, j) for i, j in itertools.permutations(range(n), 2) if R[i, j] > 0 and R[j, i] > 0]
    X = np.array([lg[j] - lg[i] for i, j in pairs])
    Y = np.array([np.log(R[i, j]) - np.log(R[j, i]) for i, j in pairs])
    key = {p: k for k, p in enumerate(pairs)}
    out = np.zeros_like(R)
    coef_full = np.polyfit(X, Y, 1)
    for i, j in itertools.permutations(range(n), 2):
        if R[j, i] <= 0:
            out[i, j] = 0.0
            continue
        if dyad_fit:
            drop = [key[p] for p in [(i, j), (j, i)] if p in key]
            keep = np.ones(len(pairs), bool)
            keep[drop] = False
            b, c = np.polyfit(X[keep], Y[keep], 1)
        else:
            b, c = coef_full
        out[i, j] = min(1.0, R[j, i] * np.exp(b * (lg[j] - lg[i]) + c))
    return out, {"slope": float(coef_full[0]), "intercept": float(coef_full[1]), "n_pairs": len(pairs)}


def soft_impute(R, mask, k, iters=200, tol=1e-7):
    """Rank-k completion of R observed on `mask` (True = observed) by iterated truncated SVD."""
    X = np.where(mask, R, 0.0)
    fill = R[mask].mean()
    X = np.where(mask, R, fill)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        L = (U[:, :k] * s[:k]) @ Vt[:k]
        Xn = np.where(mask, R, L)
        if np.abs(Xn - X).max() < tol:
            X = Xn
            break
        X = Xn
    return np.clip(L, 0, 1)


def lowrank_baseline(R, k, dyad=True, iters=200):
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    out = np.zeros_like(R)
    done = {}
    for i, j in itertools.permutations(range(n), 2):
        key = (min(i, j), max(i, j)) if dyad else (i, j)
        if key not in done:
            m = off.copy()
            m[i, j] = False
            if dyad:
                m[j, i] = False
            done[key] = soft_impute(R, m, k, iters)
        out[i, j] = done[key][i, j]
    return out


def rho(x, y):
    return float(spearmanr(x, y).statistic)


def bc_ci(boot, point, alpha=0.05):
    """Bias-corrected percentile interval (Efron 1981)."""
    from scipy.stats import norm
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    p0 = (np.sum(b < point) + 0.5 * np.sum(b == point)) / len(b)
    p0 = min(max(p0, 1 / (2 * len(b))), 1 - 1 / (2 * len(b)))
    z0 = norm.ppf(p0)
    lo, hi = norm.cdf(2 * z0 + norm.ppf(alpha / 2)), norm.cdf(2 * z0 + norm.ppf(1 - alpha / 2))
    return [float(np.quantile(b, lo)), float(np.quantile(b, hi))]
