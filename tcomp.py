"""Path-consistent completion (PCC) and related variants.

PCC (used in the paper): tc_complete(R, mask, lam, gamma, "product", two_sided=True, cells="missing") minimises
    1/2 ||P_Omega(X - R)||_F^2 + lam ||X - m||_* + gamma/2 sum_{(i,j) not in Omega} (max_k T(X_ik, X_kj) - X_ij)^2
over X in [0,1]^{n x n} by proximal gradient (m: mean of the observed cells). gamma = 0 is raw-scale soft-impute.

Variants tried during development and not adopted (kept for the record): the one-sided T-transitivity penalty
[max_k T(X_ik, X_kj) - X_ij]_+ on all cells (two_sided=False, cells="all"); pcc_logit (low-rank part on the logit scale);
per-cell weights from the observation support of each cell (support, mu_matrix).
"""
import numpy as np

EPS = 1e-12


def _tnorm_and_grads(name):
    """Return T(a, b) and its partial derivatives dT/da, dT/db (subgradients at kinks)."""
    if name == "product":
        return (lambda a, b: a * b), (lambda a, b: b), (lambda a, b: a)
    if name == "min":
        return (np.minimum, lambda a, b: (a <= b).astype(float), lambda a, b: (b < a).astype(float))
    if name == "hamacher":  # Hamacher product T(a,b) = ab / (a + b - ab)
        def T(a, b):
            d = a + b - a * b
            return np.where(d > EPS, a * b / np.maximum(d, EPS), 0.0)

        def da(a, b):
            d = a + b - a * b
            return np.where(d > EPS, b * b / np.maximum(d, EPS) ** 2, 0.0)

        def db(a, b):
            d = a + b - a * b
            return np.where(d > EPS, a * a / np.maximum(d, EPS) ** 2, 0.0)
        return T, da, db
    if name == "lukasiewicz":
        return (lambda a, b: np.maximum(0.0, a + b - 1), lambda a, b: (a + b > 1).astype(float),
                lambda a, b: (a + b > 1).astype(float))
    raise ValueError(name)


_MASKS = {}


def _triple_mask(n):
    if n not in _MASKS:
        I = np.arange(n)
        _MASKS[n] = (I[:, None, None] != I[None, :, None]) & (I[None, :, None] != I[None, None, :]) & \
                    (I[:, None, None] != I[None, None, :])
    return _MASKS[n]


def best_path(X, tnorm="product"):
    """max_k T(X_ik, X_kj) over intermediates k not in {i, j}, and the argmax k."""
    T, _, _ = _tnorm_and_grads(tnorm)
    n = X.shape[0]
    V = np.where(_triple_mask(n), T(X[:, :, None], X[None, :, :]), -1.0)  # (i, k, j)
    k = V.argmax(1)
    best = np.take_along_axis(V, k[:, None, :], 1)[:, 0, :]
    best = np.maximum(best, 0.0)
    np.fill_diagonal(best, 0.0)
    return best, k


def violation(X, tnorm="product", tol=0.0):
    """Share of ordered pairs whose best two-edge path exceeds the relation by more than tol, and the mean excess."""
    b, _ = best_path(X, tnorm)
    off = ~np.eye(X.shape[0], dtype=bool)
    ex = (b - X)[off]
    return float((ex > tol).mean()), float(np.maximum(ex, 0).mean())


def tc_complete(R, mask, lam, mu, tnorm="product", iters=400, tol=1e-6, X0=None, two_sided=False, cells="all"):
    """Proximal-gradient solver for the TC objective. mask: True where R is observed (diagonal excluded)."""
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    T, dTa, dTb = _tnorm_and_grads(tnorm)
    m = R[mask].mean()
    X = np.where(mask, R, m) if X0 is None else X0.copy()
    mu_max = float(np.max(mu))
    step = 1.0 / (1.0 + 3.0 * mu_max)        # conservative step for the smooth part
    I = np.arange(n)
    for it in range(iters):
        G = np.where(mask, X - R, 0.0)
        if mu_max > 0:
            b, k = best_path(X, tnorm)
            sel = off if cells == "all" else (off & ~mask)
            phi = np.where(sel, (b - X) if two_sided else np.maximum(b - X, 0.0), 0.0)
            if phi.any():
                ii, jj = np.nonzero(phi)
                kk = k[ii, jj]
                a, c = X[ii, kk], X[kk, jj]
                w = (mu[ii, jj] if np.ndim(mu) else mu) * phi[ii, jj]
                G[ii, jj] -= w
                np.add.at(G, (ii, kk), w * dTa(a, c))
                np.add.at(G, (kk, jj), w * dTb(a, c))
        Y = X - step * G - m
        U, s, Vt = np.linalg.svd(Y, full_matrices=False)
        s = np.maximum(s - step * lam, 0.0)
        Xn = np.clip((U * s) @ Vt + m, 0.0, 1.0)
        if np.abs(Xn - X).max() < tol:
            X = Xn
            break
        X = Xn
    X = X.copy()
    np.fill_diagonal(X, 0.0)
    return X


def tc_dyad(R, lam, mu, tnorm="product", iters=400):
    """Leave-dyad-out predictions: for every unordered pair, mask both directions, complete, read both cells."""
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    out = np.zeros_like(R)
    for i in range(n):
        for j in range(i + 1, n):
            msk = off.copy(); msk[i, j] = msk[j, i] = False
            X = tc_complete(R, msk, lam, mu, tnorm, iters)
            out[i, j], out[j, i] = X[i, j], X[j, i]
    return out


def tc_cv(R, lams, mus, tnorms=("product",), folds=5, seed=0, base=None, iters=400):
    """Choose (tnorm, lam, mu) by masking random dyads among the observed off-diagonal pairs."""
    from scipy.stats import spearmanr
    n = R.shape[0]
    base = (~np.eye(n, dtype=bool)) if base is None else base
    rng = np.random.default_rng(seed)
    dy = [(i, j) for i in range(n) for j in range(i + 1, n) if base[i, j] and base[j, i]]
    idx = rng.permutation(len(dy))
    score = {}
    for f in range(folds):
        msk = base.copy()
        for q in idx[f::folds]:
            i, j = dy[q]; msk[i, j] = msk[j, i] = False
        H = base & ~msk
        for t in tnorms:
            for lam in lams:
                for mu in mus:
                    X = tc_complete(R, msk, lam, mu, t, iters)
                    score.setdefault((t, lam, mu), []).append(float(spearmanr(X[H], R[H]).statistic))
    mean = {k: float(np.mean(v)) for k, v in score.items()}
    return max(mean, key=mean.get), mean


C_LOGIT = 0.005


def _to_logit(R):
    return np.log((R + C_LOGIT) / (1 - R + C_LOGIT))


def _from_logit(Z):
    e = np.exp(np.clip(Z, -30, 30))
    return np.clip((e * (1 + C_LOGIT) - C_LOGIT) / (1 + e), 0.0, 1.0), (1 + 2 * C_LOGIT) * e / (1 + e) ** 2


def pcc_logit(R, mask, lam, mu, tnorm="product", iters=400, tol=1e-6):
    """Path-consistent completion with the low-rank part on the logit scale (as logit soft-impute) and the path term on
    the relation scale:  F(Z) = 1/2 ||P_Omega(Z - logit R)||^2 + lam ||Z - m||_* + mu/2 sum_{(i,j) not in Omega}
    (max_k T(X_ik, X_kj) - X_ij)^2,  X = g(Z).  mu = 0 is logit soft-impute."""
    n = R.shape[0]
    off = ~np.eye(n, dtype=bool)
    T, dTa, dTb = _tnorm_and_grads(tnorm)
    Y = _to_logit(R)
    m = Y[mask].mean()
    Z = np.where(mask, Y, m)
    sel = off & ~mask
    for it in range(iters):
        G = np.where(mask, Z - Y, 0.0)
        if mu > 0:
            Xv, dX = _from_logit(Z)
            b, k = best_path(Xv, tnorm)
            phi = np.where(sel, b - Xv, 0.0)
            if phi.any():
                GX = np.zeros_like(Z)
                ii, jj = np.nonzero(phi)
                kk = k[ii, jj]
                a, c = Xv[ii, kk], Xv[kk, jj]
                w = mu * phi[ii, jj]
                GX[ii, jj] -= w
                np.add.at(GX, (ii, kk), w * dTa(a, c))
                np.add.at(GX, (kk, jj), w * dTb(a, c))
                G += GX * dX
        step = 1.0 / (1.0 + 0.75 * mu)
        U, s, Vt = np.linalg.svd(Z - step * G - m, full_matrices=False)
        s = np.maximum(s - step * lam, 0.0)
        Zn = (U * s) @ Vt + m
        if np.abs(Zn - Z).max() < tol:
            Z = Zn
            break
        Z = Zn
    Xv, _ = _from_logit(Z)
    Xv = Xv.copy(); np.fill_diagonal(Xv, 0.0)
    return Xv


def support(mask, block):
    """Observation support of every cell: the smallest of the observed shares of its row, its column and its block
    (relation type). Near 1 when gaps are scattered; 0 for a withheld block; about 1/(n-1) for a new concept."""
    n = mask.shape[0]
    off = ~np.eye(n, dtype=bool)
    row = (mask & off).sum(1) / off.sum(1)
    col = (mask & off).sum(0) / off.sum(0)
    blk = np.zeros((n, n))
    for b in np.unique(block[off]):
        sel = off & (block == b)
        blk[sel] = (mask & sel).sum() / sel.sum()
    return np.minimum(np.minimum(row[:, None], col[None, :]), blk)


def mu_matrix(mask, block, mu_lo, mu_hi):
    return mu_lo + (mu_hi - mu_lo) * (1.0 - support(mask, block))
