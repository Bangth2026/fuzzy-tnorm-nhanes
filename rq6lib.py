"""Predictors and evaluation used by revision6.py (sixth review): stronger non-path baselines,
link-prediction indices, a learned path aggregator, within-block and per-source evaluation."""
import itertools
import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.decomposition import NMF
from sklearn.isotonic import IsotonicRegression
import rq3x as X
from graph2 import CONCEPTS, TYPE

P = len(CONCEPTS)
OFF = ~np.eye(P, dtype=bool)
TYP = np.array([TYPE[c] for c in CONCEPTS])
TYPES = ["Symptom", "Socioeconomic", "Lifestyle", "Health"]
TI = np.array([TYPES.index(t) for t in TYP])


def rho(a, b):
    return float(spearmanr(a, b).statistic)


# ---------------- blocks ----------------
def undirected_block_id():
    """10 undirected type blocks (both directions)."""
    B = np.full((P, P), -1)
    pairs = [(a, b) for a in range(4) for b in range(a, 4)]
    for i, j in itertools.permutations(range(P), 2):
        a, b = sorted((TI[i], TI[j]))
        B[i, j] = pairs.index((a, b))
    return B, [f"{TYPES[a]}–{TYPES[b].lower()}" for a, b in pairs]


BLOCK, BLOCK_NAMES = undirected_block_id()


# ---------------- path-based ----------------
def path_features(R):
    """Features of indirect paths (length 2-3, never using i->j or j->i) for every ordered pair."""
    n = R.shape[0]
    f = {}
    f["prod_max"] = X.paths(R, X.t_product)
    f["prod_max2"] = X.paths(R, X.t_product, max_len=2)
    f["ham_max"] = X.paths(R, X.T_FIXED["hamacher"])
    f["min_max"] = X.paths(R, X.t_min)
    m2 = (np.arange(n)[:, None, None] != np.arange(n)[None, :, None]) & (np.arange(n)[None, :, None] != np.arange(n)[None, None, :])
    V2 = np.where(m2, R[:, :, None] * R[None, :, :], 0.0)            # (i,k,j)
    f["cn_sum"] = V2.sum(1)                                          # weighted common neighbours
    s = R.sum(1) + R.sum(0)
    f["adamic_adar"] = (V2 / np.log(2 + s)[None, :, None]).sum(1)
    f["resource_alloc"] = (V2 / np.maximum(s, 1e-9)[None, :, None]).sum(1)
    srt = np.sort(V2, axis=1)[:, ::-1, :]
    f["prod_top3_mean"] = srt[:, :3, :].mean(1)
    f["n_paths_gt_0.01"] = (V2 > 0.01).sum(1).astype(float)
    for k in f:
        np.fill_diagonal(f[k], 0)
    return f


def best_path_product(R, max_len=4):
    """Best simple path of length 2..max_len under the product, never using i->j or j->i.
    With weights <= 1, cycles cannot raise a product, so the best walk avoiding i and j as
    intermediates is a simple path; computed by max-times iteration with i, j removed."""
    n = R.shape[0]
    out = np.zeros_like(R)
    for i, j in itertools.permutations(range(n), 2):
        keep = [k for k in range(n) if k not in (i, j)]
        Rk = R[np.ix_(keep, keep)]
        a = R[i, keep]                     # i -> k (one edge)
        b = R[keep, j]                     # k -> j
        best = (a * b).max()               # length 2
        cur = a
        for _ in range(max_len - 2):       # extend to length 3, 4, ...
            cur = (cur[:, None] * Rk).max(0)
            best = max(best, (cur * b).max())
        out[i, j] = best
    return out


def katz(R, beta=None, dyad=True):
    n = R.shape[0]
    lam = np.max(np.abs(np.linalg.eigvals(R))).real
    if beta is None:
        beta = 0.5 / max(lam, 1e-9)
    out = np.zeros_like(R)
    I = np.eye(n)
    for i, j in itertools.permutations(range(n), 2):
        Rm = R.copy()
        if dyad:
            Rm[i, j] = Rm[j, i] = 0
        K = np.linalg.inv(I - beta * Rm) - I
        out[i, j] = K[i, j] - beta * Rm[i, j]
    return out


def ppr(R, alpha=0.15, dyad=True):
    n = R.shape[0]
    out = np.zeros_like(R)
    for i, j in itertools.permutations(range(n), 2):
        Rm = R.copy()
        if dyad:
            Rm[i, j] = Rm[j, i] = 0
        rs = Rm.sum(1, keepdims=True)
        Pm = np.where(rs > 0, Rm / np.maximum(rs, 1e-12), 1.0 / n)
        e = np.zeros(n); e[i] = 1
        x = np.linalg.solve(np.eye(n) - (1 - alpha) * Pm.T, alpha * e)
        out[i, j] = x[j]
    return out


# ---------------- non-path ----------------
def soft_impute_reg(R, mask, lam, iters=300, tol=1e-6, transform=None):
    """Soft-impute (Mazumder et al. 2010): nuclear-norm regularised completion."""
    Y = R.copy()
    if transform == "logit":
        Y = np.log((R + 0.005) / (1 - R + 0.005))
    elif transform == "log":
        Y = np.log(R + 0.005)
    mu = Y[mask].mean()
    Z = np.where(mask, Y - mu, 0.0)
    L = np.zeros_like(Y)
    for _ in range(iters):
        Xf = np.where(mask, Y - mu, L)
        U, s, Vt = np.linalg.svd(Xf, full_matrices=False)
        s = np.maximum(s - lam, 0)
        Ln = (U * s) @ Vt
        if np.abs(Ln - L).max() < tol:
            L = Ln
            break
        L = Ln
    out = L + mu
    if transform == "logit":
        e = np.exp(out)
        out = (e * (1 + 0.005) - 0.005) / (1 + e)
    elif transform == "log":
        out = np.exp(out) - 0.005
    return np.clip(out, 0, 1)


def choose_lambda(R, grid, transform=None, folds=5, seed=0, base=None):
    """Choose lambda by masking random dyads among observed off-diagonal pairs (within development)."""
    rng = np.random.default_rng(seed)
    base = OFF if base is None else base
    dy = [(i, j) for i in range(P) for j in range(i + 1, P) if base[i, j] and base[j, i]]
    idx = rng.permutation(len(dy))
    score = {l: [] for l in grid}
    for f in range(folds):
        test = [dy[k] for k in idx[f::folds]]
        m = base.copy()
        for i, j in test:
            m[i, j] = m[j, i] = False
        H = base & ~m
        for l in grid:
            L = soft_impute_reg(R, m, l, transform=transform)
            score[l].append(rho(L[H], R[H]))
    mean = {l: float(np.mean(v)) for l, v in score.items()}
    return max(mean, key=mean.get), mean


def softimpute_dyad(R, lam, transform=None):
    out = np.zeros_like(R)
    for i in range(P):
        for j in range(i + 1, P):
            m = OFF.copy(); m[i, j] = m[j, i] = False
            L = soft_impute_reg(R, m, lam, transform=transform)
            out[i, j], out[j, i] = L[i, j], L[j, i]
    return out


def nmf_dyad(R, k=3, iters=300):
    """Non-negative factorisation with the dyad masked (weighted multiplicative updates)."""
    out = np.zeros_like(R)
    rng = np.random.default_rng(0)
    for i in range(P):
        for j in range(i + 1, P):
            Wm = OFF.astype(float); Wm[i, j] = Wm[j, i] = 0
            A = rng.random((P, k)) * 0.3 + 0.05
            B = rng.random((k, P)) * 0.3 + 0.05
            for _ in range(iters):
                AB = A @ B
                A *= ((Wm * R) @ B.T) / np.maximum((Wm * AB) @ B.T, 1e-12)
                AB = A @ B
                B *= (A.T @ (Wm * R)) / np.maximum(A.T @ (Wm * AB), 1e-12)
            L = A @ B
            out[i, j], out[j, i] = L[i, j], L[j, i]
    return np.clip(out, 0, 1)


def prevalence_regression(R, prev, withheld=None, use_block=True):
    """log(mu + c) ~ logit p_i + logit p_j + block (or source and target type when the block is withheld),
    fitted on the other dyads (leave-dyad-out) or on the observed pairs outside a withheld block."""
    c = 0.005
    lg = np.log(prev / (1 - prev))
    y = np.log(R + c)

    def design(i, j, block_dummies):
        x = [1.0, lg[i], lg[j]]
        if block_dummies:
            d = np.zeros(10); d[BLOCK[i, j]] = 1; x += list(d[1:])
        else:
            si = np.zeros(4); si[TI[i]] = 1; tj = np.zeros(4); tj[TI[j]] = 1
            x += list(si[1:]) + list(tj[1:])
        return x

    pairs = [(i, j) for i, j in itertools.permutations(range(P), 2)]
    out = np.zeros_like(R)
    if withheld is not None:
        tr = [(i, j) for i, j in pairs if not withheld[i, j]]
        Xtr = np.array([design(i, j, False) for i, j in tr]); ytr = np.array([y[i, j] for i, j in tr])
        beta = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
        for i, j in pairs:
            out[i, j] = np.exp(np.dot(design(i, j, False), beta)) - c
        return np.clip(out, 0, 1)
    Xall = {p: design(*p, use_block) for p in pairs}
    for i, j in pairs:
        tr = [p for p in pairs if p not in ((i, j), (j, i))]
        beta = np.linalg.lstsq(np.array([Xall[p] for p in tr]), np.array([y[p] for p in tr]), rcond=None)[0]
        out[i, j] = np.exp(np.dot(Xall[(i, j)], beta)) - c
    return np.clip(out, 0, 1)


FEATS = ["prod_max", "ham_max", "min_max", "cn_sum", "adamic_adar", "resource_alloc", "prod_top3_mean", "n_paths_gt_0.01"]


def learned_aggregator(R_train, target_train, train_mask, R_apply, alpha=1.0):
    """Ridge regression on path features: trained within development (features of R_train, target the
    development relation on train_mask), applied to features of R_apply."""
    Ft, Fa = path_features(R_train), path_features(R_apply)
    Xt = np.column_stack([Ft[k][train_mask] for k in FEATS])
    m, s = Xt.mean(0), Xt.std(0) + 1e-12
    mdl = Ridge(alpha=alpha).fit((Xt - m) / s, target_train[train_mask])
    Xa = np.stack([Fa[k] for k in FEATS], -1)
    return np.clip(mdl.predict(((Xa - m) / s).reshape(-1, len(FEATS))).reshape(P, P), 0, 1)


# ---------------- evaluation ----------------
def per_source(pred, target, k=5):
    rs, nd, pk = [], [], []
    for i in range(P):
        js = [j for j in range(P) if j != i]
        p, t = pred[i, js], target[i, js]
        rs.append(rho(p, t))
        order = np.argsort(-p, kind="stable")[:k]
        ideal = np.sort(t)[::-1][:k]
        disc = 1 / np.log2(np.arange(2, k + 2))
        nd.append(float((t[order] * disc).sum() / max((ideal * disc).sum(), 1e-12)))
        topt = set(np.argsort(-t, kind="stable")[:k])
        pk.append(len(set(order) & topt) / k)
    return float(np.nanmean(rs)), float(np.mean(nd)), float(np.mean(pk))


def within_block(pred, target):
    vals, w = [], []
    for b in range(10):
        m = OFF & (BLOCK == b)
        vals.append(rho(pred[m], target[m])); w.append(m.sum())
    vals = np.array(vals); w = np.array(w)
    ok = np.isfinite(vals)
    return float((vals[ok] * w[ok]).sum() / w[ok].sum()), vals.tolist()


def block_partial(pred, target):
    """Spearman correlation after removing block means of the ranks (pairs compared within blocks)."""
    rp, rt = rankdata(pred[OFF]), rankdata(target[OFF])
    b = BLOCK[OFF]
    for v in (rp, rt):
        for k in range(10):
            v[b == k] -= v[b == k].mean()
    return float(np.corrcoef(rp, rt)[0, 1])


def evaluate(pred, target):
    ps = per_source(pred, target)
    wb, _ = within_block(pred, target)
    return {"rho": rho(pred[OFF], target[OFF]), "mae": float(np.abs(pred - target)[OFF].mean()),
            "within_block_rho": wb, "block_partial_rho": block_partial(pred, target),
            "per_source_rho": ps[0], "ndcg5": ps[1], "precision5": ps[2]}


def recalibrate(composed_train, edge_train, composed_apply, kind="linear"):
    x, y = composed_train, edge_train
    if kind == "linear":
        A = np.vstack([np.ones_like(x), x]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.clip(a + b * composed_apply, 0, 1)
    if kind == "power":
        c = 1e-4
        A = np.vstack([np.ones_like(x), np.log(x + c)]).T
        a, b = np.linalg.lstsq(A, np.log(y + c), rcond=None)[0]
        return np.clip(np.exp(a) * (composed_apply + c) ** b - c, 0, 1)
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(x, y)
        return iso.predict(composed_apply)
    raise ValueError(kind)
