"""Data-driven fuzzy knowledge graph: chance-corrected subsethood, modularity,
type-preserving and respondent-level nulls, and t-norm multi-hop inference."""
import itertools
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la
from clean2 import SYM, SOC, LIFE, HEALTH, CONCEPTS

TYPE = {**{c: "Symptom" for c in SYM}, **{c: "Socioeconomic" for c in SOC},
        **{c: "Lifestyle" for c in LIFE}, **{c: "Health" for c in HEALTH}}
P = len(CONCEPTS)
GRID = np.linspace(0, 1, 201)


def wmean(x, w):
    return float((x * w).sum() / w.sum())


def relation_matrices(df, w="w"):
    """Return raw subsethood S, independence expectation S0 and chance-corrected E (P x P)."""
    A = df[CONCEPTS].to_numpy(float)
    W = df[w].to_numpy(float)
    M = ~np.isnan(A)
    A0 = np.nan_to_num(A)
    # E_w[min(x, A_j)] on a grid, from the observed marginal of j
    G = np.empty((P, GRID.size))
    for j in range(P):
        aj, wj = A0[M[:, j], j], W[M[:, j]]
        G[j] = (np.minimum(GRID[:, None], aj[None, :]) * wj).sum(1) / wj.sum()
    S = np.zeros((P, P))
    S0 = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            if i == j:
                continue
            m = M[:, i] & M[:, j]
            wi = W[m] * A0[m, i]
            den = wi.sum()
            if den <= 0:
                continue
            S[i, j] = (W[m] * np.minimum(A0[m, i], A0[m, j])).sum() / den
            S0[i, j] = (W[m] * np.interp(A0[m, i], GRID, G[j])).sum() / den
    E = np.where(S0 < 1, np.clip((S - S0) / (1 - S0), 0, 1), 0)
    np.fill_diagonal(E, 0)
    return S, S0, E


def edges_from(Mx, eps):
    rows = []
    for i, j in itertools.permutations(range(P), 2):
        if Mx[i, j] >= eps and Mx[i, j] > 0:
            a, b = CONCEPTS[i], CONCEPTS[j]
            rows.append((a, b, float(Mx[i, j]), f"{TYPE[a]}->{TYPE[b]}"))
    return pd.DataFrame(rows, columns=["src", "dst", "mu", "rel"])


def adj(E):
    ix = {n: k for k, n in enumerate(CONCEPTS)}
    A = np.zeros((P, P))
    for s, d, m in zip(E.src, E.dst, E.mu):
        A[ix[s], ix[d]] = m
    return A


def qf(A, memb, gamma=1.0):
    W = A.sum()
    if W <= 0:
        return 0.0
    so, si = A.sum(1), A.sum(0)
    same = memb[:, None] == memb[None, :]
    return float(((A - gamma * np.outer(so, si) / W) * same).sum() / W)


def leiden(A, seed, gamma=1.0):
    src, dst = np.nonzero(A)
    if len(src) == 0:
        return np.arange(A.shape[0])
    g = ig.Graph(n=A.shape[0], edges=list(zip(src.tolist(), dst.tolist())), directed=True)
    g.es["weight"] = A[src, dst].tolist()
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=gamma, seed=int(seed), n_iterations=-1)
    return np.array(p.membership)


def best_q(A, seeds, gamma=1.0):
    best, bm, qs = -9.0, None, []
    for s in seeds:
        m = leiden(A, s, gamma)
        q = qf(A, m, gamma)
        qs.append(q)
        if q > best:
            best, bm = q, m
    return best, bm, np.array(qs)


def n2_permute(E, rng):
    E = E.copy()
    for _, g in E.groupby("rel"):
        E.loc[g.index, "mu"] = rng.permutation(g.mu.values)
    return E


def n1_rewire(E, rng, swaps_per_edge=10):
    E = E.copy().reset_index(drop=True)
    for _, g in E.groupby("rel"):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        pairs = set(zip(E.loc[idx, "src"], E.loc[idx, "dst"]))
        srcs, dsts = set(E.loc[idx, "src"]), set(E.loc[idx, "dst"])
        if len(pairs) >= sum(1 for a in srcs for b in dsts if a != b):
            E.loc[idx, "mu"] = rng.permutation(E.loc[idx, "mu"].values)
            continue
        src, dst = E.loc[idx, "src"].tolist(), E.loc[idx, "dst"].tolist()
        target, done, tries = swaps_per_edge * len(idx), 0, 0
        while done < target and tries < 50 * target:
            tries += 1
            a, b = rng.choice(len(idx), 2, replace=False)
            s1, d1, s2, d2 = src[a], dst[a], src[b], dst[b]
            if s1 == d2 or s2 == d1 or (s1, d2) in pairs or (s2, d1) in pairs:
                continue
            pairs -= {(s1, d1), (s2, d2)}
            pairs |= {(s1, d2), (s2, d1)}
            dst[a], dst[b] = d2, d1
            done += 1
        E.loc[idx, "dst"] = dst
        E.loc[idx, "mu"] = rng.permutation(E.loc[idx, "mu"].values)
    return E


def respondent_permute(df, rng):
    """Permute each concept independently across respondents (keeps marginals, removes association)."""
    d = df.copy()
    for c in CONCEPTS:
        d[c] = rng.permutation(d[c].to_numpy())
    return d


# ---------------- t-norms ----------------
TNORM = {
    "product": lambda x, y: x * y,
    "hamacher": lambda x, y: np.where((x + y - x * y) > 0, x * y / np.maximum(x + y - x * y, 1e-300), 0.0),
    "lukasiewicz": lambda x, y: np.maximum(0.0, x + y - 1.0),
    "yager2": lambda x, y: np.maximum(0.0, 1 - np.sqrt((1 - x) ** 2 + (1 - y) ** 2)),
    "min": lambda x, y: np.minimum(x, y),
}


def compose(R, T):
    """sup-T composition (R o R)[i, j] = max_k T(R[i, k], R[k, j]), excluding k in {i, j}."""
    n = R.shape[0]
    out = np.zeros_like(R)
    for k in range(n):
        v = TNORM[T](R[:, k][:, None], R[k, :][None, :])
        v[k, :] = 0
        v[:, k] = 0
        out = np.maximum(out, v)
    np.fill_diagonal(out, 0)
    return out


def indirect(R, T, max_len=3):
    """Best indirect path strength (length 2..max_len) for every ordered pair, never using the direct edge i->j."""
    n = R.shape[0]
    best = np.zeros_like(R)
    for i, j in itertools.permutations(range(n), 2):
        b = 0.0
        mids = [k for k in range(n) if k not in (i, j)]
        for k in mids:                       # length 2
            v = TNORM[T](R[i, k], R[k, j])
            b = max(b, float(v))
        if max_len >= 3:
            for k, l in itertools.permutations(mids, 2):   # length 3
                v = TNORM[T](TNORM[T](R[i, k], R[k, l]), R[l, j])
                b = max(b, float(v))
        best[i, j] = b
    return best


def binary_hops(R, alpha, max_len=3):
    """Hop count of the shortest indirect path i->...->j (length >= 2) in the binarised graph; inf if none."""
    n = R.shape[0]
    B = R >= alpha
    np.fill_diagonal(B, False)
    hops = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Bij = B.copy()
            Bij[i, j] = False
            frontier, seen, h = {i}, {i}, 0
            while frontier and h < max_len:
                h += 1
                nxt = set()
                for u in frontier:
                    for v in np.nonzero(Bij[u])[0]:
                        if v not in seen:
                            seen.add(v)
                            nxt.add(int(v))
                if j in nxt:
                    hops[i, j] = h
                    break
                frontier = nxt
    return hops
