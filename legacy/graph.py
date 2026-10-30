"""MFKG v0: schema, edge construction, modularity, nulls, pathway scoring.

The literature layer below is PROVISIONAL (v0): grades were assigned by the
technical author from the cited reviews and MUST be reviewed and calibrated
by the clinical co-first author. Results computed on v0 are labelled as such.
"""
import heapq
import itertools
import numpy as np
import pandas as pd
import igraph as ig
import leidenalg as la

SYM = ["anhedonia", "depressed_mood", "sleep", "fatigue", "appetite",
       "worthlessness", "concentration", "psychomotor"]
RISK = ["low_income", "obesity", "smoking", "inactivity", "chronic_illness"]
MECH = ["cognitive_reactivity", "emotional_reactivity", "low_mindfulness", "rumination",
        "worry", "low_self_compassion", "low_meta_awareness", "psychological_inflexibility"]
COMP = ["body_scan", "mindful_movement", "sitting_meditation", "breathing_space",
        "turning_toward_difficulty", "thoughts_not_facts", "pleasant_unpleasant_events",
        "activity_and_self_care", "relapse_action_plan"]   # listed in curriculum order
OUT = ["reduced_relapse_vulnerability"]
NODES = SYM + RISK + MECH + COMP + OUT
TYPE = {**{n: "Symptom" for n in SYM}, **{n: "Risk" for n in RISK}, **{n: "Mechanism" for n in MECH},
        **{n: "Component" for n in COMP}, **{n: "Outcome" for n in OUT}}
GRADE = {"P": 0.25, "M": 0.50, "S": 0.75, "VS": 0.917}
# Mediation-evidence cap per mechanism, from Gu et al. 2015 (ref [4]); meta-awareness
# is not graded there and is capped at preliminary (van der Velden et al. 2015, ref [3]).
MECH_CAP = {"cognitive_reactivity": "S", "emotional_reactivity": "S", "low_mindfulness": "M",
            "rumination": "M", "worry": "M", "low_self_compassion": "P",
            "psychological_inflexibility": "P", "low_meta_awareness": "P"}
SYM_MECH = [  # maintained-by, provisional grades
    ("depressed_mood", "rumination", "S"), ("depressed_mood", "cognitive_reactivity", "S"),
    ("depressed_mood", "emotional_reactivity", "M"), ("depressed_mood", "low_meta_awareness", "P"),
    ("worthlessness", "rumination", "S"), ("worthlessness", "cognitive_reactivity", "S"),
    ("worthlessness", "low_self_compassion", "M"), ("worthlessness", "low_meta_awareness", "P"),
    ("anhedonia", "rumination", "M"), ("anhedonia", "psychological_inflexibility", "P"),
    ("sleep", "worry", "M"), ("sleep", "rumination", "P"),
    ("fatigue", "psychological_inflexibility", "P"),
    ("concentration", "rumination", "M"), ("concentration", "worry", "M"),
    ("concentration", "low_mindfulness", "M"),
    ("psychomotor", "emotional_reactivity", "P"), ("appetite", "emotional_reactivity", "P")]
MECH_COMP = [  # targeted-by, provisional grades before the mechanism cap
    ("rumination", "thoughts_not_facts", "S"), ("rumination", "breathing_space", "M"),
    ("rumination", "sitting_meditation", "M"),
    ("worry", "breathing_space", "M"), ("worry", "sitting_meditation", "P"),
    ("cognitive_reactivity", "thoughts_not_facts", "S"), ("cognitive_reactivity", "relapse_action_plan", "M"),
    ("emotional_reactivity", "turning_toward_difficulty", "M"), ("emotional_reactivity", "breathing_space", "M"),
    ("low_mindfulness", "body_scan", "M"), ("low_mindfulness", "sitting_meditation", "M"),
    ("low_mindfulness", "mindful_movement", "P"), ("low_mindfulness", "pleasant_unpleasant_events", "P"),
    ("low_self_compassion", "turning_toward_difficulty", "P"), ("low_self_compassion", "activity_and_self_care", "P"),
    ("low_meta_awareness", "thoughts_not_facts", "M"), ("low_meta_awareness", "pleasant_unpleasant_events", "P"),
    ("psychological_inflexibility", "activity_and_self_care", "P"),
    ("psychological_inflexibility", "turning_toward_difficulty", "P")]
COMP_OUT = [(c, OUT[0], "M") for c in COMP]
ORDER = ["P", "M", "S", "VS"]


def cap(g, mech):
    return ORDER[min(ORDER.index(g), ORDER.index(MECH_CAP[mech]))]


def subsethood(df, i, j, w="w"):
    d = df[[i, j, w]].dropna()
    den = (d[w] * d[i]).sum()
    return float((d[w] * np.minimum(d[i], d[j])).sum() / den) if den > 0 else 0.0


def literature_edges():
    e = []
    for a, b, g in SYM_MECH:
        e.append((a, b, GRADE[g], "maintained-by", "literature-v0"))
    for a, b, g in MECH_COMP:
        e.append((a, b, GRADE[cap(g, a)], "targeted-by", "literature-v0"))
    for a, b, g in COMP_OUT:
        e.append((a, b, GRADE[g], "contributes-to", "literature-v0"))
    return e


def data_edges(df, eps=0.05):
    e = []
    for i, j in itertools.permutations(SYM, 2):
        m = subsethood(df, i, j)
        if m >= eps:
            e.append((i, j, m, "co-activates", "NHANES"))
    for r in RISK:
        for j in SYM:
            m = subsethood(df, r, j)
            if m >= eps:
                e.append((r, j, m, "associated-with", "NHANES"))
    return e


def edges_df(edges):
    return pd.DataFrame(edges, columns=["src", "dst", "mu", "rel", "prov"])


# ---------------- modularity ----------------
IDX = {n: k for k, n in enumerate(NODES)}


def adj(E, nodes=NODES):
    ix = {n: k for k, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, d, m in zip(E.src, E.dst, E.mu):
        A[ix[s], ix[d]] = m
    return A


def qf(A, memb, gamma=1.0):
    W = A.sum()
    so, si = A.sum(1), A.sum(0)
    same = memb[:, None] == memb[None, :]
    return float(((A - gamma * np.outer(so, si) / W) * same).sum() / W)


def leiden(A, seed, gamma=1.0):
    n = A.shape[0]
    src, dst = np.nonzero(A)
    g = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=True)
    g.es["weight"] = A[src, dst].tolist()
    p = la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                          resolution_parameter=gamma, seed=int(seed), n_iterations=-1)
    return np.array(p.membership)


def best_q(A, seeds, gamma=1.0):
    best, bm, qs = -9, None, []
    for s in seeds:
        m = leiden(A, s, gamma)
        q = qf(A, m, gamma)
        qs.append(q)
        if q > best:
            best, bm = q, m
    return best, bm, np.array(qs)


# ---------------- null models ----------------
def n2_permute(E, rng):
    E = E.copy()
    for r, g in E.groupby("rel"):
        E.loc[g.index, "mu"] = rng.permutation(g.mu.values)
    return E


def n1_rewire(E, rng, swaps_per_edge=10):
    E = E.copy().reset_index(drop=True)
    for r, g in E.groupby("rel"):
        idx = g.index.to_numpy()
        if len(idx) < 2:
            continue
        pairs = set(zip(E.loc[idx, "src"], E.loc[idx, "dst"]))
        srcs, dsts = set(E.loc[idx, "src"]), set(E.loc[idx, "dst"])
        possible = sum(1 for a in srcs for b in dsts if a != b)
        if len(pairs) >= possible:      # layer is complete: no valid swap exists,
            E.loc[idx, "mu"] = rng.permutation(E.loc[idx, "mu"].values)  # N1 reduces to N2 here
            continue
        target, done, tries = swaps_per_edge * len(idx), 0, 0
        while done < target and tries < 50 * target:
            tries += 1
            a, b = rng.choice(idx, 2, replace=False)
            s1, d1, s2, d2 = E.at[a, "src"], E.at[a, "dst"], E.at[b, "src"], E.at[b, "dst"]
            if s1 == d2 or s2 == d1 or (s1, d2) in pairs or (s2, d1) in pairs:
                continue
            pairs -= {(s1, d1), (s2, d2)}
            pairs |= {(s1, d2), (s2, d1)}
            E.at[a, "dst"], E.at[b, "dst"] = d2, d1
            done += 1
        E.loc[idx, "mu"] = rng.permutation(E.loc[idx, "mu"].values)
    return E


# ---------------- pathway scoring ----------------
INF = float("inf")
TNORMS = {
    "product": (lambda x: -np.log(x) if x > 0 else INF, lambda y: float(np.exp(-y)), INF),
    "hamacher": (lambda x: (1 - x) / x if x > 0 else INF, lambda y: 1.0 / (1.0 + y), INF),
    "lukasiewicz": (lambda x: 1 - x, lambda y: max(0.0, 1 - y), 1.0),
    "yager2": (lambda x: (1 - x) ** 2, lambda y: max(0.0, 1 - np.sqrt(y)) if y < 1 else 0.0, 1.0),
}


def dijkstra(adjl, starts):
    dist = dict(starts)
    pq = [(d, n) for n, d in starts.items()]
    heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v, l in adjl.get(u, []):
            nd = d + l
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def widest(adjl, starts):
    best = dict(starts)
    pq = [(-b, n) for n, b in starts.items()]
    heapq.heapify(pq)
    while pq:
        nb, u = heapq.heappop(pq)
        b = -nb
        if b < best.get(u, 0):
            continue
        for v, w in adjl.get(u, []):
            nbv = min(b, w)
            if nbv > best.get(v, 0):
                best[v] = nbv
                heapq.heappush(pq, (-nbv, v))
    return best


def path_edges(E):
    """Edges usable by pathways: symptom->symptom, symptom->mechanism, mechanism->component, component->outcome."""
    keep = E.rel.isin(["co-activates", "maintained-by", "targeted-by", "contributes-to"])
    return E[keep]


def component_scores(E, act, tnorm):
    P = path_edges(E)
    starts_raw = {s: a for s, a in act.items() if a > 0}
    if tnorm == "maxmin":
        fw = {}
        for u, v, m in zip(P.src, P.dst, P.mu):
            fw.setdefault(u, []).append((v, m))
        bw = {}
        for u, v, m in zip(P.src, P.dst, P.mu):
            bw.setdefault(v, []).append((u, m))
        f = widest(fw, starts_raw)
        b = widest(bw, {OUT[0]: 1.0})
        return {c: min(f.get(c, 0), b.get(c, 0)) for c in COMP}
    g, ginv, g0 = TNORMS[tnorm]
    fw, bw = {}, {}
    for u, v, m in zip(P.src, P.dst, P.mu):
        fw.setdefault(u, []).append((v, g(m)))
        bw.setdefault(v, []).append((u, g(m)))
    f = dijkstra(fw, {s: g(a) for s, a in starts_raw.items()})
    b = dijkstra(bw, {OUT[0]: 0.0})
    return {c: ginv(min(f.get(c, INF) + b.get(c, INF), g0)) for c in COMP}


def binary_scores(E, items, alpha=0.5):
    """Hop-count shortest paths on G_alpha from symptoms rated >= 2 (present)."""
    P = path_edges(E)
    P = P[P.mu >= alpha]
    fw, bw = {}, {}
    for u, v in zip(P.src, P.dst):
        fw.setdefault(u, []).append((v, 1.0))
        bw.setdefault(v, []).append((u, 1.0))
    starts = {s: 0.0 for s, x in items.items() if x >= 2}
    f = dijkstra(fw, starts)
    b = dijkstra(bw, {OUT[0]: 0.0})
    return {c: -(f.get(c, INF) + b.get(c, INF)) for c in COMP}   # higher = better


def rank(scores):
    """Rank components by score (desc); ties broken by curriculum order."""
    return sorted(COMP, key=lambda c: (-scores[c], COMP.index(c)))
