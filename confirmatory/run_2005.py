"""Confirmatory test on NHANES 2005-2006 as registered in confirmatory/PLAN.md.
Run from the repository root: python3 confirmatory/run_2005.py [main|no_inactivity]
Writes results2/confirm_2005_<variant>.json."""
import json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import clean2
from analysis2 import dev
from graph2 import CONCEPTS, TYPE, P, relation_matrices
import rq3x as X
import rq6lib as L

variant = sys.argv[1] if len(sys.argv) > 1 else "main"
OFF = ~np.eye(P, dtype=bool)
UP = np.triu(OFF, 1)
LAM_GRID = [0.01, 0.03, 0.1, 0.3, 0.6]
MODIFIABLE = ["smoking", "heavy_drinking", "inactivity", "short_sleep", "obesity"]
SYMPTOMS = [c for c in CONCEPTS if TYPE[c] == "Symptom"]

# 2005-2006 physical activity: vigorous (PAD200) and moderate (PAD320) leisure-time activity in the past 30 days
_rd = clean2.rd
def rd(c, s):
    d = _rd(c, s)
    if c == "PAQ" and s == "D":
        v = d.PAD200.replace({3: 2}); m = d.PAD320.replace({3: 2})
        d = d.assign(PAQ605=v, PAQ650=v, PAQ620=m, PAQ665=m)
    return d
clean2.rd = rd

c05, flow = clean2.cycle("D")
c05["w"] = c05.WTMEC2YR
c05 = c05[CONCEPTS + ["w", "strata", "psu"]]

drop = [] if variant == "main" else [CONCEPTS.index("inactivity")]
keep_idx = [i for i in range(P) if i not in drop]


def rel(df):
    E = relation_matrices(df)[2]
    return E[np.ix_(keep_idx, keep_idx)]


def r(a, b):
    return float(spearmanr(a, b).statistic)


n = len(keep_idx)
OFFn = ~np.eye(n, dtype=bool); UPn = np.triu(OFFn, 1)
BLK = L.BLOCK[np.ix_(keep_idx, keep_idx)]
nb = np.array([(OFFn & (BLK == b)).sum() for b in range(10)], float)
names = [CONCEPTS[i] for i in keep_idx]


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


def lowrank_dyad(E, k=3):
    out = np.zeros_like(E)
    for i in range(n):
        for j in range(i + 1, n):
            m = OFFn.copy(); m[i, j] = m[j, i] = False
            F = X.soft_impute(E, m, k)
            out[i, j], out[j, i] = F[i, j], F[j, i]
    return out


def katz_dyad_free(E):
    lam = np.max(np.abs(np.linalg.eigvals(E))).real
    beta = 0.5 / max(lam, 1e-9)
    K = np.linalg.inv(np.eye(n) - beta * E) - np.eye(n)
    return K


def choose_lambda(E, base):
    dy = [(i, j) for i in range(n) for j in range(i + 1, n) if base[i, j] and base[j, i]]
    g = np.random.default_rng(0); idx = g.permutation(len(dy)); sc = {}
    for lam in LAM_GRID:
        s = []
        for f in range(5):
            m = base.copy()
            for q in idx[f::5]:
                i, j = dy[q]; m[i, j] = m[j, i] = False
            H = base & ~m
            s.append(r(L.soft_impute_reg(E, m, lam)[H], E[H]))
        sc[lam] = np.mean(s)
    return max(sc, key=sc.get)


def stats(E1, E2):
    o = {}
    prod, ham, mn = X.paths(E1, X.t_product), X.paths(E1, X.T_FIXED["hamacher"]), X.paths(E1, X.t_min)
    r3 = lowrank_dyad(E1)
    o["H1_rank3"] = r(r3[OFFn], E2[OFFn]); o["product"] = r(prod[OFFn], E2[OFFn])
    o["H1_rank3_minus_product"] = o["H1_rank3"] - o["product"]
    S1, S2 = np.sqrt(E1 * E1.T), np.sqrt(E2 * E2.T)
    o["sym_product"] = r(X.paths(S1, X.t_product)[UPn], S2[UPn]); o["sym_hamacher"] = r(X.paths(S1, X.T_FIXED["hamacher"])[UPn], S2[UPn])
    o["H2_sym_product_minus_hamacher"] = o["sym_product"] - o["sym_hamacher"]
    o["min"] = r(mn[OFFn], E2[OFFn]); o["hamacher"] = r(ham[OFFn], E2[OFFn])
    o["H3_product_minus_min"] = o["product"] - o["min"]
    pb = {"product": [], "softimpute_cv": [], "katz": []}
    for b in range(10):
        H = OFFn & (BLK == b)
        if H.sum() < 3:
            pb["product"].append(np.nan); pb["softimpute_cv"].append(np.nan); pb["katz"].append(np.nan); continue
        Em = np.where(H, 0.0, E1)
        pb["product"].append(r(X.paths(Em, X.t_product)[H], E2[H]))
        lam = choose_lambda(E1, OFFn & ~H)
        pb["softimpute_cv"].append(r(L.soft_impute_reg(E1, OFFn & ~H, lam)[H], E2[H]))
        pb["katz"].append(r(katz_dyad_free(Em)[H], E2[H]))
    wm = {k: float(np.nansum(np.array(v) * nb) / nb[~np.isnan(v)].sum()) for k, v in pb.items()}
    o.update({f"block_{k}": v for k, v in wm.items()})
    o["H4_block_product_minus_softimpute"] = wm["product"] - wm["softimpute_cv"]
    o["H5_block_product_minus_katz"] = wm["product"] - wm["katz"]
    # H6 clinical query
    if variant == "main":
        ci = {c: names.index(c) for c in names}
        mod = [ci[c] for c in MODIFIABLE]
        def credit(p, t):
            top = np.where(np.isclose(p, p.max(), rtol=0, atol=1e-12))[0]
            return (1.0 / len(top)) if int(np.argmax(t)) in top else 0.0
        for k, M in {"product": prod, "min": mn, "hamacher": ham, "rank3": r3, "direct_edge": E1}.items():
            o[f"H6_{k}"] = sum(credit(np.array([M[i, ci[s]] for i in mod]), np.array([E2[i, ci[s]] for i in mod])) for s in SYMPTOMS)
    return o


Ed = rel(dev); E05 = rel(c05)
full = stats(Ed, E05)
reps = [stats(rel(d), E05) for d in jrr_reps(dev)] + [stats(Ed, rel(h)) for h in jrr_reps(c05)]
res = {"variant": variant, "n_concepts": n, "flow_2005_2006": flow, "n_adults": int(len(c05)),
       "n_strata_2005": int(c05.strata.nunique()), "n_replicates": len(reps),
       "corr_dev_vs_2005": r(Ed[OFFn], E05[OFFn])}
for k in full:
    v = np.array([x[k] for x in reps], float)
    se = float(np.sqrt(((v - full[k]) ** 2).sum()))
    res[k] = {"point": float(full[k]), "ci95": [full[k] - 1.96 * se, full[k] + 1.96 * se]}
dec = {"H1": res["H1_rank3_minus_product"]["ci95"][0] > 0,
       "H2": res["H2_sym_product_minus_hamacher"]["ci95"][0] <= 0 <= res["H2_sym_product_minus_hamacher"]["ci95"][1],
       "H3": res["H3_product_minus_min"]["ci95"][0] > 0,
       "H4": res["H4_block_product_minus_softimpute"]["ci95"][0] > 0,
       "H5": res["H5_block_product_minus_katz"]["ci95"][0] <= 0 <= res["H5_block_product_minus_katz"]["ci95"][1]}
if variant == "main":
    dec["H6"] = full["H6_product"] >= full["H6_min"]
res["supported"] = {k: bool(v) for k, v in dec.items()}
json.dump(res, open(f"results2/confirm_2005_{variant}.json", "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ["n_adults", "corr_dev_vs_2005", "supported"]}, indent=1))
