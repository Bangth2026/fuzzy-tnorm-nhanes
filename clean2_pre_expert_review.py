"""Extended NHANES dataset (24 concepts) for the data-driven fuzzy knowledge graph.

Development cycles: 2013-2014 (H), 2015-2016 (I), 2017-2018 (J)
Held-out cycles:    2007-2008 (E), 2009-2010 (F), 2011-2012 (G)
Membership functions are anchored to conventional clinical cut-points and are
varied in the sensitivity analysis (crisp versions).
"""
import json
import numpy as np
import pandas as pd

DEV, HOLD = "HIJ", "EFG"
ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SYM = ["anhedonia", "depressed_mood", "sleep_problems", "fatigue", "appetite",
       "worthlessness", "concentration", "psychomotor", "self_harm_thoughts"]
SOC = ["low_income", "low_education", "no_partner", "uninsured"]
LIFE = ["smoking", "heavy_drinking", "inactivity", "short_sleep", "obesity"]
HEALTH = ["poor_self_rated_health", "hypertension", "diabetes", "cardiovascular", "cancer", "arthritis"]
CONCEPTS = SYM + SOC + LIFE + HEALTH


def rd(c, s):
    d = pd.read_sas(f"raw/{c}_{s}.xpt", format="xport").set_index("SEQN")
    num = d.select_dtypes("number").columns
    d[num] = d[num].where((d[num].abs() > 1e-50) | d[num].isna(), 0.0)
    return d


def yn(x):
    x = np.asarray(x, dtype=float)
    return np.where(x == 1, 1.0, np.where(x == 2, 0.0, np.nan))


def lin(x, lo, hi):
    """0 at lo, 1 at hi (works for lo > hi as a decreasing ramp)."""
    x = np.asarray(x, dtype=float)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def cycle(s):
    dpq, demo = rd("DPQ", s), rd("DEMO", s)
    d = demo[["RIDAGEYR", "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU", "DMDEDUC2", "DMDMARTL"]].join(dpq[ITEMS], how="inner")
    flow = {"dpq_file_adults": int(len(dpq))}
    ok = d[ITEMS].isin([0, 1, 2, 3]).all(axis=1)
    d = d[ok & (d.WTMEC2YR > 0)].copy()
    flow["complete_phq9_positive_weight"] = int(len(d))
    for it, nm in zip(ITEMS, SYM):
        d[nm] = d[it] / 3.0
    d["phq9_total"] = d[ITEMS].sum(axis=1)
    ix = d.index

    # socioeconomic
    d["low_income"] = lin(d.INDFMPIR, 3.0, 1.0)                     # 1 at PIR <= 1, 0 at PIR >= 3
    ed = d.DMDEDUC2.where(d.DMDEDUC2.isin([1, 2, 3, 4, 5]))
    d["low_education"] = ed.map({1: 1.0, 2: 1.0, 3: 0.5, 4: 0.0, 5: 0.0})
    ms = d.DMDMARTL.where(d.DMDMARTL.isin([1, 2, 3, 4, 5, 6]))
    d["no_partner"] = ms.map({1: 0.0, 6: 0.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0})
    d["uninsured"] = 1 - yn(rd("HIQ", s).HIQ011.reindex(ix))      # HIQ011 1 = covered

    # lifestyle
    smq = rd("SMQ", s).reindex(ix)
    d["smoking"] = np.where(smq.SMQ020 == 2, 0.0, np.where(smq.SMQ040 == 1, 1.0, np.where(smq.SMQ040 == 2, 0.5,
                   np.where(smq.SMQ040 == 3, 0.0, np.nan))))
    alq = rd("ALQ", s).reindex(ix)
    freq0 = (alq.ALQ121 == 0) if "ALQ121" in alq else (alq.ALQ120Q == 0)
    per_day = alq.ALQ130.where(alq.ALQ130 < 100)                    # drinks on a drinking day
    d["heavy_drinking"] = np.where(freq0, 0.0, lin(per_day, 2.0, 5.0))  # 0 at <= 2, 1 at >= 5
    d.loc[~freq0 & per_day.isna(), "heavy_drinking"] = np.nan
    paq = rd("PAQ", s).reindex(ix)
    vig = (yn(paq.PAQ605) == 1) | (yn(paq.PAQ650) == 1)
    mod = (yn(paq.PAQ620) == 1) | (yn(paq.PAQ665) == 1)
    known = ~np.isnan(np.column_stack([yn(paq[c]) for c in ["PAQ605", "PAQ650", "PAQ620", "PAQ665"]])).any(axis=1)
    d["inactivity"] = np.where(~known, np.nan, np.where(vig, 0.0, np.where(mod, 0.5, 1.0)))
    slq = rd("SLQ", s).reindex(ix)
    hrs = slq.SLD012 if "SLD012" in slq else slq.SLD010H.where(slq.SLD010H < 25)
    d["short_sleep"] = lin(hrs, 7.0, 5.0)                           # 0 at >= 7 h, 1 at <= 5 h
    d["obesity"] = lin(rd("BMX", s).BMXBMI.reindex(ix), 25.0, 35.0)

    # health
    hs = rd("HSQ", s).HSD010.reindex(ix)
    d["poor_self_rated_health"] = lin(hs.where(hs.isin([1, 2, 3, 4, 5])), 1.0, 5.0)
    d["hypertension"] = yn(rd("BPQ", s).BPQ020.reindex(ix))
    diq = rd("DIQ", s).DIQ010.reindex(ix)
    d["diabetes"] = np.where(diq == 1, 1.0, np.where(diq == 3, 0.5, np.where(diq == 2, 0.0, np.nan)))
    mcq = rd("MCQ", s).reindex(ix)
    cv = np.column_stack([yn(mcq[c]) for c in ["MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F"]])
    d["cardiovascular"] = np.where(np.isnan(cv).all(1), np.nan, np.nanmax(np.nan_to_num(cv, nan=0), 1))
    d["cancer"] = yn(mcq.MCQ220)
    d["arthritis"] = yn(mcq.MCQ160A)

    d["cycle"] = s
    d["strata"] = s + "_" + d.SDMVSTRA.astype(int).astype(str)
    d["psu"] = d.strata + "_" + d.SDMVPSU.astype(int).astype(str)
    return d, flow


def build(cycles, name):
    parts, flows = [], {}
    for s in cycles:
        d, f = cycle(s)
        parts.append(d)
        flows[s] = f
    d = pd.concat(parts)
    d["w"] = d.WTMEC2YR / len(cycles)
    flows["total_analytic"] = int(len(d))
    flows["weighted_prev_phq_ge10"] = round(float(np.average(d.phq9_total >= 10, weights=d.w)), 4)
    flows["missing"] = {c: int(d[c].isna().sum()) for c in CONCEPTS}
    flows["weighted_mean_activation"] = {c: round(float(np.average(d[c].dropna(), weights=d.w[d[c].notna()])), 4) for c in CONCEPTS}
    d[CONCEPTS + ["w", "strata", "psu", "cycle", "phq9_total", "RIDAGEYR"]].to_pickle(f"data2_{name}.pkl")
    return flows


if __name__ == "__main__":
    out = {"development_2013_2018": build(DEV, "dev"), "heldout_2007_2012": build(HOLD, "hold")}
    json.dump(out, open("results2/flow.json", "w"), indent=2)
    print(json.dumps(out, indent=1))
