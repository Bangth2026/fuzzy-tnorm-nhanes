"""Build the analytic NHANES dataset for the MFKG study.

Development cycles: 2013-2014 (H), 2015-2016 (I), 2017-2018 (J)
Held-out cycles:    2007-2008 (E), 2009-2010 (F), 2011-2012 (G)
All membership functions for risk factors are PROVISIONAL and must be
approved by the clinical co-first author before submission.
"""
import json
import numpy as np
import pandas as pd

DEV, HOLD = "HIJ", "EFG"
ITEMS = [f"DPQ0{i}0" for i in range(1, 10)]
SYM = ["anhedonia", "depressed_mood", "sleep", "fatigue", "appetite",
       "worthlessness", "concentration", "psychomotor", "self_harm_thoughts"]


def rd(c, s):
    d = pd.read_sas(f"raw/{c}_{s}.xpt", format="xport")
    d = d.set_index("SEQN")
    num = d.select_dtypes("number").columns
    d[num] = d[num].where((d[num].abs() > 1e-50) | d[num].isna(), 0.0)
    return d


def yn(x):
    """NHANES yes/no coding: 1 yes, 2 no, other = missing."""
    return np.where(x == 1, 1.0, np.where(x == 2, 0.0, np.nan))


def cycle(s):
    dpq, demo = rd("DPQ", s), rd("DEMO", s)
    d = demo[["RIDAGEYR", "INDFMPIR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]].join(dpq[ITEMS], how="inner")
    flow = {"dpq_file_adults": int(len(dpq))}
    ok = d[ITEMS].isin([0, 1, 2, 3]).all(axis=1)
    d = d[ok & (d.WTMEC2YR > 0)].copy()
    flow["complete_phq9_positive_weight"] = int(len(d))
    for it, nm in zip(ITEMS, SYM):
        d[nm] = d[it] / 3.0
    d["phq9_total"] = d[ITEMS].sum(axis=1)

    # ---- provisional risk-factor membership functions ----
    pir = d.INDFMPIR
    d["low_income"] = np.clip((3.0 - pir) / 2.0, 0, 1)          # 1 at PIR<=1, 0 at PIR>=3
    bmx = rd("BMX", s).BMXBMI.reindex(d.index)
    d["obesity"] = np.clip((bmx - 25.0) / 10.0, 0, 1)            # 0 at BMI<=25, 1 at BMI>=35
    smq = rd("SMQ", s).reindex(d.index)
    ever, now = smq.SMQ020, smq.SMQ040
    sm = np.where(ever == 2, 0.0, np.where(now == 1, 1.0, np.where(now == 2, 0.5, np.where(now == 3, 0.0, np.nan))))
    d["smoking"] = sm
    paq = rd("PAQ", s).reindex(d.index)
    vig = (yn(paq.PAQ605) == 1) | (yn(paq.PAQ650) == 1)
    mod = (yn(paq.PAQ620) == 1) | (yn(paq.PAQ665) == 1)
    known = ~(np.isnan(yn(paq.PAQ605)) | np.isnan(yn(paq.PAQ650)) | np.isnan(yn(paq.PAQ620)) | np.isnan(yn(paq.PAQ665)))
    d["inactivity"] = np.where(~known, np.nan, np.where(vig, 0.0, np.where(mod, 0.5, 1.0)))
    mcq = rd("MCQ", s).reindex(d.index)
    diq = rd("DIQ", s).reindex(d.index)
    conds = np.column_stack([yn(diq.DIQ010), yn(mcq.MCQ160B), yn(mcq.MCQ160C), yn(mcq.MCQ160E),
                             yn(mcq.MCQ160F), yn(mcq.MCQ220)])
    cnt = np.nansum(conds, axis=1)
    allnan = np.isnan(conds).all(axis=1)
    d["chronic_illness"] = np.where(allnan, np.nan, np.clip(cnt / 2.0, 0, 1))  # 2+ conditions = 1

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
    d["safety_fail"] = ((d.self_harm_thoughts > 0) | (d.phq9_total >= 20)).astype(int)
    flows["total_analytic"] = int(len(d))
    flows["item9_positive"] = int((d.self_harm_thoughts > 0).sum())
    flows["phq_ge20"] = int((d.phq9_total >= 20).sum())
    flows["safety_fail_any"] = int(d.safety_fail.sum())
    flows["phq_ge10"] = int((d.phq9_total >= 10).sum())
    flows["weighted_prev_phq_ge10"] = float(np.average(d.phq9_total >= 10, weights=d.w))
    for r in ["low_income", "obesity", "smoking", "inactivity", "chronic_illness"]:
        flows[f"missing_{r}"] = int(d[r].isna().sum())
    d.to_pickle(f"data_{name}.pkl")
    return flows


if __name__ == "__main__":
    out = {"development_2013_2018": build(DEV, "dev"), "heldout_2007_2012": build(HOLD, "hold")}
    json.dump(out, open("results/flow.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
