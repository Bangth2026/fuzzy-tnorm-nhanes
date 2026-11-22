"""NHANES August 2021-August 2023 (cycle L) on the concepts of clean2.py.

Changes needed because the cycle-L questionnaires differ (fixed in confirmatory/PLAN_2021.md before the files were
downloaded, from the public codebooks only):
  - marital status: DMDMARTZ (1 married or living with partner, 2 widowed/divorced/separated, 3 never married)
    replaces DMDMARTL; mapped to 0, 1 and 0.5 as in clean2.py
  - self-rated health: HUQ010 (household interview) replaces HSD010 (HSQ_L has no general-health item); same five
    categories, scored as in clean2.py
  - inactivity: leisure-time activity frequency, vigorous PAD810Q > 0 -> 0, else moderate PAD790Q > 0 -> 0.5,
    else 1 (codes 7777/9999 missing)
  - the PHQ-9 was self-administered by ACASI in the mobile examination center in this cycle (face-to-face
    interview in earlier cycles)
"""
import numpy as np
import clean2
from graph2 import CONCEPTS

_rd = clean2.rd


def rd(c, s):
    if s == "L" and c == "HSQ":
        d = _rd("HUQ", s)
        return d.assign(HSD010=d.HUQ010)
    d = _rd(c, s)
    if s == "L" and c == "DEMO":
        d = d.assign(DMDMARTL=d.DMDMARTZ.map({1: 1, 2: 2, 3: 5}))
    if s == "L" and c == "PAQ":
        vq = d.PAD810Q.where(d.PAD810Q < 7777)
        mq = d.PAD790Q.where(d.PAD790Q < 7777)
        v = np.where(vq > 0, 1.0, np.where(vq == 0, 2.0, np.nan))
        m = np.where(mq > 0, 1.0, np.where(mq == 0, 2.0, np.nan))
        d = d.assign(PAQ605=v, PAQ650=v, PAQ620=m, PAQ665=m)
    return d


def load_c21():
    clean2.rd = rd
    try:
        d, flow = clean2.cycle("L")
    finally:
        clean2.rd = _rd
    d["w"] = d.WTMEC2YR
    return d[CONCEPTS + ["w", "strata", "psu"]], flow
