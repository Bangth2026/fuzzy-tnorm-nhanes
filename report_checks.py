"""Checks on reported quantities (v2.3.0; extended in v2.3.1 and v2.3.2).

Reads existing result files only; no model is refitted.

  python3 report_checks.py

For the 2007-2012, 2005-2006 and 2021-2023 targets it reports, for each of
the six methods that define regret (PCC, rank blend, product, Katz, logit
soft-impute, rank-3 completion):
  * the regret in the single-dyad and type-block designs and the design in
    which its worst case lies;
  * the worst-case regret g computed from unrounded rho, and the same
    quantity computed from rho rounded to three decimals, to show where
    the two differ in the third decimal;
and checks that adding the raw blend to the reference set would leave the
best method of each design, and therefore every regret, unchanged.
It also copies the single-dyad comparison of rank-3 completion and the
product by pair group (results2/pcclim_s1.json, from pcc_limits.py s1).

Added in v2.3.1:
  * the worst-case regret over the 11 regimes of the 2007-2012 comparison
    (results2/pcc2_summary.json) and over the nine regimes of the 2021-2023
    test (pcc2_c21.json, pcc2_c21regimes.json), from unrounded rho and from
    rho rounded to three decimals, with the gaps of the product and the rank
    blend to the raw blend;
  * cold start with one anchor relation per concept: the median per-concept
    rho of every method in the five repetitions of the regime comparison
    (pcc2_summary.json) beside the ten repetitions of the earlier cold-start
    analysis (gen_coldstart.json).

Added in v2.3.2: the relation-scale blend (w = 0.8) in the worst-case regret
over regimes, so that its nine-regime value on 2021-2023 is available beside
the 11-regime value on 2007-2012. The 2005-2006 levels of the six reference
methods in the single-dyad and type-block designs are under
2005_2006/levels_4dp.

Output: results2/report_checks.json
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results2")
SIX = ["pcc", "blend_rank", "product", "katz", "si_logit", "rank3"]
DESIGNS = ["dyad", "block"]


def load(fname, key):
    with open(os.path.join(R, fname)) as f:
        return json.load(f)[key]["full"]


def analyse(full, rounded=False):
    v = {d: {m: full[f"{d}_{m}"] for m in SIX + ["blend_raw"]} for d in DESIGNS}
    if rounded:
        v = {d: {m: round(x, 3) for m, x in v[d].items()} for d in DESIGNS}
    best = {d: max(v[d][m] for m in SIX) for d in DESIGNS}
    out = {}
    for m in SIX:
        reg = {d: best[d] - v[d][m] for d in DESIGNS}
        worst = max(DESIGNS, key=lambda d: reg[d])
        out[m] = {"regret": reg, "g": reg[worst],
                  "worst_design": worst if reg["dyad"] != reg["block"] else "tie"}
    raw = {d: {"raw_blend": v[d]["blend_raw"], "best_of_six": best[d],
               "raw_blend_is_best": v[d]["blend_raw"] > best[d]} for d in DESIGNS}
    return out, raw


METHOD_KEYS = {"pcc": "0.003|0.5", "blend_rank": "blend_rank|0.5", "blend_raw": "blend_raw|0.6",
               "blend_mu": "blend_mu|0.8",
               "product": "product", "katz": "katz", "si_logit": "si_logit", "rank3": "rank3"}


def worst_case(rho, regimes):
    out = {}
    for rounded in (False, True):
        v = {m: {g: (round(rho[m][g], 3) if rounded else rho[m][g]) for g in regimes} for m in rho}
        best = {g: max(v[m][g] for m in SIX) for g in regimes}
        g = {}
        worst = {}
        for m in v:
            reg = {r: best[r] - v[m][r] for r in regimes}
            worst[m] = max(regimes, key=lambda r: reg[r])
            g[m] = reg[worst[m]]
        out["g_from_rounded_levels" if rounded else "g"] = g
        if not rounded:
            out["worst_regime"] = worst
    g = out["g"]
    out["gaps_to_raw_blend"] = {"product": round(g["product"] - g["blend_raw"], 4),
                                "blend_rank": round(g["blend_rank"] - g["blend_raw"], 4)}
    return out


def regimes_check():
    with open(os.path.join(R, "pcc2_summary.json")) as f:
        s = json.load(f)["summary"]
    reg11 = ["dyad", "block", "random_0.25", "random_0.5", "cold_1", "cold_3", "cold_6",
             "sources_A", "sources_B", "thin_random_0.5", "thin_random_0.25"]
    rho07 = {m: {g: s["rho"][m][g] for g in reg11} for m in METHOD_KEYS}
    full = load("pcc2_c21.json", "c21")
    with open(os.path.join(R, "pcc2_c21regimes.json")) as f:
        reg21 = json.load(f)["c21regimes"]
    regs9 = ["dyad", "block"] + list(reg21)
    rho21 = {m: {"dyad": full[f"dyad_{m}"], "block": full[f"block_{m}"],
                 **{g: reg21[g][k] for g in reg21}} for m, k in METHOD_KEYS.items()}
    return {"2007_2012_11_regimes": worst_case(rho07, reg11),
            "2021_2023_9_regimes": worst_case(rho21, regs9)}


def cold_start_check():
    with open(os.path.join(R, "pcc2_summary.json")) as f:
        s = json.load(f)["summary"]
    with open(os.path.join(R, "gen_coldstart.json")) as f:
        g = json.load(f)["coldstart"]["1"]
    ten = {m: g[f"median_per_concept_{k}"] for m, k in
           [("product", "product"), ("hamacher", "hamacher"), ("katz", "katz"),
            ("rank3", "rank3"), ("si_logit", "softimpute_cv")]}
    five = {m: s["rho"][m]["cold_1"] for m in ["product", "hamacher", "katz", "rank3", "si_logit",
                                              "pcc", "blend_rank", "blend_raw"]}
    return {"five_repetitions_regime_comparison": five, "ten_repetitions_earlier_analysis": ten,
            "note": "gen_coldstart softimpute_cv is the soft-impute variant chosen by cross-validation "
                    "in that analysis; pcc2_summary si_logit is logit soft-impute"}


def main():
    cycles = {
        "2007_2012": load("pcc2_jrr.json", "jrr"),
        "2005_2006": load("pcc2_jrr05.json", "jrr05"),
        "2021_2023": load("pcc2_c21.json", "c21"),
    }
    res = {}
    for c, full in cycles.items():
        exact, raw = analyse(full)
        rnd, _ = analyse(full, rounded=True)
        res[c] = {
            "methods": exact,
            "g_from_rounded_levels": {m: rnd[m]["g"] for m in SIX},
            "g_unrounded_minus_rounded": {m: exact[m]["g"] - rnd[m]["g"] for m in SIX},
            "raw_blend_vs_reference": raw,
            "levels_4dp": {d: {m: round(full[f"{d}_{m}"], 4) for m in SIX + ["blend_raw"]}
                           for d in DESIGNS},
        }
    with open(os.path.join(R, "pcclim_lam0.json")) as f:
        lam0 = json.load(f)["lam0"]
    res["product_paths_2021_2023_single_dyads"] = {
        "pcc_lambda0": lam0["dyad"]["pcc_lam0_cap_2021"],
        "two_edge_path": lam0["paths_2021"]["dyad_path2"]["point"],
        "two_or_three_edge_path": lam0["paths_2021"]["dyad_path23"]["point"],
    }
    with open(os.path.join(R, "pcclim_s1.json")) as f:
        res["s1_by_pair_group"] = json.load(f)
    res["worst_case_over_regimes"] = regimes_check()
    res["cold_start_one_anchor"] = cold_start_check()
    with open(os.path.join(R, "report_checks.json"), "w") as f:
        json.dump(res, f, indent=1)
    for c in cycles:
        print(c)
        for m in SIX:
            e = res[c]["methods"][m]
            print(f"  {m:11s} dyad {e['regret']['dyad']:.4f}  block {e['regret']['block']:.4f}"
                  f"  g {e['g']:.4f} ({e['worst_design']})"
                  f"  g from rounded {res[c]['g_from_rounded_levels'][m]:.3f}")
        for d in DESIGNS:
            r = res[c]["raw_blend_vs_reference"][d]
            print(f"  raw blend {d}: {r['raw_blend']:.4f} vs best {r['best_of_six']:.4f}")
    print(res["product_paths_2021_2023_single_dyads"])
    for c, v in res["worst_case_over_regimes"].items():
        print(c, {m: round(x, 4) for m, x in v["g"].items()}, v["gaps_to_raw_blend"])
    print(res["cold_start_one_anchor"])


if __name__ == "__main__":
    main()
