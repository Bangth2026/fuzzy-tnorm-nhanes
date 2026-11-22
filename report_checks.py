"""Checks on reported quantities (v2.3.0).

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


if __name__ == "__main__":
    main()
