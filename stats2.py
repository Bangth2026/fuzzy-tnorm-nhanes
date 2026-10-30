"""Single source of truth for every number in the rewritten manuscript."""
import json, os
r = lambda x, n=3: round(x, n)
L = lambda f: json.load(open(f"results2/{f}"))
R1, R2, R3, F = L("rq1.json"), L("rq2.json"), L("rq3.json"), L("flow.json")
S = {"flow": F, "thresholds": {k: r(v) for k, v in R1["thresholds"].items()},
     "main": {"edges": R1["rq1_main"]["n_edges"], "Q": r(R1["rq1_main"]["Q_mean"]), "ncomm": R1["rq1_main"]["n_comm"],
              "communities": R1["rq1_main"]["communities"]},
     "binarised": {k: {"edges": v["edges_kept"], "ncomm": v["n_comm"], "Q": r(v["Q_on_GF_mean"]), "cost": r(v["cost"]),
                       "ci": [r(x) for x in R1["rq1_boot_ci"][k]], "NMI": r(v["NMI"], 2), "ARI": r(v["ARI"], 2),
                       "null_mean": r(R1["cost_null_N2"][k]["mean"]), "null_range": [r(R1["cost_null_N2"][k]["q025"]), r(R1["cost_null_N2"][k]["q975"])],
                       "null_p": r(R1["cost_null_N2"][k]["p_ge"])} for k, v in R1["rq1_main"]["binarised"].items()},
     "heldout": {"Q": r(R1["heldout"]["Q_mean"]), "ncomm": R1["heldout"]["n_comm"], **{k: r(v) for k, v in R1["heldout_vs_dev"].items()},
                 "costs": {k: r(v["cost"]) for k, v in R1["heldout"]["binarised"].items()}},
     "sensitivity": {k: {"edges": v["n_edges"], "Q": r(v["Q_mean"]), "ncomm": v["n_comm"],
                         "costs": {kk: r(vv["cost"]) for kk, vv in v["binarised"].items()}} for k, v in R1["sensitivity"].items()},
     "rq2": {k: {kk: (r(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in R2.items()},
     "nresp_density": L("nresp_density.json"), "power_N2": L("power_N2.json"),
     "power_N1": L("power_N1.json") if os.path.exists("results2/power_N1.json") else None,
     "rq3": {"held_out": {k: {kk: r(vv) for kk, vv in v.items()} for k, v in R3["held_out"].items()},
             "ci": {k: [r(x) for x in v] for k, v in R3["spearman_ci"].items()},
             "diff_vs_product": {k: [r(x) for x in v] for k, v in R3["spearman_diff_vs_product_ci"].items()},
             "transitivity": {k: {kk: r(vv) for kk, vv in v.items()} for k, v in R3["transitivity_heldout"].items()}}}
S["describe"] = L("describe.json")
S["sens_expert"] = {k: ({kk: (r(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else r(v))
                    for k, v in L("sens_expert.json").items()}
S["rq3"]["bootstrap"] = R3.get("bootstrap", "held-out only")
REV = {}
for part in ["m1", "m2", "m3", "m4", "m5", "m6", "m9", "minor"]:
    REV.update(L(f"revision_{part}.json"))
S["revision"] = REV
S["sanity"] = L("sanity2.json")
S["pre_review"] = json.load(open("results2_pre_expert_review/paper_stats.json"))
json.dump(S, open("results2/paper_stats.json", "w"), indent=1)
print(json.dumps(S["binarised"], indent=0)[:3000])
# second review
S["revision2"] = {p: L(f"revision2_{p}.json") for p in ["r3", "sim", "misc", "boot", "null", "extra"] if os.path.exists(f"results2/revision2_{p}.json")}
json.dump(S, open("results2/paper_stats.json", "w"), indent=1)
S["revision3"] = {p: L(f"revision3_{p}.json") for p in ["blocks", "sim2", "reverse"] if os.path.exists(f"results2/revision3_{p}.json")}
S["revision4"] = {p: L(f"revision4_{p}.json") for p in ["sens", "swap", "split", "simdesign"] if os.path.exists(f"results2/revision4_{p}.json")}
json.dump(S, open("results2/paper_stats.json", "w"), indent=1)
