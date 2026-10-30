import json, re
def norm(t): return re.sub(r'[^a-z0-9]','',(t or '').lower())[:120]
srcs = [("PubMed", json.load(open("pubmed_records.json"))), ("EuropePMC", json.load(open("epmc_E3_operator_clinical.json"))),
        ("SemanticScholar", json.load(open("s2_full.json"))), ("PubMed-B", json.load(open("pubmed_branchB.json"))), ("arXiv", json.load(open("arxiv.json")))]
n_ident = {k: len(v) for k, v in srcs}
merged, keys = [], {}
for name, recs in srcs:
    for r in recs:
        ks = [k for k in [("doi", (r.get("doi") or "").lower()), ("pmid", r.get("pmid") or (r["id"] if name.startswith("PubMed") else "")), ("t", norm(r["title"]))] if k[1]]
        hit = next((keys[k] for k in ks if k in keys), None)
        if hit is None:
            r = dict(r); r["sources"] = [name]; merged.append(r); idx = len(merged) - 1
        else:
            idx = hit; m = merged[idx]; m["sources"].append(name)
            if len(r.get("abstract") or "") > len(m.get("abstract") or ""): m["abstract"] = r["abstract"]
        for k in ks: keys[k] = idx
OP = re.compile(r"t-norm|triangular norm|fuzzy relation|max-product|max-min|sup-min|fuzzy operator|fuzzy conjunction|hamacher|łukasiewicz|lukasiewicz|fuzzy logic operator|fuzzy knowledge graph|fuzzy graph|compositional rule|fuzzy transitiv|fuzzy relational", re.I)
CL = re.compile(r"clinic|medic|health|disease|patient|depress|psychiatr|biomedic|diagnos|survey|epidemiolog|cancer|diabet|heart|cardi|tumou?r|covid|hospital|mental", re.I)
for i, m in enumerate(merged):
    txt = (m["title"] or "") + " " + (m.get("abstract") or "")
    m["uid"] = f"R{i+1:03d}"; m["flag_op"] = bool(OP.search(txt)); m["flag_cl"] = bool(CL.search(txt)); m["has_abs"] = bool(m.get("abstract"))
json.dump(merged, open("merged.json", "w"), indent=1)
print("identified", n_ident, "total", sum(n_ident.values()), "after dedup", len(merged))
print("both flags", sum(m["flag_op"] and m["flag_cl"] for m in merged), "no abstract", sum(not m["has_abs"] for m in merged))
