import json, time, urllib.parse, urllib.request
A='("t-norm" | "t-norms" | "triangular norm" | "triangular norms" | "fuzzy relation composition" | "max-product composition" | "max-min composition" | "sup-min composition" | "fuzzy operators" | "fuzzy conjunction" | Hamacher | Lukasiewicz | "fuzzy logic operators")'
B='(graph | graphs | network | networks | "multi-hop" | multihop | transitivity | transitive | "knowledge base" | "cognitive map" | "cognitive maps" | reasoning | inference)'
C='(clinical | medical | medicine | health | disease | patient | patients | depression | depressive | psychiatric | biomedical | diagnosis | survey | epidemiology)'
def run(q, fetch=True):
    recs, token, total = [], None, None
    while True:
        p = {"query": q, "fields": "title,abstract,year,venue,externalIds,publicationTypes", "publicationDateOrYear": ":2026-10-30"}
        if token: p["token"] = token
        u = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode(p)
        for i in range(8):
            try: r = json.loads(urllib.request.urlopen(u, timeout=120).read()); break
            except Exception as e: time.sleep(4 * (i + 1))
        else: raise SystemExit("failed " + q[:40])
        total = r.get("total")
        if not fetch: return total, []
        for w in r.get("data") or []:
            ex = w.get("externalIds") or {}
            recs.append(dict(src="SemanticScholar", id=w.get("paperId"), year=w.get("year"), journal=w.get("venue") or "", doi=ex.get("DOI",""), pmid=ex.get("PubMed",""),
                             arxiv=ex.get("ArXiv",""), title=w.get("title") or "", abstract=w.get("abstract") or ""))
        token = r.get("token")
        if not token: return total, recs
        time.sleep(1.5)
out = {}
for k, q, f in [("S2_1_operator", A, False), ("S2_2_operator_graph", f"{A} + {B}", False), ("S2_3_full", f"{A} + {B} + {C}", True)]:
    n, recs = run(q, f); out[k] = dict(query=q, count=n, n_fetched=len(recs)); print(k, n, len(recs)); time.sleep(2)
    if f: json.dump(recs, open("s2_full.json", "w"), indent=1)
json.dump(out, open("s2_search.json", "w"), indent=1)
