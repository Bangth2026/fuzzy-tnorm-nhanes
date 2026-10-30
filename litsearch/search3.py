import json, time, urllib.parse, urllib.request
A='("t-norm" OR "t-norms" OR "triangular norm" OR "triangular norms" OR "fuzzy relation composition" OR "max-product composition" OR "max-min composition" OR "sup-min composition" OR "fuzzy operators" OR "fuzzy conjunction" OR "Hamacher" OR "Lukasiewicz" OR "fuzzy logic operators")'
B='(graph OR graphs OR network OR networks OR "multi-hop" OR multihop OR transitivity OR transitive OR "knowledge base" OR "cognitive map" OR "cognitive maps" OR reasoning OR inference)'
C='(clinical OR medical OR medicine OR health OR disease OR patient OR patients OR depression OR depressive OR psychiatric OR biomedical OR diagnosis OR survey OR epidemiology)'
def run(q, fetch=True):
    recs, cursor, n = [], "*", None
    while cursor:
        u = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q},to_publication_date:2026-10-30", "per-page": 200, "cursor": cursor, "mailto": "truonghaibang@siu.edu.vn"})
        for i in range(6):
            try: r = json.loads(urllib.request.urlopen(u, timeout=120).read()); break
            except Exception as e: time.sleep(5 * (i + 1))
        else: raise SystemExit("failed")
        n = r["meta"]["count"]
        if not fetch: return n, []
        for w in r["results"]:
            inv = w.get("abstract_inverted_index") or {}
            ab = " ".join(t for t, _ in sorted(((t, p) for t, ps in inv.items() for p in ps), key=lambda z: z[1]))
            recs.append(dict(src="OpenAlex", id=w["id"], year=w.get("publication_year"), journal=((w.get("primary_location") or {}).get("source") or {}).get("display_name",""),
                             doi=(w.get("doi") or "").replace("https://doi.org/",""), title=w.get("title") or "", abstract=ab, type=w.get("type")))
        cursor = r["meta"].get("next_cursor"); time.sleep(1)
    return n, recs
out = {}
n1, _ = run(A, fetch=False); out["O1_operator"] = dict(query=A, count=n1); print("O1", n1); time.sleep(2)
n2, _ = run(f"{A} AND {B}", fetch=False); out["O2_operator_graph"] = dict(query=f"{A} AND {B}", count=n2); print("O2", n2); time.sleep(2)
n3, recs = run(f"{A} AND {B} AND {C}"); out["O3_full"] = dict(query=f"{A} AND {B} AND {C}", count=n3, n_fetched=len(recs)); print("O3", n3, len(recs))
json.dump(recs, open("openalex_O3.json", "w"), indent=1); json.dump(out, open("openalex_search.json", "w"), indent=1)
