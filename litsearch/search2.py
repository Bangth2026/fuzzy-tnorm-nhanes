import json, time, urllib.parse, urllib.request
terms = ['"t-norm"','"t-norms"','"triangular norm"','"triangular norms"','"fuzzy relation composition"','"max-product composition"',
         '"max-min composition"','"sup-min composition"','"fuzzy operators"','"fuzzy conjunction"','"Hamacher"','"Lukasiewicz"','"fuzzy logic operators"']
A = "(" + " OR ".join(f"TITLE:{t} OR ABSTRACT:{t}" for t in terms) + ")"
clin = ['clinical','medical','medicine','health','disease','patient','patients','depression','psychiatric','biomedical','diagnosis','survey','epidemiology']
C = "(" + " OR ".join(f"TITLE:{t} OR ABSTRACT:{t}" for t in clin) + ")"
def epmc(q):
    recs, cursor = [], "*"
    while True:
        u = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({"query": q, "format": "json", "pageSize": 1000, "cursorMark": cursor, "resultType": "core"})
        r = json.loads(urllib.request.urlopen(u, timeout=120).read())
        for x in r["resultList"]["result"]:
            recs.append(dict(src="EuropePMC", id=x.get("id"), source=x.get("source"), pmid=x.get("pmid",""), year=x.get("pubYear"), journal=(x.get("journalInfo") or {}).get("journal",{}).get("title","") or x.get("bookOrReportDetails",{}).get("publisher",""),
                             doi=x.get("doi",""), title=x.get("title",""), abstract=x.get("abstractText","")))
        nc = r.get("nextCursorMark")
        if not nc or nc == cursor or not r["resultList"]["result"]: return r["hitCount"], recs
        cursor = nc; time.sleep(0.3)
out = {}
for k, q in {"E1_operator": A, "E3_operator_clinical": f"{A} AND {C}"}.items():
    n, recs = epmc(q + " AND (FIRST_PDATE:[1965-01-01 TO 2026-10-30])"); out[k] = dict(query=q, count=n, n_fetched=len(recs)); print(k, n, len(recs))
    json.dump(recs, open(f"epmc_{k}.json", "w"), indent=1)
json.dump(out, open("epmc_search.json", "w"), indent=1)
