import json, time, urllib.parse, urllib.request, sys
A = ('("t-norm"[tiab] OR "t-norms"[tiab] OR "triangular norm"[tiab] OR "triangular norms"[tiab] OR "fuzzy relation composition"[tiab] '
     'OR "max-product composition"[tiab] OR "max-min composition"[tiab] OR "sup-min composition"[tiab] OR "fuzzy operators"[tiab] '
     'OR "fuzzy conjunction"[tiab] OR "Hamacher"[tiab] OR "Lukasiewicz"[tiab] OR "fuzzy logic operators"[tiab])')
B = ('(graph[tiab] OR graphs[tiab] OR network[tiab] OR networks[tiab] OR "multi-hop"[tiab] OR multihop[tiab] OR transitivity[tiab] '
     'OR transitive[tiab] OR "knowledge base"[tiab] OR "cognitive map"[tiab] OR "cognitive maps"[tiab] OR reasoning[tiab] OR inference[tiab])')
C = ('(clinical[tiab] OR medical[tiab] OR medicine[tiab] OR health[tiab] OR disease[tiab] OR diseases[tiab] OR patient[tiab] OR patients[tiab] '
     'OR depression[tiab] OR depressive[tiab] OR psychiatric[tiab] OR biomedical[tiab] OR diagnosis[tiab] OR survey[tiab] OR epidemiolog*[tiab])')
Q = {"S1_operator": A, "S2_operator_graph": f"{A} AND {B}", "S3_full": f"{A} AND {B} AND {C}"}
def get(url):
    for i in range(4):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception as e:
            time.sleep(3)
    raise
base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
out = {}
for k, q in Q.items():
    u = base + "esearch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "term": q + ' AND ("1965/01/01"[dp] : "2026/10/30"[dp])', "retmode": "json", "retmax": 5000})
    r = json.loads(get(u)); ids = r["esearchresult"]["idlist"]; out[k] = {"query": q, "count": int(r["esearchresult"]["count"]), "ids": ids}
    print(k, out[k]["count"]); time.sleep(0.5)
json.dump(out, open("pubmed_search.json", "w"), indent=1)
