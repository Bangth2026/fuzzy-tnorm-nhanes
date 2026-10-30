import json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
# Branch B: fuzzy graph / fuzzy relation constructs in clinical settings (PubMed + Europe PMC)
F_pm = ('("fuzzy knowledge graph"[tiab] OR "fuzzy knowledge graphs"[tiab] OR "fuzzy graph"[tiab] OR "fuzzy graphs"[tiab] OR "fuzzy relation"[tiab] '
        'OR "fuzzy relations"[tiab] OR "fuzzy relational"[tiab] OR "compositional rule of inference"[tiab] OR "fuzzy transitive"[tiab] OR "fuzzy transitivity"[tiab])')
C_pm = ('(clinical[tiab] OR medical[tiab] OR medicine[tiab] OR health[tiab] OR disease[tiab] OR diseases[tiab] OR patient[tiab] OR patients[tiab] '
        'OR depression[tiab] OR depressive[tiab] OR psychiatric[tiab] OR biomedical[tiab] OR diagnosis[tiab] OR survey[tiab] OR epidemiolog*[tiab])')
q = f"{F_pm} AND {C_pm}" + ' AND ("1965/01/01"[dp] : "2026/10/30"[dp])'
r = json.loads(urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode({"db":"pubmed","term":q,"retmode":"json","retmax":5000}), timeout=60).read())
ids = r["esearchresult"]["idlist"]; print("PubMed branch B", r["esearchresult"]["count"])
recs = []
for i in range(0, len(ids), 100):
    x = ET.fromstring(urllib.request.urlopen("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode({"db":"pubmed","id":",".join(ids[i:i+100]),"retmode":"xml"}), timeout=120).read())
    for a in x.findall(".//PubmedArticle"):
        t = a.find(".//ArticleTitle"); recs.append(dict(src="PubMed-B", id=a.findtext(".//PMID"), year=a.findtext(".//PubDate/Year") or (a.findtext(".//PubDate/MedlineDate") or "")[:4],
            journal=a.findtext(".//Journal/ISOAbbreviation"), doi=next((e.text for e in a.findall(".//ArticleId") if e.get("IdType")=="doi"), ""),
            title="".join(t.itertext()) if t is not None else "", abstract=" ".join("".join(e.itertext()) for e in a.findall(".//AbstractText"))))
    time.sleep(0.5)
json.dump(dict(query=q, count=int(r["esearchresult"]["count"])), open("pubmed_branchB_search.json","w"), indent=1)
json.dump(recs, open("pubmed_branchB.json","w"), indent=1); print(len(recs))
# arXiv: operator AND clinical
ops = ['"t-norm"','"t-norms"','"triangular norm"','"triangular norms"','"fuzzy logic operators"','"fuzzy operators"','Lukasiewicz','Hamacher']
cl = ['clinical','medical','health','disease','patient','patients','depression','psychiatric','biomedical','diagnosis','survey']
aq = "(" + " OR ".join(f"abs:{o}" for o in ops) + ") AND (" + " OR ".join(f"abs:{c}" for c in cl) + ")"
ns = {'a':'http://www.w3.org/2005/Atom','o':'http://a9.com/-/spec/opensearch/1.1/'}
ax, start, total = [], 0, None
while total is None or start < total:
    u = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": aq, "start": start, "max_results": 200})
    root = ET.fromstring(urllib.request.urlopen(u, timeout=120).read()); total = int(root.findtext("o:totalResults", namespaces=ns))
    for e in root.findall("a:entry", ns):
        ax.append(dict(src="arXiv", id=e.findtext("a:id",namespaces=ns), year=e.findtext("a:published",namespaces=ns)[:4], journal="arXiv", doi="",
                       title=" ".join(e.findtext("a:title",namespaces=ns).split()), abstract=" ".join(e.findtext("a:summary",namespaces=ns).split())))
    start += 200; time.sleep(3)
ax = [x for x in ax if x["year"] <= "2026"]
json.dump(dict(query=aq, count=total), open("arxiv_search.json","w"), indent=1); json.dump(ax, open("arxiv.json","w"), indent=1); print("arXiv", total, len(ax))
