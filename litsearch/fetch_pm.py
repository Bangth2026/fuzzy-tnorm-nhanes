import json, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
s = json.load(open("pubmed_search.json")); ids = s["S1_operator"]["ids"]
recs = []
for i in range(0, len(ids), 100):
    u = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids[i:i+100]), "retmode": "xml"})
    x = ET.fromstring(urllib.request.urlopen(u, timeout=120).read())
    for a in x.findall(".//PubmedArticle"):
        pmid = a.findtext(".//PMID"); t = "".join(a.find(".//ArticleTitle").itertext()) if a.find(".//ArticleTitle") is not None else ""
        ab = " ".join("".join(e.itertext()) for e in a.findall(".//AbstractText"))
        yr = a.findtext(".//PubDate/Year") or (a.findtext(".//PubDate/MedlineDate") or "")[:4]
        j = a.findtext(".//Journal/ISOAbbreviation") or a.findtext(".//Journal/Title")
        doi = next((e.text for e in a.findall(".//ArticleId") if e.get("IdType") == "doi"), "")
        recs.append(dict(src="PubMed", id=pmid, year=yr, journal=j, doi=doi, title=t, abstract=ab,
                         in_S2=pmid in s["S2_operator_graph"]["ids"], in_S3=pmid in s["S3_full"]["ids"]))
    time.sleep(0.5)
json.dump(recs, open("pubmed_records.json", "w"), indent=1); print(len(recs))
