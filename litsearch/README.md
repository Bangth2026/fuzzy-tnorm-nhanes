# Structured literature search (Supplementary Table S3)

Searched 30 October 2026. Question: has any study compared t-norms for composing clinical or health relations against an independently measured relation?

| Script | Source | Output |
| --- | --- | --- |
| `search.py`, `fetch_pm.py` | PubMed (E-utilities), operator terms | `pubmed_search.json`, `pubmed_records.json` (220) |
| `search2.py` | Europe PMC REST API, operator AND clinical terms | `epmc_search.json`, `epmc_E3_operator_clinical.json` (48) |
| `search_s2.py` | Semantic Scholar bulk API, operator AND graph/reasoning AND clinical terms | `s2_search.json`, `s2_full.json` (152) |
| `search4.py` | PubMed, fuzzy graph/relation AND clinical terms; arXiv API, operator AND clinical terms | `pubmed_branchB*.json` (46), `arxiv*.json` (5) |
| `search3.py` | OpenAlex (not used: the free daily quota for the network was exhausted on the search day) | – |
| `merge.py` | De-duplication by DOI, PMID and normalised title | `merged.json` (418 unique records) |
| `screen.py` | Title and abstract screening decisions | `screening_included.csv` (41 retained; 0 meet the criterion) |

Database counts change over time; rerunning the scripts on a later date will return more records. Scopus and Web of Science were not searched.
