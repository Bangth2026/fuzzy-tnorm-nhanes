**Table S3.** Structured literature search for studies that test the t-norm used to compose clinical relations (searched 30 October 2026). (a) Sources and records; (b) studies retained at title and abstract screening.

| Source (interface) | Search, title and abstract fields | Records |
| --- | --- | --- |
| PubMed (E-utilities) | operator terms | 220 |
| Europe PMC (REST API) | operator terms AND clinical terms | 48 |
| Semantic Scholar (bulk API) | operator terms AND graph or reasoning terms AND clinical terms | 152 |
| PubMed (E-utilities) | fuzzy graph or fuzzy relation terms AND clinical terms | 46 |
| arXiv (API) | operator terms AND clinical terms | 5 |
| Total identified; unique after removing duplicates | | 471; 418 |
| Retained at title and abstract screening | | 41 |
| Compared two or more t-norms for composition against an independently measured relation | | 0 |

Operator terms: "t-norm(s)", "triangular norm(s)", "fuzzy relation composition", "max-product composition", "max-min composition", "sup-min composition", "fuzzy operators", "fuzzy conjunction", Hamacher, Lukasiewicz, "fuzzy logic operators". Graph or reasoning terms: graph(s), network(s), multi-hop, transitivity or transitive, "knowledge base", "cognitive map(s)", reasoning, inference. Clinical terms: clinical, medical, medicine, health, disease(s), patient(s), depression or depressive, psychiatric, biomedical, diagnosis, survey, epidemiology. Fuzzy graph or relation terms: "fuzzy (knowledge) graph(s)", "fuzzy relation(s)", "fuzzy relational", "compositional rule of inference", "fuzzy transitive" or "fuzzy transitivity". One reviewer (H.B.T.) screened all records on title and abstract; full texts were not assessed, and six records without abstracts were judged on titles. Scopus and Web of Science were not searched. Abstract-level searching misses studies that use a t-norm without naming it in the abstract; for example, the query-answering studies cited in Section 1 [11,12] were not retrieved. The exact queries, retrieved records and screening decisions are in the code repository (folder litsearch/).

Eligibility: a record was retained at title and abstract screening if it applied fuzzy relational composition, a t-norm or another fuzzy conjunction operator to clinical, health or population-health data or knowledge. It met the criterion of interest if it also compared two or more t-norms for composing relations and evaluated them against a relation measured independently of the one composed. Records were excluded if the operator served only multi-criteria aggregation of expert ratings, control, image processing or algebra, or if "Hamacher", "Łukasiewicz" or "T-norm" denoted a person, an institution or a normalisation of test scores.

Flow (PRISMA-style): records identified 471 (PubMed 220 + 46, Europe PMC 48, Semantic Scholar 152, arXiv 5) → duplicates removed 53 → records screened on title and abstract 418 → excluded 377 → retained 41 (groups A–D below) → meeting the criterion of interest 0.

| Group | Studies | What they do | Comparison of operators | Evaluation target |
| --- | --- | --- | --- | --- |
| A | 30 | Compose fuzzy relations or fire fuzzy rules with one fixed operator (usually max–min) for diagnosis or risk | No | Diagnostic labels, expert agreement or none |
| B | 5 | Compare two or more operators | Yes | Classification accuracy, or simulated patients |
| C | 5 | Compose community-elicited or survey-derived health relations along paths by fuzzy transitive closure | No (one fixed operator) | None, or qualitative agreement with a trial |
| D | 1 | Fit a parametric t-norm to reproduce an existing aggregation operator | Within one family | The earlier operator |

Groups B–D:

- B. Prototype Trees to Develop an Interpretable Automated Pipeline for Nasal Cytology (2025; doi:10.1109/FUZZ62266.2025.11152079). Three t-norms compared for path selection in a prototype tree; judged by cell-classification F1.
- B. Similarity classifier using similarity measure derived from Yu's norms in classification of medical data sets (2007; doi:10.1016/j.compbiomed.2006.10.005). Similarity classifier from Yu's norms vs other models; classification accuracy.
- B. Aplicabilidade das Operações Aritméticas Difusas t-normas e t-conormas no Diagnóstico Diferencial de Eventos Epilépticos e Eventos Não Epilépticos (2007; no DOI). Applicability of t-norms/t-conorms in differential diagnosis of epileptic events (title only).
- B. Prediction of cancer using customised fuzzy rough machine learning approaches (2019; doi:10.1049/htl.2018.5055). Customised Lukasiewicz t-norm in fuzzy-rough feature selection; classification accuracy.
- B. A fuzzy probabilistic method for medical diagnosis (2015; doi:10.1007/s10916-015-0203-9). Max-min composition vs a fuzzy-probabilistic method on simulated patients.
- C. Community Views of Determinants of Men's Wellbeing in Guatemala: A Study Using Fuzzy Cognitive Mapping (2026; doi:10.1177/2752535X241312378). Fuzzy transitive closure on community fuzzy cognitive maps (men's wellbeing).
- C. Keeping birth at home: Community and service provider visions for perinatal wellness and continued Inuit childbirth in Nunavik (2024; doi:10.1016/j.wombi.2024.101839). Fuzzy transitive closure on perinatal-wellness maps.
- C. Giving birth in a good way when it must take place away from home: Participatory research into visions of Inuit families and their Montreal-based medical providers (2023; doi:10.1111/birt.12726). Fuzzy transitive closure on birth-evacuation maps.
- C. Causes of short birth interval (kunika) in Bauchi State, Nigeria: systematizing local knowledge with fuzzy cognitive mapping (2021; doi:10.1186/s12978-021-01066-2). Fuzzy transitive closure on maps of short-birth-interval causes.
- C. The women made it work: fuzzy transitive closure of the results chain in a dengue prevention trial in Mexico (2017; doi:10.1186/s12889-017-4301-0). Fuzzy transitive closure of odds-ratio relations from a dengue-trial household survey; compared qualitatively with trial findings.
- D. Aggregation of Fuzzy Nutrient-Level Evaluations Using a Parametric T-Norm: Mathematical Properties, Calibration, and a Proof-of-Concept Analysis (2026; doi:10.3390/nu18193151). Schweizer t-norm parameter fitted to reproduce an existing aggregation operator (nutrient evaluation); clinical validity not established.
