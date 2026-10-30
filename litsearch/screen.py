import json, csv
m = {r['uid']: r for r in json.load(open('merged.json'))}
A = dict(R200="max-min composition in an ultrasonography expert system; evaluated as diagnostic accuracy",
 R399="max-min composition for nursing diagnosis; agreement with expert panel", R401="fuzzy relation symptom-syndrome (gastric cancer); coincidence rate",
 R407="hierarchical fuzzy relational inference (arthritic diseases)", R409="CADIAG-IV knowledge model: findings-disease relations as fuzzy relations",
 R410="MEDUSA fuzzy relations for acute abdominal pain", R412="max-min composite fuzzy relation for diagnosis (genital/urinary inflammation)",
 R413="chained unidirectional fuzzy relations among diagnostic units; concordance with physicians", R396="fuzzy symptom-syndrome relations (TCM)",
 R397="fuzzy relation method vs naive Bayes; classification", R392="fuzzy relations for ECG classification", R386="mediative fuzzy relation/projection, post-COVID diagnosis",
 R345="t-norm rule firing in a fuzzy inference classifier (breast tumours)", R146="new t-norm in a neuro-fuzzy lung-cancer risk model",
 R144="Hamacher and OWA combination of immune parameters (dengue severity)", R145="Hamacher and OWA combination of cytokines (dengue)",
 R195="fuzzy rules for HIV-1 drug resistance; correlation with virological outcome", R220="fuzzy rule reasoning with fuzzy operators (melanoma)",
 R316="max-min composition (mouth cancer); no empirical evaluation reported", R148="fuzzy aggregation operators for dental X-ray diagnosis",
 R025="Hamacher-motivated aggregation in an autism assessment model", R233="Lukasiewicz inference for nephropathia epidemica diagnosis (title only)",
 R275="diagnosis from multidimensional fuzzy relations (title only)", R307="fuzzy diagnosis matrix (title only)", R357="fuzzy relationships in medical knowledge-based systems (title only)",
 R358="interpretation of CADIAG-2 (title only)", R415="bipolar fuzzy relation equations for clinical decision support; minimum t-norm adopted, theoretical",
 R362="Pythagorean fuzzy triple-I inference; illustrative medical example", R372="fuzzy graph propagation for Alzheimer's risk prediction", R408="fuzzy graph classification of cardiac function")
B = dict(R263="three t-norms compared for path selection in a prototype tree; judged by cell-classification F1",
 R187="similarity classifier from Yu's norms vs other models; classification accuracy", R364="applicability of t-norms/t-conorms in differential diagnosis of epileptic events (title only)",
 R129="customised Lukasiewicz t-norm in fuzzy-rough feature selection; classification accuracy", R157="max-min composition vs a fuzzy-probabilistic method on simulated patients")
C = dict(R378="fuzzy transitive closure on community fuzzy cognitive maps (men's wellbeing)", R381="fuzzy transitive closure on perinatal-wellness maps",
 R383="fuzzy transitive closure on birth-evacuation maps", R391="fuzzy transitive closure on maps of short-birth-interval causes",
 R394="fuzzy transitive closure of odds-ratio relations from a dengue-trial household survey; compared qualitatively with trial findings")
D = dict(R002="Schweizer t-norm parameter fitted to reproduce an existing aggregation operator (nutrient evaluation); clinical validity not established")
rows = []
for g, d in [("A", A), ("B", B), ("C", C), ("D", D)]:
    for u, note in d.items():
        r = m[u]; rows.append(dict(uid=u, group=g, year=r['year'], title=r['title'], venue=r.get('journal') or '', doi=r.get('doi') or '', sources="; ".join(r['sources']), note=note,
                                   compares_operators="yes" if g in "B" else "no", independent_relation_target="no", meets_criterion="no"))
with open("screening_included.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print({g: sum(r['group'] == g for r in rows) for g in "ABCD"}, len(rows))
