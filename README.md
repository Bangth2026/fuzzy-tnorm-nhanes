# Testing t-norms on a population fuzzy graph of depressive symptoms (NHANES)

[![DOI](https://zenodo.org/badge/1439206241.svg)](https://doi.org/10.5281/zenodo.23790456)

Code and derived results for the manuscript

> Truong Hai Bang, Van An Phan Thi. *Testing composition rules on a graded association graph: a temporally held-out case study of depressive symptoms in NHANES* (earlier titles: *Testing t-norm choice on a population fuzzy graph of depressive symptoms and their correlates*; *Which t-norm should a clinical fuzzy graph reason with?*). Under review at the Journal of Healthcare Informatics Research.

Version 1.3.0 contains every analysis reported in the second revision, including stronger baselines, link-prediction indices, a learned path model, per-source ranking, calibration in the withheld design, random masking, cold start, age–sex standardisation, signed relations and a respondent-level simulation of chained versus latent-factor generation (`revision6.py`, `rq6lib.py`). The literature search behind Supplementary Table S3 is in `litsearch/` (`tableS3_v7.md` gives the search strings, eligibility criteria and record flow).

The pipeline builds a typed, directed fuzzy relation graph of 24 concepts (nine PHQ-9 symptoms and 15 correlates) from NHANES 2013–2018 (development) and 2007–2012 (held-out), using a survey-weighted fuzzy certainty factor. It then audits:

- the cost of binarisation (RQ1)
- the structure against null models and the concept-type schema (RQ2)
- multi-hop inference with five t-norms against held-out relations (RQ3)

Every number in the manuscript is read from result files in `results2/`: `paper_stats.json` for the original analyses and `revision*_*.json` for the analyses added in review.

## Requirements

Python 3.13 with the versions used for the paper:

```
pip install -r requirements.txt
```

## Data

NHANES files are public and are not redistributed here. Run

```
python3 download_nhanes.py
```

This fetches the 72 XPT files (cycles E–J; components DPQ, DEMO, SMQ, BMX, DIQ, MCQ, PAQ, HIQ, ALQ, SLQ, HSQ, BPQ) into `raw/` and compares them with `raw/MANIFEST.md5`, the checksums of the files the published results were computed from.

## Run order

Run all steps with `bash run_all.sh`, or run them one at a time:

| Step | Command | Output |
| --- | --- | --- |
| 1 | `python3 clean2.py` | `data2_dev.pkl`, `data2_hold.pkl`, `results2/flow.json` |
| 2 | `python3 analysis2.py main` | RQ1: binarisation cost, bootstrap, null-calibrated cost, held-out, sensitivity |
| 3 | `python3 analysis2.py nulls` | RQ2: N1, N2, N3 in both periods |
| 4 | `python3 analysis2.py rq3` | RQ3: leave-edge-out t-norm inference, T-transitivity, bootstrap CIs |
| 5 | `python3 power2.py N2`, `python3 power2.py N1` | size and power calibration |
| 6 | `python3 nresp_check.py` | density of respondent-permutation null graphs |
| 7 | `python3 describe2.py` | descriptive results (`results2/describe.json`) |
| 8 | `python3 sens_expert.py` | membership-function variants (`results2/sens_expert.json`) |
| 9 | `python3 revision.py m1` … `minor` | analyses requested in review (`results2/revision_*.json`) |
| 10 | `python3 sanity2.py` | implementation checks (`results2/sanity2.json`) |
| 11 | `python3 revision2.py r3`, `boot`, `sim`, `null`, `misc`; `python3 revision2_extra.py` | second-review analyses: non-path predictors, parametric t-norms, scale, withheld blocks, simulation, strength-preserving null, detectable cost (`results2/revision2_*.json`) |
| 12 | `python3 revision3.py blocks`, `sim2`, `reverse` | third-review analyses: withheld blocks with cross-validated completion rank, bootstrap intervals and sign tests; simulation matched to the data's noise and scale; zero reverse edges (`results2/revision3_*.json`) |
| 13 | `python3 revision4.py sens`, `swap`, `split`, `simdesign` | fourth-review analyses: RQ3 under every sensitivity setting and with binary PHQ-9 items, periods swapped (200 replicates), 20 within-period splits by sampling unit, realised simulation design (`results2/revision4_*.json`) |
| 14 | `python3 stats2.py` | `results2/paper_stats.json` |
| 15 | `python3 revision5.py` | calibration, edge-cut sensitivity, initial vs revised membership functions (`results2/revision5.json`) |
| 16 | `python3 revision6.py main`, `blocks`, `mask`, `strat`, `signed`, `sim`, `misc` | sixth-review analyses (`results2/revision6_<part>.json`): stronger baselines and link-prediction indices with concept jackknife, withheld blocks with recalibration on measured pairs, random masking and cold start, age–sex standardised relation, signed relation, respondent-level simulation, Table S17 completion, grid error, bootstrap edge filter, sleep 2015–2018, split-half reliability, modularity interval, age–sex description |
| 17 | `python3 figures6.py calib`, `forest`, `heatmap` | Fig. 2, Fig. 3 and Fig. S1 (`figures/`) |

`rq6lib.py` can be reused on any square relation matrix with concept types and prevalences to run the procedure of Section 7.3 of the paper: leave-dyad-out composition under several t-norms, link-prediction indices and a learned path model, predictors without paths, within-block and per-source evaluation, recalibration, and withheld-block, random and cold-start designs.

Master seed: 20261029. The full run takes several hours on one CPU, mostly in the null models and bootstraps.

## Variance estimation

All intervals use the Rao–Wu rescaling bootstrap: in each stratum, n_h − 1 primary sampling units are drawn and the weights are rescaled. RQ3 intervals resample both periods.

## Clinical review of membership functions

The clinical co-author reviewed the correlate membership functions on the form in `review/`:

- `Phieu_duyet_ham_thanh_vien_VanAn.xlsx` is the blank form.
- `Phieu_duyet_ham_thanh_vien_VanAn_reviewed.xlsx` is the completed form.

The forms are in Vietnamese. The five revised functions are marked `[expert]` in `clean2.py`.

## Archived results

- Results under the initial membership functions are in `results2_pre_expert_review/`, and `clean2_pre_expert_review.py` is the cleaning script that produced them. They are reported as Supplementary Table S1.
- Results before the switch to the Rao–Wu bootstrap are in `results2_pre_revision/`.
- `legacy/` keeps an earlier, superseded design (a graph with a provisional literature layer). The manuscript does not use it; it is kept for the record.

## Licence

Code: MIT (see `LICENSE`). Derived results and review forms: CC BY 4.0.

## Citation

Archived on Zenodo. All versions (concept DOI, cited in the paper): https://doi.org/10.5281/zenodo.23790456. Version 1.1.0 (the revised manuscript): https://doi.org/10.5281/zenodo.23794509. Version 1.0.0: https://doi.org/10.5281/zenodo.23790457. See also `CITATION.cff`.
