# Confirmatory test on NHANES 2005–2006 (registered before the data were downloaded)

Registered: 16 November 2026, before any 2005–2006 file was downloaded, cleaned or analysed by this project.
The commit that adds this file is the registration; its timestamp precedes the commit that adds any 2005–2006 result.

## Sample
NHANES 2005–2006 (cycle D), adults aged 18 or older with all nine PHQ-9 items valid and a positive MEC weight.
Concepts and membership functions are those of `clean2.py` without change, except inactivity, whose 2005–2006
questions differ: 0 if vigorous leisure-time activity in the past 30 days (PAD200 = 1), 0.5 if only moderate
activity (PAD320 = 1), 1 otherwise. A sensitivity analysis removes inactivity (23 concepts).

## Design
The development relation is the 2013–2018 relation of the paper (`data2_dev.pkl`), unchanged; nothing is tuned
on 2005–2006. Target: the relation rebuilt from 2005–2006. Ranks and penalties are those chosen within development
(rank 3; soft-impute penalty by dyad cross-validation within development). Intervals: jackknife repeated replication
over the strata of both periods; two-sided 95% intervals.

## Hypotheses and decision rules
H1 (completion, scattered gaps). With one dyad withheld at a time, rank-3 completion has a higher Spearman rho with
   the 2005–2006 relation than the best path under the product (552 ordered pairs).
   Supported if the interval for rank-3 minus product lies above 0.
H2 (t-norm, symmetric relations). On the symmetric relation sqrt(mu(i->j) mu(j->i)) (276 dyads), the product and
   the Hamacher t-norm do not differ.
   Supported if the interval for product minus Hamacher contains 0.
H3 (minimum rule). On directed relations the product has a higher rho than the minimum.
   Supported if the interval for product minus minimum lies above 0.
H4 (whole relation types). With each of the 10 relation-type blocks withheld in turn, the pair-weighted mean rho of
   the product is higher than that of soft-impute with its penalty chosen on the other blocks.
   Supported if the interval for the difference lies above 0.
H5 (walks). In the same block design, the product and Katz do not differ.
   Supported if the interval for the difference contains 0.
H6 (clinical query). For the nine PHQ-9 symptoms, the product's best path recovers the 2005–2006 top modifiable
   correlate (smoking, heavy drinking, inactivity, short sleep, obesity) for at least as many symptoms as the
   minimum rule (ties in a prediction share the credit). Descriptive; no interval.

All six results are reported whatever their direction. No other analysis of 2005–2006 is planned; any added later
will be labelled exploratory.
