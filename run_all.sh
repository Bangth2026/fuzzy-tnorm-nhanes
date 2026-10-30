#!/usr/bin/env bash
set -euo pipefail
python3 download_nhanes.py
python3 clean2.py
python3 analysis2.py main
python3 analysis2.py nulls
python3 analysis2.py rq3
python3 power2.py N2
python3 power2.py N1
python3 nresp_check.py
python3 describe2.py
python3 sens_expert.py
for p in m1 m2 m3 m4 m5 m6 m9 minor; do python3 revision.py "$p"; done
python3 sanity2.py
for p in r3 boot sim null misc; do python3 revision2.py "$p"; done
python3 revision2_extra.py
for p in blocks sim2 reverse; do python3 revision3.py "$p"; done
for p in sens swap split simdesign; do python3 revision4.py "$p"; done
python3 stats2.py
