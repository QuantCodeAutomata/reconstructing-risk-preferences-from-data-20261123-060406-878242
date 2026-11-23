#!/usr/bin/env bash
# Runs the test suite and all four experiments of
# "Reconstructing risk preferences from data" (Section 3).
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1_gbm
python -m exp.exp2_heston
python -m exp.exp3_discrete
python -m exp.exp4_incompatible
