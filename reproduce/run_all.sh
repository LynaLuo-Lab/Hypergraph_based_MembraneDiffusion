#!/bin/bash
# figures and numbers of the held-out second-half evaluation from the rollouts in assets/ (level 1, no GPU).
# LEVEL=2 first redraws the rollouts, the 40,000-frame forecasts and the squash traces from the checkpoints (GPU, DEVICE=cuda by default).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p ../logs ../results ../figures
if [ "${LEVEL:-1}" = 2 ]; then
  $PY roll_heldout.py 2>&1 | grep -v '^  ' > ../logs/roll_heldout.log
  $PY roll_forecast.py 2>&1 | grep -v '^  ' > ../logs/roll_forecast.log
  $PY trace_squash.py 2>&1 | grep -v '^  ' > ../logs/trace_squash.log
fi
$PY score_histogram.py > ../logs/score_histogram.log
$PY figures.py 2>&1 | grep -v '^  ' > ../logs/figures.log
$PY fig_overview.py >> ../logs/figures.log
$PY forecast.py 2>&1 | grep -v '^  ' > ../logs/forecast.log
$PY forecast_bump.py > ../logs/forecast_bump.log
$PY results_table.py
$PY ablation_overlap_test.py
$PY depth_median_test.py
$PY derived_numbers.py
