#!/bin/bash
# Regenerate every stored reference report (CPU). Several hours on a busy 96-core host.
cd "$(dirname "$0")/.."
PY=${PY:-../.venv/bin/python}
$PY tools/run_reference_filters.py ekf_gated 12 > logs/ref_ekf_gated.log 2>&1 &
$PY tools/run_reference_filters.py ekf_huber 12 > logs/ref_ekf_huber.log 2>&1 &
wait
$PY tools/run_reference_filters.py ekf_gated_tuned 12 dev,test,deployment > logs/ref_ekf_gated_tuned.log 2>&1 &
$PY tools/run_reference_filters.py ude_ekf_gated 12 dev,test,deployment > logs/ref_ude_ekf_gated.log 2>&1 &
wait
$PY tools/summarize_reference.py ekf_gated
$PY tools/summarize_reference.py ekf_huber
$PY tools/build_leaderboard.py
