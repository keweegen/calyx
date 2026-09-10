#!/bin/sh
# Refinement of the E6.2 diagnostic resolution (rule Е2): two scales in parallel.
# The dichotomy is sequential within a scale, so there are no shards inside it.
mkdir -p results/diag_window/logs
./msvc_run.bat v1b_window_diag.py --refine 8 > results/diag_window/logs/refine_scale8.log 2>&1 &
./msvc_run.bat v1b_window_diag.py --refine 4 > results/diag_window/logs/refine_scale4.log 2>&1 &
wait
echo "refinement on both scales finished"
