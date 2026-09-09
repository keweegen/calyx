#!/bin/sh
# Уточнение разрешения диагностики E6.2 (правило Е2): два масштаба параллельно.
# Дихотомия последовательна внутри масштаба, поэтому шардов внутри нет.
mkdir -p results/diag_window/logs
./msvc_run.bat v1b_window_diag.py --refine 8 > results/diag_window/logs/refine_scale8.log 2>&1 &
./msvc_run.bat v1b_window_diag.py --refine 4 > results/diag_window/logs/refine_scale4.log 2>&1 &
wait
echo "уточнение по обоим масштабам завершено"
