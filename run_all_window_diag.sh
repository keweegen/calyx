#!/bin/sh
# Запустить все шарды диагностики зазора (V1c-E6.2) и дождаться их.
# Независимые процессы, а не multiprocessing.Pool: на этом venv Pool встаёт
# в дедлок (см. заголовок run_v1b_grid.py).
N=${1:-8}
i=0
while [ "$i" -lt "$N" ]; do
  ./run_shard_window.bat "$i" "$N" &
  i=$((i + 1))
done
wait
echo "все $N шардов диагностики зазора завершились"
