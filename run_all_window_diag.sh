#!/bin/sh
# Launch all shards of the gap diagnostic (V1c-E6.2) and wait for them.
# Independent processes, not multiprocessing.Pool: on this venv, Pool
# deadlocks (see the header of run_v1b_grid.py).
N=${1:-8}
i=0
while [ "$i" -lt "$N" ]; do
  ./run_shard_window.bat "$i" "$N" &
  i=$((i + 1))
done
wait
echo "all $N shards of the gap diagnostic finished"
