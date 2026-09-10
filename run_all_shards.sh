#!/bin/sh
# Launch all eight shards of the stage in parallel and wait for them.
# Eight independent processes, not multiprocessing.Pool: on this venv, Pool
# deadlocks, workers hang at 0% CPU (see the header of run_v1b_grid.py).
STAGE=${1:-1}
N=${2:-8}
i=0
while [ "$i" -lt "$N" ]; do
  ./run_shard.bat "$i" "$N" "$STAGE" &
  i=$((i + 1))
done
wait
echo "all $N shards of stage $STAGE finished"
