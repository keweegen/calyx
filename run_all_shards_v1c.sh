#!/bin/sh
# Launch all eight shards of the V1c stage in parallel and wait for them.
# Eight independent processes, not multiprocessing.Pool: on this venv, Pool
# deadlocks, workers hang at 0% CPU.
N=${1:-8}
i=0
while [ "$i" -lt "$N" ]; do
  ./run_shard_v1c.bat "$i" "$N" &
  i=$((i + 1))
done
wait
echo "all $N shards of the V1c stage finished"
