#!/usr/bin/env bash
# Sequential job queue: pops the first line of queue.txt, runs it, logs to queue_done.log.
# $PY = JAX/GPU env (karma_jax), $PYCPU = pinned reference env. Stop with: touch optimize/queue.STOP
cd "$(dirname "$0")/.." || exit 1
export PY=/home/chang940/mamba/envs/karma-jax/bin/python
export PYCPU=/home/chang940/mamba/envs/karma-hand-metric/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
Q=optimize/queue.txt
while true; do
  [ -f optimize/queue.STOP ] && { echo "$(date -Is) STOP file present, exiting" >> optimize/queue_done.log; exit 0; }
  job=$(grep -m1 -v '^\s*$' "$Q" 2>/dev/null)
  if [ -z "$job" ]; then sleep 60; continue; fi
  awk -v j="$job" 'BEGIN{d=0} {if(!d && $0==j){d=1; next} print}' "$Q" > "$Q.tmp"; mv "$Q.tmp" "$Q"
  echo "$(date -Is) START $job" >> optimize/queue_done.log
  t0=$(date +%s)
  bash -c "$job" >> optimize/queue_jobs.log 2>&1
  rc=$?
  echo "$(date -Is) END rc=$rc ($(( $(date +%s) - t0 ))s) $job" >> optimize/queue_done.log
done
