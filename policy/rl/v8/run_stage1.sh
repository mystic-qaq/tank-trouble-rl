#!/usr/bin/env bash
# Stage 1 pipeline: wait for the corpus to finish, then phase A then phase B.
set -u
R=$RL_ROOT/releases/20261110-v8
RUNDIR=$RL_ROOT/v8-runs
DATA=$RL_ROOT/data-v3/multi
PY=$RL_ROOT/.venv/bin/python
export TMPDIR=$RL_ROOT/tmp XDG_CACHE_HOME=$RL_ROOT/cache CUDA_VISIBLE_DEVICES=1

cd "$R" || exit 1
mkdir -p "$RUNDIR"

echo "=== $(date +%T) waiting for collection to finish"
for i in $(seq 1 480); do
  status=$($PY -c "import json;print(json.load(open('$DATA/status.json')).get('status',''))" 2>/dev/null)
  if [ "$status" = "complete" ] || [ "$status" = "budget_exhausted" ]; then
    echo "=== $(date +%T) collection $status"
    break
  fi
  sleep 15
done
$PY -c "import json;d=json.load(open('$DATA/status.json'));print(json.dumps({k:d[k] for k in ['status','worlds','trajectories','recorded_rows','gb','by_teacher']}))"

echo "=== $(date +%T) phase A"
$PY -m policy.rl.v8.train --data "$DATA" --out "$RUNDIR/s1a" --phase a \
  --steps 20000 --batch 1024 --seq 32 --lr 2e-4 --eval-every 500 \
  --device cuda:0 --seconds 14400 2>&1
echo "=== $(date +%T) phase A exit $?"

echo "=== $(date +%T) phase B"
$PY -m policy.rl.v8.train --data "$DATA" --out "$RUNDIR/s1b" --phase b \
  --steps 4000 --batch 4096 --seq 32 --lr 3e-5 --eval-every 500 \
  --init "$RUNDIR/s1a/final.pt" --device cuda:0 --seconds 7200 2>&1
echo "=== $(date +%T) phase B exit $?"

echo "=== $(date +%T) STAGE1 DONE"
