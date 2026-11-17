#!/usr/bin/env bash
set -euo pipefail
task_root="${RL_ROOT:-$RL_ROOT}"
release="$task_root/releases/20261106-60s-v1"
test -f "$release/source-manifest.json"
test -x "$task_root/.venv/bin/python"
for session in tank-rl-win60 tank-rl-frag60; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Session already exists: $session. Inspect it before starting a duplicate experiment."
    exit 1
  fi
done
mkdir -p "$task_root/runs/initial60"
tmux new-session -d -s tank-rl-win60 -c "$release" \
  "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 '$task_root/.venv/bin/python' -u policy/rl/run_experiment.py --objective win --out '$task_root/runs/initial60/win' --samples 65536 --steps 1048576 --envs 16 --seed 20261106 > '$task_root/runs/initial60/win.log' 2>&1"
tmux new-session -d -s tank-rl-frag60 -c "$release" \
  "CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 '$task_root/.venv/bin/python' -u policy/rl/run_experiment.py --objective frag --out '$task_root/runs/initial60/frag' --samples 65536 --steps 1048576 --envs 16 --seed 20261107 > '$task_root/runs/initial60/frag.log' 2>&1"
tmux list-sessions -F '#{session_name}'
