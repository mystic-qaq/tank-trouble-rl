#!/usr/bin/env bash
set -euo pipefail
task_root="${RL_ROOT:-$RL_ROOT}"
if ! tmux has-session -t tank-rl-events 2>/dev/null; then
  tmux new-session -d -s tank-rl-events -c "$task_root" \
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '$task_root/.venv/bin/python' -u policy/rl/live_tensorboard.py --root '$task_root' > '$task_root/runs/initial60/events.log' 2>&1"
fi
if ! tmux has-session -t tank-rl-tensorboard 2>/dev/null; then
  tmux new-session -d -s tank-rl-tensorboard -c "$task_root" \
    "'$task_root/.venv/bin/tensorboard' --logdir '$task_root/runs/initial60/tensorboard' --host 127.0.0.1 --port 6006 --reload_interval 5 > '$task_root/runs/initial60/tensorboard.log' 2>&1"
fi
tmux list-sessions -F '#{session_name}'
