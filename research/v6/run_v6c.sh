#!/bin/bash
# V6C: extended SHARED PPO on the WIN objective with dense kill/survival
# shaping, gated by a measured development win rate before the two objective
# forks are released. A frag-only shared phase learned to turtle (dev win
# 0.0536 < ~0.20 random baseline) because avoiding suicides is far easier
# than learning to kill; win + kill_aux shaping aligns the training signal
# with the measured objective.
#
# Design constraints inherited from the audit discipline:
#  * never overwrite an existing run directory;
#  * every segment ends normally and writes result.json (no forced kill, no
#    truncated rollout used for an update);
#  * a gate that is not met continues training, it is NEVER relaxed;
#  * 1 GPU for shared, 2 GPUs for the forks.
set -u
R=$RL_ROOT/releases/20261107-v5-2gpu
V=$RL_ROOT/v6-runs
PY=$RL_ROOT/.venv/bin/python
REC=$V/receipt2.json
DATA="$RL_ROOT/data-local/multi $RL_ROOT/data-local/skills"
PROBE_CACHE=$V/probe-cache
FIRST_SEG_SECONDS=5400     # 1.5 h first segment: get an early win-rate trend
SEG_SECONDS=10800          # 3 h for later segments
MAX_SEGMENTS=4             # 10.5 h safety valve, then stop for an operator decision
FORK_SECONDS=7200          # 5400 s training + 1800 s independent dev acceptance
ACCEPT_SECONDS=1800
cd $R

log() { echo "[$(date '+%F %T')] $*" | tee -a $V/pipeline3.log; }

log "V6C waiting for shared2/result.json + shared-gate.json (V6B fork2 is suppressed)"
for i in $(seq 1 720); do
  [ -f $V/shared2/result.json ] && [ -f $V/shared2/shared-gate.json ] && break
  # V6B writes the handoff report a few seconds AFTER result.json. If its shell
  # is gone, run the very same gate here rather than aborting the whole chain.
  if [ -f $V/shared2/result.json ] && [ ! -f $V/shared2/shared-gate.json ] && [ $((i % 6)) -eq 0 ]; then
    log "handoff report not written by V6B yet; running the identical gate locally"
    (cd $R && $PY $V/gate_shared.py $V/shared2) >> $V/pipeline3.log 2>&1 || true
  fi
  sleep 10
done
if [ ! -f $V/shared2/result.json ]; then log "ABORT: shared2 never produced result.json"; exit 1; fi
if [ ! -f $V/shared2/shared-gate.json ]; then log "ABORT: handoff gate report missing"; exit 1; fi
$PY -c "import json,sys; sys.exit(0 if json.load(open('$V/shared2/shared-gate.json'))['accepted'] else 1)" \
  || { log "ABORT: shared2 quality gate failed"; exit 1; }
log "shared2 handoff accepted; best.pt is the audited init"

INIT=$V/shared2/best.pt
SEG=0
while [ $SEG -lt $MAX_SEGMENTS ]; do
  SEG=$((SEG+1))
  OUT=$V/shared3-s$SEG
  CFG=$V/shared3-s$SEG-config.json
  # A fresh config seed per segment gives new maps / spawns / rosters.
  # The PROBE config seed stays fixed, so dev win rates remain comparable.
  $PY - "$V/shared3-base-config.json" "$CFG" "$SEG" <<'PY'
import json, pathlib, sys
src, dst, seg = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
c = json.loads(src.read_text())
c['seed'] = c['seed'] + seg * 101
dst.write_text(json.dumps(c, indent=2))
PY
  if [ -d $OUT ]; then log "ABORT: $OUT exists (never overwrite an audited run)"; exit 1; fi
  if [ $SEG -eq 1 ]; then BUDGET=$FIRST_SEG_SECONDS; else BUDGET=$SEG_SECONDS; fi
  log "=== shared3 segment $SEG/$MAX_SEGMENTS: 1 GPU / ${BUDGET}s; objective=win; init=$INIT ==="
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY -m policy.rl.v4.optimize \
      --config $CFG --data $DATA --data-receipt $REC --out $OUT \
      --objective win --device cuda:0 --seconds $BUDGET \
      --skip-acceptance --phase shared --init $INIT > $V/shared3-s$SEG.log 2>&1
  E=$?
  if [ $E -ne 0 ]; then
    log "ABORT: shared3-s$SEG exited $E"
    tail -25 $V/shared3-s$SEG.log >> $V/pipeline3.log
    exit 1
  fi
  log "shared3-s$SEG finished; probing frozen development win rate"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY $V/probe_win.py \
      $OUT/best.pt $PROBE_CACHE $V/probe-config.json $OUT/probe.json \
      > $V/probe-s$SEG.log 2>&1
  E=$?
  if [ $E -ne 0 ]; then
    log "ABORT: win-rate probe failed ($E)"
    tail -25 $V/probe-s$SEG.log >> $V/pipeline3.log
    exit 1
  fi
  INIT=$OUT/best.pt
  if $PY $V/gate_fork.py $OUT/probe.json $OUT/result.json $OUT/fork-gate.json >> $V/pipeline3.log 2>&1; then
    log "FORK GATE PASSED at segment $SEG; forking from $INIT"
    break
  fi
  log "fork gate not met at segment $SEG; continuing shared PPO"
done

if [ ! -f $OUT/fork-gate.json ] || \
   ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/fork-gate.json'))['accepted'] else 1)"; then
  log "STOP: $MAX_SEGMENTS segments without a measured win rate. Nothing forked; no gate relaxed."
  $PY -c "import json,pathlib; pathlib.Path('$V/v6c-needs-decision.json').write_text(json.dumps(dict(status='awaiting_operator_decision', segments=$MAX_SEGMENTS, last_segment=str('$OUT')), indent=2))"
  exit 2
fi

log "=== fork3: 2 GPUs / ${FORK_SECONDS}s each incl. ${ACCEPT_SECONDS}s dev ==="
for OBJ in win frag; do
  if [ "$OBJ" = "win" ]; then GPU=1; else GPU=2; fi
  CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=1 $PY -m policy.rl.v4.optimize \
      --config $V/fork3-config.json --data $DATA --data-receipt $REC \
      --init $INIT --out $V/fork3-$OBJ --objective $OBJ --device cuda:0 \
      --seconds $FORK_SECONDS --acceptance-seconds $ACCEPT_SECONDS --phase fork \
      > $V/fork3-$OBJ.log 2>&1 &
  eval "P_$OBJ=$!"
done
wait $P_win; E1=$?
wait $P_frag; E2=$?
log "fork3_win_exit=$E1 fork3_frag_exit=$E2"
tail -3 $V/fork3-win.log >> $V/pipeline3.log
tail -3 $V/fork3-frag.log >> $V/pipeline3.log
log "V6C_PIPELINE_DONE"
