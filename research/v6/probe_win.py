"""Independent win-rate probe on the frozen development split.

Lives in v6-runs/ (outside the source_digest roots) so a win rate can be
measured without changing the V6 engine/model contracts or invalidating G0.
Uses the same Evaluator.dev() and the same config seed/dev blocks as the final
acceptance, so numbers stay comparable across shared segments and with the
eventual gate. Teacher references are cached in a shared directory.

Usage: probe_win.py CHECKPOINT CACHE_ROOT PROBE_CONFIG RESULT_JSON
"""
import json
import os
import pathlib
import sys
import time

# A script launched as `python v6-runs/probe_win.py` has v6-runs/ as
# sys.path[0], not the release root. Bind the audited release explicitly so
# this probe works regardless of the caller's cwd. RL_ROOT overrides it.
_RELEASE = os.environ.get('RL_ROOT', '$RL_ROOT/releases/20261107-v5-2gpu')
if not (pathlib.Path(_RELEASE) / 'policy' / 'rl' / 'v4' / 'model.py').is_file():
    raise RuntimeError('Audited release root not found: ' + _RELEASE)
sys.path.insert(0, _RELEASE)

import torch

from policy.rl.v4.common import atomic_json
from policy.rl.v4.evaluate import Evaluator
from policy.rl.v4.model import load_policy

checkpoint = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
config_path = pathlib.Path(sys.argv[3]).resolve()
result_path = pathlib.Path(sys.argv[4]).resolve()
root.mkdir(parents=True, exist_ok=True)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model, meta = load_policy(checkpoint, device)
model.eval()
torch.set_num_threads(1)
evaluator = Evaluator(root, config_path, device, deadline=time.monotonic() + 2700)
t0 = time.monotonic()
dev = evaluator.dev(model, 'win')
report = dict(
    checkpoint=str(checkpoint), checkpoint_stage=meta.get('stage'),
    checkpoint_iteration=meta.get('iteration'), checkpoint_score=meta.get('score'),
    teacher_win=dev['teacher'], greedy_win=dev['greedy'], sampled_win=dev['sampled'],
    student_win=min(dev['greedy'], dev['sampled']),
    suicide_greedy=dev['suicide_greedy'], suicide_sampled=dev['suicide_sampled'],
    teacher_suicides=dev['teacher_suicides'],
    seconds=time.monotonic() - t0,
    note='frozen development split, same Evaluator.dev() as final acceptance')
atomic_json(result_path, report)
print('PROBE_WIN teacher=%.4f greedy=%.4f sampled=%.4f student=%.4f '
      'suicides=%.3f/%.3f seconds=%.0f' %
      (report['teacher_win'], report['greedy_win'], report['sampled_win'],
       report['student_win'], report['suicide_greedy'], report['suicide_sampled'],
       report['seconds']), flush=True)
