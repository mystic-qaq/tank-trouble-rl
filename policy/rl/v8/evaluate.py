"""Engine evaluation for V8 under the frozen acceptance convention.

The frozen Evaluator (policy/rl/v4/evaluate.py) cannot drive V8: its rollout batches
observations from different episodes into one model call, while V8 carries per-episode
state (the 346-dim map and the per-opponent memory). This driver keeps that state where
it belongs, and otherwise reproduces the frozen protocol:

  tasks      combat('development', blocks, seed=20261107)
  aggregate  gates.mean -> n-balanced macro average over n=2..8
  modes      greedy and sampled; acceptance uses min(greedy, sampled)

Teacher reference: 0.301871 macro win (hy) / 0.123044 suicides, measured twice in
v7-runs/audit and identical to v6-runs/baseline-probe.json. Hardcode it so this script
does not need to re-run the teacher.

Usage:
  python -m policy.rl.v8.evaluate --checkpoint ... --blocks 2 --mode greedy
"""
import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..v4.common import STYLES, atomic_json
from ..v4.gates import mean as macro
from ..v4.tasks import combat
from .model import Policy, load_policy
from .transport import Worker

TEACHER_WIN = 0.30187074829931976
TEACHER_SUICIDES = 0.12304421768707482
ACCEPT_FRACTION = 0.7
SUICIDE_MARGIN = 0.03


def load_model(path, device):
    model, meta = load_policy(path, device)
    model.eval()
    return model, meta


def run_task(model, device, style_index, task, stochastic, deadline):
    worker = Worker()
    slot = 0
    try:
        header, data = worker.request(op='eval_open', slot=slot, config=task, seat=task['seat'])
        if not header.get('ok'):
            raise RuntimeError(header.get('error'))
        map_vec = torch.from_numpy(np.asarray(data[:346], np.float32)).unsqueeze(0).to(device)
        obs = torch.from_numpy(np.asarray(data[346:], np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            map_feat = model.encode_map(map_vec)
            memory = model.zero_memory(1, device)
            style = torch.full((1,), style_index, dtype=torch.long, device=device)
            frames = 0
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError('task budget')
                memory = model.advance_memory(obs, memory)
                controls, _, _, _ = model.act(obs, style, map_feat, memory,
                                              deterministic=not stochastic)
                action = dict(drive=int(controls[0, 0]), steer=float(controls[0, 1]),
                              shoot=bool(controls[0, 2] > 0.5))
                header, data = worker.request(op='eval_act', slot=slot, action=action)
                frames += 1
                if header['done']:
                    return header['info']
                if header.get('active', 1) == 0:
                    # dead seats stop observing but the world keeps running to a winner
                    pass
                obs = torch.from_numpy(np.asarray(data, np.float32)).unsqueeze(0).to(device)
    finally:
        worker.close()


def summarise(rows):
    ev = lambda k: float(np.mean([(r['events'] or {}).get(k, 0) for r in rows]))
    shots = max(ev('emissions'), 1e-9)
    return dict(episodes=len(rows),
                win_macro=macro(rows, 'win'),
                win_plain=float(np.mean([r['win'] for r in rows])),
                netKills_macro=macro(rows, 'netKills'),
                kills=float(np.mean([r['kills'] for r in rows])),
                suicides_macro=macro(rows, 'suicides'),
                timeout=float(np.mean([r['timeout'] for r in rows])),
                shots=shots, hit_per_shot=ev('hits') / shots,
                self_per_shot=ev('self_hits') / shots)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default='/home/xy/rl_for_fun/v8-runs/eval.json')
    p.add_argument('--blocks', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261107)
    p.add_argument('--style', default='hy')
    p.add_argument('--mode', choices=['greedy', 'sampled', 'both'], default='both')
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=int, default=3600)
    a = p.parse_args()

    torch.set_grad_enabled(False)
    model, meta = load_model(a.checkpoint, a.device)
    tasks = combat('development', a.blocks, a.seed)
    print('checkpoint %s  tasks %d  style %s' % (a.checkpoint, len(tasks), a.style), flush=True)
    style_index = STYLES.index(a.style)
    deadline = time.monotonic() + a.seconds
    result = dict(checkpoint=a.checkpoint, blocks=a.blocks, seed=a.seed,
                  style=a.style, episodes=len(tasks),
                  teacher_win=TEACHER_WIN, teacher_suicides=TEACHER_SUICIDES,
                  accept_threshold=ACCEPT_FRACTION * TEACHER_WIN,
                  suicide_ceiling=TEACHER_SUICIDES + SUICIDE_MARGIN, runs={})

    modes = ['greedy', 'sampled'] if a.mode == 'both' else [a.mode]
    for mode in modes:
        stochastic = mode == 'sampled'
        t0 = time.monotonic()
        rows = []
        with cf.ThreadPoolExecutor(a.workers) as pool:
            futs = {pool.submit(run_task, model, a.device, style_index, t, stochastic, deadline): t
                    for t in tasks}
            for f in cf.as_completed(futs):
                rows.append(f.result())
        stats = summarise(rows)
        stats['seconds'] = round(time.monotonic() - t0, 1)
        result['runs'][mode] = stats
        ok_win = stats['win_macro'] >= result['accept_threshold']
        ok_sui = stats['suicides_macro'] <= result['suicide_ceiling']
        stats['pass_win'] = ok_win
        stats['pass_suicide'] = ok_sui
        print('%-8s win macro %.4f (need %.4f) %s | suicides %.4f (max %.4f) %s | '
              'netK %+.3f  shots %.2f  hit/shot %.3f  %.0fs'
              % (mode, stats['win_macro'], result['accept_threshold'],
                 'PASS' if ok_win else 'FAIL', stats['suicides_macro'],
                 result['suicide_ceiling'], 'PASS' if ok_sui else 'FAIL',
                 stats['netKills_macro'], stats['shots'], stats['hit_per_shot'],
                 stats['seconds']), flush=True)

    if len(modes) == 2:
        student = min(result['runs']['greedy']['win_macro'], result['runs']['sampled']['win_macro'])
        result['student_win'] = student
        result['accept_pass'] = student >= result['accept_threshold'] and all(
            result['runs'][m]['pass_suicide'] for m in modes)
        print('student_win (min of both modes) %.4f -> %s'
              % (student, 'PASS' if result['accept_pass'] else 'FAIL'), flush=True)

    atomic_json(Path(a.out), result)
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
