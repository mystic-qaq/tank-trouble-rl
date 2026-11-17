"""Central batched inference, independent per-case sampling streams, native frames."""
import concurrent.futures as cf
import subprocess
import time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from ..v2.transport import Worker as BinaryWorker
from .common import ROOT, STYLES, BudgetExhausted
from .tasks import number


class Worker(BinaryWorker):
    def __init__(self):
        self.process = subprocess.Popen(['node', '--experimental-vm-modules', '--disable-warning=ExperimentalWarning',
                                         str(Path(__file__).with_name('worker.mjs'))], cwd=ROOT,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)


def run(tasks, model=None, teacher=None, *, style='hy', workers=32, device='cpu', stochastic=False,
        trajectory=False, label_teacher=None, objective='win', deadline=float('inf'), early_death=False, on_result=None):
    if (model is None) == (teacher is None):
        raise ValueError('Exactly one policy source required')
    processes = [Worker() for _ in range(min(workers, len(tasks)))]
    pending, results, records, rng, label_audits = {}, {}, {}, {}, {}
    ready = deque()
    todo = iter(enumerate(tasks))
    start = time.monotonic()
    pool = cf.ThreadPoolExecutor(max(1, len(processes)))

    def reset(w):
        try:
            i, task = next(todo)
        except StopIteration:
            return
        records[i] = []
        label_audits[i] = dict(samples=0, motion_correct=0, fire_positives=0, fire_predictions=0, fire_tp=0)
        rng[i] = np.random.default_rng(number(task, style, 'sampling'))
        pending[pool.submit(w.request, op='reset', slot=0, options={'objective': objective, 'early_death': early_death}, config=task)] = (w, i, None)

    try:
        for w in processes:
            reset(w)
        while pending or ready:
            if time.monotonic() >= deadline:
                raise BudgetExhausted('Budget ended during rollout; incomplete results cannot certify a gate')
            if pending:
                finished, _ = cf.wait(pending, timeout=.002 if ready else .2, return_when=cf.FIRST_COMPLETED)
                finished |= {f for f in pending if f.done()}
                for f in finished:
                    w, i, transition = pending.pop(f)
                    meta, obs = f.result()
                    if transition is not None and meta.get('active') and meta.get('label') is not None:
                        predicted = transition['controls']
                        label = meta['label']
                        audit = label_audits[i]
                        audit['samples'] += 1
                        audit['motion_correct'] += int(predicted[0] == label[0] and abs(predicted[1] - label[1]) <= .125)
                        audit['fire_positives'] += int(label[2] > .5)
                        audit['fire_predictions'] += int(predicted[2] > .5)
                        audit['fire_tp'] += int(label[2] > .5 and predicted[2] > .5)
                    if trajectory and transition is not None:
                        records[i].append(dict(**transition, reward=float(meta['reward']), active=bool(meta['active']),
                                               label=meta.get('label'),
                                               kills_delta=float(meta.get('kills_delta', 0.)),
                                               suicides_delta=float(meta.get('suicides_delta', 0.))))
                    if meta.get('done'):
                        results[i] = {**tasks[i], **meta['info'], 'label_audit': label_audits[i]}
                        if on_result:
                            on_result(len(results), len(tasks))
                        reset(w)
                    else:
                        ready.append((w, i, obs))
            if ready:
                batch = [ready.popleft() for _ in range(min(128, len(ready)))]
                if model is not None:
                    x = torch.from_numpy(np.stack([v[2] for v in batch])).to(device)
                    s = torch.full((len(batch),), STYLES.index(style), device=device, dtype=torch.long)
                    uniforms = torch.tensor([rng[i].random() for _, i, _ in batch], device=device, dtype=torch.float32)
                    with torch.inference_mode():
                        actions, lp, _, controls = model.act(x, s, deterministic=not stochastic, uniforms=uniforms)
                    actions, lp, controls = actions.cpu().numpy(), lp.cpu().numpy(), controls.cpu().numpy()
                for k, (w, i, obs) in enumerate(batch):
                    action = dict(drive=int(controls[k, 0]), steer=float(controls[k, 1]), shoot=bool(controls[k, 2])) if model is not None else dict(drive=0, steer=0, shoot=False)
                    transition = dict(obs=obs, action=int(actions[k]), logprob=float(lp[k]), controls=controls[k].tolist()) if trajectory and model is not None else None
                    if not trajectory and label_teacher and model is not None:
                        transition = dict(controls=controls[k].tolist())
                    selected_teacher = ('reference' if tasks[i].get('skill') else teacher) if teacher else None
                    selected_label = ('reference' if tasks[i].get('skill') else label_teacher) if label_teacher else None
                    pending[pool.submit(w.request, op='step', slot=0, action=action, teacher=selected_teacher, label_teacher=selected_label)] = (w, i, transition)
        return [results[i] for i in range(len(tasks))], records, time.monotonic() - start
    finally:
        # Kill native workers before waiting on blocked binary reads after an exception.
        for w in processes:
            w.close()
        pool.shutdown(wait=True, cancel_futures=True)


def monte_carlo(records):
    active = []
    for episode in records.values():
        total = 0.
        backward = []
        for row in reversed(episode):
            total += row['reward']
            if row['active']:
                backward.append({**row, 'return': total})
        active.extend(reversed(backward))
    if not active:
        raise ValueError('No active actions')
    return active
