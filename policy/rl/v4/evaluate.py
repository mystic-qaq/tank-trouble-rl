"""Independent teacher/previous-policy comparisons; no gradient updates here."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .common import TEACHERS, STYLES, atomic_json, canonical, digest, source_digest, BudgetExhausted
from .tasks import combat, skills
from .model import load_policy
from .rollout import run
from .data import read_records, training_identity
from .gates import SPLITS, mean, complete


def offline(model, directories, teacher, rules, device, deadline=float('inf'), cache=None):
    key = (tuple(str(p) for p in directories), teacher)
    if cache is not None and key in cache:
        batches = cache[key]
    else:
        records = [r for r in read_records(directories) if r['validation'] and r['teacher'] == teacher and r.get('source') != 'dagger']
        batches = []
        for r in records:
            if time.monotonic() > deadline:
                raise BudgetExhausted('Offline validation incomplete')
            array = np.load(r['path'], mmap_mode='r')
            # Fixed uniform samples, no event oversampling or class weighting in validation.
            rng = np.random.default_rng(int(canonical([r['seed'], r['file']])[:8], 16))
            indices = rng.choice(len(array), min(512, len(array)), replace=False)
            data = torch.from_numpy(np.array(array[indices], dtype=np.float32))
            batches.append((data[:, :1966], data[:, 1966:]))
        # Keep the ~1.5 GB validation set in RAM: it used to be re-read from
        # NFS at every one of the six/seventeen screening points.
        if cache is not None:
            cache[key] = batches
    total = correct = positive = predicted = tp = 0
    model.eval()
    style_index = STYLES.index(teacher)
    for x_cpu, target_cpu in batches:
        if time.monotonic() > deadline:
            raise BudgetExhausted('Offline validation incomplete')
        x, target = x_cpu.to(device), target_cpu.to(device)
        style = torch.full((len(x),), style_index, device=device, dtype=torch.long)
        with torch.inference_mode():
            _, _, _, controls = model.act(x, style, deterministic=True)
        total += len(x)
        correct += int(((controls[:, 0] == target[:, 0]) & ((controls[:, 1] - target[:, 1]).abs() <= rules['steer_tolerance'])).sum())
        actual, pred = target[:, 2] > .5, controls[:, 2] > .5
        positive += int(actual.sum())
        predicted += int(pred.sum())
        tp += int((actual & pred).sum())
    return dict(samples=total, motion_accuracy=correct / max(1, total), fire_precision=tp / max(1, predicted),
                fire_recall=tp / max(1, positive), fire_positives=positive, fire_predictions=predicted)


class Evaluator:
    def __init__(self, root, config_path, device, deadline):
        self.root = Path(root)
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text())
        self.device, self.deadline = device, deadline
        self.source = source_digest()
        self.offline_cache = {}
        (self.root / 'references').mkdir(parents=True, exist_ok=True)

    def episodes(self, tasks, model=None, teacher=None, objective='win', stochastic=False, label='eval', label_teacher=None):
        def progress(done, total):
            if done % 16 == 0 or done == total:
                atomic_json(self.root / 'evaluation-status.json', dict(stage=label, complete=done, total=total, status='running' if done < total else 'complete'))
        return run(tasks, model, teacher, style=self.config.get('policy_style', TEACHERS[objective]), workers=self.config['workers'], device=self.device,
                   objective=objective, stochastic=stochastic, deadline=self.deadline, on_result=progress, label_teacher=label_teacher)[0]

    def reference(self, tasks, teacher, objective):
        key = canonical(dict(tasks=tasks, teacher=teacher, objective=objective, source=self.source))
        path = self.root / 'references' / (key + '.json')
        if path.exists():
            cached = json.loads(path.read_text())
            if cached['key'] == key and complete(cached['rows'], tasks, skill=bool(tasks[0].get('skill'))):
                return cached['rows']
            raise ValueError('Reference cache contract failed')
        rows = self.episodes(tasks, teacher=teacher, objective=objective, label='teacher:' + teacher)
        atomic_json(path, dict(key=key, rows=rows))
        return rows

    def dev(self, model, objective):
        tasks = combat('development', self.config['suites']['dev_blocks'], self.config['seed'])
        teacher = self.reference(tasks, TEACHERS[objective], objective)
        modes = {mode: self.episodes(tasks, model, objective=objective, stochastic=mode == 'sampled', label='dev:' + objective + ':' + mode)
                 for mode in ['greedy', 'sampled']}
        metric = 'win' if objective == 'win' else 'netKills'
        return dict(teacher=mean(teacher, metric), greedy=mean(modes['greedy'], metric), sampled=mean(modes['sampled'], metric),
                    suicide_greedy=mean(modes['greedy'], 'suicides'), suicide_sampled=mean(modes['sampled'], 'suicides'),
                    teacher_suicides=mean(teacher, 'suicides'))

    def evidence(self, stage, checkpoint, directories, out, objective=None, previous=None, updates=None, data_receipt=None):
        c = self.config
        split = SPLITS[stage]
        tasks = combat(split, c['suites']['gate_blocks'], c['seed'])
        micro = skills(split, c['suites']['skill_cases'], c['seed'])
        model, _ = load_policy(checkpoint, self.device)
        model.eval()
        old = load_policy(previous, self.device)[0].eval() if previous else None
        evidence = dict(split=split, objective=objective, checkpoint_sha256=digest(checkpoint), config_sha256=digest(self.config_path),
                        source_sha256=self.source, temperature=model.temperature, tasks_sha256=canonical(tasks), branches={}, updates=updates or [],
                        previous_checkpoint_sha256=digest(previous) if previous else None,
                        training_data=training_identity(directories, data_receipt))
        for target in ['win', 'frag'] if stage == 'pretrain' else [objective]:
            branch = dict(teacher=self.reference(tasks, TEACHERS[target], target), skill_teacher=self.reference(micro, 'reference', target),
                          modes={}, skills={}, previous={})
            for mode in ['greedy', 'sampled']:
                branch['modes'][mode] = self.episodes(tasks, model, objective=target, stochastic=mode == 'sampled', label=stage + ':' + target + ':' + mode,
                                                     label_teacher=TEACHERS[target] if stage == 'dagger' else None)
                branch['skills'][mode] = self.episodes(micro, model, objective=target, stochastic=mode == 'sampled', label=stage + ':skills:' + target + ':' + mode)
                if old is not None:
                    branch['previous'][mode] = self.episodes(tasks, old, objective=target, stochastic=mode == 'sampled', label=stage + ':previous:' + target + ':' + mode)
            if stage in ['pretrain', 'dagger']:
                branch['offline'] = offline(model, directories, TEACHERS[target], c['offline'], self.device, self.deadline,
                                           cache=self.offline_cache)
            if stage == 'dagger':
                branch['student_state_imitation'] = {}
                for mode, rows in branch['modes'].items():
                    counts = {k: sum(r['label_audit'][k] for r in rows) for k in ['samples', 'motion_correct', 'fire_positives', 'fire_predictions', 'fire_tp']}
                    branch['student_state_imitation'][mode] = dict(motion_accuracy=counts['motion_correct'] / max(1, counts['samples']),
                        fire_precision=counts['fire_tp'] / max(1, counts['fire_predictions']), fire_recall=counts['fire_tp'] / max(1, counts['fire_positives']),
                        fire_positives=counts['fire_positives'], samples=counts['samples'])
            evidence['branches'][target] = branch
            atomic_json(out, evidence)
        return evidence
