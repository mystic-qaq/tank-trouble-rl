"""Reuse raw demonstrations, never old weights; audit full files before training."""
import json
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch
from .common import STYLES, SKILLS, canonical, digest, atomic_json, GateFailure, BudgetExhausted


def validation(seed):
    import hashlib
    return int.from_bytes(hashlib.blake2b(str(seed).encode(), digest_size=8).digest(), 'little') % 10 == 0


def read_records(directories):
    records = []
    for directory in directories:
        directory = Path(directory).resolve()
        for line in (directory / 'trajectories.jsonl').read_text().splitlines():
            r = json.loads(line)
            path = (directory / r['file']).resolve()
            if path.parent != directory:
                raise GateFailure('Data path escapes its directory')
            records.append({**r, 'path': str(path)})
    return records


def audit(directories, rules, out, deadline=float('inf'), contracts=None):
    import time
    records = read_records(directories)
    counts, fires, files, seen = Counter(), Counter(), {}, set()
    errors = []
    for r in records:
        if time.monotonic() > deadline:
            raise BudgetExhausted('Data audit incomplete')
        p = Path(r['path'])
        if r['path'] in seen:
            raise GateFailure('Duplicate trajectory path')
        seen.add(r['path'])
        if r['teacher'] not in STYLES or bool(r['validation']) != validation(r['seed']) or r.get('physics_hz') != 60:
            raise GateFailure('Teacher/split/physics contract mismatch')
        matrix = np.load(p, mmap_mode='r', allow_pickle=False)
        if matrix.dtype != np.float32 or matrix.shape != (r['length'], 1969) or not len(matrix):
            raise GateFailure('Unexpected raw trajectory schema')
        positives = 0
        fire_rows = []
        for offset in range(0, len(matrix), 4096):
            x = matrix[offset:offset + 4096]
            if not np.isfinite(x).all() or not np.isin(x[:, -3], [-1, 0, 1]).all() or (np.abs(x[:, -2]) > 1).any() or not np.isin(x[:, -1], [0, 1]).all():
                raise GateFailure('Invalid state/action values')
            if ((x[:, -1] > .5) & (x[:, 16] < .5)).any():
                raise GateFailure('Emission label at an unavailable fire boundary')
            positives += int(x[:, -1].sum())
            fire_rows.extend((np.flatnonzero(x[:, -1] > .5) + offset).tolist())
        key = ('val' if r['validation'] else 'train') + ':' + r['teacher']
        counts[key] += 1
        fires[key] += positives
        if r.get('source') == 'skill':
            counts['skill:' + r['skill']] += 1
        stat = p.stat()
        files[str(p)] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': digest(p), 'fire_rows': fire_rows}
    for teacher in STYLES[:-1]:
        for split, threshold in [('train', rules['min_train_trajectories_per_teacher']), ('val', rules['min_validation_trajectories_per_teacher'])]:
            if counts[f'{split}:{teacher}'] < threshold:
                errors.append(f'{split}:{teacher}: insufficient independent trajectories')
        if fires[f'train:{teacher}'] < rules['min_train_fires_per_teacher']:
            errors.append(teacher + ': insufficient real fire labels')
    for skill in SKILLS:
        if counts['skill:' + skill] < rules['min_skill_trajectories']:
            errors.append('skill:' + skill + ': insufficient trajectories')
    # The legacy raw collector explicitly verified selected worlds by replaying native physics.
    for directory in directories:
        p = Path(directory)
        manifest = json.loads((p / 'manifest.json').read_text())
        if not manifest.get('continuous') or manifest.get('status') != 'complete':
            errors.append(str(p) + ': collection incomplete or not continuous')
        worlds = p / 'worlds.jsonl'
        if not worlds.exists() or not all(json.loads(line).get('replay_identical') is True for line in worlds.read_text().splitlines()):
            errors.append(str(p) + ': missing native replay evidence')
        for q in [p / 'manifest.json', p / 'trajectories.jsonl', worlds]:
            if q.exists():
                stat = q.stat()
                files[str(q.resolve())] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': digest(q)}
    if contracts is None:
        raise GateFailure('Engine/model contract evidence required before data acceptance')
    contract = json.loads(Path(contracts).read_text())
    from .common import source_digest
    if not contract.get('passed') or contract.get('source_sha256') != source_digest():
        raise GateFailure('Missing/stale engine and model contract checks')
    report = dict(kind='data', passed=not errors, errors=errors, rules=rules, counts=dict(counts), fire_labels=dict(fires), files=files,
                  contracts=str(Path(contracts).resolve()), contracts_sha256=digest(contracts),
                  directories=[str(Path(p).resolve()) for p in directories], schema='raw1969_current_fields_selected_in_model')
    report['receipt_sha256'] = canonical(report)
    atomic_json(out, report)
    if errors:
        raise GateFailure('G0 data audit failed: ' + '; '.join(errors))
    return report


def verify_data(path):
    report = json.loads(Path(path).read_text())
    expected = report.pop('receipt_sha256')
    if not report.get('passed') or canonical(report) != expected:
        raise GateFailure('Invalid data receipt')
    from .common import source_digest
    contract = json.loads(Path(report['contracts']).read_text())
    if digest(report['contracts']) != report['contracts_sha256'] or not contract.get('passed') or contract.get('source_sha256') != source_digest():
        raise GateFailure('Contract checks are missing or stale')
    for name, metadata in report['files'].items():
        stat = Path(name).stat()
        if stat.st_size != metadata['size'] or stat.st_mtime_ns != metadata['mtime_ns']:
            raise GateFailure('Audited data changed: ' + name)
    return report


class Frames:
    """Trajectory-balanced sampling with class-balanced fire-positive oversampling.

    A classification loss must NOT be re-weighted by the inverse proposal: at a
    0.6% natural fire rate that restores the imbalance and removes the fire class
    from the objective (199-1000x suppression). Unbiased fire-rate estimates come
    from offline()'s uniform validation sampling, never from a training weight.
    Quality/style/count weights define the intended trajectory distribution.
    """
    def __init__(self, directories, seed=1, fire_index=None, *,
                 fire_oversample=.15, fire_weight=1., style_boost=2.):
        self.records = [r for r in read_records(directories) if not r['validation']]
        if not self.records:
            raise GateFailure('No training data')
        self.rng = np.random.default_rng(seed)
        self.cache = OrderedDict()
        self.fire_index = fire_index or {}
        self.fire_oversample = float(fire_oversample)
        self.fire_weight = float(fire_weight)
        if not 0 <= self.fire_oversample < 1 or self.fire_weight <= 0:
            raise GateFailure('Invalid fire sampling parameters')
        sources = Counter(r.get('source', 'multi_teacher') for r in self.records)
        strata = Counter((r.get('source', 'multi_teacher'), r['teacher'], r['n']) for r in self.records)
        masses = {'multi_teacher': .3, 'skill': .2, 'dagger': .5} if 'dagger' in sources else {'multi_teacher': .8, 'skill': .2}
        weights = np.array([masses.get(r.get('source', 'multi_teacher'), .1) * max(.05, float(r.get('quality', 1))) /
                            strata[(r.get('source', 'multi_teacher'), r['teacher'], r['n'])] for r in self.records])
        # Normalize source mass separately, independent of how many teacher/count strata it has.
        for source in sources:
            mask = np.array([r.get('source', 'multi_teacher') == source for r in self.records])
            weights[mask] *= masses.get(source, .1) / weights[mask].sum()
        # Both deployable objectives must see enough of their own teacher style.
        if style_boost != 1.:
            weights *= np.array([style_boost if r['teacher'] in ('hy', 'gpt') else 1. for r in self.records])
        self.probabilities = weights / weights.sum()

    def array(self, index):
        if index not in self.cache:
            matrix = np.load(self.records[index]['path'], mmap_mode='r', allow_pickle=False)
            record = self.records[index]
            indexed = record.get('fire_rows', self.fire_index.get(record['path'], {}).get('fire_rows'))
            positives = np.asarray(indexed, dtype=np.int64) if indexed is not None else np.flatnonzero(matrix[:, -1] > .5)
            self.cache[index] = matrix, positives
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def batch(self, count, device):
        # Draw the same trajectory marginal, but reuse each open trajectory for
        # several independent frame draws. This avoids hundreds of file opens
        # per step without changing the intended BC objective.
        groups = min(32, max(1, count // 16))
        chosen = self.rng.choice(len(self.records), groups, p=self.probabilities)
        rows, styles, importance = [], [], []
        for group, index in enumerate(chosen):
            r = self.records[int(index)]
            matrix, positives = self.array(int(index))
            length = len(matrix)
            size = count // groups + int(group < count % groups)
            indices = self.rng.integers(length, size=size)
            if len(positives) and self.fire_oversample:
                selected = self.rng.random(size) < self.fire_oversample
                if selected.any():
                    indices[selected] = self.rng.choice(positives, int(selected.sum()))
            part = np.array(matrix[indices], copy=True)
            rows.append(part)
            # Teach microtasks to both deployable styles as well as the reference style.
            selected_styles = self.rng.choice(['hy', 'gpt', 'reference'], size=size) if r['teacher'] == 'reference' else [r['teacher']] * size
            styles.extend(STYLES.index(style) for style in selected_styles)
            # A bounded, normalized class weight; never an inverse-proposal correction.
            importance.extend(np.where(part[:, -1] > .5, self.fire_weight, 1.).astype(np.float32))
        data = torch.from_numpy(np.concatenate(rows)).to(device)
        weights = torch.tensor(importance, dtype=torch.float32, device=device)
        weights = weights / weights.mean().clamp_min(1e-8)
        return data[:, :1966], torch.tensor(styles, device=device), data[:, 1966:], weights


def save_dagger(directory, tasks, records, teacher):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    saved = []
    for i, task in enumerate(tasks):
        rows = [r for r in records[i] if r['active'] and r['label'] is not None]
        if not rows:
            continue
        matrix = np.concatenate([np.stack([r['obs'] for r in rows]), np.array([r['label'] for r in rows], dtype=np.float32)], 1)
        name = f'case-{i:05d}.npy'
        np.save(directory / name, matrix, allow_pickle=False)
        saved.append(dict(file=name, length=len(matrix), seed=task['seed'], n=task['n'], physics_hz=60,
                          source='dagger', quality=1., teacher='reference' if task.get('skill') else teacher,
                          skill=task.get('skill', 'combat'), validation=validation(task['seed']), fire_rows=np.flatnonzero(matrix[:, -1] > .5).tolist()))
    (directory / 'trajectories.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in saved))
    if not saved:
        raise GateFailure('DAgger produced no labeled states')
    metadata = {}
    for path in [directory / r['file'] for r in saved] + [directory / 'trajectories.jsonl']:
        stat = path.stat()
        metadata[str(path.resolve())] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path))
    atomic_json(directory / 'generated-data.json', dict(files=metadata, kind='dagger', labeling='full_teacher_motion_and_legal_emission'))
    return directory


def training_identity(directories, original_receipt):
    """Bind generated labels as well as the already fully audited original data."""
    original = verify_data(original_receipt)
    roots = set(original['directories'])
    identity = {}
    for directory in directories:
        directory = Path(directory).resolve()
        if str(directory) in roots:
            identity[str(directory)] = {'original_data_receipt_sha256': digest(original_receipt)}
            continue
        path = directory / 'generated-data.json'
        report = json.loads(path.read_text())
        for name, meta in report['files'].items():
            stat = Path(name).stat()
            if stat.st_size != meta['size'] or stat.st_mtime_ns != meta['mtime_ns']:
                raise GateFailure('Generated training data changed')
        identity[str(directory)] = {'generated_manifest_sha256': digest(path)}
    if not roots.issubset(identity):
        raise GateFailure('Original audited datasets omitted')
    return identity
