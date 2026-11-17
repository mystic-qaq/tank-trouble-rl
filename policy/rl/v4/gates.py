"""Promotion is an evidence decision, never a step/time decision.

All comparisons pair complete seat/map blocks and weight player counts equally.
Engineering acceptance margins are predeclared in config.json, not tuned to results.
"""
import json
from pathlib import Path
import numpy as np
from .common import GateFailure, atomic_json, canonical, digest, source_digest, TEACHERS, SKILLS
from .tasks import combat, skills
from .data import verify_data, training_identity

SPLITS = {'pretrain': 'promotion-bc', 'dagger': 'promotion-dagger', 'pilot': 'promotion-pilot', 'final': 'final-locked'}


def mean(rows, metric):
    return float(np.mean([np.mean([r[metric] for r in rows if r['n'] == n]) for n in range(2, 9)]))


def interval(student, teacher, metric, fraction=1., samples=4000):
    if len(student) != len(teacher):
        raise GateFailure('Unpaired result lengths')
    groups = {}
    for s, t in zip(student, teacher):
        if any(s[k] != t[k] for k in ['seed', 'spawnSeed', 'mapSeed', 'seat', 'n', 'players', 'block', 'case']):
            raise GateFailure('Unpaired scenario')
        groups.setdefault(s['n'], {}).setdefault(s['block'], []).append(float(s[metric] - fraction * t[metric]))
    if set(groups) != set(range(2, 9)):
        raise GateFailure('Missing player-count coverage')
    rng = np.random.default_rng(11072026)
    draws = []
    for blocks in groups.values():
        values = np.array([np.mean(v) for v in blocks.values()])
        draws.append(values[rng.integers(len(values), size=(samples, len(values)))].mean(1))
    low, high = np.quantile(np.mean(draws, axis=0), [.025, .975])
    return [float(low), float(high)]


def complete(rows, expected, skill=False):
    if len(rows) != len(expected):
        return False
    for row, task in zip(rows, expected):
        if any(row.get(k) != v for k, v in task.items()):
            return False
        if row.get('errors') != 0 or not 0 < row.get('frames', 0) <= 3602 or not 0 <= row.get('duration', -1) <= 60.02:
            return False
        if not skill and (not row.get('round_complete') or row.get('opponents') != task['players']):
            return False
        if not np.isfinite([row.get(k, np.nan) for k in ['win', 'netKills', 'suicides', 'active_frames', 'spin_frames', 'stuck_frames']]).all():
            return False
        if row.get('netKills') != row.get('kills', 0) - row.get('suicides'):
            return False
    return True


def evaluate_gate(stage, evidence, config):
    if stage not in SPLITS:
        raise GateFailure('Unknown promotion stage')
    checks, metrics = {}, {}
    split = SPLITS[stage]
    expected = combat(split, config['suites']['gate_blocks'], config['seed'])
    expected_skills = skills(split, config['suites']['skill_cases'], config['seed'])
    checks['fresh_complete_suite'] = evidence.get('split') == split and evidence.get('tasks_sha256') == canonical(expected)
    checks['fixed_sampling_temperature'] = evidence.get('temperature') == config['temperature']
    checks['minimum_independent_blocks'] = config['suites']['gate_blocks'] >= 32 and config['suites']['skill_cases'] >= 64
    objectives = ['win', 'frag'] if stage == 'pretrain' else [evidence.get('objective')]
    if any(o not in TEACHERS for o in objectives):
        raise GateFailure('Missing objective')
    g = config['gates']
    for objective in objectives:
        branch = evidence.get('branches', {}).get(objective, {})
        reference = branch.get('teacher', [])
        modes = branch.get('modes', {})
        if not complete(reference, expected) or not all(complete(modes.get(mode, []), expected) for mode in ['greedy', 'sampled']):
            checks[objective + ':complete_combat'] = False
            continue
        checks[objective + ':complete_combat'] = True
        metric = 'win' if objective == 'win' else 'netKills'
        teacher_mean = mean(reference, metric)
        checks[objective + ':meaningful_teacher_baseline'] = teacher_mean >= g['min_teacher_win' if objective == 'win' else 'min_teacher_net']
        fraction = g['pretrain_teacher_fraction'] if stage == 'pretrain' else g['dagger_teacher_fraction']
        if stage == 'final':
            fraction = 1.
        margin = g['win_noninferiority_margin' if objective == 'win' else 'frag_noninferiority_margin']
        for mode, student in modes.items():
            if mode not in ['greedy', 'sampled']:
                continue
            prefix = objective + ':' + mode
            ci = interval(student, reference, metric, fraction, config['suites']['bootstrap_samples'])
            # A zero boundary is acceptable for noninferiority; final superiority must be strict.
            checks[prefix + ':teacher_performance'] = ci[0] > 0 if stage == 'final' else ci[0] >= 0
            strata_ok = []
            for n in range(2, 9):
                t_mean = float(np.mean([r[metric] for r in reference if r['n'] == n]))
                floor = (g['min_count_teacher_fraction'] * t_mean if t_mean >= 0 else t_mean) - margin
                strata_ok.append(np.mean([r[metric] for r in student if r['n'] == n]) >= floor)
            checks[prefix + ':no_dead_count_stratum'] = all(strata_ok)
            suicide_ci = interval(student, reference, 'suicides', samples=config['suites']['bootstrap_samples'])
            checks[prefix + ':suicide_control'] = suicide_ci[1] <= g['suicide_margin']
            active = max(1, sum(r['active_frames'] for r in student))
            ref_active = max(1, sum(r['active_frames'] for r in reference))
            stuck = sum(r['stuck_frames'] for r in student) / active
            ref_stuck = sum(r['stuck_frames'] for r in reference) / ref_active
            checks[prefix + ':no_wall_stall_collapse'] = stuck <= max(.15, ref_stuck + .05)
            checks[prefix + ':no_spin_collapse'] = sum(r['spin_frames'] for r in student) / active <= .10
            checks[prefix + ':effective_attack'] = sum(r['events']['hits'] for r in student) >= 30 and sum(r['events']['emissions'] for r in student) >= 100
            metrics[prefix] = dict(student_mean=mean(student, metric), teacher_mean=teacher_mean, teacher_fraction=fraction,
                                   paired_ci95=ci, suicide_difference_ci95=suicide_ci, stuck_fraction=stuck)
        greedy, sampled = modes['greedy'], modes['sampled']
        checks[objective + ':sampling_no_performance_collapse'] = interval(sampled, greedy, metric, samples=config['suites']['bootstrap_samples'])[0] >= -margin
        checks[objective + ':sampling_no_suicide_collapse'] = interval(sampled, greedy, 'suicides', samples=config['suites']['bootstrap_samples'])[1] <= g['suicide_margin']
        if stage != 'pretrain':
            previous = branch.get('previous', {})
            for mode in ['greedy', 'sampled']:
                baseline = previous.get(mode, [])
                checks[objective + ':' + mode + ':previous_checkpoint_complete'] = complete(baseline, expected)
                if complete(baseline, expected):
                    checks[objective + ':' + mode + ':previous_nonregression'] = interval(modes[mode], baseline, metric, samples=config['suites']['bootstrap_samples'])[0] >= -margin
        if stage in ['pretrain', 'dagger']:
            offline = branch.get('offline', {})
            for key in ['motion_accuracy', 'fire_precision', 'fire_recall']:
                checks[objective + ':offline_' + key] = float(offline.get(key, -1)) >= config['offline'][key]
            checks[objective + ':offline_positive_support'] = offline.get('fire_positives', 0) >= config['offline']['min_fire_positives']
        if stage == 'dagger':
            for mode in ['greedy', 'sampled']:
                imitation = branch.get('student_state_imitation', {}).get(mode, {})
                for key in ['motion_accuracy', 'fire_precision', 'fire_recall']:
                    checks[f'{objective}:{mode}:student_state_{key}'] = float(imitation.get(key, -1)) >= config['offline'][key]
                checks[f'{objective}:{mode}:student_state_positive_support'] = imitation.get('fire_positives', 0) >= config['offline']['min_fire_positives']
        skill_rows = branch.get('skills', {})
        skill_reference = branch.get('skill_teacher', [])
        checks[objective + ':complete_skill_reference'] = complete(skill_reference, expected_skills, skill=True)
        for mode in ['greedy', 'sampled']:
            rows = skill_rows.get(mode, [])
            checks[objective + ':' + mode + ':complete_skills'] = complete(rows, expected_skills, skill=True)
            if not complete(rows, expected_skills, skill=True) or not complete(skill_reference, expected_skills, skill=True):
                continue
            for skill in SKILLS:
                part = [r for r in rows if r['skill'] == skill]
                baseline = [r for r in skill_reference if r['skill'] == skill]
                score = float(np.mean([r['skill_success'] for r in part]))
                ref_score = float(np.mean([r['skill_success'] for r in baseline]))
                checks[f'{objective}:{mode}:skill_{skill}'] = score >= max(g['skill_success'][skill], g['skill_teacher_fraction'] * ref_score)
                if skill == 'muzzle':
                    checks[f'{objective}:{mode}:zero_muzzle_fizzles'] = sum(r['events']['fizzles'] for r in part) == 0
        if stage == 'pilot':
            updates = evidence.get('updates', [])
            checks['pilot:at_least_three_verified_updates'] = len(updates) >= config['ppo']['pilot_iterations'] >= 3
            checks['pilot:all_updates_within_trust_bounds'] = bool(updates) and all(
                u.get('accepted') and u.get('optimizer_steps', 0) > 0 and 0 < u.get('mean_kl', np.inf) <= config['ppo']['mean_kl'] and
                u.get('p99_kl', np.inf) <= config['ppo']['p99_kl'] and u.get('replay_error', np.inf) <= config['ppo']['replay_tolerance']
                for u in updates)
    return dict(passed=bool(checks) and all(checks.values()), checks={k: bool(v) for k, v in checks.items()}, metrics=metrics)


def certify(stage, checkpoint, evidence_path, config_path, output, *, data_receipt, predecessor=None):
    config = json.loads(Path(config_path).read_text())
    evidence = json.loads(Path(evidence_path).read_text())
    if verify_data(data_receipt).get('rules') != config['data']:
        raise GateFailure('Data receipt coverage rules do not match the configured gate')
    if evidence.get('training_data') != training_identity(evidence.get('training_data', {}).keys(), data_receipt):
        raise GateFailure('Training data identity mismatch')
    if evidence.get('checkpoint_sha256') != digest(checkpoint) or evidence.get('source_sha256') != source_digest() or evidence.get('config_sha256') != digest(config_path):
        raise GateFailure('Evidence not bound to these exact weights/config/source')
    if stage != 'pretrain':
        previous_stage = {'dagger': 'pretrain', 'pilot': 'dagger', 'final': 'pilot'}[stage]
        if predecessor is None:
            raise GateFailure('Missing predecessor certificate')
        verified = verify_certificate(predecessor, previous_stage, evidence.get('objective'), config_path)
        if evidence.get('previous_checkpoint_sha256') != verified['checkpoint_sha256']:
            raise GateFailure('Nonregression baseline is not the accepted predecessor')
    decision = evaluate_gate(stage, evidence, config)
    receipt = dict(kind='promotion', stage=stage, objective=evidence.get('objective'), checkpoint=str(Path(checkpoint).resolve()),
                   checkpoint_sha256=digest(checkpoint), evidence=str(Path(evidence_path).resolve()), evidence_sha256=digest(evidence_path),
                   config=str(Path(config_path).resolve()), config_sha256=digest(config_path), source_sha256=source_digest(),
                   data_receipt=str(Path(data_receipt).resolve()), data_receipt_sha256=digest(data_receipt),
                   predecessor=str(Path(predecessor).resolve()) if predecessor else None,
                   predecessor_sha256=digest(predecessor) if predecessor else None, **decision)
    receipt['receipt_sha256'] = canonical(receipt)
    atomic_json(output, receipt)
    if not decision['passed']:
        failed = [key for key, passed in decision['checks'].items() if not passed]
        raise GateFailure(stage + ' acceptance failed: ' + ', '.join(failed))
    return receipt


def verify_certificate(path, stage, objective, config_path, checkpoint=None):
    receipt = json.loads(Path(path).read_text())
    expected = receipt.pop('receipt_sha256')
    if canonical(receipt) != expected or not receipt.get('passed') or receipt.get('stage') != stage:
        raise GateFailure('Missing, failed or corrupt promotion certificate')
    if stage != 'pretrain' and receipt.get('objective') != objective:
        raise GateFailure('Certificate objective mismatch')
    if receipt.get('source_sha256') != source_digest() or receipt.get('config_sha256') != digest(config_path):
        raise GateFailure('Source/config changed; old acceptance is invalid')
    for name, field in [('checkpoint', 'checkpoint_sha256'), ('evidence', 'evidence_sha256'), ('data_receipt', 'data_receipt_sha256')]:
        if digest(receipt[name]) != receipt[field]:
            raise GateFailure('Certified artifact changed: ' + name)
    if checkpoint is not None and digest(checkpoint) != receipt['checkpoint_sha256']:
        raise GateFailure('Attempted handoff of different weights')
    if verify_data(receipt['data_receipt']).get('rules') != json.loads(Path(config_path).read_text())['data']:
        raise GateFailure('Data coverage rules changed after acceptance')
    evidence = json.loads(Path(receipt['evidence']).read_text())
    if evidence.get('training_data') != training_identity(evidence.get('training_data', {}).keys(), receipt['data_receipt']):
        raise GateFailure('Training data changed after acceptance')
    if evidence.get('checkpoint_sha256') != receipt['checkpoint_sha256'] or not evaluate_gate(stage, evidence, json.loads(Path(config_path).read_text()))['passed']:
        raise GateFailure('Recomputed acceptance failed')
    if receipt['predecessor']:
        if digest(receipt['predecessor']) != receipt['predecessor_sha256']:
            raise GateFailure('Predecessor certificate changed')
        previous_stage = {'dagger': 'pretrain', 'pilot': 'dagger', 'final': 'pilot'}[stage]
        previous = verify_certificate(receipt['predecessor'], previous_stage, objective, config_path)
        if evidence.get('previous_checkpoint_sha256') != previous['checkpoint_sha256']:
            raise GateFailure('Predecessor checkpoint changed')
    return receipt
