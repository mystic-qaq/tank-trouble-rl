import copy
import json
from pathlib import Path
import numpy as np
import pytest
from policy.rl.v4.common import canonical, GateFailure, digest, source_digest, atomic_json
from policy.rl.v4.tasks import combat, skills
from policy.rl.v4.gates import evaluate_gate, interval, SPLITS, verify_certificate

CONFIG = json.loads(Path(__file__).with_name('config.json').read_text())


def result(task, skill=False):
    return dict(**task, opponents=task['players'], errors=0, frames=120, duration=2., round_complete=not skill,
                win=1., kills=1., netKills=1., suicides=0., active_frames=120, spin_frames=0, stuck_frames=0,
                events={'hits': 1, 'emissions': 3, 'fizzles': 0}, skill_success=True)


def evidence(stage='pretrain', objective='win'):
    split = SPLITS[stage]
    tasks = combat(split, CONFIG['suites']['gate_blocks'], CONFIG['seed'])
    rows = [result(t) for t in tasks]
    micro = [result(t, True) for t in skills(split, CONFIG['suites']['skill_cases'], CONFIG['seed'])]
    imitation = dict(motion_accuracy=1., fire_precision=1., fire_recall=1., fire_positives=1000)
    branch = dict(teacher=rows, modes={'greedy': copy.deepcopy(rows), 'sampled': copy.deepcopy(rows)},
                  skills={'greedy': micro, 'sampled': micro}, skill_teacher=micro, offline=imitation,
                  student_state_imitation={'greedy': imitation, 'sampled': imitation}, previous={'greedy': rows, 'sampled': rows})
    return dict(split=split, objective=objective, temperature=CONFIG['temperature'], tasks_sha256=canonical(tasks),
                branches={o: copy.deepcopy(branch) for o in (['win', 'frag'] if stage == 'pretrain' else [objective])},
                updates=[dict(accepted=True, optimizer_steps=2, mean_kl=.001, p99_kl=.002, replay_error=.00001)] * 3)


def test_complete_good_pretraining_can_pass():
    decision = evaluate_gate('pretrain', evidence(), CONFIG)
    assert decision['passed'], {k: v for k, v in decision['checks'].items() if not v}


def test_training_steps_cannot_substitute_for_missing_evidence():
    decision = evaluate_gate('pretrain', {'steps': 10**12, 'elapsed_hours': 100, 'status': 'complete'}, CONFIG)
    assert not decision['passed']


@pytest.mark.parametrize('failure', ['sampled_suicides', 'incomplete', 'wrong_seat', 'wall_stall', 'no_shots', 'no_fire_recall'])
def test_each_material_failure_blocks_promotion(failure):
    e = evidence()
    b = e['branches']['win']
    if failure == 'sampled_suicides':
        for r in b['modes']['sampled']:
            r.update(suicides=1., kills=2.)
    elif failure == 'incomplete':
        b['modes']['greedy'].pop()
    elif failure == 'wrong_seat':
        b['modes']['greedy'][0]['seat'] = 99
    elif failure == 'wall_stall':
        for r in b['modes']['greedy']:
            r['stuck_frames'] = r['active_frames']
    elif failure == 'no_shots':
        for r in b['modes']['greedy']:
            r['events']['hits'] = r['events']['emissions'] = 0
    else:
        b['offline']['fire_recall'] = 0.
    assert not evaluate_gate('pretrain', e, CONFIG)['passed']


def test_dagger_must_correct_the_states_student_actually_visits():
    e = evidence('dagger')
    assert evaluate_gate('dagger', e, CONFIG)['passed']
    e['branches']['win']['student_state_imitation']['greedy']['motion_accuracy'] = .2
    assert not evaluate_gate('dagger', e, CONFIG)['passed']


def test_pilot_needs_real_bounded_updates_not_just_training_return():
    e = evidence('pilot')
    assert evaluate_gate('pilot', e, CONFIG)['passed']
    e['updates'][0]['mean_kl'] = .3
    assert not evaluate_gate('pilot', e, CONFIG)['passed']


def test_final_tie_is_not_superiority():
    assert not evaluate_gate('final', evidence('final'), CONFIG)['passed']


def test_old_or_missing_receipt_cannot_enter_next_stage(tmp_path):
    p = tmp_path / 'fake.json'
    p.write_text(json.dumps(dict(passed=True, stage='dagger', steps=999999, receipt_sha256='wrong')))
    with pytest.raises(GateFailure):
        verify_certificate(p, 'dagger', 'win', Path(__file__).with_name('config.json'))


def test_valid_receipt_is_invalid_after_weight_config_or_report_change(tmp_path, monkeypatch):
    from policy.rl.v4 import gates
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(CONFIG))
    model = tmp_path / 'candidate.pt'
    model.write_bytes(b'isolated-gate-test-artifact')
    data = tmp_path / 'data-receipt.json'
    data.write_text('{}')
    e = evidence()
    e.update(checkpoint_sha256=digest(model), config_sha256=digest(config), source_sha256=source_digest(), training_data={})
    report = tmp_path / 'evidence.json'
    atomic_json(report, e)
    receipt = tmp_path / 'acceptance.json'
    # Isolate promotion integrity from separately tested data/engine audit contracts.
    monkeypatch.setattr(gates, 'verify_data', lambda _: {'rules': CONFIG['data']})
    monkeypatch.setattr(gates, 'training_identity', lambda *_: {})
    gates.certify('pretrain', model, report, config, receipt, data_receipt=data)
    gates.verify_certificate(receipt, 'pretrain', 'win', config, model)
    model.write_bytes(b'different-weights')
    with pytest.raises(GateFailure, match='artifact changed'):
        gates.verify_certificate(receipt, 'pretrain', 'win', config, model)
    model.write_bytes(b'isolated-gate-test-artifact')
    original = config.read_text()
    config.write_text(original + '\n')
    with pytest.raises(GateFailure, match='Source/config changed'):
        gates.verify_certificate(receipt, 'pretrain', 'win', config, model)
    config.write_text(original)
    report.write_text(report.read_text() + '\n')
    with pytest.raises(GateFailure, match='artifact changed'):
        gates.verify_certificate(receipt, 'pretrain', 'win', config, model)


def test_suites_are_disjoint_and_seats_form_complete_blocks():
    seen = set()
    for split in ['development', *SPLITS.values()]:
        tasks = combat(split, 2)
        layouts = {(t['seed'], t['spawnSeed']) for t in tasks}
        assert not seen & layouts
        seen |= layouts
        for n in range(2, 9):
            for b in {t['block'] for t in tasks if t['n'] == n}:
                part = [t for t in tasks if t['block'] == b]
                assert len(part) == 2 * n
                for map_seed in {t['mapSeed'] for t in part}:
                    assert {t['seat'] for t in part if t['mapSeed'] == map_seed} == set(range(n))


def test_bootstrap_keeps_seats_in_the_same_layout_cluster():
    tasks = combat('ci-test', 4)
    a = [result(t) for t in tasks]
    b = copy.deepcopy(a)
    for r in a:
        r['netKills'] += .5
    ci = interval(a, b, 'netKills')
    assert np.allclose(ci, [.5, .5])
