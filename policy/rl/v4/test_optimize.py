import numpy as np
import pytest
import torch
import json
import subprocess
import sys
from policy.rl.v4.common import GateFailure, STYLES
from policy.rl.v4.optimize import (METRIC, mean, shaped_returns, plan_deadlines, teacher_targets,
                                  curriculum_tasks, certification_start)
from policy.rl.v4.model import ACTION_COUNT, Policy
from policy.rl.v4.tasks import training

RULES = dict(gamma=1.0, survival_bonus=.0002, death_penalty=1.0)


def row(reward=0., active=True):
    return dict(reward=reward, active=active, obs=np.zeros(4, dtype=np.float32),
                action=0, logprob=-1.)


def test_win_shaping_prefers_survival_and_penalises_death():
    won = shaped_returns({0: [row(0., True), row(0., True), row(1., True)]}, 'win', RULES)
    died = shaped_returns({0: [row(0., True), row(0., True), row(0., False)]}, 'win', RULES)
    assert won[0][1] > died[0][1]
    assert died[0][1] < 0
    assert all(r['active'] for r, _ in won + died)


def test_frag_returns_are_undiscounted_future_net_kills():
    episode = [row(1., True), row(-1., True), row(0., True)]
    result = shaped_returns({0: episode}, 'frag', RULES)
    assert [round(value, 4) for _, value in result] == [0., -1., 0.]


def test_no_active_frames_is_rejected():
    with pytest.raises(GateFailure, match='No active'):
        shaped_returns({0: [row(0., False)]}, 'win', RULES)


def test_metric_and_mean_helpers():
    assert METRIC == {'win': 'win', 'frag': 'netKills'}
    assert mean([{'netKills': 1.}, {'netKills': -1.}], 'netKills') == 0.
    assert mean([], 'win') == 0.


def test_acceptance_budget_is_reserved_not_consumed_by_training():
    train, final = plan_deadlines(1000., 3600, 1200, False)
    assert train == 1000. + 2400
    assert final == 1000. + 3600
    assert final - train == 1200
    train, final = plan_deadlines(1000., 300, 3600, True)
    assert (train, final) == (1300., 1300.)
    with pytest.raises(GateFailure):
        plan_deadlines(1000., 600, 600, False)
    with pytest.raises(GateFailure):
        plan_deadlines(1000., 600, 0, True)


def test_teacher_targets_are_deployment_action_indices():
    actor = Policy().eval()
    labels = [[1., 0., 0.], [0., .125, 1.], [-1., -1., 1.]]
    rows = [dict(label=label) for label in labels]
    targets = teacher_targets(actor, rows, 'cpu')
    assert targets.shape == (3,)
    assert bool((targets >= 0).all() and (targets < ACTION_COUNT).all())
    # action = motion * 2 + fire, so the parity must reproduce the shoot bit.
    assert [int(t) % 2 for t in targets] == [0, 1, 1]


def test_teacher_targets_reject_incomplete_labels():
    assert teacher_targets(Policy(), [dict(label=None)], 'cpu') is None


def test_curriculum_ramps_difficulty_without_changing_the_acceptance_split():
    weak = ['builtin', 'minimax', 'kimi', 'qwen']
    rules = dict(episodes=16, curriculum=dict(enabled=True, start_n=2, end_n=8,
                                              ramp_iterations=10, weak_opponents=weak))
    config = dict(seed=20261107)
    early = curriculum_tasks(config, 1, rules)
    assert len(early) == 16
    assert all(task['n'] == 2 for task in early)
    assert all(all(p in weak for i, p in enumerate(task['players']) if i != task['seat'])
               for task in early)
    late = curriculum_tasks(config, 11, rules)
    assert len(late) == 16 and max(task['n'] for task in late) == 8
    assert late == training(config['seed'], 11, 16)


def test_disabled_curriculum_is_exactly_the_original_task_generator():
    rules = dict(episodes=12, curriculum=dict(enabled=False))
    assert curriculum_tasks(dict(seed=20261107), 7, rules) == training(20261107, 7, 12)


def test_only_full_strength_iterations_can_certify_a_checkpoint():
    # Easy-curriculum wins must never be selected as the accepted checkpoint.
    assert certification_start(dict(enabled=True, ramp_iterations=16)) == 16
    assert certification_start(dict(enabled=False, ramp_iterations=16)) == 0
    assert certification_start(None) == 0


def test_kill_auxiliary_is_dense_for_win_but_never_for_frag():
    def frame(reward=0., kills=0., suicides=0., active=True):
        return dict(reward=reward, active=active, obs=np.zeros(4, dtype=np.float32), action=0,
                    logprob=-1., kills_delta=kills, suicides_delta=suicides)
    rules = {**RULES, 'kill_aux': .25}
    plain = shaped_returns({0: [frame(), frame()]}, 'win', RULES)
    killed = shaped_returns({0: [frame(kills=1.), frame()]}, 'win', rules)
    # The kill signal propagates backwards to earlier frames; the final active
    # frame is pinned by the death penalty and must be unchanged.
    assert killed[0][1] > plain[0][1]
    assert abs(killed[1][1] - plain[1][1]) < 1e-6
    # The auxiliary is training-only: frag still uses the engine's raw reward.
    frag = shaped_returns({0: [frame(kills=1.), frame()]}, 'frag', rules)
    assert [round(v, 4) for _, v in frag] == [0., 0.]
    # A kill must not be mistaken for a win by the death-penalty trigger.
    without_win = shaped_returns({0: [frame(kills=1.), frame(), frame(active=False)]}, 'win', rules)
    assert without_win[0][1] < 0


def test_optimize_module_entrypoint_is_callable():
    # A missing __main__ guard makes `python -m ...` exit 0 without training.
    proc = subprocess.run([sys.executable, '-m', 'policy.rl.v4.optimize', '--help'],
                          capture_output=True, text=True, timeout=90)
    assert proc.returncode == 0, proc.stderr
    assert '--phase' in proc.stdout and '--init' in proc.stdout
    assert '--acceptance-seconds' in proc.stdout


def test_one_policy_style_is_shared_by_training_and_evaluation():
    config = json.loads(__import__('pathlib').Path(__file__).with_name('config.json').read_text())
    assert config['version'] == 6
    assert config['policy_style'] in STYLES
    assert config['optimize']['kill_aux'] > 0
    assert config['optimize']['teacher_aux'] == 0
    assert 0 < config['optimize']['bc_aux'] < .01
