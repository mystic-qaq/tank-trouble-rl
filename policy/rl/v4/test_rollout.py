import time
import pytest
import torch
from policy.rl.v4.common import BudgetExhausted
from policy.rl.v4.rollout import run


class Idle:
    def act(self, x, style, **kwargs):
        size = len(x)
        return torch.full((size,), 50), torch.zeros(size), torch.zeros(size), torch.zeros(size, 3)


def test_skill_evaluation_does_not_silently_run_reference_instead_of_student():
    tasks = [dict(seed=723, n=2, seat=0, skill='static', players=['builtin', 'builtin'])]
    student, _, _ = run(tasks, Idle(), workers=1)
    reference, _, _ = run(tasks, teacher='reference', workers=1)
    assert not student[0]['skill_success']
    assert reference[0]['skill_success']
    assert student[0]['events']['emissions'] == 0


def test_deadline_does_not_return_partial_acceptance():
    tasks = [dict(seed=1, n=2, seat=0, players=['builtin', 'builtin'])]
    with pytest.raises(BudgetExhausted):
        run(tasks, Idle(), workers=1, deadline=time.monotonic() - 1)
