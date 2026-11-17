import copy
import pytest
import torch
from policy.rl.v4.ppo import transactional_actor_update, distribution_logits, exact_kl
from policy.rl.v4.common import GateFailure
from policy.rl.v4.rollout import monte_carlo


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def logits(self, x, style):
        return self.head(x)


def batch(actor):
    x = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]).repeat(8, 1)
    s = torch.zeros(len(x), dtype=torch.long)
    a = torch.tensor([0, 1, 2, 1]).repeat(8)
    lp = actor.logits(x, s).log_softmax(-1).gather(1, a[:, None]).squeeze(1).detach()
    return x, s, a, lp, torch.tensor([1., 1., -1., -1.]).repeat(8)


def rules():
    return dict(batch=8, replay_tolerance=1e-3, backtracks=2, epochs=2, mean_kl=.01, p99_kl=.05, clip=.1, bc_aux=0.)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_equal(a[k], b[k])
    elif isinstance(a, list):
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


def test_unsafe_update_restores_weights_and_optimizer_moments():
    m = Tiny()
    opt = torch.optim.Adam(m.parameters(), lr=10.)
    (m.head.weight.sum() * 0 + m.head.bias.sum() * 0).backward()
    opt.step()  # Create Adam moments before testing their rollback.
    before, state = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    r = rules()
    r.update(mean_kl=1e-14, p99_kl=1e-14)
    with pytest.raises(GateFailure, match='rolled back'):
        transactional_actor_update(m, opt, *batch(m), r)
    assert_equal(m.state_dict(), before)
    assert_equal(opt.state_dict(), state)


def test_accepted_update_satisfies_exact_whole_rollout_kl():
    torch.manual_seed(3)
    m = Tiny()
    opt = torch.optim.Adam(m.parameters(), lr=.001)
    args = batch(m)
    old = distribution_logits(m, args[0], args[1])
    report = transactional_actor_update(m, opt, *args, rules())
    kl, p99 = exact_kl(old, distribution_logits(m, args[0], args[1]))
    assert report['accepted'] and report['optimizer_steps'] > 0
    assert 0 < kl <= rules()['mean_kl'] and p99 <= rules()['p99_kl']
    assert report['mean_kl'] == kl


def test_wrong_old_probabilities_are_rejected_before_update():
    m = Tiny()
    opt = torch.optim.Adam(m.parameters())
    args = list(batch(m))
    args[3] += .1
    before = copy.deepcopy(m.state_dict())
    with pytest.raises(GateFailure, match='before any actor update'):
        transactional_actor_update(m, opt, *args, rules())
    assert_equal(m.state_dict(), before)


def test_posthumous_rewards_and_episode_boundaries_are_preserved():
    records = {0: [dict(reward=0., active=True), dict(reward=-1., active=True), dict(reward=2., active=False)],
               1: [dict(reward=-1., active=True)]}
    assert [r['return'] for r in monte_carlo(records)] == [1., 1., -1.]
