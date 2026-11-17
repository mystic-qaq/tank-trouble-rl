import torch
import pytest
from policy.rl.v4.model import (Policy, Critic, CURRENT_SELF, ACTION_COUNT, MOTION_COUNT,
                                ARCH, save_policy, load_policy)


def observations(count=4, device='cpu'):
    x = torch.randn(count, 1966, device=device) * .1
    x[:, 6] = x[:, 16] = 1
    x[:, 454:] = 0
    return x


def test_action_history_is_used_again():
    """V5 dropped fields 56..80; V6 deliberately feeds all 80 self fields."""
    torch.set_num_threads(1)
    m = Policy().eval()
    x = observations()
    y = x.clone()
    y[:, 56:80] = torch.randn(len(x), 24) * 10
    style = torch.full((len(x),), 3)
    assert CURRENT_SELF == list(range(80))
    with torch.no_grad():
        assert not torch.equal(m.logits(x, style), m.logits(y, style))
        y[:, 4] += 1
        assert not torch.equal(m.logits(x, style), m.logits(y, style))
    assert not any(isinstance(layer, (torch.nn.GRU, torch.nn.LSTM)) for layer in m.modules())


def test_joint_controls_are_legal_and_quantization_is_bounded():
    m = Policy()
    controls = m.controls(torch.arange(ACTION_COUNT))
    assert torch.equal(m.targets(controls), torch.arange(ACTION_COUNT))
    x = observations(32)
    x[:, 16] = 0
    style = torch.zeros(32, dtype=torch.long)
    action, lp, entropy, controls = m.act(x, style)
    assert (controls[:, 2] == 0).all()
    assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()
    assert set(controls[:, 0].tolist()).issubset({-1, 0, 1})


def test_inverse_cdf_boundary_draws_never_select_masked_fire_actions():
    m = Policy().eval()
    x = observations(4)
    x[:, 16] = 0
    style = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        _, lp, _, controls = m.act(x, style, uniforms=torch.tensor([0., 1., .99999999, .5]))
    assert (controls[:, 2] == 0).all()
    assert torch.isfinite(lp).all() and (lp > -1e6).all()


def test_bc_loss_is_a_normalized_weighted_mean():
    torch.set_num_threads(1)
    torch.manual_seed(81)
    m = Policy().eval()
    x = observations(32)
    style = torch.arange(32) % 9
    controls = torch.zeros(32, 3)
    controls[:, 0] = torch.arange(32) % 3 - 1
    controls[:, 2] = (torch.arange(32) % 2).float()
    w = torch.linspace(.2, 1.8, 32)
    per = m(x, style, controls)
    assert torch.allclose(m.bc_loss(x, style, controls, w), (per * w).sum() / w.sum(), atol=1e-6)
    assert torch.allclose(m.bc_loss(x, style, controls), per.mean(), atol=1e-6)
    assert torch.allclose(m.bc_loss(x, style, controls, torch.ones(32)), per.mean(), atol=1e-6)


def test_training_and_deployment_share_one_logits_distribution():
    torch.manual_seed(4)
    m = Policy(.35).eval()
    x = observations(8)
    style = torch.full((8,), 3)
    actions = torch.arange(8) % ACTION_COUNT
    with torch.no_grad():
        logits = m.logits(x, style)
        _, lp, _, _ = m.act(x, style, actions=actions)
        expected = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
    assert torch.allclose(lp, expected, atol=1e-5)
    assert logits.shape == (8, ACTION_COUNT)


def test_fire_margin_can_depend_on_motion():
    """V5 forced P(fire | motion) to be constant; V6 must break that coupling."""
    torch.manual_seed(5)
    m = Policy().eval()
    x = observations(4)
    x[:, 16] = 1
    style = torch.zeros(4, dtype=torch.long)
    with torch.no_grad():
        marginals = m.logits(x, style).reshape(-1, MOTION_COUNT, 2).softmax(-1)[..., 1]
        assert float(marginals.std()) > 1e-4, 'fire margin must vary across manoeuvres'
        before = marginals.mean(1)
        m.fire_global.bias.add_(1.)
        after = m.logits(x, style).reshape(-1, MOTION_COUNT, 2).softmax(-1)[..., 1].mean(1)
        assert (after > before).all(), 'the global fire term must still control overall fire rate'
    assert m.motion_head.out_features == MOTION_COUNT


def test_relational_encoder_mixes_all_entity_groups():
    torch.manual_seed(11)
    m = Policy().eval()
    x = observations(3)
    style = torch.zeros(3, dtype=torch.long)
    assert len(m.encoder.blocks) >= 2
    assert m.encoder.type_embedding.num_embeddings == 5
    with torch.no_grad():
        base = m.logits(x, style)
        for start, width in ((454, 24), (622, 24), (1582, 12)):
            y = x.clone()
            y[:, start:start + width] = torch.randn(3, width) * 2
            y[:, start + width - 1] = 1.
            assert not torch.allclose(base, m.logits(y, style), atol=1e-5)
        z = x.clone()
        z[:, 80:454] = torch.randn(3, 374)
        assert not torch.allclose(base, m.logits(z, style), atol=1e-5)


def test_checkpoint_roundtrip_records_shared_lineage(tmp_path):
    """The fork phase depends on iteration/stage metadata surviving save/load."""
    m = Policy()
    path = tmp_path / 'shared.pt'
    save_policy(m, path, stage='shared', objective='frag', iteration=17, score=.42)
    loaded, meta = load_policy(path, 'cpu')
    assert meta['architecture'] == ARCH
    assert meta['observation'] == 'v2_full_self_including_action_history'
    assert meta['stage'] == 'shared' and meta['iteration'] == 17
    assert all(torch.allclose(a, b) for a, b in zip(m.parameters(), loaded.parameters()))


def test_empty_entities_and_separate_critic_have_finite_gradients():
    m = Policy()
    x = observations()
    style = torch.zeros(4, dtype=torch.long)
    loss = m.bc_loss(x, style, torch.zeros(4, 3))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    critic = Critic()
    assert not ({id(p) for p in critic.parameters()} & {id(p) for p in m.parameters()})


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_collection_and_replay_are_the_same_distribution(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA not available in this contract run')
    torch.manual_seed(9)
    m = Policy().to(device).eval()
    x = observations(23, device)
    style = torch.full((len(x),), 3, device=device)
    with torch.no_grad():
        a, lp, _, _ = m.act(x, style)
        replay = torch.cat([m.act(x[i:i + 7], style[i:i + 7], actions=a[i:i + 7])[1] for i in range(0, len(x), 7)])
    assert float((lp - replay).abs().max()) < 1e-3
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
