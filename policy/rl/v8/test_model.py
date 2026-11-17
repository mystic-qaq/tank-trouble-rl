"""Shape / gradient / contract tests for the V8 policy.

Usage: python -m policy.rl.v8.test_model
"""
import torch

from .model import (Policy, OBS, ENEMY, ENEMY_SLOTS, BULLET, BULLET_SLOTS, LIDAR_N, AIM_N,
                    INTENT_N, STYLES, MEMORY, ACTION_DIM)


def build(b=4):
    torch.manual_seed(0)
    model = Policy().eval()
    obs = torch.rand(b, OBS)
    obs[:, 6] = 1.0                      # self alive
    obs[:, ENEMY + 7:ENEMY + ENEMY_SLOTS * 9:9] = 1.0     # every enemy slot present
    obs[:, BULLET + 7:BULLET + BULLET_SLOTS * 8:8] = 0.0  # no bullets
    map_vec = torch.randint(0, 2, (b, 346)).float()
    style = torch.full((b,), STYLES.index('hy'), dtype=torch.long)
    return model, obs, map_vec, style


def main():
    model, obs, map_vec, style = build()
    map_feat = model.encode_map(map_vec)
    assert map_feat[0].shape == (4, 256), map_feat[0].shape
    assert map_feat[1].shape == (4, 16, 128), map_feat[1].shape

    mem = model.zero_memory(4, 'cpu')
    for _ in range(3):
        mem = model.advance_memory(obs, mem)
    assert mem[0].shape == (4, ENEMY_SLOTS, MEMORY)
    assert mem[1].shape == (4, MEMORY)

    out = model.step(obs, style, map_feat, mem)
    assert out['z'].shape == (4, 512)
    assert out['drive'].shape == (4, 3)
    assert out['steer_mu'].shape == (4, 1) and out['steer_logstd'].shape == (4, 1)
    assert out['fire'].shape == (4,)
    assert out['aim'].shape == (4, AIM_N, 4)
    assert out['threat'].shape == (4, BULLET_SLOTS, 3)
    assert out['intent'].shape == (4, INTENT_N, 2)

    # actions
    ctl, logp, ent, five = model.act(obs, style, map_feat, mem, deterministic=True)
    assert ctl.shape == (4, 3) and five.shape == (4, 5)
    assert ctl[:, 1].abs().max() <= 1.0
    assert set(ctl[:, 0].tolist()) <= {-1.0, 0.0, 1.0}
    assert set(ctl[:, 2].tolist()) <= {0.0, 1.0}
    assert torch.isfinite(logp).all() and torch.isfinite(ent).all()

    u = torch.rand(4)
    ctl1, lp1, _, _ = model.act(obs, style, map_feat, mem, uniforms=u)
    ctl2, lp2, _, _ = model.act(obs, style, map_feat, mem, uniforms=u)
    assert torch.equal(ctl1, ctl2) and torch.equal(lp1, lp2), 'uniforms must be deterministic'
    assert (ctl1[:, 1].abs() <= 1.0).all()

    # forward head and gradients
    model.train()
    out = model.step(obs, style, map_feat, mem)
    forward = model.forward_head(torch.cat([out['z'], torch.zeros(4, ACTION_DIM)], -1))
    assert forward.shape == (4, OBS)
    loss = (out['drive'].logsumexp(-1).mean() + out['steer_mu'].abs().mean()
            + out['fire'].abs().mean() + out['aim'].sigmoid().mean()
            + out['threat'].sigmoid().mean() + out['intent'].sigmoid().mean()
            + forward.abs().mean())
    loss.backward()
    grads = [p.grad is not None for p in model.parameters()]
    assert all(grads), 'every parameter must receive gradient: %d/%d' % (sum(grads), len(grads))
    assert torch.isfinite(torch.tensor(float(loss)))

    params = sum(p.numel() for p in model.parameters())
    print('shapes OK; uniforms deterministic; all %d tensors got gradient' % len(grads))
    print('parameters %.2fM' % (params / 1e6))
    print('token count %d' % (1 + 16 + LIDAR_N + 1 + ENEMY_SLOTS + BULLET_SLOTS))


if __name__ == '__main__':
    main()
