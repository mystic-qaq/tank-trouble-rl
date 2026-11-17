"""V6 relational entity encoder with a motion-conditional fire head.

The 1966-d observation contract is intentionally unchanged: the frozen
demonstration corpus (9319 multi-teacher + 1200 skill trajectories) stays valid,
and the engine-side analytic features in `v2/geometry.mjs` keep providing exact
multi-bounce bullet forecasts. What changes is how those fields are combined:

  * all 80 self fields are used, including the 24-d eight-frame action history;
  * enemies, bullets, aim candidates and map cells become ONE token set with
    two layers of self-attention, so bullet/wall/tank relations are learned
    jointly instead of being pooled by three independent single-query reads;
  * the fire margin is a function of (hidden, motion), so P(fire | motion) is
    no longer forced to be independent of the chosen manoeuvre, while a single
    global fire logit still keeps the rare fire class well conditioned.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .common import STYLES

OBS = 1966
ARCH = 'relational_motion_conditional_fire_v6'
TURNS = [-1., -.75, -.5, -.25, -.125, -.0625, -.03125, -.015625, 0.,
         .015625, .03125, .0625, .125, .25, .5, .75, 1.]
ACTION_COUNT = 3 * len(TURNS) * 2
MOTION_COUNT = 3 * len(TURNS)
SELF_BLOCK = 80
CURRENT_SELF = list(range(SELF_BLOCK))
MAP_SLICE = slice(80, 454)
ENEMY_SLICE = slice(454, 622)
BULLET_SLICE = slice(622, 1582)
AIM_SLICE = slice(1582, 1966)
TOKEN_WIDTH = 128
TYPE_COUNT = 5


def precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.layers(x)


class AttentionBlock(nn.Module):
    """Pre-LN self-attention with a small feed-forward block."""

    def __init__(self, width, heads=4, expansion=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * expansion), nn.SiLU(),
                                         nn.Linear(width * expansion, width))

    def forward(self, tokens, padding):
        residual = self.norm1(tokens)
        tokens = tokens + self.attention(residual, residual, residual,
                                         key_padding_mask=padding, need_weights=False)[0]
        return tokens + self.feedforward(self.norm2(tokens))


class Encoder(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.style = nn.Embedding(len(STYLES), 32)
        self.self_net = nn.Sequential(nn.Linear(SELF_BLOCK + 32, 256), nn.SiLU(), Residual(256))
        self.map_conv = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.map_vec = nn.Sequential(nn.Linear(32, 256), nn.SiLU())
        self.map_proj = nn.Linear(32, TOKEN_WIDTH)
        self.entity_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(n, TOKEN_WIDTH), nn.SiLU(), nn.Linear(TOKEN_WIDTH, TOKEN_WIDTH))
            for n in (24, 24, 12)])
        self.type_embedding = nn.Embedding(TYPE_COUNT, TOKEN_WIDTH)
        self.cls = nn.Parameter(torch.zeros(1, 1, TOKEN_WIDTH))
        self.blocks = nn.ModuleList([AttentionBlock(TOKEN_WIDTH) for _ in range(layers)])
        self.token_norm = nn.LayerNorm(TOKEN_WIDTH)
        # own 256 + map vector 256 + CLS 128 + masked entity pool 128
        self.trunk = nn.Sequential(nn.Linear(768, 512), nn.SiLU(), Residual(512),
                                   Residual(512), nn.LayerNorm(512))

    def forward(self, x, style):
        batch, device = x.shape[0], x.device
        own = self.self_net(torch.cat([x[:, :SELF_BLOCK], self.style(style)], -1))
        features = self.map_conv(x[:, MAP_SLICE].reshape(-1, 2, 11, 17))
        map_vec = self.map_vec(F.adaptive_avg_pool2d(features, 1).flatten(1))
        spatial = F.adaptive_avg_pool2d(features, (4, 4)).flatten(2).transpose(1, 2)
        map_tokens = self.map_proj(spatial)
        groups = [x[:, ENEMY_SLICE].reshape(-1, 7, 24),
                  x[:, BULLET_SLICE].reshape(-1, 40, 24),
                  x[:, AIM_SLICE].reshape(-1, 32, 12)]
        entity_tokens = [map_tokens]
        valid = [torch.ones(batch, 16, dtype=torch.bool, device=device)]
        type_ids = [torch.zeros(batch, 16, dtype=torch.long, device=device)]
        for group, encode, type_id in zip(groups, self.entity_nets, (1, 2, 3)):
            entity_tokens.append(encode(group))
            valid.append(group[..., -1] > 0)
            type_ids.append(torch.full_like(valid[-1], type_id, dtype=torch.long))
        cls_valid = torch.ones(batch, 1, dtype=torch.bool, device=device)
        cls_type = torch.full((batch, 1), 4, dtype=torch.long, device=device)
        # CLS and the 16 map tokens are always unmasked, so no row is fully masked.
        padding = ~torch.cat([cls_valid, *valid], 1)
        types = torch.cat([cls_type, *type_ids], 1)
        tokens = torch.cat([self.cls.expand(batch, -1, -1), *entity_tokens], 1)
        tokens = tokens + self.type_embedding(types)
        for block in self.blocks:
            tokens = block(tokens, padding)
        tokens = self.token_norm(tokens)
        present = (~padding[:, 1:]).unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens[:, 1:] * present).sum(1) / present.sum(1).clamp_min(1.)
        return self.trunk(torch.cat([own, map_vec, tokens[:, 0], pooled], -1))


class Policy(nn.Module):
    def __init__(self, temperature=.35):
        super().__init__()
        precision()
        self.temperature = float(temperature)
        if not 0 < self.temperature <= 1:
            raise ValueError('Invalid fixed exploration temperature')
        self.encoder = Encoder()
        self.motion_head = nn.Linear(512, MOTION_COUNT)
        self.fire_global = nn.Linear(512, 1)
        self.fire_conditional = nn.Sequential(nn.Linear(512 + MOTION_COUNT, 256), nn.SiLU(),
                                              nn.Linear(256, MOTION_COUNT))
        self.register_buffer('turns', torch.tensor(TURNS))
        self.register_buffer('fire_classes', torch.arange(ACTION_COUNT) % 2 == 1)

    def logits(self, x, style):
        hidden = self.encoder(x, style)
        motion = self.motion_head(hidden)
        # The global term keeps the rare class well conditioned; the conditional
        # term lets P(fire | motion) vary across manoeuvres. Detaching motion
        # here keeps the two gradient paths distinct.
        fire = self.fire_global(hidden) + self.fire_conditional(
            torch.cat([hidden, motion.detach()], -1))
        scores = torch.stack([motion, motion + fire], dim=-1).reshape(-1, ACTION_COUNT) / self.temperature
        unavailable = self.fire_classes[None] & (x[:, 16:17] < .5)
        return scores.masked_fill(unavailable, -1e9)

    def controls(self, actions):
        a = actions.long()
        return torch.stack([(a // (len(TURNS) * 2) - 1).float(), self.turns[(a // 2) % len(TURNS)], (a % 2).float()], -1)

    def targets(self, controls):
        turn = (controls[:, 1:2] - self.turns[None]).abs().argmin(1)
        return ((controls[:, 0].long() + 1) * len(TURNS) + turn) * 2 + controls[:, 2].long()

    def act(self, x, style, actions=None, deterministic=False, uniforms=None):
        logits = self.logits(x, style)
        distribution = Categorical(logits=logits, validate_args=False)
        if actions is None:
            if deterministic:
                actions = logits.argmax(-1)
            elif uniforms is not None:
                cdf = distribution.probs.cumsum(-1)
                cdf = (cdf / cdf[:, -1:]).contiguous()
                # Float32 conversion can round a per-case draw to 1. Right-sided
                # search skips zero-probability plateaus, including a draw of 0.
                u = uniforms.clamp(0, 1 - torch.finfo(cdf.dtype).eps)[:, None].contiguous()
                actions = torch.searchsorted(cdf, u, right=True).squeeze(1)
            else:
                actions = distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), self.controls(actions)

    def forward(self, x, style, controls, importance=None):
        """Per-example cross entropy; sample weighting is applied in bc_loss."""
        return F.cross_entropy(self.logits(x, style), self.targets(controls), reduction='none')

    def bc_loss(self, x, style, controls, importance=None):
        per = self(x, style, controls)
        if importance is None:
            return per.mean()
        w = importance.to(per.dtype)
        return (per * w).sum() / w.sum().clamp_min(1e-8)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(512, 1)

    def forward(self, x, style):
        return self.head(self.encoder(x, style)).squeeze(-1)


def save_policy(model, path, **metadata):
    path = __import__('pathlib').Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(architecture=ARCH, temperature=model.temperature, model=model.state_dict(), round_limit=60,
                    observation='v2_full_self_including_action_history', **metadata), tmp)
    tmp.replace(path)


def load_policy(path, device='cpu'):
    meta = torch.load(path, map_location='cpu', weights_only=False)
    if meta.get('architecture') != ARCH or meta.get('round_limit') != 60:
        raise ValueError('Incompatible checkpoint; V1-V5 weights cannot be resumed')
    model = Policy(meta['temperature']).to(device)
    model.load_state_dict(meta['model'])
    return model, meta
