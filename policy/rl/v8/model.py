"""V8: Schema V3 perception-only observation, factored action, per-opponent memory.

Contract: policy/rl/v8/DESIGN.md. Three load-bearing decisions:

  1. The observation carries no solver output. traceShot / intercept / flight /
     pathsFor / closest appear only in training labels, never in the input.
  2. The action is three independent distributions (drive Categorical(3), steer
     tanh-Normal, fire Bernoulli), never a 102-way joint.
  3. Memory is per-opponent: one shared-weight GRU per enemy slot, so the policy can
     answer "is *that* one tracking me" instead of muddying seven opponents into one
     hidden vector.

Tensor flow, B = batch:

  obs (B,608) ─┬─ self   11 ⊕ style(9→32) ─Linear→128
               ├─ enemy  7×9  ─shared Linear→128 ─┐
               ├─ bullet 40×8 ─shared Linear→128  │ tokens
               ├─ lidar  64×3 ─shared Linear→128  │
               └─ map           ─conv→16×128 +256 ┘

  slot input 7×16 = [enemy 9 | enemy action 3 | bearing sin/cos, range, speed, cooldown]
                          → GRUCell_shared → slot memory (B,7,128)
  global input 13 = [self 8 | self action 3 | clock | alive share]
                          → GRUCell_global → global memory (B,128)

  tokens = [CLS] + map(16) + lidar(64) + self(1) + fused-enemy(7) + bullet(40) = 129 × 128
  fused-enemy_k = MLP(enemy_embedding_k ⊕ slot_memory_k)
  3 × Pre-LN self-attention(4 heads) → LayerNorm → CLS ⊕ masked-mean
  trunk: Linear(128+256+128+128+128 = 768 → 512) SiLU Residual Residual LayerNorm → z

  action heads: drive 512→3 | steer 512→2 (mu, log-sigma) | fire 512→1
  aux heads   : aim 512→192 | threat 512→120 | intent 512→14 | forward 517→608
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal

OBS = 608
ACTION_DIM = 3
STYLES = ['deepseek', 'glm', 'gpt', 'hy', 'kimi', 'minimax', 'qwen', 'builtin', 'reference']

SELF = slice(0, 8)
SELF_ACTION = slice(8, 11)
ENEMY = 11
ENEMY_STRIDE = 9
ENEMY_ACTION = 74
ENEMY_ACTION_STRIDE = 3
BULLET = 95
BULLET_STRIDE = 8
LIDAR = 415
LIDAR_STRIDE = 3
CLOCK = 607

ENEMY_SLOTS = 7
BULLET_SLOTS = 40
LIDAR_N = 64
AIM_N = 64
INTENT_N = 7

TOKEN = 128
TYPE_COUNT = 6
TRUNK = 512
MEMORY = 128
MAP_TOKENS = 16
LAYERS = 3
STYLE_DIM = 32

EPS = 1e-6

FIRE_FEATURE = 8 + LIDAR_N * LIDAR_STRIDE + 1


def fire_features(obs):
    """(B,201) raw-observation safety input: self state + ego-centric lidar + clock.

    A Stage 1 ablation showed the 512-d z does not make "will this shot come back at me"
    linearly separable (aim_safe_recall 0.004). This slice keeps the physical cause - the
    forward wall distance and normal - directly visible to the fire decision. It contains
    no solver output.
    """
    return torch.cat([obs[:, SELF],
                      obs[:, LIDAR:LIDAR + LIDAR_N * LIDAR_STRIDE],
                      obs[:, CLOCK:CLOCK + 1]], -1)

# Explicit adaptive_avg_pool2d windows for 11x17 -> 4x4: start=floor(i*n/out),
# end=ceil((i+1)*n/out). Written out so the ONNX graph is Slice+ReduceMean instead of
# adaptive_avg_pool2d, which the TorchScript exporter rejects for non-factor sizes
# (that operator is exactly why the V7 export failed).
_H_WINDOWS = [(0, 3), (2, 6), (5, 9), (8, 11)]
_W_WINDOWS = [(0, 5), (4, 9), (8, 13), (12, 17)]


def pool_spatial(feat):
    """(B,C,11,17) -> (B,C,4,4), numerically equal to adaptive_avg_pool2d(feat,(4,4))."""
    return torch.stack(
        [torch.stack([feat[..., h0:h1, w0:w1].mean((-2, -1)) for w0, w1 in _W_WINDOWS], -1)
         for h0, h1 in _H_WINDOWS], -2)


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.SiLU(), nn.Linear(width, width))

    def forward(self, x):
        return x + self.layers(x)


class AttentionBlock(nn.Module):
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


class MoveAdapter(nn.Module):
    """Zero-initialised movement/dodge correction on raw bullet + lidar state.

    Stage 2b-dodge freezes the Stage 1 trunk and trains only this head. Its input is the
    live bullets in the tank's body frame (relative position and velocity), the nearest
    one, the ego lidar, the clock and the seat. The seat is needed to tell own bullets
    from incoming ones: Schema V3 carries bullet owner id but not the learner's index.
    At zero it reproduces the base's drive/steer exactly.
    """

    def __init__(self):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 32))
        width = 32 + 8 + 1 + 8 + LIDAR_N * LIDAR_STRIDE + 1 + 1
        self.head = nn.Sequential(nn.Linear(width, 128), nn.SiLU(), nn.Linear(128, 5))
        nn.init.zeros_(self.head[2].weight)
        nn.init.zeros_(self.head[2].bias)

    def forward(self, obs, seat):
        b = len(obs)
        me = obs[:, SELF]
        bullet = obs[:, BULLET:BULLET + BULLET_SLOTS * BULLET_STRIDE].reshape(
            b, BULLET_SLOTS, BULLET_STRIDE)
        W, H = 1280.0, 800.0
        dx = (bullet[..., 0] - me[:, None, 0]) * W
        dy = (bullet[..., 1] - me[:, None, 1]) * H
        cos, sin = me[:, None, 2], me[:, None, 3]
        rx = (dx * cos + dy * sin) / 800.0
        ry = (-dx * sin + dy * cos) / 800.0
        vx = bullet[..., 2] * cos + bullet[..., 3] * sin
        vy = -bullet[..., 2] * sin + bullet[..., 3] * cos
        own = ((bullet[..., 6] * 7.0).round().long() == seat[:, None]).float()
        present = bullet[..., 7]
        feat = torch.stack([rx, ry, vx, vy, bullet[..., 4], bullet[..., 5], own, present], -1)
        emb = self.slot(feat)
        weight = present.unsqueeze(-1)
        mean = (emb * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        dist = torch.hypot(rx, ry) + 10.0 * (1.0 - present)
        nearest = dist.argmin(1)
        near = feat[torch.arange(b, device=obs.device), nearest]
        closest = dist.min(1, keepdim=True).values.clamp(max=1.0)
        head = torch.cat([mean, near, closest, me,
                          obs[:, LIDAR:LIDAR + LIDAR_N * LIDAR_STRIDE],
                          obs[:, CLOCK:CLOCK + 1],
                          (seat.float() / 7.0).unsqueeze(-1)], -1)
        return self.head(head)


class Policy(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = float(temperature)
        self.style = nn.Embedding(len(STYLES), STYLE_DIM)

        self.self_net = nn.Sequential(nn.Linear(8 + STYLE_DIM, 128), nn.SiLU(), nn.Linear(128, TOKEN))
        self.enemy_net = nn.Sequential(nn.Linear(ENEMY_STRIDE, 128), nn.SiLU(), nn.Linear(128, TOKEN))
        self.bullet_net = nn.Sequential(nn.Linear(BULLET_STRIDE, 128), nn.SiLU(), nn.Linear(128, TOKEN))
        self.lidar_net = nn.Sequential(nn.Linear(LIDAR_STRIDE, 64), nn.SiLU(), nn.Linear(64, TOKEN))

        self.map_conv = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.SiLU())
        self.map_vec = nn.Sequential(nn.Linear(64, 256), nn.SiLU())
        self.map_proj = nn.Linear(64, TOKEN)

        self.enemy_fuse = nn.Sequential(nn.Linear(TOKEN + MEMORY, TOKEN), nn.SiLU())

        self.slot_gru = nn.GRUCell(ENEMY_STRIDE + ENEMY_ACTION_STRIDE + 5, MEMORY)
        self.global_gru = nn.GRUCell(13, MEMORY)

        self.type_embedding = nn.Embedding(TYPE_COUNT, TOKEN)
        self.cls = nn.Parameter(torch.zeros(1, 1, TOKEN))
        self.blocks = nn.ModuleList([AttentionBlock(TOKEN) for _ in range(LAYERS)])
        self.token_norm = nn.LayerNorm(TOKEN)

        self.trunk = nn.Sequential(nn.Linear(768, TRUNK), nn.SiLU(),
                                   Residual(TRUNK), Residual(TRUNK), nn.LayerNorm(TRUNK))

        self.drive_head = nn.Linear(TRUNK, 3)
        self.steer_head = nn.Linear(TRUNK, 2)
        self.fire_head = nn.Linear(TRUNK, 1)
        # Zero-initialised: a Stage 1 checkpoint therefore behaves exactly as before.
        # Stage 2a trains it to veto self-returning shots from the raw lidar/pose.
        self.fire_adapter = nn.Sequential(nn.Linear(FIRE_FEATURE, 64), nn.SiLU(),
                                          nn.Linear(64, 1))
        nn.init.zeros_(self.fire_adapter[2].weight)
        nn.init.zeros_(self.fire_adapter[2].bias)
        self.move_adapter = MoveAdapter()

        self.aim_head = nn.Linear(TRUNK, AIM_N * 4)
        self.threat_head = nn.Linear(TRUNK, BULLET_SLOTS * 3)
        self.intent_head = nn.Linear(TRUNK, INTENT_N * 2)
        self.forward_head = nn.Sequential(nn.Linear(TRUNK + ACTION_DIM, 512), nn.SiLU(),
                                          nn.Linear(512, OBS))

    # ---------------- 输入拆分 ----------------
    def split(self, obs):
        b = len(obs)
        enemy = obs[:, ENEMY:ENEMY + ENEMY_SLOTS * ENEMY_STRIDE].reshape(b, ENEMY_SLOTS, ENEMY_STRIDE)
        enemy_action = obs[:, ENEMY_ACTION:ENEMY_ACTION + ENEMY_SLOTS * ENEMY_ACTION_STRIDE].reshape(
            b, ENEMY_SLOTS, ENEMY_ACTION_STRIDE)
        bullet = obs[:, BULLET:BULLET + BULLET_SLOTS * BULLET_STRIDE].reshape(b, BULLET_SLOTS, BULLET_STRIDE)
        lidar = obs[:, LIDAR:LIDAR + LIDAR_N * LIDAR_STRIDE].reshape(b, LIDAR_N, LIDAR_STRIDE)
        return enemy, enemy_action, bullet, lidar

    def slot_inputs(self, obs):
        """(B,7,16) memory-slot input from raw state only; no solver quantities."""
        enemy, enemy_action, _, _ = self.split(obs)
        me = obs[:, SELF]
        W, H, diagonal = 1280.0, 800.0, (1280.0 ** 2 + 800.0 ** 2) ** 0.5
        dx = (enemy[..., 0] - me[:, None, 0]) * W
        dy = (enemy[..., 1] - me[:, None, 1]) * H
        bearing = torch.atan2(dy, dx) - torch.atan2(me[:, None, 3], me[:, None, 2])
        rel = torch.stack([torch.sin(bearing), torch.cos(bearing),
                           torch.hypot(dx, dy) / diagonal, enemy[..., 4], enemy[..., 5]], -1)
        return torch.cat([enemy, enemy_action, rel], -1)

    def global_input(self, obs):
        """(B,13)."""
        alive = (obs[:, ENEMY + 6:ENEMY + ENEMY_SLOTS * ENEMY_STRIDE:ENEMY_STRIDE] > 0.5)
        return torch.cat([obs[:, SELF], obs[:, SELF_ACTION],
                          obs[:, CLOCK:CLOCK + 1],
                          alive.sum(1, keepdim=True).float() / ENEMY_SLOTS], -1)

    # ---------------- 记忆 ----------------
    def zero_memory(self, batch, device):
        return (torch.zeros(batch, ENEMY_SLOTS, MEMORY, device=device),
                torch.zeros(batch, MEMORY, device=device))

    def advance_memory(self, obs, memory):
        slot, glob = memory
        b = len(obs)
        inp = self.slot_inputs(obs)
        flat = self.slot_gru(inp.reshape(b * ENEMY_SLOTS, -1), slot.reshape(b * ENEMY_SLOTS, MEMORY))
        slot = flat.reshape(b, ENEMY_SLOTS, MEMORY)
        glob = self.global_gru(self.global_input(obs), glob)
        return slot, glob

    # ---------------- 融合 ----------------
    def encode_map(self, map_vec):
        """map_vec (B,346) -> ((B,256) summary, (B,16,128) tokens). Cache per episode.

        346 = vFlag 10x17 (170) + hFlag 11x16 (176). Pad both to 11x17 so the two wall
        families share one grid, matching the engine's convention (map.js:33-34).
        """
        b = len(map_vec)
        v = F.pad(map_vec[:, :170].reshape(b, 10, 17), (0, 0, 0, 1))
        h = F.pad(map_vec[:, 170:346].reshape(b, 11, 16), (0, 1))
        feat = self.map_conv(torch.stack([v, h], 1))
        vec = self.map_vec(feat.mean((-2, -1)))
        tokens = self.map_proj(pool_spatial(feat).flatten(2).transpose(1, 2))
        return vec, tokens

    def step(self, obs, style, map_feat, memory, seat=None):
        """One frame. Returns (z, outputs dict). memory must already be advanced."""
        b = len(obs)
        device = obs.device
        if seat is None:
            seat = torch.zeros(b, dtype=torch.long, device=device)
        map_vec, map_tokens = map_feat
        enemy, enemy_action, bullet, lidar = self.split(obs)
        slot, glob = memory

        own = self.self_net(torch.cat([obs[:, SELF], self.style(style)], -1))
        enemy_emb = self.enemy_fuse(torch.cat([self.enemy_net(enemy), slot], -1))
        bullet_emb = self.bullet_net(bullet)
        lidar_emb = self.lidar_net(lidar)

        tokens = torch.cat([
            self.cls.expand(b, -1, -1),
            map_tokens,
            lidar_emb,
            own.unsqueeze(1),
            enemy_emb,
            bullet_emb,
        ], 1)
        types = torch.cat([
            torch.zeros(b, 1, dtype=torch.long, device=device),
            torch.ones(b, MAP_TOKENS, dtype=torch.long, device=device),
            torch.full((b, LIDAR_N), 2, dtype=torch.long, device=device),
            torch.full((b, 1), 3, dtype=torch.long, device=device),
            torch.full((b, ENEMY_SLOTS), 4, dtype=torch.long, device=device),
            torch.full((b, BULLET_SLOTS), 5, dtype=torch.long, device=device),
        ], 1)
        tokens = tokens + self.type_embedding(types)

        padding = torch.cat([
            torch.zeros(b, 1 + MAP_TOKENS + LIDAR_N + 1, dtype=torch.bool, device=device),
            enemy[..., 7] <= 0,
            bullet[..., 7] <= 0,
        ], 1)
        for block in self.blocks:
            tokens = block(tokens, padding)
        tokens = self.token_norm(tokens)
        present = (~padding[:, 1:]).unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens[:, 1:] * present).sum(1) / present.sum(1).clamp_min(1.0)

        z = self.trunk(torch.cat([own, map_vec, tokens[:, 0], pooled, glob], -1))

        steer = self.steer_head(z)
        move = self.move_adapter(obs, seat)
        out = dict(
            z=z,
            drive=self.drive_head(z) + move[:, :3],
            steer_mu=torch.tanh(steer[:, :1] + move[:, 3:4]),
            steer_logstd=(steer[:, 1:2] + move[:, 4:5]).clamp(-4.0, 1.0),
            fire=(self.fire_head(z) + self.fire_adapter(fire_features(obs))).squeeze(-1),
            aim=self.aim_head(z).reshape(b, AIM_N, 4),
            threat=self.threat_head(z).reshape(b, BULLET_SLOTS, 3),
            intent=self.intent_head(z).reshape(b, INTENT_N, 2),
        )
        return out

    # ---------------- 动作分布 ----------------
    @staticmethod
    def _squashed_logp(mu, logstd, u):
        """log N(tanh^-1(a); mu, sigma) + log|da/dx| for a = tanh(x)."""
        std = logstd.exp()
        x = torch.atanh(u.clamp(-1 + EPS, 1 - EPS))
        logp = -0.5 * ((x - mu) / std) ** 2 - logstd - 0.5 * torch.log(torch.tensor(2 * torch.pi, device=u.device))
        return logp - torch.log(1 - u * u + EPS)

    def act(self, obs, style, map_feat, memory, deterministic=False, uniforms=None, seat=None):
        """Sample (or take) an action. Returns (controls (B,5), logp, entropy, parts)."""
        out = self.step(obs, style, map_feat, memory, seat)
        temp = self.temperature
        drive_logits = out['drive'] / temp
        fire_logit = out['fire'] / temp
        mu, logstd = out['steer_mu'], out['steer_logstd']

        if uniforms is None:
            uniforms = torch.rand(len(obs), 3, device=obs.device)
        elif uniforms.dim() == 1:
            # the frozen Evaluator supplies one uniform per row; derive three deterministic ones
            u = uniforms
            frac = lambda t: t - t.floor()
            uniforms = torch.stack([u, frac(u * 7.3129), frac(u * 13.7731)], -1)
        u_d, u_s, u_f = uniforms[:, 0], uniforms[:, 1], uniforms[:, 2]

        if deterministic:
            drive = drive_logits.argmax(-1)
            steer = mu.squeeze(-1)
            fire = (fire_logit > 0).float()
        else:
            cdf = drive_logits.softmax(-1).cumsum(-1)
            drive = torch.searchsorted(cdf.contiguous(),
                                       (u_d.clamp(0, 1 - EPS))[:, None].contiguous(),
                                       right=True).squeeze(1).clamp(0, 2)
            dist = Normal(mu.squeeze(-1), logstd.exp().squeeze(-1))
            steer = torch.tanh(dist.icdf(u_s.clamp(EPS, 1 - EPS)))
            fire = (u_f < fire_logit.sigmoid()).float()

        logp = (Categorical(logits=drive_logits).log_prob(drive)
                + self._squashed_logp(mu.squeeze(-1), logstd.squeeze(-1), steer)
                - torch.nn.functional.binary_cross_entropy_with_logits(
                    fire_logit, fire, reduction='none'))
        entropy = (Categorical(logits=drive_logits).entropy()
                   + Normal(mu.squeeze(-1), logstd.exp().squeeze(-1)).entropy()
                   + torch.distributions.Bernoulli(logits=fire_logit).entropy())

        controls = torch.stack([drive.float() - 1.0, steer, fire], -1)
        five = torch.cat([controls, out['steer_mu'], fire_logit.unsqueeze(-1)], -1)
        return controls, logp, entropy, five


def save_policy(model, path, **metadata):
    from pathlib import Path
    path = Path(path)
    torch.save(dict(architecture='perception_factored_memory_v8', temperature=model.temperature,
                    model=model.state_dict(), observation='schema_v3', round_limit=60,
                    **metadata), path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)


def load_policy(path, device='cpu'):
    meta = torch.load(path, map_location='cpu', weights_only=False)
    if meta.get('architecture') != 'perception_factored_memory_v8':
        raise ValueError('Incompatible checkpoint')
    model = Policy(meta.get('temperature', 1.0)).to(device)
    missing, unexpected = model.load_state_dict(meta['model'], strict=False)
    if unexpected or any(not key.startswith(('fire_adapter.', 'move_adapter.')) for key in missing):
        raise ValueError('Incompatible checkpoint (missing %s, unexpected %s)'
                         % (list(missing), list(unexpected)))
    return model, meta
