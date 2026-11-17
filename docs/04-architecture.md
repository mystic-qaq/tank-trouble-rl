# 04 · Architecture

This note is about what the networks actually compute, not about the training loop. It covers the V6 relational encoder, the V7 aim-distillation head, and the V8 design that follows from their failures.

The through-line: **the input representation is a hypothesis about what the agent needs to know, and every version moved more "solver knowledge" out of the input and into training-time labels.**

---

## 1. The observation, in one table

V1–V7 share a 1,966-float per-frame observation. It is a structured state, not pixels:

| Slice | Dims | Contents |
|---|---:|---|
| Self | 80 | x, y, sin/cos, speed, cooldown, alive, ammo, last action, 8-frame action history, angular velocity, a few engine constants |
| Map | 374 | 2×11×17 wall bitmaps (vertical + horizontal boundaries) |
| Enemies | 168 | 7 slots × 24 (relative position, sin/cos, speed, cooldown, alive, presence, ammo, …) |
| Bullets | 960 | 40 slots × 24 (relative position, direction, life, bounces, owner, presence, …) |
| Aim candidates | 384 | 32 candidates × 12 (angle, hit, self-hit, time-to-hit, cone, …) |

The map slice is reshaped to `(2, 11, 17)` and passed through two 3×3 convolutions. Enemy, bullet and aim slots are each encoded by a shared small MLP with a presence mask.

**Why this matters.** The aim-candidate block is a solver output: `policy/rl/v2/geometry.mjs` traces multi-bounce trajectories and hands the network 32 pre-computed options. That makes learning easier, but it also means the network can read the answer instead of learning the geometry. V7 and V8 are attempts to remove that crutch without losing the information.

---

## 2. V6: 96 tokens and a conditional fire head

### 2.1 Tokenisation

| Token group | Count | Width | Source |
|---|---:|---:|---|
| Map | 16 | 128 | Conv features adaptive-pooled to 4×4, flattened |
| Enemies | 7 | 128 | Shared MLP on 24 dims |
| Bullets | 40 | 128 | Shared MLP on 24 dims |
| Aim candidates | 32 | 128 | Shared MLP on 12 dims |
| CLS | 1 | 128 | Learned |

Total: **96 tokens**. A type embedding (5 types) is added, and a padding mask marks absent enemies/bullets/aim candidates. Map tokens and CLS are never masked, so no row is fully masked.

### 2.2 Encoder

```text
tokens = [CLS] + 16 map + 7 enemy + 40 bullet + 32 aim
for 2 blocks:
    x = x + MHA(LN(x), key_padding_mask=padding)
    x = x + FFN(LN(x))
x = LN(x)
pooled = masked_mean(x[1:])
z = Linear([own_256, map_vec_256, CLS_128, pooled_128] = 768 -> 512)
z = Residual(512) -> Residual(512) -> LN
```

`own` comes from a separate self MLP (`80 + 32 style -> 256`); `map_vec` is the global average pool of the map conv features. The style embedding is a 9-class condition used to keep training and evaluation consistent (the final runs use `gpt` for both).

### 2.3 Action heads

The action is **51 motion classes × 2 fire states** conceptually, but it is never materialised as a 102-way softmax:

```text
motion_head:        Linear(512 -> 51)
fire_global:        Linear(512 -> 1)
fire_conditional:   Linear(512 + 51 -> 256) -> SiLU -> Linear(256 -> 51)

logit(m, f) = ( motion[m] + f * (fire_global + fire_conditional[m]) ) / T
```

- `motion[m]` is the navigation logit.
- `fire_conditional[m]` gives each manoeuvre its own fire marginal, so `P(fire | motion)` is not forced to be constant.
- `fire_global` keeps the rare fire class well conditioned: it receives gradient from every motion, so the fire signal is not diluted across 51 classes.

At deployment, `f=0` and `f=1` are compared per motion; the policy picks the best `(m, f)` pair. Invalid fire during cooldown is masked (`x[:, 16:17] < .5`), consistently in BC, PPO and export.

**Parameters:** ~2.1M, down from V5's 3.6M because the large `Linear(32*11*17 -> 256)` layer is replaced by a 3×3 conv + 4×4 pooling.

### 2.4 Why it did not win

The architecture gave the network better relational features, and the shared PPO stage did reach +0.203 net kills with 0.031 suicides at iteration 47. But the independent dev probe was 7.74% win. Better representation does not fix a reward that is satisfied by not firing. That is why V7 changed the supervision, not just the backbone.

---

## 3. V7: distil the aim servo

V7 keeps the 1,966-D observation and the V6-style encoder, but adds a dense analytic teacher for aiming. The exported model is still sensor-only.

### 3.1 Labels

For each frame:

1. Take 64 directions evenly spaced around the tank: `φ_i = θ + i·2π/64`.
2. Call `traceShot(game, me, φ_i)` for each direction. It returns whether the ray hits an enemy, whether it hits self, and the time to hit (up to 6 s).
3. From the exact candidate set (`aimCandidates()` with `intercept()` lead and mirror-bounce solutions), compute the best **safe** direction and its angular error.
4. Produce:
   - `hit_enemy[64]`, `hit_self[64]`, `time_to_hit[64]`;
   - `sin(err_best)`, `cos(err_best)`, `has_safe`, `cone_best / π`.

The 64-direction basis is **uniform and self-centred**, not the solver's candidate set. The network has to learn which direction is good; it is not handed a ranked list. The exact servo target is only a label.

### 3.2 Servo target

```text
steer* = clamp(err_best / (3.3 * dt), -1, 1)   if has_safe
drive* = teacher drive
fire*  = (ready ∧ has_safe ∧ |err_best| ≤ cone_best) ∨ teacher fire
```

At 60 Hz, `3.3 * dt ≈ 0.055 rad ≈ 3.15°` per frame. A continuous steer head can express the exact correction; a 17-bin discrete head cannot.

### 3.3 Export

The V7 ONNX export is a static batch-1 graph, opset 17. It required two substitutions that do not change the function:

- `adaptive_avg_pool2d` → an explicit windowed mean (ONNX has no adaptive pool);
- `atan2(0, 0)` guarded with `torch.where(y == 0 & x == 0, 0, atan2(y, x))` at export time only.

Verification on 6,004 real frames + 4 constructed edge cases:

| Check | Result |
|---|---|
| Drive / turn / fire / argmax agreement | 1.000 |
| Max control error | 0.0 |
| Max logit error | 3.7e-5 |
| Mismatched frames | 0 |
| Outputs finite | yes |

This matters because the browser arena runs the same weights through ONNX Runtime Web. A model that only works in PyTorch is not a deployed policy.

---

## 4. V8: sensor-only schema and solver-as-oracle

V8 is a design contract, partially implemented. It is the logical conclusion of the V7 lesson: if analytic labels help, then remove the solver from the input entirely and keep it only as a teacher.

### 4.1 Input (608 per frame + 346 per round)

| Block | Dims | Contents |
|---|---:|---|
| Self | 8 | x, y, sinθ, cosθ, speed, cooldown, alive, ammo |
| Last action | 3 | drive, steer, shoot |
| Opponents | 63 | 7 × 9 state fields |
| Opponent actions | 21 | 7 × 3, recovered from adjacent-frame differences |
| Bullets | 320 | 40 × 8 |
| Lidar | 192 | 64 rays × (distance, wall normal cos/sin) |
| Clock | 1 | `1 − roundTime/60` |
| **Per frame** | **608** | |
| Walls (round-level) | 346 | 170 vertical + 176 horizontal flags, encoded once and cached |

Opponent actions are recovered without touching the engine:

```text
steer_k = clamp(wrap(θ_k,t − θ_k,t−1) / (3.3·dt), −1, 1)
drive_k = sign(Δspeed_k) if |Δspeed_k| > 0.5·accel·dt else previous
shoot_k = (cooldown_k,t−1 > 0) ∧ (cooldown_k,t ≈ 0.42)
```

Using the cooldown jump rather than "a new bullet appeared" also captures a dry-fire attempt, because the engine sets the cooldown before the muzzle-blocked check.

### 4.2 Labels (333, training only)

| Label | Dims | Source |
|---|---:|---|
| Aim | 196 | 64 directions × 3 + 4 servo targets |
| Threat | 120 | 40 bullets × (ttc, miss distance, will-hit-me) |
| Intent | 14 | 7 opponents × (bearing rate, tracking flag) |
| Teacher action | 3 | drive, steer, shoot |
| **Total** | **333** | |

The labels are stored in the corpus but never fed to the policy. The inference graph contains no `traceShot`, `intercept`, `flight`, `pathsFor` or `closest`.

### 4.3 Model

```text
map 346 -> (2,11,17) -> Conv3x3(2->32) -> Conv3x3(32->64)
    -> global pool -> map_vec 256
    -> spatial flatten -> 16 map tokens

per frame:
  lidar 192 -> Linear(3->64) -> Linear(64->128)   64 tokens
  self 11   -> Linear(11->128)                      1 token
  bullets 40 -> shared Linear(8->128)              40 tokens
  opponents 7 -> GRU memory m_k (shared weights)    7 tokens
  + type embeddings, padding mask by presence

tokens = [CLS] + 16 map + 64 lidar + 1 self + 7 opponents + 40 bullets = 129
3 × Pre-LN self-attention(128, 4 heads)
-> CLS + masked mean -> concat with own/map/global memory
-> 768 -> 512 trunk (2 residual blocks)

heads (inference): drive 3, steer μ/σ 2, fire 1
heads (training only): aim 64×3, threat 40×3, intent 7×2, forward 608
```

Three design choices deserve emphasis:

1. **Per-opponent GRU slots, shared weights.** A single global GRU mixes all opponents into one hidden state and cannot answer "is HY tracking me, or is GPT?" Seven slots with shared weights keep the memory addressable per opponent and preserve seat permutation equivariance.
2. **Action factorisation.** Drive is categorical (3), steer is a tanh-squashed Normal, fire is Bernoulli. The joint log-probability is the sum of three independent terms, so exploration can be tempered on fire only. The 102-class joint action of V4/V5 made this impossible.
3. **Forward-dynamics auxiliary head.** Given `(z, a)`, predict the next 608-D state block. This is a representation test, not a world model for imagination: the simulator is exact and cheap enough that imagined rollouts would be strictly worse.

### 4.4 What V8 does not claim

V8's Stage-1 exit criteria were not met. The safety-bypass experiment (freeze the base policy, train a zero-initialised lidar veto on the fire logit) is preserved in `policy/rl/v8/` but is not a headline result. V8 is included as a design argument, not as evidence that the design works.

---

## 5. What was borrowed from VLA-style systems

This is not a vision-language-action model: there is no camera, no language instruction, and no action chunking. But several design patterns were borrowed from modern robot policies:

| Pattern | How it appears here | Why |
|---|---|---|
| Privileged teacher → sensor-only student | Analytic solver labels in V7/V8; solver absent from inference | Get dense geometry supervision without leaking the answer at test time |
| Factored action heads | 51+1 in V5/V6; 5-D in V8 | Rare events (fire) should not compete with dense navigation |
| Auxiliary grounding heads | Aim, threat, intent, forward dynamics (V8) | Force the trunk to represent quantities the action head needs |
| Per-entity memory | 7 opponent GRUs + 1 global GRU (V8) | Multi-agent intent is indexed by opponent, not pooled |
| Strict inference-graph separation | Export checks, no solver in ONNX | A policy that only works with privileged features is not deployable |
| Action-space continuity | Continuous steer via tanh Normal (V8) | A 3.15° cone cannot be expressed with 17 discrete bins |

The difference from a real VLA system is equally important: the "language" and "vision" halves are missing, and the environment is a deterministic simulator with an exact solver. That makes privileged distillation unusually cheap here — and it is exactly why the remaining failure is strategic rather than perceptual.

---

## 6. Version comparison

| | V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8 |
|---|---|---|---|---|---|---|---|---|
| Observation | 926 | 926 | 926 | 1,966 | 1,966 | 1,966 | 1,966 | 608 + 346 |
| Action | 3 heads | 3 heads | joint 9 × steer + fire | 102 joint | 51 + 1 | 51 + 1 conditional | 51 + 1 + servo | 3 + continuous + 1 |
| Encoder | MLP | MLP | CNN + attention + GRU | MLP | CNN + pooling | 96-token attention | V6 + aim head | 129-token attention + GRU |
| Params | 159k | 159k | 4.6M | 3.6M | 3.6M | 2.1M | ~2.1M | ~4.5M |
| Solver in input? | partial | partial | yes | yes | yes | yes | labels only | no |

The trend is deliberate: as the project learned more, the input got **smaller** and the training signal got **denser**. That is the opposite of the usual "add more features" instinct, and it is the most transferable architectural lesson from the project.
