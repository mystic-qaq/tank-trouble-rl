# 03 · Learning to play: V1–V8

This is the main research log. It records what each generation tried, what the numbers were, and why it failed. The short version: the project produced a real, reproducible improvement from 1.19% (random) to 16.4% (V7), but never reached the 21.1% acceptance gate, and the reasons are more instructive than the score.

---

## 1. The evaluation contract

All generations share one contract, inherited from the original pipeline:

| Item | Value |
|---|---|
| Development split | 140 episodes, `dev_blocks=2`, frozen seed 20261107 |
| Player counts | n = 2…8, equally represented |
| Aggregation | macro-average over n (not row-average) |
| Decoding | `min(greedy, sampled)` must pass; sampled uses the training distribution |
| Win gate | ≥ 70% of teacher win rate → **0.2113** |
| Suicide ceiling | teacher suicides + 0.03 → **0.1530** |
| Teacher | HY for win; GPT for net kills |
| Full gate | 2,240 episodes (`gate_blocks=32`) for the formal acceptance |

Two rules were never bent:

1. **Do not tune temperature after training to pass the gate.** The sampled decoder is the same distribution PPO trained on.
2. **Do not lower the 70% threshold.** If a run failed, it failed.

The measured uniform-random baseline is **1.19% win**, not the analytic 20%. The analytic number assumes every episode is decided; in practice 8-player rounds have many draws and a random policy kills itself constantly (suicide rate 0.926). Using the wrong baseline makes a 7.7% policy look "worse than random" and sends the next iteration in the wrong direction.

---

## 2. Data: 80 GB of demonstrations

The corpus was collected from the real JS engine, not from a Python approximation.

| Item | Value |
|---|---:|
| Worlds | 6,000 |
| Trajectories | 9,319 multi-agent + 1,200 skill |
| Frames | 7,614,581 |
| Size | ~80.4 GB (75 GB multi + 995 MB skills) |
| Collection time | 1,337.6 s (~22 min) |
| Throughput | ~5,692 frames/s |
| Storage | local disk, not NFS |

Each trajectory stores a continuous frame sequence (not sampled frames), because later designs need adjacent-frame differences for action recovery and forward-dynamics targets. The data receipt is a per-file SHA-256 manifest; `G0` recomputes it before every run and fails closed on a mismatch.

The observation schema evolved across versions:

- **V1–V5**: 1,966 floats. Includes a wall grid, self state, up to 7 enemies, 40 bullets, 32 aim candidates, and 24 dims of action history. Some fields were solver outputs (privileged features), which later became a design problem.
- **V6**: same 1,966 input, different processing (96 tokens, attention).
- **V8**: a new sensor-only schema, 608 per-frame + 346 per-round, with all solver outputs moved to training labels.

The corpus is not shipped with the repository. `policy/rl/v3/collect.py` and `policy/rl/v8/collect.py` are the collectors; `research/` contains the receipts and scripts.

---

## 3. V1 — behaviour cloning, then PPO

**Hypothesis.** A small MLP can imitate the HY/GPT policies from single-frame observations; PPO can then improve on the imitation.

**Setup.** 65,536 samples per objective, 30 BC epochs, then ~1M PPO frames. Observation 926 dims, three heads (drive / steer / shoot), 159k parameters.

**Result.**

| Metric | Win model | Frag model |
|---|---:|---:|
| Motion accuracy | 92.5% | 95.6% |
| Steer MAE | 0.140 | 0.309 |
| Fire recall at 0.5 | **0%** | **0%** |
| Dev win (28 rounds) | 3.6% | 3.6% |
| Dev net kills | 0 | 0 |

After 524,288 PPO frames, the win model was still at 3.6% with zero kills; the frag model reached 7.1% and still zero kills. The teacher references on the same 28 rounds were HY 39.3% and GPT 14.3%.

**Diagnosis.** BC learned to drive and never learned to shoot. The dataset contained only 354/597 fire positives, and the weighted binary cross-entropy put every fire probability below the 0.5 decision threshold. The frag model converged to "never shoot": with rare positive rewards and frequent self-kill negatives, lowering the fire probability was the fastest way to improve return.

**Lesson.** A gate on offline action accuracy is not a gate on behaviour. A policy can be 92% accurate on the dominant class and functionally dead on the class that matters.

---

## 4. V2 — DAgger and micro-skills

**Hypothesis.** If the problem is distribution shift and rare fire events, add student-state labels (DAgger) and explicit skill scenarios (static target, moving target, bank shot, dodge, volley survival, wall safety).

**Setup.** 2,655 HY worlds / 407k samples / 20,319 real shots; 3,056 GPT worlds / 207k samples / 20,335 shots; 2,400 skill worlds / 26k samples / 4,696 shots; 118 DAgger trajectories per candidate; four candidates (two objectives × two seeds/weights).

**Result.** Skills improved substantially:

| Candidate | Static | Moving | Bank | Dodge | Volley | Wall |
|---|---:|---:|---:|---:|---:|---:|
| Win A | 30→80% | 80→96% | 38→92% | 100% | 98% | 100% |
| Win B | 24→82% | 66→96% | 32→92% | 100% | 100% | 100% |
| Frag A | 2→56% | 64→74% | 4→82% | 100% | 100% | 100% |
| Frag B | 12→52% | 82→78% | 12→82% | 100% | 100% | 100% |

But the teacher-fire imitation metrics were still far from the 70% precision / 80% recall gate:

| Candidate | Fire precision | Fire recall |
|---|---:|---:|
| Win A | 37.9% | 3.5% |
| Win B | 28.0% | 13.2% |
| Frag A | 46.6% | 17.3% |
| Frag B | 43.1% | 30.8% |

**Diagnosis.** DAgger fixed local skills but not the decision of *when* to fire in a full game. The skill scenarios were too clean: a static target with a clear line is not the same state as a melee with three bullets in flight. PPO never started because the readiness gates failed.

**Lesson.** Micro-skills are necessary but not sufficient. The gap is in state distribution, not in the ability to execute a known shot.

---

## 5. V3 — recurrent pretraining and three DAgger rounds

**Hypothesis.** A larger recurrent model with a shared pretraining stage and iterative DAgger can close the gap that V2's feed-forward model could not.

**Setup.** 4.6M parameters: map CNN, entity attention, 256-D GRU, joint nine-way "throttle × turn direction" plus continuous steering and conditional fire. 30,000 pretraining steps on 4 GPUs, then three DAgger rounds, then PPO on two branches.

**Result.** After 70 development episodes, the shared model won **9.76%** with **+0.028 net kills**, fired only **6 times** and killed **3 times**. The health review also found that 70–71% of three-second windows were "moving forward, not requesting fire" in the first DAgger round.

The run was stopped at `stopped_pending_readiness_review`. PPO did not start.

**Diagnosis.** The recurrent model had more capacity but the same behavioural problem: the policy preferred not to shoot. The readiness review also caught a checkpoint-selection bug (the last DAgger round was used even when an earlier model was better) and a CPU/CUDA RNG restore bug. Both were fixed, but the policy was still not ready.

**Lesson.** More parameters do not fix a missing decision rule. The model needs a training signal that makes firing worthwhile, not just a bigger function approximator.

---

## 6. V4 — the 102-class joint head failure

**Hypothesis.** Treat the action as a single 102-way choice (2 drive directions × 51 steer/fire combinations) and scale to 4 GPUs.

**Result.** G1 stopped after 30,000 of 40,000 steps. On six evaluations, the win model predicted **zero fire actions**; the frag model's best fire recall was 0.136, with motion accuracy 0.43/0.55 against a 0.85 gate.

**Root cause.** Three independent bugs compounded:

1. **Inverse-probability weighting.** `data.py` multiplied each frame by the inverse probability of its action. The natural fire rate was 0.601%, so fire frames were suppressed by **199–1,000×**. The model was explicitly trained not to fire.
2. **A contract test encoded the bug.** `test_data.py` used a toy fixture with 5% fire and asserted that weighted fire frequency must equal the natural frequency. It certified the wrong invariant.
3. **Temperature mismatch.** BC used raw logits; rollout and PPO divided by `temperature=0.35`. The same checkpoint behaved differently in training and deployment.

There were also systems issues: 4× `DataParallel` on a 14 MB model split a batch of 512 into four tiny slices, and random reads of 32 small NFS files per step kept GPUs at 8–11% utilisation.

**Lesson.** A rare class is a data-pipeline problem before it is a model problem. And a test can be worse than no test if it locks in the wrong assumption.

---

## 7. V5 — factorise, then optimise the real objective

**Hypothesis.** Fix the action representation (51 motion logits + 1 fire logit), fix the data weighting, and stop treating frame-level imitation of a deterministic MPC as the goal. Optimise the game's real objective with PPO, using BC only as a prior.

**Changes.**

- 51 + 1 factorised head; all stages use the same temperature-scaled logits.
- Delete inverse-probability weighting; use 15% fire oversampling and bounded class weights.
- Single GPU, batch 2048, local data: **10 → 17–22 steps/s** (~7× sample throughput).
- New `optimize.py`: PPO-like clipped surrogate on real rewards, GAE, exact mean-KL, `max_kl=0.05` rollback, separate acceptance budget.
- Curriculum: opponents start weak (built-in, minimax, Kimi, Qwen), n grows from 2 to 8; the gate always uses the full-strength frozen split.
- `best.pt` only from full-strength iterations (`certified = iteration > ramp`).

**Result after switching objectives.** BC prior was still weak (win 5.1%/6.3%, net kills −0.04/−0.65, suicides 0.35/0.76). Direct PPO with curriculum improved frag substantially:

| Run | Iterations | Best | Final dev greedy | Final dev sampled | Gate |
|---|---:|---:|---:|---:|---:|
| win | 80 | 65 | 4.38% | 4.76% | 21.13% |
| frag | 66 | 41 | **+0.358** | +0.158 | +0.591 |

The frag model's greedy suicide rate fell to **0.007**, the first time any policy passed the suicide ceiling. Its net kills reached 42% of the teacher. The win model did not move: survival shaping taught it not to die, not how to win.

Teacher-guided KL was tested twice and failed both times: advantage-weighted guidance blew KL to 0.21; uniform guidance was rejected by the trust region on all 14 iterations.

**Lesson.** Representation and reward both matter, but they fix different things. Factorisation made fire learnable; the curriculum made winning reachable; neither made the policy aggressive enough to beat the teacher.

---

## 8. V6 — relational attention and conditional fire

**Hypothesis.** The remaining bottleneck is relational reasoning: bullets, enemies and aim candidates must interact, and fire probability must depend on the manoeuvre. Replace three independent pooling heads with a shared token space.

**Architecture.** 96 tokens (16 map + 7 enemy + 40 bullet + 32 aim + CLS), 2-layer Pre-LN self-attention (128-D, 4 heads), conditional fire head, ~2.1M parameters. See [04 · Architecture](04-architecture.md).

**Run.** A shared frag-objective stage on one GPU, then fork into win/frag. The first attempt failed outright: 24 iterations all negative (best −0.4375, suicides 0.50), and the run gate passed the bad checkpoint to the forks because it only checked "process finished", not quality. That was a harness design error, not a learning result.

The rerun (`shared2`, 4,200 s, 68 iterations) produced a genuine local peak:

| Metric | Iter 47 (saved) | Final iter 68 |
|---|---:|---:|
| netKills | **+0.203** | −0.203 |
| Kills/round | 0.234 | 0.125 |
| Suicides | **0.031** | 0.328 |
| KL | 0.0089 | 0.0098 |

But the independent dev probe told a different story: **7.74% win** (greedy), +0.004 net kills, 0.038 suicides. The checkpoint had learned to avoid self-kills and occasionally clean up, not to win rounds.

The next stage (V6C) switched the shared objective to `win` with survival + kill-aux shaping, and staged a gate at 0.1509. It was ready to start when the operator stopped the run to reconsider the reward design. It never executed.

**Lesson.** A good-looking training curve can be a local optimum of the wrong objective. The dev probe, not the training score, is the source of truth.

---

## 9. V7 — distill an analytic aim servo

**Hypothesis.** The network does not need to *discover* ricochet geometry from scratch if we can give it a dense, analytic teacher for aiming. Distil the geometry into labels, keep the inference graph sensor-only.

**Method.**

- For each frame, cast 64 evenly spaced rays from the tank through `traceShot`.
- Label each direction with `hit_enemy`, `hit_self`, and `time_to_hit`.
- From the exact candidate set, compute the best safe direction and its angular error, plus a `has_safe` flag and cone width.
- Train the policy to predict these labels as auxiliary heads, and to output a servo action: `steer = err / (turn × dt)`, `fire = ready ∧ has_safe ∧ |err| ≤ cone`.
- The solver is used **only at training time**. The exported ONNX graph contains no `traceShot`, no `atan2` search, no candidate set.

**Result.**

| Checkpoint | Greedy win | Sampled win | Net kills | Suicides | Draws |
|---|---:|---:|---:|---:|---:|
| V6 iter47 (control) | 7.74% | 14.92% | +0.004 | 0.038 | 22.1% |
| V7 distill1 final | **16.41%** | 9.50% | **+0.430** | 0.183 | 24.3% |
| V7 distill3 best | 13.76% | 11.34% | +0.312 | 0.167 | 25.7% |
| HY teacher | 30.19% | — | +0.653 | 0.123 | 17.9% |

V7 is the best learned policy in the project: it more than doubled V6's greedy win rate, reached 66% of the teacher's net kills, and stayed within the same order of magnitude on suicides. It still failed the gate (16.41% < 21.13%).

The per-n breakdown is the most useful diagnostic:

| n | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V7 win | **62.5%** | 8.3% | 12.5% | 15.0% | **0.0%** | 7.1% | 9.4% |
| Teacher win | 50.0% | 41.7% | 31.3% | 25.0% | 12.5% | 32.1% | 18.8% |

It is a **duel specialist**: it beats the teacher in 2-player rounds and collapses in 6-player melees. It also draws more (24.3% vs 17.9%) and wins only 15.1% of decided rounds (teacher 32.2%). The policy learned to survive and to aim, but not to create and finish a winning opportunity in a crowded map.

**Lesson.** Analytic labels can teach geometry efficiently. They cannot teach melee strategy. The next bottleneck is multi-agent intent and tempo, not ricochet math.

---

## 10. V8 — sensor-only schema and solver-as-oracle

V8 is a design contract plus a partial implementation; it is not a headline result. Its three axioms are a direct response to the failures above:

1. **The solver is a label oracle, not an input.** Inputs contain only state reads and O(1) sensor queries. `traceShot`, `intercept`, `flight`, `pathsFor` and `closest` remain in the training loss, never in the inference graph.
2. **Factorise actions.** 5-D action: categorical drive (3), tanh-squashed steer (continuous), Bernoulli fire. Three independent distributions, so fire exploration can be tempered without randomising navigation.
3. **Prove the representation can predict the next state.** A forward-dynamics auxiliary head predicts the next 608-D state block from `(z, a)`.

Other choices:

- **608-D per-frame input**: self 8 + last action 3 + 7×9 opponents + 7×3 recovered opponent actions + 40×8 bullets + 64×3 lidar + clock.
- **346-D per-round input**: wall bitmaps, encoded once and cached.
- **333-D labels**: 64-direction aim labels, 40-bullet threat labels, 7-opponent intent labels, teacher actions.
- **129-token attention** with 5 type embeddings, plus **per-opponent GRU memory** (7 slots, shared weights) and a global GRU.
- **4 auxiliary heads** (aim, threat, intent, forward), all removed at export.
- **~4.5M parameters**, inference target ≤ 4 ms/frame.

The Stage-1 exit criteria (aim AUC ≥ 0.90, steer error ≤ one cone, fire precision ≥ 0.85 / recall ≥ 0.65) were not met. The team then tried a narrower Stage-2a experiment: freeze the base policy and train a zero-initialised safety bypass on self-state + lidar to veto self-killing shots. That experiment is preserved in `policy/rl/v8/` but is not presented as a solved result.

**Why include it?** Because it is the honest next step: the V7 result says the remaining gap is not geometry, so V8 moves the solver out of the input path and gives the model memory and intent labels. It also shows where the project would go with more compute and time.

---

## 11. Systems notes

These are not incidental. Several "learning failures" were systems failures first.

| Issue | Symptom | Fix | Effect |
|---|---|---|---|
| NFS hard mount `a100:/data` hung at 06:00 | 60 processes in D state, SSH/VSCode unusable | Copy 80 GB to local disk; never train from NFS | Removed a 6-hour class of failures |
| `DataParallel` on a 14 MB model, 4 GPUs | 550–644 MiB per GPU, 8–11% util, ~10 steps/s | Single GPU, batch 2048 | 17–22 steps/s, ~7× sample throughput |
| CPU simulation vs GPU update | 60–77 s collect vs 5–11 s update; GPU util 12–15% | More workers, smaller model, accept the ratio; do not chase GPU utilisation | Stable throughput; GPU is not the bottleneck |
| Rollout memory growth | None after 68 iterations | Records are per-iteration locals; `del` after update; no disk writes; mmap data + LRU cache | ~5.5 GB resident, flat over 12 h |
| Logs on NFS | 0-byte `v4-main.log` | Local logs, `status.json`, TensorBoard | Observable runs |

---

## 12. What we would do differently

1. **Start with factored actions and a measured random baseline.** Both were cheap and both would have saved a generation.
2. **Separate "can execute" from "can decide".** Skill tests are useful, but they must be paired with full-game behaviour probes; otherwise a 92% motion accuracy hides a 0% fire policy.
3. **Treat the reward as a hypothesis, not a definition.** `netKills` is the game's score, and it still produced a passive policy. The next reward should include a tempo/engagement term or an explicit draw penalty.
4. **Evaluate per player count from day one.** A policy that wins duels and loses melees is not a good policy; the aggregate number hides it.
5. **Use the solver as a teacher, not as an input.** V7 is the evidence: dense analytic labels helped; privileged features mostly taught the network to read the answer.
6. **Keep the gate small but honest.** 140 episodes is enough to reject a bad policy and cheap enough to run often; the full 2,240-episode gate is for the final claim, not for every iteration.

The negative result is not "RL does not work on Tank Trouble". It is narrower and more useful: **a sensor-only 2M-parameter policy can learn ricochet aiming and survival from analytic labels, but not yet melee strategy, within the compute and reward design we had.** That is the claim this repository supports.
