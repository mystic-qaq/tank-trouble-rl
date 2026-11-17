# 02 · Seven agents, one arena

The arena answers a narrow question: **if seven different LLMs write a policy against the same interface, how do they compare when the rules, maps, spawns and seats are held constant?**

It is not a benchmark of the models that wrote the policies. It is a benchmark of seven concrete programs, frozen at a point in time, under a harness designed to remove the easy ways to win: privileged state access, seat advantage, cross-round state, and inconsistent denominators.

---

## 1. The interface

Every policy implements the same signature as the engine's built-in bot:

```js
updateBot(tank, game, dt) -> { drive: -1 | 0 | 1, steer: number, shoot: boolean }
```

The arena takes over `TT.updateBot` and dispatches to the policy registered for each seat. It normalises the return value, but deliberately does **not** clamp `steer`: three policies use fractional steering to hit an exact angle in one frame, and clamping would destroy their aim. The physical turn rate itself is still enforced by the engine.

Errors are swallowed and counted. A policy that throws returns `{drive: 0, steer: 0, shoot: false}` for that frame instead of crashing the round.

---

## 2. Loading seven programs without editing them

The policies were written independently and do not share a module format:

| Policy | Format | Loading strategy | Entry point |
|---|---|---|---|
| DeepSeek | classic script | `vm.runInContext` | `TT.deepseekPolicy` |
| GLM | classic script | `vm.runInContext` | `TT.Policies.glm.update` |
| GPT | classic script | `vm.runInContext` | `TT.policies.gpt` |
| HY | classic script | `vm.runInContext` | `TT.Policies.hy` |
| Qwen | classic script | `vm.runInContext` | `TT.policies.qwen.update` |
| Kimi | ES module | `vm.SourceTextModule` | default export |
| minimax | ES module | `vm.SourceTextModule` | default export |

Three details make this work without touching policy source:

1. **Capture the native bot before loading any policy.** `policy/gpt/ai.js` overwrites `TT.updateBot` when it loads. The arena saves the engine's original function first and registers it as the "built-in" seat.
2. **Stub `setTimeout`.** GLM has a fallback that takes over `TT.updateBot` after 600 ms if nobody else does. The arena owns that function, so the fallback must be disabled.
3. **ESM sees the sandbox global.** `vm.SourceTextModule` is created with the same context, so minimax's `G.TT.CFG` reads the sandbox's `TT.CFG`, not the host's.

For the browser spectator page, `arena/build-vendor.mjs` mechanically converts the two ESM policies into classic scripts (`arena/vendor/*.classic.js`) because `file://` refuses dynamic `import()`. The conversion is checked byte-for-byte against the original after stripping comments. `policy/` is never modified.

---

## 3. Fairness by construction

### 3.1 Seats and spawns

Choosing a set of spawn points and then assigning them in greedy order gives the first pick a different distribution from the last. The engine now shuffles the selected set with the run's random source before assigning seats.

Rotation alone is not enough with finite samples: changing the spawn every round does not guarantee that each policy sees the same layout. `arena/schedule.js` groups rounds by player count `N`:

- one group = `N` rounds on the same map and spawn layout;
- each policy occupies each of the `N` seats exactly once per group;
- the next group uses a new spawn layout; maps alternate A/B by group;
- the spawn RNG is independent of policy calls and visual effects.

A run is balanced when the round count is a multiple of `2 × N`. The default 208 rounds is balanced for 8 players; 60 works for 3, 40 for 2.

### 3.2 Per-round isolation

Each round is a fresh sandbox: the engine and all seven policies are reloaded, and the round gets a seed derived from `(base seed, round index)`. Policy-internal `WeakMap`/`Map` caches cannot leak across rounds, and the same `(base seed, round index, players)` always produces the same round. Parallel and serial runs agree.

### 3.3 What the arena does not claim

Maps are asymmetric, so different spawn cells still have different survival value. The schedule guarantees **equal opportunity**, not equal geography. Policy randomness also remains: most policies are deterministic, but minimax and Kimi use internal randomness. The arena reports sample sizes and does not pretend that a 2-point difference is significant.

---

## 4. Four audits

A high win rate can come from cheating, from a broken physics assumption, or from a metric artefact. We ran four independent checks.

### 4.1 Runtime write audit (`arena/audit.mjs`)

A `Proxy` wraps `game`, `tank` and `bullet`, recording every write made while a policy is executing.

| Policy | Writes | Non-public reads |
|---|---:|---|
| DeepSeek, GLM, GPT, HY, Kimi, minimax, Qwen | **0** | None (DeepSeek/HY read fields that do not exist in this engine as cross-engine fallbacks; they get `undefined`) |
| Built-in bot | 4 (`tank.ai`) | `tank.barrelTip` |

The built-in bot's `tank.ai` write is the engine's own memory slot (`js/ai.js` does the same). All seven submitted policies made zero writes.

### 4.2 Steering audit (`arena/steer-audit.mjs`)

The engine integrates `angle += steer * turn * dt`, so `|steer| > 1` would exceed the physical turn rate of 3.3 rad/s. Because the harness passes `steer` through unclamped, this is a real attack surface.

Measured maximum `|steer|` was **1.00** for all eight agents, and the fraction of `|steer| > 1` was **0.00%**. No policy gained an illegal turn.

### 4.3 Seat neutrality (`arena/seat-check.mjs`)

Three seeds, 600 rounds:

- seat marginal win rates: 8.0%–14.7% (fair line 11.8%, maximum deviation 3.8 points);
- every policy's seat bonus: **±0.0 points** — rotation is exact;
- HY ranked first on **8 of 8 seats** (25.3%–45.3%).

If HY's edge came from a lucky seat, it would disappear in its worst seat. It did not.

### 4.4 Head-to-head (`arena/head2head.mjs`)

28 pairs × 30 rounds × swapped seats = **1,680 matches**, isolating each pair from the eight-player crowd:

```text
GPT Sentinel 72.1% > DeepSeek 69.2% > HY 68.1% > GLM 64.9%
> Qwen 45.3% > Kimi 36.8% > minimax 27.1% > built-in 17.5%
```

HY is only third in 1v1. If it were cheating, the same cheat would work in a duel. It did not.

---

## 5. The metric trap: why minimax's "95%" became 26.5%

minimax's own test harness computed:

```text
rate = win / (matches × 5 × seeds)
```

That denominator assumes every match lasts exactly five rounds. A first-to-5 match actually lasts 10–20 rounds, so the denominator is far too small and the rate is inflated.

`arena/metric-check.mjs` recomputes the same data with the true number of played rounds:

```text
played rounds: 566
minimax wins: 150
README formula (180 denominator): 83.3%
true per-round win rate (566 denominator): 26.5%
```

The arena therefore reports **per-round win rate**: wins ÷ rounds actually played. The table below summarises the different claims:

| Policy | README claim (1v7) | Denominator | Same as per-round win rate? |
|---|---:|---|---|
| minimax | 95.0% | matches × 5 × seeds | no, inflated |
| GLM | ~52% | played rounds | yes |
| Qwen | 40.0% | played rounds | yes |
| DeepSeek | 44.7% | played rounds (8 seats × 80) | yes |
| HY | 54.0% | played rounds | yes |

`arena/validate.mjs` re-measures every policy against the seven built-in bots using the common denominator.

---

## 6. Results (2026-11-06 engine snapshot)

Eight-player free-for-all, 208 rounds, seat-balanced:

| Policy | Win rate | Final-two rate | Survival to end | Kills/round |
|---|---:|---:|---:|---:|
| HY | **33.2%** | **51.8%** | **37.7%** | 1.17 |
| GPT Sentinel | 21.2% | 43.8% | 21.5% | **1.79** |
| GLM | 18.2% | 38.0% | 19.0% | 0.99 |
| DeepSeek | 15.2% | 27.3% | 18.3% | 0.93 |
| Qwen | 5.0% | 20.7% | 5.0% | 0.28 |
| Kimi / minimax / built-in | see `arena/report*.json` | | | |

The interpretation is more interesting than the ranking:

- **HY is defensive**: it wins the most rounds while killing less than GPT. Survival *is* win rate in a last-tank-alive format.
- **GPT is offensive**: it kills the most and dies more. It wins the duel, not the melee.
- **The format picks the champion.** Ranking by wins (HY), by kills (GPT), and by 1v1 (GPT) gives three different answers. That is evidence that the ranking reflects style-to-rules fit, not a hidden information advantage.

Among 33 timeouts, HY was the last survivor 27 times — the "both players refuse to engage" stalemate is itself a behavioural signature.

> The report files in `arena/` were produced before the last physics fixes (muzzle-inside-wall bullets, ricochet hit resolution). They are kept for provenance and are **not** the current ranking. Rerun `arena/run.mjs` for fresh numbers.

---

## 7. Browser spectator

`arena/index.html` runs without a build step or server:

- choose 2–8 agents, apply roster, reset statistics;
- 1× uses the real frame interval and a fixed 1/60 s physics step; 0.25×–16× is available;
- pause does not skip to the next round; single-step advances one frame;
- the leaderboard accumulates wins/kills/suicides by policy identity across seat rotation;
- 60 s timeout is a draw; kills already scored are kept;
- the RL preset adds the V2 and V7 ONNX models through `arena/rl-controller.mjs`, with a sha256 check and a browser-side inference HUD.

---

## 8. Reproduce

```bash
# Full 8-agent run, 208 balanced rounds
node --experimental-vm-modules arena/run.mjs --rounds 208 --seed 20260101 --jobs 8

# One policy against the seven built-in bots, common denominator
node --experimental-vm-modules arena/validate.mjs --rounds 40

# Physical and schedule regressions
node arena/regression.mjs

# Audits
node --experimental-vm-modules arena/audit.mjs
node --experimental-vm-modules arena/steer-audit.mjs
node --experimental-vm-modules arena/seat-check.mjs
node --experimental-vm-modules arena/head2head.mjs

# Frozen source check
node policy/verify-frozen.mjs
```

The frozen manifest is `policy/manifest.sha256.json` (7 policies, 74 files). `verify-frozen.mjs` fails if any policy source changed after freezing.
