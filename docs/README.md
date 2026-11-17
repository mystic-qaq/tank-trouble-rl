# Research notes

Long-form notes behind the [main README](../README.md). They are written to be read in order, but each is self-contained.

| Note | What it covers | Why read it |
|---|---|---|
| [01 · Game and engine](01-game-and-engine.md) | Rules, map generation, physics, determinism, UI | The environment every experiment shares |
| [02 · Agent arena](02-agent-arena.md) | Seven LLM policies, sandboxing, fairness audits, metric traps | How we compare fixed agents honestly |
| [03 · Learning policies](03-learning-policies.md) | V1–V8: hypotheses, runs, numbers, post-mortems | The main research narrative |
| [04 · Architecture](04-architecture.md) | V6 tokens/attention, V7 aim distillation, V8 schema | What the networks actually compute |
| [05 · Reproduction](05-reproduction.md) | Environment, data, commands, artifact map | How to rerun or extend it |

The raw Chinese lab logs are kept next to the code that produced them:

- `policy/rl/PLAN.md` — the original V1 plan.
- `policy/rl/DIAGNOSIS.md` — the first behavioural diagnosis.
- `policy/rl/v2/RESULTS_V2B.md` — DAgger and skill-gate results.
- `policy/rl/v3/HEALTH_20261107.md` — readiness review that stopped V3 PPO.
- `policy/rl/v4/V5_TRIAL_20261107.md` — V4→V6 record, including the V5 final numbers.
- `policy/rl/v8/DESIGN.md` — the V8 design contract.

These are working documents, not polished prose. They contain cluster paths and timestamps from the original runs; the numbers in the English notes above are cross-checked against them.
