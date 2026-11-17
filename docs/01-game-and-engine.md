# 01 · The game and the engine

The game is a browser reimplementation of **Tank Trouble**: one bullet kills, bullets ricochet, and the last tank alive wins the round. It is deliberately small, deterministic, and asset-free, because it has to serve three roles at once:

1. a playable game with a readable UI;
2. the referee used by the agent arena;
3. the training environment used by PPO, with no second physics implementation in Python.

This note describes the rules, the map generator, the simulation loop, and the parts of the code that matter for reproducibility.

---

## 1. Rules that shape every policy

| Rule | Value | Consequence |
|---|---|---|
| World | 1280×800, 16×10 cells, 80 px per cell | Rectangular maps; no rotational symmetry assumptions |
| Walls | 10 px thick, on cell boundaries, merged into long segments | Ricochets are long and predictable, but only along grid-aligned surfaces |
| Tanks per round | 2–8 | 2 human seats + bots, or pure AI spectator mode |
| Movement | 178 px/s forward, 66% reverse, 3.3 rad/s turn, 920 px/s² acceleration | Driving is fast enough to dodge, slow enough to punish over-commitment |
| Bullets | 620 px/s, radius 3.2, 5 simultaneous per tank, 5 bounces, 6 s lifetime | Every shot can come back; ammunition is a resource |
| Hit | One hit kills | A single mistake ends a round; no health recovery |
| Score | Kill +1 for the shooter, suicide −1 | The game's own score is `kills − suicides` |
| Round | First to 5 wins; 60 s limit; timeout is a draw | Draws are frequent enough that they cannot be ignored in evaluation |

Two design details matter more than they look:

- **Spawn selection forbids opening crossfire.** `pickSpawns()` only considers cells with at least three open sides, and rejects any candidate pair with a clear line of fire (`hasClearShot()`). If it cannot satisfy the constraint for all tanks, it relaxes the spacing requirement — never the crossfire rule. This prevents "whoever spawns first wins" from dominating the data.
- **Spawn order is not seat order.** After choosing the set, the points are Fisher–Yates shuffled with the run's random source. The arena relies on this to rotate every agent through every seat without a first-pick advantage.

---

## 2. Map generation

`js/map.js` generates a map in four stages, then scores up to 70 candidates and keeps the best:

1. **Recursive division.** Split the grid with a long wall, leave one gap — or two gaps with probability 0.28 when the region is at least 5 cells long. Recurse on both sides. The split is biased slightly toward horizontal walls (42% vertical) so the result looks like the original's long corridors.
2. **Room carving.** Carve 6–9 rooms, mostly 2×2 and occasionally 2×3 or 3×3, by clearing the interior wall flags. Rooms create open fights and sight lines.
3. **Wall removal.** Delete each remaining wall segment with probability 0.10. This creates loops and avoids pure tree mazes.
4. **Short stubs.** Add 3–6 isolated short walls for visual variety, but only if `reachRatio()` stays at 1 — no wall may disconnect any cell.

The score combines:

- connectivity (`reachRatio()` must be 1),
- wall density close to 40%,
- at least 30 cells with three or more open sides.

Generation stops early at score ≥ 0.94; otherwise the best of 70 attempts is used. Two fixed seeds (`20260101`, `88888888`) alternate between rounds, so a match is reproducible without being symmetric.

The map is stored as two bit arrays: `vFlag[ROWS][COLS+1]` for vertical boundaries and `hFlag[ROWS+1][COLS]` for horizontal boundaries. Rendering merges consecutive flags into single wall rectangles; the collision system uses the same rectangles.

---

## 3. Simulation and determinism

The browser loop uses `requestAnimationFrame` and a **clamped variable timestep** (`dt ≤ 1/30 s`), because it must remain playable on arbitrary displays. The arena and the training harness do not use the browser loop: they advance the engine with a fixed `1/60 s` step, which is what makes rollouts reproducible.

The engine itself is time-parameterised:

- tank acceleration, braking, turning and bullet motion all integrate `dt`;
- cooldowns and lifetimes are in seconds;
- the round timer advances only during the `playing` state, not during intro/outro animations.

Determinism comes from four separate streams:

| Stream | Used for |
|---|---|
| Map seed | Wall layout |
| Spawn seed | Candidate cells, orientations, seat shuffle |
| Policy seed | Any randomness inside an agent (most are deterministic) |
| FX seed | Particles, screen shake, audio variation — never gameplay |

A round can be replayed from `(map seed, spawn seed, policy seeds, action sequence)`. The arena uses this to guarantee that a given `(base seed, round index)` always produces the same round, including under parallel execution.

---

## 4. UI and presentation

The interface is intentionally plain and high-contrast, closer to the Flash original than to a modern web app:

- 1280×800 canvas with a device-pixel-ratio-aware backing store;
- a right-hand kill board that sorts by kills and shows wins;
- an overlay state machine for menu → setup → controls → pause → result;
- keyboard hints, fullscreen, mute, and a pause that does not advance the world;
- 8-colour palette with thick black outlines; no images, no fonts beyond the system stack.

Audio is synthesised at runtime with WebAudio: shot, bounce, explosion, UI click. There are no audio files to download. This keeps the repository self-contained and makes the game load instantly from `file://`.

The UI is not cosmetics in this project: the arena's spectator page reuses the same rendering and input contracts, and the kill board is the only in-game view of the same `kills`/`suicides` counters that the learning code optimises.

---

## 5. Code map

| File | Responsibility |
|---|---|
| `js/util.js` | Constants, seeded RNG, vector/angle helpers, colour helpers |
| `js/map.js` | Recursive-division generator, wall flags, collision query, spawn selection |
| `js/entities.js` | Tank and bullet integration, ricochet, hit resolution, effects hooks |
| `js/game.js` | Round state machine, scoring, 60 s timeout, world update |
| `js/ai.js` | Built-in bot used as a baseline and as a weak curriculum opponent |
| `js/input.js` | Keyboard mapping for two local players |
| `js/audio.js` | WebAudio synthesis |
| `js/ui.js` | Menus, HUD, kill board, result table |
| `js/main.js` | Bootstrapping and the animation loop |
| `css/style.css` | Layout, panels, typography, responsive behaviour |

For the learning side, the same engine is wrapped by:

- `policy/rl/v2/geometry.mjs` — ricochet tracing, line-of-sight, aim candidates;
- `policy/rl/env.mjs` — observation construction, 60 Hz step, reward bookkeeping;
- `policy/rl/v4/worker.mjs` — Node worker used by PPO rollouts.

See [02 · Agent arena](02-agent-arena.md) for how the same engine is sandboxed for seven untrusted policies, and [04 · Architecture](04-architecture.md) for the observation schema built on top of it.
