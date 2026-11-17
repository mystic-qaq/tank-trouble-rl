'use strict';
// Run: node policy/gpt/verify.cjs [--quick] [--selfplay]
// This harness loads ONLY engine files and this directory's own policy.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { performance } = require('node:perf_hooks');
const root = path.resolve(__dirname, '../..');
const sandbox = { console, performance };
sandbox.window = sandbox;
vm.createContext(sandbox);
for (const name of ['util', 'map', 'entities', 'game']) {
  vm.runInContext(fs.readFileSync(path.join(root, 'js', name + '.js'), 'utf8'), sandbox, { filename: name + '.js' });
}
let policyCode = fs.readFileSync(path.join(__dirname, 'ai.js'), 'utf8');
policyCode = policyCode.replace('state.physics = data.physics;',
  'globalThis.__gptPhysics = data.physics; state.physics = data.physics;');
if (process.env.GPT_TRACE) policyCode = policyCode.replace('state.future = plan',
  'globalThis.__gptTrace = { goal, aim }; state.future = plan');
vm.runInContext(policyCode, sandbox, { filename: 'gpt/ai.js' });
const TT = sandbox.TT, policy = TT.updateBot;
TT.Audio = new Proxy({}, { get: () => () => {} });
TT.Input = { get: () => ({ drive: 0, steer: 0, shoot: false }) };
const noFX = () => new Proxy({}, { get: () => () => {} });
const canvas = { width: 1280, height: 800, getContext: () => ({}),
  getBoundingClientRect: () => ({ width: 1280, height: 800 }) };
let calls = 0, totalMs = 0, maxMs = 0;
const timings = [];
function decide(t, g, dt) {
  const before = performance.now(), result = policy(t, g, dt), elapsed = performance.now() - before;
  calls++; totalMs += elapsed; maxMs = Math.max(maxMs, elapsed); timings.push(elapsed);
  assert(Number.isFinite(result.steer) && Math.abs(result.steer) <= 1);
  assert([-1, 0, 1].includes(result.drive));
  assert.equal(typeof result.shoot, 'boolean');
  if (g.trace && t.index === g.tracedSlot) g.trace.push({ time: g.time, debug: sandbox.__gptTrace,
    tank: { x: t.x, y: t.y, a: t.angle, speed: t.speed, cooldown: t.cooldown }, input: result,
    enemies: g.tanks.filter(e => e.alive && e !== t).map(e => ({ id: e.index, x: e.x, y: e.y, a: e.angle, cooldown: e.cooldown })),
    bullets: g.bullets.map(b => ({ owner: b.owner, x: b.x, y: b.y, a: b.angle, bounce: b.bounces })) });
  return result;
}
function game(seed, specs) {
  const g = new TT.Game(canvas);
  g.mode = 'match'; g.state = 'playing'; g.map = new TT.GameMap(seed); g.fx = noFX();
  g.tanks = specs.map((s, i) => new TT.Tank(i, { x: s.x, y: s.y, a: s.a || 0 }, TT.COLORS[i], true));
  g.bullets = []; g.controllers = specs.map(s => s.controller || 'gpt');
  g.deaths = []; g.shots = 0;
  const kill = g.killTank.bind(g);
  g.killTank = (t, b) => { g.deaths.push({ victim: t.index, owner: b.owner, time: g.time, bounces: b.bounces }); kill(t, b); };
  return g;
}
function openMap(g) {
  g.map.vFlag.fill(0); g.map.hFlag.fill(0); g.map.buildWalls();
  return g;
}
const tick = (g, dt) => {
  TT.updateBot = (t, world, delta) => {
    const c = g.controllers[t.index];
    if (c === 'gpt') return decide(t, world, delta);
    return typeof c === 'function' ? c(t, world, delta) : { drive: 0, steer: 0, shoot: false };
  };
  const before = g.bullets.length;
  g.update(dt);
  g.shots += Math.max(0, g.bullets.length - before);
};
function run(g, seconds, dt = 1 / 60) {
  for (let i = 0; i < seconds / dt && g.state === 'playing'; i++) tick(g, dt);
  return { winner: g.state === 'over' ? g.roundWinner : null,
    seconds: +g.time.toFixed(2), deaths: g.deaths, alive: g.tanks.filter(t => t.alive).map(t => t.index), shots: g.shots };
}
// A test-only gun: no imported or external policy. It pursues via the public
// map.findPath, aims at the present target and fires whenever its barrel aligns.
function testGun(t, g, dt) {
  const foes = g.tanks.filter(e => e.alive && e.index !== t.index).sort((a, b) =>
    Math.hypot(a.x - t.x, a.y - t.y) - Math.hypot(b.x - t.x, b.y - t.y));
  if (!foes.length) return { drive: 0, steer: 0, shoot: false };
  const e = foes[0], clear = g.map.hasLOS(t.x, t.y, e.x, e.y);
  const route = clear ? [e] : g.map.findPath(t.x, t.y, e.x, e.y);
  const p = route && route.length ? route[0] : e;
  const a = Math.atan2(p.y - t.y, p.x - t.x), da = TT.util.angleDiff(t.angle, a);
  const range = Math.hypot(e.x - t.x, e.y - t.y);
  return { drive: Math.abs(da) < .65 && (!clear || range > 210) ? 1 : 0,
    steer: TT.util.clamp(da / (TT.CFG.TANK.turn * dt), -1, 1),
    shoot: clear && Math.abs(da) < .10 };
}
function checkReadOnly() {
  const g = openMap(game(1, [{ x: 200, y: 300 }, { x: 600, y: 300, controller: 'idle' }]));
  const before = JSON.stringify({ tanks: g.tanks, bullets: g.bullets, walls: g.map.walls, scores: g.scores });
  decide(g.tanks[0], g, 1 / 60);
  const after = JSON.stringify({ tanks: g.tanks, bullets: g.bullets, walls: g.map.walls, scores: g.scores });
  assert.equal(after, before, 'policy must not mutate world state');
  console.log('PASS read-only world and valid controls');
}
checkReadOnly();
for (const seed of TT.CFG.MAP_SEEDS) {
  const g = game(seed, [{ x: 120, y: 120 }, { x: 1000, y: 680, controller: 'idle' }]);
  decide(g.tanks[0], g, 1 / 60);
  const fast = sandbox.__gptPhysics, rng = TT.util.rng(seed + 17);
  for (let i = 0; i < 2500; i++) {
    const x = rng() * 1280, y = rng() * 800, xx = rng() * 1800 - 260, yy = rng() * 1200 - 200;
    const a = g.map.raycast(x, y, xx, yy), b = fast.raycast(x, y, xx, yy);
    assert.equal(!!a, !!b, 'accelerated raycast hit must match engine');
    if (a) {
      assert(Math.abs(a.t - b.t) < 1e-10, 'accelerated raycast time must match engine');
      assert.equal(a.nx, b.nx); assert.equal(a.ny, b.ny);
    }
    const p = { x, y }, q = { x, y };
    assert.equal(g.map.resolveCircle(p, 16), fast.resolveCircle(q, 16));
    assert(Math.hypot(p.x - q.x, p.y - q.y) < 1e-8, 'accelerated circle solver must match engine');
  }
}
console.log('PASS 5000 engine-equivalent raycasts and 5000 body collisions');
for (const dt of [1 / 60, 1 / 30]) {
  const g = openMap(game(3, [{ x: 200, y: 300, a: .75 }, { x: 600, y: 300, controller: 'idle' }]));
  const result = run(g, 8, dt);
  assert.equal(result.winner, 0, 'must acquire and kill a visible stationary target');
  console.log('PASS stationary target', dt, JSON.stringify(result));
}
{
  const g = openMap(game(3, [{ x: 200, y: 300, a: .2 }, { x: 600, y: 300,
    a: Math.PI / 2, controller: () => ({ drive: 1, steer: 0, shoot: false }) }]));
  const result = run(g, 8);
  assert.equal(result.winner, 0, 'must lead and hit a moving target');
  console.log('PASS moving target', JSON.stringify(result));
}
{
  const g = openMap(game(6, [{ x: 300, y: 400, a: -1.1 }, { x: 650, y: 400, controller: 'idle' }]));
  g.map.walls.push({ x: 465, y: 200, w: 10, h: 300 });
  assert(!g.map.hasLOS(300, 400, 650, 400));
  const result = run(g, 8);
  assert.equal(result.winner, 0, 'must attack an initially occluded target');
  assert(result.deaths.some(d => d.owner === 0 && d.bounces > 0), 'must land a bank shot');
  console.log('PASS bank shot around cover', JSON.stringify(result));
}
for (const dt of [1 / 60, 1 / 30]) {
  const g = openMap(game(4, [{ x: 400, y: 400, a: Math.PI / 2 }, { x: 1100, y: 700, controller: 'idle' }]));
  g.bullets.push(new TT.Bullet(650, 400, Math.PI, 1, '#000'));
  run(g, 1.1, dt);
  assert(g.tanks[0].alive, 'must evade a known frontal incoming projectile');
  console.log('PASS incoming projectile', dt, g.tanks[0].x.toFixed(1), g.tanks[0].y.toFixed(1));
}
// A real wall bounce rather than a straight extrapolation toward our tank.
{
  const g = openMap(game(5, [{ x: 160, y: 400, a: Math.PI / 2 }, { x: 1100, y: 700, controller: 'idle' }]));
  g.bullets.push(new TT.Bullet(90, 400, Math.PI, 1, '#000'));
  run(g, .9);
  assert(g.tanks[0].alive, 'must evade a projectile returning from the outer wall');
  console.log('PASS reflected incoming projectile');
}
for (let nav = 0; nav < Number(process.env.GPT_NAV_CASES || 2); nav++) {
  if (process.env.GPT_NAV_ONLY && nav + 1 !== Number(process.env.GPT_NAV_ONLY)) continue;
  const seed = TT.CFG.MAP_SEEDS[nav % 2];
  const map = new TT.GameMap(seed), spawns = map.pickSpawns(2, TT.util.rng(seed + 31 + Math.floor(nav / 2)));
  const g = game(seed, spawns.map((p, i) => ({ ...p, controller: i ? 'idle' : 'gpt' })));
  if (process.env.GPT_TRACE) { g.trace = []; g.tracedSlot = 0; }
  const result = run(g, 40);
  if (result.winner !== 0) {
    console.log('NAV FAILED', nav + 1, JSON.stringify({ result, tanks: g.tanks.map(t => ({ x: t.x, y: t.y, a: t.angle, speed: t.speed })) }));
    if (g.trace) fs.writeFileSync(path.join(__dirname, 'nav-trace.json'), JSON.stringify(g.trace));
  }
  assert.equal(result.winner, 0, 'must navigate the actual maze and finish a stationary opponent');
  console.log('PASS maze navigation', seed, JSON.stringify(result));
}
let matchSummary = null;
if (!process.argv.includes('--quick')) {
  const rounds = Number(process.env.GPT_ROUNDS || 12);
  const summary = { rounds, wins: 0, draws: 0, losses: 0, suicides: 0, completed: 0 };
  for (let r = 0; r < rounds; r++) {
    if (process.env.GPT_ONLY_ROUND && r + 1 !== Number(process.env.GPT_ONLY_ROUND)) continue;
    const seed = TT.CFG.MAP_SEEDS[r % 2], m = new TT.GameMap(seed);
    const n = Number(process.env.GPT_PLAYERS) || (process.argv.includes('--selfplay') ? 4 : (r % 3 ? 2 : 8));
    const spawns = m.pickSpawns(n, TT.util.rng(4100 + r));
    const slot = r % n;
    const g = game(seed, spawns.map((p, i) => ({ ...p,
      controller: process.argv.includes('--selfplay') || i === slot ? 'gpt' : testGun })));
    if (process.env.GPT_TRACE) { g.trace = []; g.tracedSlot = slot; }
    const result = run(g, 65, r % 2 ? 1 / 30 : 1 / 60);
    if (result.winner != null) summary.completed++;
    if (process.argv.includes('--selfplay')) assert.notEqual(result.winner, null, 'selfplay must finish within 65 seconds');
    if (result.winner === slot) summary.wins++;
    else if (result.winner == null || result.winner < 0) summary.draws++;
    else summary.losses++;
    summary.suicides += g.deaths.filter(d => d.owner === d.victim &&
      (process.argv.includes('--selfplay') || d.victim === slot)).length;
    console.log('ROUND', r + 1, 'slot', slot, JSON.stringify(result));
    if (g.trace) fs.writeFileSync(path.join(__dirname, 'trace-' + (r + 1) + '.json'), JSON.stringify(g.trace));
  }
  matchSummary = summary;
  console.log('RESULT', JSON.stringify(summary));
}
timings.sort((a, b) => a - b);
const perf = { calls, meanMs: +(totalMs / calls).toFixed(3),
  p95Ms: +timings[Math.floor(timings.length * .95)].toFixed(3), maxMs: +maxMs.toFixed(3) };
console.log('PERFORMANCE', JSON.stringify(perf));
if (process.env.GPT_REPORT) {
  const report = { policy: 'GPT Sentinel', sha256: require('node:crypto').createHash('sha256')
    .update(fs.readFileSync(path.join(__dirname, 'ai.js'))).digest('hex'),
    engineChecks: 'passed', geometrySamples: 5000, navCases: Number(process.env.GPT_NAV_CASES || 2),
    mode: process.argv.includes('--selfplay') ? 'selfplay' : 'test-gun', matches: matchSummary, performance: perf };
  fs.writeFileSync(path.join(__dirname, process.argv.includes('--selfplay') ? 'validation-selfplay.json' : 'validation.json'),
    JSON.stringify(report, null, 2) + '\n');
}
