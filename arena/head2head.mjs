/* ===========================================================
   head2head.mjs — 逐对单挑：验证排名是真实实力差异
   -----------------------------------------------------------
   如果 HY 真的在「作弊」，那它在 1v1 里也应该赢；但作弊手段
   （改状态、偷看、串通）在 1v1 里同样有效。

   所以单挑结果能区分两种可能：
     · HY 单挑也赢 → 它要么真的强，要么作弊（再用 audit.mjs 排除作弊）
     · HY 单挑不赢 → 它的优势来自混战机制，不来自实力

   另外顺带统计每个策略在 8 个座位上的胜率，验证座位轮转是否真的中性。

   用法：node --experimental-vm-modules arena/head2head.mjs [--rounds 40]
   =========================================================== */

import vm from 'node:vm';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';

import { createArena, mulberry32, roundSeed, STRATEGY_IDS, STRATEGIES } from './arena.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  if (i >= 0 && process.argv[i + 1] != null) return process.argv[i + 1];
  return dflt;
}
const ROUNDS = parseInt(arg('rounds', '40'), 10);
const SEED = parseInt(arg('seed', '20260101'), 10);
const JOBS = parseInt(arg('jobs', String(Math.min(16, os.cpus().length))), 10);

/* ---------------- 单挑：策略 A（座位 0） vs 策略 B（座位 1） ---------------- */

async function duel(a, b, seed, roundIndex) {
  const env = await createArena();
  const { TT, ctx, sandbox, adapters } = env;

  sandbox.__rnd = mulberry32(roundSeed(seed, roundIndex));
  vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

  const seats = [a, b];
  /* 两张图交替 */
  const mapSeed = (roundIndex % 2 === 0) ? 20260101 : 88888888;
  TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
  TT.CFG.ROUND_INTRO = 0;

  const game = new TT.Game(env.canvas);
  game.render = function () {}; game.resize = function () {};
  game.mode = 'match';
  game.slots = ['bot', 'bot', 'off', 'off', 'off', 'off', 'off', 'off'];
  game.roundsToWin = 9999;

  let finished = false, winnerIndex = -1, frames = 0;
  const origEnd = game.endRound.bind(game);
  game.endRound = function (w) { finished = true; winnerIndex = w; return origEnd(w); };

  TT.updateBot = function (tank, g, dt) {
    const fn = adapters[seats[tank.index]];
    const out = fn(tank, g, dt);
    if (!out) return { drive: 0, steer: 0, shoot: false };
    return {
      drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
      steer: Number(out.steer) || 0,
      shoot: !!out.shoot,
    };
  };

  game.startRound();
  while (!finished && frames < 60 * 60 + 2) { game.update(1 / 60); frames++; }

  return {
    a, b,
    winner: winnerIndex >= 0 ? seats[winnerIndex] : null,
    timeout: game.roundTimedOut || !finished,
    durationSec: game.roundTime,
  };
}

/* ---------------- 并行 ---------------- */

function runTasks(tasks) {
  return new Promise((resolve, reject) => {
    const n = Math.max(1, Math.min(JOBS, tasks.length));
    const chunk = Math.ceil(tasks.length / n);
    const results = [];
    let finished = 0;
    for (let i = 0; i < n; i++) {
      const slice = tasks.slice(i * chunk, (i + 1) * chunk);
      if (!slice.length) { finished++; continue; }
      const w = new Worker(path.join(HERE, 'h2h-worker.mjs'), {
        workerData: { seed: SEED, slice, rounds: ROUNDS },
        execArgv: ['--experimental-vm-modules'],
      });
      w.on('message', (m) => { if (m.type === 'done') results.push(...m.results); });
      w.on('error', reject);
      w.on('exit', () => { if (++finished === n) resolve(results); });
      w.postMessage({ type: 'run' });
    }
  });
}

/* ---------------- 主流程 ---------------- */

/* 1v1：每对打两次，互换座位（消除座位偏好） */
const pairs = [];
for (let i = 0; i < STRATEGY_IDS.length; i++) {
  for (let j = i + 1; j < STRATEGY_IDS.length; j++) {
    pairs.push([STRATEGY_IDS[i], STRATEGY_IDS[j]]);
  }
}
const tasks = [];
for (const [a, b] of pairs) {
  for (let r = 0; r < ROUNDS; r++) {
    tasks.push({ a, b, r });
    tasks.push({ a: b, b: a, r });
  }
}

console.log(`\n▶ 逐对单挑：${pairs.length} 对 × ${ROUNDS} 局 × 2（互换座位）= ${tasks.length} 场\n`);

const results = await runTasks(tasks);

/* 汇总 */
const tally = {};
for (const id of STRATEGY_IDS) {
  tally[id] = { wins: 0, losses: 0, draws: 0, timeouts: 0, durSum: 0, n: 0 };
}
for (const r of results) {
  const A = tally[r.a], B = tally[r.b];
  A.n++; B.n++;
  A.durSum += r.durationSec; B.durSum += r.durationSec;
  if (r.timeout) { A.timeouts++; B.timeouts++; }
  if (r.winner === r.a) { A.wins++; B.losses++; }
  else if (r.winner === r.b) { B.wins++; A.losses++; }
  else { A.draws++; B.draws++; }
}

console.log('  策略            胜     负    平   胜率    平均时长');
console.log('  ' + '─'.repeat(58));
const order = STRATEGY_IDS.slice().sort((x, y) => tally[y].wins - tally[x].wins);
for (const id of order) {
  const t = tally[id];
  const decided = t.wins + t.losses;
  const rate = decided ? (t.wins / decided * 100).toFixed(1) + '%' : '—';
  const label = (STRATEGIES.find((s) => s.id === id) || {}).label || id;
  console.log('  ' + label.padEnd(15) +
    String(t.wins).padStart(4) + String(t.losses).padStart(6) +
    String(t.draws).padStart(6) + rate.padStart(9) +
    ((t.durSum / t.n).toFixed(1) + 's').padStart(11));
}

console.log('\n  注：单挑里「平」= 120 秒仍未分出胜负（真实每场平手）。');
console.log('  胜率分母只算「分出胜负」的对局，平手不计入。');
console.log('');
