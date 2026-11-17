/* runner.mjs — 单一配置跑 M 场，输出一行 JSON（给 sweep.mjs 用）
   node runner.mjs --n 40 --cfg '{"selfMargin":10}' --seed 1000 --seats 1 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playMatch } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');

function arg(n, d) {
  const i = process.argv.indexOf('--' + n);
  return i < 0 ? d : process.argv[i + 1];
}

const N = parseInt(arg('n', '40'), 10);
const cfg = JSON.parse(arg('cfg', '{}'));
const seedBase = parseInt(arg('seed', '1000'), 10);
const kQwen = parseInt(arg('seats', '1'), 10);
const ROUNDS = parseInt(arg('rounds', '4'), 10);
const dt = parseFloat(arg('dt', String(1 / 60)));
const opponent = arg('opponent', 'builtin');   // builtin | qwen

const seats = new Array(8).fill(undefined);
for (let i = 0; i < 8; i++) {
  if (i < kQwen) seats[i] = cfg && Object.keys(cfg).length ? { policy: 'qwen', tune: cfg } : 'qwen';
  else seats[i] = opponent === 'qwen' ? 'qwen' : null;
}

let rounds = 0, wins = 0, draws = 0, suicides = 0, deaths = 0, kills = 0;
let matchWins = 0, matches = 0, timeouts = 0, durSum = 0, ricochet = 0, selfRico = 0;
let duelReach = 0, duelWin = 0, survivedToEnd = 0;
let duoReach = 0, duoWin = 0;
let stalls = 0, stallSec = 0;
const SEATIDX = 0;
const qwenIdx = [];
for (let i = 0; i < 8; i++) if (seats[i] && seats[i] !== 'builtin') qwenIdx.push(i);

for (let m = 0; m < N; m++) {
  const s = playMatch({
    seats: seats.slice(),
    policyFile: POLICY,
    roundsToWin: ROUNDS,
    seed: seedBase + m * 131,
    dt,
    maxMatchSeconds: 300,
  });
  matches++;
  if (s.timedOut) timeouts++;
  const myIdx = new Set(qwenIdx);
  for (const r of s.rounds) {
    if (r.unfinished) { stalls++; stallSec += r.duration; continue; }   // 僵死回合单独记账
    rounds++;
    durSum += r.duration;
    if (r.winner < 0) draws++;
    else if (myIdx.has(r.winner)) wins++;
    if (r.aliveAtEnd && r.aliveAtEnd.length === 2 && r.aliveAtEnd.includes(SEATIDX)) {
      duelReach++;
      if (r.winner === SEATIDX) duelWin++;
    }
    if (r.duo && r.duo.length === 2 && r.duo.includes(SEATIDX)) {
      duoReach++;
      if (r.winner === SEATIDX) duoWin++;
    }
    if (r.aliveAtEnd && r.aliveAtEnd.length === 1 && r.aliveAtEnd[0] === SEATIDX) survivedToEnd++;
  }
  if (s.winner !== null && myIdx.has(s.winner)) matchWins++;
  for (const i of qwenIdx) {
    const p = s.per[i];
    suicides += p.suicides;
    deaths += p.suicides + p.killedByOthers;
    kills += p.kills;
    ricochet += p.ricochetDeaths;
  }
  for (const d of s.deaths) if (d.suicide && myIdx.has(d.victim) && d.bounces > 0) selfRico++;
}

const out = {
  cfg,
  n: matches,
  rounds,
  winRate: wins / rounds,
  drawRate: draws / rounds,
  deathRate: deaths / rounds,
  killRate: kills / rounds,
  suicideRate: suicides / rounds,
  matchWin: matchWins / matches,
  avgRound: durSum / rounds,
  timeouts,
  ricochet,
  selfRico,
  duelReach,
  duelWinRate: duelReach ? duelWin / duelReach : null,
  duoReach,
  duoWinRate: duoReach ? duoWin / duoReach : null,
  surviveToEnd: survivedToEnd / rounds,
  stalls,
  stallSec: stallSec / (matches || 1),
  earlyDeath: null,
};
process.stdout.write(JSON.stringify(out) + '\n');
