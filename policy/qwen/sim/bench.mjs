/* ===========================================================
   bench.mjs — Arena 批量基准
   用法：
     node bench.mjs                     # qwen 单挑 7 个内置 bot，24 场
     node bench.mjs --n 40
     node bench.mjs --roster qwen,qwen,null,null,null,null,null,null
     node bench.mjs --dt 0.02 --rounds 7
   =========================================================== */

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playMatch } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return dflt;
  return process.argv[i + 1];
}

const N = parseInt(arg('n', '24'), 10);
const ROUNDS = parseInt(arg('rounds', '5'), 10);
const DT = parseFloat(arg('dt', String(1 / 60)));
const rosterSpec = arg('roster', null);

let roster;
if (rosterSpec) {
  roster = rosterSpec.split(',').map((s) => (s === 'null' ? null : s));
  while (roster.length < 8) roster.push(null);
} else {
  roster = ['qwen', null, null, null, null, null, null, null];
}

const label = (s) => (s === null ? 'builtin' : s);
const seats = new Array(8).fill(undefined);
const seatIdx = {};
for (let i = 0; i < 8; i++) {
  seats[i] = roster[i];
  if (roster[i] !== undefined) (seatIdx[label(roster[i])] = seatIdx[label(roster[i])] || []).push(i);
}

const agg = {};
for (const k of Object.keys(seatIdx)) {
  agg[k] = { rounds: 0, wins: 0, deaths: 0, suicides: 0, kills: 0, matchWins: 0, surv: 0, survN: 0, ricochet: 0, matches: 0, timeout: 0 };
}

const t0 = Date.now();
let totalRounds = 0, totalMatchSeconds = 0, timeouts = 0;
const roundDur = [];

for (let m = 0; m < N; m++) {
  const s = playMatch({
    seats: seats.slice(),
    policyFile: POLICY,
    roundsToWin: ROUNDS,
    seed: 1000 + m * 37,
    dt: DT,
    maxMatchSeconds: 420,
  });
  totalRounds += s.rounds.length;
  totalMatchSeconds += s.simSeconds;
  roundDur.push(...s.rounds.map((r) => r.duration));
  if (s.timedOut) timeouts++;

  for (const k of Object.keys(seatIdx)) {
    const a = agg[k];
    a.matches++;
    if (s.winner !== null && seatIdx[k].includes(s.winner)) a.matchWins++;
    for (const i of seatIdx[k]) {
      const p = s.per[i];
      if (!p) continue;
      a.rounds += p.rounds;
      a.wins += p.wins;
      a.kills += p.kills;
      a.suicides += p.suicides;
      a.deaths += p.suicides + p.killedByOthers;
      a.ricochet += p.ricochetDeaths;
      if (p.avgSurvival != null) { a.surv += p.avgSurvival * (p.suicides + p.killedByOthers); a.survN += p.suicides + p.killedByOthers; }
    }
  }
}

function i2(k) { return seatIdx[k]; }

const avg = (a, b) => (b ? a / b : 0);
console.log(`\n== Arena bench ==  ${N} 场 · 先到 ${ROUNDS} 胜 · dt=${DT.toFixed(4)} · roster=[${roster.map(label).join(',')}]`);
console.log(`回合总数 ${totalRounds} · 平均回合时长 ${avg(roundDur.reduce((x, y) => x + y, 0), roundDur.length).toFixed(1)}s · 超时 ${timeouts}\n`);

const rows = Object.keys(agg).map((k) => {
  const a = agg[k];
  const perSeat = a.matches * Object.keys(seatIdx[k]).length || 1;
  return {
    policy: k,
    '回合胜率%': (avg(a.wins, a.rounds) * 100).toFixed(1),
    '场胜率%': (avg(a.matchWins, a.matches) * 100).toFixed(1),
    '生存率%': (avg(a.rounds - a.deaths, a.rounds) * 100).toFixed(1),
    '自杀/回合': avg(a.suicides, a.rounds).toFixed(3),
    '死亡/回合': avg(a.deaths, a.rounds).toFixed(2),
    '击杀/回合': avg(a.kills, a.rounds).toFixed(2),
    '反弹致死': a.ricochet,
    '平均存活s': avg(a.surv, a.survN).toFixed(1),
  };
});
console.table(rows);
console.log(`仿真 ${totalMatchSeconds.toFixed(0)}s 对局 / ${(Date.now() - t0) / 1000}s 墙钟 · ${(totalMatchSeconds / ((Date.now() - t0) / 1000)).toFixed(0)}x 实时`);
