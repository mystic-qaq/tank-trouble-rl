/* ===========================================================
   minimax policy · 新防守机制下的核心参数复核（第九轮）
   ---------------------------------------------------------------
   自伤/闪避逻辑大改之后，原来调好的核心参数（瞄准容差、距离带、
   横移时长等）可能不再是最优。这一轮用新的防守配置重新复核。
   用法：node test/sweep9.mjs [每组场数] [分片号] [分片数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure, resetConfig } from '../policy.js';

const matches = parseInt(process.argv[2] || '14', 10);
const chunk = parseInt(process.argv[3] || '0', 10);
const chunkCount = parseInt(process.argv[4] || '1', 10);
const { TT } = loadGame();

const SEEDS = [21000, 25000];

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const VARIANTS = [
  { name: '新基线', cfg: {} },
  { name: '容差0.035', cfg: { aimTol: 0.035 } },
  { name: '容差0.07', cfg: { aimTol: 0.07 } },
  { name: '距离带100-280', cfg: { standoffMin: 100, standoffMax: 280, standoffIdeal: 190 } },
  { name: '距离带140-360', cfg: { standoffMin: 140, standoffMax: 360, standoffIdeal: 240 } },
  { name: '横移0.5', cfg: { displaceTime: 0.5 } },
  { name: '横移1.2', cfg: { displaceTime: 1.2 } },
  { name: '横移1.6', cfg: { displaceTime: 1.6 } },
  { name: '闪避阈值32', cfg: { dodgeWant: 32 } },
  { name: '闪避阈值48', cfg: { dodgeWant: 48 } },
  { name: '闪避时长1.6', cfg: { dodgeHorizon: 1.6 } },
  { name: '闪避时长2.0', cfg: { dodgeHorizon: 2.0 } },
  { name: '在场子弹2', cfg: { maxLiveBullets: 2 } },
  { name: '在场子弹4', cfg: { maxLiveBullets: 4 } },
  { name: '撤阈值3', cfg: { crossRetreat: 3 } },
  { name: '提前量0.9', cfg: { leadFactor: 0.9 } },
];

const TOTAL = matches * 5 * SEEDS.length;

console.log(`核心参数复核：每组 ${matches} 场 × ${SEEDS.length} 组种子 = ${TOTAL} 回合  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(88));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  resetConfig(); configure(v.cfg);
  let win = 0, self = 0, death = 0, kill = 0, rounds = 0, to = 0, errors = 0, dur = 0, dn = 0;
  for (const sb of SEEDS) {
    const res = [];
    for (let i = 0; i < matches; i++) {
      installRandom(sb + i * 7919);
      const idx = i % 8;
      res.push(runMatch(TT, policy, { policyIndex: idx, tankCount: 8, roundsToWin: 5 }));
    }
    for (const r of res) {
      win += r.scores[r.policyIndex];
      self += r.selfKills[r.policyIndex];
      death += r.deaths[r.policyIndex];
      kill += r.kills[r.policyIndex];
    }
    const s = summarize(res, 0);
    rounds += s.rounds; to += s.timeouts; errors += s.errors;
    dur += s.avgRound * s.rounds; dn += s.rounds;
  }
  rows.push({
    name: v.name, win, rate: (win / TOTAL * 100), kill, self, death,
    to, rounds, dur: dur / Math.max(1, dn), errors,
  });
}

rows.sort((a, b) => b.rate - a.rate);
for (const r of rows) {
  console.log(
    `  ${r.name.padEnd(18, ' ')} 胜率 ${r.rate.toFixed(1).padStart(5)}%` +
    `  胜场 ${String(r.win).padStart(3)}/${TOTAL}` +
    `  净击杀 ${String(r.kill).padStart(4)}` +
    `  自杀 ${String(r.self).padStart(3)}` +
    `  死亡 ${String(r.death).padStart(4)}` +
    `  超时 ${String(r.to).padStart(3)}/${String(r.rounds).padStart(4)}` +
    `  回合 ${r.dur.toFixed(1)}s` +
    (r.errors ? `  ⚠异常${r.errors}` : '')
  );
}
resetConfig();
