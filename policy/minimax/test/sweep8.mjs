/* ===========================================================
   minimax policy · 决选扫描（跨多种子，稳健性优先）
   ---------------------------------------------------------------
   前面的扫描单个种子差异可能有噪声。这一轮对每个候选配置
   跑 3 组完全不同的种子，把结果合起来看，谁稳定谁就赢。
   用法：node test/sweep8.mjs [每组场数] [分片号] [分片数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure, resetConfig } from '../policy.js';

const matches = parseInt(process.argv[2] || '16', 10);
const chunk = parseInt(process.argv[3] || '0', 10);
const chunkCount = parseInt(process.argv[4] || '1', 10);
const { TT } = loadGame();

const SEEDS = [7000, 11000, 15000];

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
  { name: '默认(余量6,增益12)', cfg: {} },
  { name: '余量0', cfg: { selfMargin: 0 } },
  { name: '余量0+增益24', cfg: { selfMargin: 0, dodgeGain: 24 } },
  { name: '余量0+不信阻挡', cfg: { selfMargin: 0, blockRule: 0 } },
  { name: '余量0+增益24+不信阻挡', cfg: { selfMargin: 0, dodgeGain: 24, blockRule: 0 } },
  { name: '余量3+增益18', cfg: { selfMargin: 3, dodgeGain: 18 } },
  { name: '余量0+增益18', cfg: { selfMargin: 0, dodgeGain: 18 } },
  { name: '余量0+增益36', cfg: { selfMargin: 0, dodgeGain: 36 } },
];

console.log(`决选扫描：每组 ${matches} 场 × 8 辆 × ${SEEDS.length} 组种子 = ${matches * 5 * SEEDS.length} 回合`);
console.log('─'.repeat(92));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  resetConfig(); configure(v.cfg);
  let win = 0, self = 0, death = 0, kill = 0, rounds = 0, to = 0, errors = 0, dur = 0, dn = 0;
  const perSeed = [];
  for (const sb of SEEDS) {
    const res = [];
    for (let i = 0; i < matches; i++) {
      installRandom(sb + i * 7919);
      const idx = i % 8;
      res.push(runMatch(TT, policy, { policyIndex: idx, tankCount: 8, roundsToWin: 5 }));
    }
    let w = 0;
    for (const r of res) {
      w += r.scores[r.policyIndex];
      self += r.selfKills[r.policyIndex];
      death += r.deaths[r.policyIndex];
      kill += r.kills[r.policyIndex];
    }
    win += w;
    perSeed.push((w / (matches * 5) * 100).toFixed(1) + '%');
    const s = summarize(res, 0);
    rounds += s.rounds; to += s.timeouts; errors += s.errors;
    dur += s.avgRound * s.rounds; dn += s.rounds;
  }
  rows.push({
    name: v.name, win, rate: (win / (matches * 5 * SEEDS.length) * 100),
    kill, self, death, to, rounds, dur: dur / Math.max(1, dn), errors,
    perSeed: perSeed.join(' '),
  });
}

rows.sort((a, b) => b.rate - a.rate);
for (const r of rows) {
  console.log(
    `  ${r.name.padEnd(24, ' ')} 胜率 ${r.rate.toFixed(1).padStart(5)}%` +
    `  胜场 ${String(r.win).padStart(4)}/${matches * 5 * SEEDS.length}` +
    `  净击杀 ${String(r.kill).padStart(4)}` +
    `  自杀 ${String(r.self).padStart(3)}` +
    `  死亡 ${String(r.death).padStart(4)}` +
    `  超时 ${String(r.to).padStart(3)}/${String(r.rounds).padStart(4)}` +
    `  回合 ${r.dur.toFixed(1)}s` +
    (r.errors ? `  ⚠异常${r.errors}` : '')
  );
  console.log(`      分种子：${r.perSeed}`);
}
resetConfig();
