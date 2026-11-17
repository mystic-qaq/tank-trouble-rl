/* ===========================================================
   minimax policy · 第五轮扫描的确认（更大样本）
   ---------------------------------------------------------------
   第五轮里 selfMargin=0 意外拿了 98.8%，比默认的 6 高 8.8 个点。
   样本只有 80 回合，可能是噪声，这里用 2~3 倍样本 + 不同种子复核。
   用法：node test/sweep6.mjs [每组场数] [分片号] [分片数] [种子基数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure, resetConfig } from '../policy.js';

const matches = parseInt(process.argv[2] || '24', 10);
const chunk = parseInt(process.argv[3] || '0', 10);
const chunkCount = parseInt(process.argv[4] || '1', 10);
const seedBase = parseInt(process.argv[5] || '7000', 10);
const { TT } = loadGame();

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
  { name: '余量0', cfg: { selfMargin: 0 } },
  { name: '余量3', cfg: { selfMargin: 3 } },
  { name: '余量6(当前默认)', cfg: { selfMargin: 6 } },
  { name: '余量0+增益18', cfg: { selfMargin: 0, dodgeGain: 18 } },
  { name: '余量0+增益24', cfg: { selfMargin: 0, dodgeGain: 24 } },
  { name: '余量0+增益12', cfg: { selfMargin: 0, dodgeGain: 12 } },
  { name: '余量0+阻挡1', cfg: { selfMargin: 0, blockRule: 1 } },
  { name: '余量0+阻挡0', cfg: { selfMargin: 0, blockRule: 0 } },
  { name: '余量0+只按当前位置', cfg: { selfMargin: 0, guardMode: 0 } },
  { name: '余量0+闪避增益0', cfg: { selfMargin: 0, dodgeGain: 0 } },
];

const TOTAL = matches * 5;

console.log(`确认扫描：每组 ${matches} 场 × 8 辆（席位轮换）种子基数 ${seedBase}  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(88));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  resetConfig(); configure(v.cfg);
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(seedBase + i * 7919);
    const idx = i % 8;
    res.push(runMatch(TT, policy, { policyIndex: idx, tankCount: 8, roundsToWin: 5 }));
  }
  const s = summarize(res, 0);
  let win = 0, self = 0, death = 0, kill = 0;
  for (const r of res) {
    win += r.scores[r.policyIndex];
    self += r.selfKills[r.policyIndex];
    death += r.deaths[r.policyIndex];
    kill += r.kills[r.policyIndex];
  }
  rows.push({
    name: v.name, win, rate: (win / TOTAL * 100), kill, self, death,
    to: s.timeouts, rounds: s.rounds, dur: s.avgRound, errors: s.errors,
  });
}

rows.sort((a, b) => b.rate - a.rate);
for (const r of rows) {
  console.log(
    `  ${r.name.padEnd(22, ' ')} 胜率 ${r.rate.toFixed(1).padStart(5)}%` +
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
