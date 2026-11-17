/* ===========================================================
   minimax policy · 镜像场景 A/B（最接近 Arena）
   ---------------------------------------------------------------
   8 辆坦克全部用本策略互相打 —— 这是 Arena 的真实形态：
   每个座位都是"有模有样"的策略，比拼的是谁活得更久。
   这里对比几组配置，看新加的防守逻辑是不是变得太保守。
   用法：node test/sweep7.mjs [每组场数] [分片号] [分片数] [种子基数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure, resetConfig } from '../policy.js';

const matches = parseInt(process.argv[2] || '16', 10);
const chunk = parseInt(process.argv[3] || '0', 10);
const chunkCount = parseInt(process.argv[4] || '1', 10);
const seedBase = parseInt(process.argv[5] || '9000', 10);
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
  { name: '新默认', cfg: {} },
  { name: '旧:全旧行为', cfg: { guardMode: 0, dodgeGain: 0, dodgeWins: 0, blockRule: 1 } },
  { name: '余量0', cfg: { selfMargin: 0 } },
  { name: '新默认+闪避增益24', cfg: { dodgeGain: 24 } },
  { name: '新默认+只按当前位置', cfg: { guardMode: 0 } },
  { name: '新默认+横移盖过闪避', cfg: { dodgeWins: 0 } },
  { name: '新默认+闪避只看最优', cfg: { dodgeGain: 0 } },
  { name: '新默认+余量0+增益24', cfg: { selfMargin: 0, dodgeGain: 24 } },
];

console.log(`镜像 A/B：每组 ${matches} 场 × 8 辆 minimax，种子基数 ${seedBase}  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(92));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  resetConfig(); configure(v.cfg);
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(seedBase + i * 7919);
    res.push(runMatch(TT, policy, { policyIndex: 0, tankCount: 8, roundsToWin: 5, roundTimeout: 60 }));
  }
  const s = summarize(res, 0);
  /* 镜像：8 个座位全是自己，谁赢得最多就是最强 */
  const seats = s.wins.slice(0, 8);
  const best = Math.max(...seats);
  const avg = seats.reduce((a, b) => a + b, 0) / 8;
  const selfTotal = s.selfKills.reduce((a, b) => a + b, 0);
  rows.push({
    name: v.name, best, avg, s,
    rounds: s.rounds, selfTotal,
    rate: (avg / Math.max(1, s.rounds) * 100),
    to: s.timeouts, dur: s.avgRound, errors: s.errors,
  });
}

rows.sort((a, b) => b.avg - a.avg);
for (const r of rows) {
  console.log(
    `  ${r.name.padEnd(22, ' ')} 平均席位胜场 ${r.avg.toFixed(1).padStart(5)}` +
    `  (${r.rate.toFixed(1).padStart(5)}% 回合)  最佳席位 ${String(r.best).padStart(3)}` +
    `  全场自杀 ${String(r.selfTotal).padStart(4)}` +
    `  超时 ${String(r.to).padStart(3)}/${String(r.rounds).padStart(4)}` +
    `  回合 ${r.dur.toFixed(1)}s` +
    (r.errors ? `  ⚠异常${r.errors}` : '')
  );
}
resetConfig();
