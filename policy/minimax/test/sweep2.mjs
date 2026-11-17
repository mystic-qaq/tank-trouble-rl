/* ===========================================================
   minimax policy · 聚焦参数扫描（第二轮）
   ---------------------------------------------------------------
   第一轮 24 组扫描给出的方向：
     闪避更激进(dodgeWant 30~40) / 开火容差 0.05 / 距离带更近(120-320)
     / 关闭撤退 / 瞄准时始终移动 / 横移 0.8s
   这一轮把这些单独效果好的参数组合起来，再逐个微调。
   用法：node test/sweep2.mjs [每组场数] [分片号] [分片数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure } from '../policy.js';

const matches = parseInt(process.argv[2] || '16', 10);
const chunk = parseInt(process.argv[3] || '0', 10);
const chunkCount = parseInt(process.argv[4] || '1', 10);
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

/* 第二轮候选：先定"新基准"，再逐项加减 */
const NEW = {
  dodgeWant: 30,
  aimTol: 0.05,
  standoffMin: 120, standoffMax: 320, standoffIdeal: 210,
  crossRetreat: 0,
  aimMove: 2,
  displaceTime: 0.8,
  maxLiveBullets: 3,
};

const VARIANTS = [
  { name: '新基准', cfg: {} },
  { name: '撤退阈值2', cfg: { crossRetreat: 2 } },
  { name: '撤退阈值3', cfg: { crossRetreat: 3 } },
  { name: '瞄准时被瞄准才动', cfg: { aimMove: 1 } },
  { name: '瞄准时站定', cfg: { aimMove: 0 } },
  { name: '横移0.5s', cfg: { displaceTime: 0.5 } },
  { name: '横移1.0s', cfg: { displaceTime: 1.0 } },
  { name: '距离带100-280', cfg: { standoffMin: 100, standoffMax: 280, standoffIdeal: 180 } },
  { name: '距离带160-400', cfg: { standoffMin: 160, standoffMax: 400, standoffIdeal: 260 } },
  { name: '闪避阈值24', cfg: { dodgeWant: 24 } },
  { name: '闪避阈值36', cfg: { dodgeWant: 36 } },
  { name: '开火容差0.04', cfg: { aimTol: 0.04 } },
  { name: '开火容差0.065', cfg: { aimTol: 0.065 } },
  { name: '闪避时长1.5s', cfg: { dodgeHorizon: 1.5 } },
  { name: '闪避时长1.0s', cfg: { dodgeHorizon: 1.0 } },
  { name: '弹道验证2.9s', cfg: { traceHorizon: 2.9 } },
  { name: '冷枪范围420', cfg: { bankRange: 420, bankWalls: 12 } },
  { name: '候选3个', cfg: { planCandidates: 3 } },
];

const TOTAL = matches * 5;

console.log(`聚焦扫描：每组 ${matches} 场 × 8 辆坦克  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(74));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  configure({ ...NEW, ...v.cfg });
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(1000 + i * 7919);
    res.push(runMatch(TT, policy, { policyIndex: 0, tankCount: 8, roundsToWin: 5 }));
  }
  const s = summarize(res, 0);
  rows.push({
    name: v.name,
    win: s.wins[0],
    rate: (s.wins[0] / TOTAL * 100),
    kill: s.kills[0],
    self: s.selfKills[0],
    death: s.deaths[0],
    to: s.timeouts,
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
    `  超时 ${r.to}`
  );
}
configure(NEW);
