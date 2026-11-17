/* ===========================================================
   minimax policy · 参数扫描
   ---------------------------------------------------------------
   每个变体跑同样的对局序列（同一 seed → 同一批出生点），
   这样差异只来自策略本身。
   用法：node test/sweep.mjs [每组场数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure } from '../policy.js';

const matches = parseInt(process.argv[2] || '8', 10);
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

const VARIANTS = [
  { name: '基准（当前）', cfg: {} },
  { name: '瞄准时永远移动', cfg: { aimMove: 2 } },
  { name: '瞄准时不动', cfg: { aimMove: 0 } },
  { name: '关闭交叉火力撤退', cfg: { crossRetreat: 0 } },
  { name: '交叉火力撤退阈值3', cfg: { crossRetreat: 3 } },
  { name: '在场子弹上限2', cfg: { maxLiveBullets: 2 } },
  { name: '在场子弹上限4', cfg: { maxLiveBullets: 4 } },
  { name: '在场子弹上限5', cfg: { maxLiveBullets: 5 } },
  { name: '闪避更保守(70)', cfg: { dodgeWant: 70 } },
  { name: '闪避更激进(40)', cfg: { dodgeWant: 40 } },
  { name: '安全距离带更近(120-320)', cfg: { standoffMin: 120, standoffMax: 320, standoffIdeal: 210 } },
  { name: '安全距离带更远(220-520)', cfg: { standoffMin: 220, standoffMax: 520, standoffIdeal: 330 } },
  { name: '开火角度容差 0.05', cfg: { aimTol: 0.05 } },
  { name: '开火角度容差 0.015', cfg: { aimTol: 0.015 } },
  { name: '横移时间 0.6s', cfg: { displaceTime: 0.6 } },
  { name: '横移时间 0.25s', cfg: { displaceTime: 0.25 } },
  { name: '闪避阈值30', cfg: { dodgeWant: 30 } },
  { name: '闪避阈值50', cfg: { dodgeWant: 50 } },
  { name: '开火容差0.08', cfg: { aimTol: 0.08 } },
  { name: '开火容差0.12', cfg: { aimTol: 0.12 } },
  { name: '横移0.8s', cfg: { displaceTime: 0.8 } },
  { name: '组合A', cfg: { crossRetreat: 0, dodgeWant: 40, aimTol: 0.05, displaceTime: 0.6, aimMove: 2 } },
  { name: '组合B', cfg: { crossRetreat: 0, dodgeWant: 30, aimTol: 0.08, displaceTime: 0.6, aimMove: 2 } },
  { name: '组合C', cfg: { crossRetreat: 3, dodgeWant: 40, aimTol: 0.05, displaceTime: 0.6, aimMove: 1 } },
];

const BASE = { aimMove: 1, crossRetreat: 2, maxLiveBullets: 3, dodgeWant: 56 };
const TOTAL = matches * 5;

console.log(`参数扫描：每组 ${matches} 场 × 8 辆坦克  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(74));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  configure({ ...BASE, ...v.cfg });
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
configure(BASE);
