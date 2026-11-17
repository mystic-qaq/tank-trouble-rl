/* ===========================================================
   minimax policy · 组合确认扫描
   ---------------------------------------------------------------
   把终轮扫描里单项最好的参数组合起来，跟单项最优逐一对照，
   用更大的样本量确认最终配置。
   用法：node test/sweep4.mjs [每组场数] [分片号] [分片数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure } from '../policy.js';

const matches = parseInt(process.argv[2] || '20', 10);
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

/* 组合最优 */
const BEST = {
  dodgeWant: 40,
  aimTol: 0.05,
  standoffMin: 120, standoffMax: 320, standoffIdeal: 210,
  crossRetreat: 2,
  aimMove: 0,
  displaceTime: 0.8,
  maxLiveBullets: 3,
  leadFactor: 0.94,
  rageAfter: 25,
  dodgeHorizon: 1.2,
  bankRange: 320, bankWalls: 8,
};

const VARIANTS = [
  { name: '组合最优', cfg: {} },
  { name: '组合+瞄准时移动', cfg: { aimMove: 1 } },
  { name: '组合+闪避45', cfg: { dodgeWant: 45 } },
  { name: '组合+横移1.0', cfg: { displaceTime: 1.0 } },
  { name: '组合+距离带160-400', cfg: { standoffMin: 160, standoffMax: 400, standoffIdeal: 270 } },
  { name: '组合+容差0.04', cfg: { aimTol: 0.04 } },
  { name: '组合+撤退阈值3', cfg: { crossRetreat: 3 } },
  { name: '组合+冷枪420', cfg: { bankRange: 420, bankWalls: 10 } },
  { name: '终轮基线', cfg: { dodgeWant: 36, aimTol: 0.035, standoffMin: 160, standoffMax: 400, standoffIdeal: 270, aimMove: 1, displaceTime: 0.8, bankRange: 420, bankWalls: 10 } },
  { name: '组合+闪避35', cfg: { dodgeWant: 35 } },
  { name: '组合+提前量0.9', cfg: { leadFactor: 0.9 } },
  { name: '组合+激进40s', cfg: { rageAfter: 40 } },
];

const TOTAL = matches * 5;

console.log(`组合确认：每组 ${matches} 场 × 8 辆坦克  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(74));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  configure({ ...BEST, ...v.cfg });
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(2000 + i * 7919);
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
    dur: s.avgRound,
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
    `  超时 ${r.to}` +
    `  回合 ${r.dur.toFixed(1)}s`
  );
}
configure(BEST);
