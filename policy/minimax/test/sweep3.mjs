/* ===========================================================
   minimax policy · 终轮参数扫描
   ---------------------------------------------------------------
   基线 = 第三轮改进后的策略（比例转向 / 卡住自救 / 僵局激进度）。
   在新基线上逐项加减。
   用法：node test/sweep3.mjs [每组场数] [分片号] [分片数]
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

const NEW = {
  dodgeWant: 36,
  aimTol: 0.035,
  standoffMin: 160, standoffMax: 400, standoffIdeal: 270,
  crossRetreat: 2,
  aimMove: 1,
  displaceTime: 0.8,
  maxLiveBullets: 3,
  leadFactor: 0.94,
  rageAfter: 25,
  dodgeHorizon: 1.2,
  bankRange: 420, bankWalls: 10,
};

const VARIANTS = [
  { name: '新基准', cfg: {} },
  { name: '开火容差0.02', cfg: { aimTol: 0.02 } },
  { name: '开火容差0.05', cfg: { aimTol: 0.05 } },
  { name: '闪避阈值30', cfg: { dodgeWant: 30 } },
  { name: '闪避阈值45', cfg: { dodgeWant: 45 } },
  { name: '距离带120-320', cfg: { standoffMin: 120, standoffMax: 320, standoffIdeal: 210 } },
  { name: '距离带200-450', cfg: { standoffMin: 200, standoffMax: 450, standoffIdeal: 300 } },
  { name: '瞄准时站定', cfg: { aimMove: 0 } },
  { name: '瞄准时始终移动', cfg: { aimMove: 2 } },
  { name: '横移0.5s', cfg: { displaceTime: 0.5 } },
  { name: '横移1.0s', cfg: { displaceTime: 1.0 } },
  { name: '关闭激进度', cfg: { rageAfter: 0 } },
  { name: '激进度15s', cfg: { rageAfter: 15 } },
  { name: '提前量0.85', cfg: { leadFactor: 0.85 } },
  { name: '提前量1.0', cfg: { leadFactor: 1.0 } },
  { name: '冷枪范围320', cfg: { bankRange: 320, bankWalls: 8 } },
  { name: '关闭撤退', cfg: { crossRetreat: 0 } },
  { name: '撤退阈值3', cfg: { crossRetreat: 3 } },
];

const TOTAL = matches * 5;

console.log(`终轮扫描：每组 ${matches} 场 × 8 辆坦克  [分片 ${chunk + 1}/${chunkCount}]`);
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
configure(NEW);
