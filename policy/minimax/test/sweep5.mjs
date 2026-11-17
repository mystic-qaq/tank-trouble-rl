/* ===========================================================
   minimax policy · 自伤/闪避机制扫描（第五轮）
   ---------------------------------------------------------------
   这一轮针对三个新发现的机制开关做 A/B：
     guardMode  自伤判定：只看当前位置(0) 还是看"我移动后的位置"(1)
     dodgeGain  闪避触发：只按"最优躲避净空"判(0) 还是和"原地不动"比(>0)
     dodgeWins  闪避是否压过"开火后横移"
     blockRule  敌人能不能替我挡枪
     selfMargin 自伤判定的额外余量
   席位轮换，避免 P1 出生点红利干扰结论。
   用法：node test/sweep5.mjs [每组场数] [分片号] [分片数]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy, { configure, resetConfig } from '../policy.js';

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

const BASE = {};   /* 用 policy.js 里的默认值 */

const VARIANTS = [
  { name: '新默认', cfg: {} },
  { name: '旧:只看当前位置', cfg: { guardMode: 0 } },
  { name: '旧:闪避只看最优净空', cfg: { dodgeGain: 0 } },
  { name: '旧:横移盖过闪避', cfg: { dodgeWins: 0 } },
  { name: '旧:按速度判阻挡', cfg: { blockRule: 1 } },
  { name: '不信阻挡', cfg: { blockRule: 0 } },
  { name: '余量0', cfg: { selfMargin: 0 } },
  { name: '余量20', cfg: { selfMargin: 20 } },
  { name: '闪避增益6', cfg: { dodgeGain: 6 } },
  { name: '闪避增益24', cfg: { dodgeGain: 24 } },
  { name: '闪避增益40', cfg: { dodgeGain: 40 } },
  { name: '全旧行为', cfg: { guardMode: 0, dodgeGain: 0, dodgeWins: 0, blockRule: 1 } },
  { name: '仅运动判定', cfg: { dodgeGain: 0, dodgeWins: 0, blockRule: 2 } },
  { name: '仅闪避修正', cfg: { guardMode: 0, dodgeGain: 12, dodgeWins: 1, blockRule: 2 } },
];

const TOTAL = matches * 5;

console.log(`自伤/闪避扫描：每组 ${matches} 场 × 8 辆（席位轮换）  [分片 ${chunk + 1}/${chunkCount}]`);
console.log('─'.repeat(88));

const rows = [];
const mine = VARIANTS.filter((_, i) => i % chunkCount === chunk);
for (const v of mine) {
  resetConfig(); configure({ ...BASE, ...v.cfg });
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(3000 + i * 7919);
    /* 席位轮换：把 P1 的出生红利摊平 */
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
    name: v.name,
    win, rate: (win / TOTAL * 100),
    kill, self, death,
    to: s.timeouts, rounds: s.rounds,
    dur: s.avgRound, errors: s.errors,
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
