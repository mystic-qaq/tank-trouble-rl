/* ===========================================================
   minimax policy · 基准测试
   ---------------------------------------------------------------
   用法：
     node test/bench.mjs [场数] [坦克数] [--seed=xxx]

   会跑两轮：
     A) index 0 = minimax 策略，其余 = 游戏内置 AI
     B) 全部 = 游戏内置 AI（基线，用来对比）
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy from '../policy.js';

const args = process.argv.slice(2);
const matches = parseInt(args[0] || '12', 10);
const tankCount = parseInt(args[1] || '8', 10);
const noBase = args.includes('--nobase');
const shuffle = args.includes('--shuffle');
const seedArg = args.find((a) => a.startsWith('--seed='));
let seed = seedArg ? parseInt(seedArg.split('=')[1], 10) : 20260101;

/* 简单的可复现随机：覆盖 Math.random，让同一 seed 结果一致 */
function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function printTable(title, sum, policyIndex, markPolicy = true) {
  console.log('\n' + title);
  console.log('─'.repeat(64));
  const rows = [];
  for (let i = 0; i < 8; i++) {
    if (sum.n === 0) break;
    rows.push({ i, win: sum.wins[i], kill: sum.kills[i], death: sum.deaths[i], self: sum.selfKills[i] });
  }
  rows.sort((a, b) => b.win - a.win || b.kill - a.kill);
  for (const r of rows) {
    const tag = (markPolicy && r.i === policyIndex) ? '  ← minimax' : '';
    console.log(
      `  P${r.i + 1}  胜场 ${String(r.win).padStart(4)}  净击杀 ${String(r.kill).padStart(5)}` +
      `  死亡 ${String(r.death).padStart(4)}  自杀 ${String(r.self).padStart(4)}${tag}`
    );
  }
  console.log('─'.repeat(64));
}

console.log(`基准测试：${matches} 场 × ${tankCount} 辆坦克   seed=${seed}${shuffle ? '  [席位随机]' : ''}`);

const { TT } = loadGame();
const builtin = TT.updateBot;   // 原始内置 AI（runMatch 结束后会还原，先存一份）

/* ---- A) minimax vs 内置 AI ---- */
installRandom(seed);
const mine = [];
const usedIdx = [];
for (let i = 0; i < matches; i++) {
  const idx = shuffle ? Math.floor(Math.random() * tankCount) : 0;
  usedIdx.push(idx);
  const r = runMatch(TT, policy, { policyIndex: idx, tankCount, roundsToWin: 5 });
  mine.push(r);
  if (r.policyError) {
    console.error('策略异常！', r.policyError);
    break;
  }
}
const sumMine = summarize(mine, 0);
printTable('A. minimax vs 内置 AI', sumMine, 0, !shuffle);

/* 策略自己的成绩（随机席位时也按实际席位统计） */
const pMine = sumMine.policy;
if (shuffle) {
  const bySeat = new Array(8).fill(0);
  for (const r of mine) bySeat[r.policyIndex] += r.scores[r.policyIndex];
  console.log('  策略按席位分布：' + bySeat.map((v, i) => `P${i + 1}:${v}`).join('  '));
}

/* ---- B) 基线：全是内置 AI ---- */
let sumBase = null;
if (!noBase) {
  installRandom(seed);
  const base = [];
  for (let i = 0; i < matches; i++) {
    base.push(runMatch(TT, (t, g, d) => builtin(t, g, d), {
      policyIndex: 0, tankCount, roundsToWin: 5,
    }));
  }
  sumBase = summarize(base, 0);
  printTable('B. 基线：全是内置 AI', sumBase, -1, false);
}

/* ---- 结论 ---- */
const mineWin = pMine.wins;
const baseAvg = sumBase ? sumBase.wins.reduce((a, b) => a + b, 0) / Math.max(1, tankCount) : 0;
console.log('\n结论');
console.log(`  minimax 胜场        : ${mineWin} / ${matches * 5}`);
if (sumBase) console.log(`  内置 AI 平均胜场    : ${baseAvg.toFixed(1)} / ${matches * 5}`);
console.log(`  minimax 净击杀      : ${pMine.kills}（自杀 ${pMine.selfKills} 次，死亡 ${pMine.deaths} 次）`);
if (sumBase) console.log(`  基线平均自杀        : ${(sumBase.selfKills.reduce((a, b) => a + b, 0) / 8).toFixed(1)} 次`);
console.log(`  minimax 回合超时    : ${sumMine.timeouts} / ${sumMine.rounds} 回合`);
if (sumBase) console.log(`  基线回合超时        : ${sumBase.timeouts} / ${sumBase.rounds} 回合`);
console.log(`  平均回合时长        : minimax ${sumMine.avgRound.toFixed(1)}s` + (sumBase ? ` / 基线 ${sumBase.avgRound.toFixed(1)}s` : ''));
console.log(`  minimax 策略异常    : ${sumMine.errors}`);
if (sumBase) console.log(`  相对基线            : ${mineWin >= baseAvg ? '+' : ''}${(mineWin - baseAvg).toFixed(1)} 场`);
