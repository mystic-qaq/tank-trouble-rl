/* ===========================================================
   metric-check.mjs — 用 minimax 自带测试台同时算两种「胜率」
   -----------------------------------------------------------
   minimax 的 sweep8 报告「回合胜率 95.8%（115/120 回合）」，
   但它的分母是 matches × roundsToWin(5) × seeds —— 假设每场正好 5 回合。
   实际一场「先到 5 胜」往往打十几回合，所以真实分母应该是实际回合数。

   本脚本用 minimax 自己的 harness，同时输出：
     · match points 率（= 它 README 用的口径）
     · 真实每回合胜率（= 实际赢下的回合 / 实际打出的回合）
   用来判断竞技场口径和各家 README 口径的差异。
   =========================================================== */

import { loadGame, runMatch, summarize } from '../policy/minimax/test/harness.mjs';
import policy from '../policy/minimax/policy.js';

const MATCHES = parseInt(process.argv[2] || '12', 10);
const SEEDS = [11, 22, 33];

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const { TT } = loadGame();

let points = 0;      // minimax 赢下的回合数（README 口径的分子）
let realRounds = 0;  // 实际打出的回合总数（真实分母）
let matches = 0;

for (const sb of SEEDS) {
  for (let i = 0; i < MATCHES; i++) {
    installRandom(sb + i * 7919);
    const idx = i % 8;
    const r = runMatch(TT, policy, { policyIndex: idx, tankCount: 8, roundsToWin: 5 });
    points += r.scores[idx];
    realRounds += r.rounds;
    matches++;
  }
}

console.log('\n▶ minimax 自带测试台 · ' + MATCHES + ' 场/种子 × ' + SEEDS.length + ' 种子\n');
console.log('  实际打出的回合总数        : ' + realRounds);
console.log('  minimax 赢下的回合数      : ' + points);
console.log('');
console.log('  README 口径 (分母 = 场数×5×种子数 = ' + (MATCHES * 5 * SEEDS.length) + ') : ' +
  (points / (MATCHES * 5 * SEEDS.length) * 100).toFixed(1) + '%');
console.log('  真实每回合胜率 (分母 = 实际回合数 ' + realRounds + ')              : ' +
  (points / realRounds * 100).toFixed(1) + '%');
console.log('');
console.log('  → README 的「95%」是「match points 率」，分母假设每场正好 5 回合；');
console.log('    真实每回合胜率要低得多。竞技场用的就是后者（真实每回合胜率）。');
console.log('');
