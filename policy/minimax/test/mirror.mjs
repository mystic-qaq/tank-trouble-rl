/* ===========================================================
   minimax policy · 镜像对局（自己打自己）
   ---------------------------------------------------------------
   8 辆坦克全用 minimax 策略：
     - 如果胜场分布均匀 → 没有"固定席位/先手"优势，可被针对的漏洞少
     - 如果某席位明显吃紧 → 说明策略对出生位置有偏好，需要修
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy from '../policy.js';

const matches = parseInt(process.argv[2] || '10', 10);
const { TT } = loadGame();

/* 让每个席位都用 minimax 策略 */
const results = [];
for (let i = 0; i < matches; i++) {
  const r = runMatch(TT, policy, { policyIndex: 0, tankCount: 8, roundsToWin: 5 });
  results.push(r);
}

const sum = summarize(results, 0);
console.log(`镜像对局：${matches} 场 × 8 × minimax`);
console.log('─'.repeat(40));
const avg = sum.wins.reduce((a, b) => a + b, 0) / 8;
for (let i = 0; i < 8; i++) {
  const dev = sum.wins[i] - avg;
  const bar = '█'.repeat(Math.round(sum.wins[i] / 4));
  console.log(`  P${i + 1}  胜场 ${String(sum.wins[i]).padStart(4)}  ${bar} ${dev >= 0 ? '+' : ''}${dev.toFixed(1)}`);
}
console.log('─'.repeat(40));
console.log(`  平均 ${avg.toFixed(1)}（理论 ${(matches * 5 / 8).toFixed(1)}）`);
console.log(`  自杀合计 ${sum.selfKills.reduce((a, b) => a + b, 0)} 次`);
console.log(`  回合超时 ${sum.timeouts} / ${sum.rounds}`);
const spread = Math.max(...sum.wins) - Math.min(...sum.wins);
console.log(`  席位差距 ${spread}（越小越说明没有位置偏好）`);
