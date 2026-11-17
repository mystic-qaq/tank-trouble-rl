/* ===========================================================
   minimax policy · 位置偏差溯源
   ---------------------------------------------------------------
   镜像对局里 P1 明显更强。两种可能：
     (a) 出生点分配有系统性优势（第一个挑到的格子更"好"）
     (b) 子弹命中判定按 tanks 数组顺序"先到先得"，
         同一发子弹擦到两辆坦克时，数组里靠前的先死 → 低 index 占便宜
   实验：把每回合的 tanks 数组顺序反转，(b) 的优势应该跟着数组位置跑。
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy from '../policy.js';

const matches = parseInt(process.argv[2] || '10', 10);
const { TT } = loadGame();

function run(rotate) {
  const orig = TT.Game.prototype.startRound;
  if (rotate) {
    TT.Game.prototype.startRound = function () {
      orig.call(this);
      this.tanks.reverse();
    };
  }
  const res = [];
  for (let i = 0; i < matches; i++) {
    res.push(runMatch(TT, policy, { policyIndex: 0, tankCount: 8, roundsToWin: 5 }));
  }
  TT.Game.prototype.startRound = orig;
  return summarize(res, 0);
}

function show(title, s) {
  const avg = s.wins.reduce((a, b) => a + b, 0) / 8;
  console.log('\n' + title);
  const line = s.wins.map((w, i) => `P${i + 1}:${w}`).join('  ');
  console.log('  ' + line);
  console.log(`  平均 ${avg.toFixed(1)}  最高 ${Math.max(...s.wins)}  最低 ${Math.min(...s.wins)}  差距 ${Math.max(...s.wins) - Math.min(...s.wins)}`);
}

const normal = run(false);
show(`正常顺序（${matches} 场）`, normal);

const rev = run(true);
show(`数组顺序反转（${matches} 场）`, rev);

console.log('\n判读：');
console.log('  若反转后优势跟着"数组位置"跑（原 P8 变强） → 是命中判定的顺序优势（b）');
console.log('  若优势仍固定在 P1 → 是出生点分配带来的（a）');
