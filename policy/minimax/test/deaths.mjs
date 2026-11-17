/* ===========================================================
   minimax policy · 阵亡分析
   看看自己是怎么死的：被谁打死、距离多远、当时在干嘛。
   =========================================================== */

import { loadGame, runMatch } from './harness.mjs';
import policy from '../policy.js';

const matches = parseInt(process.argv[2] || '14', 10);
const { TT } = loadGame();

const all = [];
let roundsWon = 0, rounds = 0;
for (let i = 0; i < matches; i++) {
  const r = runMatch(TT, policy, { policyIndex: 0, tankCount: 8, roundsToWin: 5 });
  all.push(...r.deathLog);
  rounds += r.rounds;
  for (const w of r.roundWinners) if (w === 0) roundsWon++;
}

console.log(`阵亡分析：${matches} 场，共 ${all.length} 次阵亡，胜率 ${(roundsWon / rounds * 100).toFixed(1)}%`);

const byDist = { '近(<120)': 0, '中(120-260)': 0, '远(>260)': 0 };
const byIncoming = { 0: 0, 1: 0, '2+': 0 };
const byMyBullets = { 0: 0, 1: 0, '2+': 0 };
const bySpeed = { '慢(<40)': 0, '中(40-120)': 0, '快(>120)': 0 };
const killers = new Map();

for (const d of all) {
  if (d.self) continue;
  if (d.dist < 120) byDist['近(<120)']++;
  else if (d.dist < 260) byDist['中(120-260)']++;
  else byDist['远(>260)']++;

  if (d.incoming === 0) byIncoming[0]++;
  else if (d.incoming === 1) byIncoming[1]++;
  else byIncoming['2+']++;

  if (d.myBullets === 0) byMyBullets[0]++;
  else if (d.myBullets === 1) byMyBullets[1]++;
  else byMyBullets['2+']++;

  if (d.speed < 40) bySpeed['慢(<40)']++;
  else if (d.speed < 120) bySpeed['中(40-120)']++;
  else bySpeed['快(>120)']++;

  killers.set(d.killer, (killers.get(d.killer) || 0) + 1);
}

function row(title, obj) {
  console.log('\n' + title);
  const tot = Object.values(obj).reduce((a, b) => a + b, 0) || 1;
  for (const [k, v] of Object.entries(obj)) {
    console.log(`  ${String(k).padStart(12)} : ${String(v).padStart(4)}  ${(v / tot * 100).toFixed(0)}%  ${'█'.repeat(Math.round(v / tot * 30))}`);
  }
}

row('被打死时 子弹距离', byDist);
row('被打死时 场上有几发敌方子弹', byIncoming);
row('被打死时 我自己有几发在场', byMyBullets);
row('被打死时 我的速度', bySpeed);
console.log('\n凶手分布（P编号 → 击杀次数）');
console.log('  ' + [...killers.entries()].sort((a, b) => b[1] - a[1])
  .map(([k, v]) => `P${k + 1}:${v}`).join('  '));
