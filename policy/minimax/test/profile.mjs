/* ===========================================================
   minimax policy · 性能剖析
   用法：node test/profile.mjs [场数]
   =========================================================== */

import { loadGame, runMatch } from './harness.mjs';
import policy from '../policy.js';

const matches = parseInt(process.argv[2] || '2', 10);
const { TT } = loadGame();

/* 统计策略调用次数与耗时 */
let calls = 0;
let total = 0;
let worst = 0;
const wrapped = function (tank, game, dt) {
  const t0 = process.hrtime.bigint();
  const r = policy(tank, game, dt);
  const t1 = process.hrtime.bigint();
  const ms = Number(t1 - t0) / 1e6;
  calls++;
  total += ms;
  if (ms > worst) worst = ms;
  return r;
};

const t0 = Date.now();
const res = [];
for (let i = 0; i < matches; i++) res.push(runMatch(TT, wrapped, { policyIndex: 0, tankCount: 8 }));
const wall = Date.now() - t0;

const simSeconds = res.reduce((a, r) => a + r.seconds, 0);
const frames = Math.round(simSeconds * 60);

console.log(`场次            : ${matches}`);
console.log(`模拟时长        : ${simSeconds.toFixed(1)}s（${frames} 帧）`);
console.log(`策略调用        : ${calls} 次（平均每帧 ${(calls / frames).toFixed(2)} 次）`);
console.log(`策略总耗时      : ${total.toFixed(0)}ms`);
console.log(`平均单次        : ${(total / calls).toFixed(3)}ms`);
console.log(`最慢单次        : ${worst.toFixed(2)}ms`);
console.log(`策略占实时比例  : ${((total / (wall)) * 100).toFixed(1)}%（单进程内）`);
console.log(`平均帧预算占比  : ${((total / frames) / 16.67 * 100).toFixed(1)}%（16.67ms/帧）`);
