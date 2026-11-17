/* ===========================================================
   steer-audit.mjs — 审计各策略输出的 steer 是否「超物理」
   -----------------------------------------------------------
   背景：竞技场刻意**不截断** steer（因为 deepseek/hy/minimax 用分数转向
   做一帧精确落角，截断会毁掉瞄准）。但引擎的 Tank.update 是
       angle += steer * T.turn * dt
   也就是说 |steer| > 1 意味着**转向速度超过物理上限 3.3 rad/s**。

   如果某个策略系统性输出大 |steer|，它就获得了别人没有的「超速转向」，
   那是真正的 unfair advantage —— 必须查清楚。

   本脚本统计每个策略输出 steer 的分布（均值/绝对值均值/P95/最大值，
   以及 |steer|>1 的占比）。

   用法：node --experimental-vm-modules arena/steer-audit.mjs
   =========================================================== */

import vm from 'node:vm';
import { createArena, mulberry32, roundSeed, seatPlan, mapPlan, STRATEGY_IDS, STRATEGIES } from './arena.mjs';

const env = await createArena();
const { TT, ctx, sandbox, adapters } = env;

sandbox.__rnd = mulberry32(roundSeed(20260101, 0));
vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

const seats = seatPlan(0);
const mapSeed = mapPlan(0);
TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
TT.CFG.ROUND_INTRO = 0;

const game = new TT.Game(env.canvas);
game.render = function () {}; game.resize = function () {};
game.mode = 'match';
game.slots = ['bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'];
game.roundsToWin = 9999;

/* 每个策略的 steer 分布 */
const stats = {};
for (const id of STRATEGY_IDS) {
  stats[id] = { n: 0, sum: 0, absSum: 0, over1: 0, max: 0, hist: new Array(11).fill(0) };
}

TT.updateBot = function (tank, g, dt) {
  const id = seats[tank.index];
  const out = adapters[id](tank, g, dt);
  if (!out) return { drive: 0, steer: 0, shoot: false };

  const s = Number(out.steer) || 0;
  const st = stats[id];
  st.n++;
  st.sum += s;
  st.absSum += Math.abs(s);
  if (Math.abs(s) > 1) st.over1++;
  if (Math.abs(s) > st.max) st.max = Math.abs(s);
  /* 分桶：0 / ±0.1 / ... / >1 */
  const a = Math.abs(s);
  const b = a <= 1 ? Math.min(10, Math.floor(a * 10)) : 10;
  st.hist[b]++;

  return {
    drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
    steer: s,
    shoot: !!out.shoot,
  };
};

game.startRound();
let frames = 0;
while (frames < 60 * 60) {
  game.update(1 / 60);
  frames++;
  if (game.state === 'over') game.startRound();
}

/* ---------------- 输出 ---------------- */

console.log('\n▶ steer 输出审计（60 局 × 8 车）');
console.log('  引擎转向上限 T.turn = 3.3 rad/s；|steer| > 1 即为「超物理转向」\n');
console.log('  策略            样本数    平均|steer|   P95      最大     |steer|>1 占比');
console.log('  ' + '─'.repeat(72));

const order = STRATEGY_IDS.slice().sort((x, y) => stats[y].absSum / stats[y].n - stats[x].absSum / stats[x].n);
for (const id of order) {
  const s = stats[id];
  const label = (STRATEGIES.find((x) => x.id === id) || {}).label || id;
  const avgAbs = s.absSum / s.n;
  /* P95：从直方图估 */
  let acc = 0, p95 = 0;
  for (let b = 0; b < 11; b++) {
    acc += s.hist[b];
    if (acc >= s.n * 0.95) { p95 = b / 10; break; }
  }
  const overPct = (s.over1 / s.n * 100);
  console.log('  ' + label.padEnd(15) + String(s.n).padStart(8) +
    avgAbs.toFixed(3).padStart(13) + p95.toFixed(2).padStart(9) +
    s.max.toFixed(2).padStart(9) + (overPct.toFixed(2) + '%').padStart(14));
}

console.log('\n  直方图（|steer| 分桶，0=0.0-0.1 … 9=0.9-1.0, 10=>1.0）:');
for (const id of order) {
  const s = stats[id];
  const label = (STRATEGIES.find((x) => x.id === id) || {}).label || id;
  console.log('  ' + label.padEnd(15) + s.hist.map((c) => String(c).padStart(7)).join(''));
}
console.log('');
