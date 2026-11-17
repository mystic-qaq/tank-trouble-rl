/* ===========================================================
   minimax policy · 弹道推演保真度校验
   ---------------------------------------------------------------
   策略的一切都建立在"traceBullet 算出来的弹道 == 游戏里真实的弹道"
   这个假设上。这里直接在真实对局里抓一颗子弹的每一帧位置，
   跟 traceBullet 的结果逐帧比对。
   用法：node test/verify-trace.mjs [次数]
   =========================================================== */

import { loadGame } from './harness.mjs';
import { traceBullet, getGrid } from '../policy.js';

const tries = parseInt(process.argv[2] || '40', 10);
const { TT } = loadGame();

/* 造一个假 DOM + 一个真实 Game 对象 */
const canvas = {
  width: 1280, height: 800, style: {},
  getContext: () => new Proxy({}, { get: () => () => {} }),
  getBoundingClientRect: () => ({ width: 1280, height: 800, left: 0, top: 0, right: 1280, bottom: 800 }),
  addEventListener() {}, removeEventListener() {},
};

const game = new TT.Game(canvas);
game.startMatch(['bot'], 3);

/* 随机挑一辆坦克，随机角度开一枪，逐帧记录子弹位置 */
let maxErr = 0, tested = 0, mismatch = 0;
const DT = 1 / 60;

for (let n = 0; n < tries; n++) {
  const tank = game.tanks[0];
  if (!tank || !tank.alive) { game.startRound(); continue; }

  const ang = Math.random() * Math.PI * 2;
  const tip = { x: tank.x + Math.cos(tank.angle) * (tank.radius + 12), y: tank.y + Math.sin(tank.angle) * (tank.radius + 12) };

  /* 先把游戏里的子弹清干净，只留一颗 */
  game.bullets.length = 0;
  game.bullets.push(new TT.Bullet(tip.x, tip.y, ang, tank.index, tank.color));

  const real = [];
  const b = game.bullets[0];
  for (let f = 0; f < 180 && !b.dead; f++) {
    b.update(DT, game);
    real.push({ x: b.x, y: b.y, bounces: b.bounces });
    if (b.dead) break;
  }

  /* 用策略的 traceBullet 重算同一条弹道 */
  const tr = traceBullet(game.map, game.tanks, tank.index, tip.x, tip.y, ang, 3.0, DT, null, true);
  const st = tr.steps;

  /* 逐帧比对（真实子弹死了就只比到它死为止） */
  const n2 = Math.min(real.length, st.length - 1);
  let err = 0;
  for (let f = 0; f < n2; f++) {
    const d = Math.hypot(real[f].x - st[f + 1].x, real[f].y - st[f + 1].y);
    if (d > err) err = d;
  }
  tested++;
  if (err > maxErr) maxErr = err;
  if (err > 1.0) {
    mismatch++;
    console.log(`  #${n}: 偏差 ${err.toFixed(3)}px  真实弹跳 ${real[real.length - 1].bounces} / 推演 ${tr.bounces}`);
  }
}

console.log(`弹道保真度：比对 ${tested} 条弹道，帧级最大偏差 ${maxErr.toFixed(4)}px，偏差>1px 的有 ${mismatch} 条`);
console.log(maxErr <= 1.0
  ? '✅ traceBullet 与游戏真实弹道一致'
  : '❌ 存在偏差，策略的弹道判断不可信');
