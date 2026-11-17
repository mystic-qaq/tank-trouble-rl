/* ===========================================================
   minimax policy · 自杀机理判别实验
   ---------------------------------------------------------------
   回答一个问题：自杀的时候，子弹的路径有没有经过"我开火那一刻的位置"？
     A. 经过 → 用"当前位置"判就能拦住，问题在别处
     B. 没经过 → 是我自己走进子弹的轨迹里，
                 必须用"我移动后的位置"（运动轨迹）来判
   用法：node test/selfwhy.mjs [对手名] [场数]
   =========================================================== */

import { loadGame, runMatch } from './harness.mjs';
import policy from '../policy.js';

const which = (process.argv[2] || 'sniper');
const matches = parseInt(process.argv[3] || '6', 10);

const { TT } = loadGame();
const U = TT.util;

/* ---- 对手风格 ---- */
function sniper(tank, game, dt) {
  const inp = { drive: 0, steer: 0, shoot: false };
  if (!tank.ai) tank.ai = { j: Math.random() * 6.28 };
  const st = tank.ai;
  st.j += dt * 0.5;
  let target = null, best = Infinity;
  for (const o of game.tanks) {
    if (o === tank || !o.alive) continue;
    if (!game.map.hasLOS(tank.x, tank.y, o.x, o.y)) continue;
    const d = Math.hypot(o.x - tank.x, o.y - tank.y);
    if (d < best) { best = d; target = o; }
  }
  if (!target) {
    const cx = 640 + Math.cos(st.j) * 380;
    const cy = 400 + Math.sin(st.j * 1.3) * 260;
    const want = Math.atan2(cy - tank.y, cx - tank.x);
    const diff = U.angleDiff(tank.angle, want);
    if (Math.abs(diff) > 0.05) inp.steer = U.sign(diff);
    else inp.drive = 1;
    return inp;
  }
  const lead = best / TT.CFG.BULLET.speed;
  const px = target.x + Math.cos(target.angle) * target.speed * lead;
  const py = target.y + Math.sin(target.angle) * target.speed * lead;
  const want = Math.atan2(py - tank.y, px - tank.x);
  const diff = U.angleDiff(tank.angle, want);
  if (Math.abs(diff) > 0.02) inp.steer = U.sign(diff);
  if (best < 260) inp.drive = -1;
  else if (best > 420) inp.drive = 1;
  if (Math.abs(diff) < 0.035) inp.shoot = true;
  return inp;
}

const OPPONENTS = { sniper };

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---- 记录每颗子弹的"出生档案"：出生点、角度、主人当时在哪 ---- */
const births = new Map();     /* bullet -> {x,y,angle,owner,ownerX,ownerY} */
const OrigBullet = TT.Bullet;
TT.Bullet = class extends OrigBullet {
  constructor(x, y, angle, owner, color) {
    super(x, y, angle, owner, color);
    const t = TT.__curTank;
    births.set(this, {
      x, y, angle, owner,
      ownerX: t ? t.x : 0, ownerY: t ? t.y : 0,
    });
  }
};

const opponent = OPPONENTS[which] || sniper;

/* 让 killTank 记录自杀时的详细现场 */
const deaths = [];
const origKill = TT.Game.prototype.killTank;
TT.Game.prototype.killTank = function (tank, bullet) {
  if (tank.alive && bullet && bullet.owner === tank.index) {
    const b = births.get(bullet);
    /* 子弹从出生到杀死我，一路走过的所有点 */
    const path = bullet.__path || [];
    const minToBirth = b ? minDist(path, b.ownerX, b.ownerY) : -1;
    const minToDeath = minDist(path, tank.x, tank.y);
    deaths.push({
      minToBirth, minToDeath,
      bounces: bullet.bounces,
      traveled: path.length,
      ownerMoved: b ? Math.hypot(b.ownerX - tank.x, b.ownerY - tank.y) : -1,
    });
  }
  return origKill.call(this, tank, bullet);
};

function minDist(path, px, py) {
  let m = Infinity;
  for (let i = 0; i < path.length; i++) {
    const d = Math.hypot(path[i].x - px, path[i].y - py);
    if (d < m) m = d;
  }
  return m === Infinity ? -1 : m;
}

/* 给子弹挂上轨迹记录（每帧记一次位置） */
const origUpdate = TT.Bullet.prototype.update;
TT.Bullet.prototype.update = function (dt, game) {
  origUpdate.call(this, dt, game);
  if (!this.__path) this.__path = [];
  this.__path.push({ x: this.x, y: this.y });
};

/* 让 Bullet 构造时能拿到"当前开火的那辆坦克" */
const origTankUpdate = TT.Tank.prototype.update;
TT.Tank.prototype.update = function (dt, input, game) {
  TT.__curTank = this;
  origTankUpdate.call(this, dt, input, game);
  TT.__curTank = null;
};

for (let i = 0; i < matches; i++) {
  installRandom(4242 + i * 7919);
  runMatch(TT, policy, {
    policyIndex: 0, tankCount: 8, roundsToWin: 5, opponent, roundTimeout: 90,
  });
}

const HIT_R = TT.CFG.TANK.radius + TT.CFG.BULLET.radius;
console.log(`自杀机理：对手=${which}，${matches} 场，自杀 ${deaths.length} 次`);
console.log(`  判定半径 ${HIT_R.toFixed(1)}px\n`);

let passed = 0, movedIn = 0, other = 0;
for (const d of deaths) {
  if (d.minToBirth >= 0 && d.minToBirth <= HIT_R + 2) passed++;
  else if (d.minToDeath <= HIT_R + 2) movedIn++;
  else other++;
}
const tot = Math.max(1, deaths.length);
console.log(`  A 子弹经过"我开火时的位置"（当前位置判就能拦住）: ${passed}  ${(passed / tot * 100).toFixed(0)}%`);
console.log(`  B 没经过开火位置，是我后来走进弹道的（必须算运动轨迹）: ${movedIn}  ${(movedIn / tot * 100).toFixed(0)}%`);
console.log(`  C 两者都不是（路径没记录到）: ${other}  ${(other / tot * 100).toFixed(0)}%`);

const moved = deaths.filter((d) => d.ownerMoved >= 0).map((d) => d.ownerMoved);
if (moved.length) {
  const avg = moved.reduce((a, b) => a + b, 0) / moved.length;
  console.log(`\n  从开火到死亡，我平均移动了 ${avg.toFixed(0)}px`);
}
