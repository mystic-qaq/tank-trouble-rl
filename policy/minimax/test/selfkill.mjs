/* ===========================================================
   minimax policy · 自杀诊断
   ---------------------------------------------------------------
   统计「自己把自己打死」的那些子弹：飞了多久？弹了几次？
   如果大部分自杀是"子弹绕了一圈弹回来"，说明开火前的弹道推演
   时长（traceHorizon）不够长；如果都是刚射出的直球，说明是
   炮口判定或擦边判定的问题。
   用法：node test/selfkill.mjs [对手名] [场数]
   =========================================================== */

import { loadGame, runMatch } from './harness.mjs';
import policy from '../policy.js';

const which = (process.argv[2] || 'sniper');
const matches = parseInt(process.argv[3] || '8', 10);

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

function camper(tank, game, dt) {
  const inp = { drive: 0, steer: 0, shoot: false };
  if (!tank.ai) tank.ai = { x: 60 + tank.index * 30, y: 60 + tank.index * 20 };
  const home = tank.ai;
  let target = null, best = Infinity;
  for (const o of game.tanks) {
    if (o === tank || !o.alive) continue;
    if (!game.map.hasLOS(tank.x, tank.y, o.x, o.y)) continue;
    const d = Math.hypot(o.x - tank.x, o.y - tank.y);
    if (d < best) { best = d; target = o; }
  }
  if (target) {
    const want = Math.atan2(target.y - tank.y, target.x - tank.x);
    const diff = U.angleDiff(tank.angle, want);
    if (Math.abs(diff) > 0.03) inp.steer = U.sign(diff);
    if (Math.abs(diff) < 0.05) inp.shoot = true;
  } else {
    const want = Math.atan2(home.y - tank.y, home.x - tank.x);
    const diff = U.angleDiff(tank.angle, want);
    if (Math.abs(diff) > 0.05) inp.steer = U.sign(diff);
    else if (Math.hypot(home.x - tank.x, home.y - tank.y) > 30) inp.drive = 1;
  }
  return inp;
}

const OPPONENTS = { sniper, camper };

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const opponent = OPPONENTS[which] || sniper;

let rounds = 0, wins = 0;
const self = [];        // 自杀事件
const byEnemy = [];     // 被别人打死

for (let i = 0; i < matches; i++) {
  installRandom(4242 + i * 7919);
  const r = runMatch(TT, policy, {
    policyIndex: 0, tankCount: 8, roundsToWin: 5, opponent, roundTimeout: 90,
  });
  rounds += r.rounds;
  wins += r.roundWinners.filter((w) => w === 0).length;
  for (const d of r.deathLog) (d.self ? self : byEnemy).push(d);
}

console.log(`自杀诊断：对手=${which}，${matches} 场，${rounds} 回合，我方胜 ${wins}`);
console.log(`  阵亡 ${self.length + byEnemy.length} 次（其中自杀 ${self.length} 次，占 ` +
  `${(self.length / Math.max(1, self.length + byEnemy.length) * 100).toFixed(1)}%）`);

if (self.length) {
  /* 按子弹年龄分桶 */
  const buckets = [
    ['<0.3s  刚出膛', 0.3], ['0.3-1s', 1], ['1-2s', 2], ['2-3s', 3], ['3-4s', 4], ['>4s', 99],
  ];
  console.log('\n  自杀子弹的飞行时长分布：');
  for (const [label, hi] of buckets) {
    const lo = label.startsWith('<') ? 0 : buckets[buckets.findIndex((b) => b[0] === label) - 1][1];
    const n = self.filter((d) => d.bulletAge >= lo && d.bulletAge < hi).length;
    const bar = '█'.repeat(Math.round(n / self.length * 30));
    console.log(`    ${label.padEnd(12)} : ${String(n).padStart(3)}  ${(n / self.length * 100).toFixed(0)}%  ${bar}`);
  }
  const bc = {};
  for (const d of self) bc[d.bulletBounces] = (bc[d.bulletBounces] || 0) + 1;
  console.log('\n  自杀子弹的弹跳次数：' +
    Object.entries(bc).sort((a, b) => a[0] - b[0]).map(([k, v]) => `${k}次×${v}`).join('  '));
  console.log('\n  明细（前 12 条）：');
  for (const d of self.slice(0, 12)) {
    console.log(`    飞行 ${d.bulletAge.toFixed(2)}s  弹跳 ${d.bulletBounces} 次  ` +
      `回合第 ${d.roundTime.toFixed(1)}s  当时场上我方子弹 ${d.myBullets} 发`);
  }
}
