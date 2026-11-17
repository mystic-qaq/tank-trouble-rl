/* ===========================================================
   minimax policy · 对照实验
   ---------------------------------------------------------------
   同一个对手风格，分别让 minimax 和「游戏内置 AI」坐在 P1，
   看谁赢得多。用来区分两种情况：
     A) minimax 明显更强      → 是 minimax 的优势
     B) 两者都很差            → 是这个对手风格把游戏拖成僵局，跟策略无关
   用法：node test/versus.mjs [对手名] [场数] [回合超时秒]
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy from '../policy.js';

const which = (process.argv[2] || 'camper');
const matches = parseInt(process.argv[3] || '8', 10);
const roundTimeout = parseInt(process.argv[4] || '40', 10);

const { TT } = loadGame();
const builtin = TT.updateBot;
const U = TT.util;

/* ---------- 对手风格（与 opponents.mjs 保持一致） ---------- */
function berserker(tank, game, dt) {
  const inp = { drive: 0, steer: 0, shoot: false };
  let target = null, best = Infinity;
  for (const o of game.tanks) {
    if (o === tank || !o.alive) continue;
    const d = Math.hypot(o.x - tank.x, o.y - tank.y);
    if (d < best) { best = d; target = o; }
  }
  if (!target) return inp;
  const want = Math.atan2(target.y - tank.y, target.x - tank.x);
  const diff = U.angleDiff(tank.angle, want);
  if (Math.abs(diff) > 0.04) inp.steer = U.sign(diff);
  inp.drive = Math.abs(diff) > 2 ? -1 : 1;
  if (Math.abs(diff) < 0.09) inp.shoot = true;
  return inp;
}

function camper(tank, game, dt) {
  const inp = { drive: 0, steer: 0, shoot: false };
  if (!tank.ai) tank.ai = { x: 60 + tank.index * 30, y: 60 + tank.index * 20, timer: 0 };
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

function sniper(tank, game, dt) {
  const inp = { drive: 0, steer: 0, shoot: false };
  if (!tank.ai) tank.ai = { j: Math.random() * 6.28, ang: Math.random() * 6.28 };
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

function bouncer(tank, game, dt) {
  const inp = { drive: 0, steer: 0, shoot: false };
  if (!tank.ai) tank.ai = { t: 0 };
  const b = tank.ai;
  b.t -= dt;
  let target = null, best = Infinity;
  for (const o of game.tanks) {
    if (o === tank || !o.alive) continue;
    const d = Math.hypot(o.x - tank.x, o.y - tank.y);
    if (d < best) { best = d; target = o; }
  }
  if (!target) return inp;
  const want = Math.atan2(target.y - tank.y, target.x - tank.x);
  const diff = U.angleDiff(tank.angle, want);
  if (Math.abs(diff) > 0.04) inp.steer = U.sign(diff);
  if (Math.abs(diff) > 1.2) inp.drive = 1;
  if (b.t <= 0 && Math.abs(diff) < 0.12) {
    b.t = 0.8 + Math.random() * 0.8;
    inp.shoot = true;
  }
  return inp;
}

const OPPONENTS = { berserker, camper, sniper, bouncer };
const opponent = OPPONENTS[which];
if (!opponent) {
  console.error('未知对手：' + which + '，可选：' + Object.keys(OPPONENTS).join(' / '));
  process.exit(1);
}

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* 跑一批：controlled 是坐在 P1 的那个策略 */
function run(controlled, tankCount = 8) {
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(4242 + i * 7919);
    res.push(runMatch(TT, controlled, {
      policyIndex: 0, tankCount, roundsToWin: 5, opponent, roundTimeout,
    }));
  }
  return summarize(res, 0);
}

function report(name, s) {
  const rounds = s.rounds || 1;
  const p = s.policy;
  console.log(
    `  ${name.padEnd(22)} 胜 ${String(p.wins).padStart(3)}/${String(rounds).padStart(4)} 回合` +
    ` (${(p.wins / rounds * 100).toFixed(1).padStart(5)}%)` +
    `  平局 ${String(s.roundDraws).padStart(4)}` +
    `  净击杀 ${String(p.kills).padStart(4)}` +
    `  自杀 ${String(p.selfKills).padStart(3)}` +
    `  死亡 ${String(p.deaths).padStart(4)}` +
    `  超时 ${String(s.timeouts).padStart(4)}`
  );
}

console.log(`对照实验：对手 = ${which}（其余 7 席），每组 ${matches} 场，回合超时上限 ${roundTimeout}s`);
console.log('─'.repeat(100));

console.log('\n【8 辆坦克混战】');
report('minimax（坐 P1）', run(policy, 8));
report('内置 AI（坐 P1）', run((t, g, d) => builtin(t, g, d), 8));

console.log('\n【2 辆坦克单挑】');
report('minimax（坐 P1）', run(policy, 2));
report('内置 AI（坐 P1）', run((t, g, d) => builtin(t, g, d), 2));
