/* ===========================================================
   minimax policy · 对手风格鲁棒性测试
   ---------------------------------------------------------------
   Arena 里其他 7 辆坦克是别人写的策略，风格未知。
   这里写几种典型风格当对手，看 minimax 是否照样能赢：
     1. 冲锋型  berserker ：无脑贴脸，见人就打（不查自伤）
     2. 蹲坑型  camper    ：守在角落，谁进来打谁
     3. 狙击型  sniper    ：拉开距离，精确瞄准
     4. 反弹型  bouncer   ：喜欢打墙折射冷枪
   这些对手只用于测试，不属于交付的策略本体。
   =========================================================== */

import { loadGame, runMatch, summarize } from './harness.mjs';
import policy from '../policy.js';

const matches = parseInt(process.argv[2] || '8', 10);
/* 回合超时上限（秒）。游戏本体没有回合时限，回合只在剩 ≤1 辆坦克时结束；
   这里的超时只是测试台的安全帽，把它调大可以看到"打得久但终究分出胜负"的真实走势。 */
const roundTimeout = parseInt(process.argv[3] || '40', 10);
const { TT } = loadGame();

const CFG = TT.CFG;
const U = TT.util;

/* ---------- 对手 1：冲锋型 ---------- */
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

/* ---------- 对手 2：蹲坑型 ---------- */
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

/* ---------- 对手 3：狙击型（保持距离 + 精确瞄准 + 预判，看不见人就绕圈巡逻） ---------- */
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
    /* 看不见人：绕着地图中心巡逻 */
    const cx = 640 + Math.cos(st.j) * 380;
    const cy = 400 + Math.sin(st.j * 1.3) * 260;
    const want = Math.atan2(cy - tank.y, cx - tank.x);
    const diff = U.angleDiff(tank.angle, want);
    if (Math.abs(diff) > 0.05) inp.steer = U.sign(diff);
    else inp.drive = 1;
    return inp;
  }
  const lead = best / CFG.BULLET.speed;
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

/* ---------- 对手 4：反弹冷枪型 ---------- */
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
  /* 朝着"目标背后那面墙"打，制造折射 */
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

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* 让 1..7 号席位都用指定对手，0 号用 minimax */
function runVs(opponent, tankCount = 8) {
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(4242 + i * 7919);
    res.push(runMatch(TT, policy, {
      policyIndex: 0, tankCount, roundsToWin: 5, opponent, roundTimeout,
    }));
  }
  return summarize(res, 0);
}

/* 全 minimax 的基准，用来对比"同水平对手" */
function runMirror(tankCount = 8) {
  const res = [];
  for (let i = 0; i < matches; i++) {
    installRandom(4242 + i * 7919);
    res.push(runMatch(TT, policy, { policyIndex: 0, tankCount, roundsToWin: 5, roundTimeout }));
  }
  return summarize(res, 0);
}

function report(name, s) {
  const rounds = s.rounds || 1;
  const draws = s.roundDraws || 0;
  const p = s.policy;
  console.log(
    `  ${name.padEnd(20, ' ')} 胜 ${String(p.wins).padStart(3)}/${String(rounds).padStart(4)} 回合` +
    ` (${(p.wins / rounds * 100).toFixed(1).padStart(5)}%)` +
    `  平局 ${String(draws).padStart(4)}` +
    `  净击杀 ${String(p.kills).padStart(4)}` +
    `  自杀 ${String(p.selfKills).padStart(3)}` +
    `  死亡 ${String(p.deaths).padStart(4)}` +
    `  超时 ${String(s.timeouts).padStart(3)}`
  );
}

console.log(`对手风格测试：每组 ${matches} 场（minimax 坐 P1，其余席位是对手），回合超时上限 ${roundTimeout}s`);
console.log('─'.repeat(96));
console.log('\n【8 辆坦克混战】');
{
  const base = runMirror(8);
  report('同为 minimax（镜像）', base);
  for (const [name, fn] of Object.entries(OPPONENTS)) report(name, runVs(fn, 8));
}

console.log('\n【2 辆坦克单挑】');
{
  const base = runMirror(2);
  report('同为 minimax（镜像）', base);
  for (const [name, fn] of Object.entries(OPPONENTS)) report(name, runVs(fn, 2));
}
