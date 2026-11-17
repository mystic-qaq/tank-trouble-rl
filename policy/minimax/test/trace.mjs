/* ===========================================================
   minimax policy · 回合级追踪
   ---------------------------------------------------------------
   把每一回合的经过打印出来：谁活到最后、打了多久、我怎么死的。
   用来诊断"打不赢某种对手"的真实原因。
   用法：node test/trace.mjs [对手名] [场数] [回合超时秒]
   =========================================================== */

import { loadGame, runMatch } from './harness.mjs';
import policy from '../policy.js';

const which = (process.argv[2] || 'camper');
const matches = parseInt(process.argv[3] || '3', 10);
const roundTimeout = parseInt(process.argv[4] || '60', 10);

const { TT } = loadGame();
const U = TT.util;

/* ---- 对手风格（与 opponents.mjs 一致） ---- */
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

const OPPONENTS = { camper, berserker };

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* 逐回合记录存活人数曲线 */
function runTraced(TT2, policyFn, opponent, matches) {
  const rows = [];
  for (let i = 0; i < matches; i++) {
    installRandom(4242 + i * 7919);
    const r = runMatch(TT2, policyFn, {
      policyIndex: 0, tankCount: 8, roundsToWin: 5, opponent, roundTimeout,
    });
    rows.push(r);
  }
  return rows;
}

const opponent = OPPONENTS[which] || camper;
console.log(`回合追踪：对手=${which}，${matches} 场，回合超时 ${roundTimeout}s\n`);

const res = runTraced(TT, policy, opponent, matches);
let idx = 0;
for (const r of res) {
  console.log('─'.repeat(70));
  console.log(`场次 #${++idx}`);
  let sum = 0;
  r.roundDur.forEach((d, i) => {
    const w = r.roundWinners[i];
    const aliveLeft = w >= 0 ? 1 : 0;
    sum += d;
    console.log(
      `  回合 ${String(i + 1).padStart(2)}  时长 ${d.toFixed(1).padStart(6)}s` +
      `  胜者 ${w >= 0 ? 'P' + (w + 1) : '无（超时）'}` +
      `  我方阵亡 ${r.deaths[0]}  自杀 ${r.selfKills[0]}`
    );
  });
  console.log(`  小计：${r.rounds} 回合，平均 ${(sum / Math.max(1, r.rounds)).toFixed(1)}s，` +
    `我方胜 ${r.roundWinners.filter((w) => w === 0).length}，超时 ${r.timeouts}`);
  /* 阵亡记录 */
  const dl = r.deathLog;
  if (dl.length) {
    const byKiller = new Map();
    for (const d of dl) {
      const k = d.self ? '自己' : (d.killer >= 0 ? 'P' + (d.killer + 1) : '环境');
      byKiller.set(k, (byKiller.get(k) || 0) + 1);
    }
    console.log('  死于：' + [...byKiller.entries()].sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}×${v}`).join('  '));
  }
}
