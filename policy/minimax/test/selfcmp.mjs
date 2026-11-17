/* ===========================================================
   minimax policy · 配置 × 自伤率 对比（同种子，配对比较）
   ---------------------------------------------------------------
   对内置 AI 和"狙击型"对手分别测几个配置的自杀次数/死亡次数。
   用法：node test/selfcmp.mjs [对手] [场数]
   =========================================================== */

import { loadGame, runMatch } from './harness.mjs';
import policy, { configure, resetConfig } from '../policy.js';

const which = (process.argv[2] || 'sniper');
const matches = parseInt(process.argv[3] || '8', 10);

const { TT } = loadGame();
const U = TT.util;

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

function installRandom(s) {
  let a = s >>> 0 || 1;
  Math.random = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const CONFIGS = [
  { name: '最终(余量0,增益24)', cfg: {} },
  { name: '余量6,增益12', cfg: { selfMargin: 6, dodgeGain: 12 } },
  { name: '余量0,增益12', cfg: { selfMargin: 0, dodgeGain: 12 } },
  { name: '余量0,增益24,增益阈值32', cfg: { dodgeWant: 32 } },
  { name: '全旧行为', cfg: { guardMode: 0, dodgeGain: 0, dodgeWins: 0, blockRule: 1, selfMargin: 6 } },
];

console.log(`配置 × 自伤对比：对手=${which}，每组 ${matches} 场\n`);
console.log('─'.repeat(92));

for (const c of CONFIGS) {
  resetConfig(); configure(c.cfg);
  let self = 0, death = 0, win = 0, rounds = 0, kill = 0;
  for (let i = 0; i < matches; i++) {
    installRandom(4242 + i * 7919);
    const r = runMatch(TT, policy, {
      policyIndex: 0, tankCount: 8, roundsToWin: 5,
      opponent: which === 'builtin' ? null : sniper,
      roundTimeout: 60,
    });
    self += r.selfKills[0];
    death += r.deaths[0];
    kill += r.kills[0];
    win += r.roundWinners.filter((w) => w === 0).length;
    rounds += r.rounds;
  }
  console.log(
    `  ${c.name.padEnd(24, ' ')} 胜 ${String(win).padStart(3)}/${String(rounds).padStart(4)}` +
    `  自杀 ${String(self).padStart(3)}` +
    `  死亡 ${String(death).padStart(4)}` +
    `  自杀占死亡 ${(death ? (self / death * 100) : 0).toFixed(1).padStart(5)}%` +
    `  净击杀 ${String(kill).padStart(4)}`
  );
}
resetConfig();
