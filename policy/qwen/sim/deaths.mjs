/* deaths.mjs — 死因剖析：qwen 在什么处境下被打死
   node deaths.mjs [matches] [seat] */
import path from 'node:path';
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadTT, mulberry32 } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');

const N = parseInt(process.argv[2] || '12', 10);
const SEAT = parseInt(process.argv[3] || '0', 10);

const rows = [];
let rounds = 0, wins = 0;

for (let m = 0; m < N; m++) {
  const { ctx, TT, builtinBot, canvas } = loadTT({ seed: 500 + m * 91 });
  vm.runInContext(fs.readFileSync(POLICY, 'utf8'), ctx, { filename: POLICY });
  const QWEN = TT.policies.qwen;
  const brainsMap = QWEN._internals.brains;

  const game = new TT.Game(canvas);
  game.render = () => {};
  game.resize = () => {};
  TT.updateBot = (t, g, d) => (t.index === SEAT ? QWEN.update(t, g, d) : builtinBot(t, g, d));

  const origKill = game.killTank.bind(game);
  game.killTank = function (tank, bullet) {
    if (tank.alive && tank.index === SEAT) {
      const alive = game.tanks.filter((t) => t.alive).length;
      let killer = null;
      if (bullet && bullet.owner != null && bullet.owner !== SEAT) {
        killer = game.tanks.find((t) => t.index === bullet.owner);
      }
      const b = brainsMap.get(tank);
      const d = (b && b.dbg) || {};
      let kd = null, kdLos = null;
      if (killer) {
        kd = Math.hypot(killer.x - tank.x, killer.y - tank.y);
        kdLos = game.map.hasLOS(tank.x, tank.y, killer.x, killer.y);
      }
      rows.push({
        round: game.round,
        aliveAtDeath: alive,
        t: Math.min(30, game.time % 60).toFixed(0),
        suicide: !!(bullet && bullet.owner === SEAT),
        bounces: bullet ? bullet.bounces : -1,
        kd: kd === null ? '—' : kd.toFixed(0),
        kdLos: kdLos === null ? '—' : (kdLos ? 'Y' : 'N'),
        speed: Math.abs(tank.speed).toFixed(0),
        st: d.st || '?',
        nearBullet: game.bullets.filter((x) => !x.dead && Math.hypot(x.x - tank.x, x.y - tank.y) < 200).length,
      });
    }
    return origKill(tank, bullet);
  };

  const origEnd = game.endRound.bind(game);
  game.endRound = function (winner) {
    if (game.state === 'playing') { rounds++; if (winner === SEAT) wins++; }
    return origEnd(winner);
  };

  game.startMatch(new Array(8).fill('bot'), 4);
  const dt = 1 / 60;
  let tick = 0;
  while (game.state !== 'matchover' && tick < 60 * 300) { game.update(dt); tick++; }
}

const q = (f) => rows.filter(f);
const pc = (n, d) => ((100 * n) / (d || 1)).toFixed(1) + '%';
console.log(`\n== qwen 死亡剖析：${rows.length} 次死亡 / ${rounds} 回合，回合胜率 ${pc(wins, rounds)} ==`);

console.log('\n按存活人数（死亡时的场上人数）:');
for (const a of [8, 7, 6, 5, 4, 3, 2, 1]) {
  const s = q((r) => r.aliveAtDeath === a);
  if (s.length) console.log(`  ${a} 人时死: ${s.length} (${pc(s.length, rows.length)})`);
}

console.log('\n按死因:');
console.log('  自杀        :', q((r) => r.suicide).length);
console.log('  反弹弹(≥1跳):', q((r) => !r.suicide && +r.bounces > 0).length);
console.log('  直射        :', q((r) => !r.suicide && +r.bounces === 0).length);

console.log('\n按死亡时状态:');
const st = {};
for (const r of rows) st[r.st] = (st[r.st] || 0) + 1;
Object.entries(st).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => console.log(`  ${k}: ${v} (${pc(v, rows.length)})`));

const still = q((r) => +r.speed < 25);
console.log(`\n  几乎静止时死亡: ${still.length} (${pc(still.length, rows.length)})`);
const seen = q((r) => r.kdLos === 'Y');
console.log(`  死时看得见凶手: ${seen.length} / 看不见: ${q((r) => r.kdLos === 'N').length} / 无凶手: ${q((r) => r.kdLos === '—').length}`);
const close = q((r) => r.kd !== '—' && +r.kd < 200);
console.log(`  凶手在 200px 内: ${close.length} (${pc(close.length, rows.length)})`);
const many = q((r) => r.nearBullet >= 2);
console.log(`  死时 200px 内≥2发子弹: ${many.length} (${pc(many.length, rows.length)})`);

console.log('\n样本（前 25 条）:');
console.table(rows.slice(0, 25));
