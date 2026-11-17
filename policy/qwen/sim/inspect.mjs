/* inspect.mjs — 单场诊断回放
   node inspect.mjs [seed] [seat] [--table] */
import path from 'node:path';
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadTT } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');

const SEED = parseInt(process.argv[2] || '7', 10);
const SEAT = parseInt(process.argv[3] || '0', 10);
const SHOW_TABLE = process.argv.includes('--table');

const { ctx, TT, builtinBot, canvas } = loadTT({ seed: SEED });
vm.runInContext(fs.readFileSync(POLICY, 'utf8'), ctx, { filename: POLICY });

const QWEN = TT.policies.qwen;
const brainsMap = QWEN._internals.brains;

const game = new TT.Game(canvas);
game.render = () => {};
game.resize = () => {};

TT.updateBot = (tank, g, dt) => (tank.index === SEAT ? QWEN.update(tank, g, dt) : builtinBot(tank, g, dt));

/* 统计 */
const deaths = [];
const shots = new Array(8).fill(0);
const origFire = TT.Tank.prototype.fire;
TT.Tank.prototype.fire = function (g) { shots[this.index]++; return origFire.call(this, g); };

const origKill = game.killTank.bind(game);
game.killTank = function (tank, bullet) {
  if (tank.alive) {
    deaths.push({
      round: game.round, victim: tank.index,
      killer: bullet && bullet.owner != null ? bullet.owner : null,
      suicide: !!(bullet && bullet.owner === tank.index),
      bounces: bullet ? bullet.bounces : null,
    });
  }
  return origKill(tank, bullet);
};

game.startMatch(new Array(8).fill('bot'), 3);

const dt = 1 / 60;
let tick = 0;
const states = {};
let still = 0, moving = 0, samples = 0;
let aliveTicks = 0, dodge = 0, fireCmds = 0;
const logs = [];

while (game.state !== 'matchover' && tick < 60 * 240) {
  game.update(dt);
  tick++;

  const me = game.tanks.find((t) => t.index === SEAT);
  if (me && me.alive) {
    samples++;
    if (Math.abs(me.speed) < 20) still++; else moving++;
    const b = brainsMap.get(me);
    if (b) {
      const st = b.dbg.st || '?';
      states[st] = (states[st] || 0) + 1;
      if (st === 'dodge') dodge++;
      const inp = b.lastInp;
      if (inp && inp.shoot) fireCmds++;
      if (SHOW_TABLE && tick % 30 === 0) {
        logs.push({
          r: game.round, t: (tick * dt).toFixed(1),
          pos: me.x.toFixed(0) + ',' + me.y.toFixed(0), v: me.speed.toFixed(0),
          alive: game.tanks.filter((x) => x.alive).length, bl: game.bullets.length,
          st, shot: b.dbg.shot, aim: b.dbg.aim, goal: b.dbg.goal, pr: b.dbg.pressure,
        });
      }
    }
  }
}

console.log(`\n== seed ${SEED} seat P${SEAT} ==`);
console.log('scores :', game.scores.join(' '), '(P' + SEAT + ' = ' + game.scores[SEAT] + ')');
console.log('kills  :', game.kills.join(' '), '| 开枪次数', shots.join(' '));
console.log('回合   :', game.round, '| 平局', game.roundsToWin, '| 超时', game.state !== 'matchover');
console.log('状态占比:', JSON.stringify(Object.fromEntries(Object.entries(states).map(([k, v]) => [k, (v / samples * 100).toFixed(0) + '%']))));
console.log(`静止 ${(still / samples * 100).toFixed(0)}% · 移动 ${(moving / samples * 100).toFixed(0)}% · 规避 ${(dodge / samples * 100).toFixed(0)}%`);
console.log('qwen 死亡:', deaths.filter((d) => d.victim === SEAT).map((d) => 'R' + d.round + (d.suicide ? '(自杀)' : '←P' + d.killer + '/弹' + d.bounces)).join(' '));
console.log('全员死亡:', deaths.length, '自杀', deaths.filter((d) => d.suicide).length, '反弹', deaths.filter((d) => !d.suicide && d.bounces > 0).length);
console.log('policy stats:', JSON.stringify(QWEN.stats));
if (SHOW_TABLE) console.table(logs);
