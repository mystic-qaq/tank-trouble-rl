/* Local headless regression/arena test. Does not read any other policy. */
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import kimi from './policy.mjs';

globalThis.window = globalThis;
globalThis.TT = globalThis.TT || {};
globalThis.TT.Audio = {
  shoot() {}, bounce() {}, explode() {}, roundStart() {}, win() {},
};

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
for (const file of ['util.js', 'map.js', 'entities.js', 'ai.js', 'game.js']) {
  vm.runInThisContext(fs.readFileSync(path.join(root, 'js', file), 'utf8'), { filename: file });
}

function runRound(seed, count, spawnSeed, maxSeconds = 60) {
  const map = new TT.GameMap(seed);
  const game = {
    map,
    tanks: [],
    bullets: [],
    kills: Array(8).fill(0),
    selfKills: Array(8).fill(0),
    time: 0,
    round: 1,
    roundSerial: spawnSeed,
    fx: { muzzle() {}, fizzle() {}, bounceSpark() {}, explosion() {} },
    hooks: { hud() {} },
    shake() {},
  };
  game.killTank = function (tank, bullet) {
    if (bullet && bullet.owner === tank.index) game.selfKills[tank.index]++;
    if (process.env.DEBUG) {
      console.log('kill', JSON.stringify({ time: game.time.toFixed(2), tank: tank.index,
        owner: bullet && bullet.owner, x: tank.x.toFixed(1), y: tank.y.toFixed(1),
        bounces: bullet && bullet.bounces }));
    }
    TT.Game.prototype.killTank.call(game, tank, bullet);
  };

  const spawns = map.pickSpawns(count, TT.util.rng(spawnSeed));
  for (let i = 0; i < count; i++) {
    game.tanks.push(new TT.Tank(i, spawns[i], TT.COLORS[i], true));
  }

  const dt = 1 / 30;
  const steps = Math.floor(maxSeconds / dt);
  for (let step = 0; step < steps; step++) {
    game.time += dt;
    for (const tank of game.tanks) {
      if (!tank.alive) continue;
      const input = tank.index === 0 ? kimi(tank, game, dt) : TT.updateBot(tank, game, dt);
      if (process.env.DEBUG && input.shoot) {
        console.log('shoot', JSON.stringify({ time: game.time.toFixed(2), tank: tank.index,
          x: tank.x.toFixed(1), y: tank.y.toFixed(1), angle: tank.angle.toFixed(3),
          drive: input.drive, steer: input.steer.toFixed(3) }));
      }
      tank.update(dt, input, game);
    }
    TT.Game.prototype.resolveTankCollisions.call(game);
    for (const b of game.bullets) if (!b.dead) b.update(dt, game);

    for (const b of game.bullets) {
      if (b.dead) continue;
      for (const tank of game.tanks) {
        if (!tank.alive || (b.owner === tank.index && b.grace > 0)) continue;
        const t = TT.util.segCircle(b.px, b.py, b.x, b.y, tank.x, tank.y, tank.radius + b.radius);
        if (t >= 0) {
          game.killTank(tank, b);
          b.dead = true;
          break;
        }
      }
    }
    game.bullets = game.bullets.filter((b) => !b.dead);
    const alive = game.tanks.filter((t) => t.alive);
    if (alive.length <= 1) {
      return { winner: alive.length ? alive[0].index : -1, kills: game.kills.slice(),
        selfKills: game.selfKills.slice(), time: game.time };
    }
  }
  return { winner: -2, kills: game.kills.slice(), selfKills: game.selfKills.slice(), time: game.time };
}

const rounds = Number(process.argv[2] || 12);
const counts = [2, 4, 8];
const started = performance.now();
for (const count of counts) {
  let wins = 0, draws = 0, kills = 0, self = 0;
  for (let i = 0; i < rounds; i++) {
    const result = runRound(TT.CFG.MAP_SEEDS[i % 2], count, 1009 + i * 7919);
    if (result.winner === 0) wins++;
    if (result.winner < 0) draws++;
    kills += result.kills[0];
    self += result.selfKills[0];
  }
  console.log(JSON.stringify({ opponents: count - 1, rounds, wins, winRate: wins / rounds, draws, kills, self }));
}
console.log('elapsed_ms=' + Math.round(performance.now() - started));
