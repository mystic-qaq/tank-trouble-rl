/* ===========================================================
   Headless simulator for the DeepSeek Tank Trouble policy.
   Loads the original game scripts (util/map/entities/ai) and the
   policy, then runs full free-for-all rounds with real physics.

   Usage (from anywhere):
     node policy/deepseek/test/sim.mjs
     node policy/deepseek/test/sim.mjs --rounds 200 --ours 1
     node policy/deepseek/test/sim.mjs --mode duel --rounds 200
     node policy/deepseek/test/sim.mjs --selfplay 1 --rounds 100
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../..');

/* ---------------- command line ---------------- */

const argv = process.argv.slice(2);
function arg(name, def) {
  const i = argv.indexOf('--' + name);
  if (i >= 0 && i + 1 < argv.length) return argv[i + 1];
  return def;
}
const OPT = {
  rounds: Number(arg('rounds', 100)),
  ours: Number(arg('ours', 1)),
  mode: arg('mode', 'ffa'),          // ffa | duel | selfplay | baseline
  seat: Number(arg('seat', 0)),
  maxTime: Number(arg('maxTime', 75)),
  quiet: arg('quiet', '0') === '1',
  debug: arg('debug', '0') === '1',
  seed: Number(arg('seed', 20261105)),
  params: arg('params', ''),
  variants: arg('variants', ''),
  bots: arg('bots', 'bot'),
  matches: Number(arg('matches', 50)),
  roundsToWin: Number(arg('roundsToWin', 5)),
};

/* ---------------- load game into a vm sandbox ---------------- */

const ctx = {
  console, Math, Date, JSON, Number, Array, Object, String, Boolean,
  setTimeout, clearTimeout, setInterval, clearInterval,
  document: {
    createElement() {
      return {
        width: 0, height: 0,
        getContext() { return null; },
      };
    },
  },
};
ctx.window = ctx;
ctx.globalThis = ctx;
vm.createContext(ctx);

/* 让内置 bot 的 Math.random 也可复现，便于定位问题 */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
ctx.__rng = mulberry32(OPT.seed >>> 0);
vm.runInContext('Math.random = __rng;', ctx);

function load(rel) {
  const file = path.join(ROOT, rel);
  const code = fs.readFileSync(file, 'utf8');
  vm.runInContext(code, ctx, { filename: rel });
}

load('js/util.js');
load('js/map.js');
load('js/entities.js');
ctx.TT.Audio = {
  shoot() {}, bounce() {}, explode() {},
  roundStart() {}, win() {}, click() {},
};
load('js/ai.js');
load('policy/deepseek/policy.js');

const TT = ctx.TT;
const U = TT.util;
const CFG = TT.CFG;
if (OPT.debug) TT.DS_DEBUG = true;
if (OPT.params) {
  try { TT.deepseekPolicy.setParams(JSON.parse(OPT.params)); }
  catch (e) { console.error('bad params', e.message); }
}

/* ---------------- one round ---------------- */

function runRound({ seed, policies, maxTime, debugOurs, debugTag, numTanks }) {
  ctx.__roundRng = mulberry32((seed ^ 0x5bf03635) >>> 0);
  vm.runInContext('Math.random = __roundRng;', ctx);
  const roundSerial = seed;
  const map = new TT.GameMap(seed);
  const rng = U.rng((seed ^ 0x9e3779b9) >>> 0);
  const n = numTanks || 8;
  const spawns = map.pickSpawns(n, rng);

  const tanks = [];
  for (let i = 0; i < n; i++) {
    const tank = new TT.Tank(i, spawns[i], TT.COLORS[i], true);
    tank.alive = true;
    tanks.push(tank);
  }

  const kills = new Array(Math.max(8, n)).fill(0);
  const perTank = [];
  for (let i = 0; i < n; i++) {
    perTank.push({
      shots: 0, suicides: 0, kills: 0, deaths: 0, aliveTime: 0, won: false,
      deathTime: null, killer: null, hist: [], deathInfo: null,
    });
  }
  const stats = { suicides: 0, kills: 0, errors: 0, shots: 0 };

  const game = {
    map, tanks, bullets: [], kills,
    time: 0, roundSerial, state: 'playing',
    fx: {
      muzzle() {}, explosion() {}, fizzle() {}, bounceSpark() {}, shake() {},
    },
    shake() {},
    killTank(tank, bullet) {
      if (!tank.alive) return;
      tank.alive = false;
      perTank[tank.index].deaths++;
      perTank[tank.index].deathTime = game.time;
      if (bullet && bullet.owner != null) {
        perTank[tank.index].killer = bullet.owner;
        var killer = tanks[bullet.owner];
        perTank[tank.index].deathInfo = {
          bounces: bullet.bounces,
          age: +(6 - bullet.life).toFixed(2),
          bAngle: +bullet.angle.toFixed(2),
          dist: killer ? +Math.hypot(killer.x - tank.x, killer.y - tank.y).toFixed(0) : -1,
          lastShotAge: +tank.lastShotAge.toFixed(2),
          alive: tanks.filter(function (x) { return x.alive; }).length + 1,
          mode: tank.__dsMode || '',
        };
        if (bullet.owner === tank.index) {
          kills[tank.index] -= 1;
          stats.suicides++;
          perTank[tank.index].suicides++;
          if (OPT.debug && debugOurs && debugOurs.has(tank.index)) {
            console.log('SELFKILL', JSON.stringify({
              tag: debugTag, seed,
              t: +game.time.toFixed(2),
              idx: tank.index,
              x: +tank.x.toFixed(1), y: +tank.y.toFixed(1),
              angle: +tank.angle.toFixed(2),
              speed: +tank.speed.toFixed(1),
              lastShotAge: +tank.lastShotAge.toFixed(2),
              bounces: bullet.bounces,
              bAngle: +bullet.angle.toFixed(2),
              bLife: +bullet.life.toFixed(2),
              hist: perTank[tank.index].hist.slice(-14),
              log: (TT.DS_LOG || []).filter((e) =>
                e.r === roundSerial && e.idx === tank.index && e.t > game.time - 2.2
              ),
            }));
          }
        } else {
          kills[bullet.owner] += 1;
          stats.kills++;
          perTank[bullet.owner].kills++;
        }
      }
      stats.deaths = (stats.deaths || 0) + 1;
    },
  };

  const dt = 1 / 60;
  let t = 0;
  let step = 0;
  let finalTwoAt = null;

  while (t < maxTime) {
    t += dt;
    step++;
    game.time = t;

    for (let i = 0; i < tanks.length; i++) {
      const tank = tanks[i];
      if (!tank.alive) continue;
      const fn = policies[i];
      let inp;
      try {
        inp = fn(tank, game, dt);
      } catch (err) {
        stats.errors++;
        if (stats.errors <= 3) console.error('policy error:', err && err.stack || err);
        inp = { drive: 0, steer: 0, shoot: false };
      }
      if (!inp || typeof inp !== 'object') inp = { drive: 0, steer: 0, shoot: false };
      inp.drive = Number(inp.drive) || 0;
      inp.steer = Number(inp.steer) || 0;
      inp.shoot = !!inp.shoot;
      if (inp.drive > 1) inp.drive = 1;
      if (inp.drive < -1) inp.drive = -1;
      if (inp.steer > 1) inp.steer = 1;
      if (inp.steer < -1) inp.steer = -1;
      if (inp.shoot) stats.shots++;
      if (inp.shoot) perTank[tank.index].shots++;
      tank.update(dt, inp, game);
      perTank[tank.index].aliveTime += dt;
      if (OPT.debug && step % 3 === 0) {
        const h = perTank[tank.index].hist;
        h.push([
          +game.time.toFixed(2), +inp.drive.toFixed(2), +inp.steer.toFixed(2),
          inp.shoot ? 1 : 0, Math.round(tank.x), Math.round(tank.y),
          +tank.angle.toFixed(2), Math.round(tank.speed),
        ]);
        if (h.length > 40) h.shift();
      }
    }

    /* tank vs tank */
    for (let i = 0; i < tanks.length; i++) {
      for (let j = i + 1; j < tanks.length; j++) {
        const a = tanks[i], b = tanks[j];
        if (!a.alive || !b.alive) continue;
        const dx = b.x - a.x, dy = b.y - a.y;
        const minD = a.radius + b.radius;
        const d2 = dx * dx + dy * dy;
        if (d2 >= minD * minD || d2 < 1e-6) continue;
        const d = Math.sqrt(d2);
        const push = (minD - d) / 2;
        const nx = dx / d, ny = dy / d;
        a.x -= nx * push; a.y -= ny * push;
        b.x += nx * push; b.y += ny * push;
        map.resolveCircle(a, a.radius);
        map.resolveCircle(b, b.radius);
        a.speed *= 0.72;
        b.speed *= 0.72;
      }
    }

    /* bullets */
    for (let i = 0; i < game.bullets.length; i++) {
      const b = game.bullets[i];
      if (!b.dead) b.update(dt, game);
    }

    /* bullet vs tank */
    for (let i = 0; i < game.bullets.length; i++) {
      const b = game.bullets[i];
      if (b.dead) continue;
      for (let j = 0; j < tanks.length; j++) {
        const tank = tanks[j];
        if (!tank.alive) continue;
        if (b.owner === tank.index && b.grace > 0) continue;
        const hit = U.segCircle(b.px, b.py, b.x, b.y, tank.x, tank.y, tank.radius + b.radius);
        if (hit >= 0) {
          game.killTank(tank, b);
          b.dead = true;
          break;
        }
      }
    }

    game.bullets = game.bullets.filter((b) => !b.dead);

    let alive = 0;
    let last = null;
    for (let i = 0; i < tanks.length; i++) {
      if (tanks[i].alive) { alive++; last = tanks[i]; }
    }
    if (alive === 2 && finalTwoAt == null) finalTwoAt = t;
    if (alive <= 1) {
      if (alive === 1) perTank[last.index].won = true;
      return {
        winner: alive === 1 ? last.index : -1,
        time: t,
        kills, stats, perTank,
        finalTwoAt,
        aliveCount: alive,
        survivors: tanks.filter((x) => x.alive).map((x) => ({
          i: x.index, x: +x.x.toFixed(1), y: +x.y.toFixed(1),
        })),
        timedOut: false,
      };
    }
  }

  return {
    winner: -1, time: t, kills, stats, perTank, aliveCount: 0, timedOut: true,
    finalTwoAt,
    survivors: tanks.filter((x) => x.alive).map((x) => ({
      i: x.index, x: +x.x.toFixed(1), y: +x.y.toFixed(1),
    })),
  };
}

/* ---------------- match runners ---------------- */

function makePolicySet(mode, ours, seat) {
  const policies = [];
  const kinds = [];
  for (let i = 0; i < 8; i++) {
    const isOurs =
      mode === 'selfplay' ? true :
      mode === 'baseline' ? false :
      mode === 'duel' ? (i === seat) :
      ours === 1 ? (i === seat) : (i < ours);
    if (isOurs) {
      policies.push(TT.deepseekPolicy.update);
      kinds.push('deepseek');
    } else if (mode === 'sniper' || OPT.bots === 'sniper') {
      policies.push((tank, game, dt) => {
        if (!tank.ai) {
          TT.updateBot(tank, game, dt);
          tank.ai.skill = 1;
          tank.ai.jitter = 0;
          tank.ai.jitterTarget = 0;
          tank.ai.shootDelay = 0.05;
        }
        return TT.updateBot(tank, game, dt);
      });
      kinds.push('bot');
    } else {
      policies.push((tank, game, dt) => TT.updateBot(tank, game, dt));
      kinds.push('bot');
    }
  }
  return { policies, kinds };
}

function simulate(label, mode, opts) {
  const total = opts.rounds;
  let winCount = 0;
  let timeout = 0;
  let totalTime = 0;
  let suicides = 0;
  let kills = 0;
  let errors = 0;
  let shots = 0;
  let oursSuicides = 0;
  let oursKills = 0;
  let oursShots = 0;
  let oursAliveAtTimeout = 0;
  let oursTimeoutRounds = 0;
  let draws = 0;
  let ourFinalTwo = 0;
  let ourFinalTwoWins = 0;
  let ourDeaths = 0;
  const ourDeathTimes = [];
  const ourDeathModes = {};
  let ourDeathBounce = 0;
  let ourDeathRecentShot = 0;
  const ourDeathAlive = {};
  const winBySeat = new Array(8).fill(0);
  const roundTimes = [];
  const timeoutSamples = [];

  const oursSet = new Set();
  if (mode === 'selfplay') for (let i = 0; i < 8; i++) oursSet.add(i);
  else if (mode === 'duel') oursSet.add(opts.seat);
  else if (mode === 'sniper') oursSet.add(opts.seat);
  else if (mode === 'ffa') {
    if (opts.ours === 1) oursSet.add(opts.seat);
    else for (let i = 0; i < opts.ours; i++) oursSet.add(i);
  }

  const t0 = Date.now();

  for (let r = 0; r < total; r++) {
    const seed = (opts.seed + r * 7919) >>> 0;
    const { policies, kinds } = makePolicySet(mode, opts.ours, opts.seat);
    if (OPT.debug && r === 0) console.error('KINDS', kinds.join(','), 'match', policies.map((p,i)=>p===TT.deepseekPolicy.update?i:null).filter(x=>x!=null).join(','));
    const res = runRound({
      seed, policies, maxTime: opts.maxTime,
      debugOurs: oursSet, debugTag: label,
      numTanks: mode === 'duel' ? 2 : 8,
    });
    totalTime += res.time;
    roundTimes.push(res.time);
    suicides += res.stats.suicides;
    kills += res.stats.kills;
    errors += res.stats.errors;
    shots += res.stats.shots;

    for (const i of oursSet) {
      oursSuicides += res.perTank[i].suicides;
      oursKills += res.perTank[i].kills;
      oursShots += res.perTank[i].shots;
      if (res.perTank[i].deathTime != null) {
        ourDeaths++;
        ourDeathTimes.push(res.perTank[i].deathTime);
        const di = res.perTank[i].deathInfo;
        if (di) {
          ourDeathModes[di.mode] = (ourDeathModes[di.mode] || 0) + 1;
          if (di.bounces > 0) ourDeathBounce++;
          if (di.lastShotAge < 1) ourDeathRecentShot++;
          ourDeathAlive[di.alive] = (ourDeathAlive[di.alive] || 0) + 1;
        }
      }
    }

    if (res.finalTwoAt != null) {
      let oursInFinal = false;
      for (const i of oursSet) {
        const dtDeath = res.perTank[i].deathTime;
        if (dtDeath == null || dtDeath > res.finalTwoAt) oursInFinal = true;
      }
      if (oursInFinal) {
        ourFinalTwo++;
        if (res.winner >= 0 && kinds[res.winner] === 'deepseek') ourFinalTwoWins++;
      }
    }

    if (res.winner >= 0) {
      if (kinds[res.winner] === 'deepseek') winCount++;
      winBySeat[res.winner]++;
    } else if (res.timedOut) {
      timeout++;
      let oursAlive = false;
      for (const i of oursSet) {
        if (res.perTank[i].aliveTime > res.time - 0.1) oursAlive = true;
      }
      if (oursAlive) {
        oursAliveAtTimeout++;
        if (timeoutSamples.length < 5) {
          timeoutSamples.push({
            seed,
            time: +res.time.toFixed(1),
            survivors: res.survivors,
          });
        }
      }
      if (mode === 'ffa' && oursSet.size > 0) oursTimeoutRounds++;
    } else {
      draws++;
    }
  }

  roundTimes.sort((a, b) => a - b);
  const elapsed = (Date.now() - t0) / 1000;
  const median = roundTimes[Math.floor(roundTimes.length / 2)] || 0;
  const oursN = mode === 'selfplay' ? 8 : (mode === 'duel' ? 1 : opts.ours);
  const expPerRound = mode === 'duel' ? 0.5 : (mode === 'selfplay' ? 1 : oursN / 8);

  const out = {
    label,
    mode,
    rounds: total,
    winRate: +(winCount / total).toFixed(4),
    expectedRandom: +expPerRound.toFixed(4),
    timeoutRate: +(timeout / total).toFixed(4),
    timeoutsOursAlive: oursAliveAtTimeout,
    drawRate: +(draws / total).toFixed(4),
    ourFinalTwo,
    ourFinalTwoWinRate: ourFinalTwo ? +(ourFinalTwoWins / ourFinalTwo).toFixed(4) : 0,
    ourDeaths,
    ourAvgDeathTime: ourDeaths ? +(ourDeathTimes.reduce((a, b) => a + b, 0) / ourDeaths).toFixed(2) : 0,
    ourDeathModes,
    ourDeathBounce,
    ourDeathRecentShot,
    ourDeathAlive,
    avgTime: +(roundTimes.reduce((a, b) => a + b, 0) / total).toFixed(2),
    medianTime: +median.toFixed(2),
    suicides,
    kills,
    errors,
    shots,
    oursSuicides,
    oursKills,
    oursShots,
    timeoutSamples,
    simSeconds: +elapsed.toFixed(2),
    winBySeat,
  };

  if (!opts.quiet) {
    console.log(
      `${label.padEnd(30)} win ${(out.winRate * 100).toFixed(1)}%` +
      ` (random ${(out.expectedRandom * 100).toFixed(1)}%)` +
      ` | timeouts ${(out.timeoutRate * 100).toFixed(1)}% (ours alive ${oursAliveAtTimeout})` +
      ` | draws ${(out.drawRate * 100).toFixed(1)}%` +
      ` | avg ${out.avgTime}s med ${out.medianTime}s` +
      ` | final2 ${ourFinalTwo}/${total} won ${out.ourFinalTwoWinRate * 100 ? (out.ourFinalTwoWinRate * 100).toFixed(1) + '%' : '0%'}` +
      ` | OUR sui ${oursSuicides} deaths ${ourDeaths} kills ${oursKills} shots ${oursShots}` +
      ` | all sui ${suicides} kills ${kills} errors ${errors}` +
      ` | ${elapsed.toFixed(1)}s cpu`
    );
    if (timeoutSamples.length) {
      console.log('  timeout samples:', JSON.stringify(timeoutSamples));
    }
  }

  return out;
}

/* ---------------- match runner (first to N wins) ---------------- */

function simulateMatch(label, mode, opts) {
  const matches = opts.matches;
  let ourWins = 0;
  let timeouts = 0;
  let totalRounds = 0;
  let errors = 0;
  const t0 = Date.now();
  const numTanks = mode === 'duel-match' ? 2 : 8;
  const duel = numTanks === 2;

  for (let m = 0; m < matches; m++) {
    const scores = new Array(8).fill(0);
    const { policies, kinds } = makePolicySet(duel ? 'duel' : 'ffa', 1, opts.seat);
    let round = 0;
    let decided = false;
    let aborted = false;

    while (round < 300) {
      const seed = (opts.seed + m * 104729 + round * 7919) >>> 0;
      const res = runRound({
        seed, policies, maxTime: opts.maxTime,
        numTanks,
      });
      totalRounds++;
      errors += res.stats.errors;
      round++;

      if (res.winner >= 0) {
        scores[res.winner]++;
        if (scores[res.winner] >= opts.roundsToWin) {
          decided = true;
          if (kinds[res.winner] === 'deepseek') ourWins++;
          break;
        }
      }
    }

    if (!decided) { timeouts++; aborted = true; }
    if (aborted && m === 0) {
      console.log('  match aborted after 300 rounds');
    }
  }

  const elapsed = (Date.now() - t0) / 1000;
  const out = {
    label,
    mode,
    matches,
    roundsToWin: opts.roundsToWin,
    matchWinRate: +(ourWins / matches).toFixed(4),
    expectedRandom: duel ? 0.5 : 0.125,
    aborted: timeouts > 0,
    totalRounds,
    avgRoundsPerMatch: +(totalRounds / matches).toFixed(2),
    errors,
    simSeconds: +elapsed.toFixed(2),
  };

  if (!opts.quiet) {
    console.log(
      `${label.padEnd(26)} MATCH win ${(out.matchWinRate * 100).toFixed(1)}%` +
      ` (random ${(out.expectedRandom * 100).toFixed(1)}%)` +
      ` | avg rounds/match ${out.avgRoundsPerMatch}` +
      ` | aborted ${timeouts} | errors ${errors} | ${elapsed.toFixed(1)}s cpu`
    );
  }
  return out;
}

/* ---------------- main ---------------- */

const base = { rounds: OPT.rounds, maxTime: OPT.maxTime, quiet: OPT.quiet, seed: OPT.seed };
const results = [];

if (OPT.mode === 'ffa' || OPT.mode === 'baseline') {
  results.push(simulate('8 built-in bots (baseline)', 'baseline', { ...base, ours: 0, seat: 0 }));
  if (OPT.mode === 'ffa') {
    results.push(simulate(`1 deepseek + 7 bots (seat ${OPT.seat})`, 'ffa', { ...base, ours: OPT.ours, seat: OPT.seat }));
  }
} else if (OPT.mode === 'ours') {
  results.push(simulate(`1 deepseek + 7 bots (seat ${OPT.seat})`, 'ffa', { ...base, ours: 1, seat: OPT.seat }));
} else if (OPT.mode === 'sniper') {
  results.push(simulate('1 deepseek + 7 snipers', 'sniper', { ...base, ours: 1, seat: OPT.seat }));
} else if (OPT.mode === 'duel') {
  results.push(simulate(`duel: deepseek at seat ${OPT.seat}`, 'duel', { ...base, seat: OPT.seat }));
} else if (OPT.mode === 'seats') {
  for (let s = 0; s < 8; s++) {
    results.push(simulate(`ffa seat ${s}`, 'ffa', { ...base, ours: 1, seat: s }));
  }
} else if (OPT.mode === 'tune') {
  const vf = OPT.variants || path.join(HERE, 'variants.json');
  const variants = JSON.parse(fs.readFileSync(vf, 'utf8'));
  for (const v of variants) {
    TT.deepseekPolicy.setParams(v.p || {});
    TT.deepseekPolicy.reset();
    results.push(simulate(v.name, 'ffa', {
      ...base, ours: 1, seat: Number(v.seat != null ? v.seat : OPT.seat),
    }));
  }
} else if (OPT.mode === 'match') {
  results.push(simulateMatch('match ffa seat ' + OPT.seat, 'match', { ...base, seat: OPT.seat, matches: OPT.matches, roundsToWin: OPT.roundsToWin }));
} else if (OPT.mode === 'duel-match') {
  results.push(simulateMatch('match duel seat ' + OPT.seat, 'duel-match', { ...base, seat: OPT.seat, matches: OPT.matches, roundsToWin: OPT.roundsToWin }));
} else if (OPT.mode === 'selfplay') {
  results.push(simulate('8 deepseek self-play', 'selfplay', { ...base, ours: 8, seat: 0 }));
}

if (!OPT.quiet) {
  console.log('\nJSON:', JSON.stringify(results, null, 0));
}

export { runRound, simulate };
