/* =====================================================================
   Headless Arena simulation using the REAL game engine.

     node policy/hy/test/sim.js --matches 10 --rounds 3 --frames 12000
     node policy/hy/test/sim.js --baseline            # 8 built-in bots

   Only reads files outside policy/hy; writes nothing.
   ===================================================================== */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '../../..');
const POLICY = path.resolve(__dirname, '../policy.js');
const argv = process.argv.slice(2);

function arg(name, def) {
  const i = argv.indexOf('--' + name);
  return (i >= 0 && argv[i + 1]) ? argv[i + 1] : def;
}
const MATCHES = parseInt(arg('matches', '10'), 10);
const MAX_FRAMES = parseInt(arg('frames', '12000'), 10);
const ROUNDS = parseInt(arg('rounds', '3'), 10);
const HY_INDEX = parseInt(arg('hy', '0'), 10);
const BASELINE = argv.includes('--baseline');
const VERBOSE = argv.includes('--verbose');
const DIAG = argv.includes('--diag');
const DT = 1 / 60;

/* ---------- deterministic RNG so runs are comparable ---------- */
function mulberry(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const M2 = Object.create(Math);
M2.random = mulberry(parseInt(arg('seed', '424242'), 10));

/* ---------- DOM / audio stubs ---------- */
function fakeCtx() {
  const noop = function () {};
  return new Proxy({}, {
    get(t, k) {
      if (k === 'createRadialGradient' || k === 'createLinearGradient') {
        return function () { return { addColorStop: noop }; };
      }
      if (k in t) return t[k];
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; }
  });
}
function fakeCanvas() {
  return {
    width: 0, height: 0,
    getBoundingClientRect: () => ({ width: 1280, height: 800, left: 0, top: 0 }),
    getContext: () => fakeCtx()
  };
}

let clock = 0;
const realWarn = console.warn;
let errorCount = 0;
const sandbox = {
  Math: M2,
  console: {
    log: console.log,
    warn: function () {
      errorCount++;
      if (errorCount <= 6) realWarn.apply(console, arguments);
    }
  },
  Date, JSON,
  Uint8Array, Int8Array, Int32Array, Float32Array, Float64Array,
  Map, WeakMap, Set, Proxy, isFinite,
  performance: { now: () => clock },
  setTimeout: function () { return 0; },   /* hook is installed explicitly below */
  requestAnimationFrame: function () { return 0; },
  addEventListener: function () {},
  document: { createElement: () => fakeCanvas() },
  AudioContext: function () {}
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

/* ---------- load engine ---------- */
const files = ['js/util.js', 'js/input.js', 'js/audio.js', 'js/map.js',
  'js/entities.js', 'js/ai.js', 'js/game.js'];
for (const f of files) {
  vm.runInContext(fs.readFileSync(path.join(ROOT, f), 'utf8'), sandbox, { filename: f });
}
sandbox.TT.Audio = new Proxy({}, { get: () => function () {} });

/* ---------- load policy ---------- */
vm.runInContext(fs.readFileSync(POLICY, 'utf8'), sandbox, { filename: 'policy.js' });
const HY = sandbox.HYPolicy;
if (typeof HY !== 'function') throw new Error('policy did not register a callable API');

sandbox.__HY_DEBUG__ = true;
if (!BASELINE) HY.install();          /* wrap TT.updateBot for tanks marked __hy */

const TT = sandbox.TT;
const game = new TT.Game(fakeCanvas());
const slots = new Array(8).fill('bot');

/* ---------- instrumentation ---------- */
const stats = {
  matches: 0, champ: 0, roundWins: 0, rounds: 0,
  selfKill: 0, hyKill: 0, hyDeath: 0, timeouts: 0, frames: 0
};
const deaths = [];
const origKill = game.killTank;
game.killTank = function (tank, bullet) {
  if (bullet && bullet.owner != null) {
    if (bullet.owner === HY_INDEX && tank.index === HY_INDEX) stats.selfKill++;
    if (bullet.owner === HY_INDEX && tank.index !== HY_INDEX) stats.hyKill++;
  }
  if (tank.index === HY_INDEX) {
    stats.hyDeath++;
    if (DIAG && deaths.length < 14) {
      deaths.push({
        round: game.round,
        owner: bullet ? bullet.owner : -1,
        bounces: bullet ? bullet.bounces : -1,
        speed: Math.round(tank.speed),
        distToShooter: (function () {
          if (!bullet) return -1;
          for (let i = 0; i < this.tanks.length; i++) {
            if (this.tanks[i].index === bullet.owner) {
              return Math.round(Math.hypot(this.tanks[i].x - tank.x, this.tanks[i].y - tank.y));
            }
          }
          return -1;
        }).call(this)
      });
    }
  }
  return origKill.call(this, tank, bullet);
};

/* ---------- run ---------- */
const started = Date.now();
for (let m = 0; m < MATCHES; m++) {
  game.startMatch(slots, ROUNDS);
  let frames = 0;
  while (game.state !== 'matchover' && frames < MAX_FRAMES) {
    if (!BASELINE) {
      for (let i = 0; i < game.tanks.length; i++) {
        game.tanks[i].__hy = (game.tanks[i].index === HY_INDEX);
      }
    }
    clock += DT * 1000;
    game.update(DT);
    frames++;
  }
  stats.frames += frames;
  stats.matches++;
  if (game.matchWinner === HY_INDEX) stats.champ++;
  stats.roundWins += game.scores[HY_INDEX];
  stats.rounds += game.round;
  if (game.state !== 'matchover') {
    stats.timeouts++;
    if (VERBOSE || DIAG) {
      const hyTank = game.tanks.filter(t => t.index === HY_INDEX)[0];
      console.log('  TIMEOUT detail | hyAlive:', hyTank ? hyTank.alive : false,
        '| alive:', game.tanks.filter(t => t.alive).length,
        '| round:', game.round, '| HY wins:', game.scores[HY_INDEX],
        '| scores:', JSON.stringify(game.scores));
    }
  }
  if (VERBOSE) {
    console.log('match', m + 1,
      '| winner', game.matchWinner,
      '| rounds', game.round,
      '| HY wins', game.scores[HY_INDEX],
      '| HY kills', game.kills[HY_INDEX],
      '| frames', frames);
  }
}

const secs = (Date.now() - started) / 1000;
console.log('\n================ HY ARENA REPORT ================');
console.log('mode        :', BASELINE ? 'BASELINE (8 built-in bots)' : 'HY vs 7 built-in bots');
console.log('matches     :', stats.matches, '| rounds:', stats.rounds,
  '| frames:', stats.frames, '| timeouts:', stats.timeouts);
console.log('champion    :', stats.champ + '/' + stats.matches,
  '=', (100 * stats.champ / stats.matches).toFixed(1) + '%');
console.log('round wins  :', stats.roundWins + '/' + stats.rounds,
  '=', (100 * stats.roundWins / Math.max(1, stats.rounds)).toFixed(1) + '%');
console.log('HY kills    :', stats.hyKill, '| self kills:', stats.selfKill,
  '| HY deaths:', stats.hyDeath);
console.log('policy errs :', errorCount);
console.log('wall clock  :', secs.toFixed(1) + 's',
  '|', (stats.frames / Math.max(0.001, secs)).toFixed(0), 'frames/s');
if (DIAG && deaths.length) {
  console.log('--- HY deaths ---');
  for (const d of deaths) console.log(JSON.stringify(d));
}
console.log('==================================================');
