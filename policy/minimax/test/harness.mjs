/* ===========================================================
   minimax policy · 无头测试台
   ---------------------------------------------------------------
   直接把项目里真实的 js/*.js 装进 Node 的 vm 里跑，
   地图、碰撞、子弹反弹全部是游戏本体逻辑，不做任何简化，
   这样测出来的胜率才作数。
   =========================================================== */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');

/* ---------------- 假 DOM ---------------- */
function makeCtx() {
  const grad = { addColorStop() {} };
  const base = {
    createRadialGradient: () => grad,
    createLinearGradient: () => grad,
    canvas: null,
  };
  return new Proxy(base, {
    get(t, p) {
      if (p in t) return t[p];
      return function () {};
    },
    set(t, p, v) { t[p] = v; return true; },
  });
}

function makeCanvas() {
  const c = {
    width: 1280,
    height: 800,
    style: {},
    getContext: () => makeCtx(),
    getBoundingClientRect: () => ({ width: 1280, height: 800, left: 0, top: 0, right: 1280, bottom: 800 }),
    addEventListener() {},
    removeEventListener() {},
  };
  return c;
}

/* ---------------- 载入真实游戏代码 ---------------- */
export function loadGame() {
  const sandbox = {
    console,
    Math,
    Date,
    JSON,
    performance,
    devicePixelRatio: 1,
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
    addEventListener() {},
    removeEventListener() {},
    document: {
      addEventListener() {},
      removeEventListener() {},
      createElement(tag) {
        if (tag === 'canvas') return makeCanvas();
        return { style: {}, appendChild() {}, getContext: () => makeCtx() };
      },
      getElementById: () => null,
      querySelector: () => null,
      querySelectorAll: () => [],
      body: { appendChild() {}, style: {} },
    },
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;

  const ctx = vm.createContext(sandbox);

  const files = [
    'js/util.js',
    'js/input.js',
    'js/audio.js',
    'js/map.js',
    'js/entities.js',
    'js/ai.js',
    'js/game.js',
  ];
  for (const f of files) {
    const code = readFileSync(path.join(ROOT, f), 'utf8');
    vm.runInContext(code, ctx, { filename: f });
  }

  /* 给子弹记一个"出生时的寿命"，这样 deathLog 里能算出子弹飞了多久（诊断自杀用） */
  if (sandbox.TT && sandbox.TT.Bullet) {
    const OrigBullet = sandbox.TT.Bullet;
    sandbox.TT.Bullet = class extends OrigBullet {
      constructor(...args) {
        super(...args);
        this.life0 = this.life;
      }
    };
  }

  return { TT: sandbox.TT, sandbox, ctx };
}

/* ---------------- 跑一整场 ---------------- */
/**
 * @param {object} TT       载入后的 TT
 * @param {function} policy 策略函数 (tank, game, dt) -> {drive,steer,shoot}
 * @param {object} opt
 */
export function runMatch(TT, policy, opt = {}) {
  const {
    policyIndex = 0,
    opponent = null,
    tankCount = 8,
    roundsToWin = 5,
    dt = 1 / 60,
    maxSeconds = 900,
    roundTimeout = 40,
  } = opt;

  const canvas = makeCanvas();
  const game = new TT.Game(canvas);

  const slots = [];
  for (let i = 0; i < 8; i++) slots.push(i < tankCount ? 'bot' : 'off');

  /* 把内置 AI 包一层：策略席位走 policy，其余走 opponent（没给就用内置 AI） */
  const orig = TT.updateBot;
  let policyError = null;
  TT.updateBot = function (tank, g, d) {
    if (tank.index === policyIndex) {
      try {
        return policy(tank, g, d);
      } catch (e) {
        if (!policyError) policyError = e;
        return { drive: 0, steer: 0, shoot: false };
      }
    }
    if (opponent) {
      try {
        return opponent(tank, g, d);
      } catch (e) {
        return { drive: 0, steer: 0, shoot: false };
      }
    }
    return orig(tank, g, d);
  };

  game.startMatch(slots, roundsToWin);

  /* 统计死亡、自杀、回合时长 */
  const deaths = new Array(8).fill(0);
  const selfKills = new Array(8).fill(0);
  const killEvents = new Array(8).fill(0);
  const roundDur = [];
  const deathLog = [];          // 被策略控制的坦克阵亡时的现场快照
  const origKill = TT.Game.prototype.killTank;
  TT.Game.prototype.killTank = function (tank, bullet) {
    if (tank.alive) {
      deaths[tank.index]++;
      if (bullet && bullet.owner != null) {
        if (bullet.owner === tank.index) selfKills[tank.index]++;
        else killEvents[bullet.owner]++;
      }
      if (tank.index === policyIndex) {
        const snap = {
          killer: bullet ? bullet.owner : -1,
          self: bullet ? bullet.owner === tank.index : false,
          dist: bullet ? Math.hypot(bullet.x - tank.x, bullet.y - tank.y) : -1,
          /* 自杀诊断：子弹飞了多久、弹了几次 */
          bulletAge: bullet && bullet.life != null ? (bullet.life0 != null ? bullet.life0 - bullet.life : -1) : -1,
          bulletBounces: bullet ? bullet.bounces : -1,
          speed: tank.speed,
          alive: this.tanks.filter((t) => t.alive).length,
          myBullets: this.bullets.filter((b) => !b.dead && b.owner === tank.index).length,
          incoming: this.bullets.filter((b) => !b.dead && b.owner !== tank.index).length,
          angle: tank.angle,
          roundTime,
        };
        deathLog.push(snap);
      }
    }
    return origKill.call(this, tank, bullet);
  };

  let t = 0;
  let roundTime = 0;
  let timeouts = 0;
  let rounds = 0;
  const roundWinners = [];

  while (game.state !== 'matchover' && t < maxSeconds) {
    const before = game.state;
    game.update(dt);
    t += dt;
    if (before === 'playing') {
      if (game.state !== 'playing') {
        /* 回合刚结束：此刻 roundWinner 还有效 */
        rounds++;
        roundWinners.push(game.roundWinner);
        roundDur.push(roundTime);
        roundTime = 0;
      } else {
        roundTime += dt;
        if (roundTime > roundTimeout) {
          timeouts++;
          game.endRound(-1);
          rounds++;
          roundWinners.push(-1);
          roundDur.push(roundTime);
          roundTime = 0;
        }
      }
    }
  }
  TT.Game.prototype.killTank = origKill;

  const result = {
    scores: game.scores.slice(),
    kills: game.kills.slice(),
    deaths,
    selfKills,
    killEvents,
    roundDur,
    rounds,
    timeouts,
    seconds: t,
    roundWinners,
    deathLog,
    policyError,
    policyIndex,          /* 策略这一场坐的是几号席位（随机席位测试要用） */
    winner: game.matchWinner,
  };

  TT.updateBot = orig;
  return result;
}

/* ---------------- 汇总 ---------------- */
/**
 * @param {Array} results     runMatch 的结果数组
 * @param {number} policyIndex 策略的默认席位（结果里带 policyIndex 时以它为准）
 */
export function summarize(results, policyIndex = 0) {
  const n = results.length;
  const wins = new Array(8).fill(0);
  const kills = new Array(8).fill(0);
  const deaths = new Array(8).fill(0);
  const selfKills = new Array(8).fill(0);
  const killEvents = new Array(8).fill(0);
  let selfHits = 0, timeouts = 0, errors = 0, rounds = 0;
  let roundDraws = 0;
  let durSum = 0, durN = 0;

  /* 策略自己的成绩（席位随机时按每场实际席位统计） */
  let pWins = 0, pKills = 0, pDeaths = 0, pSelf = 0;

  for (const r of results) {
    for (let i = 0; i < 8; i++) {
      wins[i] += r.scores[i];
      kills[i] += r.kills[i];
      deaths[i] += r.deaths[i];
      selfKills[i] += r.selfKills[i];
      killEvents[i] += r.killEvents[i];
    }
    const pi = (r.policyIndex != null) ? r.policyIndex : policyIndex;
    pWins += r.scores[pi];
    pKills += r.kills[pi];
    pDeaths += r.deaths[pi];
    pSelf += r.selfKills[pi];

    timeouts += r.timeouts;
    rounds += r.rounds;
    for (const w of r.roundWinners) if (w < 0) roundDraws++;
    for (const d of r.roundDur) { durSum += d; durN++; }
    if (r.policyError) errors++;
    if (r.kills[pi] < 0) selfHits += -r.kills[pi];
  }

  return {
    n, wins, kills, deaths, selfKills, killEvents,
    selfHits, timeouts, errors, rounds, roundDraws,
    avgRound: durN ? durSum / durN : 0,
    /* 策略本体的成绩（与席位无关） */
    policy: { wins: pWins, kills: pKills, deaths: pDeaths, selfKills: pSelf },
  };
}

export { ROOT };
