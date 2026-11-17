/* ===========================================================
   harness.mjs — 无头 Arena
   直接加载游戏真实源码（util/map/entities/ai/game），把 DOM 与
   Audio 桩掉，用固定 dt 手动驱动 game.update()。
   这样策略测的就是真实规则，而不是我脑补的规则。
   只在 policy/qwen/ 下写文件，不改动游戏本体。
   =========================================================== */

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const GAME_ROOT = path.resolve(HERE, '..', '..', '..');

/* ---------------- 可复现随机 ---------------- */

export function mulberry32(seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------------- DOM / Audio 桩 ---------------- */

function makeCtx2d() {
  const grad = { addColorStop() {} };
  const ctx = {
    canvas: null,
    fillStyle: '', strokeStyle: '', lineWidth: 1, lineJoin: '', font: '',
    globalAlpha: 1, globalCompositeOperation: 'source-over',
    save() {}, restore() {}, scale() {}, translate() {}, rotate() {},
    setTransform() {}, transform() {}, clearRect() {},
    fillRect() {}, strokeRect() {}, beginPath() {}, closePath() {},
    moveTo() {}, lineTo() {}, arc() {}, arcTo() {}, rect() {},
    fill() {}, stroke() {}, clip() {},
    fillText() {}, strokeText() {}, measureText() { return { width: 0 }; },
    createRadialGradient() { return grad; },
    createLinearGradient() { return grad; },
    createPattern() { return null; },
    drawImage() {},
    getImageData() { return { data: new Uint8ClampedArray(4) }; },
    putImageData() {},
  };
  return ctx;
}

function makeCanvas(w = 1280, h = 800) {
  const cv = {
    width: w, height: h, style: {},
    getContext() { return makeCtx2d(); },
    getBoundingClientRect() { return { width: cv.width, height: cv.height, left: 0, top: 0 }; },
    addEventListener() {},
    toDataURL() { return ''; },
  };
  return cv;
}

function makeDocument() {
  return {
    readyState: 'complete',
    hidden: false,
    createElement(tag) {
      if (tag === 'canvas') return makeCanvas();
      return { style: {}, dataset: {}, classList: { add() {}, remove() {}, toggle() {} }, appendChild() {}, addEventListener() {} };
    },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    getElementById() { return null; },
    addEventListener() {},
    removeEventListener() {},
    documentElement: { style: {} },
    body: { appendChild() {}, style: {} },
  };
}

/* ---------------- 沙箱装载 ---------------- */

const GAME_FILES = ['js/util.js', 'js/map.js', 'js/entities.js', 'js/ai.js', 'js/game.js'];

export function loadTT({ seed = 1, root = GAME_ROOT, files = GAME_FILES } = {}) {
  const rand = mulberry32(seed);

  const sandbox = {
    console,
    Math,
    JSON,
    Date,
    isNaN, isFinite, parseFloat, parseInt,
    Object, Array, String, Number, Boolean, Error, TypeError, RangeError,
    Map, Set, WeakMap, WeakSet, Promise, Symbol, Int32Array, Uint8Array, Uint8ClampedArray, Float64Array, ArrayBuffer,
    __rnd: rand,
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.document = makeDocument();
  sandbox.devicePixelRatio = 1;
  sandbox.requestAnimationFrame = () => 0;
  sandbox.cancelAnimationFrame = () => {};
  sandbox.setTimeout = (fn) => 0;
  sandbox.clearTimeout = () => {};
  sandbox.addEventListener = () => {};
  sandbox.removeEventListener = () => {};
  sandbox.performance = { now: () => 0 };
  sandbox.AudioContext = function () { throw new Error('no audio in headless arena'); };

  const ctx = vm.createContext(sandbox);
  vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

  // 音效整体桩掉：无头环境不需要，也不能依赖 audio.js 的实现细节
  vm.runInContext('window.TT = {}; TT.Audio = {init(){},resume(){},mute(){},roundStart(){},win(){},explode(){},shoot(){},bounce(){},click(){},};', ctx);

  for (const f of files) {
    const code = fs.readFileSync(path.join(root, f), 'utf8');
    vm.runInContext(code, ctx, { filename: f });
  }

  const TT = ctx.TT;
  if (!TT || !TT.Game || !TT.updateBot) throw new Error('game scripts failed to load');

  // 保留内置 bot，之后按座位分发
  const builtinBot = TT.updateBot.bind(TT);

  return { ctx, TT, builtinBot, rand, canvas: makeCanvas() };
}

/* ---------------- 对局驱动 ---------------- */

function makeGame(TT, canvas) {
  const g = new TT.Game(canvas);
  g.render = function () {};        // 无头不渲染
  g.resize = function () {};
  return g;
}

/**
 * 跑一场 Arena 对局。
 * @param {object} o
 * @param {Array<null|string|Function>} o.seats null=内置bot，'qwen'=沙箱内注册的策略，函数=直接注入
 * @param {string} [o.policyFile] 注入沙箱的策略文件（与浏览器同款经典脚本）
 * @param {number} o.roundsToWin
 * @param {number} o.seed
 * @param {number} o.dt 固定步长
 * @param {number} o.maxMatchSeconds 兜底上限
 */
export function playMatch({
  seats,
  policyFile = null,
  roundsToWin = 5,
  seed = 1,
  dt = 1 / 60,
  maxMatchSeconds = 900,
  mapSeeds = null,
} = {}) {
  const { ctx, TT, builtinBot, rand, canvas } = loadTT({ seed });
  if (mapSeeds) TT.CFG.MAP_SEEDS = mapSeeds.slice();

  if (policyFile) {
    const code = fs.readFileSync(policyFile, 'utf8');
    vm.runInContext(code, ctx, { filename: policyFile });
  }

  const brains = seats.map((s) => {
    if (typeof s === 'function') return s;
    if (s === null || s === undefined || s === 'builtin') return null;
    if (typeof s === 'object') {
      const pol = TT.policies && TT.policies[s.policy];
      if (!pol) throw new Error('policy not registered in sandbox: ' + s.policy);
      if (s.tune && typeof pol.variant === 'function') return pol.variant(s.tune);
      return pol.update || pol.think || pol.decide;
    }
    const pol = TT.policies && TT.policies[s];
    if (!pol) throw new Error('policy not registered in sandbox: ' + s);
    return pol.update || pol.think || pol.decide;
  });

  const game = makeGame(TT, canvas);

  const botCalls = new Array(8).fill(0);
  TT.updateBot = function (tank, g, d) {
    const idx = tank.index;
    botCalls[idx]++;
    const fn = brains[idx];
    return fn ? fn(tank, g, d) : builtinBot(tank, g, d);
  };

  /* 死因统计 */
  const deaths = [];
  const origKill = game.killTank.bind(game);
  game.killTank = function (tank, bullet) {
    if (tank.alive) {
      deaths.push({
        round: roundNo,
        t: roundTime,
        victim: tank.index,
        killer: bullet && bullet.owner != null ? bullet.owner : null,
        suicide: !!(bullet && bullet.owner === tank.index),
        bounces: bullet ? bullet.bounces : null,
        x: tank.x, y: tank.y,
      });
    }
    return origKill(tank, bullet);
  };

  /* 回合结果统计（endRound 里才能拿到真实 winner） */
  const roundLog = [];
  let roundNo = 0;
  let roundTime = 0;
  const origEnd = game.endRound.bind(game);
  let duo = null;
  game.endRound = function (winner) {
    if (game.state === 'playing') {
      const alive = game.tanks.filter((t) => t.alive).map((t) => t.index);
      roundLog.push({ round: roundNo, winner, duration: roundTime, aliveAtEnd: alive, duo: duo });
    }
    return origEnd(winner);
  };
  const origStart = game.startRound.bind(game);
  game.startRound = function () {
    origStart();
    roundNo++;
    roundTime = 0;
    duo = null;
  };

  const slots = [];
  for (let i = 0; i < 8; i++) slots.push(seats[i] === undefined ? 'off' : 'bot');

  game.startMatch(slots, roundsToWin);

  const maxTicks = Math.ceil(maxMatchSeconds / dt);
  let ticks = 0;

  while (game.state !== 'matchover' && ticks < maxTicks) {
    if (game.state === 'playing') {
      roundTime += dt;
      /* 记录本回合是否进入过 1v1，以及是哪两个 */
      if (!duo) {
        const al = game.tanks.filter((t) => t.alive);
        if (al.length === 2) duo = al.map((t) => t.index);
      }
    }
    game.update(dt);
    ticks++;
  }

  /* 还没打完的那一回合也计入，否则「僵死回合」正好被统计口径藏起来 */
  if (game.state !== 'matchover') {
    roundLog.push({
      round: roundNo,
      winner: -2,
      duration: roundTime,
      unfinished: true,
      duo: duo,
      aliveAtEnd: game.tanks.filter((t) => t.alive).map((t) => t.index),
    });
  }

  const n = slots.filter((s) => s !== 'off').length;
  const stats = {
    winner: game.matchWinner,
    scores: game.scores.slice(),
    kills: game.kills.slice(),
    slots,
    tankCount: n,
    roundsToWin,
    rounds: roundLog,
    roundsPlayed: roundLog.length,
    deaths,
    simSeconds: ticks * dt,
    timedOut: game.state !== 'matchover',
    seed,
    dt,
    botCalls,
    mapSeeds: TT.CFG.MAP_SEEDS.slice(),
  };

  /* 每座位：胜场 / 击杀 / 自杀 / 死于他人 / 平均存活时长 / 活到回合结束的比率 */
  const per = [];
  for (let i = 0; i < 8; i++) {
    const mine = deaths.filter((d) => d.victim === i);
    const survSum = mine.reduce((a, d) => a + d.t, 0);
    const played = roundLog.length;
    per[i] = {
      idx: i,
      wins: roundLog.filter((r) => r.winner === i).length,
      kills: game.kills[i],
      suicides: mine.filter((d) => d.suicide).length,
      killedByOthers: mine.filter((d) => !d.suicide).length,
      ricochetDeaths: mine.filter((d) => !d.suicide && d.bounces > 0).length,
      avgSurvival: mine.length ? survSum / mine.length : null,
      survivedRate: played ? (played - mine.length) / played : 0,
      rounds: played,
    };
  }
  stats.per = per;
  stats.suicides = deaths.filter((d) => d.suicide).length;
  stats.draws = roundLog.filter((r) => r.winner < 0).length;

  return stats;
}
