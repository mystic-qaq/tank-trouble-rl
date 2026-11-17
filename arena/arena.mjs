/* ===========================================================
   arena.mjs — Tank Trouble 统一竞技场（无头）
   -----------------------------------------------------------
   从 policy/ 下的 7 份 AI 策略 + 游戏内置 Bot 中任选 2–8 个参赛，
   按真实引擎规则跑完整回合，统计胜率与击杀。

   设计原则：**不改动任何策略源码**。
   - 5 份 classic script（deepseek/glm/gpt/hy/qwen）原样 vm.runInContext 载入；
   - 2 份 ESM（kimi/minimax）用 vm.SourceTextModule 载入同一个沙箱上下文，
     这样它们的 globalThis 就是沙箱全局，能读到 TT.CFG。
   - 策略一律通过它自己注册到 TT 上的入口取出（见 STRATEGIES.adapter），
     调用签名统一为 (tank, game, dt) -> {drive, steer, shoot}。

   竞技场侧只做三件事（都不碰策略内部逻辑）：
   1) 布局配平：固定地图和出生点，让所有策略轮换遍历每个席位；
   2) 地图配额：按完整轮换组交替两张图，2×人数局即可完整配平；
   3) 每局独立沙箱 + 独立随机种子，保证结果可复现且互不串状态。
   =========================================================== */

import './schedule.js';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, '..');

/* ---------------- 参赛名单 ---------------- */

export const STRATEGIES = [
  {
    id: 'deepseek', label: 'DeepSeek', kind: 'classic',
    file: 'policy/deepseek/policy.js',
    adapter: (TT) => TT.deepseekPolicy || (TT.policies && TT.policies.deepseek),
  },
  {
    id: 'glm', label: 'GLM', kind: 'classic',
    file: 'policy/glm/glm_policy.js',
    adapter: (TT) => (TT.Policies && TT.Policies.glm && TT.Policies.glm.update) || TT.GLM_POLICY,
  },
  {
    id: 'gpt', label: 'GPT Sentinel', kind: 'classic',
    file: 'policy/gpt/ai.js',
    adapter: (TT) => (TT.policies && TT.policies.gpt) || TT.updateBotGPT,
  },
  {
    id: 'hy', label: 'HY', kind: 'classic',
    file: 'policy/hy/policy.js',
    adapter: (TT) => (TT.Policies && TT.Policies.hy) || TT.HYPolicy,
  },
  {
    id: 'kimi', label: 'Kimi', kind: 'esm',
    file: 'policy/kimi/policy.mjs',
    adapter: (TT, ns) => (ns && ns.default) || (ns && ns.policy),
  },
  {
    id: 'minimax', label: 'minimax', kind: 'esm',
    file: 'policy/minimax/policy.js',
    adapter: (TT, ns) => (ns && ns.default) || (ns && ns.updateBot),
  },
  {
    id: 'qwen', label: 'Qwen', kind: 'classic',
    file: 'policy/qwen/policy.js',
    adapter: (TT) => (TT.policies && TT.policies.qwen && TT.policies.qwen.update) || TT.qwenPolicy,
  },
  {
    id: 'builtin', label: '内置 Bot', kind: 'builtin',
    file: 'js/ai.js',
    adapter: null,   // 用引擎原生 TT.updateBot
  },
];

export const STRATEGY_IDS = STRATEGIES.map((s) => s.id);
export const MAP_SEEDS = [20260101, 88888888];
export const RULES_VERSION = '2026-11-06-60s-v1';

/* ---------------- 可复现随机 ---------------- */

export function mulberry32(seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* 每局独立种子：同样的 (baseSeed, roundIndex) 永远得到同一局 */
export function roundSeed(baseSeed, roundIndex) {
  let h = (baseSeed >>> 0) ^ 0x9e3779b9;
  h = Math.imul(h ^ (roundIndex + 0x85ebca6b), 0xc2b2ae35);
  h ^= h >>> 13;
  return (h >>> 0) || 1;
}

/* ---------------- 座位轮转 + 地图配额 ---------------- */

/* 共用浏览器排程，每组使用相同出生种子，策略逐局轮换。 */
export const validateRoster = globalThis.TTArenaSchedule.roster;
export function seatPlan(roundIndex, players = STRATEGY_IDS) {
  return globalThis.TTArenaSchedule.plan(roundIndex, players).seats;
}

export function mapPlan(roundIndex, players = STRATEGY_IDS) {
  return globalThis.TTArenaSchedule.plan(roundIndex, players).mapSeed;
}

/* ---------------- DOM / Canvas / Audio 桩 ---------------- */

function makeCtx2d() {
  const grad = { addColorStop() {} };
  return {
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

/* ---------------- 沙箱 ---------------- */

const GAME_FILES = ['js/util.js', 'js/map.js', 'js/entities.js', 'js/ai.js', 'js/game.js'];

/**
 * 建一个全新的竞技场沙箱：载入引擎本体 + 全部 7 份策略。
 * 每次调用得到一个完全隔离的环境（策略内部状态互不串味）。
 */
export async function createArena() {
  const sandbox = {
    console,
    JSON, Date,
    isNaN, isFinite, parseFloat, parseInt,
    Object, Array, String, Number, Boolean, Error, TypeError, RangeError,
    Map, Set, WeakMap, WeakSet, Promise, Symbol,
    Int32Array, Uint8Array, Uint8ClampedArray, Float64Array, ArrayBuffer,
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  sandbox.document = makeDocument();
  sandbox.devicePixelRatio = 1;
  sandbox.requestAnimationFrame = () => 0;
  sandbox.cancelAnimationFrame = () => {};
  /* setTimeout 桩：不让 glm 的「600ms 后接管 TT.updateBot」兜底逻辑生效，
     否则它会抢走整张竞技场的控制权。竞技场自己接管 TT.updateBot。 */
  sandbox.setTimeout = () => 0;
  sandbox.clearTimeout = () => {};
  sandbox.addEventListener = () => {};
  sandbox.removeEventListener = () => {};
  sandbox.performance = { now: () => 0 };

  const ctx = vm.createContext(sandbox);
  const TT = sandbox.TT = {};

  /* 音频整体桩掉：无头环境不需要 */
  TT.Audio = {
    init() {}, resume() {}, mute() {}, roundStart() {}, win() {},
    explode() {}, shoot() {}, bounce() {}, click() {},
  };
  TT.Input = { get: () => ({ drive: 0, steer: 0, shoot: false }), clear() {} };

  for (const f of GAME_FILES) {
    const code = fs.readFileSync(path.join(ROOT, f), 'utf8');
    vm.runInContext(code, ctx, { filename: f });
  }
  if (!TT.Game || !TT.updateBot) throw new Error('引擎本体加载失败');

  /* 内置 Bot 原件 —— 必须在任何策略加载之前抓住，
     因为 policy/gpt/ai.js 在加载时会把 TT.updateBot 覆盖成它自己。 */
  const builtinBot = TT.updateBot;

  /* 载入 7 份策略 */
  const adapters = {};
  for (const st of STRATEGIES) {
    if (st.kind === 'builtin') { adapters[st.id] = builtinBot; continue; }

    const full = path.join(ROOT, st.file);
    const code = fs.readFileSync(full, 'utf8');
    let ns = null;
    if (st.kind === 'esm') {
      if (typeof vm.SourceTextModule !== 'function') {
        throw new Error('加载 ESM 策略需要 node --experimental-vm-modules');
      }
      const mod = new vm.SourceTextModule(code, { context: ctx, identifier: st.file });
      await mod.link(() => { throw new Error(st.file + ' 不应有 import'); });
      await mod.evaluate();
      ns = mod.namespace;
    } else {
      vm.runInContext(code, ctx, { filename: st.file });
    }
    const fn = st.adapter(TT, ns);
    if (typeof fn !== 'function') {
      throw new Error('策略 ' + st.id + ' 未注册出可用的入口函数');
    }
    adapters[st.id] = fn;
  }

  return { ctx, TT, sandbox, adapters, canvas: makeCanvas() };
}

/* ---------------- 单局 ---------------- */

const DT = 1 / 60;
const MAX_FRAMES = 60 * 60 + 2; // 60 秒交战 + intro 状态转换帧；时限由引擎判定

/**
 * 跑一局 2–8 车混战。
 * @param {number} baseSeed 基础种子
 * @param {number} roundIndex 第几局（0 起）
 * @param {boolean} verbose 是否打印逐局信息
 */
export async function playRound(baseSeed, roundIndex, verbose = false, players = STRATEGY_IDS) {
  players = validateRoster(players);
  const env = await createArena();
  const { TT, ctx, sandbox, adapters } = env;

  /* 本局随机源：每局独立，互不影响 */
  sandbox.__rnd = mulberry32(roundSeed(baseSeed, roundIndex));
  vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

  const { seats, mapSeed, spawnSeed } = globalThis.TTArenaSchedule.plan(roundIndex, players, baseSeed);

  /* 本局配置：跳过 1.15s 的「ROUND n」横幅等待（该阶段不做任何决策） */
  TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
  TT.CFG.ROUND_INTRO = 0;

  const game = new TT.Game(env.canvas);
  game.render = function () {};
  game.resize = function () {};
  game.mode = 'match';
  game.slots = Array.from({ length: 8 }, (_, i) => i < seats.length ? 'bot' : 'off');
  game.spawnRandom = TT.util.rng(spawnSeed);
  game.roundsToWin = 9999;

  /* 统计 */
  const errors = {};
  const kills = {};
  const suicides = {};
  const deaths = {};
  for (const id of players) { errors[id] = 0; kills[id] = 0; suicides[id] = 0; deaths[id] = 0; }

  /* 淘汰顺序与死亡时刻：用于统计「决赛率」「平均存活时长」 */
  const eliminationOrder = [];
  const deathTime = {};
  const deathBy = {};      /* 被谁打死 */
  const deathBounces = {}; /* 死于几段反弹的子弹 */

  let finished = false;
  let winnerIndex = -1;
  let timeout = false;
  let frames = 0;
  let elapsed = 0;

  /* 死亡归因：按本局座位表把 slot 映射回策略 */
  const origKill = game.killTank.bind(game);
  game.killTank = function (tank, bullet) {
    if (tank.alive) {
      const victim = seats[tank.index] || ('slot' + tank.index);
      deaths[victim] = (deaths[victim] || 0) + 1;
      eliminationOrder.push(victim);
      deathTime[victim] = +game.roundTime.toFixed(2);
      deathBy[victim] = (bullet && bullet.owner != null && bullet.owner !== tank.index)
        ? (seats[bullet.owner] || ('slot' + bullet.owner))
        : (bullet && bullet.owner === tank.index ? victim : 'unknown');
      deathBounces[victim] = bullet ? bullet.bounces : null;
      if (bullet && bullet.owner != null && bullet.owner !== tank.index) {
        const killer = seats[bullet.owner] || ('slot' + bullet.owner);
        kills[killer] = (kills[killer] || 0) + 1;
      } else {
        suicides[victim] = (suicides[victim] || 0) + 1;
      }
    }
    return origKill(tank, bullet);
  };

  const origEnd = game.endRound.bind(game);
  game.endRound = function (winner) {
    finished = true;
    winnerIndex = winner;
    return origEnd(winner);
  };

  /* 竞技场接管 TT.updateBot：按座位分发给对应策略 */
  TT.updateBot = function (tank, g, dt) {
    const id = seats[tank.index];
    const fn = adapters[id] || builtinFallback(adapters);
    try {
      const out = fn(tank, g, dt);
      if (!out) return { drive: 0, steer: 0, shoot: false };
      /* 归一化字段：drive 只需要符号，steer 原样透传（不截断 ——
         deepseek/hy 用分数转向精确落角，截断会破坏其瞄准） */
      return {
        drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
        steer: Number(out.steer) || 0,
        shoot: !!out.shoot,
      };
    } catch (e) {
      errors[id] = (errors[id] || 0) + 1;
      return { drive: 0, steer: 0, shoot: false };
    }
  };

  /* 开局 */
  game.startRound();

  /* 出生布局快照：让「出生点」成为可分析的协变量，而不只是一个 seed。
     openSides = 该格子朝外的通路数，越小越像「死角」。 */
  const spawns = game.tanks.map((t) => {
    const r = Math.floor(t.y / TT.CFG.CELL);
    const c = Math.floor(t.x / TT.CFG.CELL);
    return {
      seat: t.index,
      x: +t.x.toFixed(1), y: +t.y.toFixed(1), a: +t.angle.toFixed(3),
      open: game.map.openSides(r, c),
    };
  });

  while (!finished && frames < MAX_FRAMES) {
    game.update(DT);
    frames++;
    elapsed = frames * DT;
    if (game.state === 'over') { finished = true; winnerIndex = game.roundWinner; }
  }
  if (!finished) { timeout = true; winnerIndex = -1; }

  const alive = game.tanks.filter((t) => t.alive).map((t) => seats[t.index]);

  /* 决赛：最后活着的两人 = 胜者 + 最后被淘汰的那个 */
  const finalists = [];
  if (winnerIndex >= 0) finalists.push(seats[winnerIndex]);
  if (eliminationOrder.length) {
    const last = eliminationOrder[eliminationOrder.length - 1];
    if (finalists.indexOf(last) < 0) finalists.push(last);
  }

  const result = {
    rulesVersion: RULES_VERSION,
    roundTimeLimitSec: TT.CFG.ROUND_TIME_LIMIT,
    round: roundIndex,
    winner: winnerIndex >= 0 ? (seats[winnerIndex] || null) : null,
    timeout: timeout || game.roundTimedOut,
    durationSec: +game.roundTime.toFixed(2),
    seats: seats.slice(),
    mapSeed, spawnSeed, spawns,
    kills, suicides, deaths,
    errors,
    aliveAtEnd: alive,
    survivors: alive.length,
    eliminationOrder,
    deathTime,
    deathBy,
    deathBounces,
    finalists,
  };

  if (verbose) {
    console.log(
      `#${String(roundIndex + 1).padStart(3)}  胜者=${result.winner || '平局'}` +
      `  时长=${result.durationSec}s` +
      `  存活=${alive.join(',')}` +
      `  击杀=${Object.entries(kills).filter(([, v]) => v).map(([k, v]) => k + ':' + v).join(' ')}`
    );
  }
  return result;
}

function builtinFallback(adapters) {
  return adapters.builtin;
}

/* ---------------- 批量 ---------------- */

/**
 * 跑一批局。逐局独立沙箱，结果与顺序无关（可并行）。
 */
export async function playRounds({ baseSeed, from, to, verbose = false, onRound = null, players = STRATEGY_IDS }) {
  const out = [];
  for (let r = from; r < to; r++) {
    const res = await playRound(baseSeed, r, verbose, players);
    out.push(res);
    if (onRound) onRound(res);
  }
  return out;
}
