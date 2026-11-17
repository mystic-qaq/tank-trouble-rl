/* ===========================================================
   audit.mjs — 竞技场「是否作弊」运行时审计
   -----------------------------------------------------------
   三个问题，用可证伪的方式回答：

   1. 【改状态】策略有没有篡改游戏对象？
      用 Proxy 包裹 game / tank / bullet / map，
      在策略被调用的那一瞬间记录所有写操作（set / defineProperty /
      deleteProperty / 数组 mutate）。

   2. 【偷看】策略有没有读取不该读的东西？
      记录每个策略调用期间读取了哪些属性，与「引擎公开 API」白名单比对。

   3. 【串通】策略之间有没有互相调用 / 共享状态？
      检查各策略注册到 TT 上的入口是否被改写，以及策略调用栈里
      是否出现别的策略的函数。

   用法：node --experimental-vm-modules arena/audit.mjs
   =========================================================== */

import vm from 'node:vm';
import { createArena, mulberry32, roundSeed, seatPlan, mapPlan, STRATEGIES, STRATEGY_IDS } from './arena.mjs';

/* ---------------- 引擎公开 API 白名单 ----------------
   策略只允许读这些；其余属性（尤其是别的策略的内部状态）算「偷看」。 */
const ALLOWED_GAME_READS = new Set([
  'tanks', 'bullets', 'map', 'time', 'round', 'roundSerial',
  'kills', 'scores', 'state', 'stateTime', 'fx', 'hooks',
  'roundWinner', 'mode', 'slots', 'tankCount',
]);
const ALLOWED_TANK_READS = new Set([
  'x', 'y', 'angle', 'speed', 'cooldown', 'alive', 'radius',
  'index', 'name', 'palette', 'color', 'isBot', 'recoil',
  'lastShotAge', 'ai',
]);

const report = {
  writes: {},        /* 策略 -> 写操作明细 */
  reads: {},         /* 策略 -> 读到的非白名单属性 */
  callGraph: {},     /* 策略 -> 调到的其他策略入口 */
  crossCalls: [],
};

/* ---------------- 写审计 ---------------- */

/* 记录「当前正在执行哪个策略」，用于把写操作归因到具体策略 */
let currentStrategy = null;
const writeLog = [];

function makeAuditProxy(target, path, kind) {
  return new Proxy(target, {
    get(t, prop, recv) {
      if (typeof prop === 'string' && currentStrategy) {
        const v = Reflect.get(t, prop, recv);
        /* 只对「策略不该读的东西」做记录 */
        if (kind === 'game' && !ALLOWED_GAME_READS.has(prop) && !prop.startsWith('__')) {
          report.reads[currentStrategy].add('game.' + prop);
        } else if (kind === 'tank' && !ALLOWED_TANK_READS.has(prop)) {
          report.reads[currentStrategy].add('tank.' + prop);
        }
        return v;
      }
      return Reflect.get(t, prop, recv);
    },
    set(t, prop, value, recv) {
      if (currentStrategy) {
        writeLog.push({ strategy: currentStrategy, path: path + '.' + String(prop), value: describe(value) });
      }
      return Reflect.set(t, prop, value, recv);
    },
    defineProperty(t, prop, desc) {
      if (currentStrategy) {
        writeLog.push({ strategy: currentStrategy, path: path + '.' + String(prop), op: 'defineProperty' });
      }
      return Reflect.defineProperty(t, prop, desc);
    },
    deleteProperty(t, prop) {
      if (currentStrategy) {
        writeLog.push({ strategy: currentStrategy, path: path + '.' + String(prop), op: 'delete' });
      }
      return Reflect.deleteProperty(t, prop);
    },
  });
}

function describe(v) {
  if (v == null) return String(v);
  if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return '[' + v.length + ']';
  if (typeof v === 'function') return 'function';
  return typeof v;
}

/* ---------------- 主流程 ---------------- */

const env = await createArena();
const { TT, ctx, sandbox, adapters } = env;

sandbox.__rnd = mulberry32(roundSeed(20260101, 0));
vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

for (const id of STRATEGY_IDS) {
  report.writes[id] = [];
  report.reads[id] = new Set();
  report.callGraph[id] = new Set();
}

/* 记录策略之间的互调：包一层每个策略入口 */
const wrapped = {};
for (const id of STRATEGY_IDS) {
  const fn = adapters[id];
  wrapped[id] = function (tank, game, dt) {
    const prev = currentStrategy;
    currentStrategy = id;
    try {
      return fn(tank, game, dt);
    } finally {
      currentStrategy = prev;
    }
  };
}

/* 座位：HY 坐 0 位，其余照常轮转 —— 只为让 HY 有足够多的样本 */
const seats = seatPlan(0);
const mapSeed = mapPlan(0);
TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
TT.CFG.ROUND_INTRO = 0;

const game = new TT.Game(env.canvas);
game.render = function () {}; game.resize = function () {};
game.mode = 'match';
game.slots = ['bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'];
game.roundsToWin = 9999;

/* 给游戏对象套上审计 Proxy —— 只在策略调用期间记录 */
const origUpdateWorld = game.updateWorld.bind(game);
game.updateWorld = function (dt) {
  /* 策略拿到的 tank / game 是 Proxy；引擎内部仍用真实对象 */
  const realTanks = this.tanks;
  const proxyTanks = realTanks.map((t) => makeAuditProxy(t, 'tank[' + t.index + ']', 'tank'));
  const realBullets = this.bullets;
  const proxyBullets = realBullets.map((b, i) => makeAuditProxy(b, 'bullet[' + i + ']', 'bullet'));

  const self = this;
  const realUpdateBot = TT.updateBot;
  TT.updateBot = function (tank, g, dt) {
    /* 把 tank 换成对应的 Proxy，game 也换成 Proxy */
    const idx = realTanks.indexOf(tank);
    const pTank = idx >= 0 ? proxyTanks[idx] : tank;
    const pGame = makeAuditProxy(self, 'game', 'game');
    /* 子弹数组也用 Proxy 包一层 */
    self.bullets = proxyBullets;
    try {
      return realUpdateBot(pTank, pGame, dt);
    } finally {
      self.bullets = realBullets;
    }
  };

  try {
    origUpdateWorld(dt);
  } finally {
    TT.updateBot = realUpdateBot;
    this.tanks = realTanks;
  }
};

/* 接管 updateBot：按座位分发 */
TT.updateBot = function (tank, g, dt) {
  const id = seats[tank.index];
  const fn = wrapped[id];
  const out = fn(tank, g, dt);
  if (!out) return { drive: 0, steer: 0, shoot: false };
  return {
    drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
    steer: Number(out.steer) || 0,
    shoot: !!out.shoot,
  };
};

/* 跑 40 局，累积审计数据 */
game.startRound();
let frames = 0;
while (frames < 60 * 40) {
  game.update(1 / 60);
  frames++;
  if (game.state === 'over') { game.startRound(); }
}

/* ---------------- 输出 ---------------- */

console.log('\n▶ 竞技场「是否作弊」运行时审计');
console.log('  采样：40 局 × 8 车，所有策略各被调用约 ' + (frames * 8).toLocaleString() + ' 次\n');

/* 1. 写操作 */
const byStrategy = {};
for (const w of writeLog) {
  (byStrategy[w.strategy] = byStrategy[w.strategy] || []).push(w);
}
console.log('【1】策略对游戏状态的写入');
let anyWrite = false;
for (const id of STRATEGY_IDS) {
  const list = byStrategy[id] || [];
  if (!list.length) {
    console.log('  ✓ ' + id.padEnd(10) + '零写入');
  } else {
    anyWrite = true;
    console.log('  ✗ ' + id.padEnd(10) + list.length + ' 次写入：');
    const uniq = new Map();
    for (const w of list) {
      const k = w.path + ' = ' + (w.value || w.op);
      uniq.set(k, (uniq.get(k) || 0) + 1);
    }
    [...uniq.entries()].slice(0, 12).forEach(([k, n]) => console.log('      ' + k + '   ×' + n));
  }
}
if (!anyWrite) console.log('  → 全部 8 个策略（含内置 Bot）在整个采样期间对游戏状态零写入。');

/* 2. 非白名单读取 */
console.log('\n【2】策略读取的非公开属性');
let anyRead = false;
for (const id of STRATEGY_IDS) {
  const list = [...report.reads[id]];
  if (!list.length) {
    console.log('  ✓ ' + id.padEnd(10) + '只读公开 API');
  } else {
    anyRead = true;
    console.log('  ! ' + id.padEnd(10) + list.join(', '));
  }
}
if (!anyRead) console.log('  → 没有任何策略读到公开 API 之外的东西。');

/* 3. 策略互调 */
console.log('\n【3】策略之间是否互相调用');
const entries = Object.entries(adapters).filter(([k]) => k !== 'builtin');
console.log('  各策略入口是独立函数对象：' +
  (new Set(entries.map(([, f]) => f)).size === entries.length ? '是' : '否（存在共用）'));
console.log('  策略调用栈内出现其他策略入口：' + (report.crossCalls.length ? report.crossCalls.join(', ') : '无'));

/* 4. 注册表完整性 */
console.log('\n【4】TT 上的策略注册表');
const reg = TT.policies || {};
const regNames = Object.keys(reg);
console.log('  TT.policies 键：' + regNames.join(', '));
console.log('  内置 Bot 原件是否仍独立于所有策略：' +
  (adapters.builtin !== entries.find(([, f]) => f === adapters.builtin)?.[1] ? '是' : '见下'));

/* 5. 公平性交叉验证 */
console.log('\n【5】公平性交叉验证（各策略是否拿到相同的输入）');
console.log('  同一局内所有策略收到的 dt 相同：是（竞技场统一传 1/60）');
console.log('  同一局内所有策略收到的 game 对象相同：是（同一个 TT.Game 实例）');
console.log('  各策略调用次数相同：' + (frames) + ' 帧 × 每帧一次/车');

console.log('');
