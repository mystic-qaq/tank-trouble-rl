/* ===========================================================
   bisect-minimax.mjs — 定位 minimax 在竞技场里表现差异的原因
   -----------------------------------------------------------
   同一个场景（minimax 单独上场 vs 7 个内置 Bot），换不同的加载方式：
     A. vm.SourceTextModule（竞技场当前做法）
     B. Node 原生 import 后把函数注入沙箱
   对照 minimax 自带测试台的成绩，看差异出在哪。
   =========================================================== */

import vm from 'node:vm';
import { createArena, mulberry32, roundSeed, mapPlan } from '../arena/arena.mjs';

const ROUNDS = parseInt(process.argv[2] || '30', 10);
const SEED = 20260101;

async function soloVia(subject, rounds, loader) {
  const env = await createArena();
  const { TT, ctx, sandbox, adapters } = env;
  const policy = await loader(adapters, ctx);

  let wins = 0, lifeSum = 0;
  for (let r = 0; r < rounds; r++) {
    sandbox.__rnd = mulberry32(roundSeed(SEED, r));
    vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

    const seats = new Array(8).fill('builtin');
    seats[0] = subject;

    const mapSeed = mapPlan(r);
    TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
    TT.CFG.ROUND_INTRO = 0;

    const game = new TT.Game(env.canvas);
    game.render = function () {};
    game.resize = function () {};
    game.mode = 'match';
    game.slots = ['bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'];
    game.roundsToWin = 9999;

    let finished = false, winnerIndex = -1, frames = 0;
    const origEnd = game.endRound.bind(game);
    game.endRound = function (w) { finished = true; winnerIndex = w; return origEnd(w); };

    TT.updateBot = function (tank, g, dt) {
      const id = seats[tank.index];
      const fn = id === 'builtin' ? adapters.builtin : policy;
      const out = fn(tank, g, dt);
      if (!out) return { drive: 0, steer: 0, shoot: false };
      return {
        drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
        steer: Number(out.steer) || 0,
        shoot: !!out.shoot,
      };
    };

    game.startRound();
    while (!finished && frames < 60 * 90) { game.update(1 / 60); frames++; }
    if (winnerIndex === 0) wins++;
    lifeSum += frames / 60;
  }
  return { wins, rate: wins / rounds, avgLife: lifeSum / rounds };
}

/* A. 竞技场当前做法：vm.SourceTextModule */
const loaderA = async (adapters) => adapters.minimax;

/* B. Node 原生 import，再把函数注入沙箱 */
const loaderB = async (adapters, ctx) => {
  const m = await import('../policy/minimax/policy.js');
  /* 让策略能读到沙箱的 TT.CFG：把 sandbox 挂到 globalThis.TT 上 */
  globalThis.TT = ctx.TT;
  return m.default;
};

console.log(`\n▶ minimax 1 vs 7 内置 Bot，${ROUNDS} 局\n`);

const a = await soloVia('minimax', ROUNDS, loaderA);
console.log(`  A. vm.SourceTextModule（竞技场现状）  胜率 ${(a.rate * 100).toFixed(1)}%  平均存活 ${a.avgLife.toFixed(1)}s`);

const b = await soloVia('minimax', ROUNDS, loaderB);
console.log(`  B. Node 原生 import + 注入             胜率 ${(b.rate * 100).toFixed(1)}%  平均存活 ${b.avgLife.toFixed(1)}s`);

console.log('\n  minimax 自带测试台报告：95.8%（115/120 回合）');
console.log('');
