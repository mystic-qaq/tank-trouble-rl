/* ===========================================================
   timeout-probe.mjs — 检查超时局如果给更长时间会不会分出胜负
   -----------------------------------------------------------
   正式游戏与竞技场统一为 60 秒；本诊断脚本可显式延长以研究僵局。
   这里用 300 秒重跑那几局，看它们是「很快能分出胜负」还是「彻底僵死」。
   =========================================================== */

import vm from 'node:vm';
import { createArena, mulberry32, roundSeed, seatPlan, mapPlan, STRATEGY_IDS } from './arena.mjs';

const rounds = process.argv.slice(2).map(Number);
const LIMIT = 300;

for (const r of rounds) {
  const env = await createArena();
  const { TT, ctx, sandbox, adapters } = env;
  sandbox.__rnd = mulberry32(roundSeed(20260101, r));
  vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

  const seats = seatPlan(r);
  const mapSeed = mapPlan(r);
  TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
  TT.CFG.ROUND_INTRO = 0;
  TT.CFG.ROUND_TIME_LIMIT = LIMIT;

  const game = new TT.Game(env.canvas);
  game.render = function () {}; game.resize = function () {};
  game.mode = 'match';
  game.slots = ['bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'];
  game.roundsToWin = 9999;

  let finished = false, winnerIndex = -1, frames = 0;
  const origEnd = game.endRound.bind(game);
  game.endRound = function (w) { finished = true; winnerIndex = w; return origEnd(w); };

  TT.updateBot = function (tank, g, dt) {
    const fn = adapters[seats[tank.index]];
    const out = fn(tank, g, dt);
    if (!out) return { drive: 0, steer: 0, shoot: false };
    return {
      drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
      steer: Number(out.steer) || 0, shoot: !!out.shoot,
    };
  };

  game.startRound();
  while (!finished && frames < 60 * LIMIT) { game.update(1 / 60); frames++; }

  const alive = game.tanks.filter((t) => t.alive).map((t) => seats[t.index]);
  console.log(
    '局' + String(r + 1).padStart(4) +
    '  ' + (finished ? '分出胜负' : '仍然僵死') +
    '  用时 ' + (frames / 60).toFixed(0) + 's' +
    '  胜者=' + (winnerIndex >= 0 ? seats[winnerIndex] : '无') +
    '  存活=' + (alive.join(',') || '无')
  );
}
