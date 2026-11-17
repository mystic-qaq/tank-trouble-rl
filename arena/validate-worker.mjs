/* ===========================================================
   validate-worker.mjs — 集成校验的工作线程
   -----------------------------------------------------------
   每个策略单独上场（座位 0），其余 7 个位置交给内置 Bot。
   用法场景与各策略 README 的「1 席 vs 7 内置 Bot」自测一致。
   =========================================================== */

import { parentPort, workerData } from 'node:worker_threads';
import vm from 'node:vm';
import { createArena, mulberry32, roundSeed, mapPlan } from './arena.mjs';

const { baseSeed, slice } = workerData;

async function playSolo(subject, roundIndex) {
  const env = await createArena();
  const { TT, ctx, sandbox, adapters } = env;

  sandbox.__rnd = mulberry32(roundSeed(baseSeed, roundIndex));
  vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

  /* 座位 0 = 被测策略，其余 7 位 = 内置 Bot */
  const seats = new Array(8).fill('builtin');
  seats[0] = subject;

  const mapSeed = mapPlan(roundIndex);
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
    const fn = id === 'builtin' ? adapters.builtin : adapters[subject];
    try {
      const out = fn(tank, g, dt);
      if (!out) return { drive: 0, steer: 0, shoot: false };
      return {
        drive: Math.max(-1, Math.min(1, Number(out.drive) || 0)),
        steer: Number(out.steer) || 0,
        shoot: !!out.shoot,
      };
    } catch (e) {
      return { drive: 0, steer: 0, shoot: false };
    }
  };

  game.startRound();
  while (!finished && frames < 60 * 60 + 2) { game.update(1 / 60); frames++; }

  return {
    subject,
    round: roundIndex,
    winner: winnerIndex >= 0 ? seats[winnerIndex] : null,
    durationSec: game.roundTime,
    timeout: game.roundTimedOut || !finished,
  };
}

parentPort.on('message', async (msg) => {
  if (msg.type !== 'run') return;
  const out = [];
  for (const t of slice) {
    try {
      out.push(await playSolo(t.subject, t.round));
    } catch (e) {
      out.push({ subject: t.subject, round: t.round, winner: null, durationSec: 0, timeout: true, error: String(e && e.message || e) });
    }
  }
  parentPort.postMessage({ type: 'done', results: out });
  parentPort.close();
});
