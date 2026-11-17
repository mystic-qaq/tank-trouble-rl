/* head2head.mjs 的工作线程：跑一批单挑对局 */
import { parentPort, workerData } from 'node:worker_threads';
import vm from 'node:vm';
import { createArena, mulberry32, roundSeed } from './arena.mjs';

const { seed, slice } = workerData;

async function duel(a, b, roundIndex) {
  const env = await createArena();
  const { TT, ctx, sandbox, adapters } = env;

  sandbox.__rnd = mulberry32(roundSeed(seed, roundIndex));
  vm.runInContext('Math.random = function () { return __rnd(); };', ctx);

  const seats = [a, b];
  const mapSeed = (roundIndex % 2 === 0) ? 20260101 : 88888888;
  TT.CFG.MAP_SEEDS = [mapSeed, mapSeed];
  TT.CFG.ROUND_INTRO = 0;

  const game = new TT.Game(env.canvas);
  game.render = function () {}; game.resize = function () {};
  game.mode = 'match';
  game.slots = ['bot', 'bot', 'off', 'off', 'off', 'off', 'off', 'off'];
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
      steer: Number(out.steer) || 0,
      shoot: !!out.shoot,
    };
  };

  game.startRound();
  while (!finished && frames < 60 * 60 + 2) { game.update(1 / 60); frames++; }

  return {
    a, b,
    winner: winnerIndex >= 0 ? seats[winnerIndex] : null,
    timeout: game.roundTimedOut || !finished,
    durationSec: game.roundTime,
  };
}

parentPort.on('message', async (msg) => {
  if (msg.type !== 'run') return;
  const out = [];
  for (const t of slice) {
    try {
      out.push(await duel(t.a, t.b, t.r));
    } catch (e) {
      out.push({ a: t.a, b: t.b, winner: null, timeout: true, durationSec: 0, error: String(e && e.message || e) });
    }
  }
  parentPort.postMessage({ type: 'done', results: out });
  parentPort.close();
});
