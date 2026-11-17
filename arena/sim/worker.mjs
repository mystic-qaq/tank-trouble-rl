/* sim/worker.mjs — 执行单局的工作线程（一局一次沙箱，互相隔离） */
import { parentPort } from 'node:worker_threads';
import { playRound } from '../arena.mjs';

parentPort.on('message', async (msg) => {
  if (!msg) return;
  if (msg.type === 'stop') { process.exit(0); return; }
  if (msg.type !== 'round') return;
  const { job, index } = msg;
  let res;
  try {
    res = await playRound(job.seed, index, false, job.players);
  } catch (e) {
    res = {
      round: index, winner: null, timeout: false, durationSec: 0, seats: job.players.slice(),
      mapSeed: null, spawnSeed: null, spawns: [], kills: {}, suicides: {}, deaths: {},
      errors: { engine: String((e && e.stack) || e) }, aliveAtEnd: [], survivors: 0,
      eliminationOrder: [], deathTime: {}, deathBy: {}, deathBounces: {}, finalists: [],
      error: String((e && e.message) || e),
    };
  }
  res.jobId = job.id;
  res.tag = job.tag;
  res.n = job.players.length;
  parentPort.postMessage({ type: 'result', key: job.id + '#' + index, res });
});
