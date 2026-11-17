/* ===========================================================
   worker.mjs — 竞技场并行工作线程
   -----------------------------------------------------------
   每个 worker 独立跑一批局，结果回传给主线程。
   每一局都用全新沙箱 + 独立随机种子，所以并行与串行结果完全一致。
   =========================================================== */

import { parentPort, workerData } from 'node:worker_threads';
import { playRound } from './arena.mjs';

const { baseSeed, from, to, players } = workerData;

async function run() {
  const out = [];
  for (let r = from; r < to; r++) {
    try {
      out.push(await playRound(baseSeed, r, false, players));
    } catch (e) {
      out.push({
        round: r, error: String((e && e.message) || e),
        winner: null, timeout: false, durationSec: 0,
        seats: [], mapSeed: null,
        kills: {}, suicides: {}, deaths: {}, errors: {}, aliveAtEnd: [], survivors: 0,
      });
    }
    parentPort.postMessage({ type: 'progress', done: 1, total: to - from });
  }
  parentPort.postMessage({ type: 'done', results: out });
  /* 主动断开端口，否则 message 监听器会让线程一直活着 */
  parentPort.close();
}

parentPort.on('message', (msg) => {
  if (msg.type === 'run') run();
});
