/* ===========================================================
   bench.mjs — 吞吐标定：一局到底花在哪里，单核能跑多快
   -----------------------------------------------------------
   用法：
     node --experimental-vm-modules arena/bench.mjs
     node --experimental-vm-modules arena/bench.mjs --players gpt,hy --rounds 20

   产出（用于估算总规模与并行度）：
     · createArena 单次成本（建沙箱 + 编译 8 份策略）
     · 单局成本、单局模拟时长、实时倍率（sim秒 / 墙钟秒）
     · 拆出「每帧逻辑」与「每局建场」的占比
   =========================================================== */

import { createArena, playRound, STRATEGY_IDS } from './arena.mjs';

const arg = (n, d) => {
  const i = process.argv.indexOf('--' + n);
  return i >= 0 && process.argv[i + 1] != null ? process.argv[i + 1] : d;
};

const PLAYERS = (arg('players', STRATEGY_IDS.join(','))).split(',').map(s => s.trim());
const N = parseInt(arg('rounds', '12'), 10);

const now = () => Number(process.hrtime.bigint()) / 1e6;

/* ---- 1. 建场成本 ---- */
const arenaTimes = [];
for (let i = 0; i < 3; i++) {
  const t = now();
  await createArena();
  arenaTimes.push(now() - t);
}
const arenaAvg = arenaTimes.reduce((a, b) => a + b, 0) / arenaTimes.length;

/* ---- 2. 整局成本 ---- */
const rows = [];
for (let r = 0; r < N; r++) {
  const t = now();
  const res = await playRound(20260101, r, false, PLAYERS);
  const wall = now() - t;
  rows.push({ wall, sim: res.durationSec, winner: res.winner, timeout: res.timeout });
}

const sum = (k) => rows.reduce((a, b) => a + b[k], 0);
const wallAvg = sum('wall') / N;
const simAvg = sum('sim') / N;
const frameAvg = (wallAvg - arenaAvg) / (simAvg * 60) * 1000;   // ms / 帧（估算）

console.log(`\n参赛 ${PLAYERS.length} 车：${PLAYERS.join(', ')}`);
console.log(`createArena（建沙箱+编译 8 份策略）: ${arenaAvg.toFixed(1)} ms / 次`);
console.log(`整局墙钟                          : ${wallAvg.toFixed(1)} ms / 局（模拟时长 ${simAvg.toFixed(1)} s）`);
console.log(`建场占比                          : ${(arenaAvg / wallAvg * 100).toFixed(0)}%`);
console.log(`逻辑帧成本（扣掉建场，估算）      : ${frameAvg.toFixed(3)} ms / 帧 = ${(1000 / frameAvg).toFixed(0)} 帧/秒/核`);
console.log(`实时倍率                          : ${(simAvg / (wallAvg / 1000)).toFixed(0)}× 实时`);
console.log(`单核吞吐                          : ${(1000 / wallAvg * 60).toFixed(0)} 局/分钟（按 60s/局 归一）`);
console.log(`单核实际                          : ${(1000 / wallAvg).toFixed(0)} 局/分钟\n`);
