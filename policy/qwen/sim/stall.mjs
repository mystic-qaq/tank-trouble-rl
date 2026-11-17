/* stall.mjs — 找出「打完一局要多久」以及长回合里我在干什么 */
import path from 'node:path';
import fs from 'node:fs';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';
import { loadTT } from './harness.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const POLICY = path.resolve(HERE, '..', 'policy.js');
const N = parseInt(process.argv[2] || '8', 10);
const SEAT = 0;

const buckets = { '<5s': 0, '5-15s': 0, '15-30s': 0, '30-60s': 0, '>60s': 0 };
const longRounds = [];
let totalRounds = 0;

for (let m = 0; m < N; m++) {
  const { ctx, TT, builtinBot, canvas } = loadTT({ seed: 9000 + m * 577 });
  vm.runInContext(fs.readFileSync(POLICY, 'utf8'), ctx, { filename: POLICY });
  const QWEN = TT.policies.qwen;
  const brainsMap = QWEN._internals.brains;

  const game = new TT.Game(canvas);
  game.render = () => {}; game.resize = () => {};
  TT.updateBot = (t, g, d) => (t.index === SEAT ? QWEN.update(t, g, d) : builtinBot(t, g, d));

  const stateMix = {};
  let roundNo = 0, roundTime = 0;
  const origStart = game.startRound.bind(game);
  game.startRound = function () { origStart(); roundNo++; roundTime = 0; for (const k in stateMix) delete stateMix[k]; };

  const origEnd = game.endRound.bind(game);
  game.endRound = function (winner) {
    if (game.state === 'playing') {
      totalRounds++;
      const b = roundTime < 5 ? '<5s' : roundTime < 15 ? '5-15s' : roundTime < 30 ? '15-30s' : roundTime < 60 ? '30-60s' : '>60s';
      buckets[b]++;
      if (roundTime >= 30) {
        longRounds.push({
          round: roundNo, dur: roundTime.toFixed(1), winner,
          alive: game.tanks.filter((t) => t.alive).length,
          myState: JSON.stringify(stateMix),
          shots: QWEN.stats.calls,
        });
      }
    }
    return origEnd(winner);
  };

  /* 记录每回合 qwen 的状态占比 */
  game.startMatch(new Array(8).fill('bot'), 4);
  const dt = 1 / 60;
  let tick = 0;
  while (game.state !== 'matchover' && tick < 60 * 300) {
    if (game.state === 'playing') roundTime += dt;
    game.update(dt);
    tick++;
    if (game.state === 'playing') {
      const me = game.tanks.find((t) => t.index === SEAT);
      if (me && me.alive) {
        const b = brainsMap.get(me);
        const st = (b && b.dbg.st) || '?';
        stateMix[st] = (stateMix[st] || 0) + 1;
      }
    }
  }
}

console.log('回合时长分布:', JSON.stringify(buckets), '总回合', totalRounds);
console.log('\n≥30s 的长回合（前 12 条）:');
console.table(longRounds.slice(0, 12));
