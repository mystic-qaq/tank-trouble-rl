/* Where does Schema V3 label generation spend its time?
   Usage: node --experimental-vm-modules policy/rl/v8/bench_labels.mjs
*/
import { TankEnvV2 } from '../v2/env.mjs';
import { aimCandidates, flight } from '../v2/geometry.mjs';
import { traceShotV8, servoTarget, selfPose } from './geometry_v8.mjs';
import { createTracker, snapshotTracker, recordAction, observe, labels } from './observe_v3.mjs';

const e = new TankEnvV2({ objective: 'frag' });
e.skipFeatures = true;
await e.reset({ seed: 5551, n: 8, seat: 0, mapSeed: 20260101,
                players: ['hy', 'gpt', 'glm', 'qwen', 'kimi', 'builtin', 'minimax', 'deepseek'] });
const tr = createTracker(e.game, e.seat);
const frames = [];
let guard = 0;
while (!e.done && guard++ < 200) {
  const me = e.game.tanks[e.seat];
  if (me.alive) {
    frames.push({ input: observe(tr), label: labels(tr, { drive: 1, steer: 0, shoot: false }) });
    snapshotTracker(tr);
  }
  const acts = e.game.tanks.map(t => t.alive ? e.policyAction(e.players[t.index], t) : { drive: 0, steer: 0, shoot: false });
  e.step(acts[0]);
  recordAction(tr, acts[e.seat]);
}
console.log('frames captured', frames.length);

const N = 200;
const t = (label, fn) => {
  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) fn();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / N;
  console.log(label.padEnd(34) + '  ' + ms.toFixed(3) + ' ms/frame');
  return ms;
};

const game = e.game, me = game.tanks[e.seat];

const tObserve = t('observe()', () => observe(tr));
const tLabels = t('labels()  (full)', () => labels(tr, { drive: 1, steer: 0, shoot: false }));
const tCands = t('aimCandidates()', () => aimCandidates(game, me));
const cands = aimCandidates(game, me);
const tServo = t('servoTarget()', () => servoTarget(game, me, cands));
const tTrace = t('traceShotV8() x1', () => traceShotV8(game, me, me.angle + 0.3));
const tFlight = t('flight() x1', () => flight(game, me.x, me.y, me.angle, 620, 6));
const tPose = t('selfPose() x1', () => selfPose(me, me.angle + 0.3, 0.5, 3.3));

console.log('\n--- 推断 ---');
console.log('64 x traceShotV8  = ' + (64 * tTrace).toFixed(2) + ' ms');
console.log('labels 其余部分    = ' + Math.max(0, tLabels - 64 * tTrace - tCands - tServo).toFixed(2) + ' ms');
console.log('observe + labels   = ' + (tObserve + tLabels).toFixed(2) + ' ms -> 单 worker 上限 ' +
            Math.round(1000 / (tObserve + tLabels)) + ' 行/秒');
