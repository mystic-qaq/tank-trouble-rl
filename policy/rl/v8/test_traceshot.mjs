/* Quantify the traceShot self-hit defect before it is baked into V8 labels.
   Runs real teacher play, and on every frame compares v2 traceShot against
   traceShotV8 over the engine's own aim candidates.

   The dangerous direction is "v2 says safe, v8 says self-hit": that is a firing
   solution the V4-V7 label pipeline told the policy was free, and which then
   rebounded into it. V7 measured self/shot 0.055 against teacher hy 0.016.

   Usage: node --experimental-vm-modules policy/rl/v8/test_traceshot.mjs [frames]
*/
import assert from 'node:assert/strict';
import { TankEnvV2 } from '../v2/env.mjs';
import { aimCandidates, traceShot } from '../v2/geometry.mjs';
import { traceShotV8, selfPose, aimTruthBasis } from './geometry_v8.mjs';

const FRAMES = Number(process.argv[2] || 4000);
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

/* --- selfPose unit checks: the arc must degenerate to a straight line --- */
{
  const me = { x: 100, y: 100, angle: 0, speed: 178, radius: 16, alive: true };
  const straight = selfPose(me, 0, 1.0, 3.3);
  assert.ok(Math.abs(straight.x - (100 + 178)) < 1e-9, 'aligned target must give straight-line motion');
  assert.ok(Math.abs(straight.y - 100) < 1e-9);
  // quarter turn at max rate takes (pi/2)/3.3 s; after that the tank faces the target
  const q = selfPose(me, Math.PI / 2, (Math.PI / 2) / 3.3, 3.3);
  assert.ok(Math.abs(q.angle - Math.PI / 2) < 1e-9, 'must be aligned exactly at tAlign');
  // arc length must equal speed * time
  const chord = Math.hypot(q.x - 100, q.y - 100);
  const arc = 178 * ((Math.PI / 2) / 3.3);
  assert.ok(chord < arc && chord > arc * 0.6, 'chord must be shorter than arc but not degenerate');
  // long after alignment it must be straight along the target angle
  const far = selfPose(me, Math.PI / 2, 10, 3.3);
  assert.ok(Math.abs(far.x - q.x) < 1e-6, 'x must stop changing once heading is +y');
  console.log('selfPose: straight-line degeneration, alignment time, arc-vs-chord, post-align hold  OK');
}

const tally = {
  frames: 0, candidates: 0,
  v2_safe: 0, v8_safe: 0,
  false_safe: 0,      // v2 safe, v8 self-hit  -> the dangerous direction
  false_alarm: 0,     // v2 self-hit, v8 safe
  hit_disagree: 0,
  has_safe_v2: 0, has_safe_v8: 0, has_safe_flip: 0,
  err_of_false_safe: [],
};

for (const teacher of ['hy', 'gpt']) {
  for (const n of [2, 4, 8]) {
    const e = new TankEnvV2({ objective: 'frag' });
    await e.reset({ seed: 91237 + n * 31, n, seat: 0, players: Array(n).fill(teacher) });
    const game = e.game;
    game.__cfg = game.__cfg || globalThis.TT.CFG;
    game.__util = game.__util || globalThis.TT.util;
    let frames = 0;
    while (!e.done && frames < FRAMES / 6) {
      e.step(null, teacher);
      frames++;
      const me = game.tanks[e.seat];
      if (!me.alive) continue;
      const cands = aimCandidates(game, me);
      tally.frames++;
      let hs2 = false, hs8 = false;
      for (const c of cands) {
        tally.candidates++;
        const v2 = traceShot(game, me, c.angle);
        const v8 = traceShotV8(game, me, c.angle);
        const safe2 = c !== null && v2.hit && !v2.self && !v2.blocked;
        const safe8 = v8.hit && !v8.self && !v8.blocked;
        if (safe2) { tally.v2_safe++; hs2 = true; }
        if (safe8) { tally.v8_safe++; hs8 = true; }
        if (safe2 && v8.self) {
          tally.false_safe++;
          tally.err_of_false_safe.push(Math.abs(wrap(c.angle - me.angle)) * 180 / Math.PI);
        }
        if (!safe2 && safe8) tally.false_alarm++;
        if (!!v2.hit !== !!v8.hit) tally.hit_disagree++;
      }
      if (hs2) tally.has_safe_v2++;
      if (hs8) tally.has_safe_v8++;
      if (hs2 !== hs8) tally.has_safe_flip++;
    }
  }
}

const pct = (a, b) => b ? (100 * a / b).toFixed(2) + '%' : 'n/a';
const errs = tally.err_of_false_safe.sort((a, b) => a - b);
const q = p => errs.length ? errs[Math.min(errs.length - 1, Math.floor(p * errs.length))].toFixed(1) : 'n/a';

console.log(JSON.stringify({
  frames: tally.frames, candidates: tally.candidates,
  v2_safe: tally.v2_safe, v8_safe: tally.v8_safe,
  safe_rate_v2: pct(tally.v2_safe, tally.candidates),
  safe_rate_v8: pct(tally.v8_safe, tally.candidates),
  false_safe: tally.false_safe,
  false_safe_share_of_v2_safe: pct(tally.false_safe, tally.v2_safe),
  false_alarm: tally.false_alarm,
  hit_verdict_disagreements: tally.hit_disagree,
  frames_with_any_safe_v2: tally.has_safe_v2,
  frames_with_any_safe_v8: tally.has_safe_v8,
  frames_where_has_safe_flips: tally.has_safe_flip,
  has_safe_flip_rate: pct(tally.has_safe_flip, tally.frames),
  false_safe_aim_error_deg: { p10: q(0.1), p50: q(0.5), p90: q(0.9) },
}, null, 2));
