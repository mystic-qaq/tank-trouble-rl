/* Prove the allocation-free raycast and flight are bit-identical to the engine ones.

   The optimised path exists only for throughput; if it disagreed with js/map.js even on a
   handful of rays the labels would be silently wrong, so this compares exhaustively over
   many random rays and shots before any collection is trusted.

   Usage: node --experimental-vm-modules policy/rl/v8/test_fastgeom.mjs
*/
import assert from 'node:assert/strict';
import { TankEnvV2 } from '../v2/env.mjs';
import { flight } from '../v2/geometry.mjs';
import { raycastInto, flightV8, RC_T, RC_NX, RC_NY } from './geometry_v8.mjs';

const MAP_SEEDS = [20260101, 88888888, 12345, 99887766];
let rays = 0, hits = 0, shots = 0, segs = 0;

for (const mapSeed of MAP_SEEDS) {
  const e = new TankEnvV2({ objective: 'frag' });
  e.skipFeatures = true;
  await e.reset({ seed: 777 + mapSeed % 1000, n: 8, seat: 0, mapSeed,
                  players: ['hy', 'gpt', 'glm', 'qwen', 'kimi', 'builtin', 'minimax', 'deepseek'] });
  const game = e.game;

  /* --- raycast: random origins, directions, radii --- */
  let seed = 0x2545F491;
  const rnd = () => {
    seed ^= seed << 13; seed >>>= 0; seed ^= seed >> 17; seed ^= seed << 5; seed >>>= 0;
    return seed / 0x100000000;
  };
  for (let i = 0; i < 40000; i++) {
    const x0 = rnd() * 1280, y0 = rnd() * 800;
    const a = rnd() * Math.PI * 2, len = 40 + rnd() * 1800;
    const radius = [0, 3.2, 16][i % 3];
    const ref = game.map.raycast(x0, y0, x0 + Math.cos(a) * len, y0 + Math.sin(a) * len, radius);
    const got = raycastInto(game.map, x0, y0, x0 + Math.cos(a) * len, y0 + Math.sin(a) * len, radius);
    assert.equal(got, !!ref, 'hit/miss mismatch at ray ' + i);
    if (ref) {
      hits++;
      assert.ok(RC_T === ref.t, 't mismatch %s vs %s', RC_T, ref.t);
      assert.ok(RC_NX === (ref.nx || 0), 'nx mismatch');
      assert.ok(RC_NY === (ref.ny || 0), 'ny mismatch');
    }
    rays++;
  }
  /* axis-aligned degenerate cases (dx or dy == 0) */
  for (let i = 0; i < 4000; i++) {
    const x0 = rnd() * 1280, y0 = rnd() * 800, len = 40 + rnd() * 1800;
    for (const [ex, ey] of [[x0 + len, y0], [x0, y0 + len], [x0 - len, y0], [x0, y0 - len]]) {
      const ref = game.map.raycast(x0, y0, ex, ey, 0);
      const got = raycastInto(game.map, x0, y0, ex, ey, 0);
      assert.equal(got, !!ref, 'axis-aligned hit/miss mismatch');
      if (ref) { assert.ok(RC_T === ref.t && RC_NX === (ref.nx || 0) && RC_NY === (ref.ny || 0)); hits++; }
      rays++;
    }
  }

  /* --- flight: random shots from random poses, all bounce budgets --- */
  for (let i = 0; i < 3000; i++) {
    const me = game.tanks[i % game.tanks.length];
    const a = rnd() * Math.PI * 2;
    const from = [me.x, me.y, a];
    for (const bounces of [0, 2, 5]) {
      const ref = flight(game, from[0], from[1], from[2], 620, 6, bounces);
      const got = flightV8(game, from[0], from[1], from[2], 620, 6, bounces);
      assert.equal(got.length, ref.length, 'segment count mismatch');
      for (let k = 0; k < ref.length; k++) {
        for (const key of ['x', 'y', 'vx', 'vy', 'time', 'duration']) {
          assert.ok(got[k][key] === ref[k][key],
                    'flight ' + key + ' mismatch at segment ' + k + ': ' + got[k][key] + ' vs ' + ref[k][key]);
        }
      }
      segs += ref.length;
      shots++;
    }
  }
  console.log('mapSeed ' + mapSeed + ': walls ' + game.map.walls.length);
}

console.log(JSON.stringify({
  raycast_cases: rays, raycast_hits: hits,
  flight_cases: shots, flight_segments: segs,
  exact_match: true,
}));
