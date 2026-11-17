/* Verifies every way an Arena might load / call the policy.
   Run: node policy/hy/test/interface.js                       */
'use strict';

const assert = require('assert');
const path = require('path');
const { pathToFileURL } = require('url');

function checkAction(a, label) {
  assert(a && typeof a === 'object', label + ': not an object');
  for (const k of ['drive', 'steer', 'shoot', 'throttle', 'turn', 'fire', 'left', 'right']) {
    assert(k in a, label + ': missing key ' + k);
  }
  assert(Number.isFinite(a.drive) && Number.isFinite(a.steer), label + ': non-finite drive/steer');
  assert(a.drive >= -1.001 && a.drive <= 1.001, label + ': drive out of range');
  assert(a.steer >= -1.001 && a.steer <= 1.001, label + ': steer out of range');
  assert(typeof a.shoot === 'boolean', label + ': shoot not boolean');
}

/* ---- 1. CommonJS ---- */
const p = require('../policy.js');
assert.strictEqual(typeof p, 'function', 'CommonJS default export is not a function');
assert.strictEqual(typeof p.decide, 'function');
assert.strictEqual(typeof p.updateBot, 'function');
console.log('[ok] CommonJS require("../policy.js")');

const index = require('../index.js');
assert.strictEqual(typeof index, 'function');
console.log('[ok] CommonJS require("../index.js")');

/* ---- 2. Global registration ---- */
assert.strictEqual(globalThis.HYPolicy, p, 'global HYPolicy not registered');
assert(globalThis.TT && globalThis.TT.Policies && globalThis.TT.Policies.hy === p,
  'TT.Policies.hy not registered');
assert(globalThis.Policies && globalThis.Policies.hy === p, 'Policies.hy not registered');
console.log('[ok] global / TT.Policies / Policies registration');

/* ---- 3. Engine-native call shape ---- */
const walls = [
  { x: 0, y: 0, w: 1280, h: 10 }, { x: 0, y: 790, w: 1280, h: 10 },
  { x: 0, y: 0, w: 10, h: 800 }, { x: 1270, y: 0, w: 10, h: 800 },
  { x: 400, y: 200, w: 10, h: 300 }
];
const tanks = [
  { index: 0, x: 200, y: 200, angle: 0, speed: 80, cooldown: 0, alive: true },
  { index: 1, x: 700, y: 400, angle: Math.PI, speed: 0, cooldown: 0, alive: true }
];
const bullets = [
  { x: 500, y: 300, px: 490, py: 300, angle: 0.2, owner: 1, bounces: 1, life: 4, grace: 0 }
];
const game = { tanks, bullets, map: { walls }, time: 10, dt: 1 / 60, roundSerial: 1 };

checkAction(p(tanks[0], game, 1 / 60), 'tank,game,dt');
checkAction(p(game, tanks[0], 1 / 60), 'game,tank,dt');
console.log('[ok] (tank, game, dt) and (game, tank, dt)');

/* ---- 4. Snapshot call shapes ---- */
const snap = {
  self: tanks[0], tanks, bullets, walls,
  time: 10, tick: 600, dt: 1 / 60
};
checkAction(p(snap), 'snapshot');
checkAction(p(snap, 1 / 60), 'snapshot,dt');
const snap2 = Object.assign({}, snap, { self: undefined, me: tanks[0] });
checkAction(p(snap2), 'snapshot me=');
console.log('[ok] snapshot shapes (self / me / +dt)');

/* ---- 5. No wall data -> deterministic embedded maps ---- */
const t0 = Date.now();
const snap3 = { self: tanks[0], tanks, bullets, time: 11, tick: 660, dt: 1 / 60 };
checkAction(p(snap3), 'snapshot without walls');
console.log('[ok] wall-less fallback (embedded map build ' + (Date.now() - t0) + 'ms)');

/* ---- 6. Weird / degenerate input must never throw ---- */
checkAction(p({ self: { x: 640, y: 400, angle: 0 }, tanks: [], bullets: [] }), 'empty world');
checkAction(p(tanks[0]), 'tank only');
checkAction(p({}), 'empty object');
console.log('[ok] degenerate inputs handled');

/* ---- 7. ESM ---- */
(async function () {
  const mod = await import(pathToFileURL(path.resolve(__dirname, '../policy.mjs')).href);
  assert.strictEqual(typeof mod.default, 'function', 'ESM default is not a function');
  assert.strictEqual(typeof mod.decide, 'function');
  checkAction(mod.default(tanks[0], game, 1 / 60), 'esm default');
  console.log('[ok] ESM import("../policy.mjs")');

  const mod2 = await import(pathToFileURL(path.resolve(__dirname, '../index.mjs')).href);
  assert.strictEqual(typeof mod2.default, 'function');
  console.log('[ok] ESM import("../index.mjs")');

  console.log('\nALL INTERFACE CHECKS PASSED');
})().catch(function (e) {
  console.error(e);
  process.exit(1);
});
