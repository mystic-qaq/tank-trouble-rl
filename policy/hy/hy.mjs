/* =====================================================================
   HY POLICY  ·  Tank Trouble Arena
   ---------------------------------------------------------------------
   Self-contained. No dependency on the game source or on any other policy.

   Design priorities (the round is won by the LAST SURVIVOR, kills only
   affect the scoreboard):
     1. Never kill ourselves. Every candidate shot is traced through the
        exact bullet physics and vetoed if it comes back at us.
     2. Never get hit. A model-predictive controller rolls forward the
        position of every live bullet (including bounces) and picks the
        manoeuvre with the lowest collision risk.
     3. Only take shots with a high hit probability: exact same-frame
        aiming plus simulated bank shots. No blind / suppressive spam.
     4. Camp patiently. In an 8-tank free-for-all the opponents kill each
        other; staying alive is worth more than chasing. Aggression ramps
        up slowly to break stalemates.

   Accepted call signatures:
     decide(tank, game, dt)      engine-native bot hook
     decide(game, tank, dt)      swapped
     decide(state)               snapshot {self/me, tanks, bullets, walls}
     decide(state, dt)

   Returned action carries every common key:
     {drive, steer, shoot} plus throttle/move/turn/rotate/fire and
     boolean up/down/left/right/forward/backward.
   ===================================================================== */
var HY_POLICY_API = (function () {
  'use strict';

  var root = (typeof globalThis !== 'undefined') ? globalThis
    : (typeof window !== 'undefined') ? window
      : (typeof self !== 'undefined') ? self : this;

  /* Every generated entry point reuses the very same API instance. */
  if (root && root.__HY_API__) return root.__HY_API__;

  /* ------------------------------------------------------------------
     Engine constants (mirrors js/util.js)
     ------------------------------------------------------------------ */
  var C = {
    W: 1280, H: 800, CELL: 80, COLS: 16, ROWS: 10, WT: 10,
    TR: 16, TS: 178, TREV: 0.66, TACC: 920, TBRK: 1150,
    TTURN: 3.3, TCD: 0.42,
    BS: 620, BR: 3.2, BBOUNCE: 5, BLIFE: 6, BGRACE: 0.07, BMAX: 5,
    MUZZLE: 28, HITR: 19.2
  };
  var TAU = Math.PI * 2;
  var STEP = 1 / 60;
  var K = 54;                 /* 0.9 s prediction horizon */
  var BMAX = 64, EMAX = 10;
  var SAFE_R = 36;            /* hit radius + safety margin (covers sampling error) */
  var SAFE_R2 = SAFE_R * SAFE_R;

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function sgn(v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }
  function dist2(ax, ay, bx, by) { var dx = bx - ax, dy = by - ay; return dx * dx + dy * dy; }
  function hypot(x, y) { return Math.sqrt(x * x + y * y); }
  function angleDiff(a, b) { var d = (b - a + Math.PI) % TAU; if (d < 0) d += TAU; return d - Math.PI; }
  function num(a, b, c, d) {
    if (typeof a === 'number' && isFinite(a)) return a;
    if (typeof b === 'number' && isFinite(b)) return b;
    if (typeof c === 'number' && isFinite(c)) return c;
    if (typeof d === 'number' && isFinite(d)) return d;
    return 0;
  }
  function sameId(a, b) { return a === b || String(a) === String(b); }
  function nowMs() {
    if (typeof performance !== 'undefined' && performance.now) return performance.now();
    return Date.now();
  }

  /* ------------------------------------------------------------------
     Deterministic map generation (identical to js/map.js), used only as
     a fallback when the caller does not expose walls.
     ------------------------------------------------------------------ */
  var VC = C.COLS + 1, HC = C.ROWS + 1;
  var TOTAL = C.ROWS * C.COLS;

  function mulberry(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeFlags(seed) {
    var v = new Uint8Array(C.ROWS * VC), h = new Uint8Array(HC * C.COLS);
    var bestV = null, bestH = null, best = -1;

    function vAt(r, c) { return v[r * VC + c]; }
    function hAt(r, c) { return h[r * C.COLS + c]; }

    function reachRatio() {
      var seen = new Uint8Array(TOTAL);
      var stack = [1 * C.COLS + 1];
      seen[stack[0]] = 1;
      var count = 0;
      while (stack.length) {
        var idx = stack.pop(); count++;
        var r = (idx / C.COLS) | 0, c = idx % C.COLS;
        if (c < C.COLS - 1 && !vAt(r, c + 1) && !seen[idx + 1]) { seen[idx + 1] = 1; stack.push(idx + 1); }
        if (c > 0 && !vAt(r, c) && !seen[idx - 1]) { seen[idx - 1] = 1; stack.push(idx - 1); }
        if (r < C.ROWS - 1 && !hAt(r + 1, c) && !seen[idx + C.COLS]) { seen[idx + C.COLS] = 1; stack.push(idx + C.COLS); }
        if (r > 0 && !hAt(r, c) && !seen[idx - C.COLS]) { seen[idx - C.COLS] = 1; stack.push(idx - C.COLS); }
      }
      return count / TOTAL;
    }

    function openSides(r, c) {
      var n = 0;
      if (!hAt(r, c)) n++;
      if (!hAt(r + 1, c)) n++;
      if (!vAt(r, c)) n++;
      if (!vAt(r, c + 1)) n++;
      return n;
    }

    function score() {
      var reach = reachRatio();
      if (reach < 1) return reach * 0.55;
      var walls = 0;
      for (var r = 0; r < C.ROWS; r++) for (var c = 1; c < C.COLS; c++) if (vAt(r, c)) walls++;
      for (var r2 = 1; r2 < C.ROWS; r2++) for (var c2 = 0; c2 < C.COLS; c2++) if (hAt(r2, c2)) walls++;
      var density = walls / ((C.COLS - 1) * C.ROWS + (C.ROWS - 1) * C.COLS);
      var s = 1 - Math.abs(density - 0.40) * 2.4;
      var open = 0;
      for (var r3 = 1; r3 < C.ROWS - 1; r3++) for (var c3 = 1; c3 < C.COLS - 1; c3++)
        if (openSides(r3, c3) >= 3) open++;
      if (open < 30) s -= (30 - open) * 0.012;
      return s < 0 ? 0 : s;
    }

    function buildFlags(rnd) {
      v.fill(0); h.fill(0);
      function divide(r0, c0, r1, c1, depth) {
        var hh = r1 - r0, ww = c1 - c0;
        if (hh < 2 && ww < 2) return;
        if (depth > 18) return;
        var vertical;
        if (ww < 2) vertical = false;
        else if (hh < 2) vertical = true;
        else if (ww > hh * 1.3) vertical = true;
        else if (hh > ww * 1.3) vertical = false;
        else vertical = rnd() < 0.42;

        if (vertical) {
          var cs = c0 + 1 + Math.floor((ww - 1) * (0.3 + rnd() * 0.4));
          for (var r = r0; r < r1; r++) v[r * VC + cs] = 1;
          var gaps = (r1 - r0 >= 5 && rnd() < 0.28) ? 2 : 1;
          for (var g = 0; g < gaps; g++) { var rr = r0 + Math.floor(rnd() * hh); v[rr * VC + cs] = 0; }
          divide(r0, c0, r1, cs, depth + 1);
          divide(r0, cs, r1, c1, depth + 1);
        } else {
          var rs = r0 + 1 + Math.floor((hh - 1) * (0.3 + rnd() * 0.4));
          for (var c = 0 + c0; c < c1; c++) h[rs * C.COLS + c] = 1;
          var gaps2 = (c1 - c0 >= 5 && rnd() < 0.28) ? 2 : 1;
          for (var g2 = 0; g2 < gaps2; g2++) { var cc = c0 + Math.floor(rnd() * ww); h[rs * C.COLS + cc] = 0; }
          divide(r0, c0, rs, c1, depth + 1);
          divide(rs, c0, r1, c1, depth + 1);
        }
      }
      divide(0, 0, C.ROWS, C.COLS, 0);

      var rooms = 6 + Math.floor(rnd() * 4);
      for (var i = 0; i < rooms; i++) {
        var rw = 2 + (rnd() < 0.34 ? 1 : 0), rh = 2 + (rnd() < 0.28 ? 1 : 0);
        var rr0 = 1 + Math.floor(rnd() * (C.ROWS - 1 - rh));
        var cc0 = 1 + Math.floor(rnd() * (C.COLS - 1 - rw));
        for (var r4 = rr0; r4 < rr0 + rh; r4++)
          for (var c4 = cc0 + 1; c4 < cc0 + rw; c4++) v[r4 * VC + c4] = 0;
        for (var r5 = rr0 + 1; r5 < rr0 + rh; r5++)
          for (var c5 = cc0; c5 < cc0 + rw; c5++) h[r5 * C.COLS + c5] = 0;
      }
      for (var r6 = 0; r6 < C.ROWS; r6++)
        for (var c6 = 1; c6 < C.COLS; c6++) if (v[r6 * VC + c6] && rnd() < 0.10) v[r6 * VC + c6] = 0;
      for (var r7 = 1; r7 < C.ROWS; r7++)
        for (var c7 = 0; c7 < C.COLS; c7++) if (h[r7 * C.COLS + c7] && rnd() < 0.10) h[r7 * C.COLS + c7] = 0;

      var stubs = 3 + Math.floor(rnd() * 4);
      for (var i2 = 0; i2 < stubs; i2++) {
        var rs2 = 1 + Math.floor(rnd() * (C.ROWS - 2));
        var cs2 = 1 + Math.floor(rnd() * (C.COLS - 2));
        var isV = rnd() < 0.5;
        var slot = isV ? (rs2 * VC + cs2) : (rs2 * C.COLS + cs2);
        var arr = isV ? v : h;
        if (arr[slot]) continue;
        if (isV && (vAt(rs2 - 1, cs2) || vAt(rs2 + 1, cs2))) continue;
        if (!isV && (hAt(rs2, cs2 - 1) || hAt(rs2, cs2 + 1))) continue;
        arr[slot] = 1;
        if (reachRatio() < 1) arr[slot] = 0;
      }
    }

    /*刻意复制引擎 generate() 的语义：提前 break 时不恢复 best */
    for (var attempt = 0; attempt < 70; attempt++) {
      var rnd = mulberry(seed + attempt * 7919);
      buildFlags(rnd);
      var s = score();
      if (s > best) { best = s; bestV = v.slice(); bestH = h.slice(); }
      if (s >= 0.94) break;
    }
    if (best >= 0 && best < 0.94) { v = bestV; h = bestH; }
    return { v: v, h: h };
  }

  function wallsFromFlags(v, h) {
    var walls = [], W = C.W, H = C.H, T = C.WT;
    walls.push({ x: 0, y: 0, w: W, h: T });
    walls.push({ x: 0, y: H - T, w: W, h: T });
    walls.push({ x: 0, y: 0, w: T, h: H });
    walls.push({ x: W - T, y: 0, w: T, h: H });

    function vAt(r, c) { return v[r * VC + c]; }
    function hAt(r, c) { return h[r * C.COLS + c]; }

    for (var c = 1; c < C.COLS; c++) {
      var r = 0;
      while (r < C.ROWS) {
        if (!vAt(r, c)) { r++; continue; }
        var r2 = r;
        while (r2 + 1 < C.ROWS && vAt(r2 + 1, c)) r2++;
        var y0 = r * C.CELL - T / 2, y1 = (r2 + 1) * C.CELL + T / 2;
        walls.push({ x: c * C.CELL - T / 2, y: y0, w: T, h: y1 - y0 });
        r = r2 + 1;
      }
    }
    for (var r3 = 1; r3 < C.ROWS; r3++) {
      var c3 = 0;
      while (c3 < C.COLS) {
        if (!hAt(r3, c3)) { c3++; continue; }
        var c4 = c3;
        while (c4 + 1 < C.COLS && hAt(r3, c4 + 1)) c4++;
        var x0 = c3 * C.CELL - T / 2, x1 = (c4 + 1) * C.CELL + T / 2;
        walls.push({ x: x0, y: r3 * C.CELL - T / 2, w: x1 - x0, h: T });
        c3 = c4 + 1;
      }
    }
    return walls;
  }

  function outerWalls() {
    var T = C.WT;
    return [
      { x: 0, y: 0, w: C.W, h: T },
      { x: 0, y: C.H - T, w: C.W, h: T },
      { x: 0, y: 0, w: T, h: C.H },
      { x: C.W - T, y: 0, w: T, h: C.H }
    ];
  }

  /* ------------------------------------------------------------------
     Geometry / world
     ------------------------------------------------------------------ */
  function rayBox(ox, oy, dx, dy, minx, miny, maxx, maxy) {
    var tmin = 0, tmax = 1, nx = 0, ny = 0;
    if (Math.abs(dx) < 1e-9) { if (ox < minx || ox > maxx) return null; }
    else {
      var inv = 1 / dx, t1 = (minx - ox) * inv, t2 = (maxx - ox) * inv, s = -1;
      if (t1 > t2) { var tmp = t1; t1 = t2; t2 = tmp; s = 1; }
      if (t1 > tmin) { tmin = t1; nx = s; ny = 0; }
      if (t2 < tmax) tmax = t2;
      if (tmin > tmax) return null;
    }
    if (Math.abs(dy) < 1e-9) { if (oy < miny || oy > maxy) return null; }
    else {
      var inv2 = 1 / dy, u1 = (miny - oy) * inv2, u2 = (maxy - oy) * inv2, s2 = -1;
      if (u1 > u2) { var tmp2 = u1; u1 = u2; u2 = tmp2; s2 = 1; }
      if (u1 > tmin) { tmin = u1; nx = 0; ny = s2; }
      if (u2 < tmax) tmax = u2;
      if (tmin > tmax) return null;
    }
    return { t: tmin, nx: nx, ny: ny };
  }

  function World(walls) {
    this.walls = walls;
    this.field = this.buildField();
    this.grid = this.buildGrid();
  }

  World.prototype.raycast = function (x0, y0, dx, dy) {
    var minx = Math.min(x0, x0 + dx), maxx = Math.max(x0, x0 + dx);
    var miny = Math.min(y0, y0 + dy), maxy = Math.max(y0, y0 + dy);
    var best = null, W = this.walls;
    for (var i = 0; i < W.length; i++) {
      var w = W[i];
      if (w.x > maxx || w.x + w.w < minx || w.y > maxy || w.y + w.h < miny) continue;
      var hit = rayBox(x0, y0, dx, dy, w.x, w.y, w.x + w.w, w.y + w.h);
      if (hit && hit.t >= 0 && hit.t <= 1 && (!best || hit.t < best.t)) best = hit;
    }
    return best;
  };

  World.prototype.solid = function (x, y, pad) {
    var p = pad || 0, W = this.walls;
    for (var i = 0; i < W.length; i++) {
      var w = W[i];
      if (x >= w.x - p && x <= w.x + w.w + p && y >= w.y - p && y <= w.y + w.h + p) return true;
    }
    return false;
  };

  World.prototype.los = function (x0, y0, x1, y1) {
    return !this.raycast(x0, y0, x1 - x0, y1 - y0);
  };

  /* 与 map.resolveCircle 完全一致的推挤 */
  World.prototype.resolve = function (x, y, r) {
    var W = this.walls;
    for (var pass = 0; pass < 2; pass++) {
      var hit = false;
      for (var i = 0; i < W.length; i++) {
        var w = W[i];
        if (x + r < w.x || x - r > w.x + w.w || y + r < w.y || y - r > w.y + w.h) continue;
        var nx = clamp(x, w.x, w.x + w.w), ny = clamp(y, w.y, w.y + w.h);
        var dx = x - nx, dy = y - ny, d2 = dx * dx + dy * dy;
        if (d2 >= r * r) continue;
        if (d2 > 1e-8) {
          var d = Math.sqrt(d2), push = r - d;
          x += (dx / d) * push; y += (dy / d) * push;
        } else {
          var left = x - w.x, right = w.x + w.w - x, top = y - w.y, bottom = w.y + w.h - y;
          var m = Math.min(left, right, top, bottom);
          if (m === left) x = w.x - r;
          else if (m === right) x = w.x + w.w + r;
          else if (m === top) y = w.y - r;
          else y = w.y + w.h + r;
        }
        hit = true;
      }
      if (!hit) break;
    }
    return { x: x, y: y };
  };

  /* 每格 8 个方向的可通行度，供走位评分查表 */
  World.prototype.buildField = function () {
    var f = new Int8Array(TOTAL);
    var dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [0.7, 0.7], [0.7, -0.7], [-0.7, 0.7], [-0.7, -0.7]];
    for (var r = 0; r < C.ROWS; r++) {
      for (var c = 0; c < C.COLS; c++) {
        var cx = c * C.CELL + C.CELL / 2, cy = r * C.CELL + C.CELL / 2, open = 0;
        for (var i = 0; i < 8; i++) {
          var blocked = false;
          for (var j = 1; j <= 3; j++) {
            var d = j * 19;
            if (this.solid(cx + dirs[i][0] * d, cy + dirs[i][1] * d, 3)) { blocked = true; break; }
          }
          if (!blocked) open++;
        }
        f[r * C.COLS + c] = open;
      }
    }
    return f;
  };

  function fieldAt(world, x, y) {
    var c = clamp(Math.floor(x / C.CELL), 0, C.COLS - 1);
    var r = clamp(Math.floor(y / C.CELL), 0, C.ROWS - 1);
    return world.field[r * C.COLS + c];
  }

  /* 用格心→相邻格心的射线反推每格的 4 个出口，供 BFS 寻路 */
  World.prototype.buildGrid = function () {
    var g = new Uint8Array(TOTAL * 4);   /* E, W, S, N */
    for (var r = 0; r < C.ROWS; r++) {
      for (var c = 0; c < C.COLS; c++) {
        var cx = c * C.CELL + C.CELL / 2, cy = r * C.CELL + C.CELL / 2;
        var base = (r * C.COLS + c) * 4;
        if (c < C.COLS - 1) g[base] = this.raycast(cx, cy, C.CELL, 0) ? 0 : 1;
        if (c > 0) g[base + 1] = this.raycast(cx, cy, -C.CELL, 0) ? 0 : 1;
        if (r < C.ROWS - 1) g[base + 2] = this.raycast(cx, cy, 0, C.CELL) ? 0 : 1;
        if (r > 0) g[base + 3] = this.raycast(cx, cy, 0, -C.CELL) ? 0 : 1;
      }
    }
    return g;
  };

  /* 网格 BFS：局部 MPC 不会绕墙，残局必须靠它走到最后一个敌人面前 */
  function pathTo(world, sx, sy, tx, ty) {
    var sc = clamp(Math.floor(sx / C.CELL), 0, C.COLS - 1);
    var sr = clamp(Math.floor(sy / C.CELL), 0, C.ROWS - 1);
    var tc = clamp(Math.floor(tx / C.CELL), 0, C.COLS - 1);
    var tr = clamp(Math.floor(ty / C.CELL), 0, C.ROWS - 1);
    var start = sr * C.COLS + sc, goal = tr * C.COLS + tc;
    if (start === goal) return [{ x: tx, y: ty }];

    var prev = new Int16Array(TOTAL);
    prev.fill(-2);
    var q = new Int16Array(TOTAL), head = 0, tail = 0;
    prev[start] = -1; q[tail++] = start;
    var g = world.grid, found = false;

    while (head < tail) {
      var cur = q[head++];
      if (cur === goal) { found = true; break; }
      var base = cur * 4;
      if (g[base] && prev[cur + 1] === -2) { prev[cur + 1] = cur; q[tail++] = cur + 1; }
      if (g[base + 1] && prev[cur - 1] === -2) { prev[cur - 1] = cur; q[tail++] = cur - 1; }
      if (g[base + 2] && prev[cur + C.COLS] === -2) { prev[cur + C.COLS] = cur; q[tail++] = cur + C.COLS; }
      if (g[base + 3] && prev[cur - C.COLS] === -2) { prev[cur - C.COLS] = cur; q[tail++] = cur - C.COLS; }
    }
    if (!found) return null;

    var out = [], node = goal, guard = 0;
    while (node !== start && guard++ < TOTAL) {
      var r = (node / C.COLS) | 0, c = node % C.COLS;
      out.push({ x: c * C.CELL + C.CELL / 2, y: r * C.CELL + C.CELL / 2 });
      node = prev[node];
      if (node < 0) break;
    }
    out.reverse();
    if (out.length) out[out.length - 1] = { x: tx, y: ty };
    return out;
  }

  /* ------------------------------------------------------------------
     Exact bullet tracing, including up to 5 wall bounces
     ------------------------------------------------------------------ */
  function trace(world, x, y, angle, maxLen, b0) {
    var segs = [];
    var cx = x, cy = y, vx = Math.cos(angle), vy = Math.sin(angle);
    var remaining = maxLen, total = 0, b = b0 || 0;
    for (var it = 0; it < 9 && remaining > 0.5; it++) {
      var dx = vx * remaining, dy = vy * remaining;
      var hit = world.raycast(cx, cy, dx, dy);
      if (!hit) {
        segs.push({ x0: cx, y0: cy, x1: cx + dx, y1: cy + dy, len: remaining, s: total, nx: 0, ny: 0 });
        break;
      }
      var t = clamp(hit.t, 0, 1), L = remaining * t;
      var hx = cx + dx * t, hy = cy + dy * t;
      segs.push({ x0: cx, y0: cy, x1: hx, y1: hy, len: L, s: total, nx: hit.nx, ny: hit.ny });
      total += L;
      b++;
      if (b > 5 || (hit.nx === 0 && hit.ny === 0) || L < 0.02) break;
      var dot = vx * hit.nx + vy * hit.ny;
      if (dot < 0) { vx -= 2 * dot * hit.nx; vy -= 2 * dot * hit.ny; }
      var n = hypot(vx, vy) || 1; vx /= n; vy /= n;
      cx = hx + hit.nx * 0.8; cy = hy + hit.ny * 0.8;
      remaining = remaining - L - 0.7;
    }
    return segs;
  }

  function segCircle(x0, y0, x1, y1, cx, cy, r) {
    var dx = x1 - x0, dy = y1 - y0, fx = x0 - cx, fy = y0 - cy;
    var a = dx * dx + dy * dy;
    if (a < 1e-9) return (fx * fx + fy * fy <= r * r) ? 0 : -1;
    var b = 2 * (fx * dx + fy * dy), c = fx * fx + fy * fy - r * r;
    var disc = b * b - 4 * a * c;
    if (disc < 0) return -1;
    disc = Math.sqrt(disc);
    var t1 = (-b - disc) / (2 * a), t2 = (-b + disc) / (2 * a);
    if (t1 >= 0 && t1 <= 1) return t1;
    if (t2 >= 0 && t2 <= 1) return t2;
    if (t1 < 0 && t2 > 1) return 0;
    return -1;
  }

  /* 轨迹在 [0.14, tMax] 内是否会经过我身边 —— 自伤否决的核心 */
  function selfRisk(segs, sx, sy, tMax, speed) {
    var risk = 0;
    for (var i = 0; i < segs.length; i++) {
      var s = segs[i];
      var dx = s.x1 - s.x0, dy = s.y1 - s.y0, len2 = dx * dx + dy * dy;
      var t = len2 > 1e-9 ? clamp(((sx - s.x0) * dx + (sy - s.y0) * dy) / len2, 0, 1) : 0;
      var px = s.x0 + dx * t, py = s.y0 + dy * t;
      var d = hypot(px - sx, py - sy);
      var when = (s.s + t * s.len) / speed;
      if (when >= 0.14 && when <= tMax && d < 54) {
        risk += (1 - d / 54) * (1.35 - 0.35 * Math.min(1, when / 2));
      }
    }
    return risk;
  }

  /* 返回轨迹首次穿过圆 (cx,cy,r) 的时间与坐标 */
  function firstHit(segs, cx, cy, r, speed) {
    var best = null;
    for (var i = 0; i < segs.length; i++) {
      var s = segs[i];
      var t = segCircle(s.x0, s.y0, s.x1, s.y1, cx, cy, r);
      if (t < 0) continue;
      var when = (s.s + t * s.len) / speed;
      if (!best || when < best.when) best = { when: when, x: s.x0 + (s.x1 - s.x0) * t, y: s.y0 + (s.y1 - s.y0) * t };
    }
    return best;
  }

  function barrelClear(world, self, angle) {
    var ca = Math.cos(angle), sa = Math.sin(angle);
    var tx = self.x + ca * C.MUZZLE, ty = self.y + sa * C.MUZZLE;
    if (world.solid(tx, ty, 1)) return false;
    if (world.solid(tx + ca * 12, ty + sa * 12, 1)) return false;
    return true;
  }

  /* ------------------------------------------------------------------
     Shared state (all copies of this file share brains / caches)
     ------------------------------------------------------------------ */
  var SHARED = root.__HY_SHARED__;
  if (!SHARED) {
    SHARED = {
      brains: new Map(),
      worldCache: new WeakMap(),
      fb: null,
      mapScore: [0, 0],
      takeoverAll: false,
      quality: 1,
      TX: new Float32Array(BMAX * K), TY: new Float32Array(BMAX * K), TW: new Float32Array(BMAX * K),
      EX: new Float32Array(EMAX * K), EY: new Float32Array(EMAX * K)
    };
    root.__HY_SHARED__ = SHARED;
  }

  function getBrain(id) {
    var key = String(id);
    var b = SHARED.brains.get(key);
    if (!b) {
      b = {
        lastToken: -1, roundKey: null, roundTime: 0,
        lastX: -1, lastY: -1, stuck: 0, escape: 0, escapeSteer: 1, flip: 1,
        scanCd: 0, bankTtl: 0, bankPlan: null,
        suppTtl: 0, suppPlan: null, lastSteer: 0, lastDrive: 0,
        prevEnemies: new Map(), wp: null, wpCd: 0, shots: 0
      };
      SHARED.brains.set(key, b);
    }
    return b;
  }

  function normaliseWalls(raw) {
    var out = [], T = C.WT;
    for (var i = 0; i < raw.length; i++) {
      var w = raw[i];
      if (!w) continue;
      if (Array.isArray(w)) {
        out.push({ x: w[0], y: w[1], w: w[2] || T, h: w[3] || T });
      } else if (typeof w.x1 === 'number') {
        var x1 = w.x1, y1 = w.y1, x2 = w.x2, y2 = w.y2;
        if (Math.abs(x2 - x1) >= Math.abs(y2 - y1)) {
          out.push({ x: Math.min(x1, x2), y: y1 - T / 2, w: Math.abs(x2 - x1), h: T });
        } else {
          out.push({ x: x1 - T / 2, y: Math.min(y1, y2), w: T, h: Math.abs(y2 - y1) });
        }
      } else {
        var x = num(w.x, 0), y = num(w.y, 0);
        var ww = num(w.w, w.width, T), hh = num(w.h, w.height, T);
        out.push({ x: x, y: y, w: ww, h: hh });
      }
    }
    return out;
  }

  function getWorld(raw) {
    var cached = SHARED.worldCache.get(raw);
    if (cached) return cached;
    var w = new World(normaliseWalls(raw));
    SHARED.worldCache.set(raw, w);
    return w;
  }

  function ensureFallback() {
    if (!SHARED.fb) {
      var f1 = makeFlags(20260101), f2 = makeFlags(88888888);
      SHARED.fb = [
        new World(wallsFromFlags(f1.v, f1.h)),
        new World(wallsFromFlags(f2.v, f2.h)),
        new World(outerWalls())
      ];
    }
    return SHARED.fb;
  }

  /* 没有墙信息时：用观测在两张固定地图中投票，都不吻合则退化为外墙 */
  function chooseFallback(S) {
    var fb = ensureFallback(), sc = SHARED.mapScore;
    sc[0] *= 0.995; sc[1] *= 0.995;
    for (var m = 0; m < 2; m++) {
      var w = fb[m];
      for (var i = 0; i < S.tanks.length; i++)
        if (w.solid(S.tanks[i].x, S.tanks[i].y, 10)) sc[m] -= 3;
      for (var j = 0; j < S.bullets.length; j++) {
        var b = S.bullets[j];
        if (w.solid(b.x, b.y, 0)) sc[m] -= 1.5;
        var dx = b.x - b.px, dy = b.y - b.py;
        if (dx * dx + dy * dy > 4) {
          var hit = w.raycast(b.px, b.py, dx, dy);
          if (hit && hit.t < 0.85) sc[m] -= 6;   /* 子弹穿墙 = 这张图不对 */
        }
      }
    }
    var best = sc[0] >= sc[1] ? 0 : 1;
    if (sc[best] < -70) return fb[2];
    return fb[best];
  }

  /* ------------------------------------------------------------------
     Input normalisation
     ------------------------------------------------------------------ */
  function isTank(o) {
    return !!(o && typeof o.x === 'number' && typeof o.y === 'number' && typeof o.angle === 'number');
  }
  function isWorld(o) {
    return !!(o && (Array.isArray(o.tanks) || Array.isArray(o.bullets) || Array.isArray(o.walls) ||
      (o.map && (o.map.walls || o.map.vFlag))));
  }

  function normTank(o, i) {
    return {
      raw: o,
      x: num(o.x), y: num(o.y), a: num(o.angle, o.heading, o.dir),
      speed: num(o.speed, o.velocity, 0),
      id: (o.index != null ? o.index : (o.id != null ? o.id : (o.playerId != null ? o.playerId : i))),
      alive: o.alive !== false && o.dead !== true && o.destroyed !== true,
      cd: num(o.cooldown, o.coolDown, o.reload, o.fireCooldown, 0),
      r: num(o.radius, 16)
    };
  }

  function normBullet(o, i) {
    var vx = num(o.vx, o.dx, 0), vy = num(o.vy, o.dy, 0);
    var a = typeof o.angle === 'number' ? o.angle :
      (typeof o.heading === 'number' ? o.heading : Math.atan2(vy, vx || 1));
    var ca = Math.cos(a), sa = Math.sin(a);
    var owner = (o.owner != null ? o.owner : (o.ownerId != null ? o.ownerId :
      (o.player != null ? o.player : (o.shooter != null ? o.shooter : -1))));
    return {
      raw: o, x: num(o.x), y: num(o.y),
      px: num(o.px, o.prevX, o.lastX, o.x - ca * 10),
      py: num(o.py, o.prevY, o.lastY, o.y - sa * 10),
      angle: a, owner: owner,
      bounces: num(o.bounces, o.bounceCount, 0),
      life: num(o.life, o.ttl, 6),
      grace: num(o.grace, o.spawnGrace, 0),
      dead: o.dead === true || o.alive === false
    };
  }

  function normalise(a, b, c) {
    var selfRaw = null, wo = null, dt = 0, i;

    if (isTank(a) && (b == null || isWorld(b) || typeof b === 'number')) {
      selfRaw = a; wo = (b && typeof b === 'object') ? b : null;
      if (typeof b === 'number') dt = b;
    } else if (isWorld(a) && isTank(b)) {
      selfRaw = b; wo = a;
    } else if (a && (a.self || a.me || a.you || a.tank || a.player || a.ego) && !isTank(a)) {
      wo = a; selfRaw = a.self || a.me || a.you || a.tank || a.player || a.ego;
    } else if (isWorld(a)) {
      wo = a;
      var want = (a.selfIndex != null ? a.selfIndex : (a.index != null ? a.index : (a.id != null ? a.id : null)));
      if (want != null && Array.isArray(a.tanks)) {
        for (i = 0; i < a.tanks.length; i++) {
          var t = a.tanks[i];
          if (t.isSelf === true || t.self === true ||
            (t.index != null && String(t.index) === String(want)) ||
            (t.id != null && String(t.id) === String(want))) { selfRaw = t; break; }
        }
      }
      if (!selfRaw && Array.isArray(a.tanks) && a.tanks.length) selfRaw = a.tanks[0];
    } else if (isTank(a)) {
      selfRaw = a; wo = null;
    }

    if (typeof c === 'number') dt = c;
    if (!dt && wo && typeof wo.dt === 'number') dt = wo.dt;
    if (!dt && wo && typeof wo.timeStep === 'number') dt = wo.timeStep;
    dt = clamp(dt || STEP, 1 / 240, 1 / 20);

    var tanksRaw = (wo && (wo.tanks || wo.players || wo.entities ||
      (wo.world && wo.world.tanks))) || [];
    var bulletsRaw = (wo && (wo.bullets || wo.projectiles || wo.shots || wo.missiles ||
      (wo.world && wo.world.bullets))) || [];
    var self = selfRaw ? normTank(selfRaw, 0) : null;
    var tanks = [], bullets = [], seen = false;

    for (i = 0; i < tanksRaw.length; i++) {
      var nt = normTank(tanksRaw[i], i);
      if (self && sameId(nt.id, self.id) && tanksRaw[i] === selfRaw) seen = true;
      tanks.push(nt);
    }
    if (self && !seen) tanks.push(self);
    for (i = 0; i < bulletsRaw.length; i++) bullets.push(normBullet(bulletsRaw[i], i));

    /* 地图来源：优先调用方给的墙，其次 flag 重建，最后固定种子推理 */
    var world = null;
    var wallRaw = null, mapObj = null;
    if (wo) {
      wallRaw = wo.walls || (wo.world && wo.world.walls) || null;
      mapObj = wo.map || (wo.world && wo.world.map) || null;
      if (!wallRaw && mapObj) wallRaw = mapObj.walls || mapObj.wallList || null;
    }
    if (wallRaw && wallRaw.length) world = getWorld(wallRaw);
    else if (mapObj && (mapObj.vFlag || mapObj.hFlag)) {
      var flags = { v: mapObj.vFlag || new Uint8Array(C.ROWS * VC), h: mapObj.hFlag || new Uint8Array(HC * C.COLS) };
      world = getWorld(wallsFromFlags(flags.v, flags.h));
    }

    var tkey;
    if (wo && typeof wo.time === 'number') tkey = 't' + wo.time.toFixed(4);
    else if (wo && typeof wo.tick === 'number') tkey = 'k' + wo.tick;
    else if (wo && typeof wo.frame === 'number') tkey = 'f' + wo.frame;
    else tkey = 'b' + Math.floor(nowMs() * 0.06);

    var roundKey = (wo && (wo.roundSerial != null ? wo.roundSerial : wo.round)) != null
      ? 'r' + (wo.roundSerial != null ? wo.roundSerial : wo.round) : 'r0';

    return {
      self: self, tanks: tanks, bullets: bullets, world: world, dt: dt,
      tkey: tkey, roundKey: roundKey, raw: wo
    };
  }

  /* ------------------------------------------------------------------
     Threat sampling
     ------------------------------------------------------------------ */
  function fillSamples(segs, base, weight) {
    var TX = SHARED.TX, TY = SHARED.TY, TW = SHARED.TW;
    var si = 0, total = 0;
    if (segs.length) total = segs[segs.length - 1].s + segs[segs.length - 1].len;
    for (var k = 0; k < K; k++) {
      var d = C.BS * k * STEP;
      if (!segs.length || d > total + 0.001) {
        TW[base + k] = 0; TX[base + k] = 1e9; TY[base + k] = 1e9; continue;
      }
      while (si < segs.length - 1 && d > segs[si].s + segs[si].len) si++;
      var s = segs[si];
      var t = s.len > 1e-6 ? clamp((d - s.s) / s.len, 0, 1) : 0;
      TX[base + k] = s.x0 + (s.x1 - s.x0) * t;
      TY[base + k] = s.y0 + (s.y1 - s.y0) * t;
      TW[base + k] = weight * (1 - 0.45 * k / K);
    }
  }

  function estimateVel(brain, e, dt, fresh) {
    var vx = Math.cos(e.a) * e.speed, vy = Math.sin(e.a) * e.speed;
    if (e.speed === 0) {
      var p = brain.prevEnemies.get(String(e.id));
      if (p && fresh && dt > 0) {
        vx = clamp((e.x - p.x) / dt, -C.TS, C.TS);
        vy = clamp((e.y - p.y) / dt, -C.TS, C.TS);
      }
    }
    if (fresh) brain.prevEnemies.set(String(e.id), { x: e.x, y: e.y });
    var m = hypot(vx, vy);
    if (m > C.TS) { vx *= C.TS / m; vy *= C.TS / m; }
    return { vx: vx, vy: vy };
  }

  /* ------------------------------------------------------------------
     Shot planning
     ------------------------------------------------------------------ */
  function evalDirect(world, self, e, vel, aggression) {
    var dx = e.x - self.x, dy = e.y - self.y;
    var d = hypot(dx, dy);
    if (d > 820 || d < 24) return null;

    var t = d / C.BS;
    var px = e.x + vel.vx * t * 0.92, py = e.y + vel.vy * t * 0.92;
    t = hypot(px - self.x, py - self.y) / C.BS;
    px = e.x + vel.vx * t * 0.92; py = e.y + vel.vy * t * 0.92;

    var aim = Math.atan2(py - self.y, px - self.x);
    var ca = Math.cos(aim), sa = Math.sin(aim);
    var tx = self.x + ca * C.MUZZLE, ty = self.y + sa * C.MUZZLE;
    if (!barrelClear(world, self, aim)) return null;
    /* 中心线 + 两条侧偏射线，避免擦墙哑火 */
    var nx = -sa, ny = ca;
    var toX = px - tx, toY = py - ty;
    if (world.raycast(tx, ty, toX, toY)) return null;
    if (world.raycast(tx + nx * 7, ty + ny * 7, toX, toY) &&
      world.raycast(tx - nx * 7, ty - ny * 7, toX, toY)) return null;

    var hitT = hypot(px - tx, py - ty) / C.BS;
    if (hitT > 1.5 + 0.5 * aggression) return null;

    /* 命中前的轨迹若回到自己身边则否决；残局收窄缓冲，否则永远打不出枪 */
    var segs = trace(world, tx, ty, aim, C.BS * 2.2, 0);
    var risk = selfRisk(segs, self.x, self.y,
      Math.min(hitT + 0.40 - 0.25 * aggression, 2.2), C.BS);
    if (risk > 0.20 + 0.35 * aggression) return null;

    var conf = clamp(1.15 - d / 900, 0.42, 0.95);
    var value = 20 * conf * (1 + 0.35 * aggression) - risk * 22 - hitT * 2;
    if (value < 5) return null;
    return { aim: aim, value: value, safe: true };
  }

  /* 反弹猎杀：扫描一圈角度，用真实弹道找能打到敌人的一跳 */
  function scanBank(world, self, enemies, vels, aggression) {
    var N = 72, best = null;
    var maxLen = C.BS * 2.6;
    for (var i = 0; i < N; i++) {
      var a = -Math.PI + (i + 0.5) * TAU / N;
      var ca = Math.cos(a), sa = Math.sin(a);
      var tx = self.x + ca * C.MUZZLE, ty = self.y + sa * C.MUZZLE;
      if (world.solid(tx, ty, 1)) continue;
      var segs = trace(world, tx, ty, a, maxLen, 0);
      for (var j = 0; j < enemies.length; j++) {
        var e = enemies[j], vel = vels[j];
        var hit = firstHit(segs, e.x + vel.vx * 0.35, e.y + vel.vy * 0.35, 21, C.BS);
        if (!hit || hit.when > 1.9 + 0.6 * aggression || hit.when < 0.06) continue;
        /* 敌人会移动：检查命中点是否仍在其外推位置附近 */
        var ex = e.x + vel.vx * hit.when, ey = e.y + vel.vy * hit.when;
        var tol = 34 + 12 * aggression;
        if (dist2(hit.x, hit.y, ex, ey) > tol * tol) continue;
        var risk = selfRisk(segs, self.x, self.y, Math.min(hit.when, 2.4), C.BS);
        if (risk > 0.16 + 0.20 * aggression) continue;
        var value = 15 / (0.75 + hit.when) * (1 + 0.5 * aggression) - risk * 20;
        if (value > 5 && (!best || value > best.value)) best = { aim: a, value: value, safe: true };
      }
    }
    return best;
  }

  /* 残局破局：对手（内置 bot）完全不躲子弹，而我们的 MPC 会躲自己的子弹。
     向敌方区域打一发"低自伤"的反弹弹，靠弹道覆盖把僵局磨掉。
     只在弹道前期不回自己身边、且会穿过敌人附近时才打，绝不盲射。 */
  function planSuppression(world, self, enemies) {
    if (!enemies.length) return null;
    var e = enemies[0], bd = Infinity;
    for (var i = 0; i < enemies.length; i++) {
      var d = dist2(self.x, self.y, enemies[i].x, enemies[i].y);
      if (d < bd) { bd = d; e = enemies[i]; }
    }
    var toE = Math.atan2(e.y - self.y, e.x - self.x);
    var offs = [0, 0.12, -0.12, 0.25, -0.25, 0.42, -0.42, 0.62, -0.62];
    var best = null;
    for (var k = 0; k < offs.length; k++) {
      var a = toE + offs[k];
      if (!barrelClear(world, self, a)) continue;
      var ca = Math.cos(a), sa = Math.sin(a);
      var tx = self.x + ca * C.MUZZLE, ty = self.y + sa * C.MUZZLE;
      var segs = trace(world, tx, ty, a, C.BS * 2.0, 0);
      var risk = selfRisk(segs, self.x, self.y, 1.8, C.BS);
      if (risk > 0.09) continue;
      var pass = firstHit(segs, e.x, e.y, 95, C.BS);
      if (!pass || pass.when > 2.2) continue;
      var value = 6.5 - Math.abs(offs[k]) * 2.0 - risk * 8
        + (1 - Math.min(1, pass.when / 2)) * 1.5;
      if (value > 3 && (!best || value > best.value)) {
        best = { aim: a, value: value, safe: true, suppress: true };
      }
    }
    return best;
  }

  function planShot(world, self, enemies, vels, brain, dt, aggression, fresh) {
    var best = null, i;
    for (i = 0; i < enemies.length; i++) {
      var cand = evalDirect(world, self, enemies[i], vels[i], aggression);
      if (cand && (!best || cand.value > best.value)) best = cand;
    }
    if (fresh) {
      brain.scanCd -= dt;
      brain.bankTtl -= dt;
      if (brain.bankTtl <= 0) brain.bankPlan = null;
    }
    if (brain.scanCd <= 0 && (!best || best.value < 12)) {
      brain.bankPlan = scanBank(world, self, enemies, vels, aggression);
      brain.bankTtl = 0.30;
      brain.scanCd = 0.19 - 0.08 * aggression;
    }
    if (brain.bankPlan && (!best || brain.bankPlan.value > best.value)) best = brain.bankPlan;
    return (best && best.value > 5.5) ? best : null;
  }

  /* ------------------------------------------------------------------
     Model-predictive manoeuvre selection
     ------------------------------------------------------------------ */
  function pickManoeuvre(world, self, brain, shotPlan, shotSteer, shotValue,
    nb, ne, aggression, dt, wp, wWay) {
    var steers = [-1, -0.55, 0, 0.55, 1];
    if (shotPlan) steers.push(shotSteer);
    var drives = [-1, 0, 1];
    var best = null, bestCost = Infinity;
    var TW = SHARED.TW, TX = SHARED.TX, TY = SHARED.TY, EX = SHARED.EX, EY = SHARED.EY;

    for (var si = 0; si < steers.length; si++) {
      for (var di = 0; di < drives.length; di++) {
        var steer = steers[si], drive = drives[di];
        var canShoot = !!(shotPlan && Math.abs(steer - shotSteer) < 1e-6);
        var angle = self.a, speed = self.speed, x = self.x, y = self.y, cost = 0;

        for (var k = 0; k < K; k++) {
          angle += steer * C.TTURN * STEP;
          if (drive !== 0) {
            var max = C.TS * (drive < 0 ? C.TREV : 1);
            var target = drive > 0 ? max : -max;
            var rate = (speed * drive < 0) ? C.TBRK : C.TACC;
            if (speed < target) speed = Math.min(target, speed + rate * STEP);
            else if (speed > target) speed = Math.max(target, speed - rate * STEP);
          } else {
            var drop = C.TBRK * 0.75 * STEP;
            if (Math.abs(speed) <= drop) speed = 0;
            else speed -= sgn(speed) * drop;
          }
          var nx = x + Math.cos(angle) * speed * STEP;
          var ny = y + Math.sin(angle) * speed * STEP;
          var res = world.resolve(nx, ny, C.TR);
          if (Math.abs(res.x - nx) + Math.abs(res.y - ny) > 0.4) {
            cost += 0.22;
            if (Math.abs(speed) > 60) speed *= 0.55;
          }
          x = res.x; y = res.y;

          var tw = 1 - 0.5 * k / K;
          for (var bi = 0; bi < nb; bi++) {
            var base = bi * K + k, w = TW[base];
            if (w <= 0) continue;
            var ddx = x - TX[base], ddy = y - TY[base], dd = ddx * ddx + ddy * ddy;
            if (dd < SAFE_R2) cost += 200 * w * tw * (1 - Math.sqrt(dd) / SAFE_R);
          }
          if (ne) {
            var minD2 = Infinity;
            for (var ei = 0; ei < ne; ei++) {
              var eb = ei * K + k;
              var ex = EX[eb] - x, ey = EY[eb] - y, e2 = ex * ex + ey * ey;
              if (e2 < minD2) minD2 = e2;
            }
            var md = Math.sqrt(minD2);
            /* 近距直射弹 0.2s 就到，来不及躲：中强度规避，但激进期仍允许压上
               （压得太死会打不完回合，反而输掉整场比赛）。 */
            if (md < 240) {
              cost += (240 - md) / 240 * 3.8 * tw * (1 - 0.65 * aggression);
            } else if (md > 390 && aggression > 0.04) {
              cost += (md - 390) / 500 * 3.4 * aggression * tw;
            }
          }
          if (k % 6 === 0) cost += (8 - fieldAt(world, x, y)) * 0.055;
        }

        var progress = hypot(x - self.x, y - self.y);
        cost += 1.5 - Math.min(1.5, progress * 0.013);
        cost += (8 - fieldAt(world, x, y)) * 0.13;
        if (x < 22 || x > C.W - 22 || y < 22 || y > C.H - 22) cost += 4;
        cost += Math.abs(steer - brain.lastSteer) * 0.22 + Math.abs(drive - brain.lastDrive) * 0.18;
        if (wp && wWay > 0) {
          var d0 = hypot(wp.x - self.x, wp.y - self.y);
          var d1 = hypot(wp.x - x, wp.y - y);
          cost -= clamp(d0 - d1, -55, 55) * wWay;
        }
        if (canShoot) cost -= shotValue;

        if (cost < bestCost) {
          bestCost = cost;
          best = { steer: steer, drive: drive, canShoot: canShoot, cost: cost };
        }
      }
    }
    return best || { steer: 0, drive: 0, canShoot: false, cost: 0 };
  }

  /* ------------------------------------------------------------------
     Output
     ------------------------------------------------------------------ */
  function idle() { return makeOut(0, 0, false); }
  function makeOut(drive, steer, shoot) {
    return {
      drive: drive, steer: steer, shoot: !!shoot,
      throttle: drive, move: drive, turn: steer, rotate: steer,
      fire: !!shoot, shooting: !!shoot,
      up: drive > 0.2, down: drive < -0.2,
      forward: drive > 0.2, backward: drive < -0.2, back: drive < -0.2,
      reverse: drive < -0.2, accelerate: drive > 0.2,
      left: steer < -0.15, right: steer > 0.15
    };
  }

  /* ------------------------------------------------------------------
     Main decision
     ------------------------------------------------------------------ */
  function decideRaw(a, b, c) {
    ensureHook();
    var S = normalise(a, b, c);
    var self = S.self;
    if (!self || !self.alive) return idle();
    if (!S.world) S.world = chooseFallback(S);
    var world = S.world, dt = S.dt, tanks = S.tanks, bullets = S.bullets;

    var brain = getBrain(self.id);
    var fresh = (brain.lastToken !== S.tkey);
    brain.lastToken = S.tkey;

    if (brain.roundKey !== S.roundKey ||
      (brain.lastX > 0 && Math.abs(self.x - brain.lastX) > 260) ||
      (brain.lastY > 0 && Math.abs(self.y - brain.lastY) > 230)) {
      brain.roundKey = S.roundKey; brain.roundTime = 0;
      brain.bankPlan = null; brain.scanCd = 0; brain.shots = 0;
      brain.suppPlan = null; brain.suppTtl = 0;
      brain.wp = null; brain.wpCd = 0;
      brain.prevEnemies.clear();
    }
    if (fresh) brain.roundTime += dt;

    var moved = hypot(self.x - brain.lastX, self.y - brain.lastY);
    if (fresh) {
      if (moved < 1.3 && Math.abs(brain.lastDrive) > 0.2) brain.stuck += dt;
      else brain.stuck = Math.max(0, brain.stuck - dt * 2.5);
    }
    brain.lastX = self.x; brain.lastY = self.y;

    var enemies = [], vels = [], live = 0, i;
    for (i = 0; i < tanks.length; i++) {
      var t = tanks[i];
      if (!t.alive || sameId(t.id, self.id)) continue;
      enemies.push(t);
    }
    for (i = 0; i < bullets.length; i++) {
      if (!bullets[i].dead && sameId(bullets[i].owner, self.id)) live++;
    }
    for (i = 0; i < enemies.length; i++) vels.push(estimateVel(brain, enemies[i], dt, fresh));

    /* ---- 威胁采样 ---- */
    var TW = SHARED.TW;
    TW.fill(0);
    var nb = 0, maxLen = C.BS * (K * STEP + 0.03);
    for (i = 0; i < bullets.length && nb < BMAX; i++) {
      var bl = bullets[i];
      if (bl.dead) continue;
      var mine = sameId(bl.owner, self.id);
      var segs = trace(world, bl.x, bl.y, bl.angle, maxLen, bl.bounces);
      var base = nb * K;
      fillSamples(segs, base, mine ? 0.95 : 1.0);
      if (mine && bl.grace > 0) {
        for (var gk = 0; gk < K && gk * STEP < bl.grace; gk++) TW[base + gk] = 0;
      }
      nb++;
    }
    /* 任何看得见我的敌人都是潜在射手：沿"敌人→我"的直线布一条虚拟弹道。
       这样在子弹真的出现之前，我就已经侧向离开那条射线。 */
    var myvx = Math.cos(self.a) * self.speed, myvy = Math.sin(self.a) * self.speed;
    for (i = 0; i < enemies.length && nb < BMAX; i++) {
      var e = enemies[i];
      var toMe = Math.atan2(self.y - e.y, self.x - e.x);
      var dd = hypot(self.x - e.x, self.y - e.y);
      if (dd > 620 || !world.los(e.x, e.y, self.x, self.y)) continue;
      var off = Math.abs(angleDiff(e.a, toMe));
      var align = Math.max(0, 1 - off / 0.55);
      var near = clamp(1 - dd / 700, 0.25, 1);
      var w = 0.20 + 0.42 * align * near;
      var etx = e.x + Math.cos(toMe) * C.MUZZLE, ety = e.y + Math.sin(toMe) * C.MUZZLE;
      if (world.solid(etx, ety, 1)) continue;
      fillSamples(trace(world, etx, ety, toMe, maxLen, 0), nb * K, w);
      nb++;
      /* 对手普遍会打提前量，再布一条指向"我 0.3s 后位置"的弹道，
         逼自己不要匀速直线送人头。 */
      if (nb < BMAX && dd < 520) {
        var fx = self.x + myvx * 0.30, fy = self.y + myvy * 0.30;
        var leadA = Math.atan2(fy - e.y, fx - e.x);
        var ltx = e.x + Math.cos(leadA) * C.MUZZLE, lty = e.y + Math.sin(leadA) * C.MUZZLE;
        if (!world.solid(ltx, lty, 1)) {
          fillSamples(trace(world, ltx, lty, leadA, maxLen, 0), nb * K, 0.30 * near);
          nb++;
        }
      }
    }

    /* ---- 敌人匀速外推 ---- */
    var ne = Math.min(enemies.length, EMAX), EX = SHARED.EX, EY = SHARED.EY;
    for (i = 0; i < ne; i++) {
      for (var k = 0; k < K; k++) {
        EX[i * K + k] = enemies[i].x + vels[i].vx * k * STEP;
        EY[i * K + k] = enemies[i].y + vels[i].vy * k * STEP;
      }
    }

    /* ---- 侵略度：先苟住；残局必须主动收掉最后一个人，否则平局拖死 ---- */
    var aliveN = 1 + ne;
    var aggression = clamp((brain.roundTime - 6) / 18, 0, 1) * 0.80
      + (ne <= 1 ? 0.45 : 0) + (aliveN <= 3 ? 0.18 : 0);
    aggression = clamp(aggression, 0, 1);
    var maxLive = aggression > 0.75 ? 4 : (aggression > 0.35 ? 3 : 2);

    /* ---- BFS 路点：只有局部评分会永远绕不过墙，残局卡成超时 ---- */
    var wWay = 0.015 + 0.10 * aggression;
    if (fresh) brain.wpCd -= dt;
    if (brain.wpCd <= 0 || !brain.wp) {
      brain.wp = null;
      if (enemies.length) {
        var nearE = enemies[0], nearD = Infinity;
        for (i = 0; i < enemies.length; i++) {
          var dE = dist2(self.x, self.y, enemies[i].x, enemies[i].y);
          if (dE < nearD) { nearD = dE; nearE = enemies[i]; }
        }
        var pathCells = pathTo(world, self.x, self.y, nearE.x, nearE.y);
        if (pathCells && pathCells.length) {
          var last = Math.min(pathCells.length - 1, 4), wp0 = pathCells[0];
          for (var pi = last; pi >= 0; pi--) {
            if (world.los(self.x, self.y, pathCells[pi].x, pathCells[pi].y)) {
              wp0 = pathCells[pi]; break;
            }
          }
          brain.wp = wp0;
        }
      }
      brain.wpCd = 0.30;
    }

    var shotPlan = null, shotSteer = 0, shotValue = 0;
    if (self.cd <= 0.001 && live < maxLive && enemies.length) {
      shotPlan = planShot(world, self, enemies, vels, brain, dt, aggression, fresh);
      if (!shotPlan && aggression > 0.45 && live < 3 && ne <= 2) {
        if (fresh) brain.suppTtl -= dt;
        if (brain.suppTtl <= 0 || !brain.suppPlan) {
          brain.suppPlan = planSuppression(world, self, enemies);
          brain.suppTtl = 0.45;
        }
        shotPlan = brain.suppPlan;
      }
    }
    if (shotPlan) {
      var diff = angleDiff(self.a, shotPlan.aim);
      shotSteer = clamp(diff / (C.TTURN * dt), -1, 1);
      var after = self.a + shotSteer * C.TTURN * dt;
      var canHit = Math.abs(angleDiff(after, shotPlan.aim)) < 0.035;
      shotValue = shotPlan.value * (canHit ? 1 : 0.30) * (shotPlan.suppress ? 0.40 : 1);
      if (!canHit) shotPlan.canShoot = false; else shotPlan.canShoot = true;
    }

    var chosen = pickManoeuvre(world, self, brain, shotPlan, shotSteer, shotValue,
      nb, ne, aggression, dt, brain.wp, wWay);

    var drive = chosen.drive, steer = chosen.steer, shoot = false;

    /* ---- 卡住自救（用位移判定，不能用速度） ---- */
    if (fresh && brain.escape > 0) brain.escape -= dt;
    if (brain.stuck > 0.6 && brain.escape <= 0) {
      brain.stuck = 0;
      brain.escape = 0.5;
      brain.flip = -brain.flip;
      brain.escapeSteer = brain.flip;
    }
    if (brain.escape > 0) {
      drive = -1; steer = brain.escapeSteer; shoot = false;
    } else if (shotPlan && shotPlan.canShoot && shotPlan.safe &&
      self.cd <= 0.001 && live < maxLive) {
      var newAngle = self.a + steer * C.TTURN * dt;
      if (Math.abs(angleDiff(newAngle, shotPlan.aim)) < 0.035 &&
        barrelClear(world, self, shotPlan.aim)) {
        shoot = true;
        brain.shots++;
      }
    }

    brain.lastSteer = steer;
    brain.lastDrive = drive;
    return makeOut(drive, steer, shoot);
  }

  function decide(a, b, c) {
    try { return decideRaw(a, b, c); }
    catch (err) {
      /* 任何异常都不能让引擎崩掉，退化成原地不动 */
      if (root && root.__HY_DEBUG__) console.warn('[HY]', err);
      return idle();
    }
  }

  /* ------------------------------------------------------------------
     Multi-shape registration
     ------------------------------------------------------------------ */
  var api = decide;
  api.decide = decide;
  api.act = decide;
  api.step = decide;
  api.policy = decide;
  api.getAction = decide;
  api.updateBot = decide;
  try { Object.defineProperty(api, 'name', { value: 'hy', configurable: true }); } catch (e) { /* name is read-only in strict mode */ }
  api.id = 'hy';
  api.version = '1.0.0';
  api.configure = function (opts) {
    if (!opts) return api;
    if (opts.takeoverAll != null) SHARED.takeoverAll = !!opts.takeoverAll;
    if (opts.quality != null) SHARED.quality = opts.quality;
    if (opts.reset) { SHARED.brains.clear(); SHARED.mapScore[0] = 0; SHARED.mapScore[1] = 0; }
    return api;
  };
  api.install = ensureHook;
  /* Harmless lifecycle aliases for Arenas that expect an object-like policy */
  api.meta = { name: 'hy', version: '1.0.0', game: 'tank-trouble' };
  api.init = function () { return api; };
  api.start = function () { return api; };
  api.onMatchStart = function () { return api; };
  api.onMatchEnd = function () { return api; };
  api.onRoundStart = function () { return api; };
  api.onRoundEnd = function () { return api; };
  api.destroy = function () { return api; };
  api.reset = function () {
    SHARED.brains.clear();
    SHARED.mapScore[0] = 0; SHARED.mapScore[1] = 0;
    return api;
  };

  function isHyTank(tank, game) {
    if (SHARED.takeoverAll) return true;
    if (!tank) return false;
    var marks = [tank.__hy, tank.hy, tank.policy, tank.policyName, tank.policyId,
      tank.aiName, tank.brainName, tank.controllerName, tank.strategy, tank.agent];
    for (var i = 0; i < marks.length; i++) {
      var m = marks[i];
      if (m === true || m === 'hy' || m === 'HY' || m === 'Hunyuan' || m === api) return true;
    }
    if (tank.controller === api || tank.policyFn === api || tank.brain === api) return true;
    if (game) {
      var idx = tank.index;
      var lists = [game.policies, game.controllers, game.policyIds, game.slots];
      for (var j = 0; j < lists.length; j++) {
        var L = lists[j];
        if (L && (L[idx] === api || L[idx] === 'hy' || L[idx] === 'HY')) return true;
      }
      if (game.hyTankIndex === idx) return true;
      if (root.TT && root.TT.hyTankIndex === idx) return true;
    }
    return false;
  }

  var hooked = false;
  function ensureHook() {
    if (hooked) return;
    var TT = root && root.TT;
    if (!TT || typeof TT.updateBot !== 'function') return;
    hooked = true;
    var orig = TT.updateBot;
    if (orig && orig.__hyWrapped) return;
    var wrapped = function (tank, game, dt) {
      if (isHyTank(tank, game)) return decide(tank, game, dt);
      return orig(tank, game, dt);
    };
    wrapped.__hyWrapped = true;
    wrapped.__hyOriginal = orig;
    TT.updateBot = wrapped;
  }
  api.ensureHook = ensureHook;

  if (root) {
    root.__HY_API__ = api;
    root.HYPolicy = api;
    root.HunyuanPolicy = api;
    if (!root.policy) root.policy = api;
    if (!root.Policy) root.Policy = api;
    var TTg = root.TT || (root.TT = {});
    var P = TTg.Policies || (TTg.Policies = {});
    P.hy = api; P.HY = api; P.hunyuan = api;
    var GP = root.Policies || (root.Policies = {});
    GP.hy = api;
    if (root.TANK_POLICIES) root.TANK_POLICIES.hy = api;
  }
  if (typeof setTimeout === 'function') {
    try { setTimeout(ensureHook, 0); } catch (e) { /* ignore */ }
  }

  return api;
})();

/* CommonJS export lives outside the IIFE so the shared-API early return
   still produces a correct module.exports for every generated entry. */
if (typeof module !== 'undefined' && module && module.exports) {
  module.exports = HY_POLICY_API;
  module.exports.default = HY_POLICY_API;
  module.exports.decide = HY_POLICY_API;
  module.exports.updateBot = HY_POLICY_API;
}

/* ESM builds append: export default HY_POLICY_API;  (see build.mjs) */

export default HY_POLICY_API;
export const decide = HY_POLICY_API;
export const act = HY_POLICY_API;
export const policy = HY_POLICY_API;
export const updateBot = HY_POLICY_API;
export const name = "hy";
