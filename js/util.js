/* ===========================================================
   util.js — 常量、数学、随机数、绘图助手
   =========================================================== */

window.TT = window.TT || {};

TT.CFG = {
  W: 1280,
  H: 800,
  CELL: 80,
  COLS: 16,
  ROWS: 10,

  WALL_THICK: 10,

  MAX_BULLETS_PER_TANK: 5,

  TANK: {
    radius: 16,
    speed: 178,          // 最大前进速度 px/s
    reverseRatio: 0.66,  // 倒车速度比例
    accel: 920,
    brake: 1150,
    turn: 3.3,           // rad/s
    fireCooldown: 0.42,
  },

  BULLET: {
    speed: 620,
    radius: 3.2,
    bounces: 5,
    life: 6,
    spawnGrace: 0.07,    // 出生后短暂不判定自伤
  },

  ROUND_INTRO: 1.15,
  ROUND_OUTRO: 2.25,
  ROUND_TIME_LIMIT: 60,  // 正式游戏、Arena 与训练共用的交战时限（秒）

  /* 两张固定地图，按回合交替 */
  MAP_SEEDS: [20260101, 88888888],
};

/* 原版 TT 的扁平配色：车体 / 履带 / 炮塔，全部纯色 + 黑描边 */
TT.COLORS = [
  { key: 'p1', name: 'P1', main: '#EF3B2D', body: '#C0271F', tread: '#8A1913' },
  { key: 'p2', name: 'P2', main: '#2E8BE8', body: '#1F6BC0', tread: '#144C8C' },
  { key: 'p3', name: 'P3', main: '#3CC44F', body: '#2A9B3C', tread: '#1B6E28' },
  { key: 'p4', name: 'P4', main: '#F5C518', body: '#D9A30C', tread: '#9E7505' },
  { key: 'p5', name: 'P5', main: '#A855F7', body: '#7E3AC4', tread: '#5A278F' },
  { key: 'p6', name: 'P6', main: '#FF8A3D', body: '#D9661F', tread: '#9E4713' },
  { key: 'p7', name: 'P7', main: '#22D3EE', body: '#0E9DB5', tread: '#076D7E' },
  { key: 'p8', name: 'P8', main: '#F472B6', body: '#C4408A', tread: '#8E2A62' },
];

TT.PALETTE = {
  floor: '#E6E6E6',
  wall: '#3A3A3A',
  wallEdge: '#141414',
  bullet: '#000000',
  ink: '#141414',
};

const U = TT.util = {};

U.TAU = Math.PI * 2;

U.clamp = function (v, a, b) { return v < a ? a : (v > b ? b : v); };
U.lerp = function (a, b, t) { return a + (b - a) * t; };
U.sign = function (v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); };

U.dist = function (ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  return Math.sqrt(dx * dx + dy * dy);
};

U.dist2 = function (ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  return dx * dx + dy * dy;
};

/* 返回 a → b 的最短角度差，范围 (-PI, PI] */
U.angleDiff = function (a, b) {
  let d = (b - a + Math.PI) % U.TAU;
  if (d < 0) d += U.TAU;
  return d - Math.PI;
};

/* 角度插值（走最短路径） */
U.angleTowards = function (from, to, maxStep) {
  const d = U.angleDiff(from, to);
  if (Math.abs(d) <= maxStep) return to;
  return from + U.sign(d) * maxStep;
};

/* mulberry32 —— 可复现的伪随机 */
U.rng = function (seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

U.randRange = function (rnd, a, b) { return a + rnd() * (b - a); };
U.randInt = function (rnd, a, b) { return Math.floor(a + rnd() * (b - a + 1)); };
U.pick = function (rnd, arr) { return arr[Math.floor(rnd() * arr.length)]; };

/* 圆角矩形路径 */
U.rrect = function (ctx, x, y, w, h, r) {
  const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.arcTo(x + w, y, x + w, y + rr, rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.arcTo(x + w, y + h, x + w - rr, y + h, rr);
  ctx.lineTo(x + rr, y + h);
  ctx.arcTo(x, y + h, x, y + h - rr, rr);
  ctx.lineTo(x, y + rr);
  ctx.arcTo(x, y, x + rr, y, rr);
  ctx.closePath();
};

/* 十六进制颜色 → rgba() */
U.rgba = function (hex, alpha) {
  if (hex.charAt(0) !== '#') return hex;
  const h = hex.slice(1);
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
  return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
};

/* 混色：t=0 取 a，t=1 取 b */
U.mix = function (a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ar = (pa >> 16) & 255, ag = (pa >> 8) & 255, ab = pa & 255;
  const br = (pb >> 16) & 255, bg = (pb >> 8) & 255, bb = pb & 255;
  const r = Math.round(ar + (br - ar) * t);
  const g = Math.round(ag + (bg - ag) * t);
  const bl = Math.round(ab + (bb - ab) * t);
  return 'rgb(' + r + ',' + g + ',' + bl + ')';
};

/* 线段与圆求交，返回最近命中参数 t ∈ [0,1]，否则 -1 */
U.segCircle = function (x0, y0, x1, y1, cx, cy, r) {
  const dx = x1 - x0, dy = y1 - y0;
  const fx = x0 - cx, fy = y0 - cy;
  const a = dx * dx + dy * dy;
  if (a < 1e-9) return (fx * fx + fy * fy <= r * r) ? 0 : -1;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;
  if (c <= 0) return 0;
  let disc = b * b - 4 * a * c;
  if (disc < 0) return -1;
  disc = Math.sqrt(disc);
  const t1 = (-b - disc) / (2 * a);
  const t2 = (-b + disc) / (2 * a);
  if (t1 >= 0 && t1 <= 1) return t1;
  if (t2 >= 0 && t2 <= 1) return t2;
  if (t1 < 0 && t2 > 1) return 0;
  return -1;
};

/* 射线 vs AABB（方向向量不归一化，t ∈ [0,1]） */
U.rayBox = function (ox, oy, dx, dy, minx, miny, maxx, maxy) {
  let tmin = 0, tmax = 1;
  let nx = 0, ny = 0;

  if (Math.abs(dx) < 1e-9) {
    if (ox < minx || ox > maxx) return null;
  } else {
    const inv = 1 / dx;
    let t1 = (minx - ox) * inv;
    let t2 = (maxx - ox) * inv;
    let s = -1;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; s = 1; }
    if (t1 > tmin) { tmin = t1; nx = s; ny = 0; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return null;
  }

  if (Math.abs(dy) < 1e-9) {
    if (oy < miny || oy > maxy) return null;
  } else {
    const inv = 1 / dy;
    let t1 = (miny - oy) * inv;
    let t2 = (maxy - oy) * inv;
    let s = -1;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; s = 1; }
    if (t1 > tmin) { tmin = t1; nx = 0; ny = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return null;
  }

  return { t: tmin, nx: nx, ny: ny };
};
