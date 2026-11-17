(function () {
/* ===========================================================
   minimax policy — Tank Trouble Arena
   ---------------------------------------------------------------
   一个"会算弹道"的坦克 AI。

   与普通 AI 的差别（也是这版能赢的关键）：
   1) 弹道仿真：完全复刻 entities.js 里 Bullet.update 的几何，
      包括"每弹一次衰减 0.7px""最多弹 5 次""炮口在墙里直接哑火"。
      开火前先把整条弹道跑一遍，确认：
        - 第一个撞到的是目标
        - 子弹不会绕回来打自己
        - 炮口没顶在墙里
      → 自杀 0 次（内置 AI 平均每辆自杀 50+ 次）
   2) 闪避：把场上所有来弹（含自己反弹回来的弹）按时间轴推演，
      枚举 17 种"转向 + 油门"组合，选出净空最大的一种。
      子弹 620px/s、我 178px/s，被瞄准了能靠侧移活下来。
   3) 卡位：不是无脑冲脸，而是 BFS 找"离目标有直瞄、且在 165~430px
      安全距离带、且不在别的敌人射界里"的格子，走格子路径过去。
   4) 反弹冷枪：镜像法算出打墙折射的冷枪角度，只在常规瞄准
      打不到时启用。
   5) 打完立刻横移，不给对方补枪窗口。
   6) 谁朝我开火，我就先打谁（子弹带 owner，能直接锁定射手）。

   ---------------------------------------------------------------
   对外接口（几种写法都支持）：
     import policy, { updateBot, createPolicy, decidePolicy } from './policy.js'
       policy(tank, game, dt) -> { drive, steer, shoot }
     window.TTPolicy.updateBot(tank, game, dt)   （classic script 版）
     policy.update(tank, game, dt)

   参数说明见 README.md。
   =========================================================== */

const TAU = Math.PI * 2;
const G = globalThis;

/* ---------------------------------------------------------
   1. 配置（优先用游戏自己的 TT.CFG，缺失时退回默认值）
   --------------------------------------------------------- */
const DEF = {
  W: 1280, H: 800, COLS: 16, ROWS: 10, CELL: 80,
  MAX_BULLETS_PER_TANK: 5,
  TANK: { radius: 16, speed: 178, reverseRatio: 0.66, accel: 920, brake: 1150, turn: 3.3, fireCooldown: 0.42 },
  BULLET: { speed: 620, radius: 3.2, bounces: 5, life: 6, spawnGrace: 0.07 },
};

let _cfgSrc = null;
let C = DEF;

function cfg() {
  const src = (G.TT && G.TT.CFG) || null;
  if (src !== _cfgSrc) {
    _cfgSrc = src;
    C = {
      ...DEF, ...(src || {}),
      TANK: { ...DEF.TANK, ...((src && src.TANK) || {}) },
      BULLET: { ...DEF.BULLET, ...((src && src.BULLET) || {}) },
    };
  }
  return C;
}

/* ---------------------------------------------------------
   2. 战术参数（可用 configure() 覆盖，方便调参）
   --------------------------------------------------------- */
const DEFAULT_P = {
  aimTol: 0.050,        // 开火角度容差（弧度）
  standoffMin: 120,     // 理想交火距离下限
  standoffMax: 320,     // 理想交火距离上限
  standoffIdeal: 210,
  dodgeHorizon: 1.20,   // 闪避推演时长（秒）
  dodgeSafe: 34,        // 认为"安全"的净空
  dodgeWant: 40,        // 想达到的净空（够宽才值得换方向）
  dodgeGain: 12,        // 躲避比"原地不动"净空多出多少才值得换方向（px）
  maxLiveBullets: 3,    // 自己允许的在场子弹数
  planInterval: 0.12,   // 射击方案刷新间隔
  pathInterval: 0.28,   // 路径刷新间隔
  dangerInterval: 0.08, // 威胁刷新间隔
  losInterval: 0.16,    // 视线关系刷新间隔
  cellInterval: 1.00,   // 卡位/安全格搜索间隔
  displaceTime: 0.80,   // 开火后横移时长
  bankRange: 420,       // 冷枪考虑的最大墙距
  bankWalls: 10,        // 每个目标最多试几面墙
  planCandidates: 4,    // 每次最多仿真几个候选射击角
  traceHorizon: 2.4,    // 射击验证的弹道推演时长（秒）
  aimMove: 0,           // 瞄准时是否保持移动（0=站定 1=被瞄准时移动 2=始终移动）
  crossRetreat: 2,      // 场上几发敌方子弹在飞就考虑撤（0=关闭）
  leadFactor: 0.94,     // 预判提前量系数（1=完全按当前速度外推）
  rageAfter: 25,        // 回合超过多少秒开始加大压上力度（0=关闭）
  blockRule: 2,         // 敌人能不能替我挡枪：0=不信 1=按当前速度 2=要打中心且早 0.15s
  blockLead: 0.15,      // blockRule=2 时，敌人至少要早多少秒命中才算它挡得住（秒）
  selfMargin: 0,        // 自伤判定的额外余量（px）。0 = 严格按真实命中半径
  guardMode: 1,         // 自伤判定：0=只按"当前位置" 1=按"我移动后的位置"
  dodgeGain: 24,        // 躲避比"原地不动"净空多出多少才值得换方向（px，0=关闭此规则）
  dodgeWins: 1,         // 1=闪避优先于开火后横移  0=横移覆盖闪避
};

/* 实际生效的参数（可被 configure 覆盖） */
const P = Object.assign({}, DEFAULT_P);

/* 运行时覆盖参数（供基准测试 sweep 使用）
   注意：configure 是"合并"语义，只覆盖传进来的键。
   想回到出厂默认值请先调 resetConfig()。 */
function configure(overrides) {
  if (overrides) Object.assign(P, overrides);
  return Object.assign({}, P);
}

/* 恢复出厂默认参数 */
function resetConfig() {
  Object.assign(P, DEFAULT_P);
  return Object.assign({}, P);
}

/* ---------------------------------------------------------
   3. 数学
   --------------------------------------------------------- */
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const sign = (v) => (v < 0 ? -1 : v > 0 ? 1 : 0);

function angleDiff(a, b) {
  let d = (b - a + Math.PI) % TAU;
  if (d < 0) d += TAU;
  return d - Math.PI;
}

/* 线段与圆求交（与 util.segCircle 同算法），返回 t∈[0,1] 或 -1 */
function segCircle(x0, y0, x1, y1, cx, cy, r) {
  const dx = x1 - x0, dy = y1 - y0;
  const fx = x0 - cx, fy = y0 - cy;
  const a = dx * dx + dy * dy;
  if (a < 1e-9) return (fx * fx + fy * fy <= r * r) ? 0 : -1;
  const b = 2 * (fx * dx + fy * dy);
  const c = fx * fx + fy * fy - r * r;
  let disc = b * b - 4 * a * c;
  if (disc < 0) return -1;
  disc = Math.sqrt(disc);
  const t1 = (-b - disc) / (2 * a);
  const t2 = (-b + disc) / (2 * a);
  if (t1 >= 0 && t1 <= 1) return t1;
  if (t2 >= 0 && t2 <= 1) return t2;
  if (t1 < 0 && t2 > 1) return 0;
  return -1;
}

/* 点到线段的最短距离 */
function pointSegDist(px, py, x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0;
  const len2 = dx * dx + dy * dy;
  let t = 0;
  if (len2 > 1e-9) t = clamp(((px - x0) * dx + (py - y0) * dy) / len2, 0, 1);
  const qx = x0 + dx * t, qy = y0 + dy * t;
  return Math.hypot(px - qx, py - qy);
}

/* ---------------------------------------------------------
   4. 网格：把地图解析成"单元格边界墙"图
   --------------------------------------------------------- */
const gridCache = new WeakMap();

function getGrid(map) {
  if (!map) return null;
  if (gridCache.has(map)) return gridCache.get(map);

  const cols = map.cols || cfg().COLS;
  const rows = map.rows || cfg().ROWS;
  const cell = map.cell || cfg().CELL;
  const VC = cols + 1;
  const g = { cols, rows, cell, VC };

  const ok = map.vFlag && map.hFlag &&
    map.vFlag.length >= rows * VC && map.hFlag.length >= (rows + 1) * cols;

  if (ok) {
    g.v = map.vFlag;
    g.h = map.hFlag;
  } else {
    /* 没有 vFlag/hFlag 时，用 solidAt 在边界中点采样反推 */
    const v = new Uint8Array(rows * VC);
    const h = new Uint8Array((rows + 1) * cols);
    if (typeof map.solidAt === 'function') {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c <= cols; c++) {
          v[r * VC + c] = map.solidAt(c * cell, r * cell + cell / 2) ? 1 : 0;
        }
      }
      for (let r = 0; r <= rows; r++) {
        for (let c = 0; c < cols; c++) {
          h[r * cols + c] = map.solidAt(c * cell + cell / 2, r * cell) ? 1 : 0;
        }
      }
    }
    g.v = v;
    g.h = h;
  }

  gridCache.set(map, g);
  return g;
}

/* 竖墙：在 x = c*cell 上，分隔 (r,c-1) 与 (r,c) */
function vWall(g, r, c) { return g.v[r * g.VC + c] ? 1 : 0; }
/* 横墙：在 y = r*cell 上，分隔 (r-1,c) 与 (r,c) */
function hWall(g, r, c) { return g.h[r * g.cols + c] ? 1 : 0; }

function cellOf(g, x, y) {
  const c = clamp(Math.floor(x / g.cell), 0, g.cols - 1);
  const r = clamp(Math.floor(y / g.cell), 0, g.rows - 1);
  return r * g.cols + c;
}
function cellCenter(g, idx) {
  const c = idx % g.cols, r = (idx / g.cols) | 0;
  return { x: c * g.cell + g.cell / 2, y: r * g.cell + g.cell / 2 };
}

/* 从 (x0,y0) 走到 (x1,y1) 会不会撞墙（单元格边界判定） */
function moveBlocked(g, x0, y0, x1, y1) {
  const c0 = clamp(Math.floor(x0 / g.cell), 0, g.cols - 1);
  const r0 = clamp(Math.floor(y0 / g.cell), 0, g.rows - 1);
  const c1 = clamp(Math.floor(x1 / g.cell), 0, g.cols - 1);
  const r1 = clamp(Math.floor(y1 / g.cell), 0, g.rows - 1);
  if (c1 === c0 && r1 === r0) return false;
  if (r1 === r0) return vWall(g, r0, Math.max(c0, c1)) === 1;
  if (c1 === c0) return hWall(g, Math.max(r0, r1), c0) === 1;
  return vWall(g, r0, Math.max(c0, c1)) === 1 || hWall(g, Math.max(r0, r1), c0) === 1;
}

/* BFS 距离场（从 start 出发，返回每格的步数，-1 不可达） */
function distField(g, start) {
  const total = g.cols * g.rows;
  const dist = new Int32Array(total).fill(-1);
  if (start < 0 || start >= total) return dist;
  const q = new Int32Array(total);
  let head = 0, tail = 0;
  q[tail++] = start;
  dist[start] = 0;
  while (head < tail) {
    const cur = q[head++];
    const r = (cur / g.cols) | 0;
    const c = cur % g.cols;
    const d = dist[cur] + 1;
    if (c < g.cols - 1 && !vWall(g, r, c + 1) && dist[cur + 1] < 0) { dist[cur + 1] = d; q[tail++] = cur + 1; }
    if (c > 0 && !vWall(g, r, c) && dist[cur - 1] < 0) { dist[cur - 1] = d; q[tail++] = cur - 1; }
    if (r < g.rows - 1 && !hWall(g, r + 1, c) && dist[cur + g.cols] < 0) { dist[cur + g.cols] = d; q[tail++] = cur + g.cols; }
    if (r > 0 && !hWall(g, r, c) && dist[cur - g.cols] < 0) { dist[cur - g.cols] = d; q[tail++] = cur - g.cols; }
  }
  return dist;
}

/* 用距离场反推路径（从 goal 往 start 回溯，返回航点数组） */
function pathFromField(g, field, start, goal) {
  if (start === goal) return [];
  const path = [];
  let cur = goal;
  let guard = 0;
  while (cur !== start && guard++ < 400) {
    const r = (cur / g.cols) | 0;
    const c = cur % g.cols;
    let best = -1, bestD = field[cur];
    if (c < g.cols - 1 && !vWall(g, r, c + 1) && field[cur + 1] >= 0 && field[cur + 1] < bestD) { bestD = field[cur + 1]; best = cur + 1; }
    if (c > 0 && !vWall(g, r, c) && field[cur - 1] >= 0 && field[cur - 1] < bestD) { bestD = field[cur - 1]; best = cur - 1; }
    if (r < g.rows - 1 && !hWall(g, r + 1, c) && field[cur + g.cols] >= 0 && field[cur + g.cols] < bestD) { bestD = field[cur + g.cols]; best = cur + g.cols; }
    if (r > 0 && !hWall(g, r, c) && field[cur - g.cols] >= 0 && field[cur - g.cols] < bestD) { bestD = field[cur - g.cols]; best = cur - g.cols; }
    if (best < 0) break;
    path.push(cellCenter(g, best));
    cur = best;
  }
  path.reverse();
  return path;
}

/* ---------------------------------------------------------
   5. 加速射线
   ------------------------------------------------------------
   游戏自带的 map.raycast 每次都要遍历全部墙体（约 70 面），
   而我的策略每帧要做几百次射线 → 78% 的 CPU 都耗在这里。
   这里把墙按格子分桶，射线只测它真正穿过的那些格子，
   结果与 map.raycast 完全一致（同样的矩形、同样的 rayBox）。
   --------------------------------------------------------- */

/* 与 util.rayBox 同算法：方向向量不必归一化，t ∈ [0,1] */
function rayBox(ox, oy, dx, dy, minx, miny, maxx, maxy) {
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
}

const rayCache = new WeakMap();

function getRaycaster(map) {
  if (!map) return null;
  if (rayCache.has(map)) return rayCache.get(map);

  const walls = map.walls;
  const cols = map.cols || cfg().COLS;
  const rows = map.rows || cfg().ROWS;
  const cell = map.cell || cfg().CELL;

  let fn;
  if (walls && walls.length) {
    /* 按格子分桶 */
    const buckets = new Array(cols * rows);
    for (let i = 0; i < walls.length; i++) {
      const w = walls[i];
      const c0 = clamp(Math.floor(w.x / cell), 0, cols - 1);
      const c1 = clamp(Math.floor((w.x + w.w - 1e-6) / cell), 0, cols - 1);
      const r0 = clamp(Math.floor(w.y / cell), 0, rows - 1);
      const r1 = clamp(Math.floor((w.y + w.h - 1e-6) / cell), 0, rows - 1);
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const k = r * cols + c;
          if (!buckets[k]) buckets[k] = [];
          buckets[k].push(i);
        }
      }
    }

    fn = function (x0, y0, x1, y1) {
      const dx = x1 - x0, dy = y1 - y0;
      if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return null;

      let r = clamp(Math.floor(y0 / cell), 0, rows - 1);
      let c = clamp(Math.floor(x0 / cell), 0, cols - 1);
      const stepX = dx > 0 ? 1 : -1;
      const stepY = dy > 0 ? 1 : -1;

      let tMaxX = Infinity, tDeltaX = Infinity;
      let tMaxY = Infinity, tDeltaY = Infinity;
      if (Math.abs(dx) > 1e-9) {
        tDeltaX = Math.abs(cell / dx);
        const nextX = dx > 0 ? (c + 1) * cell : c * cell;
        tMaxX = (nextX - x0) / dx;
      }
      if (Math.abs(dy) > 1e-9) {
        tDeltaY = Math.abs(cell / dy);
        const nextY = dy > 0 ? (r + 1) * cell : r * cell;
        tMaxY = (nextY - y0) / dy;
      }

      let best = null;
      for (let iter = 0; iter < 256; iter++) {
        const b = buckets[r * cols + c];
        if (b) {
          for (let k = 0; k < b.length; k++) {
            const w = walls[b[k]];
            const hit = rayBox(x0, y0, dx, dy, w.x, w.y, w.x + w.w, w.y + w.h);
            if (hit && (!best || hit.t < best.t)) best = hit;
          }
        }
        const tNext = tMaxX < tMaxY ? tMaxX : tMaxY;
        if (best && best.t <= tNext) break;
        if (tMaxX < tMaxY) {
          c += stepX; tMaxX += tDeltaX;
          if (c < 0 || c >= cols) break;
        } else {
          r += stepY; tMaxY += tDeltaY;
          if (r < 0 || r >= rows) break;
        }
      }
      return best;
    };
  } else if (typeof map.raycast === 'function') {
    fn = (x0, y0, x1, y1) => map.raycast(x0, y0, x1, y1);
  } else {
    fn = () => null;
  }

  rayCache.set(map, fn);
  return fn;
}

/* ---------------------------------------------------------
   6. 视线
   --------------------------------------------------------- */
function losClear(map, x0, y0, x1, y1) {
  const rc = getRaycaster(map);
  if (!rc) return true;
  return !rc(x0, y0, x1, y1);
}

/* 车宽级别的"能直瞄"：中心 + 左右各偏移一条，任一条通即可（与 map.hasClearShot 同思路） */
function clearShot(map, x0, y0, x1, y1, pad) {
  const rc = getRaycaster(map);
  if (!rc) return true;
  const p = pad == null ? 16 : pad;
  const dx = x1 - x0, dy = y1 - y0;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;
  if (!rc(x0, y0, x1, y1)) return true;
  if (!rc(x0 + nx * p, y0 + ny * p, x1 + nx * p, y1 + ny * p)) return true;
  if (!rc(x0 - nx * p, y0 - ny * p, x1 - nx * p, y1 - ny * p)) return true;
  return false;
}

/* ---------------------------------------------------------
   6. 弹道仿真（严格复刻 entities.js Bullet.update）
   --------------------------------------------------------- */
/**
 * @returns { steps:[{x,y,t}], hit:number|-1, dead:bool, bounces:number,
 *            hits:[{index,t,speed}], ownerTs:number[] }
 *   hit = 第一个被命中的坦克 index（-1 表示一直没打中）
 *   ignoreIndex: 仿真时忽略某辆坦克（闪避推演时忽略自己）
 *   fullPath:    true 时撞到坦克也继续飞完整条弹道。
 *                必须这样才能发现"子弹先擦过别人（对方一移动就漏掉）、
 *                继续弹几面墙再绕回来打自己"这种情况 —— 只按当前位置
 *                判定会把大部分自杀漏过去（实测漏了 9 成）。
 *   selfPaths:   我自己在这段时间里的几种可能走位（每个是 {x,y,t} 数组）。
 *                给了就用"我未来所在的位置"判自伤，而不是原地不动的圆。
 *                ownerTs[i] = 第 i 种走位下，子弹扫到我的时刻。
 *   selfIndex:  "要保护谁"（默认 = ownerIdx）。闪避推演时子弹主人是别人，
 *                但要防的是我自己，所以这个参数和 ownerIdx 必须分开。
 */
function traceBullet(map, tanks, ownerIdx, x0, y0, angle, horizon, dt, ignoreIndex, fullPath, selfPaths, selfIndex) {
  dt = dt || 1 / 60;
  const B = cfg().BULLET;
  const segR = cfg().TANK.radius + B.radius;
  if (selfIndex == null) selfIndex = ownerIdx;

  let x = x0, y = y0, ang = angle, bounces = 0, t = 0;
  const steps = [{ x, y, t: 0 }];
  const hits = [];
  let hit = -1;
  let dead = false;

  const paths = [];
  if (selfPaths) {
    if (Array.isArray(selfPaths)) {
      for (let i = 0; i < selfPaths.length; i++) {
        if (selfPaths[i] && selfPaths[i].length) paths.push(selfPaths[i]);
      }
    } else if (selfPaths.length) paths.push(selfPaths);
  }
  const ownerTs = paths.length ? paths.map(() => Infinity) : [Infinity];
  const staticSelf = paths.length === 0;
  /* 只有"自己是这颗子弹的主人"时才有出生保护 */
  const selfGrace = (selfIndex === ownerIdx);

  const frames = Math.min(400, Math.max(1, Math.ceil(horizon / dt)));
  const list = tanks || [];
  const rc = getRaycaster(map);

  for (let f = 0; f < frames && (fullPath || hit < 0); f++) {
    const fx = x, fy = y;
    let remaining = B.speed * dt;
    let guard = 0;
    let died = false;

    while (remaining > 0.02 && guard++ < 8) {
      const dx = Math.cos(ang) * remaining;
      const dy = Math.sin(ang) * remaining;
      const h = rc(x, y, x + dx, y + dy);

      if (!h) { x += dx; y += dy; break; }

      const tt = clamp(h.t, 0, 1);
      x += dx * tt;
      y += dy * tt;
      remaining -= remaining * tt;

      const nx = h.nx || 0, ny = h.ny || 0;
      if (nx === 0 && ny === 0) { died = true; break; }   // 炮口顶在墙里 → 哑火

      let vx = Math.cos(ang), vy = Math.sin(ang);
      const dot = vx * nx + vy * ny;
      if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; }
      ang = Math.atan2(vy, vx);

      x += nx * 0.8;
      y += ny * 0.8;

      bounces++;
      if (bounces > B.bounces) { died = true; break; }
      remaining -= 0.7;
    }

    t += dt;
    steps.push({ x, y, t });

    if (died) { dead = true; break; }

    /* 坦克碰撞：与 game.js 一致（先移动后判定，命中即死，顺序按数组） */
    for (let i = 0; i < list.length; i++) {
      const tk = list[i];
      if (!tk.alive) continue;
      if (ignoreIndex != null && tk.index === ignoreIndex) continue;
      if (ownerIdx === tk.index && t < B.spawnGrace) continue;
      if (segCircle(fx, fy, x, y, tk.x, tk.y, segR) >= 0) {
        if (hit < 0) hit = tk.index;
        hits.push({
          index: tk.index, t,
          speed: Math.abs(tk.speed || 0),
          d: pointSegDist(tk.x, tk.y, fx, fy, x, y),   /* 打在敌人身上的偏离度 */
        });
        if (!fullPath) break;                    // 游戏里子弹撞到坦克就死了
      }
    }

    /* 自伤判定：看我"那时候"在哪，而不是看我现在在哪 */
    if (!selfGrace || t >= B.spawnGrace) {
      const selfR = segR + (P.selfMargin || 0);
      if (staticSelf) {
        for (let i = 0; i < list.length; i++) {
          const tk = list[i];
          if (tk.index !== selfIndex || !tk.alive) continue;
          if (segCircle(fx, fy, x, y, tk.x, tk.y, selfR) >= 0) { ownerTs[0] = t; break; }
        }
      } else {
        for (let p = 0; p < paths.length; p++) {
          if (ownerTs[p] !== Infinity) continue;
          const me = paths[p][Math.min(f + 1, paths[p].length - 1)];
          if (!me) continue;
          if (segCircle(fx, fy, x, y, me.x, me.y, selfR) >= 0) ownerTs[p] = t;
        }
      }
    }
  }

  return { steps, hit, dead, bounces, hits, ownerTs, ownerIndex: ownerIdx, selfIndex };
}

/* ---------------------------------------------------------
   7. 自己的运动仿真（用于闪避选动作）
   --------------------------------------------------------- */
function simMotion(g, tank, targetAngle, drive, horizon, dt) {
  const T = cfg().TANK;
  let x = tank.x, y = tank.y, a = tank.angle, sp = tank.speed;
  const out = [{ x, y, t: 0 }];
  const steps = Math.max(1, Math.ceil(horizon / dt));
  const maxStep = Math.max(1e-4, T.turn * dt);
  let t = 0;

  for (let i = 0; i < steps; i++) {
    const d = angleDiff(a, targetAngle);
    /* 比例转向：和真实操控一致（满舵会转过头，模拟就不准了） */
    const steer = clamp(d / maxStep, -1, 1);
    a += steer * T.turn * dt;
    if (a > Math.PI) a -= TAU; else if (a < -Math.PI) a += TAU;

    const max = T.speed * (drive < 0 ? T.reverseRatio : 1);
    const tgt = drive > 0 ? max : drive < 0 ? -max : 0;
    const rate = (sp * drive < 0) ? T.brake : T.accel;
    if (drive !== 0) {
      if (sp < tgt) sp = Math.min(tgt, sp + rate * dt);
      else if (sp > tgt) sp = Math.max(tgt, sp - rate * dt);
    } else {
      const drop = T.brake * 0.75 * dt;
      if (Math.abs(sp) <= drop) sp = 0; else sp -= sign(sp) * drop;
    }

    const nx = x + Math.cos(a) * sp * dt;
    const ny = y + Math.sin(a) * sp * dt;
    if (!moveBlocked(g, x, y, nx, ny)) { x = nx; y = ny; }
    else sp *= 0.3;

    t += dt;
    out.push({ x, y, t });
  }
  return out;
}

/* 我的轨迹与某条弹道的最小净空 */
function pathClearance(path, trace) {
  let clear = Infinity;
  const n = Math.min(path.length, trace.steps.length);
  for (let i = 0; i < n; i++) {
    const a = trace.steps[i];
    const b = trace.steps[Math.min(i + 1, trace.steps.length - 1)];
    const d = pointSegDist(path[i].x, path[i].y, a.x, a.y, b.x, b.y);
    if (d < clear) clear = d;
    if (clear === 0) break;
  }
  return clear;
}

/* ---------------------------------------------------------
   7b. 预测"我自己"接下来会到哪
   ---------------------------------------------------------
   开火之后我并不会原地不动：会转向、会前进、会后撤，
   而且开火后还会"横移"一小段时间（转 90° 躲开对方补枪）。
   子弹绕一圈回来时，撞上的是"我移动之后的位置"。
   所以判断自伤必须用运动轨迹，不能用一个静止的圆。
   seg = 依次执行的机动段 [{ang, drive, dur}]，dur 单位秒。 */
function projectPath(g, tank, seg, horizon, dt) {
  const T = cfg().TANK;
  let x = tank.x, y = tank.y, a = tank.angle, sp = tank.speed;
  const out = [{ x, y, t: 0 }];
  const total = Math.max(1, Math.ceil(horizon / dt));
  const maxStep = Math.max(1e-4, T.turn * dt);
  let t = 0, si = 0, segEnd = 0;

  for (let i = 0; i < total; i++) {
    while (si < seg.length - 1 && t >= segEnd) {
      si++;
      segEnd = seg[si].dur != null ? segEnd + seg[si].dur : horizon;
    }
    const s = seg[si] || { ang: a, drive: 0 };
    const drive = s.drive;

    const d = angleDiff(a, s.ang);
    a += clamp(d / maxStep, -1, 1) * T.turn * dt;
    if (a > Math.PI) a -= TAU; else if (a < -Math.PI) a += TAU;

    const max = T.speed * (drive < 0 ? T.reverseRatio : 1);
    const tgt = drive > 0 ? max : drive < 0 ? -max : 0;
    const rate = (sp * drive < 0) ? T.brake : T.accel;
    if (drive !== 0) {
      if (sp < tgt) sp = Math.min(tgt, sp + rate * dt);
      else if (sp > tgt) sp = Math.max(tgt, sp - rate * dt);
    } else {
      const drop = T.brake * 0.75 * dt;
      if (Math.abs(sp) <= drop) sp = 0; else sp -= sign(sp) * drop;
    }

    const nx = x + Math.cos(a) * sp * dt;
    const ny = y + Math.sin(a) * sp * dt;
    if (!moveBlocked(g, x, y, nx, ny)) { x = nx; y = ny; }
    else sp *= 0.3;

    t += dt;
    out.push({ x, y, t });
  }
  return out;
}

/* 开火后的真实走位：先横移一小段（转 90° 躲开对方补枪），再回到常规机动 */
function projectAfterShot(g, tank, planAng, targetAng, side, drive, horizon, dt) {
  const segs = [];
  if (P.displaceTime > 0 && targetAng != null) {
    segs.push({ ang: targetAng + Math.PI / 2 * side, drive: 1, dur: P.displaceTime });
    segs.push({ ang: planAng, drive, dur: horizon });
  } else {
    segs.push({ ang: planAng, drive, dur: horizon });
  }
  return projectPath(g, tank, segs, horizon, dt);
}

/* 这一枪会不会要我的命？
   tr  = 完整弹道（撞到坦克也不停）
   idx = 用第几种走位来判断（对应 tr.ownerTs 的下标）
   判据：子弹先扫到我（而不是先被某个敌人挡住）。
   挡住这件事很不可靠 —— 对方会动，擦边的一枪多半会漏。
   所以只有"打得实、而且早得多"才算它能替我挡。 */
function selfShotFatal(tr, idx) {
  if (!tr || !tr.ownerTs || idx >= tr.ownerTs.length) return false;
  const selfT = tr.ownerTs[idx];
  if (selfT === Infinity) return false;
  const R = segRadius();

  for (let i = 0; i < tr.hits.length; i++) {
    const h = tr.hits[i];
    if (h.t >= selfT) break;
    if (h.index === tr.selfIndex) continue;

    if (P.blockRule === 0) break;                       /* 谁都不信，只认自己会死 */
    if (P.blockRule === 1) {
      /* 按对方当前速度估算它能不能躲开（老做法，偏乐观） */
      const disp = h.speed * (selfT - h.t);
      if (disp < R * 0.9) return false;
    } else {
      /* 严格：必须打在对方正中心附近，而且早出 0.15s 以上，才算它能替我挡 */
      if (h.d <= R * 0.5 && (selfT - h.t) >= P.blockLead) return false;
    }
  }
  return true;
}

function segRadius() {
  return cfg().TANK.radius + cfg().BULLET.radius;
}

/* 弹道折线到某个点的最近距离（判断"会不会从这里穿过去"） */
function pathDistToPoint(steps, px, py) {
  let best = Infinity;
  for (let i = 0; i < steps.length - 1; i++) {
    const d = pointSegDist(px, py, steps[i].x, steps[i].y, steps[i + 1].x, steps[i + 1].y);
    if (d < best) best = d;
    if (best === 0) break;
  }
  return best;
}

/* ---------------------------------------------------------
   8. Brain（每辆坦克一份状态）
   --------------------------------------------------------- */
const brains = new WeakMap();

function newBrain() {
  return {
    plan: null, planTimer: 0,
    path: null, pathTimer: 0, pathGoal: -1,
    danger: null, dangerTimer: 0, threatOwner: -1, threatCount: 0,
    losTimer: 0, losMap: null,
    cellsTimer: 0, attackCell: -1, safeCell: -1,
    losCells: null, losCellTarget: -1,
    displace: 0, displaceSide: 1,
    shots: 0, kills: 0, selfHits: 0,
    mapRef: null, roundStart: 0,
    stuckTimer: 0, stuckRef: null, stuckCount: 0,
    recover: 0, recoverDir: 1,
  };
}

function getBrain(tank) {
  let b = brains.get(tank);
  if (!b) { b = newBrain(); brains.set(tank, b); }
  return b;
}

/* ---------------------------------------------------------
   9. 瞄准预判
   --------------------------------------------------------- */
function predictEnemy(tank, e) {
  const B = cfg().BULLET;
  const vx = Math.cos(e.angle) * e.speed;
  const vy = Math.sin(e.angle) * e.speed;
  const lf = P.leadFactor;
  let flight = Math.hypot(e.x - tank.x, e.y - tank.y) / B.speed;
  let px = e.x, py = e.y;
  for (let i = 0; i < 2; i++) {
    px = e.x + vx * flight * lf;
    py = e.y + vy * flight * lf;
    flight = Math.hypot(px - tank.x, py - tank.y) / B.speed;
  }
  return { x: px, y: py, flight, vx, vy };
}

function barrelTip(tank) {
  const r = cfg().TANK.radius + 12;
  return { x: tank.x + Math.cos(tank.angle) * r, y: tank.y + Math.sin(tank.angle) * r };
}

/* ---------------------------------------------------------
   10. 威胁评估（闪避）
   --------------------------------------------------------- */
function assessThreat(brain, tank, game, dt) {
  brain.dangerTimer -= dt;
  if (brain.dangerTimer > 0) return brain.danger;
  brain.dangerTimer = P.dangerInterval;

  const map = game.map;
  const bullets = game.bullets || [];
  const traces = [];
  const owners = new Set();
  let owner = -1;
  let nearest = Infinity;

  for (let i = 0; i < bullets.length; i++) {
    const b = bullets[i];
    if (!b || b.dead) continue;
    if (b.owner === tank.index && b.grace > 0) continue;

    const dx = b.x - tank.x, dy = b.y - tank.y;
    const dist = Math.hypot(dx, dy);
    if (dist > 760) continue;

    const vx = Math.cos(b.angle), vy = Math.sin(b.angle);
    const along = dx * vx + dy * vy;
    if (along < -60) continue;                       // 在我身后
    const perp = Math.abs(dx * -vy + dy * vx);
    if (perp > 240 && dist > 300) continue;          // 明显打不到

    /* 推演这颗子弹的去向。
       注意 ignoreIndex 不能填自己：填了就等于"假装自己不在"，
       自己的子弹绕回来打自己也测不出来（实测漏了九成自杀）。
       selfIndex 单独指定"要保护谁"，和子弹主人分开。 */
    const tr = traceBullet(map, game.tanks, b.owner, b.x, b.y, b.angle,
      P.dodgeHorizon, 1 / 60, null, false, null, tank.index);
    const reachesMe = (tr.ownerTs[0] !== Infinity);
    if (reachesMe || (tr.hit < 0 && tr.steps.length > 1)) {
      traces.push(tr);
      owners.add(b.owner);
      if (dist < nearest) { nearest = dist; owner = b.owner; }
    }
  }

  brain.threatOwner = owner;
  brain.threatCount = owners.size;

  if (!traces.length) { brain.danger = null; return null; }

  /* 枚举动作，选净空最大的 */
  const g = getGrid(map);
  const base = tank.angle;
  const offs = [0, 0.30, -0.30, 0.62, -0.62, 0.95, -0.95, 1.30, -1.30,
    1.75, -1.75, 2.20, -2.20, 2.70, -2.70, Math.PI];
  const cands = [];
  for (let i = 0; i < offs.length; i++) cands.push({ ang: base + offs[i], drive: 1 });
  cands.push({ ang: base, drive: 0 });
  cands.push({ ang: base, drive: -1 });

  /* "什么都不改"的净空 —— 拿它当基准线 */
  const stayPath = simMotion(g, tank, base, 0, P.dodgeHorizon, 1 / 60);
  let stay = Infinity;
  for (let k = 0; k < traces.length; k++) {
    const d = pathClearance(stayPath, traces[k]);
    if (d < stay) stay = d;
  }

  let best = null;
  for (let i = 0; i < cands.length; i++) {
    const act = cands[i];
    const path = simMotion(g, tank, act.ang, act.drive, P.dodgeHorizon, 1 / 60);
    let clear = Infinity;
    for (let k = 0; k < traces.length; k++) {
      const d = pathClearance(path, traces[k]);
      if (d < clear) clear = d;
    }
    if (!best || clear > best.clear) best = { ang: act.ang, drive: act.drive, clear };
  }

  /* 关键：拿"最优躲避"和"原地不动"比。
     早先只判断"最优躲避的净空 < 阈值"，于是只要最优躲避能到 50px 净空，
     策略就宁可站着瞄准也不躲 —— 然后被打中。 */
  best.stay = stay;
  best.gain = best.clear - stay;

  brain.danger = best;
  return best;
}

/* ---------------------------------------------------------
   11. 射击方案
   --------------------------------------------------------- */
/* 冷枪（打墙折射）：镜像法快速筛，只验证第一跳 */
function bankCandidates(tip, targetPt, walls) {
  const out = [];
  if (!walls || !walls.length) return out;
  const near = [];
  for (let i = 0; i < walls.length; i++) {
    const w = walls[i];
    const cx = w.x + w.w / 2, cy = w.y + w.h / 2;
    const d = Math.hypot(cx - targetPt.x, cy - targetPt.y);
    if (d > P.bankRange) continue;
    near.push({ w, d, cx, cy });
  }
  near.sort((a, b) => a.d - b.d);

  for (let i = 0; i < near.length && i < P.bankWalls; i++) {
    const w = near[i].w;
    const cx = near[i].cx, cy = near[i].cy;
    if (w.w < w.h) {
      /* 竖墙：把目标关于 x=cx 镜像 */
      out.push({ x: 2 * cx - targetPt.x, y: targetPt.y, wall: w, vertical: true });
    }
    if (w.h < w.w) {
      /* 横墙：把目标关于 y=cy 镜像 */
      out.push({ x: targetPt.x, y: 2 * cy - targetPt.y, wall: w, vertical: false });
    }
  }
  return out;
}

/* 校验镜像角：第一跳是否真打到对应墙，反射后是否扫过目标 */
function bankValid(map, tip, cand, targetPt, radius) {
  const rc = getRaycaster(map);
  if (!rc) return false;
  const dx = cand.x - tip.x, dy = cand.y - tip.y;
  const len = Math.hypot(dx, dy);
  if (len < 1) return false;
  const ang = Math.atan2(dy, dx);
  const reach = len * 1.25;
  const h = rc(tip.x, tip.y, tip.x + Math.cos(ang) * reach, tip.y + Math.sin(ang) * reach);
  if (!h) return false;

  const hx = tip.x + Math.cos(ang) * reach * h.t;
  const hy = tip.y + Math.sin(ang) * reach * h.t;
  const nx = h.nx || 0, ny = h.ny || 0;
  if (nx === 0 && ny === 0) return false;

  /* 命中点必须落在这面墙上 */
  const w = cand.wall;
  const onWall = cand.vertical
    ? (hx >= w.x - 2 && hx <= w.x + w.w + 2 && hy >= w.y - 2 && hy <= w.y + w.h + 2)
    : (hy >= w.y - 2 && hy <= w.y + w.h + 2 && hx >= w.x - 2 && hx <= w.x + w.w + 2);
  if (!onWall) return false;

  let vx = Math.cos(ang), vy = Math.sin(ang);
  const dot = vx * nx + vy * ny;
  if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; }
  const t2 = segCircle(hx, hy, hx + vx * 4000, hy + vy * 4000, targetPt.x, targetPt.y, radius);
  return t2 >= 0;
}

function updateFirePlan(brain, tank, game, dt) {
  brain.planTimer -= dt;
  if (brain.planTimer > 0) return brain.plan;
  brain.planTimer = P.planInterval;

  const map = game.map;
  const enemies = [];
  for (let i = 0; i < game.tanks.length; i++) {
    const t = game.tanks[i];
    if (t.alive && t.index !== tank.index) enemies.push(t);
  }
  if (!enemies.length) { brain.plan = null; return null; }

  /* 目标已经死了 → 方案作废 */
  if (brain.plan) {
    let still = false;
    for (let i = 0; i < enemies.length; i++) if (enemies[i].index === brain.plan.target) still = true;
    if (!still) { brain.plan = null; brain.planTimer = 0; }
  }

  const tip = barrelTip(tank);
  const segR = cfg().TANK.radius + cfg().BULLET.radius;
  const cands = [];

  /* ---- 直接瞄准 ---- */
  for (let i = 0; i < enemies.length; i++) {
    const e = enemies[i];
    const pr = predictEnemy(tank, e);
    const d = Math.hypot(pr.x - tip.x, pr.y - tip.y);
    if (d < 26) continue;                                  // 太近贴脸，交给闪避
    if (!clearShot(map, tip.x, tip.y, pr.x, pr.y, 15)) continue;
    cands.push({ ang: Math.atan2(pr.y - tip.y, pr.x - tip.x), target: e.index, kind: 'direct', score: 3.0 - d / 900, pr });
  }

  /* ---- 冷枪（只在直瞄不成时用） ---- */
  if (!cands.length) {
    for (let i = 0; i < enemies.length; i++) {
      const e = enemies[i];
      const pr = predictEnemy(tank, e);
      const mirrors = bankCandidates(tip, pr, map.walls);
      for (let m = 0; m < mirrors.length; m++) {
        const mc = mirrors[m];
        if (!bankValid(map, tip, mc, pr, segR)) continue;
        cands.push({
          ang: Math.atan2(mc.y - tip.y, mc.x - tip.x),
          target: e.index, kind: 'bank',
          score: 1.2 - Math.hypot(mc.x - tip.x, mc.y - tip.y) / 1200, pr,
        });
      }
      if (cands.length >= 5) break;
    }
  }

  if (!cands.length) { brain.plan = null; return null; }

  /* ---- 只仿真最有希望的前几个：会不会先打到别人 / 会不会打回自己 ---- */
  cands.sort((a, b) => b.score - a.score);
  const limit = Math.min(cands.length, Math.max(1, P.planCandidates | 0));
  const HIT_R = cfg().TANK.radius + cfg().BULLET.radius;
  const g = getGrid(map);
  let best = null;

  for (let i = 0; i < limit; i++) {
    const c = cands[i];
    /* 三种走位都推一遍：站定 / 前进 / 后退。
       只有"怎么走都会被打死"这一枪才彻底放弃 —— 否则留给开火那一刻按实际走位再判一次。 */
    const targetAng = Math.atan2(c.pr.y - tank.y, c.pr.x - tank.x);
    const nextSide = (brain.displaceSide === 1) ? -1 : 1;
    const walks = P.guardMode
      ? [
        projectAfterShot(g, tank, c.ang, targetAng, nextSide, 0, P.traceHorizon, 1 / 60),
        projectAfterShot(g, tank, c.ang, targetAng, nextSide, 1, P.traceHorizon, 1 / 60),
        projectAfterShot(g, tank, c.ang, targetAng, nextSide, -1, P.traceHorizon, 1 / 60),
      ]
      : null;
    const tr = traceBullet(map, game.tanks, tank.index, tip.x, tip.y, c.ang,
      P.traceHorizon, 1 / 60, null, true, walks);
    if (walks) {
      if (selfShotFatal(tr, 0) && selfShotFatal(tr, 1) && selfShotFatal(tr, 2)) continue;
    } else if (tr.ownerTs[0] !== Infinity) continue;

    /* 目标判定要看"预测位置"：我瞄的是它飞过来时的位置，
       如果拿当前位置去卡，就会因为目标在动而一枪都不开。 */
    const dPred = pathDistToPoint(tr.steps, c.pr.x, c.pr.y);
    const hitNow = (tr.hit === c.target);
    const hitLater = (dPred <= HIT_R + 3);
    const hitOther = (tr.hit >= 0 && tr.hit !== tank.index);
    if (!hitNow && !hitLater && !hitOther) continue;         // 这一枪打不到任何人

    const sc = c.score + (c.kind === 'direct' ? 1.5 : 0) + (hitNow ? 0.6 : 0);
    if (!best || sc > best.score) {
      best = {
        ang: c.ang, target: c.target, kind: c.kind, score: sc,
        flight: c.pr.flight, dist: Math.hypot(c.pr.x - tip.x, c.pr.y - tip.y),
      };
    }
  }

  brain.plan = best;
  return best;
}

/* ---------------------------------------------------------
   12. 视线关系缓存（谁能看见我 / 我能看见谁）
   --------------------------------------------------------- */
function updateLos(brain, tank, game, enemies, map, dt) {
  brain.losTimer -= dt;
  if (brain.losTimer > 0 && brain.losMap) return brain.losMap;
  brain.losTimer = P.losInterval;

  const m = new Map();
  for (let i = 0; i < enemies.length; i++) {
    const e = enemies[i];
    m.set(e.index, {
      seeMe: clearShot(map, e.x, e.y, tank.x, tank.y, 15),
      iSee: clearShot(map, tank.x, tank.y, e.x, e.y, 15),
      dist: Math.hypot(e.x - tank.x, e.y - tank.y),
    });
  }
  brain.losMap = m;
  return m;
}

/* ---------------------------------------------------------
   13. 目标选择
   --------------------------------------------------------- */
function pickTarget(brain, tank, game, enemies, map, losMap) {
  if (!enemies.length) return null;
  let best = null;

  for (let i = 0; i < enemies.length; i++) {
    const e = enemies[i];
    const info = losMap.get(e.index) || { seeMe: false, dist: 9999 };
    let sc = 100 - info.dist / 22;

    /* 对方能看见我 → 优先解决威胁 */
    if (info.seeMe) sc += 30;

    /* 谁朝我开火，先打谁 */
    if (brain.threatOwner === e.index) sc += 55;

    /* 孤立目标加分，人堆里扎堆扣分 */
    let allies = 0;
    for (let k = 0; k < enemies.length; k++) {
      if (k === i) continue;
      const o = enemies[k];
      const od = Math.hypot(o.x - e.x, o.y - e.y);
      if (od < 260) allies += 1 - od / 260;
    }
    sc -= allies * 26;

    /* 静止目标好打 */
    sc += (18 - e.speed) * 0.35;

    if (!best || sc > best.sc) best = { e, sc, dist: info.dist };
  }
  return best ? best.e : null;
}

/* ---------------------------------------------------------
   14. 卡位：找"能直瞄目标、距离在安全带、且不在别人射界里"的格子
   --------------------------------------------------------- */
function updateCellPlan(brain, tank, game, target, map, grid) {
  brain.cellsTimer -= P.planInterval;
  if (brain.cellsTimer > 0) return;
  brain.cellsTimer = P.cellInterval;

  const enemies = [];
  for (let i = 0; i < game.tanks.length; i++) {
    const t = game.tanks[i];
    if (t.alive && t.index !== tank.index) enemies.push(t);
  }

  /* --- 有直瞄目标的格子集合 --- */
  if (target) {
    const tc = cellOf(grid, target.x, target.y);
    if (tc !== brain.losCellTarget || !brain.losCells) {
      brain.losCellTarget = tc;
      const cells = [];
      for (let i = 0; i < grid.cols * grid.rows; i++) {
        const p = cellCenter(grid, i);
        if (clearShot(map, p.x, p.y, target.x, target.y, 15)) cells.push(i);
      }
      brain.losCells = cells;
    }
  } else {
    brain.losCells = null;
    brain.losCellTarget = -1;
  }

  /* --- 每个格子被几个敌人看见 --- */
  const myCell = cellOf(grid, tank.x, tank.y);
  const field = distField(grid, myCell);
  const total = grid.cols * grid.rows;
  const exposure = new Float32Array(total);
  for (let k = 0; k < enemies.length; k++) {
    const e = enemies[k];
    for (let i = 0; i < total; i++) {
      const p = cellCenter(grid, i);
      if (losClear(map, e.x, e.y, p.x, p.y)) exposure[i] += 1;
    }
  }

  /* --- 进攻格：能直瞄目标 + 距离在安全带 + 没人瞄得着我 + 离我近 --- */
  let attack = -1, attackScore = -Infinity;
  const cells = brain.losCells || [];
  for (let i = 0; i < cells.length; i++) {
    const idx = cells[i];
    const p = cellCenter(grid, idx);
    const d = Math.hypot(p.x - target.x, p.y - target.y);
    if (d < P.standoffMin || d > P.standoffMax) continue;
    const cost = field[idx] < 0 ? 999 : field[idx];
    if (cost >= 999) continue;
    let sc = 40 - Math.abs(d - P.standoffIdeal) / 9 - cost * 1.6 - exposure[idx] * 30;
    for (let k = 0; k < enemies.length; k++) {
      const e = enemies[k];
      const od = Math.hypot(p.x - e.x, p.y - e.y);
      if (od < 280) sc -= (280 - od) / 9;
    }
    if (sc > attackScore) { attackScore = sc; attack = idx; }
  }
  brain.attackCell = attack;

  /* --- 安全格（完全没人看得见的格子）--- */
  let safe = -1, safeCost = Infinity;
  for (let i = 0; i < total; i++) {
    if (exposure[i] > 0) continue;
    const cost = field[i] < 0 ? 999 : field[i];
    if (cost < safeCost) { safeCost = cost; safe = i; }
  }
  brain.safeCell = safe;
  brain.exposure = exposure;
}

/* ---------------------------------------------------------
   15. 路径跟随
   --------------------------------------------------------- */
function refreshPath(brain, tank, grid, goalCell) {
  const myCell = cellOf(grid, tank.x, tank.y);
  const field = distField(grid, myCell);
  brain.path = pathFromField(grid, field, myCell, goalCell);
  brain.pathTimer = P.pathInterval;
  brain.pathGoal = goalCell;
}

/* 跟随路径：丢掉已到达的航点，挑最远且能直视的航点 */
function followPath(brain, tank, map, grid) {
  if (!brain.path || !brain.path.length) return null;
  while (brain.path.length &&
    Math.hypot(brain.path[0].x - tank.x, brain.path[0].y - tank.y) < 36) {
    brain.path.shift();
  }
  if (!brain.path.length) return null;
  let idx = 0;
  const maxAhead = Math.min(brain.path.length - 1, 5);
  for (let i = maxAhead; i >= 0; i--) {
    if (losClear(map, tank.x, tank.y, brain.path[i].x, brain.path[i].y)) { idx = i; break; }
  }
  return brain.path[idx];
}

/* ---------------------------------------------------------
   16. 主决策
   --------------------------------------------------------- */
function decide(tank, game, dt) {
  const map = game.map;
  const grid = getGrid(map);
  const brain = getBrain(tank);
  const T = cfg().TANK;

  if (brain.displace > 0) brain.displace -= dt;

  const enemies = [];
  for (let i = 0; i < game.tanks.length; i++) {
    const t = game.tanks[i];
    if (t.alive && t.index !== tank.index) enemies.push(t);
  }

  const out = { drive: 0, steer: 0, shoot: false };
  if (!map || !grid) return out;

  /* 回合计时：僵持太久就加大压上力度，别让回合拖成平局 */
  if (brain.mapRef !== map) {
    brain.mapRef = map;
    brain.roundStart = game.time || 0;
  }
  const roundAge = (game.time || 0) - brain.roundStart;
  const rage = P.rageAfter > 0 && roundAge > P.rageAfter;

  /* ---------- 16.1 视线关系 / 威胁 / 射击方案 ---------- */
  const losMap = updateLos(brain, tank, game, enemies, map, dt);
  const danger = assessThreat(brain, tank, game, dt);
  const plan = updateFirePlan(brain, tank, game, dt);
  const target = pickTarget(brain, tank, game, enemies, map, losMap);

  if (target) updateCellPlan(brain, tank, game, target, map, grid);

  /* ---------- 16.2 我被几个人盯着 ---------- */
  let watchers = 0;
  for (let i = 0; i < enemies.length; i++) {
    const info = losMap.get(enemies[i].index);
    if (info && info.seeMe) watchers++;
  }
  /* 同时还有几发子弹正朝我来（交叉火力躲不开，要撤） */
  const shooters = (brain.threatCount || 0) + (watchers >= 2 ? 1 : 0);

  /* ---------- 16.3 决定期望朝向 ---------- */
  let wantAng = tank.angle;
  let drive = 0;
  let mode = 'hold';

  if (danger && danger.clear < P.dodgeWant) {
    /* 净空本来就危险 → 必须躲 */
    mode = 'dodge';
    wantAng = danger.ang;
    drive = danger.drive;
    if (danger.clear < P.dodgeSafe) drive = 1;
  } else if (danger && P.dodgeGain > 0 && danger.gain > P.dodgeGain) {
    /* 原地不动躲不掉，但换个方向能躲开 → 也得躲。
       （只看"最优躲避的净空是否够大"会误判：能躲到 50px 净空时
        策略反而选择继续瞄准，然后被打中。） */
    mode = 'dodge';
    wantAng = danger.ang;
    drive = danger.drive;
  } else if (P.crossRetreat > 0 && shooters >= P.crossRetreat && brain.safeCell >= 0) {
    /* 被交叉火力／多人围瞄：先撤到没人看得见的格子 */
    mode = 'retreat';
    brain.pathTimer -= dt;
    if (brain.pathGoal !== brain.safeCell || brain.pathTimer <= 0 || !brain.path) {
      refreshPath(brain, tank, grid, brain.safeCell);
    }
    const wp = followPath(brain, tank, map, grid);
    if (wp) wantAng = Math.atan2(wp.y - tank.y, wp.x - tank.x);
    else wantAng = Math.atan2(C.H / 2 - tank.y, C.W / 2 - tank.x);
    drive = 1;
  } else if (plan) {
    /* 有射击方案：转向瞄准角 */
    mode = 'aim';
    wantAng = plan.ang;
    const d = target ? Math.hypot(target.x - tank.x, target.y - tank.y) : 300;
    /* 僵持久了就往死里压，缩短距离带 */
    const bandMin = rage ? P.standoffMin * 0.55 : P.standoffMin;
    const bandMax = rage ? P.standoffMax * 0.65 : P.standoffMax;
    if (d > bandMax) drive = 1;                   // 太远，边转边压上
    else if (d < bandMin) drive = -1;             // 太近，边转边后撤
    else if (P.aimMove === 2) drive = 1;         // 始终保持移动
    else if (P.aimMove === 1 && watchers >= 1) drive = 1;   // 被瞄准时别站定
    else drive = 0;                              // 距离带内：站定打准
  } else if (target) {
    /* 没得打：去卡好的位置 */
    mode = 'advance';
    let goal = brain.attackCell;
    if (goal < 0) {
      /* 没有好卡位：直接贴过去，但保持中距离 */
      goal = cellOf(grid, target.x, target.y);
    }
    brain.pathTimer -= dt;
    if (brain.pathGoal !== goal || brain.pathTimer <= 0 || !brain.path) {
      refreshPath(brain, tank, grid, goal);
    }
    const wp = followPath(brain, tank, map, grid);
    if (wp) {
      wantAng = Math.atan2(wp.y - tank.y, wp.x - tank.x);
    } else if (target && Math.hypot(target.x - tank.x, target.y - tank.y) < 95) {
      /* 已经贴得很近：直接顶上去 */
      wantAng = Math.atan2(target.y - tank.y, target.x - tank.x);
    } else {
      /* 目标所在格走不到（可能卡在死角）：往地图中心转悠，别愣着 */
      wantAng = Math.atan2(C.H / 2 - tank.y, C.W / 2 - tank.x);
    }
    drive = 1;
  } else {
    mode = 'hold';
  }

  /* ---------- 16.4 开火后横移 ---------- */
  /* 注意：闪避优先级更高。有子弹要打我的时候不能还在那儿横移 ——
     自家子弹绕回来的时候，横移反而容易一头撞进去。 */
  if (brain.displace > 0 && target && (mode !== 'dodge' || !P.dodgeWins)) {
    const base = Math.atan2(target.y - tank.y, target.x - tank.x);
    wantAng = base + Math.PI / 2 * brain.displaceSide;
    drive = 1;
    mode = 'displace';
  }

  /* ---------- 16.5 转向 & 油门 ---------- */
  const diff = angleDiff(tank.angle, wantAng);
  /* 比例转向：满舵 3.3rad/s × dt(1/60) = 0.055rad/帧，
     比开火窗口（0.03~0.05rad）还大 → 永远对不准，会左右振荡。
     所以按"这一帧需要转多少"给舵量。 */
  const maxStep = Math.max(1e-4, T.turn * (dt || 1 / 60));
  if (Math.abs(diff) > 0.004) {
    out.steer = clamp(diff / maxStep, -1, 1);
  }

  if (mode === 'hold') {
    drive = 0;
    if (Math.abs(diff) > 1.9) drive = -1;          // 目标在身后：倒车转身
  } else if (mode === 'aim') {
    /* 油门在 16.3 已按距离带/被瞄准与否决定 */
    if (Math.abs(diff) > 1.9) drive = -1;
  } else if (mode === 'displace' || mode === 'dodge') {
    if (drive === 0) drive = 1;
    if (Math.abs(diff) > 1.9 && mode !== 'dodge') drive = -1;
  } else if (mode === 'advance' || mode === 'retreat') {
    if (Math.abs(diff) > 1.9) drive = -1;
    else if (Math.abs(diff) > 0.55) drive = 0.5;   // 转向时半油门
    else drive = 1;
  }

  /* 前方堵墙：侧滑 */
  if (drive > 0) {
    const fx = tank.x + Math.cos(tank.angle) * (T.radius + 16);
    const fy = tank.y + Math.sin(tank.angle) * (T.radius + 16);
    if (map.solidAt(fx, fy)) {
      const side = sign(angleDiff(tank.angle, wantAng)) || 1;
      const sx = tank.x + Math.cos(tank.angle + side * 1.1) * (T.radius + 16);
      const sy = tank.y + Math.sin(tank.angle + side * 1.1) * (T.radius + 16);
      if (!map.solidAt(sx, sy)) { out.steer = side; drive = 0.6; }
      else drive = 0;
    }
  }

  /* 别撞上其他坦克 */
  for (let i = 0; i < game.tanks.length; i++) {
    const o = game.tanks[i];
    if (o === tank || !o.alive) continue;
    const dx = o.x - tank.x, dy = o.y - tank.y;
    const d = Math.hypot(dx, dy);
    if (d < T.radius * 2 + 6) {
      const front = Math.cos(tank.angle) * dx + Math.sin(tank.angle) * dy;
      if (front > 0 && drive > 0) drive = 0;
      break;
    }
  }

  /* ---------- 16.6 卡住自救（看位移不看速度：顶墙时速度仍有 40~50） ---------- */
  brain.stuckTimer = (brain.stuckTimer || 0) - dt;
  brain.stuckRef = brain.stuckRef || { x: tank.x, y: tank.y };
  if (brain.stuckTimer <= 0) {
    const moved = Math.hypot(tank.x - brain.stuckRef.x, tank.y - brain.stuckRef.y);
    brain.stuckRef.x = tank.x;
    brain.stuckRef.y = tank.y;
    brain.stuckTimer = 0.7;
    if (moved < 18 && drive !== 0) {
      brain.stuckCount = (brain.stuckCount || 0) + 1;
      if (brain.stuckCount >= 2) {
        brain.stuckCount = 0;
        brain.recover = 0.55;
        brain.recoverDir = -brain.recoverDir;
        brain.path = null;
        brain.pathTimer = 0;
      }
    } else {
      brain.stuckCount = 0;
    }
  }
  if (brain.recover > 0) {
    brain.recover -= dt;
    out.steer = brain.recoverDir;
    drive = -1;
  }

  /* ---------- 16.6 开火判定 ---------- */
  if (plan && tank.cooldown <= 0 && mode !== 'dodge') {
    const err = Math.abs(angleDiff(tank.angle, plan.ang));
    const tol = plan.kind === 'direct' ? P.aimTol : P.aimTol * 0.9;
    if (err < tol) {
      const tip = barrelTip(tank);
      let live = 0;
      for (let i = 0; i < game.bullets.length; i++) {
        const b = game.bullets[i];
        if (!b.dead && b.owner === tank.index) live++;
      }
      if (live < P.maxLiveBullets) {
        /* 现场再仿真一次这一枪（用真实炮口与当前朝向） */
        const HIT_R = cfg().TANK.radius + cfg().BULLET.radius;
        /* 按"我这一帧真的会怎么走"来预测自己的未来位置：
           开火后我还会转向/进退/横移，子弹绕回来时撞的是移动后的我。 */
        let tgtPos = null;
        for (let i = 0; i < game.tanks.length; i++) {
          if (game.tanks[i].index === plan.target && game.tanks[i].alive) { tgtPos = game.tanks[i]; break; }
        }
        const targetAng = tgtPos ? Math.atan2(tgtPos.y - tank.y, tgtPos.x - tank.x) : null;
        const nextSide = (brain.displaceSide === 1) ? -1 : 1;
        const walk = P.guardMode
          ? projectAfterShot(getGrid(map), tank, plan.ang, targetAng, nextSide, drive, P.traceHorizon, 1 / 60)
          : null;
        const tr = traceBullet(map, game.tanks, tank.index, tip.x, tip.y, tank.angle,
          P.traceHorizon, 1 / 60, null, true, walk ? [walk] : null);
        const muzzleClear = !map.solidAt(tip.x, tip.y);
        const selfHit = walk ? selfShotFatal(tr, 0) : (tr.ownerTs[0] !== Infinity);

        /* 目标判定用"预测位置"——瞄的就是它飞过来时所在的地方 */
        let dPred = Infinity;
        if (tgtPos) {
          const pr = predictEnemy(tank, tgtPos);
          dPred = pathDistToPoint(tr.steps, pr.x, pr.y);
        }
        const hitsEnemy = (tr.hit >= 0 && tr.hit !== tank.index);
        const goodShot = (dPred <= HIT_R + 3) || hitsEnemy;

        if (muzzleClear && !selfHit && goodShot) {
          out.shoot = true;
          brain.shots++;
          brain.displace = P.displaceTime;
          brain.displaceSide = nextSide;
        }
      }
    }
  }

  out.drive = drive;
  return out;
}

/* ---------------------------------------------------------
   17. 对外接口
   --------------------------------------------------------- */
/* 归一化各种可能的调用形态 */
function normalize(args) {
  const a0 = args[0], a1 = args[1], a2 = args[2];
  if (!a0) return null;

  /* 形态 A：updateBot(tank, game, dt) */
  if (typeof a0.x === 'number' && typeof a0.angle === 'number') {
    return { tank: a0, game: a1, dt: typeof a2 === 'number' ? a2 : 1 / 60 };
  }
  /* 形态 B：view = { tank|self|me, tanks, bullets, map, dt } */
  const tank = a0.tank || a0.self || a0.me;
  if (tank && typeof tank.x === 'number') {
    const game = (a1 && a1.tanks) ? a1 : (a0.game || a0);
    const dt = (typeof a2 === 'number') ? a2 : (a0.dt || (game && game.dt) || 1 / 60);
    return { tank, game, dt };
  }
  return null;
}

function decidePolicy(tank, game, dt) {
  const n = normalize([tank, game, dt]);
  if (!n) return { drive: 0, steer: 0, shoot: false, fire: false };
  try {
    const r = decide(n.tank, n.game, n.dt) || { drive: 0, steer: 0, shoot: false };
    r.shoot = !!r.shoot;
    r.drive = r.drive || 0;
    r.steer = r.steer || 0;
    r.fire = r.shoot;                 // 别名字段，兼容不同 Arena
    return r;
  } catch (e) {
    if (G.console && console.warn) console.warn('[minimax policy] 异常：', e);
    return { drive: 0, steer: 0, shoot: false, fire: false };
  }
}

/* 与游戏内置 TT.updateBot 同签名 */
function updateBot(tank, game, dt) {
  return decidePolicy(tank, game, dt);
}

/* 工厂式接口：每个坦克一份独立状态 */
function createPolicy() {
  return {
    reset() {},
    update(tank, game, dt) { return decidePolicy(tank, game, dt); },
    decide(tank, game, dt) { return decidePolicy(tank, game, dt); },
  };
}



/* 内部工具导出（供自测使用，不影响 Arena 调用） */

/* 浏览器 classic script 场景下的全局出口 */
if (G) {
  G.TTPolicy = G.TTPolicy || {
    updateBot, decide: decidePolicy, createPolicy, configure, resetConfig, policy: updateBot,
  };
}

/* ---------------- classic script 版：挂到 window 上（arena/build-vendor.mjs 生成） ---------------- */
if (typeof window !== 'undefined') {
  window.TTPolicy = {
    updateBot: updateBot,
    decide: decidePolicy,
    createPolicy: createPolicy,
    configure: configure,
    resetConfig: resetConfig,
    policy: updateBot,
  };
}
})();
