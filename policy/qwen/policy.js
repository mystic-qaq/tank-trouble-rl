/* ===========================================================
   Tank Trouble · Qwen Arena Policy
   -----------------------------------------------------------
   入口（与游戏内置 bot 同签名，可直接替换 TT.updateBot）：
       update(tank, game, dt) -> { drive: -1..1, steer: -1..1, shoot: bool }

   为什么这样设计（来自对真实规则的分析与无头 Arena 实测）：

   1. 回合唯一胜负条件 = 「最后一个活着的」。击杀只是手段。
      内置 bot 群里 23% 的死是自己子弹反弹打死的 —— 只要零自杀，
      就等于白捡一大截胜率。所以第一优先级永远是不死。
   2. 开火前用与游戏同一套 raycast 反射物理完整模拟弹道
      （反弹 5 次 / 6 秒寿命 / 0.8px 法向偏移 / 每弹扣 0.7px 能量），
      只在「解析扫掠判定确定命中」且「全程不可能弹回自己身上」时扣扳机。
   3. 履带车不能侧移，炮管=车体朝向。所以用「倒车风筝」：
      倒车时炮管依然锁着目标，距离却在拉开 —— 对手的提前量全部打飞。
   4. 掩体后的敌人用墙面镜像求入射点，打反弹弹（bank shot）。
   5. 躲避是几何题：枚举 9 种机动，模拟各自 1 秒后的轨迹，
      选出让「来袭弹到我的最近距离」最大的那个，而不是无脑乱转。
   6. 走位用地图模型打分：单元格互相可见性（暴露度）、
      全局共享的子弹走廊热力、BFS 全对最短路、贴墙与退路。
   7. 阶段机：潜伏 → 清扫 → 单挑，随回合时长升级，绝不僵住挂机。
   =========================================================== */

(function (root) {
  'use strict';

  var VERSION = '1.2.0';

  /* =========================================================
     0. 数学
     ========================================================= */

  var PI = Math.PI;
  var TAU = PI * 2;

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function sgn(v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }
  function len2(x, y) { return x * x + y * y; }

  function angDiff(from, to) {
    var d = (to - from + PI) % TAU;
    if (d < 0) d += TAU;
    return d - PI;
  }

  /* =========================================================
     1. 可调参数（sim/bench.mjs 实测调校，可外部覆盖）
     ========================================================= */

  var P = {
    /* 节奏 */
    planHz: 15,             // 重规划频率
    dodgeHz: 22,            // 规避解算频率（其余帧沿用上次机动）

    /* 开火 */
    fireDelay: 0.06,        // 两发最小间隔
    maxLive: 3,             // 自己在场子弹上限（少流弹 = 少自杀）
    minShotScore: 46,       // 射击评分门槛
    spamPenalty: 16,        // 场上自己子弹多时的扣分
    missPenalty: 30,        // 打空的扣分（暴露位置 + 留下流弹）
    killBonus: 128,         // 命中奖励
    punishBonus: 30,        // 对手还在开火冷却里 → 这一枪特别安全

    /* 自伤防护 */
    selfMargin: 15,         // 弹道离我的最小时间相关余量（杀伤半径之外）
    selfHorizon: 2.2,       // 自伤检查窗口
    hitHorizon: 1.9,        // 命中检查窗口
    pocketR: 80,            // 「子弹在我周围乱弹」半径
    pocketW: 2.4,           // 上述情况的扣分权重（按采样数）
    nearRiskW: 2.6,         // 距离余量不足的扣分权重

    /* 交火几何 */
    wantRange: 210,         // 理想交火距离
    closeRange: 135,        // 近于此就倒车风筝
    farRange: 300,          // 远于此才推进
    duelRange: 120,         // 单挑时更近，命中率高
    danceRange: 330,        // 对手看到且近于此 → 立刻进入对枪姿态

    /* 开枪—侧移—回身（人类的核心节奏） */
    orbitArc: 1.35,         // 侧移阶段相对对手方向的角度
    runTime: 0.34,          // 侧移持续
    aimHoldMax: 0.45,       // 锁定太久还开不了枪就先走位

    /* 走位权重 */
    exposureW: 9.5,         // 被敌人直视的格子数
    visFree: 1,             // 允许被几个敌人同时直视
    visPunish: 15,          // 超出部分的硬惩罚
    dangerW: 13,            // 子弹走廊热力
    approachW: 4.2,         // 向猎杀目标推进
    crowdW: 0.09,           // 离敌人过近的惩罚
    coverW: 1.6,            // 贴墙（开口少）加分
    exitW: 2.4,             // 有退路加分
    deadEndP: 14,           // 死格惩罚
    hyst: 5.5,              // 保持原目标
    reachW: 1.2,            // 舍近求远的惩罚

    /* 阶段 */
    huntAfter: 7,           // 这么多人头之后开始清扫
    escalate: 15,           // 回合这么久还没进展 → 主动找人打
    force: 32,              // 再不上就永远打不完（防僵局）
    pressureCap: 4,         // 附近弹多就先别浪

    /* 反僵局：长时间没人死就逐渐不要命 */
    staleBerserk: 20,       // 这么久没人死 → 升级
    forceBerserk: 45,       // 不管如何，这么久必须强攻
    staleDesperate: 45,     // 更久 → 放宽自伤余量，但绝不放弃底线
    desperateAge: 80,
    desperateMargin: 0.55,
    fuseAge: 75,            // 熔断：这一回合拖太久了，不惜代价也要收掉
    fuseStale: 60,
    fuseMargin: 0.45,       // 熔断后的余量下限（约 7px）——再低就是自杀式开枪
    berserkMargin: 0.7,     // 自伤余量打折
    badGoalPenalty: 60,     // 走不通的格子短期不再选
    badGoalTtl: 7,          // 该记忆的秒数
    suppressAfter: 12,      // 僵局到这个程度开始压制射击（打不给目标也逼走位）

    /* 规避 */
    dodgeGap: 30,           // 最近距离小于此才启动规避机动
    threatR: 620,           // 只关心这个半径内的子弹
    threatNearR: 170,       // 弹道擦过这个范围才算威胁
    snapRange: 430,         // 对手枪口指着我、且近于此 → 提前规避
    snapArc: 0.13,          // 指向我方向的容差（rad）
    preWeight: 16,          // 未开火威胁的折让（他可能不打我）
    wallPenalty: 14,        // 机动撞墙的扣分

    /* 开火精度：近距离不需要那么准（几何上允许更大张角） */
    arcMin: 0.022,
    arcMax: 0.115,
    arcK: 0.85,             // arc ≈ atan(killR*arcK / range)
  };

  /* Q = 当前生效参数（ Brain.update 入口按 brain 实例切换），
     这样同一局里可以让两个座位跑两套不同参数做 A/B 对比。 */
  var Q = P;

  function deriveParams(overrides) {
    if (!overrides) return P;
    var o = Object.create(P);          // 原型链 → 未覆盖的项始终跟随 P
    for (var k in overrides) if (Object.prototype.hasOwnProperty.call(overrides, k)) o[k] = overrides[k];
    return o;
  }

  /* =========================================================
     2. 弹道模拟（严格对齐游戏 Bullet.update）
     ========================================================= */

  function traceShot(map, x, y, angle, maxDist, maxBounce) {
    var pts = [{ x: x, y: y, d: 0 }];
    var dist = maxDist;
    var a = angle;
    var bounces = 0;
    var guard = 0;
    var end = 'life';

    while (dist > 1.5 && guard++ < 9) {
      var dx = Math.cos(a) * dist, dy = Math.sin(a) * dist;
      var hit = map.raycast(x, y, x + dx, y + dy);
      var last = pts[pts.length - 1];

      if (!hit) {
        pts.push({ x: x + dx, y: y + dy, d: last.d + dist });
        break;
      }

      var tt = clamp(hit.t, 0, 1);
      var seg = dist * tt;
      x += dx * tt; y += dy * tt;
      var cur = { x: x, y: y, d: last.d + seg };
      dist -= seg;

      var nx = hit.nx || 0, ny = hit.ny || 0;
      if (nx === 0 && ny === 0) {         // 炮口顶墙 = 哑火
        pts.push(cur);
        end = 'fizzle';
        break;
      }

      var vx = Math.cos(a), vy = Math.sin(a);
      var dot = vx * nx + vy * ny;
      if (dot < 0) {
        var rx = vx - 2 * dot * nx, ry = vy - 2 * dot * ny;
        a = Math.atan2(ry, rx);
      }
      x += nx * 0.8; y += ny * 0.8;
      cur = { x: x, y: y, d: cur.d + 0.8 };
      pts.push(cur);
      dist -= 0.8 + 0.7;

      bounces++;
      if (bounces > maxBounce) { end = 'expire'; break; }
    }
    return { pts: pts, bounces: bounces, end: end };
  }

  function pathAt(pts, d) {
    if (d <= 0) return pts[0];
    var last = pts[pts.length - 1];
    if (d >= last.d) return last;
    for (var i = 1; i < pts.length; i++) {
      if (pts[i].d >= d) {
        var a = pts[i - 1], b = pts[i];
        var span = b.d - a.d;
        var t = span > 1e-6 ? (d - a.d) / span : 0;
        return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
      }
    }
    return last;
  }

  /* 折线到点的最短距离是否小于 lim（lim 为「距离平方」） */
  function pathClear2(pts, x, y, lim2) {
    for (var i = 0; i < pts.length; i++) {
      var dx = pts[i].x - x, dy = pts[i].y - y;
      if (dx * dx + dy * dy > lim2) continue;   // 粗筛
      if (i === 0) continue;
      var ax = pts[i - 1].x, ay = pts[i - 1].y, bx = pts[i].x, by = pts[i].y;
      var ddx = bx - ax, ddy = by - ay;
      var l2 = ddx * ddx + ddy * ddy;
      var t = l2 > 1e-9 ? clamp(((x - ax) * ddx + (y - ay) * ddy) / l2, 0, 1) : 0;
      var qx = ax + ddx * t - x, qy = ay + ddy * t - y;
      if (qx * qx + qy * qy < lim2) return true;
    }
    return false;
  }
  var pathClear = pathClear2;

  /* =========================================================
     3. 地图模型（按地图缓存，一次算清）
     ========================================================= */

  var modelCache = new Map();

  function MapModel(map) {
    var cols = this.cols = map.cols;
    var rows = this.rows = map.rows;
    this.cell = map.cell;
    var n = this.n = cols * rows;
    this.map = map;

    /* 每格四个方向的开口：1=上 2=下 4=左 8=右 */
    var open = this.open = new Uint8Array(n);
    var deg = this.deg = new Uint8Array(n);
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        var i = r * cols + c;
        var o = 0;
        if (r > 0 && !map.hAt(r, c)) o |= 1;
        if (r < rows - 1 && !map.hAt(r + 1, c)) o |= 2;
        if (c > 0 && !map.vAt(r, c)) o |= 4;
        if (c < cols - 1 && !map.vAt(r, c + 1)) o |= 8;
        open[i] = o;
        deg[i] = (o & 1 ? 1 : 0) + (o & 2 ? 1 : 0) + (o & 4 ? 1 : 0) + (o & 8 ? 1 : 0);
      }
    }

    var cx = this.cx = new Float64Array(n);
    var cy = this.cy = new Float64Array(n);
    for (var k = 0; k < n; k++) {
      cx[k] = (k % cols) * this.cell + this.cell / 2;
      cy[k] = ((k / cols) | 0) * this.cell + this.cell / 2;
    }

    /* 格间互相可见性：惰性计算 + 记忆化。
       一次性算全对是 12800 次 raycast（约 130ms 卡帧），改成按需算。 */
    var los = this.los = new Uint8Array(n * n);
    los.fill(2);                       // 2 = 未计算
    for (var d0 = 0; d0 < n; d0++) los[d0 * n + d0] = 1;
    this.exposure = null;

    /* 全对 BFS：steps[s][t] = 格数，next[s][t] = 从 s 出发去 t 的第一格 */
    var steps = this.steps = new Int16Array(n * n);
    steps.fill(-1);
    var next = this.next = new Int16Array(n * n);
    next.fill(-1);
    var dist = new Int16Array(n);
    var parent = new Int16Array(n);
    var q = new Int32Array(n);

    for (var s = 0; s < n; s++) {
      dist.fill(-1); parent.fill(-1);
      var head = 0, tail = 0;
      dist[s] = 0; q[tail++] = s;
      while (head < tail) {
        var cur = q[head++];
        var o2 = open[cur];
        var d0 = dist[cur];
        if (o2 & 1) { var u1 = cur - cols; if (u1 >= 0 && dist[u1] < 0) { dist[u1] = d0 + 1; parent[u1] = cur; q[tail++] = u1; } }
        if (o2 & 2) { var u2 = cur + cols; if (u2 < n && dist[u2] < 0) { dist[u2] = d0 + 1; parent[u2] = cur; q[tail++] = u2; } }
        if (o2 & 4) { var u3 = cur - 1; if (u3 >= 0 && (cur % cols) > 0 && dist[u3] < 0) { dist[u3] = d0 + 1; parent[u3] = cur; q[tail++] = u3; } }
        if (o2 & 8) { var u4 = cur + 1; if (u4 < n && (cur % cols) < cols - 1 && dist[u4] < 0) { dist[u4] = d0 + 1; parent[u4] = cur; q[tail++] = u4; } }
      }
      steps[s * n + s] = 0;
      for (var t = 0; t < n; t++) {
        if (dist[t] <= 0) continue;
        steps[s * n + t] = dist[t];
        /* 回溯到距离为 1 的那格 = 从 s 出发走第一步要踏进的格子 */
        var cur2 = t, guard = 0;
        while (dist[cur2] > 1 && guard++ < n + 2) cur2 = parent[cur2];
        next[s * n + t] = cur2;
      }
    }

    /* 墙面（反弹弹的镜像面，全部轴对齐） */
    var faces = [];
    var wl = map.walls;
    for (var wi = 0; wi < wl.length; wi++) {
      var w = wl[wi];
      if (w.w > 3) {
        faces.push({ v: false, y: w.y, x0: w.x, x1: w.x + w.w });
        faces.push({ v: false, y: w.y + w.h, x0: w.x, x1: w.x + w.w });
      }
      if (w.h > 3) {
        faces.push({ v: true, x: w.x, y0: w.y, y1: w.y + w.h });
        faces.push({ v: true, x: w.x + w.w, y0: w.y, y1: w.y + w.h });
      }
    }
    this.faces = faces;
  }

  MapModel.prototype.cellOf = function (x, y) {
    var c = clamp((x / this.cell) | 0, 0, this.cols - 1);
    var r = clamp((y / this.cell) | 0, 0, this.rows - 1);
    return r * this.cols + c;
  };

  /* 惰性可见性查询（结果写回矩阵，同一对只算一次） */
  MapModel.prototype.losCells = function (a, b) {
    if (a === b) return true;
    var n = this.n;
    var v = this.los[a * n + b];
    if (v !== 2) return v === 1;
    var ok = this.map.hasLOS(this.cx[a], this.cy[a], this.cx[b], this.cy[b]) ? 1 : 0;
    this.los[a * n + b] = ok;
    this.los[b * n + a] = ok;
    return ok === 1;
  };

  function getModel(map) {
    var key = [map.seed, map.cols, map.rows, map.walls.length, map.w === undefined ? 0 : map.w].join(':');
    var m = modelCache.get(key);
    if (!m) { m = new MapModel(map); modelCache.set(key, m); }
    return m;
  }

  /* =========================================================
     4. 共享世界快照（同帧多辆坦克复用，省算力）
     ========================================================= */

  var worldCache = new WeakMap();

  function World() {
    this.time = -1;
    this.frameId = 0;
    this.map = null;
    this.model = null;
    this.rules = null;
    this.bullets = [];
    this.tanks = [];
    this.alive = [];
    this.paths = new Map();
    this.danger = null;
    this.kills = null;
  }

  function readRules(game, TTRef) {
    var CFG = (TTRef && TTRef.CFG) || {};
    var T = CFG.TANK || {};
    var B = CFG.BULLET || {};
    var tk = game.tanks && game.tanks[0];
    return {
      bs: B.speed || 620,
      br: B.radius || 3.2,
      bl: B.life || 6,
      bounces: B.bounces == null ? 5 : B.bounces,
      grace: B.spawnGrace == null ? 0.07 : B.spawnGrace,
      tr: (tk && tk.radius) || T.radius || 16,
      turn: T.turn || 3.3,
      vmax: T.speed || 178,
      vrev: T.reverseRatio || 0.66,
      accel: T.accel || 920,
      brake: T.brake || 1150,
      cool: T.fireCooldown || 0.42,
      maxB: CFG.MAX_BULLETS_PER_TANK || 5,
    };
  }

  World.prototype.refresh = function (game, TTRef) {
    this.time = game.time;
    this.frameId++;
    this.game = game;
    this.map = game.map;
    if (!this.model || this.model.map !== game.map) {
      this.model = getModel(game.map);
      this.rules = readRules(game, TTRef);
      this.danger = new Float64Array(this.model.n);
    }
    var i;
    this.bullets = game.bullets;
    this.tanks = game.tanks;

    var alive = [];
    for (i = 0; i < this.tanks.length; i++) if (this.tanks[i].alive) alive.push(this.tanks[i]);
    this.alive = alive;

    /* 子弹路径：本帧重建（数量很少），并顺带生成全局「子弹走廊热力」 */
    this.paths = new Map();
    var r = this.rules;
    var dg = this.danger;
    dg.fill(0);
    var cell = this.model.cell;
    for (i = 0; i < this.bullets.length; i++) {
      var b = this.bullets[i];
      if (b.dead) continue;
      var maxD = r.bs * Math.max(0.05, b.life);
      var tr = traceShot(this.map, b.x, b.y, b.angle, maxD, Math.max(0, r.bounces - b.bounces));
      this.paths.set(b, tr.pts);
      /* 前 1.2 秒的航段权重高 */
      var hot = r.bs * 1.2;
      for (var d = 0; d < tr.pts[tr.pts.length - 1].d; d += cell * 0.5) {
        var p = pathAt(tr.pts, d);
        var c = this.model.cellOf(p.x, p.y);
        var w = d < hot ? 1 : 0.35;
        dg[c] += w * 0.6;
      }
    }
    /* 邻格扩散一点点，避免「正好在隔壁就无感」 */
    var tmp = new Float64Array(dg.length);
    for (i = 0; i < dg.length; i++) {
      if (dg[i] <= 0) continue;
      var o = this.model.open[i];
      var spread = dg[i] * 0.32;
      tmp[i] += dg[i];
      if (o & 1) tmp[i - this.model.cols] += spread;
      if (o & 2) tmp[i + this.model.cols] += spread;
      if (o & 4) tmp[i - 1] += spread;
      if (o & 8) tmp[i + 1] += spread;
    }
    this.danger = tmp;
    this.kills = game.kills;
  };

  function getWorld(game, TTRef) {
    var w = worldCache.get(game);
    if (!w) { w = new World(); worldCache.set(game, w); }
    if (w.time !== game.time) w.refresh(game, TTRef);
    return w;
  }

  /* =========================================================
     5. 解析扫掠：直线段 vs 匀速移动的圆
     ========================================================= */

  /* 返回命中时间；-1 表示没打中 */
  function sweepHit(ax, ay, bx, by, tA, tB, p0x, p0y, p1x, p1y, killR) {
    var span = tB - tA;
    if (span <= 1e-6) return -1;
    var bvx = (bx - ax) / span, bvy = (by - ay) / span;
    var tvx = (p1x - p0x) / span, tvy = (p1y - p0y) / span;
    var ux = bvx - tvx, uy = bvy - tvy;
    var rx = ax - p0x, ry = ay - p0y;
    var uu = ux * ux + uy * uy;
    var t = uu > 1e-9 ? -(rx * ux + ry * uy) / uu : 0;
    if (t < 0) t = 0;
    if (t > span) t = span;
    var dx = rx + ux * t, dy = ry + uy * t;
    if (dx * dx + dy * dy <= killR * killR) {
      /* 还要确认「进入」而不是「已经在里面往外走」 */
      var d0x = rx, d0y = ry;
      if (d0x * d0x + d0y * d0y <= killR * killR) return tA;
      return tA + t;
    }
    return -1;
  }

  /* 同一段里，我到子弹的最小距离（用于自伤余量） */
  function sweepMin(ax, ay, bx, by, tA, tB, p0x, p0y, p1x, p1y) {
    var span = tB - tA;
    if (span <= 1e-6) return Math.sqrt(len2(ax - p0x, ay - p0y));
    var bvx = (bx - ax) / span, bvy = (by - ay) / span;
    var tvx = (p1x - p0x) / span, tvy = (p1y - p0y) / span;
    var ux = bvx - tvx, uy = bvy - tvy;
    var rx = ax - p0x, ry = ay - p0y;
    var uu = ux * ux + uy * uy;
    var t = uu > 1e-9 ? -(rx * ux + ry * uy) / uu : 0;
    t = clamp(t, 0, span);
    var dx = rx + ux * t, dy = ry + uy * t;
    return Math.sqrt(dx * dx + dy * dy);
  }

  /* =========================================================
     6. 大脑
     ========================================================= */

  var FUT_DT = 1 / 30;

  function Track() {
    this.x = 0; this.y = 0; this.vx = 0; this.vy = 0; this.speed = 0;
    this.t = -1; this.ox = 0; this.oy = 0; this.ot = -1; this.angle = 0;
    this.dyingSoon = 0; this.lastSeen = -1;
    this.lastShotT = -9;      // 他刚开过枪 = 0.42 秒内还不了手，这是我的窗口
  }

  Track.prototype.posAt = function (t) {
    var x = this.x + this.vx * t;
    var y = this.y + this.vy * t;
    return { x: x, y: y };
  };

  function Brain(idx) {
    this.idx = idx;
    this.p = P;
    this.t = 0;
    this.roundAge = 0;
    this.tracks = {};
    this.planTimer = 0;
    this.dodgeTimer = 0;
    this.dodge = null;
    this.dodgeHold = 0;
    this.fireDelay = 0;
    this.plan = null;
    this.goal = -1;
    this.stuck = 0;
    this.recover = 0;
    this.recoverDir = 1;
    this.lastX = 0; this.lastY = 0; this.checkT = 0;
    this.prevSpeed = 0;
    this.pressure = 0;
    this.snap = false;
    this.aliveCount = 8;
    this.dbg = {};
    this.noShotT = 0;
    this.wasEngaged = 0;
    this.jink = 1;
    this.jinkT = 0;
    this.orbitPhase = 'aim';
    this.orbitT = 0;
    this.aimStall = 0;
    this.recoverDrive = -1;
    this.badGoals = {};        // cell -> 过期时间：走不通的格子短期内不再选
    this.seenB = new Set();    // 已经见过的子弹（用来检测「谁刚开火」）
    this.routeBest = Infinity;
    this.routeT = 0;
    this.prevAlive = null;
    this.lastKillT = 0;
    this.stale = 0;
    this.berserk = false;
    this.suppressT = 0;
  }

  /* ---- 感知 ---- */

  Brain.prototype.perceive = function (tank, w) {
    var r = w.rules;
    for (var i = 0; i < w.alive.length; i++) {
      var o = w.alive[i];
      if (o.index === tank.index) continue;
      var h = this.tracks[o.index];
      if (!h) { h = this.tracks[o.index] = new Track(); h.x = o.x; h.y = o.y; h.t = this.t; h.ox = o.x; h.oy = o.y; h.ot = this.t; }
      h.x = o.x; h.y = o.y; h.angle = o.angle;
      /* 0.1~0.2s 窗口回归速度，避免抖一抖就变向 */
      var dt = this.t - h.ot;
      if (dt >= 0.09) {
        var vx = (o.x - h.ox) / dt, vy = (o.y - h.oy) / dt;
        var sp = Math.sqrt(vx * vx + vy * vy);
        var cap = r.vmax * 1.1;
        if (sp > cap) { vx *= cap / sp; vy *= cap / sp; sp = cap; }
        if (sp < 14) { vx *= 0.35; vy *= 0.35; }
        h.vx += (vx - h.vx) * 0.6;
        h.vy += (vy - h.vy) * 0.6;
        h.speed = Math.sqrt(h.vx * h.vx + h.vy * h.vy);
        h.ox = o.x; h.oy = o.y; h.ot = this.t;
      }
      h.dyingSoon = Math.max(0, h.dyingSoon - 0.25);
    }

    /* 谁刚开火（新子弹出现）→ 他 0.42 秒内还不了手，这就是我的进攻窗口 */
    var bs = w.bullets;
    for (var bi = 0; bi < bs.length; bi++) {
      var bu = bs[bi];
      if (bu.dead || this.seenB.has(bu)) continue;
      this.seenB.add(bu);
      var tk = this.tracks[bu.owner];
      if (tk) tk.lastShotT = this.t;
    }
    if (this.seenB.size > bs.length + 16) {
      var live = new Set(bs), self = this;
      this.seenB.forEach(function (b) { if (!live.has(b)) self.seenB.delete(b); });
    }
    this.aliveCount = w.alive.length;
  };

  /* 谁快被别的子弹打死（规划帧算一次） */
  Brain.prototype.markDoomed = function (tank, w) {
    var r = w.rules;
    var i, j;
    for (i = 0; i < w.alive.length; i++) if (w.alive[i].index !== tank.index) this.tracks[w.alive[i].index] && (this.tracks[w.alive[i].index].dyingSoon = 0);

    for (i = 0; i < w.bullets.length; i++) {
      var b = w.bullets[i];
      if (b.dead) continue;
      var path = w.paths.get(b);
      if (!path) continue;
      var maxD = path[path.length - 1].d;
      var tA = 0;
      var ownerTrack = this.tracks[b.owner];
      for (var s = 1; s < path.length; s++) {
        var tB = path[s].d / r.bs;
        for (j = 0; j < w.alive.length; j++) {
          var o = w.alive[j];
          if (o.index === tank.index) continue;
          if (b.owner === o.index && tB < r.grace) continue;
          var th = this.tracks[o.index];
          if (!th) continue;
          var p0 = th.posAt(tA), p1 = th.posAt(tB);
          var kt = sweepHit(path[s - 1].x, path[s - 1].y, path[s].x, path[s].y, tA, tB, p0.x, p0.y, p1.x, p1.y, r.tr + r.br);
          if (kt >= 0) { th.dyingSoon = 1; j = w.alive.length; s = path.length; break; }
        }
        tA = tB;
      }
    }
  };

  /* ---- 我的未来（自伤判定的最坏情况） ---- */

  Brain.prototype.futures = function (tank, r, horizon) {
    var list = [];
    var base = [
      { steer: 0, speed: tank.speed },
      { steer: 0, speed: 0 },
    ];
    if (this.plan) base.push({ steer: this.plan.steer || 0, speed: tank.speed });
    if (this.dodge) base.push({ steer: this.dodge.steer, speed: tank.speed });

    for (var f = 0; f < base.length; f++) {
      var b = base[f];
      var x = tank.x, y = tank.y, a = tank.angle, sp = tank.speed;
      var pts = [{ x: x, y: y, t: 0 }];
      for (var t = FUT_DT; t <= horizon + 1e-6; t += FUT_DT) {
        a += b.steer * r.turn * FUT_DT;
        var maxv = b.speed >= 0 ? r.vmax : -r.vmax * r.vrev;
        sp = clamp(sp + (b.speed - sp) * 0.5, -r.vmax * r.vrev, r.vmax);
        x += Math.cos(a) * sp * FUT_DT;
        y += Math.sin(a) * sp * FUT_DT;
        pts.push({ x: x, y: y, t: t });
      }
      pts.horizon = horizon;
      list.push(pts);
    }
    return list;
  };

  function futAt(pts, t) {
    if (t <= 0) return pts[0];
    var i = Math.min(pts.length - 1, Math.round(t / FUT_DT));
    return pts[i];
  }

  /* =========================================================
     7. 射击
     ========================================================= */

  Brain.prototype.intercept = function (mx, my, th, r) {
    var t = Math.sqrt(len2(th.x - mx, th.y - my)) / r.bs;
    var ang = Math.atan2(th.y - my, th.x - mx);
    for (var i = 0; i < 4; i++) {
      var px = th.x + th.vx * t, py = th.y + th.vy * t;
      ang = Math.atan2(py - my, px - mx);
      var d = Math.sqrt(len2(px - mx, py - my));
      t = d / r.bs;
    }
    return ang;
  };

  /* 墙面镜像求反弹入射角 */
  Brain.prototype.bankAngles = function (tank, w, mx, my, th, r, budget) {
    var out = [];
    var faces = w.model.faces;
    var tx = th.x + th.vx * 0.3, ty = th.y + th.vy * 0.3;
    var maxLeg = r.bs * 1.4;
    var tried = 0;

    for (var i = 0; i < faces.length && out.length < budget && tried < 220; i++) {
      var f = faces[i];
      tried++;
      var ix, iy;

      if (f.v) {
        if (Math.abs(f.x - mx) > maxLeg || Math.abs(f.x - tx) > maxLeg) continue;
        var ddx = tx - (2 * f.x - mx);
        if (Math.abs(ddx) < 20) continue;
        var s = (f.x - (2 * f.x - mx)) / ddx;
        if (s <= 0.02 || s >= 0.98) continue;
        iy = my + (ty - my) * s;
        ix = f.x;
        if (iy < f.y0 + 3 || iy > f.y1 - 3) continue;
        if (Math.abs(ix - tank.x) < r.tr + 8) continue;
        var dirx = tx > f.x ? 1 : -1;
        if (w.map.raycast(mx, my, ix, iy)) continue;
        if (w.map.raycast(ix + dirx * 3, iy, tx, ty)) continue;
      } else {
        if (Math.abs(f.y - my) > maxLeg || Math.abs(f.y - ty) > maxLeg) continue;
        var ddy = ty - (2 * f.y - my);
        if (Math.abs(ddy) < 20) continue;
        var s2 = (f.y - (2 * f.y - my)) / ddy;
        if (s2 <= 0.02 || s2 >= 0.98) continue;
        ix = mx + (tx - mx) * s2;
        iy = f.y;
        if (ix < f.x0 + 3 || ix > f.x1 - 3) continue;
        if (Math.abs(iy - tank.y) < r.tr + 8) continue;
        var diry = ty > f.y ? 1 : -1;
        if (w.map.raycast(mx, my, ix, iy)) continue;
        if (w.map.raycast(ix, iy + diry * 3, tx, ty)) continue;
      }

      var ang = Math.atan2(iy - my, ix - mx);
      if (Math.abs(angDiff(tank.angle, ang)) > 1.35) continue;
      out.push(ang);
    }
    return out;
  };

  /* 模拟一发 */
  Brain.prototype.simShot = function (tank, w, angle, fut) {
    var r = w.rules;
    var tip = r.tr + 12;
    var mx = tank.x + Math.cos(angle) * tip;
    var my = tank.y + Math.sin(angle) * tip;

    if (w.map.solidAt(mx + Math.cos(angle) * 5, my + Math.sin(angle) * 5)) {
      return null;   // 炮口被挡，会哑火
    }

    var horizon = Math.max(Q.hitHorizon, Q.selfHorizon);
    var tr = traceShot(w.map, mx, my, angle, r.bs * horizon, r.bounces);
    if (tr.end === 'fizzle') return null;
    var pts = tr.pts;

    var killR = r.tr + r.br;
    var hitIdx = -1, hitT = 0;
    var selfMin = Infinity;
    var pocket = 0;
    var tSelf = r.grace + 0.008;   // 出膛 0.07s 内不会伤到自己，别把这段算进风险

    var tA = 0;
    for (var s = 1; s < pts.length; s++) {
      var tB = pts[s].d / r.bs;

      /* 命中谁 */
      for (var i = 0; i < w.alive.length; i++) {
        var o = w.alive[i];
        if (o.index === tank.index) continue;
        var th = this.tracks[o.index];
        if (!th) continue;
        var p0 = th.posAt(tA), p1 = th.posAt(tB);
        var kt = sweepHit(pts[s - 1].x, pts[s - 1].y, pts[s].x, pts[s].y, tA, tB, p0.x, p0.y, p1.x, p1.y, killR);
        if (kt >= 0) {
          if (hitIdx < 0 || kt < hitT) { hitIdx = o.index; hitT = kt; }
        }
      }

      /* 自己的最坏情况：把时间窗裁到 grace 之后 */
      var sA = tA > tSelf ? tA : tSelf;
      if (tB > sA) {
        var k = (tB > tA) ? (sA - tA) / (tB - tA) : 0;
        var ax = pts[s - 1].x + (pts[s].x - pts[s - 1].x) * k;
        var ay = pts[s - 1].y + (pts[s].y - pts[s - 1].y) * k;
        for (var f = 0; f < fut.length; f++) {
          var a0 = futAt(fut[f], sA), a1 = futAt(fut[f], tB);
          var m = sweepMin(ax, ay, pts[s].x, pts[s].y, sA, tB, a0.x, a0.y, a1.x, a1.y);
          if (m < selfMin) selfMin = m;
          if (m < Q.pocketR) pocket += 1;
        }
      }

      if (hitIdx >= 0 && hitT <= tB) break;    // 子弹在命中后就消失
      tA = tB;
    }

    return {
      angle: angle,
      hitIdx: hitIdx,
      hitT: hitT,
      selfMin: selfMin === Infinity ? 999 : selfMin,
      pocket: pocket,
    };
  };

  Brain.prototype.liveBullets = function (tank, w) {
    var n = 0;
    for (var i = 0; i < w.bullets.length; i++) {
      var b = w.bullets[i];
      if (!b.dead && b.owner === tank.index) n++;
    }
    return n;
  };

  Brain.prototype.pickShot = function (tank, w) {
    var r = w.rules;
    var fut = this.futures(tank, r, Math.max(Q.hitHorizon, Q.selfHorizon));
    var tip = r.tr + 12;
    var mx = tank.x + Math.cos(tank.angle) * tip;
    var my = tank.y + Math.sin(tank.angle) * tip;
    var live = this.liveBullets(tank, w);

    /* 候选目标：按距离排序，最多 5 个 */
    var list = [];
    for (var i = 0; i < w.alive.length; i++) {
      var o = w.alive[i];
      if (o.index === tank.index) continue;
      var th = this.tracks[o.index];
      if (!th) continue;
      var d = Math.sqrt(len2(o.x - tank.x, o.y - tank.y));
      if (d > 700) continue;
      var los = w.map.hasLOS(tank.x, tank.y, o.x, o.y);
      list.push({ o: o, th: th, d: d, los: los });
    }
    if (!list.length) return null;
    list.sort(function (a, b) { return a.d - b.d; });

    var cands = [];
    var direct = 0;
    var n = Math.min(list.length, 5);
    for (var k = 0; k < n; k++) {
      var e = list[k];
      if (e.los && direct < 3) {
        cands.push({ ang: this.intercept(mx, my, e.th, r), target: e.o, kind: 'direct' });
        direct++;
      } else if (!e.los && k < 3) {
        var ba = this.bankAngles(tank, w, mx, my, e.th, r, 3);
        for (var q = 0; q < ba.length; q++) cands.push({ ang: ba[q], target: e.o, kind: 'bank' });
      }
    }
    /* 当前朝向（省一次转向） */
    cands.push({ ang: tank.angle, target: null, kind: 'now' });

    var best = null;
    for (var c = 0; c < cands.length; c++) {
      var res = this.simShot(tank, w, cands[c].ang, fut);
      if (!res) continue;
      var score = 0;

      /* --- 自伤闸门：这是活下来的根本 --- */
      var selfKill = res.selfMin < r.tr + r.br;
      if (selfKill) continue;
      var need = Q.selfMargin * (this.fuse ? Q.fuseMargin : (this.desperate ? Q.desperateMargin : (this.berserk ? Q.berserkMargin : 1)));
      var margin = res.selfMin - (r.tr + r.br);
      if (margin < need) continue;
      score -= Math.max(0, 45 - margin) * Q.nearRiskW;
      score -= res.pocket * Q.pocketW;

      if (res.hitIdx >= 0) {
        var victim = null;
        for (var v = 0; v < w.alive.length; v++) if (w.alive[v].index === res.hitIdx) victim = w.alive[v];
        score += Q.killBonus;
        score -= res.hitT * 14;
        if (victim) {
          var vt = this.tracks[victim.index];
          if (vt && vt.dyingSoon) score -= 62;            // 快被打死的别补枪
          if (vt && this.t - vt.lastShotT < 0.42) score += Q.punishBonus;   // 他还在冷却，打他最安全
          score += Math.min(16, (w.kills[victim.index] || 0) * 2.2);   // 强的先清
          if (this.aliveCount <= 3) score += 40;          // 终局收益极大
        }
        if (cands[c].kind === 'bank') score += 8;
      } else {
        score -= Q.missPenalty;
        /* 打空不是完全没用：封锁走廊、逼走位。但价值低 */
        if (res.pocket < 2 && margin > 60) score += 6;
      }

      if (live >= 2) score -= (live - 1) * Q.spamPenalty;
      var liveCap = this.fuse ? r.maxB : (this.berserk ? Math.min(r.maxB, this.desperate ? r.maxB : Q.maxLive + 1) : Q.maxLive);
      if (live >= liveCap) score -= 1e6;

      if (!best || score > best.score) {
        best = {
          score: score, angle: cands[c].ang, hitIdx: res.hitIdx, hitT: res.hitT,
          kind: cands[c].kind, margin: margin, canFire: live < liveCap,
        };
      }
    }
    /* 僵局压制：长时间没人死时，朝对手方向打「逼位弹」，
       流弹会替我封走廊，也把躲在角里的对手逼出来 */
    var suppress = null;
    if ((this.stale > Q.suppressAfter || this.fuse) && (!best || best.hitIdx < 0)) {
      var nes = this.nearest(tank, w);
      if (nes && nes.d < 620) {
        var thn = this.tracks[nes.o.index];
        var sang = thn ? this.intercept(mx, my, thn, r) : Math.atan2(nes.o.y - my, nes.o.x - mx);
        var sres = this.simShot(tank, w, sang, fut);
        var sNeed = Q.selfMargin * (this.fuse ? Q.fuseMargin : (this.desperate ? Q.desperateMargin : (this.berserk ? Q.berserkMargin : 1)));
        if (sres && sres.selfMin - (r.tr + r.br) >= Math.max(sNeed, 8)) {
          suppress = { angle: sang, hitIdx: sres.hitIdx };
        }
      }
    }
    if (best) best.suppress = suppress;
    else if (suppress) best = { score: -5, angle: suppress.angle, hitIdx: suppress.hitIdx, kind: 'suppress', canFire: live < liveCap, suppress: suppress, margin: 0 };
    return best;
  };

  /* =========================================================
     8. 走位
     ========================================================= */

  Brain.prototype.mode = function (tank, w) {
    var alive = this.aliveCount;
    var age = this.roundAge;
    if (alive <= 2) return 'duel';
    if (alive <= 4 && age > Q.huntAfter) return 'hunt';
    if (age > Q.force) return 'hunt';
    if (age > Q.escalate && this.pressure < Q.pressureCap) return 'hunt';
    return 'prowl';
  };

  Brain.prototype.nearest = function (tank, w) {
    var best = null, bd = Infinity;
    for (var i = 0; i < w.alive.length; i++) {
      var o = w.alive[i];
      if (o.index === tank.index) continue;
      var d = len2(o.x - tank.x, o.y - tank.y);
      if (d < bd) { bd = d; best = o; }
    }
    return best ? { o: best, d: Math.sqrt(bd) } : null;
  };

  Brain.prototype.pickHunt = function (tank, w) {
    var model = w.model;
    var N = model.n;
    var myCell = model.cellOf(tank.x, tank.y);
    var best = null, bs = -Infinity;

    for (var i = 0; i < w.alive.length; i++) {
      var o = w.alive[i];
      if (o.index === tank.index) continue;
      var th = this.tracks[o.index];
      var d = Math.sqrt(len2(o.x - tank.x, o.y - tank.y));
      var cell = model.cellOf(o.x, o.y);
      var steps = model.steps[myCell * N + cell];
      if (steps < 0) continue;
      var s = 0;
      s -= steps * 6.5;
      s -= d * 0.02;
      s += Math.min(24, (w.kills[o.index] || 0) * 2.4);
      if (th) { s -= th.speed * 0.06; if (th.dyingSoon) s -= 90; }
      if (w.map.hasLOS(tank.x, tank.y, o.x, o.y)) s += 14;
      if (s > bs) { bs = s; best = o; }
    }
    return best;
  };

  /* 相邻两格之间的「门洞」中心 —— 沿着它走才能穿过缺口，而不是怼在墙角 */
  function doorway(model, a, b) {
    var cols = model.cols, cell = model.cell;
    var ar = (a / cols) | 0, ac = a % cols;
    var br = (b / cols) | 0, bc = b % cols;
    if (bc === ac + 1) return { x: (ac + 1) * cell, y: (ar + 0.5) * cell };
    if (bc === ac - 1) return { x: ac * cell, y: (ar + 0.5) * cell };
    if (br === ar + 1) return { x: (ac + 0.5) * cell, y: (ar + 1) * cell };
    if (br === ar - 1) return { x: (ac + 0.5) * cell, y: ar * cell };
    return { x: model.cx[b], y: model.cy[b] };
  }

  /* 沿最短路生成门洞序列，之后逐个推进 */
  Brain.prototype.rebuildNav = function (tank, w, goalCell) {
    var model = w.model;
    var my = model.cellOf(tank.x, tank.y);
    var doors = [];
    var cur = my, guard = 0;
    while (cur !== goalCell && guard++ < 120) {
      var nx = model.next[cur * model.n + goalCell];
      if (nx < 0 || nx === cur) break;
      doors.push(doorway(model, cur, nx));
      cur = nx;
    }
    doors.push({ x: model.cx[goalCell], y: model.cy[goalCell] });
    this.nav = { from: my, goal: goalCell, doors: doors, idx: 0, look: 0 };
    this.advanceNav(tank, w);
    this.routeBest = Infinity;
    this.routeT = this.t;
  };

  Brain.prototype.advanceNav = function (tank, w) {
    var nv = this.nav;
    if (!nv) return;
    while (nv.idx < nv.doors.length - 1 &&
           Math.sqrt(len2(nv.doors[nv.idx].x - tank.x, nv.doors[nv.idx].y - tank.y)) < 34) nv.idx++;
    var best = nv.idx;
    var top = Math.min(nv.doors.length - 1, nv.idx + 3);
    for (var i = top; i > nv.idx; i--) {
      if (w.map.hasLOS(tank.x, tank.y, nv.doors[i].x, nv.doors[i].y)) { best = i; break; }
    }
    nv.look = best;
  };

  Brain.prototype.navWp = function (tank, w) {
    var model = w.model;
    var cur = model.cellOf(tank.x, tank.y);
    if (!this.nav || this.nav.from !== cur || this.nav.goal !== this.goal) {
      if (this.goal >= 0) this.rebuildNav(tank, w, this.goal);
      else return null;
    }
    this.advanceNav(tank, w);
    var d = this.nav.doors[this.nav.look];
    return d ? { x: d.x, y: d.y } : null;
  };

  Brain.prototype.chooseGoal = function (tank, w, md) {
    var model = w.model;
    var N = model.n;
    var myCell = model.cellOf(tank.x, tank.y);
    var cr = (myCell / model.cols) | 0, cc = myCell % model.cols;
    var R = this.berserk ? 5 : 4;
    var now = this.t;
    for (var bg in this.badGoals) if (this.badGoals[bg] < now) delete this.badGoals[bg];

    var enemies = [];
    for (var i = 0; i < w.alive.length; i++) {
      var o = w.alive[i];
      if (o.index === tank.index) continue;
      enemies.push({ o: o, cell: model.cellOf(o.x, o.y), d: Math.sqrt(len2(o.x - tank.x, o.y - tank.y)) });
    }

    var hunt = (md !== 'prowl') ? this.pickHunt(tank, w) : null;
    if (!hunt && (md === 'prowl' && (this.roundAge > Q.escalate || this.berserk))) hunt = this.pickHunt(tank, w);
    var huntCell = hunt ? model.cellOf(hunt.x, hunt.y) : -1;

    var approach = md === 'duel' ? 1.0 : md === 'hunt' ? 0.55 : 0.12;
    if (this.roundAge > Q.force) approach = 1.3;
    if (this.berserk) approach = 1.9;
    if (this.aliveCount >= 6 && md === 'prowl' && !this.berserk) approach = 0.05;

    var bestScore = -Infinity, bestCell = myCell;

    for (var dr = -R; dr <= R; dr++) {
      for (var dc = -R; dc <= R; dc++) {
        var rr = cr + dr, ccl = cc + dc;
        if (rr < 1 || ccl < 1 || rr > model.rows - 2 || ccl > model.cols - 2) continue;
        var cell = rr * model.cols + ccl;
        var fromMe = model.steps[myCell * N + cell];
        if (fromMe < 0 || fromMe > R + 2) continue;

        var x = model.cx[cell], y = model.cy[cell];
        var s = 0;

        /* 暴露度：多少敌人能直视这格（近处权重高） */
        var expo = 0, vis = 0;
        for (var e = 0; e < enemies.length; e++) {
          if (model.losCells(cell, enemies[e].cell)) { vis++; expo += 1 / (1 + enemies[e].d / 240); }
        }
        s -= expo * Q.exposureW * (this.snap ? 1.7 : 1) * (this.berserk ? 0.5 : 1);
        /* 硬约束：同时被多个人直视的格子基本等于死地 */
        if (vis > Q.visFree) s -= (vis - Q.visFree) * Q.visPunish * (md === 'prowl' ? 1 : 0.6) * (this.berserk ? 0.35 : 1);
        if (this.badGoals[cell]) s -= Q.badGoalPenalty;

        /* 子弹走廊 */
        s -= w.danger[cell] * Q.dangerW;

        /* 推进 */
        if (huntCell >= 0) {
          var st = model.steps[cell * N + huntCell];
          if (st >= 0) s -= st * Q.approachW * approach;
        }

        /* 掩体与退路 */
        var deg = model.deg[cell];
        if (deg <= 1) s -= Q.deadEndP;
        else s += (4 - deg) * Q.coverW;
        if (deg >= 3) s += Q.exitW;

        /* 别挤在人堆里 */
        for (var q2 = 0; q2 < enemies.length; q2++) {
          var dd = Math.sqrt(len2(enemies[q2].o.x - x, enemies[q2].o.y - y));
          var near = 150;
          if (dd < near) s -= (near - dd) * Q.crowdW * (md === 'duel' ? 0.2 : 1);
          if (dd < 60) s -= 30;
        }

        s -= fromMe * Q.reachW;
        if (cell === this.goal) s += Q.hyst;

        if (s > bestScore) { bestScore = s; bestCell = cell; }
      }
    }

    var prevGoal = this.goal;
    this.goal = bestCell;
    /* 暴走时不讲究-score-了：直接锁定要猎的敌人格子，一路走过去 */
    if (this.berserk && hunt) this.goal = model.cellOf(hunt.x, hunt.y);
    if (prevGoal !== this.goal) { this.routeBest = Infinity; this.routeT = this.t; }

    /* 门洞导航 */
    var navp = this.navWp(tank, w);
    var gcell = this.goal;
    /* 暴走且已经很近 */
    if (this.berserk && hunt) {
      var hd = Math.sqrt(len2(hunt.x - tank.x, hunt.y - tank.y));
      if (hd < 170 && w.map.hasLOS(tank.x, tank.y, hunt.x, hunt.y)) {
        /* 看得见：直接怼到脸上 */
        return { x: hunt.x, y: hunt.y, gx: hunt.x, gy: hunt.y, door: false };
      }
      if (hd < 330) {
        /* 看不见但就在附近：找一个能望见他的邻格绕出去取射击线，而不是顶在墙上磨 */
        var peek = -1, peekScore = -Infinity;
        var myc = model.cellOf(tank.x, tank.y);
        var mr = (myc / model.cols) | 0, mc2 = myc % model.cols;
        for (var pr = mr - 2; pr <= mr + 2; pr++) {
          for (var pc = mc2 - 2; pc <= mc2 + 2; pc++) {
            if (pr < 0 || pc < 0 || pr >= model.rows || pc >= model.cols) continue;
            var pcell = pr * model.cols + pc;
            if (model.steps[myc * N + pcell] < 0 || model.steps[myc * N + pcell] > 3) continue;
            if (!w.map.hasLOS(model.cx[pcell], model.cy[pcell], hunt.x, hunt.y)) continue;
            var pd = Math.sqrt(len2(model.cx[pcell] - tank.x, model.cy[pcell] - tank.y));
            var ps = -pd + (model.deg[pcell] >= 2 ? 12 : 0) - (model.losCells(pcell, myc) ? 0 : 8);
            if (ps > peekScore) { peekScore = ps; peek = pcell; }
          }
        }
        if (peek >= 0) return { x: model.cx[peek], y: model.cy[peek], gx: model.cx[peek], gy: model.cy[peek], door: false };
      }
    }
    return {
      x: navp ? navp.x : model.cx[gcell], y: navp ? navp.y : model.cy[gcell],
      gx: model.cx[gcell], gy: model.cy[gcell],
      door: !!navp && this.nav && this.nav.look < this.nav.doors.length - 1,
    };
  };

  /* =========================================================
     9. 规避：枚举机动，选「弹到我最远距离」最大的
     ========================================================= */

  var MANEUVER = [
    { steer: -1, drive: 1 }, { steer: 1, drive: 1 },
    { steer: -1, drive: -1 }, { steer: 1, drive: -1 },
    { steer: 0, drive: 1 }, { steer: 0, drive: -1 },
    { steer: -1, drive: 0 }, { steer: 1, drive: 0 },
    { steer: 0, drive: 0 },        // 仅作为「什么都不做」的基线，不会被选中
  ];

  Brain.prototype.threats = function (tank, w) {
    var r = w.rules;
    var out = [];
    for (var i = 0; i < w.bullets.length; i++) {
      var b = w.bullets[i];
      if (b.dead) continue;
      if (b.owner === tank.index && b.grace > 0) continue;
      var d = Math.sqrt(len2(b.x - tank.x, b.y - tank.y));
      if (d > Q.threatR) continue;
      var path = w.paths.get(b);
      if (!path) continue;
      if (!pathClear(path, tank.x, tank.y, Q.threatNearR * Q.threatNearR)) continue;
      var eta = (d - (r.tr + r.br)) / r.bs;
      out.push({ b: b, path: path, dist: d, eta: eta });
    }
    out.sort(function (a, b) { return a.dist - b.dist; });
    return out;
  };

  Brain.prototype.rollout = function (tank, r, m, horizon) {
    var x = tank.x, y = tank.y, a = tank.angle, sp = tank.speed;
    var step = 1 / 40;
    var pts = [];
    for (var t = step; t <= horizon + 1e-9; t += step) {
      a += m.steer * r.turn * step;
      var maxv = m.drive < 0 ? -r.vmax * r.vrev : r.vmax;
      var target = m.drive === 0 ? 0 : maxv * sgn(m.drive) * (m.drive < 0 ? 1 : 1);
      var want = m.drive > 0 ? r.vmax : (m.drive < 0 ? -r.vmax * r.vrev : 0);
      var rate = (sp * want < 0) ? r.brake : r.accel;
      if (sp < want) sp = Math.min(want, sp + rate * step);
      else if (sp > want) sp = Math.max(want, sp - rate * step);
      x += Math.cos(a) * sp * step;
      y += Math.sin(a) * sp * step;
      pts.push({ x: x, y: y, t: t, a: a });
    }
    return pts;
  };

  /* 对手「枪口正指着我、且随时能开火」→ 提前规避，不等子弹出来 */
  Brain.prototype.preThreats = function (tank, w) {
    var r = w.rules;
    if (w._pre && w._preId === w.frameId) {
      var got = w._pre[tank.index];
      if (got) return got;
    } else {
      w._pre = {}; w._preId = w.frameId;
    }
    var out = [];
    for (var i = 0; i < w.alive.length; i++) {
      var o = w.alive[i];
      if (o.index === tank.index) continue;
      var d = Math.sqrt(len2(o.x - tank.x, o.y - tank.y));
      if (d > Q.snapRange) continue;
      if (!w.map.hasLOS(tank.x, tank.y, o.x, o.y)) continue;
      var aim = Math.atan2(tank.y - o.y, tank.x - o.x);
      if (Math.abs(angDiff(o.angle, aim)) > Q.snapArc) continue;
      var mx = o.x + Math.cos(o.angle) * (r.tr + 12);
      var my = o.y + Math.sin(o.angle) * (r.tr + 12);
      var delay = Math.max(0, o.cooldown || 0) + 0.02;
      var path = traceShot(w.map, mx, my, o.angle, r.bs * 1.6, r.bounces).pts;
      /* 这条路确实擦过我附近才算 */
      if (!pathClear(path, tank.x, tank.y, Q.threatNearR * Q.threatNearR)) continue;
      out.push({ path: path, delay: delay, dist: d, owner: o.index });
    }
    w._pre[tank.index] = out;
    return out;
  };

  Brain.prototype.solveDodge = function (tank, w) {
    var th = this.threats(tank, w);
    var pre = this.preThreats(tank, w);
    if (!th.length && !pre.length) return null;
    var r = w.rules;
    var killR = r.tr + r.br;
    var horizon = 0.95;

    var baseGap = Infinity;
    var best = null;

    for (var m = 0; m < MANEUVER.length; m++) {
      var mv = MANEUVER[m];
      var isBase = mv.steer === 0 && mv.drive === 0;
      var pts = this.rollout(tank, r, mv, horizon);
      var minGap = Infinity;
      var bad = 0;

      for (var i = 0; i < pts.length; i++) {
        var p = pts[i];
        if (w.map.solidAt(p.x, p.y) ||
            w.map.solidAt(p.x + Math.cos(p.a) * (r.tr + 4), p.y + Math.sin(p.a) * (r.tr + 4))) {
          bad += 1;
        }
        var d = p.t * r.bs;
        for (var k = 0; k < th.length; k++) {
          var bp = pathAt(th[k].path, d);
          var gap = Math.sqrt(len2(bp.x - p.x, bp.y - p.y)) - killR;
          if (gap < minGap) minGap = gap;
        }
        /* 预判威胁：按对手的冷却延后起算 */
        for (var u = 0; u < pre.length; u++) {
          var dd = (p.t - pre[u].delay) * r.bs;
          if (dd < 0) continue;
          var pp = pathAt(pre[u].path, dd);
          var g2 = Math.sqrt(len2(pp.x - p.x, pp.y - p.y)) - killR;
          /* 还没开火的威胁打折：他可能不打我 */
          if (g2 < minGap) minGap = g2 + Q.preWeight;
        }
      }

      if (isBase) { baseGap = minGap; continue; }   // 基线不参与选择：绝不原地站桩

      var score = minGap - bad * Q.wallPenalty;
      if (!best || score > best.score) best = { score: score, steer: mv.steer, drive: mv.drive, gap: minGap };
    }

    if (!best) return null;
    /* 只有「待着不动有危险、且某个机动明显更好」才交给规避 */
    if (baseGap > Q.dodgeGap && best.gap - baseGap < 6) return null;
    if (best.gap > Q.dodgeGap && baseGap > Q.dodgeGap) return null;
    return best;
  };

  /* =========================================================
     10. 主循环
     ========================================================= */

  /* 前方哪个方向最空（用来拐弯绕墙角，而不是顶在门上发抖） */
  Brain.prototype.clearance = function (tank, w, ang, r, spread) {
    var best = null;
    var offs = [0, 0.45, -0.45, 0.9, -0.9, 1.4, -1.4, 2.1, -2.1];
    for (var i = 0; i < offs.length; i++) {
      var a = ang + offs[i];
      var cx = tank.x + Math.cos(a) * (r.tr + 12);
      var cy = tank.y + Math.sin(a) * (r.tr + 12);
      if (w.map.solidAt(cx, cy)) continue;
      var far = !w.map.solidAt(tank.x + Math.cos(a) * (r.tr + 42), tank.y + Math.sin(a) * (r.tr + 42));
      var score = (far ? 40 : 12) - Math.abs(offs[i]) * (spread ? 8 : 14);
      if (!best || score > best.score) best = { a: a, score: score, far: far };
      if (i === 0 && far) break;   // 正前方就很空，不用再看
    }
    return best;
  };

  Brain.prototype.update = function (tank, game, dt, TTRef) {
    Q = this.p || P;
    var w = getWorld(game, TTRef);
    var r = w.rules;
    if (!tank.alive) return { drive: 0, steer: 0, shoot: false };

    this.t += dt;
    this.roundAge += dt;
    this.perceive(tank, w);

    /* ---- 僵局监测：场上多久没人死了 ---- */
    if (this.prevAlive === null) { this.prevAlive = w.alive.length; this.lastKillT = this.t; }
    if (w.alive.length < this.prevAlive) this.lastKillT = this.t;
    this.prevAlive = w.alive.length;
    this.stale = this.t - this.lastKillT;
    this.berserk = this.stale > Q.staleBerserk || this.roundAge > Q.forceBerserk;
    this.desperate = this.stale > Q.staleDesperate || this.roundAge > Q.desperateAge;
    this.fuse = this.stale > Q.fuseStale || this.roundAge > Q.fuseAge;

    var inp = { drive: 0, steer: 0, shoot: false };

    /* ---- 卡住检测：只要原地磨蹭就强制脱困 ---- */
    if (this.t - this.checkT >= 0.5) {
      var moved = Math.sqrt(len2(tank.x - this.lastX, tank.y - this.lastY));
      if (moved < 14) this.stuck += 0.5;
      else this.stuck = 0;
      this.lastX = tank.x; this.lastY = tank.y; this.checkT = this.t;
      if (this.stuck > 1.0 && this.recover <= 0) {
        this.stuck = 0;
        this.recover = 0.55;
        this.recoverSeq = (this.recoverSeq || 0) + 1;
        /* 后面空就走，后面堵就转着走 */
        var bkClear = !w.map.solidAt(
          tank.x - Math.cos(tank.angle) * (r.tr + 22),
          tank.y - Math.sin(tank.angle) * (r.tr + 22)
        );
        this.recoverDir = Math.random() < 0.5 ? -1 : 1;
        this.recoverDrive = bkClear ? -1 : 1;
        this.goal = -1;
      }
    }

    /* ---- 规划 ---- */
    this.planTimer -= dt;
    if (this.planTimer <= 0) {
      this.planTimer = 1 / Q.planHz;
      this.markDoomed(tank, w);
      var md = this.mode(tank, w);
      var shot = this.pickShot(tank, w);
      this.pressure = this.threats(tank, w).length;
      this.snap = this.preThreats(tank, w).length > 0;
      var wp = this.chooseGoal(tank, w, md);
      this.plan = { mode: md, shot: shot, wp: wp, steer: 0, drive: 0 };
      this.dodge = null;
    }

    var pl = this.plan;
    var md2 = pl ? pl.mode : 'prowl';
    var shot2 = pl ? pl.shot : null;

    /* ---- 恢复：先脱困 ---- */
    if (this.recover > 0) {
      this.recover -= dt;
      this.dbg.st = 'recover';
      return { drive: this.recoverDrive || -1, steer: this.recoverDir, shoot: false };
    }

    /* ---- 规避：真会被弹到时最高优先 ---- */
    this.dodgeTimer -= dt;
    if (this.dodgeTimer <= 0) {
      this.dodgeTimer = 1 / Q.dodgeHz;
      this.dodge = this.solveDodge(tank, w);
    }
    if (this.dodge) {
      pl.steer = this.dodge.steer;
      pl.drive = this.dodge.drive;
      this.fireDelay = Math.max(this.fireDelay, 0.18);
      this.dbg.st = 'dodge';
      return { drive: this.dodge.drive, steer: this.dodge.steer, shoot: false };
    }

    /* ---- 交火对象与姿态 ---- */
    var ne = this.nearest(tank, w);
    var visNe = null;
    if (ne && w.map.hasLOS(tank.x, tank.y, ne.o.x, ne.o.y)) visNe = ne;

    var usable = shot2 && shot2.hitIdx >= 0 && shot2.hitIdx !== tank.index && shot2.score > Q.minShotScore;
    var sup = (shot2 && shot2.suppress) ? shot2.suppress : null;
    var wantSuppress = !usable && !!sup;
    var aimAng = null;
    var targetAng = null;

    if (visNe) targetAng = Math.atan2(visNe.o.y - tank.y, visNe.o.x - tank.x);

    /* 什么时候进入「对枪姿态」：本来就要打，或者被看得见的人逼到近前 */
    var dance = (md2 !== 'prowl' && (usable || visNe)) || (visNe && visNe.d < Q.danceRange) || this.snap || wantSuppress;

    if (usable) aimAng = shot2.angle;
    else if (wantSuppress) aimAng = sup.angle;
    else if (dance && visNe) {
      var th = this.tracks[visNe.o.index];
      var lead = visNe.d / r.bs;
      aimAng = Math.atan2(visNe.o.y + th.vy * lead - tank.y, visNe.o.x + th.vx * lead - tank.x);
    }

    var moveAng = pl ? Math.atan2(pl.wp.y - tank.y, pl.wp.x - tank.x) : tank.angle;

    /* 交火精度随距离变化：19px 杀伤半径在 100px 处允许 9°，在 400px 处只允许 2.4° */
    function arcFor(range, r) {
      var a = Math.atan((r.tr + r.br) * Q.arcK / Math.max(45, range));
      return clamp(a, Q.arcMin, Q.arcMax);
    }

    var shotRange = 999;
    if (usable) {
      for (var vi = 0; vi < w.alive.length; vi++) {
        if (w.alive[vi].index === shot2.hitIdx) {
          shotRange = Math.sqrt(len2(w.alive[vi].x - tank.x, w.alive[vi].y - tank.y));
          break;
        }
      }
    }
    var fireArc = arcFor(shotRange, r);

    /* ---- 姿态机：开枪—侧移—回身 ---- */
    this.orbitT -= dt;
    if (this.orbitT <= 0 && this.orbitPhase === 'run') this.orbitPhase = 'aim';
    if (!dance) this.orbitPhase = 'aim';

    var primary;
    var drive;

    if (dance) {
      if (this.orbitPhase === 'run' && targetAng !== null) {
        /* 侧移阶段：把车头转到对手方向的切线，沿弧线跑 —— 他的提前量全废 */
        var arc = targetAng + this.jink * Q.orbitArc;
        if (w.map.solidAt(tank.x + Math.cos(arc) * (r.tr + 26), tank.y + Math.sin(arc) * (r.tr + 26))) {
          arc = targetAng - this.jink * Q.orbitArc;
          if (w.map.solidAt(tank.x + Math.cos(arc) * (r.tr + 26), tank.y + Math.sin(arc) * (r.tr + 26))) {
            arc = targetAng + PI;        // 两头都堵：直接倒车脱离
          }
        }
        primary = arc;
        var runBack = Math.abs(angDiff(tank.angle, arc)) > 2.3;
        drive = runBack ? -0.85 : 1;
        this.aimStall = 0;
      } else {
        /* 锁定阶段：炮管跟着目标，用倒车维持距离 */
        primary = aimAng !== null ? aimAng : targetAng;
        var rng = visNe ? visNe.d : (usable ? 200 : 999);
        var backClear = !w.map.solidAt(
          tank.x - Math.cos(tank.angle) * (r.tr + 24),
          tank.y - Math.sin(tank.angle) * (r.tr + 24)
        );
        var wantMin = md2 === 'duel' ? Q.duelRange : Q.closeRange;
        /* 对手刚开火 → 0.42 秒内他打不了我，这是唯一的免费突进窗口 */
        var vt2 = visNe ? this.tracks[visNe.o.index] : null;
        var punish = !!(vt2 && this.t - vt2.lastShotT < 0.4);

        if (rng < 110 && !usable && !backClear) {
          /* 贴脸开不了枪、后面又是墙：转身朝最开的方向脱离，先拉开距离 */
          var away = targetAng !== null ? targetAng + PI : primary + PI;
          var cl = this.clearance(tank, w, away, r, true);
          primary = cl ? cl.a : away;
          drive = 1;
        } else if (punish && rng > 95) {
          /* 他还在冷却，可以压上——但仍是有控制地压上，不是白给 */
          drive = 0.65;
        } else if (usable && Math.abs(angDiff(tank.angle, aimAng)) < 0.3) {
          /* 有靠谱枪且基本对准 —— 站住微调，别浪费这发 */
          drive = Math.abs(angDiff(tank.angle, aimAng)) < fireArc * 1.4 ? (backClear ? -0.22 : 0.22) : (backClear ? -0.5 : 0.4);
        } else if (rng < wantMin && backClear) drive = -0.9;
        else if (rng > Q.farRange) drive = 0.8;
        else if (rng > Q.wantRange) drive = 0.5;
        else drive = backClear ? -0.35 : 0.25;

        this.aimStall += dt;
        if (this.aimStall > Q.aimHoldMax && visNe) {
          this.orbitPhase = 'run';
          this.orbitT = Q.runTime;
          this.aimStall = 0;
        }
      }
    } else {
      /* 纯走位：把「行进方向」找出一条最空的路线，顶在门口发抖是最坏的 */
      var e2 = angDiff(tank.angle, moveAng);
      var back = Math.abs(e2) > 2.3;
      var travel = back ? moveAng + PI : moveAng;
      var clr = this.clearance(tank, w, travel, r, back);
      primary = (clr ? clr.a : travel) + (back ? PI : 0);
      var align = Math.abs(angDiff(tank.angle, primary));
      if (back) drive = align < 0.55 ? -0.9 : -0.45;
      else drive = align < 0.7 ? 1 : 0.5;

      /* 路线进度监控：长时间靠近不了当前路点（门洞）就拉黑这条路，换路线 */
      if (pl) {
        var gd = Math.sqrt(len2(pl.wp.x - tank.x, pl.wp.y - tank.y));
        if (gd < this.routeBest - 12) { this.routeBest = gd; this.routeT = this.t; }
        else if (this.t - this.routeT > 1.6) {
          var bad = w.model.cellOf(pl.wp.x, pl.wp.y);
          this.badGoals[bad] = this.t + Q.badGoalTtl;
          if (this.goal >= 0) this.badGoals[this.goal] = this.t + Q.badGoalTtl;
          this.goal = -1;
          this.nav = null;
          this.routeBest = Infinity; this.routeT = this.t;
          if (this.recover <= 0) {
            this.recover = 0.3; this.recoverDir = Math.random() < 0.5 ? -1 : 1; this.recoverDrive = -1;
          }
        }
      }
    }

    var err = angDiff(tank.angle, primary);
    if (Math.abs(err) > 0.022) inp.steer = sgn(err);

    /* ---- 绝不停着当靶子 ---- */
    if (Math.abs(tank.speed) < 16 && drive === 0) {
      drive = Math.random() < 0.5 ? -0.6 : 0.6;
    }

    pl.steer = inp.steer;
    pl.drive = drive;
    inp.drive = drive;

    /* ---- 开火 ---- */
    this.fireDelay -= dt;
    if (usable && this.orbitPhase === 'aim' && this.fireDelay <= 0 && tank.cooldown <= 0 &&
        Math.abs(angDiff(tank.angle, aimAng)) <= fireArc && (shot2.canFire !== false)) {
      inp.shoot = true;
      this.fireDelay = Q.fireDelay;
      /* 打完就走：开火 → 立刻转入侧移 */
      this.orbitPhase = 'run';
      this.orbitT = Q.runTime;
      this.aimStall = 0;
      if (Math.random() < 0.4) this.jink = -this.jink;
    } else if (wantSuppress && this.orbitPhase === 'aim' && this.fireDelay <= 0 &&
               tank.cooldown <= 0 && shot2.canFire !== false &&
               Math.abs(angDiff(tank.angle, sup.angle)) < (this.fuse ? 0.4 : 0.12)) {
      /* 僵局压制弹 */
      inp.shoot = true;
      this.fireDelay = this.fuse ? 0.1 : 0.55;
    }

    this.dbg.st = this.orbitPhase === 'run' && dance ? 'orbit' : md2;
    this.dbg.mode = md2;
    this.dbg.aim = aimAng !== null;
    this.dbg.shot = shot2 ? (shot2.kind + ':' + shot2.hitIdx + ':' + shot2.score.toFixed(0)) : '-';
    this.dbg.goal = pl ? (pl.wp.x.toFixed(0) + ',' + pl.wp.y.toFixed(0)) : '-';
    this.dbg.pressure = this.pressure;
    this.dbg.alive = this.aliveCount;
    return inp;
  };

  /* =========================================================
     11. 对外接口
     ========================================================= */

  var brains = new WeakMap();
  var TTRef = null;
  var STATS = { calls: 0, errors: 0, lastError: '', totalMs: 0, maxMs: 0 };

  function makeUpdater(overrides) {
    var params = deriveParams(overrides);

    function run(tank, game, dt) {
      var b = brains.get(tank);
      if (!b) {
        b = new Brain(tank.index);
        b.p = params;
        b.lastX = tank.x; b.lastY = tank.y; b.checkT = 0;
        brains.set(tank, b);
      } else if (b.p !== params) {
        b.p = params;
      }
      STATS.calls++;
      var t0 = Date.now();
      try {
        var out = b.update(tank, game, dt, TTRef);
        b.lastInp = out;
        var ms = Date.now() - t0;
        STATS.totalMs += ms;
        if (ms > STATS.maxMs) STATS.maxMs = ms;
        return out;
      } catch (e) {
        STATS.errors++;
        STATS.lastError = (e && e.message ? e.message : String(e)) + ' @ ' + ((e && e.stack || '').split('\n')[1] || '').trim();
        /* 任何意外都不能让坦克趴窝：退回「慢慢往前蹭 + 别撞墙」 */
        return { drive: 0.4, steer: Math.sin((game && game.time) || 0) > 0 ? 1 : -1, shoot: false };
      }
    }
    run.params = params;
    run.brainOf = function (tank) { return brains.get(tank); };
    return run;
  }

  var update = makeUpdater(null);

  var API = {
    name: 'qwen',
    version: VERSION,
    update: update,
    think: update,
    step: update,
    decide: update,
    onTick: update,
    play: update,
    act: update,
    control: update,
    /* 工厂式接口（按「一辆车一个实例」注册的场景） */
    create: function (tank, game) {
      var b = new Brain(tank ? tank.index : 0);
      var wrap = function (t, g, d) { return b.update(t, g, d, TTRef); };
      return { update: wrap, think: wrap, step: wrap, decide: wrap, brain: b, dbg: b.dbg };
    },
    tune: P,
    params: P,
    stats: STATS,
    /* 调参/自测用：返回一个带参数覆盖的独立 updater */
    variant: makeUpdater,
    setTT: function (tt) { TTRef = tt; },
    selftest: function () {
      return {
        name: API.name, version: VERSION,
        rules: 'bullet 620px/s bounce5 life6 · tank r16 178px/s turn3.3 cd0.42',
        keys: Object.keys(P).length,
      };
    },
    _internals: { Brain: Brain, World: World, MapModel: MapModel, traceShot: traceShot, models: modelCache, brains: brains },
  };

  root.QWEN_TANK = API;
  root.QwenTank = API;
  root.TT_QWEN = API;

  if (root.TT) {
    TTRef = root.TT;
    root.TT.policies = root.TT.policies || {};
    root.TT.POLICIES = root.TT.policies;
    root.TT.policies.qwen = API;
    root.TT.qwenPolicy = API;
    API.setTT(root.TT);
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = API;

})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
