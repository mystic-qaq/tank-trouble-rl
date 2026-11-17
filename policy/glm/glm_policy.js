/* ===========================================================
   glm_policy.js — GLM Tank Trouble 竞技策略 v1.0
   ===========================================================
   设计核心（相对内置 bot 的四大优势）：
   1. 弹道预知：子弹物理完全确定 → 首见时模拟整条反弹轨迹并缓存，
      对每颗子弹预知未来约 1.3s 的路径（含反弹、哑火、超时消亡）
   2. 主动规避：被瞄准时对 9 组机动（转向 x 油门）做 0.6s 滚动模拟，
      选存活期望最高的机动；自己的反弹流弹同样纳入威胁
   3. 反弹猎杀：72 向快速弹道扫描（1.2s / 含反弹 / 敌人位移外推），
      无直瞄时也能绕墙击杀；先弹回自己的弹道自动剔除（防自杀 -1）
   4. 生存走位：BFS 距离场 + 格子评分（隐蔽性 / 敌距 / 弹道密度 / 出口），
      开火后立刻侧向机动规避还击

   计分逻辑：击杀 +1 / 自杀 -1 / 活到最后 = 胜场 → 生存第一，出手必中。
   接口：update(tank, game, dt) -> { drive, steer, shoot }，与 TT.updateBot 同签名。
   =========================================================== */

(function () {
  'use strict';

  /* ======================= 0. 基础工具 ======================= */

  var TAU = Math.PI * 2;
  var DT = 1 / 60;

  function CFG() {
    var TT = (typeof window !== 'undefined' && window.TT) || (typeof global !== 'undefined' && global.TT);
    return (TT && TT.CFG) || {
      MAX_BULLETS_PER_TANK: 5,
      TANK: { radius: 16, speed: 178, reverseRatio: 0.66, accel: 920, brake: 1150, turn: 3.3 },
      BULLET: { speed: 620, radius: 3.2, bounces: 5, life: 6, spawnGrace: 0.07 },
    };
  }

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function sign(v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }
  function dist(ax, ay, bx, by) { var dx = bx - ax, dy = by - ay; return Math.sqrt(dx * dx + dy * dy); }
  /* 最短角度差，(-PI, PI]，>0 表示需要增大角度（顺时针） */
  function angleDiff(a, b) {
    var d = (b - a + Math.PI) % TAU;
    if (d < 0) d += TAU;
    return d - Math.PI;
  }

  /* 可调参数 */
  var P = {
    dodgeWindow: 0.55,        // 紧急威胁窗口（秒）
    threatLook: 1.30,         // 威胁前瞻窗口（秒）
    hitPad: 3.5,              // 命中判定安全余量 px
    scanInterval: 0.16,       // 反弹扫描间隔（秒）
    scanAngles: 72,           // 反弹扫描角度数
    scanDist: 700,            // 反弹扫描最大弹程（px）
    bounceKillMax: 1.10,      // 接受的反弹击杀最长耗时（秒）
    engageMax: 560,           // 直瞄开火最大距离
    evadeTime: 0.22,          // 射后机动时长（秒）
    goalInterval: 0.45,       // 走位目标重选间隔（秒）
    targetInterval: 0.30,     // 攻击目标重选间隔（秒）
    stuckTime: 0.60,          // 卡住判定（秒）
    unstickTime: 0.50,        // 自救时长（秒）
    rampStart: 35,            // 僵局打破：开始提激进度的回合时间（秒）
    rampRate: 1 / 45,         // 激进度每秒增量（35s 后线性上升）
    suppressCd: 1.0,          // 压制射击最小间隔（秒）
  };

  /* ================= 1. 子弹轨迹：确定性模拟 + 缓存 ================= */
  /* 物理与引擎 Bullet.update 完全一致（同一 map.raycast），逐帧 1/60。 */

  var bulletCache = new WeakMap();   // Bullet -> { pts: [{life, x, y}], origin }

  function simulateTrail(map, b) {
    var cfg = CFG();
    var pts = [];
    var x = b.x, y = b.y, angle = b.angle;
    var bounces = b.bounces || 0;
    var life = b.life;
    var speed = b.speed || cfg.BULLET.speed;
    var guardOuter = 0;

    while (life > 0 && guardOuter++ < 420) {
      pts.push({ life: life, x: x, y: y });
      life -= DT;
      var remaining = speed * DT;
      var guard = 0;
      while (remaining > 0.02 && life > 0 && guard++ < 8) {
        var dx = Math.cos(angle) * remaining;
        var dy = Math.sin(angle) * remaining;
        var hit = map.raycast(x, y, x + dx, y + dy);
        if (!hit) { x += dx; y += dy; break; }
        var t = clamp(hit.t, 0, 1);
        x += dx * t; y += dy * t;
        remaining -= remaining * t;
        var nx = hit.nx || 0, ny = hit.ny || 0;
        if (nx === 0 && ny === 0) { life = 0; break; }        // 炮口在墙里：哑火
        var vx = Math.cos(angle), vy = Math.sin(angle);
        var dot = vx * nx + vy * ny;
        if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; }
        angle = Math.atan2(vy, vx);
        x += nx * 0.8; y += ny * 0.8;
        bounces++;
        if (bounces > cfg.BULLET.bounces) { life = 0; break; } // 第 6 次撞墙消失
        remaining -= 0.7;
      }
    }
    return { pts: pts };
  }

  function getTrail(b, map) {
    var c = bulletCache.get(b);
    if (!c) { c = simulateTrail(map, b); bulletCache.set(b, c); }
    return c;
  }

  /* ================= 2. 快速弹道（DDA 网格步进，用于扫描） ================= */
  /* 只近似格边界墙（忽略墙段端点 5px 延伸），比引擎全墙扫描快两个数量级。 */

  var gridCache = new WeakMap();     // GameMap -> grid

  function grid(map) {
    var g = gridCache.get(map);
    if (g) return g;
    g = {
      cols: map.cols, rows: map.rows, cell: map.cell,
      vAt: function (r, c) { return map.vAt(r, c); },
      hAt: function (r, c) { return map.hAt(r, c); },
    };
    gridCache.set(map, g);
    return g;
  }

  /* 从 (x,y) 沿方向前进，找到 maxT 内的第一面格边界墙；无则 null */
  function marchWall(g, x, y, nx, ny, maxT) {
    var cell = g.cell;
    var c0 = Math.floor(x / cell), r0 = Math.floor(y / cell);
    var bestT = Infinity, bnx = 0, bny = 0;

    if (nx > 1e-9 || nx < -1e-9) {
      var cc = nx > 0 ? c0 + 1 : c0;
      var t = (cc * cell - x) / nx;
      if (t >= 0 && t <= maxT && t < bestT) {
        var rr = Math.floor((y + ny * t) / cell);
        if (rr < 0) rr = 0; else if (rr >= g.rows) rr = g.rows - 1;
        var wall = (cc <= 0 || cc >= g.cols) ? 1 : g.vAt(rr, cc);
        if (wall) { bestT = t; bnx = nx > 0 ? -1 : 1; bny = 0; }
      }
    }
    if (ny > 1e-9 || ny < -1e-9) {
      var r2 = ny > 0 ? r0 + 1 : r0;
      var t2 = (r2 * cell - y) / ny;
      if (t2 >= 0 && t2 <= maxT && t2 < bestT) {
        var cc2 = Math.floor((x + nx * t2) / cell);
        if (cc2 < 0) cc2 = 0; else if (cc2 >= g.cols) cc2 = g.cols - 1;
        var wall2 = (r2 <= 0 || r2 >= g.rows) ? 1 : g.hAt(r2, cc2);
        if (wall2) { bestT = t2; bnx = 0; bny = ny > 0 ? -1 : 1; }
      }
    }
    if (bestT === Infinity) return null;
    return { t: bestT, nx: bnx, ny: bny };
  }

  /*
   * traceShot: 模拟一发子弹（含反弹），每帧步进 probe。
   * probe(x, y, frame, dir) 返回非空 verdict 时立即终止并返回。
   * 返回 { verdict, x, y, dir }
   */
  function traceShot(g, sx, sy, angle, maxDist, probe) {
    var cfg = CFG();
    var x = sx, y = sy, dir = angle, traveled = 0, bounces = 0;
    var STEP = cfg.BULLET.speed * DT;
    var maxFrames = Math.min(150, Math.ceil(maxDist / STEP));

    for (var frame = 1; frame <= maxFrames; frame++) {
      var rem = STEP, guard = 0;
      var dead = false;
      while (rem > 0.4 && guard++ < 5) {
        var dx = Math.cos(dir), dy = Math.sin(dir);
        var res = marchWall(g, x, y, dx, dy, rem);
        if (!res) { x += dx * rem; y += dy * rem; rem = 0; break; }
        x += dx * res.t; y += dy * res.t;
        var dot = dx * res.nx + dy * res.ny;
        if (dot < 0) {
          var rx = dx - 2 * dot * res.nx, ry = dy - 2 * dot * res.ny;
          dir = Math.atan2(ry, rx);
        }
        x += res.nx * 0.8; y += res.ny * 0.8;
        bounces++;
        rem -= res.t + 0.7;
        if (rem < 0) rem = 0;
        if (bounces > cfg.BULLET.bounces) { dead = true; break; }
      }
      if (dead) return { verdict: null, x: x, y: y, dir: dir };
      traveled += STEP;
      if (probe) {
        var v = probe(x, y, frame, dir);
        if (v) return { verdict: v, x: x, y: y, dir: dir };
      }
    }
    return { verdict: null, x: x, y: y, dir: dir };
  }

  /* 敌人直线外推：沿当前速度方向，撞墙即停（一次 raycast 求自由程） */
  function makeExts(map, enemies, frames) {
    var exts = new Array(enemies.length);
    for (var i = 0; i < enemies.length; i++) {
      var e = enemies[i];
      var sp = e.speed || 0;
      var cos = Math.cos(e.angle), sin = Math.sin(e.angle);
      var dir = sp >= 0 ? 1 : -1;
      var spd = Math.abs(sp);
      var free = 400;
      if (spd > 5) {
        var hit = map.raycast(e.x, e.y, e.x + cos * dir * 300, e.y + sin * dir * 300);
        free = (hit ? hit.t * 300 : 300) - 14;
        if (free < 0) free = 0;
      }
      var arr = new Array(frames);
      for (var k = 0; k < frames; k++) {
        var d = spd * DT * k;
        if (d > free) d = free;
        arr[k] = { x: e.x + cos * dir * d, y: e.y + sin * dir * d };
      }
      exts[i] = arr;
    }
    return exts;
  }

  /* ===================== 3. 威胁评估 ===================== */

  /*
   * 对场上所有子弹（含自己的流弹）检测「未来 threatLook 秒内是否会撞上我」。
   * 返回 [{b, pts, idx, tHit, hitFrame}]，按 tHit 升序。
   */
  function assessThreats(tank, game) {
    var map = game.map;
    var R = CFG().TANK.radius + CFG().BULLET.radius + P.hitPad;
    var R2 = R * R;
    var look = Math.ceil(P.threatLook / DT);
    var out = [];

    for (var i = 0; i < game.bullets.length; i++) {
      var b = game.bullets[i];
      if (b.dead) continue;
      var trail = getTrail(b, map);
      var pts = trail.pts;
      if (!pts.length) continue;
      var idx = Math.round((pts[0].life - b.life) / DT);
      if (idx < 0) idx = 0;
      var end = Math.min(pts.length, idx + look);
      var meX = tank.x, meY = tank.y;

      for (var k = idx; k < end; k++) {
        var p = pts[k];
        var dx = p.x - meX, dy = p.y - meY;
        if (dx * dx + dy * dy < R2) {
          out.push({ b: b, pts: pts, idx: idx, tHit: (k - idx) * DT, hitFrame: k });
          break;
        }
      }
    }
    out.sort(function (a, c) { return a.tHit - c.tHit; });
    return out;
  }

  /* ===================== 4. 紧急躲避 ===================== */

  /* 9 组候选机动（steer x drive），滚动模拟 0.6s，选存活期望最高者 */
  function decideDodge(tank, game, urgent, out) {
    var cfg = CFG();
    var map = game.map;
    var T = cfg.TANK;
    var R = T.radius + cfg.BULLET.radius + P.hitPad;
    var R2 = R * R;
    var STEPS = 36;
    var best = null, bestScore = -Infinity;

    for (var si = 0; si < 3; si++) {
      var steer = si - 1;
      for (var di = 0; di < 3; di++) {
        var drive = di - 1;
        var sim = { x: tank.x, y: tank.y, speed: tank.speed, angle: tank.angle };
        var deadAt = -1, minClear = 1e9;
        var cosA = Math.cos(sim.angle), sinA = Math.sin(sim.angle);

        for (var k = 0; k < STEPS; k++) {
          /* 速度模型（照抄引擎） */
          var target = drive > 0 ? T.speed : (drive < 0 ? -T.speed * T.reverseRatio : 0);
          var rate = (sim.speed * drive < 0) ? T.brake : (drive === 0 ? T.brake * 0.75 : T.accel);
          if (sim.speed < target) sim.speed = Math.min(target, sim.speed + rate * DT);
          else if (sim.speed > target) sim.speed = Math.max(target, sim.speed - rate * DT);

          sim.angle += steer * T.turn * DT;
          cosA = Math.cos(sim.angle); sinA = Math.sin(sim.angle);
          sim.x += cosA * sim.speed * DT;
          sim.y += sinA * sim.speed * DT;
          map.resolveCircle(sim, T.radius);

          for (var u = 0; u < urgent.length; u++) {
            var th = urgent[u];
            var p = th.pts[th.idx + k];
            if (!p) continue;
            var ddx = p.x - sim.x, ddy = p.y - sim.y;
            var dd2 = ddx * ddx + ddy * ddy;
            if (dd2 < R2) {
              if (deadAt < 0) deadAt = k;
            } else {
              var dd = Math.sqrt(dd2);
              if (dd < minClear) minClear = dd;
            }
          }
          if (deadAt >= 0) break;   // 已死，后面不用算
        }

        var s = (deadAt < 0 ? 1000 : deadAt * 30) + Math.min(minClear, 90) * 0.6;
        s -= Math.abs(steer) * 0.4 + Math.abs(drive) * 0.25;   // 减少无谓抖动
        if (s > bestScore) { bestScore = s; best = { steer: steer, drive: drive }; }
      }
    }
    if (best) { out.steer = best.steer; out.drive = best.drive; }
  }

  /* 躲避途中若恰好已对准直瞄目标，白赚一枪 */
  function opportuneShot(tank, game, enemies, out, brain) {
    var map = game.map;
    for (var i = 0; i < enemies.length; i++) {
      var e = enemies[i];
      if (!map.hasLOS(tank.x, tank.y, e.x, e.y)) continue;
      var d = dist(tank.x, tank.y, e.x, e.y);
      var aim = Math.atan2(e.y - tank.y, e.x - tank.x);
      if (Math.abs(angleDiff(tank.angle, aim)) < 0.05 &&
          canFire(tank, game) && muzzleClear(tank, map)) {
        out.shoot = true;
        brain.lastShotAt = brain.t;
      }
      break;   // 只看最近的一个
    }
  }

  /* ===================== 5. 直瞄交战 ===================== */

  /* 比例转向：接近目标角时降速，避免满速转向在窄窗口上振荡。
     返回角度差（供调用方做开火判定）。 */
  function steerTowards(out, tank, want, dt) {
    var diff = angleDiff(tank.angle, want);
    var ad = Math.abs(diff);
    var frame = CFG().TANK.turn * dt;
    if (ad > 0.02) {
      var k = ad < frame * 2 ? ad / (frame * 2) : 1;
      out.steer = sign(diff) * k;
    }
    return diff;
  }

  function canFire(tank, game) {
    var cfg = CFG();
    var live = 0;
    for (var i = 0; i < game.bullets.length; i++) {
      var b = game.bullets[i];
      if (!b.dead && b.owner === tank.index) live++;
    }
    return live < cfg.MAX_BULLETS_PER_TANK;
  }

  /* 炮口是否可以开火：仅要求炮口点不在墙内（引擎哑火条件）。
     贴墙反弹的弹道由 DDA 扫描全程模拟并已剔除自伤角，无需额外保守。 */
  function muzzleClear(tank, map) {
    var r = tank.radius + 12;
    var tx = tank.x + Math.cos(tank.angle) * r;
    var ty = tank.y + Math.sin(tank.angle) * r;
    return !map.solidAt(tx, ty);
  }

  /* signMul = 1 查前方，-1 查后方 */
  function blockedAhead(map, tank, signMul) {
    var d = (tank.radius + 18) * signMul;
    return map.solidAt(tank.x + Math.cos(tank.angle) * d,
                       tank.y + Math.sin(tank.angle) * d);
  }

  /*
   * 直瞄交战：预判瞄准 + 距离管理 + 开火闸门。
   * evading = true 时（射后机动中）只补枪、不重设 evade。
   */
  function tryEngage(tank, target, game, brain, out, evading, dt) {
    var cfg = CFG();
    var map = game.map;
    var d = dist(tank.x, tank.y, target.x, target.y);

    /* 预判（迭代两次），0.85 折减敌机转向损失 */
    var t = d / cfg.BULLET.speed;
    var ax = target.x, ay = target.y;
    for (var i = 0; i < 2; i++) {
      ax = target.x + Math.cos(target.angle) * (target.speed || 0) * t * 0.85;
      ay = target.y + Math.sin(target.angle) * (target.speed || 0) * t * 0.85;
      t = dist(tank.x, tank.y, ax, ay) / cfg.BULLET.speed;
    }
    /* 预判点被墙挡住则退回目标当前位置 */
    if (!map.hasClearShot(tank.x, tank.y, ax, ay, 10)) { ax = target.x; ay = target.y; }

    var aim = Math.atan2(ay - tank.y, ax - tank.x);
    var ad = Math.abs(steerTowards(out, tank, aim, dt));

    /* 距离管理 */
    var frontBlocked = blockedAhead(map, tank, 1);
    var backBlocked = blockedAhead(map, tank, -1);
    if (d < 150) {
      out.drive = backBlocked ? 0 : -1;
    } else if (d > 380) {
      out.drive = frontBlocked ? 0 : 1;
    } else if (ad < 0.4) {
      out.drive = frontBlocked ? 0 : 0.5;
    }

    /* 开火闸门 */
    var thr = clamp(0.085 - d * 0.00012, 0.03, 0.085);
    if (d < 130) thr = 0.11;                       // 贴脸必中窗口放宽
    if (d > P.engageMax * (brain.aggression || 1)) return;
    if (ad < thr && canFire(tank, game) && muzzleClear(tank, map)) {
      out.shoot = true;
      brain.lastShotAt = brain.t;
      if (!evading) {
        brain.evade = P.evadeTime;
        brain.evadeDir = Math.random() < 0.5 ? -1 : 1;
      }
      return;
    }

    /* 压制射击：对峙时朝敌人方向施压，逼其移动露出破绽。
       自己的流弹威胁由弹道预知兜底（assessThreats 含自弹）。 */
    var aggr = brain.aggression || 1;
    if (ad < 0.3 && d < 420 && canFire(tank, game) && muzzleClear(tank, map) &&
        brain.t - brain.lastShotAt > P.suppressCd / aggr) {
      out.shoot = true;
      brain.lastShotAt = brain.t;
    }
  }

  /* ===================== 6. 反弹猎杀（弹道扫描） ===================== */

  /* 全向扫描：找出「先命中敌人、且不会先弹回自己」的最佳角度；
     同时记录掠过敌人最近的角度作为僵局骚扰候选 */
  function bounceScan(tank, game, enemies, phase, maxDist) {
    var map = game.map;
    var g = grid(map);
    var cfg = CFG();
    var FRAMES = 70;
    var exts = makeExts(map, enemies, FRAMES);
    var n = P.scanAngles;
    var off = (phase & 1) ? (TAU / (n * 2)) : 0;
    var best = null, bestScore = -Infinity;
    var harass = null, harassScore = Infinity;
    var dist = maxDist || P.scanDist;

    for (var i = 0; i < n; i++) {
      var a = i / n * TAU + off;
      var sx = tank.x + Math.cos(a) * (tank.radius + 12);
      var sy = tank.y + Math.sin(a) * (tank.radius + 12);
      if (map.solidAt(sx, sy)) continue;           // 炮口在墙里：必哑火

      var probe = makeProbe(tank, enemies, exts, FRAMES);
      var res = traceShot(g, sx, sy, a, dist, probe);
      if (res.verdict && res.verdict.kind === 'kill') {
        var score = 2.0 - res.verdict.frame * DT;
        if (res.verdict.target === enemies[0]) score += 0.25;   // 首选目标加成
        if (score > bestScore) {
          bestScore = score;
          best = { angle: a, tKill: res.verdict.frame * DT, targetIdx: res.verdict.target.index };
        }
      } else if (!res.verdict && probe.state.minD < 75 * 75) {
        /* 骚扰候选：弹道从敌人身边掠过 */
        if (probe.state.minD < harassScore) {
          harassScore = probe.state.minD;
          harass = { angle: a, minD: Math.sqrt(probe.state.minD), targetIdx: probe.state.target ? probe.state.target.index : -1 };
        }
      }
    }
    return { kill: best, harass: harass };
  }

  function makeProbe(tank, enemies, exts, frames) {
    var killR2 = 19.2 * 19.2;    // 敌 r16 + 弹 r3.2
    var selfR2 = 23 * 23;        // 自己 + 余量
    var meX = tank.x, meY = tank.y;
    var st = { minD: Infinity, target: null };
    var fn = function (x, y, frame, dir) {
      for (var i = 0; i < enemies.length; i++) {
        var p = exts[i][frame < frames ? frame : frames - 1];
        var dx = x - p.x, dy = y - p.y;
        var d2 = dx * dx + dy * dy;
        if (d2 < st.minD) { st.minD = d2; st.target = enemies[i]; }
        if (d2 < killR2) return { kind: 'kill', target: enemies[i], frame: frame };
      }
      if (frame > 6) {           // 跳过出生 grace 期（约 0.1s）
        var ddx = x - meX, ddy = y - meY;
        if (ddx * ddx + ddy * ddy < selfR2) return { kind: 'self', frame: frame };
      }
      return null;
    };
    fn.state = st;
    return fn;
  }

  /* 转向过程中每帧校验缓存角度是否仍然成立 */
  function verifyBounce(tank, game, bounce) {
    var map = game.map;
    var enemies = aliveEnemies(tank, game);
    if (!enemies.length) return null;
    var g = grid(map);
    var exts = makeExts(map, enemies, 70);
    var sx = tank.x + Math.cos(bounce.angle) * (tank.radius + 12);
    var sy = tank.y + Math.sin(bounce.angle) * (tank.radius + 12);
    if (map.solidAt(sx, sy)) return null;

    var res = traceShot(g, sx, sy, bounce.angle, P.scanDist, makeProbe(tank, enemies, exts, 70));
    if (res.verdict && res.verdict.kind === 'kill') {
      bounce.tKill = res.verdict.frame * DT;
      bounce.targetIdx = res.verdict.target.index;
      return bounce;
    }
    return null;
  }

  /* ===================== 7. 走位（BFS 距离场 + 格子评分） ===================== */

  function bfsField(map, r0, c0) {
    var COLS = map.cols, ROWS = map.rows;
    var f = new Int16Array(ROWS * COLS);
    for (var z = 0; z < f.length; z++) f[z] = -1;
    var q = [r0 * COLS + c0];
    f[q[0]] = 0;
    for (var h = 0; h < q.length; h++) {
      var cur = q[h], r = (cur / COLS) | 0, c = cur % COLS, d = f[cur];
      if (c < COLS - 1 && !map.vAt(r, c + 1) && f[cur + 1] < 0) { f[cur + 1] = d + 1; q.push(cur + 1); }
      if (c > 0 && !map.vAt(r, c) && f[cur - 1] < 0) { f[cur - 1] = d + 1; q.push(cur - 1); }
      if (r < ROWS - 1 && !map.hAt(r + 1, c) && f[cur + COLS] < 0) { f[cur + COLS] = d + 1; q.push(cur + COLS); }
      if (r > 0 && !map.hAt(r, c) && f[cur - COLS] < 0) { f[cur - COLS] = d + 1; q.push(cur - COLS); }
    }
    return f;
  }

  /* 选巡逻目标格：隐蔽、敌距适中、远离流弹、出口多 */
  function pickGoal(tank, game, enemies, threats, brain) {
    var map = game.map;
    var cfg = CFG();
    var cell = map.cell;
    var myC = clamp(Math.floor(tank.x / cell), 0, map.cols - 1);
    var myR = clamp(Math.floor(tank.y / cell), 0, map.rows - 1);
    var field = bfsField(map, myR, myC);

    var nAlive = enemies.length + 1;
    var aggr = brain.aggression || 1;
    var ideal = (nAlive <= 2 ? 3.2 * cell : 5.2 * cell) / aggr;   // 与最近敌人的理想距离（僵局时贴身）
    /* 只对最近的 3 个敌人做直瞄检查（便宜） */
    var near = enemies.slice().sort(function (a, b) {
      return dist2(tank, a) - dist2(tank, b);
    }).slice(0, 3);

    var best = null, bestScore = -1e9;
    for (var r = 1; r < map.rows - 1; r++) {
      for (var c = 1; c < map.cols - 1; c++) {
        var d = field[r * map.cols + c];
        if (d < 1 || d > 9) continue;
        var gx = c * cell + cell / 2, gy = r * cell + cell / 2;
        var s = 0;

        /* 与最近敌人的距离保持理想区间 */
        var nearest = 1e9;
        for (var i = 0; i < enemies.length; i++) {
          var e = enemies[i];
          var dd = dist(gx, gy, e.x, e.y);
          if (dd < nearest) nearest = dd;
        }
        if (nearest < 1e8) s -= Math.abs(nearest - ideal) * 0.0045;

        /* 隐蔽性：敌人看不到这格加分，看得到扣分（僵局时权重衰减，鼓励贴身） */
        var stealthW = 1 / aggr;
        for (var j = 0; j < near.length; j++) {
          var e2 = near[j];
          if (map.hasClearShot(e2.x, e2.y, gx, gy, 14)) s -= 0.7 * stealthW;
          else s += 0.5 * stealthW;
        }

        /* 远离逼近中的流弹 */
        for (var u = 0; u < threats.length; u++) {
          var th = threats[u];
          var p = th.pts[th.idx + 12] || th.b;   // 约 0.2s 后的位置
          if (dist(gx, gy, p.x, p.y) < 75) s -= 1.4;
        }

        s += map.openSides(r, c) * 0.14;         // 机动性

        /* 开局向中心靠拢找敌人 */
        if (brain.t < 8) {
          s -= dist(gx, gy, map.w / 2, map.h / 2) * 0.0012;
        }

        if (r === myR && c === myC) s += 0.35;   // 留在原地的小惯性
        /* 目标格惯性：避免评分抖动导致走廊内往返 */
        if (brain.goalCell && r === brain.goalCell.r && c === brain.goalCell.c) s += 1.0;
        s += Math.random() * 0.3;

        if (s > bestScore) { bestScore = s; best = { x: gx, y: gy, r: r, c: c }; }
      }
    }
    if (best) { brain.goalCell = { r: best.r, c: best.c }; return { x: best.x, y: best.y }; }
    return null;
  }

  function dist2(a, b) { var dx = b.x - a.x, dy = b.y - a.y; return dx * dx + dy * dy; }

  /* 沿 BFS 路径推进（取直视最远路点），无路径时直奔最近敌人 */
  function patrol(tank, game, brain, enemies, threats, out) {
    var map = game.map;
    brain.goalTimer -= 1 / 60;
    if (brain.goalTimer <= 0 || !brain.path || !brain.path.length) {
      brain.goalTimer = P.goalInterval;
      var goal = pickGoal(tank, game, enemies, threats, brain);
      brain.path = goal ? map.findPath(tank.x, tank.y, goal.x, goal.y) : null;
      if (brain.path && brain.path.length) brain.path.shift();   // 第一个点是脚下
    }

    var wp = null;
    if (brain.path && brain.path.length) {
      var guard = 0;
      while (brain.path.length > 1 && guard++ < 8 &&
             dist(tank.x, tank.y, brain.path[0].x, brain.path[0].y) < 42) {
        brain.path.shift();
      }
      var idx = 0;
      var maxAhead = Math.min(brain.path.length - 1, 4);
      for (var i = maxAhead; i >= 0; i--) {
        if (map.hasLOS(tank.x, tank.y, brain.path[i].x, brain.path[i].y)) { idx = i; break; }
      }
      wp = brain.path[idx];
    }
    if (!wp && enemies.length) wp = enemies[0];

    if (wp) {
      var want = Math.atan2(wp.y - tank.y, wp.x - tank.x);
      var ad = Math.abs(steerTowards(out, tank, want, 1 / 60));
      out.drive = ad > 1.9 ? -1 : (ad > 0.6 ? 0.45 : 1);

      /* 前方堵住就小幅转向 */
      if (out.drive > 0 && map.solidAt(
            tank.x + Math.cos(tank.angle) * (tank.radius + 18),
            tank.y + Math.sin(tank.angle) * (tank.radius + 18))) {
        out.drive = 0.3;
        out.steer = brain.reverseDir;
      }
    }
  }

  /* ===================== 8. 主决策状态机 ===================== */

  var brains = new WeakMap();    // Tank -> brain

  function getBrain(tank) {
    var b = brains.get(tank);
    if (!b) {
      b = {
        t: 0,
        scanTimer: 0,
        scanPhase: 0,
        bounce: null,
        path: null,
        goalTimer: 0,
        target: null,
        targetTimer: 0,
        evade: 0,
        evadeDir: 1,
        stuck: 0,
        reverse: 0,
        reverseDir: 1,
        lastShotAt: -9,
      };
      brains.set(tank, b);
    }
    return b;
  }

  function aliveEnemies(tank, game) {
    var out = [];
    for (var i = 0; i < game.tanks.length; i++) {
      var t = game.tanks[i];
      if (t !== tank && t.alive) out.push(t);
    }
    return out;
  }

  /* 目标选择：最近优先 + 背对我加成 */
  function pickTarget(tank, game, brain) {
    brain.targetTimer -= 1 / 60;
    var cur = brain.target;
    if (cur && cur.alive && cur !== tank && brain.targetTimer > 0) return cur;
    brain.targetTimer = P.targetInterval;

    var best = null, bestScore = -Infinity;
    for (var i = 0; i < game.tanks.length; i++) {
      var t = game.tanks[i];
      if (t === tank || !t.alive) continue;
      var d = dist(tank.x, tank.y, t.x, t.y);
      var s = -d * 0.01;
      /* 敌人背对我（他的朝向背离我）时优先偷 */
      var toMe = Math.atan2(tank.y - t.y, tank.x - t.x);
      if (Math.abs(angleDiff(t.angle, toMe)) < 1.2) s += 0.6;
      if (s > bestScore) { bestScore = s; best = t; }
    }
    brain.target = best;
    return best;
  }

  /* 卡住自救（最高优先级） */
  function handleStuck(tank, brain, dt, out) {
    /* 位置卡住检测：想动但 0.5s 位移 < 6px（顶墙空转）→ 强制倒车 */
    brain.posTimer = (brain.posTimer || 0) + dt;
    if (brain.posTimer >= 0.5) {
      var moved = dist(tank.x, tank.y,
        brain.lastX === undefined ? tank.x - 99 : brain.lastX,
        brain.lastY === undefined ? tank.y - 99 : brain.lastY);
      if (moved < 6 && brain.lastDrive) {
        brain.reverse = Math.max(brain.reverse, P.unstickTime);
        brain.reverseDir = Math.random() < 0.5 ? -1 : 1;
        brain.path = null;
        brain.goalTimer = 0;
        brain.stuck = 0;
      }
      brain.lastX = tank.x; brain.lastY = tank.y; brain.posTimer = 0;
    }

    if (Math.abs(tank.speed) < 24 && brain.lastDrive) brain.stuck += dt;
    else brain.stuck = Math.max(0, brain.stuck - dt * 2.5);

    if (brain.stuck > P.stuckTime) {
      brain.stuck = 0;
      brain.reverse = P.unstickTime;
      brain.reverseDir = Math.random() < 0.5 ? -1 : 1;
      brain.path = null;
      brain.goalTimer = 0;
    }
    if (brain.reverse > 0) {
      brain.reverse -= dt;
      out.drive = -1;
      out.steer = brain.reverseDir;
      return true;
    }
    return false;
  }

  /* 主入口：与 TT.updateBot 同签名 */
  function step(tank, game, dt) {
    var out = { drive: 0, steer: 0, shoot: false };
    if (!game || !game.map || !tank || !tank.alive) return out;
    var cfg = CFG();
    var map = game.map;
    var brain = getBrain(tank);
    brain.t += dt;

    var enemies = aliveEnemies(tank, game);
    if (!enemies.length) return out;

    /* 僵局打破：回合时间越长越激进（更贴身 / 更频开火 / 接受更长反弹弹道） */
    brain.aggression = 1 + Math.max(0, brain.t - P.rampStart) * P.rampRate;

    /* 1. 威胁评估（子弹轨迹缓存惰性更新） */
    var threats = assessThreats(tank, game);
    var urgent = [];
    for (var i = 0; i < threats.length; i++) {
      if (threats[i].tHit <= P.dodgeWindow) urgent.push(threats[i]);
    }

    /* 2. 卡住自救 */
    var prevDrive = out.drive;
    if (handleStuck(tank, brain, dt, out)) { brain.lastDrive = out.drive; return out; }

    /* 3. 紧急躲避（最高战斗优先级） */
    if (urgent.length) {
      decideDodge(tank, game, urgent, out);
      opportuneShot(tank, game, enemies, out, brain);
      brain.lastDrive = out.drive;
      return out;
    }

    /* 4. 射后机动：开完枪立刻横移规避还击 */
    if (brain.evade > 0) {
      brain.evade -= dt;
      out.drive = 1;
      out.steer = brain.evadeDir;
      var tgt = brain.target && brain.target.alive ? brain.target : pickTarget(tank, game, brain);
      if (tgt && map.hasLOS(tank.x, tank.y, tgt.x, tgt.y)) {
        tryEngage(tank, tgt, game, brain, out, true, dt);
      }
      brain.lastDrive = out.drive;
      return out;
    }

    /* 5. 直瞄交战 */
    var target = pickTarget(tank, game, brain);
    if (target && map.hasClearShot(tank.x, tank.y, target.x, target.y, 10)) {
      tryEngage(tank, target, game, brain, out, false, dt);
      brain.bounce = null;
      brain.lastDrive = out.drive;
      return out;
    }

    /* 6. 反弹猎杀：无直瞄时绕墙打；僵局时用骚扰流弹逼死守者移动 */
    var aggr = brain.aggression || 1;
    brain.scanTimer -= dt;
    if (brain.bounce) brain.bounce = verifyBounce(tank, game, brain.bounce);
    if (!brain.bounce && brain.scanTimer <= 0) {
      brain.scanTimer = P.scanInterval / aggr;
      brain.scanPhase++;
      var scan = bounceScan(tank, game, enemies, brain.scanPhase, P.scanDist * Math.min(aggr, 2.2));
      brain.bounce = scan.kill;
      brain.harass = scan.harass;
    }
    var killMax = P.bounceKillMax + Math.min((aggr - 1) * 0.5, 1.5);
    if (brain.bounce && brain.bounce.tKill < killMax) {
      var ad = Math.abs(steerTowards(out, tank, brain.bounce.angle, dt));
      if (ad < 0.5) out.drive = 0.4;                     // 缓慢跟进
      if (ad < 0.045 && canFire(tank, game) && muzzleClear(tank, map)) {
        out.shoot = true;
        brain.lastShotAt = brain.t;
        brain.evade = 0.18;
        brain.evadeDir = Math.random() < 0.5 ? -1 : 1;
        brain.bounce = null;
      }
      brain.lastDrive = out.drive;
      return out;
    }
    /* 骚扰射击：僵局中朝「掠过敌人身边」的角度打流弹（1.5s 内对不准就放弃） */
    if (aggr > 2.5 && brain.harass && canFire(tank, game)) {
      if (brain.harassUntil === undefined) brain.harassUntil = brain.t + 1.5;
      if (brain.t > brain.harassUntil) {
        brain.harass = null;
        brain.harassUntil = undefined;
      } else {
        var adH = Math.abs(steerTowards(out, tank, brain.harass.angle, dt));
        if (adH < 0.05 && muzzleClear(tank, map)) {
          out.shoot = true;
          brain.lastHarassAt = brain.t;
          brain.harass = null;
          brain.harassUntil = undefined;
        }
      }
    }

    /* 7. 生存走位 */
    patrol(tank, game, brain, enemies, threats, out);
    brain.lastDrive = out.drive;
    return out;
  }

  /* ===================== 9. 注册（多形态兼容） ===================== */

  var root = (typeof window !== 'undefined') ? window :
             (typeof global !== 'undefined') ? global : {};
  root.TT = root.TT || {};

  var API = {
    name: 'glm',
    label: 'GLM Competitive Policy v1.0',
    version: '1.0.0',
    params: P,                                           // 可调参数（测试/调参用）
    update: step,                                        // updateBot 同签名
    createController: function (tank) {                  // 工厂形态：返回 (game, dt) => input
      getBrain(tank);
      return function (game, dt) { return step(tank, game, dt); };
    },
  };

  root.GLM_POLICY = API;
  root.policy = API;                                     // 单槽兜底约定
  root.TT.Policies = root.TT.Policies || {}; root.TT.Policies.glm = API;
  root.TT.POLICIES = root.TT.POLICIES || {}; root.TT.POLICIES.glm = API;
  root.TT.Policy = root.TT.Policy || {}; root.TT.Policy.glm = API;
  root.TT.GLM_POLICY = API;
  if (typeof root.TT.registerPolicy === 'function') {
    try { root.TT.registerPolicy(API); } catch (e) { /* ignore */ }
  }

  /* 兜底：600ms 后若没有任何装载器接管 TT.updateBot（仍是原生函数），
     则自动接管 —— 浏览器直接打开 index.html 观战即可看到本策略。 */
  var builtinRef = (typeof root.TT.updateBot === 'function') ? root.TT.updateBot : null;
  setTimeout(function () {
    if (builtinRef && typeof root.TT.updateBot === 'function' && root.TT.updateBot === builtinRef) {
      root.TT.updateBot = step;
      if (typeof console !== 'undefined') {
        console.info('[glm-policy] 未检测到策略装载器，已接管 TT.updateBot（内置 AI 全部替换为 GLM 策略）');
      }
    }
  }, 600);

})();
