/* ===========================================================
   Tank Trouble · DeepSeek Arena Policy
   -----------------------------------------------------------
   设计目标：在 8 车混战 / 1v1 中尽可能拿到「回合胜场」。
   核心原则：
     1. 回合胜者 = 最后存活者，所以「活着」优先于击杀；
     2. 一枪毙命，自己的子弹反弹后同样会打死自己 —— 开火前
        必须完整模拟弹道，确认不会回到自己身上；
     3. 时刻模拟所有在飞子弹，用短程 MPC 选择最优躲避动作；
     4. 利用分数转向精确「落角」，对准即开火，不浪费机会；
     5. 没有直射视线时，尝试单次墙面反弹的「台球」射击；
     6. 前期占位伏击 + 中期清场 + 残局拒绝换命。

   接口（兼容常见 Arena 挂载方式）：
     TT.deepseekPolicy(tank, game, dt) -> {drive, steer, shoot}
     TT.deepseekPolicy.update(tank, game, dt)
     TT.deepseekPolicy.think / getInput / updateBot / decide ...
     const brain = TT.deepseekPolicy.create(tank, game);
     brain.update(...) / brain.think(...) / brain(game, dt)
   本文件无外部依赖（只读取 TT.CFG / game 对象），可独立加载。
   =========================================================== */

(function (root) {
  'use strict';

  var TT = (root.TT = root.TT || {});

  /* ---------------- 规则常量（优先读游戏配置） ---------------- */

  var CFG = TT.CFG || {};
  var TK = CFG.TANK || {};
  var BL = CFG.BULLET || {};

  var R_TANK = TK.radius || 16;
  var R_BULLET = BL.radius || 3.2;
  var HIT_R = R_TANK + R_BULLET;
  var BULLET_SPEED = BL.speed || 620;
  var MAX_BOUNCE = BL.bounces != null ? BL.bounces : 5;
  var BULLET_LIFE = BL.life || 6;
  var TURN = TK.turn || 3.3;
  var ACCEL = TK.accel || 920;
  var BRAKE = TK.brake || 1150;
  var MAX_SPEED = TK.speed || 178;
  var REV_RATIO = TK.reverseRatio || 0.66;
  var REV_SPEED = MAX_SPEED * REV_RATIO;
  var MAX_SHOT_LIVE = CFG.MAX_BULLETS_PER_TANK || 5;

  var PI = Math.PI;
  var TAU = PI * 2;

  /* ---------------- 可调参数（测试台/Arena 可覆盖） ---------------- */

  var P_DEFAULT = {
    guardHold: 5.5,          // 营地驻守时长（秒），之后转入猎杀
    campDist: 650,           // 营地与最近敌人的理想距离
    campCenterWeight: 0.012, // 营地偏中心的惩罚权重
    fightClose: 300,         // 近于此距离开始倒车
    fightFar: 650,           // 远于此距离开始推进
    driftFwd: 0.6,           // 中距离前压速度
    driftBack: 0.7,          // 中距离后撤速度
    maxLive: 2,              // 同时在场自弹上限
    huntStop: 26,             // 追击时的停车距离
    postFireDrive: 0.65,     // 开火后扭动的前进力度
    postFireTime: 0.26,      // 开火后扭动时长
    postFireReverse: 0,      // 开火后立刻反转速度方向，破坏对方提前量
    campAtSpawn: 0,          // 直接在出生点驻守
    campOpenBias: 6,         // 营地开口越少越好的权重
    campMinOpens: 1,         // 允许选择的最少开口数
    guardExitDist: 300,      // 敌人近于此距离时离开营地转入猎杀
    guardExitAlive: 3,       // 存活数 <= 此值时离开营地
    guardAimOpen: 1,         // 守卫时瞄准营地入口而不是穿墙瞄敌人
    campSpawnWeight: 0,      // 营地离出生点的距离惩罚
    driftMin: 0.7,          // 中距离漂移换向最短时间
    driftMax: 1.6,          // 中距离漂移换向最长时间
    aimWeight: -3,          // 目标正在瞄准我时的权重（负值=回避）
    engagedBonus: 2,        // 目标正在和别人交火时的权重
    threatDist: 300,        // 瞄准我的敌人近于此距离时仍优先处理
    preJink: 0,              // 被瞄准且无法还手时预判扭动
    preJinkDist: 430,
    coverUnderFire: 1,       // 多敌环境下被瞄准时寻找掩体
    coverDist: 260,          // 掩体搜索半径
    coverHold: 0.55,         // 掩体移动持续时间
    coverPhaseMin: 4,        // 敌人数 >= 此值才启用掩体战术
    bankShots: 1,
    dodgeMargin: 13,
    disengageLos: 0,        // 同时被 >=N 个敌人看见时脱离找掩体（0 关闭）
    avoidTrade: 0,          // 多人局不与正瞄准我的敌人对枪
    avoidTradeCd: 0.08,
    closeTurnDist: 0         // 近距离背后转身时倒车拉开（0 关闭）
  };

  var PARAMS = {};
  function applyParams() {
    for (var k in P_DEFAULT) {
      PARAMS[k] = P_DEFAULT[k];
      if (TT.DS_PARAMS && TT.DS_PARAMS[k] != null) PARAMS[k] = TT.DS_PARAMS[k];
    }
  }
  applyParams();

  /* ---------------- 小工具 ---------------- */

  function clamp(v, a, b) { return v < a ? a : (v > b ? b : v); }
  function sgn(v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }
  function dist2(ax, ay, bx, by) {
    var dx = bx - ax, dy = by - ay;
    return dx * dx + dy * dy;
  }
  function dist(ax, ay, bx, by) { return Math.sqrt(dist2(ax, ay, bx, by)); }

  function angDiff(a, b) {
    var d = (b - a + PI) % TAU;
    if (d < 0) d += TAU;
    return d - PI;
  }

  function rnd(st) {
    st.rng = (Math.imul(st.rng, 1664525) + 1013904223) >>> 0;
    return st.rng / 4294967296;
  }

  function dsLog(ev) {
    if (!TT.DS_DEBUG) return;
    if (!TT.DS_LOG) TT.DS_LOG = [];
    TT.DS_LOG.push(ev);
    if (TT.DS_LOG.length > 400) TT.DS_LOG.shift();
  }

  /* ---------------- 地图访问（全部做防御式判断） ---------------- */

  function mapSolid(game, x, y) {
    var m = game && game.map;
    if (!m || typeof m.solidAt !== 'function') return false;
    return m.solidAt(x, y);
  }

  function mapRay(game, x0, y0, x1, y1) {
    var m = game && game.map;
    if (!m || typeof m.raycast !== 'function') return null;
    return m.raycast(x0, y0, x1, y1);
  }

  function lineClear(game, x0, y0, x1, y1) {
    return !mapRay(game, x0, y0, x1, y1);
  }

  function hasLOS(game, ax, ay, bx, by) {
    var m = game && game.map;
    if (m && typeof m.hasLOS === 'function') return m.hasLOS(ax, ay, bx, by);
    return lineClear(game, ax, ay, bx, by);
  }

  function muzzleOf(tank, angle) {
    var a = angle == null ? tank.angle : angle;
    return {
      x: tank.x + Math.cos(a) * (R_TANK + 12),
      y: tank.y + Math.sin(a) * (R_TANK + 12)
    };
  }

  function muzzleUsable(game, tank, angle) {
    var m = muzzleOf(tank, angle);
    if (mapSolid(game, m.x, m.y)) return false;
    if (mapSolid(game, m.x + Math.cos(angle) * 5, m.y + Math.sin(angle) * 5)) return false;
    return true;
  }

  /* ---------------- 逐帧坦克状态 ---------------- */

  var STATES = new WeakMap();

  function freshState(tank, game) {
    return {
      mapRef: game.map,
      serial: game.roundSerial != null ? game.roundSerial : game.round,
      lastX: tank.x,
      lastY: tank.y,
      spawnX: tank.x,
      spawnY: tank.y,
      rng: (function () {
        var rs = game.roundSerial != null ? game.roundSerial : (game.round || 0);
        var v = (Math.imul((tank.index || 0) * 2654435761, 1) ^ Math.imul(rs * 2246822519, 1) ^ 0x9E3779B9) >>> 0;
        return v || 1;
      })(),
      init: false,
      time: 0,

      mode: 'camp',            // camp | guard | hunt
      camp: null,
      guardTimer: 0,
      campTimer: 0,
      modeTimer: 0,

      path: null,
      pathGoal: null,
      pathTimer: 0,

      targetIndex: -1,
      targetHold: 0,

      chkX: tank.x,
      chkY: tank.y,
      chkTimer: 0,
      intendedDrive: 0,
      stuck: 0,
      reverseTimer: 0,
      reverseSteer: 1,

      driftDir: 1,
      driftTimer: 0.6,

      jinkTimer: 0,
      jinkDir: 1,

      lastDodgeDrive: 0,
      lastDodgeSteer: 0,
      jinkDrive: 0.65,

      bank: null,
      bankTimer: 0,

      shotSafeAngle: null,
      shotSafePos: null,
      shotSafeTime: -99,
      shotSafeOK: false,

      coverTimer: 0,
      cover: null,
      seenEnemy: false,
      firstLOS: -1
    };
  }

  function stateFor(tank, game) {
    var st = STATES.get(tank);
    if (!st) {
      st = freshState(tank, game);
      STATES.set(tank, st);
    } else {
      var serial = game.roundSerial != null ? game.roundSerial : game.round;
      if (st.mapRef !== game.map || (serial != null && st.serial !== serial)) {
        var rng = st.rng;
        st = freshState(tank, game);
        st.rng = (rng + 0x9E3779B9) >>> 0;
        STATES.set(tank, st);
      }
    }
    return st;
  }

  /* ===========================================================
     弹道模拟：与游戏 Bullet.update 的反射逻辑一致
     =========================================================== */

  function bulletSim(x, y, angle, game, horizon, step) {
    var pts = [{ x: x, y: y, t: 0 }];
    var t = 0, bounces = 0;
    var dx = Math.cos(angle), dy = Math.sin(angle);
    var px = x, py = y;
    var guard = 0;

    while (t < horizon && bounces <= MAX_BOUNCE && guard++ < 400) {
      var len = BULLET_SPEED * step;
      var nx = px + dx * len;
      var ny = py + dy * len;
      var hit = mapRay(game, px, py, nx, ny);

      if (!hit) {
        px = nx; py = ny; t += step;
        pts.push({ x: px, y: py, t: t });
        continue;
      }

      var nxw = hit.nx || 0, nyw = hit.ny || 0;
      if (nxw === 0 && nyw === 0) break; // 炮口卡墙：哑火

      var tt = t + step * clamp(hit.t, 0, 1);
      var hx = px + (nx - px) * clamp(hit.t, 0, 1);
      var hy = py + (ny - py) * clamp(hit.t, 0, 1);
      pts.push({ x: hx, y: hy, t: tt });

      var dot = dx * nxw + dy * nyw;
      if (dot < 0) {
        dx -= 2 * dot * nxw;
        dy -= 2 * dot * nyw;
      }
      var dl = Math.sqrt(dx * dx + dy * dy) || 1;
      dx /= dl; dy /= dl;

      px = hx + nxw * 0.8;
      py = hy + nyw * 0.8;
      t = tt;
      bounces++;
      if (bounces > MAX_BOUNCE) break;
      pts.push({ x: px, y: py, t: t });
    }
    return pts;
  }

  /* 折线轨迹与点的最近距离（只看 t >= tMin 的部分） */
  function closestApproach(pts, x, y, tMin) {
    var best = { d2: Infinity, t: 0 };
    for (var i = 1; i < pts.length; i++) {
      var a = pts[i - 1], b = pts[i];
      if (b.t < tMin) continue;
      var abx = b.x - a.x, aby = b.y - a.y;
      var len2 = abx * abx + aby * aby;
      var u = 0;
      if (len2 > 1e-9) {
        u = ((x - a.x) * abx + (y - a.y) * aby) / len2;
        u = clamp(u, 0, 1);
      }
      var cx = a.x + abx * u;
      var cy = a.y + aby * u;
      var d2 = dist2(cx, cy, x, y);
      if (d2 < best.d2) {
        best.d2 = d2;
        best.t = a.t + (b.t - a.t) * u;
      }
    }
    return best;
  }

  /* 按固定时间采样轨迹，方便和自身运动在相同时间点比较 */
  function sampleTraj(pts, horizon, step) {
    var out = [];
    var idx = 1;
    for (var t = 0; t <= horizon + 1e-6; t += step) {
      while (idx < pts.length - 1 && pts[idx].t < t) idx++;
      var a = pts[idx - 1], b = pts[idx];
      var span = Math.max(1e-6, b.t - a.t);
      var u = clamp((t - a.t) / span, 0, 1);
      out.push({
        x: a.x + (b.x - a.x) * u,
        y: a.y + (b.y - a.y) * u
      });
    }
    return out;
  }

  /* ===========================================================
     射击评估
     =========================================================== */

  function targetSpeedVec(t) {
    var sp = t.speed || 0;
    return { x: Math.cos(t.angle) * sp, y: Math.sin(t.angle) * sp };
  }

  /* 带提前量的瞄准解（迭代求解命中时间） */
  function aimSolution(me, game, target) {
    var v = targetSpeedVec(target);
    var t = dist(me.x, me.y, target.x, target.y) / BULLET_SPEED;
    var px = target.x, py = target.y;
    var i, nx, ny;

    for (i = 0; i < 3; i++) {
      var lt = clamp(t, 0, 0.62);
      nx = target.x + v.x * lt;
      ny = target.y + v.y * lt;
      if (mapSolid(game, nx, ny)) break;
      px = nx; py = ny;
      t = dist(me.x, me.y, px, py) / BULLET_SPEED;
    }

    var ddx = px - me.x, ddy = py - me.y;
    var d = Math.sqrt(ddx * ddx + ddy * ddy) || 1;
    var angle = Math.atan2(ddy, ddx);
    var muzzle = muzzleOf(me, angle);

    var clear = !mapSolid(game, muzzle.x, muzzle.y) &&
      lineClear(game, muzzle.x, muzzle.y, px, py);

    /* 目标点不能落在墙里（例如预判穿墙） */
    if (mapSolid(game, px, py)) {
      px = target.x; py = target.y;
      d = dist(me.x, me.y, px, py) || 1;
      angle = Math.atan2(py - me.y, px - me.x);
      muzzle = muzzleOf(me, angle);
      clear = !mapSolid(game, muzzle.x, muzzle.y) &&
        lineClear(game, muzzle.x, muzzle.y, px, py);
    }

    return {
      angle: angle,
      px: px, py: py,
      d: d,
      t: d / BULLET_SPEED,
      clear: clear,
      muzzle: muzzle
    };
  }

  /* 开火后子弹是否会绕回自己身上（严格安全检查） */
  function shotSafe(me, game, angle, st) {
    var now = st ? st.time : 0;
    var posOK = st && st.shotSafePos &&
      dist2(st.shotSafePos.x, st.shotSafePos.y, me.x, me.y) < 64;

    if (st && st.shotSafeAngle != null && posOK &&
        Math.abs(angDiff(st.shotSafeAngle, angle)) < 0.02 &&
        now - st.shotSafeTime < 0.12) {
      return st.shotSafeOK;
    }

    var ok = false;
    if (!muzzleUsable(game, me, angle)) {
      ok = false;
    } else {
      var m = muzzleOf(me, angle);
      var pts = bulletSim(m.x, m.y, angle, game, 3.2, 0.04);
      var ca = closestApproach(pts, me.x, me.y, 0.16);
      var safeR = HIT_R + 10;
      ok = ca.d2 > safeR * safeR;
      if (TT.DS_DEBUG && ok && ca.d2 < 60 * 60) {
        dsLog({
          kind: 'safeCheck', t: st ? st.time : 0, r: st ? st.serial : 0, idx: me.index,
          x: +me.x.toFixed(1), y: +me.y.toFixed(1),
          angle: +angle.toFixed(3), margin: +Math.sqrt(ca.d2).toFixed(1),
          at: +ca.t.toFixed(2)
        });
      }
    }

    if (st) {
      st.shotSafeAngle = angle;
      st.shotSafePos = { x: me.x, y: me.y };
      st.shotSafeTime = now;
      st.shotSafeOK = ok;
    }
    return ok;
  }

  function ownLiveBullets(tank, game) {
    var n = 0;
    var list = game.bullets || [];
    for (var i = 0; i < list.length; i++) {
      var b = list[i];
      if (!b.dead && b.owner === tank.index) n++;
    }
    return n;
  }

  /* 分数转向：当角度差在一帧步长内时，精确落角 */
  function steerToAngle(me, aim, dt, out) {
    var diff = angDiff(me.angle, aim);
    var step = TURN * dt;
    if (Math.abs(diff) <= step) {
      out.steer = clamp(diff / Math.max(step, 1e-6), -1, 1);
      return true;
    }
    out.steer = diff > 0 ? 1 : -1;
    return false;
  }

  function canSeeEnemy(me, game, e) {
    return e && e.alive &&
      hasLOS(game, me.x, me.y, e.x, e.y);
  }

  function enemyAimsAt(e, me, game) {
    var a = Math.atan2(me.y - e.y, me.x - e.x);
    if (Math.abs(angDiff(e.angle, a)) > 0.11) return false;
    var m = muzzleOf(e, e.angle);
    return lineClear(game, m.x, m.y, me.x, me.y);
  }

  /* ===========================================================
     威胁扫描 + 避弹 MPC
     =========================================================== */

  function scanThreats(me, game, st) {
    var list = [];
    var bullets = game.bullets || [];

    for (var i = 0; i < bullets.length; i++) {
      var b = bullets[i];
      if (!b || b.dead) continue;

      var bd = dist(me.x, me.y, b.x, b.y);
      if (bd > 980) continue;

      var pts = bulletSim(b.x, b.y, b.angle, game, 1.15, 0.03);
      var ca = closestApproach(pts, me.x, me.y, 0.035);

      if (ca.t < 1.08 && ca.d2 < (HIT_R + 13) * (HIT_R + 13)) {
        list.push({
          pts: pts,
          t: ca.t,
          d: Math.sqrt(ca.d2),
          own: b.owner === me.index
        });
      }
    }

    return list;
  }

  function simMyMotion(me, game, steer, drive, horizon, step) {
    var x = me.x, y = me.y;
    var angle = me.angle;
    var speed = me.speed || 0;
    var out = [];
    var fake = { x: x, y: y, radius: R_TANK };

    for (var t = step; t <= horizon + 1e-6; t += step) {
      angle += steer * TURN * step;
      if (angle > PI) angle -= TAU;
      if (angle < -PI) angle += TAU;

      var target;
      if (drive > 0) target = MAX_SPEED * drive;
      else if (drive < 0) target = -REV_SPEED;
      else target = 0;

      if (drive !== 0) {
        var rate = (speed * drive < 0) ? BRAKE : ACCEL;
        if (speed < target) speed = Math.min(target, speed + rate * step);
        else if (speed > target) speed = Math.max(target, speed - rate * step);
      } else {
        var drop = BRAKE * 0.75 * step;
        if (Math.abs(speed) <= drop) speed = 0;
        else speed -= sgn(speed) * drop;
      }

      x += Math.cos(angle) * speed * step;
      y += Math.sin(angle) * speed * step;

      var bumped = false;
      if (game.map && typeof game.map.resolveCircle === 'function') {
        fake.x = x; fake.y = y;
        bumped = game.map.resolveCircle(fake, R_TANK);
        x = fake.x; y = fake.y;
      }
      out.push({ x: x, y: y, angle: angle, speed: speed, bump: bumped });
    }
    return out;
  }

  function pickDodge(me, game, st, threats, dt) {
    var maxT = 0;
    for (var i = 0; i < threats.length; i++) {
      if (threats[i].t > maxT) maxT = threats[i].t;
    }

    var horizon = clamp(maxT + 0.18, 0.32, 1.05);
    var step = 0.055;
    var samples = [];
    var tMax = [];

    for (i = 0; i < threats.length; i++) {
      samples.push(sampleTraj(threats[i].pts, horizon, step));
      tMax.push(threats[i].t + 0.12);
    }

    var steerOpts = [-1, -0.45, 0, 0.45, 1];
    var driveOpts = [-1, -0.4, 0.45, 1];
    var best = null;

    for (var si = 0; si < steerOpts.length; si++) {
      for (var di = 0; di < driveOpts.length; di++) {
        var steer = steerOpts[si];
        var drive = driveOpts[di];
        var path = simMyMotion(me, game, steer, drive, horizon, step);
        var minD2 = Infinity;
        var bumps = 0;

        for (var k = 0; k < path.length; k++) {
          var p = path[k];
          if (p.bump) bumps++;
          var sampleT = (k + 1) * step;
          for (var ti = 0; ti < samples.length; ti++) {
            if (sampleT > tMax[ti]) continue;
            var arr = samples[ti];
            var idx = Math.min(k + 1, arr.length - 1);
            var bp = arr[idx];
            var d2 = dist2(p.x, p.y, bp.x, bp.y);
            if (d2 < minD2) minD2 = d2;
          }
        }

        var score = Math.sqrt(minD2);
        score -= bumps * 1.2;
        if (steer === 0) score += 0.5;
        if (steer === st.lastDodgeSteer && drive === st.lastDodgeDrive) score += 0.35;
        if (Math.abs(drive) > 0.3) score += 0.12;

        if (!best || score > best.score) {
          best = { steer: steer, drive: drive, score: score };
        }
      }
    }

    if (!best) best = { steer: 0, drive: 1 };

    st.lastDodgeSteer = best.steer;
    st.lastDodgeDrive = best.drive;

    return {
      drive: best.drive,
      steer: best.steer,
      shoot: false
    };
  }

  /* 精准落角后若视野里有敌人，边躲边开火 */
  function opportunisticFire(me, game, st, steer, dt) {
    if (!me.alive || me.cooldown > 0) return false;
    if (ownLiveBullets(me, game) >= PARAMS.maxLive) return false;

    var post = me.angle + steer * TURN * dt;
    var enemies = game.tanks || [];
    for (var i = 0; i < enemies.length; i++) {
      var e = enemies[i];
      if (e === me || !e.alive) continue;
      var d = dist(me.x, me.y, e.x, e.y);
      if (d > 620) continue;
      if (!hasLOS(game, me.x, me.y, e.x, e.y)) continue;

      var a = Math.atan2(e.y - me.y, e.x - me.x);
      var maxErr = Math.max(0.02, Math.atan2(HIT_R * 0.8, d));
      if (Math.abs(angDiff(post, a)) > maxErr) continue;
      if (!muzzleUsable(game, me, post)) continue;
      if (!shotSafe(me, game, post, st)) continue;
      dsLog({
        kind: 'fire-opp', t: st.time, r: st.serial, idx: me.index,
        x: +me.x.toFixed(1), y: +me.y.toFixed(1),
        angle: +post.toFixed(3), steer: +steer.toFixed(2), d: +d.toFixed(1)
      });
      return true;
    }
    return false;
  }

  /* ===========================================================
     地图移动
     =========================================================== */

  function clearance(me, game, angle, look) {
    var ca = Math.cos(angle), sa = Math.sin(angle);
    var px = Math.cos(angle + PI / 2), py = Math.sin(angle + PI / 2);
    for (var d = 8; d <= look; d += 8) {
      for (var oi = -1; oi <= 1; oi++) {
        var off = oi * 11;
        if (mapSolid(game, me.x + ca * d + px * off, me.y + sa * d + py * off)) {
          return d;
        }
      }
    }
    return look + 1;
  }

  function getPath(me, game, st, tx, ty) {
    var m = game.map;
    if (!m || typeof m.findPath !== 'function') return null;
    var path = m.findPath(me.x, me.y, tx, ty);
    if (!path || !path.length) return null;
    return path;
  }

  function followPath(me, game, st, tx, ty, dt, stopDist) {
    var out = { drive: 0, steer: 0, shoot: false };
    st.pathTimer -= dt;

    var goalMoved = !st.pathGoal ||
      dist(st.pathGoal.x, st.pathGoal.y, tx, ty) > 60;

    if (st.pathTimer <= 0 || !st.path || !st.path.length || goalMoved) {
      st.path = getPath(me, game, st, tx, ty);
      st.pathGoal = { x: tx, y: ty };
      st.pathTimer = 0.55 + rnd(st) * 0.2;
    }

    if (!st.path || !st.path.length) {
      return driveToPoint(me, game, st, tx, ty, dt, stopDist);
    }

    /* 丢弃已经到达/越过的路点 */
    var guard = 0;
    while (st.path.length > 1 && guard++ < 10 &&
           dist(me.x, me.y, st.path[0].x, st.path[0].y) < 30) {
      st.path.shift();
    }

    /* 选取前方 0~3 个路点中能直视的最远点，让行驶更顺 */
    var idx = 0;
    var maxAhead = Math.min(st.path.length - 1, 3);
    for (var i = maxAhead; i >= 1; i--) {
      if (lineClear(game, me.x, me.y, st.path[i].x, st.path[i].y)) { idx = i; break; }
    }
    var wp = st.path[idx];

    if (dist(me.x, me.y, tx, ty) <= stopDist) {
      /* 已到目标附近 */
      return out;
    }

    return driveToPoint(me, game, st, wp.x, wp.y, dt, 0);
  }

  function driveToPoint(me, game, st, px, py, dt, stopDist) {
    var out = { drive: 0, steer: 0, shoot: false };
    var dx = px - me.x, dy = py - me.y;
    var d = Math.sqrt(dx * dx + dy * dy);

    if (d < Math.max(stopDist, 12)) return out;

    var want = Math.atan2(dy, dx);
    var diff = angDiff(me.angle, want);
    var ad = Math.abs(diff);

    steerToAngle(me, want, dt, out);

    if (ad > 1.15) {
      out.drive = 0;                 // 大角度先原地转向
    } else if (ad > 0.55) {
      out.drive = 0.42;
    } else {
      out.drive = 1;
    }

    if (d < 90) out.drive *= clamp(d / 90, 0.28, 1);

    /* 前方碰撞检测 */
    var look = 38 + Math.abs(me.speed || 0) * 0.14;
    var clr = clearance(me, game, me.angle, look);

    if (clr < look) {
      out.drive = Math.min(out.drive, 0.28);
      if (clr < 26) {
        var left = clearance(me, game, me.angle - 0.55, 42);
        var right = clearance(me, game, me.angle + 0.55, 42);
        out.steer = left >= right ? -1 : 1;
        out.drive = 0;
      }
    }

    return out;
  }

  /* ===========================================================
     目标选择
     =========================================================== */

  function selectTarget(me, game, st, enemies) {
    if (!enemies.length) return null;
    if (enemies.length === 1) return enemies[0];

    var current = null;
    for (var i = 0; i < enemies.length; i++) {
      if (enemies[i].index === st.targetIndex && enemies[i].alive) {
        current = enemies[i];
        break;
      }
    }

    var best = null, bestScore = -1e9;

    for (i = 0; i < enemies.length; i++) {
      var e = enemies[i];
      if (!e.alive) continue;
      var d = dist(me.x, me.y, e.x, e.y);
      var s = 6 - d / 180;

      var los = hasLOS(game, me.x, me.y, e.x, e.y);
      if (los) s += 3.2;
      if (los && enemyAimsAt(e, me, game)) {
        s += d < PARAMS.threatDist ? 2.6 : PARAMS.aimWeight;
      }
      if (PARAMS.engagedBonus && los && !enemyAimsAt(e, me, game)) {
        for (var ei2 = 0; ei2 < game.tanks.length; ei2++) {
          var o2 = game.tanks[ei2];
          if (o2 === e || o2 === me || !o2.alive) continue;
          if (hasLOS(game, e.x, e.y, o2.x, o2.y) && enemyAimsAt(e, o2, game)) {
            s += PARAMS.engagedBonus;
            break;
          }
        }
      }
      if (e === current) s += 1.6;
      if (st.mode === 'hunt') s -= d / 240;

      if (s > bestScore) { bestScore = s; best = e; }
    }

    if (best) {
      if (best.index !== st.targetIndex) {
        st.targetIndex = best.index;
        st.targetHold = 1.2;
      }
    }
    return best;
  }

  /* ===========================================================
     弹墙（单次反弹）射击
     =========================================================== */

  function computeBankShot(me, game, st, target) {
    if (!PARAMS.bankShots) return null;
    var m = game.map;
    if (!m || !m.walls || !m.walls.length) return null;
    if (typeof m.raycast !== 'function') return null;

    var sol0 = aimSolution(me, game, target);

    /* 目标快速移动时，反弹预测误差太大；除非我们已经大致对准 */
    var tspd = Math.abs(target.speed || 0);
    if (tspd > 150 && Math.abs(angDiff(me.angle, sol0.angle)) > 0.7) return null;

    var best = null;
    var walls = m.walls;
    var margin = 2;
    var v = targetSpeedVec(target);
    var leadCap = 0.75;

    for (var wi = 0; wi < walls.length; wi++) {
      var w = walls[wi];

      /* 只考虑离双方都不太远的墙 */
      var wx = w.x + w.w / 2, wy = w.y + w.h / 2;
      if (dist(me.x, me.y, wx, wy) > 760) continue;
      if (dist(target.x, target.y, wx, wy) > 780) continue;

      for (var face = 0; face < 4; face++) {
        var vertical = face < 2;
        var fx, fy, nx, ny, lim0, lim1;

        if (face === 0) {          // 左表面 x = w.x，法线朝 -x
          fx = w.x; nx = -1; ny = 0;
          lim0 = w.y; lim1 = w.y + w.h;
        } else if (face === 1) {   // 右表面 x = w.x+w.w，法线朝 +x
          fx = w.x + w.w; nx = 1; ny = 0;
          lim0 = w.y; lim1 = w.y + w.h;
        } else if (face === 2) {   // 上表面 y = w.y，法线朝 -y
          fy = w.y; nx = 0; ny = -1;
          lim0 = w.x; lim1 = w.x + w.w;
        } else {                   // 下表面
          fy = w.y + w.h; nx = 0; ny = 1;
          lim0 = w.x; lim1 = w.x + w.w;
        }

        if (vertical) {
          if ((me.x - fx) * nx <= 2) continue;
          if ((target.x - fx) * nx <= 2) continue;
        } else {
          if ((me.y - fy) * ny <= 2) continue;
          if ((target.y - fy) * ny <= 2) continue;
        }

        /* 不动点迭代：命中点 T 与飞行时间互相依赖 */
        var tx = target.x, ty = target.y;
        var bounce = null;

        for (var iter = 0; iter < 4; iter++) {
          var reflX, reflY, bx, by, u, denom;

          if (vertical) {
            reflX = 2 * fx - tx;
            reflY = ty;
            denom = reflX - me.x;
            if (Math.abs(denom) < 1e-6) { bounce = null; break; }
            u = (fx - me.x) / denom;
            if (u <= 0.02 || u >= 0.98) { bounce = null; break; }
            bx = fx;
            by = me.y + (reflY - me.y) * u;
            if (by < lim0 + margin || by > lim1 - margin) { bounce = null; break; }
          } else {
            reflX = tx;
            reflY = 2 * fy - ty;
            denom = reflY - me.y;
            if (Math.abs(denom) < 1e-6) { bounce = null; break; }
            u = (fy - me.y) / denom;
            if (u <= 0.02 || u >= 0.98) { bounce = null; break; }
            by = fy;
            bx = me.x + (reflX - me.x) * u;
            if (bx < lim0 + margin || bx > lim1 - margin) { bounce = null; break; }
          }

          var tImp = (dist(me.x, me.y, bx, by) + dist(bx, by, tx, ty)) / BULLET_SPEED;
          var ntx = target.x + v.x * Math.min(tImp, leadCap);
          var nty = target.y + v.y * Math.min(tImp, leadCap);

          bounce = { bx: bx, by: by, t: tImp };

          if (dist2(ntx, nty, tx, ty) < 4) break;
          tx = ntx; ty = nty;
        }

        if (!bounce) continue;

        /* 用最终命中点重算一次反弹点，保证反射几何严格成立 */
        var finalB = null;
        (function () {
          var bx2, by2, u2, denom2, reflX2, reflY2;
          if (vertical) {
            reflX2 = 2 * fx - tx;
            reflY2 = ty;
            denom2 = reflX2 - me.x;
            if (Math.abs(denom2) < 1e-6) return;
            u2 = (fx - me.x) / denom2;
            if (u2 <= 0.02 || u2 >= 0.98) return;
            bx2 = fx;
            by2 = me.y + (reflY2 - me.y) * u2;
            if (by2 < lim0 + margin || by2 > lim1 - margin) return;
          } else {
            reflX2 = tx;
            reflY2 = 2 * fy - ty;
            denom2 = reflY2 - me.y;
            if (Math.abs(denom2) < 1e-6) return;
            u2 = (fy - me.y) / denom2;
            if (u2 <= 0.02 || u2 >= 0.98) return;
            by2 = fy;
            bx2 = me.x + (reflX2 - me.x) * u2;
            if (bx2 < lim0 + margin || bx2 > lim1 - margin) return;
          }
          finalB = { bx: bx2, by: by2 };
        })();
        if (!finalB) continue;

        /* 目标最终位置不能落在墙里 */
        if (mapSolid(game, tx, ty)) continue;

        var bInX = finalB.bx + nx * 0.9;
        var bInY = finalB.by + ny * 0.9;
        if (!lineClear(game, me.x, me.y, bInX, bInY)) continue;
        if (!lineClear(game, bInX, bInY, tx, ty)) continue;

        var totalLen = dist(me.x, me.y, finalB.bx, finalB.by) + dist(finalB.bx, finalB.by, tx, ty);
        var tHit = totalLen / BULLET_SPEED;
        if (tHit > 1.15 || tHit < 0.06) continue;

        var aimAng = Math.atan2(finalB.by - me.y, finalB.bx - me.x);
        var mzl = muzzleOf(me, aimAng);
        if (mapSolid(game, mzl.x, mzl.y)) continue;
        if (!shotSafe(me, game, aimAng, st)) continue;

        /* 评分：命中越快越好，路径越短越好，目标越慢越可靠 */
        var score = 8 - tHit * 4 - totalLen * 0.0015;
        score += clamp((140 - tspd) / 140, 0, 1) * 2.5;

        if (!best || score > best.score) {
          best = {
            score: score,
            angle: aimAng,
            t: tHit,
            dist: totalLen,
            bx: finalB.bx, by: finalB.by,
            tx: tx, ty: ty
          };
        }
      }
    }

    return best;
  }

  /* ===========================================================
     营地/巡逻
     =========================================================== */

  function chooseCamp(me, game, st, enemies) {
    var m = game.map;
    if (!m) return { x: me.x, y: me.y };

    var cols = m.cols || 16;
    var rows = m.rows || 10;
    var cell = m.cell || 80;
    var best = null;

    for (var r = 1; r < rows - 1; r++) {
      for (var c = 1; c < cols - 1; c++) {
        var opens = 2;
        if (typeof m.openSides === 'function') opens = m.openSides(r, c);
        if (opens < PARAMS.campMinOpens) continue;

        var x = c * cell + cell / 2;
        var y = r * cell + cell / 2;
        if (mapSolid(game, x, y)) continue;

        var occupied = false;
        for (var oi = 0; oi < enemies.length; oi++) {
          if (dist(x, y, enemies[oi].x, enemies[oi].y) < 100) {
            occupied = true;
            break;
          }
        }
        if (occupied) continue;

        var dmin = 9999;
        for (var i = 0; i < enemies.length; i++) {
          var d = dist(x, y, enemies[i].x, enemies[i].y);
          if (d < dmin) dmin = d;
        }

        var cx = m.w ? m.w / 2 : 640;
        var cy = m.h ? m.h / 2 : 400;
        var dc = dist(x, y, cx, cy);

        var score = 18;                            // 基础分
        score += (3 - opens) * PARAMS.campOpenBias;
        score -= Math.abs(dmin - PARAMS.campDist) * 0.055;  // 离最近敌人理想距离
        score -= dc * PARAMS.campCenterWeight;             // 略偏中心
        score -= dist(x, y, st.spawnX, st.spawnY) * PARAMS.campSpawnWeight;
        score += rnd(st) * 26;                     // 随机化

        if (!best || score > best.score) {
          best = { x: x, y: y, r: r, c: c, score: score };
        }
      }
    }

    if (!best) return { x: me.x, y: me.y };
    return { x: best.x, y: best.y };
  }

  /* 守卫时朝向最可能来人的方向（最近敌人），否则朝最长的通道 */
  function guardAim(me, game, enemies) {
    var nearest = null, nd = 1e9;
    for (var i = 0; i < enemies.length; i++) {
      var d = dist(me.x, me.y, enemies[i].x, enemies[i].y);
      if (d < nd) { nd = d; nearest = enemies[i]; }
    }

    var eb = nearest
      ? Math.atan2(nearest.y - me.y, nearest.x - me.x)
      : me.angle;

    if (!PARAMS.guardAimOpen) return eb;

    var m = game.map;
    if (!m || typeof m.vAt !== 'function' || typeof m.hAt !== 'function') return eb;

    var cell = m.cell || 80;
    var cols = m.cols || 16;
    var rows = m.rows || 10;
    var c = clamp(Math.floor(me.x / cell), 0, cols - 1);
    var r = clamp(Math.floor(me.y / cell), 0, rows - 1);

    var dirs = [];
    if (c < cols - 1 && !m.vAt(r, c + 1)) dirs.push(0);
    if (c > 0 && !m.vAt(r, c)) dirs.push(PI);
    if (r < rows - 1 && !m.hAt(r + 1, c)) dirs.push(PI / 2);
    if (r > 0 && !m.hAt(r, c)) dirs.push(-PI / 2);

    if (!dirs.length) return eb;

    var best = dirs[0], bd = Infinity;
    for (var k = 0; k < dirs.length; k++) {
      var dd = Math.abs(angDiff(eb, dirs[k]));
      if (dd < bd) { bd = dd; best = dirs[k]; }
    }
    return best;
  }

  /* ===========================================================
     战斗
     =========================================================== */

  /* 在多敌环境里被瞄准时，找一个能断掉所有相关视线的近点 */
  function findCoverPoint(me, game, enemies, maxDist, st) {
    var m = game.map;
    if (!m) return null;

    var cols = m.cols || 16;
    var rows = m.rows || 10;
    var cell = m.cell || 80;
    var c0 = Math.floor(me.x / cell);
    var r0 = Math.floor(me.y / cell);
    var threat = [];

    for (var ei = 0; ei < enemies.length; ei++) {
      var e = enemies[ei];
      if (!e.alive) continue;
      if (hasLOS(game, me.x, me.y, e.x, e.y)) threat.push(e);
    }
    if (!threat.length) return null;

    var best = null;

    for (var dr = -3; dr <= 3; dr++) {
      for (var dc = -3; dc <= 3; dc++) {
        if (!dr && !dc) continue;
        var r = r0 + dr, c = c0 + dc;
        if (r < 1 || r > rows - 2 || c < 1 || c > cols - 2) continue;

        var x = c * cell + cell / 2;
        var y = r * cell + cell / 2;
        if (mapSolid(game, x, y)) continue;

        var d = dist(me.x, me.y, x, y);
        if (d < 55 || d > maxDist) continue;

        var open = typeof m.openSides === 'function' ? m.openSides(r, c) : 2;
        if (open < 2) continue;

        var safe = true;
        for (var ti = 0; ti < threat.length; ti++) {
          if (hasLOS(game, x, y, threat[ti].x, threat[ti].y)) { safe = false; break; }
        }
        if (!safe) continue;

        var score = 40 - d * 0.22 + (open === 2 ? 26 : 8) + rnd(st) * 18;
        if (!best || score > best.score) {
          best = { x: x, y: y, score: score };
        }
      }
    }

    return best;
  }

  function chooseJinkDir(me, game, st) {
    var left = clearance(me, game, me.angle - 0.65, 46);
    var right = clearance(me, game, me.angle + 0.65, 46);
    if (Math.abs(left - right) < 4) return rnd(st) < 0.5 ? -1 : 1;
    return left > right ? -1 : 1;
  }

  function fight(me, game, st, target, enemies, dt) {
    var out = { drive: 0, steer: 0, shoot: false };
    var sol = aimSolution(me, game, target);
    var diff = angDiff(me.angle, sol.angle);
    var ad = Math.abs(diff);
    var step = TURN * dt;
    var land = Math.abs(diff) <= step * 1.08;

    steerToAngle(me, sol.angle, dt, out);

    var d = dist(me.x, me.y, target.x, target.y);
    var aimThreat = null;
    for (var aut = 0; aut < enemies.length; aut++) {
      if (enemies[aut].alive && enemyAimsAt(enemies[aut], me, game)) {
        aimThreat = enemies[aut];
        break;
      }
    }
    var aimedAtMe = !!aimThreat;
    var aimerCd = aimThreat ? aimThreat.cooldown : 1;
    var aimerDist = aimThreat
      ? dist(me.x, me.y, aimThreat.x, aimThreat.y)
      : d;
    var canShootNow = land && sol.clear && me.cooldown <= 0 &&
      ownLiveBullets(me, game) < PARAMS.maxLive;

    var totalAlive = enemies.length + 1;
    var losCount = 0;
    for (var li = 0; li < enemies.length; li++) {
      if (enemies[li].alive &&
          hasLOS(game, me.x, me.y, enemies[li].x, enemies[li].y)) losCount++;
    }

    /* 被多敌同时锁定：先脱离视线，别站在交叉火力里 */
    if (PARAMS.disengageLos && totalAlive >= PARAMS.coverPhaseMin &&
        losCount >= PARAMS.disengageLos) {
      st.coverTimer -= dt;
      if (st.coverTimer <= 0) {
        st.coverTimer = 0.35;
        st.cover = findCoverPoint(me, game, enemies, Math.max(PARAMS.coverDist, 360), st);
      }
      if (st.cover) {
        var esInp = followPath(me, game, st, st.cover.x, st.cover.y, dt, 22);
        esInp.shoot = opportunisticFire(me, game, st, esInp.steer, dt);
        return esInp;
      }
    }

    /* 距离控制：太近倒车、太远推进、中距离漂移 */
    var frontClr = clearance(me, game, me.angle, 40);
    var backClr = clearance(me, game, me.angle + PI, 40);

    if (ad > 1.05) {
      out.drive = (PARAMS.closeTurnDist && d < PARAMS.closeTurnDist && backClr > 30) ? -1 : 0;
    } else if (d < PARAMS.fightClose) {
      out.drive = backClr > 30 ? -1 : (frontClr > 30 ? 0.55 : 0);
    } else if (d > PARAMS.fightFar) {
      out.drive = frontClr > 30 ? 1 : 0;
    } else {
      st.driftTimer -= dt;
      if (st.driftTimer <= 0) {
        st.driftTimer = PARAMS.driftMin + rnd(st) * (PARAMS.driftMax - PARAMS.driftMin);
        st.driftDir = -st.driftDir;
      }

      if (st.driftDir > 0) out.drive = frontClr > 30 ? PARAMS.driftFwd : 0;
      else out.drive = backClr > 30 ? -PARAMS.driftBack : (frontClr > 30 ? 0.3 : 0);
    }

    /* 开火：精确落角 + 直射通畅 + 不自伤 */
    var holdFire = PARAMS.avoidTrade && totalAlive >= 4 &&
      aimedAtMe && aimerCd <= PARAMS.avoidTradeCd;

    if (!holdFire && land && sol.clear && me.cooldown <= 0 &&
        target.alive && ownLiveBullets(me, game) < PARAMS.maxLive) {

      var post = me.angle + out.steer * TURN * dt;
      var err = Math.abs(angDiff(post, sol.angle));
      var tol = Math.max(0.014, Math.atan2(HIT_R * 0.62, Math.max(30, sol.d)));

      if (err <= tol && muzzleUsable(game, me, post) &&
          shotSafe(me, game, post, st)) {
        out.shoot = true;
        dsLog({
          kind: 'fire-fight', t: st.time, r: st.serial, idx: me.index,
          x: +me.x.toFixed(1), y: +me.y.toFixed(1),
          angle: +post.toFixed(3), steer: +out.steer.toFixed(2),
          drive: +out.drive.toFixed(2), d: +sol.d.toFixed(1), err: +err.toFixed(4)
        });
        st.jinkTimer = PARAMS.postFireTime + rnd(st) * 0.08;
        st.jinkDir = chooseJinkDir(me, game, st);
        st.jinkDrive = PARAMS.postFireReverse ?
          (me.speed > 25 ? -1 : (me.speed < -25 ? 1 : (rnd(st) < 0.5 ? 1 : -1))) :
          PARAMS.postFireDrive;
      }
    }

    /* 被瞄准且这一帧打不出去：多敌时先找掩体，否则预判性扭动 */
    if (!out.shoot && aimedAtMe && aimerCd < 0.16 && aimerDist < 560) {
      var useCover = PARAMS.coverUnderFire && enemies.length + 1 >= PARAMS.coverPhaseMin;

      if (useCover && !canShootNow) {
        st.coverTimer -= dt;
        if (st.coverTimer <= 0) {
          st.coverTimer = 0.4;
          st.cover = findCoverPoint(me, game, enemies, PARAMS.coverDist, st);
        }
        if (st.cover) {
          var covInp = followPath(me, game, st, st.cover.x, st.cover.y, dt, 22);
          covInp.shoot = opportunisticFire(me, game, st, covInp.steer, dt);
          return covInp;
        }
      }

      if (PARAMS.preJink && !canShootNow && aimerDist < PARAMS.preJinkDist) {
        st.jinkTimer = Math.max(st.jinkTimer, 0.2);
        st.jinkDir = chooseJinkDir(me, game, st);
        st.jinkDrive = PARAMS.postFireReverse ?
          (me.speed > 25 ? -1 : (me.speed < -25 ? 1 : (rnd(st) < 0.5 ? 1 : -1))) :
          PARAMS.postFireDrive;
      }
    } else {
      st.cover = null;
    }

    /* 开火后的短暂规避：反正还在冷却，扭一下破坏对方预判 */
    if (!out.shoot && me.cooldown > 0.09 && st.jinkTimer > 0) {
      st.jinkTimer -= dt;
      var jclr = clearance(me, game, me.angle, 44);
      if (jclr > 30) {
        out.steer = st.jinkDir;
        out.drive = st.jinkDrive || PARAMS.postFireDrive;
      } else {
        out.steer = st.jinkDir;
        out.drive = 0;
      }
    }

    return out;
  }

  /* ===========================================================
     主更新
     =========================================================== */

  function update(tank, game, dt) {
    var out = { drive: 0, steer: 0, shoot: false };

    if (!tank || tank.alive === false) return out;
    if (!game || !game.map) return out;

    var st = stateFor(tank, game);

    /* 若 Arena 没传 dt，用 game.time 差分自动推算 */
    var gnow = game.time != null ? game.time : null;
    var provided = Number(dt);
    var autoDt = (gnow != null && st.time > 0 && gnow > st.time)
      ? clamp(gnow - st.time, 0.001, 0.05)
      : (1 / 60);
    dt = clamp(provided > 0 ? provided : autoDt, 0.001, 0.05);

    /* 同一辆坦克被传送/重生（某些 Arena 会复用对象）：重置状态 */
    if (st.lastX != null &&
        dist2(tank.x, tank.y, st.lastX, st.lastY) > 300 * 300) {
      st = freshState(tank, game);
      STATES.set(tank, st);
    }
    st.lastX = tank.x;
    st.lastY = tank.y;

    if (TT.DS_DEBUG) tank.__dsMode = st.mode;
    if (gnow != null) st.time = gnow;
    else st.time += dt;
    st.seenEnemy = false;

    var allTanks = game.tanks || [];
    var enemies = [];

    for (var i = 0; i < allTanks.length; i++) {
      var t = allTanks[i];
      if (t && t !== tank && t.alive) enemies.push(t);
    }

    if (!enemies.length) return out;

    if (!st.init) {
      st.init = true;
      if (PARAMS.campAtSpawn) {
        st.camp = { x: tank.x, y: tank.y };
        st.mode = 'guard';
      } else {
        st.camp = chooseCamp(tank, game, st, enemies);
        st.mode = 'camp';
      }
      st.guardTimer = 0;
    }

    /* ---- 1. 危险子弹躲避（最高优先级） ---- */
    var threats = scanThreats(tank, game, st);
    var urgent = false;
    for (i = 0; i < threats.length; i++) {
      if (threats[i].t < 1.0) { urgent = true; break; }
    }

    if (urgent) {
      var dodge = pickDodge(tank, game, st, threats, dt);
      dodge.shoot = opportunisticFire(tank, game, st, dodge.steer, dt);
      st.intendedDrive = dodge.drive;
      return dodge;
    }

    /* ---- 2. 选目标 ---- */
    var target = selectTarget(tank, game, st, enemies);
    st.targetHold = Math.max(0, st.targetHold - dt);

    var los = target && canSeeEnemy(tank, game, target);
    if (los) {
      st.seenEnemy = true;
      if (st.firstLOS < 0) st.firstLOS = st.time;
    }

    /* ---- 3. 预判对方即将开火：不能还手时先闪 ---- */
    if (!los || !target) {
      var pre = null;
      for (i = 0; i < enemies.length; i++) {
        var e = enemies[i];
        var ed = dist(tank.x, tank.y, e.x, e.y);
        if (ed > 520) continue;
        if (e.cooldown > 0.07) continue;
        if (!enemyAimsAt(e, tank, game)) continue;
        if (ed < 340) { pre = e; break; }
        if (!pre) pre = e;
      }

      if (pre) {
        var fakePts = bulletSim(
          muzzleOf(pre, pre.angle).x,
          muzzleOf(pre, pre.angle).y,
          pre.angle, game, 0.85, 0.03
        );
        var fca = closestApproach(fakePts, tank.x, tank.y, 0.04);
        if (fca.d2 < (HIT_R + 16) * (HIT_R + 16)) {
          var fthreats = [{ pts: fakePts, t: fca.t, d: Math.sqrt(fca.d2), own: false }];
          var fd = pickDodge(tank, game, st, fthreats, dt);
          st.intendedDrive = fd.drive;
          return fd;
        }
      }
    }

    /* ---- 4. 有视线：战斗 ---- */
    if (target && los) {
      st.mode = 'hunt';
      var finp = fight(tank, game, st, target, enemies, dt);
      st.intendedDrive = finp.drive;
      return postProcess(tank, game, st, finp, dt);
    }

    /* ---- 5. 无视线：尝试弹墙射击 ---- */
    if (target) {
      st.bankTimer -= dt;
      if (st.bankTimer <= 0) {
        st.bankTimer = 0.14;
        st.bank = computeBankShot(tank, game, st, target);
      }

      var bank = st.bank;
      if (bank && target.alive) {
        var bankInp = { drive: 0, steer: 0, shoot: false };
        var bdiff = angDiff(tank.angle, bank.angle);
        var bstep = TURN * dt;
        steerToAngle(tank, bank.angle, dt, bankInp);

        if (Math.abs(bdiff) <= bstep * 1.08 && tank.cooldown <= 0 &&
            ownLiveBullets(tank, game) < PARAMS.maxLive &&
            muzzleUsable(game, tank, bank.angle) &&
            shotSafe(tank, game, bank.angle, st)) {
          var bpost = tank.angle + bankInp.steer * TURN * dt;
          var berr = Math.abs(angDiff(bpost, bank.angle));
          var btol = Math.max(0.01, Math.atan2(HIT_R * 0.55, Math.max(40, bank.dist)));
          if (berr <= btol) {
            bankInp.shoot = true;
            dsLog({
              kind: 'fire-bank', t: st.time, r: st.serial, idx: tank.index,
              x: +tank.x.toFixed(1), y: +tank.y.toFixed(1),
              angle: +bpost.toFixed(3), target: target.index, tHit: +bank.t.toFixed(2)
            });
            st.jinkTimer = 0.2 + rnd(st) * 0.1;
            st.jinkDir = rnd(st) < 0.5 ? -1 : 1;
          }
        }

        /* 站桩瞄准时若背后有危险，交给下一帧的威胁扫描 */
        st.intendedDrive = bankInp.drive;
        return postProcess(tank, game, st, bankInp, dt);
      }
    }

    /* ---- 6. 移动：营地 → 守卫 → 猎杀 ---- */
    var goal = null;
    var stopDist = 0;

    if (st.mode === 'camp') {
      if (!st.camp) st.camp = chooseCamp(tank, game, st, enemies);
      var cd = dist(tank.x, tank.y, st.camp.x, st.camp.y);
      if (cd < 36) {
        st.mode = 'guard';
        st.guardTimer = 0;
      } else {
        goal = st.camp;
        stopDist = 24;
      }
    }

    if (st.mode === 'guard') {
      st.guardTimer += dt;
      var nearest = 1e9;
      for (i = 0; i < enemies.length; i++) {
        nearest = Math.min(nearest, dist(tank.x, tank.y, enemies[i].x, enemies[i].y));
      }

      if (st.guardTimer > PARAMS.guardHold ||
          enemies.length + 1 <= PARAMS.guardExitAlive ||
          nearest < PARAMS.guardExitDist) {
        st.mode = 'hunt';
      } else {
        /* 面向最近敌人，等待对方撞进视线 */
        var aim = guardAim(tank, game, enemies);
        var ginp = { drive: 0, steer: 0, shoot: false };
        steerToAngle(tank, aim, dt, ginp);
        st.intendedDrive = 0;
        return postProcess(tank, game, st, ginp, dt);
      }
    }

    if (st.mode === 'hunt' && target) {
      goal = { x: target.x, y: target.y };
      /* 只有真的贴脸才停；只要出现视线，战斗逻辑会接管 */
      stopDist = PARAMS.huntStop;
    }

    if (!goal) {
      st.intendedDrive = 0;
      return postProcess(tank, game, st, out, dt);
    }

    var ninp = followPath(tank, game, st, goal.x, goal.y, dt, stopDist);
    st.intendedDrive = ninp.drive;
    return postProcess(tank, game, st, ninp, dt);
  }

  /* 卡住自救 + 倒车覆盖 */
  function postProcess(tank, game, st, inp, dt) {
    var out = inp;

    st.chkTimer += dt;
    if (st.chkTimer >= 0.3) {
      var moved = dist(tank.x, tank.y, st.chkX, st.chkY);
      if (Math.abs(st.intendedDrive) > 0.25 && moved < 7) {
        st.stuck++;
      } else {
        st.stuck = Math.max(0, st.stuck - 1);
      }
      st.chkTimer = 0;
      st.chkX = tank.x;
      st.chkY = tank.y;
    }

    if (st.stuck >= 2 && st.reverseTimer <= 0) {
      st.stuck = 0;
      st.reverseTimer = 0.45;
      st.reverseSteer = rnd(st) < 0.5 ? -1 : 1;
      st.path = null;
    }

    if (st.reverseTimer > 0) {
      st.reverseTimer -= dt;
      var back = clearance(tank, game, tank.angle + PI, 34);
      out = {
        drive: -1,
        steer: back > 22 ? st.reverseSteer : -st.reverseSteer,
        shoot: false
      };
    }

    return out;
  }

  /* ===========================================================
     对外接口（尽量兼容不同 Arena 的挂载约定）
     =========================================================== */

  function makeBrain(tank, game) {
    var boundGame = game || null;
    var boundTank = tank || null;

    function brainArg(args) {
      var tk = boundTank, gm = boundGame, d = null;
      for (var i = 0; i < args.length; i++) {
        var a = args[i];
        if (typeof a === 'number') { d = a; continue; }
        if (!a || typeof a !== 'object') continue;
        if (a.tanks && a.map) { gm = a; continue; }
        if (a.index != null && a.alive != null) { tk = a; continue; }
        if (a.tank && a.tank.index != null) {
          tk = a.tank;
          if (a.game && a.game.map) gm = a.game;
          if (a.dt != null) d = a.dt;
        }
      }
      return [tk, gm, d];
    }

    function brain() {
      var a = brainArg(arguments);
      if (!a[0] || !a[1]) return { drive: 0, steer: 0, shoot: false };
      var n = normalizeCall(a[0], a[1], a[2]);
      try {
        return update(n[0], n[1], n[2]);
      } catch (err) {
        return { drive: 0, steer: 0, shoot: false };
      }
    }

    brain.update = brain;
    brain.think = brain;
    brain.getInput = brain;
    brain.getAction = brain;
    brain.getMove = brain;
    brain.chooseInput = brain;
    brain.act = brain;
    brain.decide = brain;
    brain.step = brain;
    brain.tick = brain;
    brain.tank = boundTank;
    brain.game = boundGame;
    brain.reset = function () {
      if (boundTank) STATES.delete(boundTank);
    };
    return brain;
  }

  function normalizeCall(a, b, c) {
    /* 单对象 { tank, game, dt } */
    if (a && a.tank && a.tank.index != null && a.game && a.game.map) {
      return [a.tank, a.game, a.dt != null ? a.dt : c];
    }
    /* (game, tank, dt) 反序调用 */
    if (a && a.tanks && a.map && b && b.index != null && b.alive != null) {
      return [b, a, c];
    }
    return [a, b, c];
  }

  function API(tank, game, dt) {
    try {
      var a = normalizeCall(tank, game, dt);
      return update(a[0], a[1], a[2]);
    } catch (err) {
      return { drive: 0, steer: 0, shoot: false };
    }
  }

  API.policyName = 'deepseek';
  API.displayName = 'DeepSeek';
  API.version = '1.0.0';
  try {
    Object.defineProperty(API, 'name', { value: 'deepseek', configurable: true });
  } catch (e) { /* ignore */ }
  API.update = API;
  API.think = API;
  API.getInput = API;
  API.getAction = API;
  API.getMove = API;
  API.chooseInput = API;
  API.act = API;
  API.updateBot = API;
  API.decide = API;
  API.step = API;
  API.tick = API;
  API.create = makeBrain;
  API.createBrain = makeBrain;
  API.makeBrain = makeBrain;
  API.makeAI = makeBrain;
  API.createAI = makeBrain;
  API.bind = function (tank, game) { return makeBrain(tank, game); };
  API.init = makeBrain;
  API.setup = makeBrain;
  API.attach = function (tank, game) {
    var b = makeBrain(tank, game);
    tank.policy = b;
    tank.brain = b;
    return b;
  };
  API.reset = function () {
    STATES = new WeakMap();
  };
  API.setParams = function (p) {
    if (p && typeof p === 'object') TT.DS_PARAMS = p;
    applyParams();
    return PARAMS;
  };
  API.getParams = function () {
    var out = {};
    for (var k in PARAMS) out[k] = PARAMS[k];
    return out;
  };
  API._debug = {
    bulletSim: bulletSim,
    closestApproach: closestApproach,
    shotSafe: shotSafe,
    computeBankShot: computeBankShot,
    sampleTraj: sampleTraj
  };
  API.install = function (tt) {
    var target = tt || TT;
    target.updateBot = API;
    return API;
  };

  /* 固定挂载点（约定：TT.<name>Policy） */
  TT.deepseekPolicy = API;
  TT.DeepSeekPolicy = API;
  TT.deepseek = API;

  if (root && root !== TT) {
    root.deepseekPolicy = root.deepseekPolicy || API;
  }

  if (TT.policies && typeof TT.policies === 'object') {
    TT.policies.deepseek = API;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = API;
  }
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
