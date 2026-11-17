/* ===========================================================
   ai.js — 电脑对手（BFS 寻路 + 视线判断 + 预判射击）
   =========================================================== */

(function () {
  const CFG = TT.CFG;
  const U = TT.util;

  function makeBrain(tank) {
    return {
      jitter: 0,
      jitterTarget: 0,
      jitterTimer: 0,
      stuck: 0,
      reverse: 0,
      reverseDir: 1,
      shootDelay: 0.45,
      skill: 0.54 + Math.random() * 0.32,
      path: null,
      pathTimer: 0,
      retargetTimer: 0,
      targetIdx: -1,
    };
  }

  function probe(game, tank, dist, angleOffset) {
    const a = tank.angle + (angleOffset || 0);
    return game.map.solidAt(tank.x + Math.cos(a) * dist, tank.y + Math.sin(a) * dist);
  }

  TT.updateBot = function (tank, game, dt) {
    if (!tank.ai) tank.ai = makeBrain(tank);
    const b = tank.ai;
    const inp = { drive: 0, steer: 0, shoot: false };

    /* ---- 选目标：优先最近的敌人 ---- */
    let target = null, bestD = Infinity;
    for (let i = 0; i < game.tanks.length; i++) {
      const o = game.tanks[i];
      if (o === tank || !o.alive) continue;
      const d = U.dist2(tank.x, tank.y, o.x, o.y);
      if (d < bestD) { bestD = d; target = o; }
    }
    if (!target) return inp;

    const dist = Math.sqrt(bestD);
    b.pathTimer -= dt;
    b.shootDelay = Math.max(0, b.shootDelay - dt);

    /* 电脑不会无脑倾泻子弹：同时在场最多 3 发，降低流弹自伤 */
    let live = 0;
    for (let i = 0; i < game.bullets.length; i++) {
      const bl = game.bullets[i];
      if (!bl.dead && bl.owner === tank.index) live++;
    }
    const canShoot = live < 3;

    /* 瞄准抖动，制造"人味" */
    b.jitterTimer -= dt;
    if (b.jitterTimer <= 0) {
      b.jitterTimer = 0.25 + Math.random() * 0.5;
      b.jitterTarget = (Math.random() - 0.5) * 0.12 * (1.25 - b.skill);
    }
    b.jitter += (b.jitterTarget - b.jitter) * Math.min(1, dt * 8);

    const los = game.map.hasLOS(tank.x, tank.y, target.x, target.y);

    /* ---- 有视线：瞄准并交火 ---- */
    if (los) {
      const lead = dist / CFG.BULLET.speed;
      const px = target.x + Math.cos(target.angle) * target.speed * lead * 0.9;
      const py = target.y + Math.sin(target.angle) * target.speed * lead * 0.9;
      const aim = Math.atan2(py - tank.y, px - tank.x) + b.jitter;
      const diff = U.angleDiff(tank.angle, aim);

      if (Math.abs(diff) > 0.025) inp.steer = U.sign(diff);

      /* 保持距离：太近倒车，太远推进，中距离慢慢压上 */
      const frontBlocked = probe(game, tank, tank.radius + 20);
      const backBlocked = probe(game, tank, -tank.radius - 20);

      if (dist < 165) {
        inp.drive = backBlocked ? 0 : -1;
      } else if (dist > 340) {
        inp.drive = frontBlocked ? 0 : 1;
      } else if (Math.abs(diff) < 0.3) {
        inp.drive = frontBlocked ? 0 : 0.5;
      }

      if (frontBlocked && Math.abs(diff) < 0.5 && Math.random() < 0.5) {
        inp.steer = b.reverseDir;
      }

      /* 开火：对准了就扣扳机 */
      if (Math.abs(diff) < 0.072 + (1 - b.skill) * 0.05) {
        const tip = tank.barrelTip();
        const clear = !game.map.solidAt(
          tip.x + Math.cos(tank.angle) * 9,
          tip.y + Math.sin(tank.angle) * 9
        );
        if (clear && canShoot && b.shootDelay <= 0) {
          inp.shoot = true;
          b.shootDelay = 0.1 + (1 - b.skill) * 0.3;
        }
      }
      b.path = null;
      b.pathTimer = 0.2;
    } else {
      /* ---- 无视线：沿着 BFS 路径包抄 ---- */
      if (b.pathTimer <= 0 || !b.path || !b.path.length) {
        b.path = game.map.findPath(tank.x, tank.y, target.x, target.y);
        b.pathTimer = 0.35 + Math.random() * 0.3;
      }

      let wp = target;
      if (b.path && b.path.length) {
        let guard = 0;
        while (b.path.length > 1 && guard++ < 8 &&
               U.dist(tank.x, tank.y, b.path[0].x, b.path[0].y) < 42) {
          b.path.shift();
        }
        /* 挑最远的一个能直视的路点，走得更顺 */
        let idx = 0;
        const maxAhead = Math.min(b.path.length - 1, 4);
        for (let i = maxAhead; i >= 0; i--) {
          if (game.map.hasLOS(tank.x, tank.y, b.path[i].x, b.path[i].y)) { idx = i; break; }
        }
        wp = b.path[idx];
      }

      const want = Math.atan2(wp.y - tank.y, wp.x - tank.x);
      const diff = U.angleDiff(tank.angle, want);
      if (Math.abs(diff) > 0.03) inp.steer = U.sign(diff);
      inp.drive = Math.abs(diff) > 1.9 ? -1 : (Math.abs(diff) > 0.6 ? 0.45 : 1);

      /* 前方堵住就小幅转向 */
      if (inp.drive > 0 && probe(game, tank, tank.radius + 18)) {
        inp.drive = 0.3;
        inp.steer = b.reverseDir;
      }

      /* 盲射：偶尔朝走廊里打一发，弹来弹去制造压力 */
      if (canShoot && live < 2 && tank.cooldown <= 0 && Math.random() < dt * 0.35) {
        const tip = tank.barrelTip();
        const clear = !game.map.solidAt(
          tip.x + Math.cos(tank.angle) * 12,
          tip.y + Math.sin(tank.angle) * 12
        );
        if (clear) inp.shoot = true;
      }
    }

    /* ---- 卡住自救 ---- */
    if (inp.drive !== 0 && Math.abs(tank.speed) < 24) b.stuck += dt;
    else b.stuck = Math.max(0, b.stuck - dt * 2.5);

    if (b.stuck > 0.65) {
      b.stuck = 0;
      b.reverse = 0.5 + Math.random() * 0.35;
      b.reverseDir = Math.random() < 0.5 ? -1 : 1;
      b.path = null;
      b.pathTimer = 0;
    }

    if (b.reverse > 0) {
      b.reverse -= dt;
      inp.drive = -1;
      inp.steer = b.reverseDir;
      inp.shoot = false;
    }

    return inp;
  };
})();
