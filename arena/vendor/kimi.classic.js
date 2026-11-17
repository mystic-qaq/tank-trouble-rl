(function () {
/*
 * Kimi Arena policy for Tank Trouble.
 * Self-contained ES module: no DOM, randomness, or outside-policy dependencies.
 *
 * Priorities:
 *   1. Predict ricochets and choose a short-horizon safe driving action.
 *   2. Take only simulated hits (direct or one-bank), never blind spam.
 *   3. Navigate to targets and use wall reflections when direct LOS is blocked.
 */

const PI = Math.PI;
const TAU = PI * 2;
const TANK_R = 16;
const TANK_SPEED = 178;
const REVERSE_RATIO = 0.66;
const ACCEL = 920;
const BRAKE = 1150;
const TURN = 3.3;
const BARREL = 28;
const BULLET_SPEED = 620;
const BULLET_R = 3.2;
const HIT_R = TANK_R + BULLET_R;
const MAX_BOUNCES = 5;
const DODGE_DT = 0.06;
const DODGE_STEPS = 18;

const brains = new WeakMap();
const trajectoryFrames = new WeakMap();

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
function sign(v) { return v < 0 ? -1 : (v > 0 ? 1 : 0); }
function dist(x0, y0, x1, y1) { return Math.hypot(x1 - x0, y1 - y0); }
function angleDiff(from, to) {
  let d = (to - from + PI) % TAU;
  if (d < 0) d += TAU;
  return d - PI;
}
function turnInput(from, to, dt) {
  return clamp(angleDiff(from, to) / Math.max(0.001, TURN * dt), -1, 1);
}
function velocityOf(t) {
  return { x: Math.cos(t.angle) * t.speed, y: Math.sin(t.angle) * t.speed };
}

function getBrain(tank) {
  let b = brains.get(tank);
  if (!b) {
    b = {
      targetIdx: -1,
      targetTimer: 0,
      path: null,
      pathTimer: 0,
      bank: null,
      bankTimer: 0,
      stuck: 0,
      recovery: 0,
      recoverySteer: 1,
    };
    brains.set(tank, b);
  }
  return b;
}

function liveBullets(game, tank) {
  let n = 0;
  for (const b of game.bullets) {
    if (!b.dead && b.owner === tank.index) n++;
  }
  return n;
}

/* Advance a speed value using the same acceleration/braking rules as Tank.update. */
function nextSpeed(speed, drive, dt) {
  if (drive !== 0) {
    const target = drive > 0 ? TANK_SPEED : -TANK_SPEED * REVERSE_RATIO;
    const rate = speed * drive < 0 ? BRAKE : ACCEL;
    if (speed < target) return Math.min(target, speed + rate * dt);
    if (speed > target) return Math.max(target, speed - rate * dt);
    return speed;
  }
  const drop = BRAKE * 0.75 * dt;
  if (Math.abs(speed) <= drop) return 0;
  return speed - sign(speed) * drop;
}

function movePose(pose, drive, steer, dt, map) {
  pose.angle += steer * TURN * dt;
  if (pose.angle > PI) pose.angle -= TAU;
  if (pose.angle < -PI) pose.angle += TAU;
  pose.speed = nextSpeed(pose.speed, drive, dt);
  pose.x += Math.cos(pose.angle) * pose.speed * dt;
  pose.y += Math.sin(pose.angle) * pose.speed * dt;
  if (map.resolveCircle(pose, TANK_R)) pose.bumps = (pose.bumps || 0) + 1;
  return pose;
}

/* ------------------------- Bullet forecasting ------------------------- */

function forecastBullet(source, map) {
  const points = [{ x: source.x, y: source.y, grace: source.grace }];
  let x = source.x, y = source.y, angle = source.angle;
  let life = source.life, grace = source.grace, bounces = source.bounces;

  for (let step = 0; step < DODGE_STEPS; step++) {
    life -= DODGE_DT;
    grace = Math.max(0, grace - DODGE_DT);
    if (life <= 0 || bounces > MAX_BOUNCES) break;

    let remaining = BULLET_SPEED * DODGE_DT;
    let guard = 0;
    while (remaining > 0.02 && guard++ < 8) {
      const dx = Math.cos(angle) * remaining;
      const dy = Math.sin(angle) * remaining;
      const hit = map.raycast(x, y, x + dx, y + dy);
      if (!hit) {
        x += dx;
        y += dy;
        break;
      }
      const t = clamp(hit.t, 0, 1);
      x += dx * t;
      y += dy * t;
      remaining *= 1 - t;
      const nx = hit.nx || 0, ny = hit.ny || 0;
      if (nx === 0 && ny === 0) {
        bounces = MAX_BOUNCES + 1;
        break;
      }
      let vx = Math.cos(angle), vy = Math.sin(angle);
      const dot = vx * nx + vy * ny;
      if (dot < 0) {
        vx -= 2 * dot * nx;
        vy -= 2 * dot * ny;
      }
      angle = Math.atan2(vy, vx);
      x += nx * 0.8;
      y += ny * 0.8;
      bounces++;
      remaining -= 0.7;
      if (bounces > MAX_BOUNCES) break;
    }
    points.push({ x, y, grace });
  }
  return { owner: source.owner, points };
}

/* All tanks see the same bullet state in a frame, so forecast it only once. */
function getTrajectories(game) {
  let frame = trajectoryFrames.get(game);
  if (frame && frame.time === game.time && frame.serial === game.roundSerial) {
    return frame.paths;
  }
  const paths = [];
  for (const b of game.bullets) {
    if (!b.dead && b.life > 0) paths.push(forecastBullet(b, game.map));
  }
  frame = { time: game.time, serial: game.roundSerial, paths };
  trajectoryFrames.set(game, frame);
  return paths;
}

/* Distance from the origin to a relative-motion segment. */
function relativeDistance2(x0, y0, x1, y1) {
  const dx = x1 - x0, dy = y1 - y0;
  const len2 = dx * dx + dy * dy;
  const t = len2 > 1e-9 ? clamp(-(x0 * dx + y0 * dy) / len2, 0, 1) : 0;
  const x = x0 + dx * t, y = y0 + dy * t;
  return x * x + y * y;
}

function quickThreat(tank, paths) {
  const limit2 = 62 * 62;
  for (const path of paths) {
    const pts = path.points;
    for (let i = 1; i < pts.length; i++) {
      if (path.owner === tank.index && pts[i].grace > 0) continue;
      const d2 = relativeDistance2(
        pts[i - 1].x - tank.x, pts[i - 1].y - tank.y,
        pts[i].x - tank.x, pts[i].y - tank.y
      );
      if (d2 < limit2) return true;
    }
  }
  return false;
}

function controlRisk(tank, game, paths, control) {
  const pose = { x: tank.x, y: tank.y, angle: tank.angle, speed: tank.speed, bumps: 0 };
  let oldX = pose.x, oldY = pose.y;
  let score = 0;
  let firstHit = Infinity;

  for (let i = 1; i <= DODGE_STEPS; i++) {
    movePose(pose, control.drive, control.steer, DODGE_DT, game.map);
    const time = i * DODGE_DT;

    for (const path of paths) {
      const pts = path.points;
      if (i >= pts.length) continue;
      if (path.owner === tank.index && pts[i].grace > 0) continue;
      const d2 = relativeDistance2(
        pts[i - 1].x - oldX, pts[i - 1].y - oldY,
        pts[i].x - pose.x, pts[i].y - pose.y
      );
      if (d2 < (HIT_R + 3) * (HIT_R + 3)) {
        if (time < firstHit) firstHit = time;
      }
      if (d2 < 46 * 46) {
        const d = Math.sqrt(d2);
        score += (46 - d) * (1.1 - 0.45 * i / DODGE_STEPS);
      }
    }
    oldX = pose.x, oldY = pose.y;
  }

  if (firstHit < Infinity) score += 100000 - firstHit * 12000;
  score += (pose.bumps || 0) * 4;
  return { score, firstHit };
}

function dodge(tank, game, desired) {
  const paths = getTrajectories(game);
  if (!paths.length || !quickThreat(tank, paths)) return desired;

  const base = { drive: desired.drive, steer: desired.steer };
  const baseRisk = controlRisk(tank, game, paths, base);
  if (baseRisk.firstHit === Infinity && baseRisk.score < 2.5) return desired;

  const controls = [base];
  const seen = new Set([base.drive.toFixed(2) + ':' + base.steer.toFixed(2)]);
  for (const drive of [1, -1, 0]) {
    for (const steer of [-1, 0, 1]) {
      const key = drive.toFixed(2) + ':' + steer.toFixed(2);
      if (!seen.has(key)) {
        seen.add(key);
        controls.push({ drive, steer });
      }
    }
  }

  let best = base, bestRisk = baseRisk, bestScore = baseRisk.score;
  for (let i = 1; i < controls.length; i++) {
    const c = controls[i];
    const risk = controlRisk(tank, game, paths, c);
    const score = risk.score
      + Math.abs(c.drive - desired.drive) * 1.5
      + Math.abs(c.steer - desired.steer) * 0.25;
    if (score < bestScore) {
      best = c;
      bestRisk = risk;
      bestScore = score;
    }
  }

  /* If everything is fatal, maximize survival time rather than freezing. */
  if (bestRisk.firstHit < Infinity) {
    for (const c of controls) {
      const risk = controlRisk(tank, game, paths, c);
      if (risk.firstHit > bestRisk.firstHit) {
        best = c;
        bestRisk = risk;
      }
    }
  }
  return { drive: best.drive, steer: best.steer, shoot: false };
}

/* ------------------------- Shooting ------------------------- */

function earliestSegmentTank(tank, game, x, y, angle, length, startTime) {
  const vx = Math.cos(angle) * BULLET_SPEED;
  const vy = Math.sin(angle) * BULLET_SPEED;
  const duration = length / BULLET_SPEED;
  let best = null;

  for (const other of game.tanks) {
    if (!other.alive) continue;
    const tv = velocityOf(other);
    const rx = x - other.x, ry = y - other.y;
    const rvx = vx - tv.x, rvy = vy - tv.y;
    const a = rvx * rvx + rvy * rvy;
    const b = 2 * (rx * rvx + ry * rvy);
    const c = rx * rx + ry * ry - HIT_R * HIT_R;
    let t = Infinity;

    if (c <= 0) {
      t = 0;
    } else if (a > 1e-9) {
      const disc = b * b - 4 * a * c;
      if (disc >= 0) {
        const root = Math.sqrt(disc);
        const t0 = (-b - root) / (2 * a);
        const t1 = (-b + root) / (2 * a);
        if (t0 >= 0) t = t0;
        else if (t1 >= 0) t = t1;
      }
    }
    if (t > duration + 1e-7 || t < 0) continue;
    const absoluteTime = startTime + Math.max(0, t);
    if (other.index === tank.index && absoluteTime < 0.035) continue;
    if (!best || absoluteTime < best.time) {
      best = { tank: other, time: absoluteTime };
    }
  }
  return best;
}

/* Simulate the actual shot after this frame's movement. Any enemy is a valid hit. */
function shotOutcome(tank, game, angle, maxBounces = 1, maxTime = 2.5) {
  let x = tank.x + Math.cos(angle) * BARREL;
  let y = tank.y + Math.sin(angle) * BARREL;
  if (game.map.solidAt(x, y)) return { blocked: true, enemy: false, self: false };

  let time = 0, bounces = 0;
  while (time < maxTime && bounces <= maxBounces) {
    const length = (maxTime - time) * BULLET_SPEED;
    const dx = Math.cos(angle) * length, dy = Math.sin(angle) * length;
    const wall = game.map.raycast(x, y, x + dx, y + dy);
    const segLength = wall ? length * clamp(wall.t, 0, 1) : length;
    const hit = earliestSegmentTank(tank, game, x, y, angle, segLength, time);
    if (hit) {
      return {
        blocked: false,
        enemy: hit.tank.index !== tank.index,
        self: hit.tank.index === tank.index,
        targetIdx: hit.tank.index,
        time: hit.time,
      };
    }
    if (!wall || bounces === maxBounces) break;

    x += dx * wall.t;
    y += dy * wall.t;
    time += segLength / BULLET_SPEED + 0.7 / BULLET_SPEED;
    const nx = wall.nx || 0, ny = wall.ny || 0;
    if (nx === 0 && ny === 0) break;
    let vx = Math.cos(angle), vy = Math.sin(angle);
    const dot = vx * nx + vy * ny;
    if (dot < 0) {
      vx -= 2 * dot * nx;
      vy -= 2 * dot * ny;
    }
    angle = Math.atan2(vy, vx);
    x += nx * 0.8;
    y += ny * 0.8;
    bounces++;
  }
  return { blocked: false, enemy: false, self: false };
}

function directAim(tank, target) {
  let px = target.x, py = target.y;
  const tv = velocityOf(target);
  let travel = dist(tank.x, tank.y, px, py) / BULLET_SPEED;

  for (let i = 0; i < 3; i++) {
    px = target.x + tv.x * travel;
    py = target.y + tv.y * travel;
    travel = dist(tank.x, tank.y, px, py) / BULLET_SPEED;
  }
  travel = dist(tank.x, tank.y, px, py) / BULLET_SPEED;
  const angle = Math.atan2(py - tank.y, px - tank.x);
  return {
    angle, kind: 0, targetIdx: target.index,
    x: px, y: py, distance: travel * BULLET_SPEED,
  };
}

function clearDirectAim(tank, game, target) {
  const option = directAim(tank, target);
  const sx = tank.x + Math.cos(option.angle) * BARREL;
  const sy = tank.y + Math.sin(option.angle) * BARREL;
  if (game.map.solidAt(sx, sy)) return null;
  if (game.map.raycast(sx, sy, option.x, option.y)) return null;
  return option;
}

/* Mirror the predicted target across a wall face to obtain a one-bank aim. */
function bankAim(tank, game, target) {
  const tv = velocityOf(target);
  let lead = dist(tank.x, tank.y, target.x, target.y) / BULLET_SPEED;
  let best = null;

  for (let iteration = 0; iteration < 2; iteration++) {
    const px = target.x + tv.x * lead;
    const py = target.y + tv.y * lead;
    best = null;

    for (const w of game.map.walls) {
      const vertical = w.h > w.w * 1.15;
      const horizontal = w.w > w.h * 1.15;
      if (!vertical && !horizontal) continue;

      const planes = vertical ? [w.x, w.x + w.w] : [w.y, w.y + w.h];
      for (const plane of planes) {
        const tankSide = vertical ? tank.x - plane : tank.y - plane;
        const targetSide = vertical ? px - plane : py - plane;
        if (Math.abs(tankSide) < BARREL || Math.abs(targetSide) < 4 || tankSide * targetSide <= 0) continue;

        const vx = vertical ? 2 * plane - px : px;
        const vy = vertical ? py : 2 * plane - py;
        const angle = Math.atan2(vy - tank.y, vx - tank.x);
        const sx = tank.x + Math.cos(angle) * BARREL;
        const sy = tank.y + Math.sin(angle) * BARREL;
        if (game.map.solidAt(sx, sy)) continue;

        const denom = vertical ? vx - sx : vy - sy;
        if (Math.abs(denom) < 1e-6) continue;
        const t = (plane - (vertical ? sx : sy)) / denom;
        if (t <= 0 || t >= 1) continue;
        const bx = vertical ? plane : sx + (vx - sx) * t;
        const by = vertical ? sy + (vy - sy) * t : plane;
        if (bx < w.x + 1 || bx > w.x + w.w - 1 || by < w.y + 1 || by > w.y + w.h - 1) continue;

        const incoming = game.map.raycast(sx, sy, bx, by);
        if (!incoming || incoming.t < 0.94) continue;
        const side = sign(tankSide);
        const ox = bx + (vertical ? side : 0) * 1.1;
        const oy = by + (horizontal ? side : 0) * 1.1;
        if (game.map.raycast(ox, oy, px, py)) continue;

        const pathLength = dist(sx, sy, bx, by) + dist(bx, by, px, py);
        if (!best || pathLength < best.distance) {
          best = { angle, kind: 1, targetIdx: target.index, distance: pathLength };
        }
      }
    }
    if (best) lead = best.distance / BULLET_SPEED;
    else break;
  }
  return best;
}

function selectTarget(tank, game, brain, dt) {
  brain.targetTimer -= dt;
  let current = null;
  for (const t of game.tanks) {
    if (t.alive && t !== tank && t.index === brain.targetIdx) current = t;
  }
  if (brain.targetTimer > 0 && current) return current;

  let best = null, bestScore = Infinity;
  for (const other of game.tanks) {
    if (!other.alive || other === tank) continue;
    const d = dist(tank.x, tank.y, other.x, other.y);
    const visible = !game.map.raycast(tank.x, tank.y, other.x, other.y);
    const aimedAtMe = Math.abs(angleDiff(other.angle, Math.atan2(tank.y - other.y, tank.x - other.x))) < 0.32;
    let score = d - (visible ? 170 : 0) - (visible && aimedAtMe ? 120 : 0);
    if (other === current) score -= 35;
    if (score < bestScore) {
      bestScore = score;
      best = other;
    }
  }
  brain.targetIdx = best ? best.index : -1;
  brain.targetTimer = 0.18;
  brain.pathTimer = 0;
  brain.bankTimer = 0;
  return best;
}

function blockedAhead(tank, game, reverse = false) {
  const base = tank.angle + (reverse ? PI : 0);
  const distance = TANK_R + 23;
  for (const offset of [-0.42, 0, 0.42]) {
    const a = base + offset;
    if (game.map.solidAt(tank.x + Math.cos(a) * distance, tank.y + Math.sin(a) * distance)) return true;
  }
  return false;
}

function navigationInput(tank, game, target, brain, dt) {
  brain.pathTimer -= dt;
  if (brain.pathTimer <= 0 || !brain.path || !brain.path.length) {
    brain.path = game.map.findPath(tank.x, tank.y, target.x, target.y);
    brain.pathTimer = 0.24;
  }

  let waypoint = target;
  if (brain.path && brain.path.length) {
    let guard = 0;
    while (brain.path.length > 1 && guard++ < 6 &&
           dist(tank.x, tank.y, brain.path[0].x, brain.path[0].y) < 36) {
      brain.path.shift();
    }
    let index = 0;
    const max = Math.min(4, brain.path.length - 1);
    for (let i = max; i >= 0; i--) {
      if (!game.map.raycast(tank.x, tank.y, brain.path[i].x, brain.path[i].y)) {
        index = i;
        break;
      }
    }
    waypoint = brain.path[index];
  }

  const wanted = Math.atan2(waypoint.y - tank.y, waypoint.x - tank.x);
  const diff = angleDiff(tank.angle, wanted);
  let drive = Math.abs(diff) > 0.9 ? 0 : 1;
  if (Math.abs(diff) > 2.25 && !blockedAhead(tank, game, true)) drive = -1;
  if (drive > 0 && blockedAhead(tank, game)) drive = 0;
  return { drive, steer: turnInput(tank.angle, wanted, dt), shoot: false };
}

function combatInput(tank, game, target, option, dt) {
  const d = dist(tank.x, tank.y, target.x, target.y);
  let drive = 0;
  if (option.kind === 0) {
    if (d < 165 && !blockedAhead(tank, game, true)) drive = -1;
    else if (d > 410 && !blockedAhead(tank, game)) drive = 1;
    else {
      /* Do not become a stationary target while waiting for the cooldown. */
      const enemyAim = Math.abs(angleDiff(target.angle, Math.atan2(tank.y - target.y, tank.x - target.x)));
      if (enemyAim < 0.22 && tank.cooldown > 0.12) {
        if (!blockedAhead(tank, game, true)) drive = -1;
        else if (!blockedAhead(tank, game)) drive = 1;
      }
    }
  } else if (Math.abs(angleDiff(tank.angle, option.angle)) < 0.18 && d < 170 &&
             !blockedAhead(tank, game, true)) {
    drive = -1;
  }
  if (drive > 0 && blockedAhead(tank, game)) drive = 0;
  return { drive, steer: turnInput(tank.angle, option.angle, dt), shoot: false };
}

/* Tank.update rotates and moves before firing, so validate at that future pose. */
function firePose(tank, game, input, dt) {
  const pose = { x: tank.x, y: tank.y, angle: tank.angle, speed: tank.speed, index: tank.index };
  return movePose(pose, input.drive, input.steer, dt, game.map);
}

function updateBot(tank, game, dt) {
  let input = { drive: 0, steer: 0, shoot: false };
  if (!tank.alive || !game.map || dt <= 0) return input;
  const brain = getBrain(tank);
  const target = selectTarget(tank, game, brain, dt);
  if (!target) return input;

  let option = clearDirectAim(tank, game, target);
  brain.bankTimer -= dt;
  if (!option) {
    if (brain.bankTimer <= 0 || !brain.bank || brain.bank.targetIdx !== target.index) {
      brain.bank = bankAim(tank, game, target);
      brain.bankTimer = 0.09;
    }
    option = brain.bank;
  } else {
    brain.bank = null;
    brain.bankTimer = 0;
  }

  if (option) input = combatInput(tank, game, target, option, dt);
  else input = navigationInput(tank, game, target, brain, dt);

  /* Wall recovery is deliberately simple; bullet evasion may still override it. */
  if (input.drive !== 0 && Math.abs(tank.speed) < 20) brain.stuck += dt;
  else brain.stuck = Math.max(0, brain.stuck - dt * 2.5);
  if (brain.stuck > 0.55) {
    brain.stuck = 0;
    brain.recovery = 0.48;
    brain.recoverySteer = ((tank.index + game.round) & 1) ? 1 : -1;
    brain.path = null;
    brain.pathTimer = 0;
  }
  if (brain.recovery > 0) {
    brain.recovery -= dt;
    input.drive = -1;
    input.steer = brain.recoverySteer;
    input.shoot = false;
  }

  const dodged = dodge(tank, game, input);

  let alive = 0;
  for (const t of game.tanks) if (t.alive) alive++;
  const bulletCap = alive <= 2 ? 4 : 3;
  if (dodged === input && tank.cooldown <= dt + 1e-6 && liveBullets(game, tank) < bulletCap) {
    const pose = firePose(tank, game, input, dt);
    const outcome = shotOutcome(pose, game, pose.angle, 1, 2.5);
    if (outcome.enemy) input.shoot = true;
  }

  return input;
}

const policy = updateBot;


/* ---------------- classic script 版：挂到 window 上（arena/build-vendor.mjs 生成） ---------------- */
if (typeof window !== 'undefined') {
  window.KIMI_TANK = {
    update: updateBot,
    policy: updateBot,
  };
}
})();
