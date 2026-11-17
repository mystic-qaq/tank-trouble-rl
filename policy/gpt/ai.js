/* GPT / Sentinel — Tank Trouble policy. No dependencies, no access to other policies.
 * Interface: updateBot(tank, game, dt) -> { drive, steer, shoot }.
 * The world is read-only; all prediction operates on private copies.
 */
(function (root, factory) {
  'use strict';
  const policy = factory();
  if (typeof module === 'object' && module.exports) module.exports = policy;
  if (root.TT) {
    root.TT.policies = root.TT.policies || {};
    root.TT.policies.gpt = policy;
    root.TT.updateBotGPT = policy;
    // Drop-in compatibility with the updateBot call in js/game.js.
    root.TT.updateBot = policy;
  }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';
  const PI = Math.PI, TAU = 2 * PI;
  const maps = new WeakMap(), memories = new WeakMap();
  const clamp = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);
  const diff = (a, b) => { let d = (b - a + PI) % TAU; if (d < 0) d += TAU; return d - PI; };
  const d2 = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
  const idle = () => ({ drive: 0, steer: 0, shoot: false });

  function config(game) {
    const tt = typeof window !== 'undefined' ? window.TT : globalThis.TT;
    const c = (tt && tt.CFG) || game.cfg || {};
    return {
      tank: Object.assign({ radius: 16, speed: 178, reverseRatio: .66,
        accel: 920, brake: 1150, turn: 3.3, fireCooldown: .42 }, c.TANK),
      bullet: Object.assign({ speed: 620, radius: 3.2, bounces: 5,
        life: 6, spawnGrace: .07 }, c.BULLET),
      cap: c.MAX_BULLETS_PER_TANK || 5,
    };
  }

  function boxHit(ox, oy, dx, dy, w, pad) {
    let min = 0, max = 1, nx = 0, ny = 0;
    for (let axis = 0; axis < 2; axis++) {
      const p = axis ? oy : ox, v = axis ? dy : dx;
      const lo = (axis ? w.y : w.x) - pad;
      const hi = (axis ? w.y + w.h : w.x + w.w) + pad;
      if (v > -1e-9 && v < 1e-9) { if (p < lo || p > hi) return null; continue; }
      const inv = 1 / v;
      let a = (lo - p) * inv, b = (hi - p) * inv, sign = -1;
      if (a > b) { const tmp = a; a = b; b = tmp; sign = 1; }
      if (a > min) { min = a; nx = axis ? 0 : sign; ny = axis ? sign : 0; }
      if (b < max) max = b;
      if (min > max) return null;
    }
    return { t: min, nx, ny };
  }

  // Sweep against expanded rectangles, including the full body between ray
  // offsets. This is conservative at rounded corners and cannot miss a stub.
  function clear(map, a, b, radius) {
    const dx = b.x - a.x, dy = b.y - a.y;
    for (const w of map.walls) if (boxHit(a.x, a.y, dx, dy, w, radius)) return false;
    return true;
  }

  function mapData(map, cfg) {
    let data = maps.get(map);
    if (data) return data;
    const cell = map.cell || 80, cols = map.cols || Math.round(map.w / cell);
    const rows = map.rows || Math.round(map.h / cell), n = cols * rows;
    const nodes = Array.from({ length: n }, (_, i) => ({
      x: (i % cols + .5) * cell, y: (Math.floor(i / cols) + .5) * cell,
    }));
    const dist = new Float32Array(n * n).fill(Infinity);
    const next = new Int16Array(n * n).fill(-1);
    for (let i = 0; i < n; i++) {
      dist[i * n + i] = 0; next[i * n + i] = i;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if ((!dx && !dy) || i % cols + dx < 0 || i % cols + dx >= cols ||
            Math.floor(i / cols) + dy < 0 || Math.floor(i / cols) + dy >= rows) continue;
        const j = i + dx + dy * cols;
        if (clear(map, nodes[i], nodes[j], cfg.tank.radius + 1.5)) {
          dist[i * n + j] = Math.hypot(dx, dy) * cell; next[i * n + j] = j;
        }
      }
    }
    for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) {
      const ik = dist[i * n + k];
      if (!Number.isFinite(ik)) continue;
      for (let j = 0; j < n; j++) {
        const alt = ik + dist[k * n + j];
        if (alt < dist[i * n + j]) {
          dist[i * n + j] = alt; next[i * n + j] = next[i * n + k];
        }
      }
    }
    // Local broad phase for repeated tank rollouts. Wall order and the exact
    // two-pass circle solver match GameMap.resolveCircle.
    const localWalls = nodes.map(p => map.walls.filter(w =>
      w.x <= p.x + cell * 1.5 && w.x + w.w >= p.x - cell * 1.5 &&
      w.y <= p.y + cell * 1.5 && w.y + w.h >= p.y - cell * 1.5));
    const bins = nodes.map(p => map.walls.map((w, i) => ({ w, i })).filter(({ w }) =>
      w.x <= p.x + cell * .5 && w.x + w.w >= p.x - cell * .5 &&
      w.y <= p.y + cell * .5 && w.y + w.h >= p.y - cell * .5));
    const stamps = new Uint32Array(map.walls.length);
    let serial = 0;
    const physics = { walls: map.walls, raycast(x, y, xx, yy) {
      const dx = xx - x, dy = yy - y;
      if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return null;
      let c = Math.floor(x / cell), r = Math.floor(y / cell);
      if (c < 0 || c >= cols || r < 0 || r >= rows) return map.raycast(x, y, xx, yy);
      serial = (serial + 1) >>> 0;
      if (!serial) { stamps.fill(0); serial = 1; }
      const sx = Math.sign(dx), sy = Math.sign(dy);
      const deltaX = sx ? Math.abs(cell / dx) : Infinity;
      const deltaY = sy ? Math.abs(cell / dy) : Infinity;
      let tx = sx ? ((sx > 0 ? c + 1 : c) * cell - x) / dx : Infinity;
      let ty = sy ? ((sy > 0 ? r + 1 : r) * cell - y) / dy : Infinity;
      let best = null, bestIndex = Infinity;
      for (let guard = 0; guard < cols + rows + 4; guard++) {
        for (const entry of bins[r * cols + c]) {
          if (stamps[entry.i] === serial) continue;
          stamps[entry.i] = serial;
          const hit = boxHit(x, y, dx, dy, entry.w, 0);
          if (hit && (!best || hit.t < best.t || (hit.t === best.t && entry.i < bestIndex))) {
            best = hit; bestIndex = entry.i;
          }
        }
        if (Math.min(tx, ty) > (best ? best.t : 1)) break;
        if (tx < ty) { c += sx; tx += deltaX; }
        else if (ty < tx) { r += sy; ty += deltaY; }
        else { c += sx; r += sy; tx += deltaX; ty += deltaY; }
        if (c < 0 || c >= cols || r < 0 || r >= rows) break;
      }
      return best;
    }, clearance(p) {
      const c = clamp(Math.floor(p.x / cell), 0, cols - 1);
      const r = clamp(Math.floor(p.y / cell), 0, rows - 1);
      let minimum2 = 10000;
      for (const w of localWalls[r * cols + c]) {
        const dx = Math.max(w.x - p.x, 0, p.x - w.x - w.w);
        const dy = Math.max(w.y - p.y, 0, p.y - w.y - w.h);
        const distance2 = dx * dx + dy * dy;
        if (distance2 < minimum2) minimum2 = distance2;
      }
      return Math.sqrt(minimum2) - p.radius;
    }, resolveCircle(p, radius) {
      let touched = false;
      for (let pass = 0; pass < 2; pass++) {
        const c = clamp(Math.floor(p.x / cell), 0, cols - 1);
        const r = clamp(Math.floor(p.y / cell), 0, rows - 1);
        let hit = false;
        for (const w of localWalls[r * cols + c]) {
          if (p.x + radius < w.x || p.x - radius > w.x + w.w ||
              p.y + radius < w.y || p.y - radius > w.y + w.h) continue;
          const x = clamp(p.x, w.x, w.x + w.w), y = clamp(p.y, w.y, w.y + w.h);
          const dx = p.x - x, dy = p.y - y, length2 = dx * dx + dy * dy;
          if (length2 >= radius * radius) continue;
          if (length2 > 1e-8) {
            const length = Math.sqrt(length2), push = radius - length;
            p.x += dx / length * push; p.y += dy / length * push;
          } else {
            const left = p.x - w.x, right = w.x + w.w - p.x;
            const top = p.y - w.y, bottom = w.y + w.h - p.y;
            const m = Math.min(left, right, top, bottom);
            if (m === left) p.x = w.x - radius;
            else if (m === right) p.x = w.x + w.w + radius;
            else if (m === top) p.y = w.y - radius;
            else p.y = w.y + w.h + radius;
          }
          touched = true; hit = true;
        }
        if (!hit) break;
      }
      return touched;
    } };
    data = { nodes, n, cols, rows, cell, dist, next, physics };
    maps.set(map, data);
    return data;
  }

  function nodeFor(p, data) {
    return clamp(Math.floor(p.y / data.cell), 0, data.rows - 1) * data.cols +
      clamp(Math.floor(p.x / data.cell), 0, data.cols - 1);
  }

  function copyTank(t) {
    return { x: t.x, y: t.y, angle: t.angle, speed: t.speed || 0,
      radius: t.radius || 16 };
  }

  // Mirrors Tank.update's order, acceleration, braking, reverse and wall drag.
  function move(p, drive, steer, dt, map, T) {
    p.angle += steer * T.turn * dt;
    const max = T.speed * (drive < 0 ? T.reverseRatio : 1);
    if (drive) {
      const target = drive > 0 ? max : -max;
      const rate = p.speed * drive < 0 ? T.brake : T.accel;
      p.speed += clamp(target - p.speed, -rate * dt, rate * dt);
    } else {
      p.speed -= clamp(p.speed, -T.brake * .75 * dt, T.brake * .75 * dt);
    }
    const x = p.x, y = p.y;
    p.x += Math.cos(p.angle) * p.speed * dt;
    p.y += Math.sin(p.angle) * p.speed * dt;
    const bumped = map.resolveCircle(p, p.radius);
    if (bumped && Math.abs(p.speed) > 60) p.speed *= .55;
    return Math.hypot(p.x - x, p.y - y);
  }

  function control(action, p, goal, aim, dt, T) {
    let steer = action.turn;
    if (action.mode) {
      let desired = action.mode === 2 ? aim : Math.atan2(goal.y - p.y, goal.x - p.x);
      if (action.mode === 1 && action.drive < 0) desired += PI;
      steer = clamp(diff(p.angle, desired) / (T.turn * Math.max(dt, .001)), -1, 1);
    }
    return { drive: action.drive, steer, shoot: false };
  }

  // The real Bullet.update integrates wall hits inside each tick, displaces the
  // reflected bullet by .8px and consumes .7px of travel. Preserve these details.
  function bulletStep(b, dt, map, B) {
    b.life -= dt; b.grace = Math.max(0, b.grace - dt);
    if (b.life <= 0) { b.dead = true; return; }
    let remaining = b.speed * dt, guard = 0;
    while (remaining > .02 && !b.dead && guard++ < 8) {
      const dx = Math.cos(b.angle) * remaining, dy = Math.sin(b.angle) * remaining;
      const hit = map.raycast(b.x, b.y, b.x + dx, b.y + dy);
      if (!hit) { b.x += dx; b.y += dy; break; }
      const t = clamp(hit.t, 0, 1), nx = hit.nx || 0, ny = hit.ny || 0;
      b.x += dx * t; b.y += dy * t; remaining *= 1 - t;
      if (!nx && !ny) { b.dead = true; break; }
      let vx = Math.cos(b.angle), vy = Math.sin(b.angle);
      const dot = vx * nx + vy * ny;
      if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; }
      b.angle = Math.atan2(vy, vx);
      b.x += nx * .8; b.y += ny * .8;
      if (++b.bounces > B.bounces) b.dead = true;
      remaining -= .7;
    }
  }

  // Minimum synchronous separation between two moving segments, not a static
  // distance to an infinite bullet ray. This also handles fast crossing shots.
  function separation(ax, ay, bx, by, cx, cy, dx, dy) {
    const rx = ax - cx, ry = ay - cy;
    const vx = bx - ax - dx + cx, vy = by - ay - dy + cy;
    const v2 = vx * vx + vy * vy;
    const t = v2 > 1e-9 ? clamp(-(rx * vx + ry * vy) / v2, 0, 1) : 0;
    return Math.hypot(rx + t * vx, ry + t * vy);
  }

  function observe(state, enemies, dt, map, cfg) {
    const models = [];
    for (const tank of enemies) {
      const old = state.observed.get(tank.index);
      const elapsed = old ? state.time - old.time : dt;
      let turn = old && elapsed > .001 ? diff(old.angle, tank.angle) / elapsed : 0;
      turn = clamp(turn, -cfg.tank.turn, cfg.tank.turn);
      if (old) turn = .65 * turn + .35 * old.turn;
      state.observed.set(tank.index, { angle: tank.angle, time: state.time, turn });
      const model = { tank, turn, paths: [] };
      for (let variant = 0; variant < 2; variant++) {
        const p = copyTank(tank), path = [{ x: p.x, y: p.y }];
        for (let k = 1; k <= 24; k++) {
          const t = (k - .5) * .1;
          const angular = variant === 0 ? turn * Math.exp(-Math.max(0, t - .22) * 3) : 0;
          p.angle += angular * .1;
          p.x += Math.cos(p.angle) * p.speed * .1;
          p.y += Math.sin(p.angle) * p.speed * .1;
          map.resolveCircle(p, p.radius);
          path.push({ x: p.x, y: p.y });
        }
        model.paths.push(path);
      }
      models.push(model);
    }
    return models;
  }

  function predicted(model, t, variant) {
    const path = model.paths[variant || 0];
    const x = clamp(t * 10, 0, path.length - 1), i = Math.floor(x), f = x - i;
    const a = path[i], b = path[Math.min(i + 1, path.length - 1)];
    return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f };
  }

  function trajectories(tank, game, cfg, step, count, models) {
    const frames = Array.from({ length: count }, () => []);
    const add = (source, weight, delay) => {
      const b = { x: source.x, y: source.y, angle: source.angle,
        speed: source.speed || cfg.bullet.speed, life: source.life == null ? cfg.bullet.life : source.life,
        bounces: source.bounces || 0, grace: source.grace || 0, dead: false };
      for (let k = 0; k < count && !b.dead; k++) {
        if (k * step < delay) continue;
        const x = b.x, y = b.y;
        // Two substeps reduce 30Hz/60Hz bounce timing differences; retain the
        // actual per-substep collision chord used by updateWorld.
        for (let sub = 0; sub < 2 && !b.dead; sub++) {
          const px = b.x, py = b.y;
          bulletStep(b, step / 2, game.map, cfg.bullet);
          const reach = tank.radius + 40 + cfg.tank.speed * (k + 1) * step;
          const nearby = Math.min(Math.hypot(px - tank.x, py - tank.y),
            Math.hypot(b.x - tank.x, b.y - tank.y)) < reach;
          if (nearby && !b.dead && !(source.owner === tank.index && b.grace > 0)) {
            frames[k].push({ x: px, y: py, xx: b.x, yy: b.y,
              sub, weight, radius: tank.radius + (source.radius || cfg.bullet.radius) });
          }
        }
        if (Math.hypot(b.x - x, b.y - y) < .001) break;
      }
    };
    for (const b of game.bullets) if (!b.dead) add(b, 1, 0);
    // A small virtual danger term anticipates a loaded enemy's current barrel.
    // Observed bullets always outweigh these uncertain hypothetical shots.
    for (const model of models) {
      const e = model.tank, range = Math.sqrt(d2(tank, e));
      if (range > 490 || (e.cooldown || 0) > .3) continue;
      const a = e.angle, muzzle = (e.radius || 16) + 12;
      const source = { x: e.x + Math.cos(a) * muzzle, y: e.y + Math.sin(a) * muzzle,
        angle: a, owner: e.index, grace: 0, life: .8 };
      add(source, .24, Math.max(0, e.cooldown || 0));
      const bearing = Math.atan2(tank.y - e.y, tank.x - e.x);
      const acquisition = Math.abs(diff(a, bearing)) / cfg.tank.turn;
      const delay = Math.max(e.cooldown || 0, acquisition);
      if (delay < .33 && Math.abs(diff(a, bearing)) > .05 &&
          !game.map.raycast(e.x, e.y, tank.x, tank.y)) {
        // Predict a competent enemy acquiring us, without assuming knowledge
        // of that enemy's inputs or controller implementation.
        add({ x: e.x + Math.cos(bearing) * muzzle, y: e.y + Math.sin(bearing) * muzzle,
          angle: bearing, owner: e.index, life: .8 }, .18, delay);
      }
    }
    return frames;
  }

  function chooseTarget(tank, models, data, state, game) {
    const here = nodeFor(tank, data);
    let best = null, bestScore = Infinity;
    for (const model of models) {
      const enemy = model.tank;
      let distance = data.dist[here * data.n + nodeFor(enemy, data)];
      if (!Number.isFinite(distance)) distance = Math.sqrt(d2(tank, enemy)) * 2;
      const visible = !game.map.raycast(tank.x, tank.y, enemy.x, enemy.y);
      const score = distance + (visible ? -110 : 0) - (enemy.index === state.target ? 65 : 0);
      if (score < bestScore) { best = model; bestScore = score; }
    }
    state.target = best ? best.tank.index : -1;
    return best;
  }

  function waypoint(tank, enemy, map, data, radius) {
    if (clear(map, tank, enemy, radius + 2)) return { x: enemy.x, y: enemy.y };
    let here = nodeFor(tank, data), goal = nodeFor(enemy, data), point = data.nodes[here];
    // Return to our cell centre if an off-centre body cannot make the first turn.
    for (let i = 0; i < 9 && here !== goal; i++) {
      const next = data.next[here * data.n + goal];
      if (next < 0 || next === here) break;
      const p = data.nodes[next];
      if (clear(map, tank, p, radius + 2)) point = p;
      else if (i === 0) break;
      here = next;
    }
    return { x: point.x, y: point.y };
  }

  // Trace a proposed shot, retaining its full ricochet path. Used for both aim
  // ranking and the final post-movement, post-turn trigger decision.
  function shotSegments(origin, angle, map, cfg, limit) {
    let x = origin.x + Math.cos(angle) * (origin.radius + 12);
    let y = origin.y + Math.sin(angle) * (origin.radius + 12);
    let t = 0, bounces = 0;
    const segments = [], speed = cfg.bullet.speed;
    while (t < limit && bounces <= cfg.bullet.bounces) {
      const remaining = (limit - t) * speed;
      const vx = Math.cos(angle), vy = Math.sin(angle);
      const hit = map.raycast(x, y, x + vx * remaining, y + vy * remaining);
      const travel = hit ? remaining * clamp(hit.t, 0, 1) : remaining;
      const duration = travel / speed;
      if (duration > .00001) segments.push({ x, y, xx: x + vx * travel,
        yy: y + vy * travel, t, end: t + duration, bounce: bounces });
      if (!hit || (!hit.nx && !hit.ny)) break;
      x += vx * travel + hit.nx * .8; y += vy * travel + hit.ny * .8;
      let rx = vx, ry = vy;
      const dot = vx * hit.nx + vy * hit.ny;
      if (dot < 0) { rx -= 2 * dot * hit.nx; ry -= 2 * dot * hit.ny; }
      angle = Math.atan2(ry, rx);
      t += duration + .7 / speed; bounces++;
    }
    return segments;
  }

  function shotValue(segments, models, cfg) {
    let best = 0, earliest = Infinity;
    for (const model of models) {
      let chance = 0, impact = Infinity;
      for (let variant = 0; variant < 2; variant++) {
        let variantBest = 0;
        for (const s of segments) {
          for (let t = s.t; t < s.end - 1e-7; t += .1) {
            const end = Math.min(s.end, t + .1), span = s.end - s.t;
            const a = (t - s.t) / span, b = (end - s.t) / span;
            const p = predicted(model, t, variant), q = predicted(model, end, variant);
            const miss = separation(s.x + (s.xx - s.x) * a, s.y + (s.yy - s.y) * a,
              s.x + (s.xx - s.x) * b, s.y + (s.yy - s.y) * b, p.x, p.y, q.x, q.y);
            const r = (model.tank.radius || 16) + cfg.bullet.radius;
            const sigma = 3 + t * (Math.abs(model.tank.speed || 0) > 25 ? 20 : 7);
            const hit = clamp((r + sigma - miss) / (2 * sigma), 0, 1);
            const value = hit * Math.exp(-.5 * t) * (s.bounce ? .91 : 1);
            if (value > variantBest) { variantBest = value; impact = t; }
          }
        }
        chance += variantBest * (variant === 0 ? .72 : .28);
      }
      if (chance > best) { best = chance; earliest = impact; }
    }
    return { value: best, time: earliest };
  }

  function aiming(tank, target, models, game, cfg, state) {
    const candidates = [], muzzle = tank.radius + 12, speed = cfg.bullet.speed;
    const add = (a, priority) => candidates.push({ a, priority });
    for (const model of models) {
      let flight = Math.max(0, (Math.sqrt(d2(tank, model.tank)) - muzzle) / speed);
      let p;
      for (let k = 0; k < 4; k++) {
        p = predicted(model, flight);
        flight = Math.max(0, (Math.sqrt(d2(tank, p)) - muzzle) / speed);
      }
      const direct = Math.atan2(p.y - tank.y, p.x - tank.x);
      add(direct, 1 - flight * .2 - Math.abs(diff(tank.angle, direct)) * .055);
      // A separate present-position aim covers enemies braking at a corner.
      const now = Math.atan2(model.tank.y - tank.y, model.tank.x - tank.x);
      if (Math.abs(diff(now, direct)) > .035) add(now, .74 - flight * .2);
      if (model !== target && d2(tank, model.tank) > 650 * 650) continue;
      for (const wall of game.map.walls) {
        const faces = [];
        if (tank.x < wall.x) faces.push({ axis: 0, plane: wall.x, lo: wall.y, hi: wall.y + wall.h });
        else if (tank.x > wall.x + wall.w) faces.push({ axis: 0, plane: wall.x + wall.w, lo: wall.y, hi: wall.y + wall.h });
        if (tank.y < wall.y) faces.push({ axis: 1, plane: wall.y, lo: wall.x, hi: wall.x + wall.w });
        else if (tank.y > wall.y + wall.h) faces.push({ axis: 1, plane: wall.y + wall.h, lo: wall.x, hi: wall.x + wall.w });
        for (const face of faces) {
          let time = flight, q, mirror;
          for (let k = 0; k < 3; k++) {
            q = predicted(model, time);
            mirror = face.axis ? { x: q.x, y: 2 * face.plane - q.y } :
              { x: 2 * face.plane - q.x, y: q.y };
            time = Math.max(0, (Math.sqrt(d2(tank, mirror)) - muzzle) / speed);
          }
          if (time > 1.65) continue;
          const own = face.axis ? tank.y : tank.x, other = face.axis ? q.y : q.x;
          if ((own - face.plane) * (other - face.plane) <= 0) continue;
          const dest = face.axis ? mirror.y : mirror.x;
          const f = (face.plane - own) / (dest - own);
          const cross = face.axis ? tank.x + (mirror.x - tank.x) * f : tank.y + (mirror.y - tank.y) * f;
          if (f <= 0 || f >= 1 || cross < face.lo + 1 || cross > face.hi - 1) continue;
          const angle = Math.atan2(mirror.y - tank.y, mirror.x - tank.x);
          const startX = tank.x + Math.cos(angle) * muzzle;
          const startY = tank.y + Math.sin(angle) * muzzle;
          const first = game.map.raycast(startX, startY, mirror.x, mirror.y);
          if (!first || (!first.nx && !first.ny)) continue;
          const hitX = startX + (mirror.x - startX) * first.t;
          const hitY = startY + (mirror.y - startY) * first.t;
          const expectedX = face.axis ? cross : face.plane;
          const expectedY = face.axis ? face.plane : cross;
          if (Math.hypot(hitX - expectedX, hitY - expectedY) > 1.5 ||
              game.map.raycast(hitX + first.nx * .8, hitY + first.ny * .8, q.x, q.y)) continue;
          add(angle, .88 - time * .25 - Math.abs(diff(tank.angle, angle)) * .055);
        }
      }
    }
    add(tank.angle, .9);
    if (state.aim != null) add(state.aim, .94);
    candidates.sort((a, b) => b.priority - a.priority);
    let best = { angle: Math.atan2(target.tank.y - tank.y, target.tank.x - tank.x), value: 0 };
    const tried = [];
    for (const c of candidates) {
      if (tried.some(a => Math.abs(diff(a, c.a)) < .012)) continue;
      tried.push(c.a);
      const score = shotValue(shotSegments(tank, c.a, game.map, cfg, 1.8), models, cfg);
      // A turning enemy can move during our aim acquisition. Discount long turns.
      const turnTime = Math.abs(diff(tank.angle, c.a)) / cfg.tank.turn;
      const value = score.value * Math.exp(-.7 * turnTime);
      if (value > best.value) best = { angle: c.a, value };
      if (tried.length >= 20) break;
    }
    return best;
  }

  function tacticalGoal(tank, target, aim, game, data, cfg, state) {
    const enemy = target.tank, range = Math.sqrt(d2(tank, enemy));
    const visible = !game.map.raycast(tank.x, tank.y, enemy.x, enemy.y);
    // In a crowded free-for-all, exchanging lives loses the round. The spawn
    // is chosen behind cover by the engine: use that protection while the
    // crowd thins. A bounded wait prevents passive opponents from stalemating.
    if (game.tanks.filter(t => t.alive).length >= 5 && state.time < 6 &&
        (!visible || range > 260)) return state.anchor;
    if (!visible || range > 440 || aim.value < .18) {
      return waypoint(tank, enemy, game.map, data, tank.radius);
    }
    const away = Math.atan2(tank.y - enemy.y, tank.x - enemy.x);
    const enemyAim = Math.abs(diff(enemy.angle, away));
    // Stay out of point-blank trades; after a shot, use the reload interval to
    // move across the enemy barrel instead of sitting on the firing line.
    if (range < 150 && enemyAim < 1.1) {
      const p = { x: tank.x + Math.cos(away) * 110, y: tank.y + Math.sin(away) * 110 };
      if (clear(game.map, tank, p, tank.radius + 2)) return p;
    }
    const a = Math.atan2(enemy.y - tank.y, enemy.x - tank.x);
    let best = null, bestValue = -Infinity;
    for (const sign of [-1, 1]) {
      const b = a + sign * 1.22;
      const p = { x: tank.x + Math.cos(b) * 90, y: tank.y + Math.sin(b) * 90 };
      if (!clear(game.map, tank, p, tank.radius + 2)) continue;
      const align = Math.min(Math.abs(diff(tank.angle, b)), Math.abs(diff(tank.angle, b + PI)));
      const value = -align + (state.side === sign ? .17 : 0);
      if (value > bestValue) { bestValue = value; best = p; state.side = sign; }
    }
    return best || { x: tank.x, y: tank.y };
  }

  function plan(tank, game, cfg, goal, aim, models, state, dt) {
    const step = 1 / 30, stages = [4, 5, 7, 8], count = 24;
    const frames = trajectories(tank, game, cfg, step, count, models);
    const bodies = models.filter(m => d2(tank, m.tank) < 360 * 360).map(model => ({
      radius: model.tank.radius || 16,
      path: Array.from({ length: count }, (_, k) => predicted(model, (k + 1) * step)),
    }));
    const actions = [];
    for (const drive of [-1, 0, 1]) {
      for (const turn of [-1, 0, 1]) actions.push({ drive, turn });
      actions.push({ drive, mode: 1 });
      if (aim.value > .10) actions.push({ drive, mode: 2 });
    }
    let beam = [{ p: copyTank(tank), cost: 0, risk: 0, path: [], first: null }], frame = 0;
    const aimWeight = aim.value > .10 ? (tank.cooldown < .14 ? 35 : 14) * aim.value : 0;
    const initialDistance = Math.sqrt(d2(tank, goal));
    for (let stage = 0; stage < stages.length; stage++) {
      const pool = [], steps = stages[stage];
      for (const parent of beam) for (const action of actions) {
        const p = Object.assign({}, parent.p), path = parent.path.slice();
        let cost = parent.cost, risk = parent.risk, first = parent.first;
        for (let sub = 0; sub < steps; sub++) {
          const k = frame + sub, time = (k + 1) * step;
          const input = control(action, p, goal, aim.angle, step, cfg.tank);
          if (!first) first = control(action, p, goal, aim.angle, dt, cfg.tank);
          const x = p.x, y = p.y;
          const progress = move(p, input.drive, input.steer, step, state.physics, cfg.tank);
          if (input.drive && progress < .5) cost += .8;
          // No assumption that another tank will absorb a dangerous bullet.
          for (const b of frames[k]) {
            const f0 = b.sub * .5, f1 = f0 + .5;
            const miss = separation(x + (p.x - x) * f0, y + (p.y - y) * f0,
              x + (p.x - x) * f1, y + (p.y - y) * f1, b.x, b.y, b.xx, b.yy);
            const margin = miss - b.radius;
            if (margin < 2) {
              const hit = b.weight * (1 - time * .30);
              cost += 13000 * hit; risk += hit;
            } else if (margin < 33) {
              cost += b.weight * 32 * Math.exp(-margin / 9) * (1 - time * .35);
            }
          }
          for (const body of bodies) {
            const q = body.path[k];
            const distance = Math.sqrt(d2(p, q));
            if (distance < p.radius + body.radius + 4) cost += 110;
          }
          cost += Math.abs(input.steer) * .04;
          // Charge aim delay along the path, not only at the endpoint. A purely
          // terminal aim reward otherwise keeps postponing the shot forever.
          cost += aimWeight * (tank.cooldown < .14 ? .12 : .025) *
            Math.min(2, Math.abs(diff(p.angle, aim.angle)));
          path.push({ x: p.x, y: p.y, angle: p.angle, speed: p.speed });
        }
        const wallGap = state.physics.clearance(p);
        if (wallGap < 32) cost += steps * 1.3 * ((32 - wallGap) / 32) ** 2;
        const distance = Math.sqrt(d2(p, goal));
        const turn = Math.abs(diff(p.angle, aim.angle));
        const navigation = clamp(distance - initialDistance, -150, 200) * .26;
        const bearing = Math.atan2(goal.y - p.y, goal.x - p.x);
        const navTurn = distance > 23 ? Math.min(Math.abs(diff(p.angle, bearing)),
          Math.abs(diff(p.angle, bearing + PI))) * (aim.value > .18 ? 1.2 : 7) : 0;
        const aimCost = aimWeight * Math.min(turn, 2);
        const inertia = stage === 0 && state.lastInput ?
          (first.drive !== state.lastInput.drive ? .65 : 0) + Math.abs(first.steer - state.lastInput.steer) * .12 : 0;
        pool.push({ p, cost, risk, path, first, rank: cost + navigation + navTurn + aimCost + inertia });
      }
      pool.sort((a, b) => a.rank - b.rank);
      // Keep distinct manoeuvres so one greedy early branch does not exclude
      // a safe late reversal. The search remains strictly bounded.
      beam = [];
      for (const candidate of pool) {
        if (beam.some(b => d2(b.p, candidate.p) < 9 &&
          Math.abs(diff(b.p.angle, candidate.p.angle)) < .12)) continue;
        beam.push(candidate);
        if (beam.length === 3) break;
      }
      if (!beam.length) beam = [pool[0]];
      frame += steps;
    }
    return { input: beam[0].first, path: beam[0].path, risk: beam[0].risk, step };
  }

  function safeShot(tank, future, input, game, cfg, dt) {
    const p = copyTank(tank);
    move(p, input.drive, input.steer, dt, game.map, cfg.tank);
    const b = { x: p.x + Math.cos(p.angle) * (p.radius + 12),
      y: p.y + Math.sin(p.angle) * (p.radius + 12), angle: p.angle,
      speed: cfg.bullet.speed, life: cfg.bullet.life, grace: cfg.bullet.spawnGrace,
      bounces: 0, dead: false };
    // Test the actual first tick; a barrel embedded in a wall simply fizzles.
    const bx = b.x, by = b.y;
    bulletStep(b, dt, game.map, cfg.bullet);
    if (b.dead) return null;
    if (b.grace <= 0 && separation(p.x, p.y, p.x, p.y, bx, by, b.x, b.y) < p.radius + cfg.bullet.radius + 1) return null;
    for (let k = 1; k < future.path.length && !b.dead; k++) {
      const prev = future.path[k - 1], next = future.path[k];
      const x = b.x, y = b.y;
      bulletStep(b, future.step, game.map, cfg.bullet);
      if (!b.dead && b.grace <= 0 && separation(prev.x, prev.y, next.x, next.y,
        x, y, b.x, b.y) < p.radius + cfg.bullet.radius + 4) return null;
    }
    return p;
  }

  function updateBot(tank, game, dt) {
    if (!tank || !tank.alive || !game || !game.map || !Array.isArray(game.tanks)) return idle();
    const enemies = game.tanks.filter(t => t.alive && t.index !== tank.index);
    if (!enemies.length) return idle();
    dt = clamp(Number.isFinite(dt) ? dt : 1 / 60, .001, 1 / 30);
    let state = memories.get(tank);
    if (!state || state.map !== game.map) {
      state = { map: game.map, time: 0, observed: new Map(), target: -1,
        aim: tank.angle, aimAt: -99, planAt: -99, lastInput: null, side: tank.index % 2 ? 1 : -1,
        anchor: { x: tank.x, y: tank.y } };
      memories.set(tank, state);
    }
    state.time += dt;
    const cfg = config(game), data = mapData(game.map, cfg);
    state.physics = data.physics;
    const world = { map: data.physics, tanks: game.tanks, bullets: game.bullets };
    const models = observe(state, enemies, dt, data.physics, cfg);
    const target = chooseTarget(tank, models, data, state, world);
    if (!target) return idle();
    if (state.time - state.aimAt >= .09 || !state.aimResult) {
      state.aimResult = aiming(tank, target, models, world, cfg, state);
      state.aim = state.aimResult.angle; state.aimAt = state.time;
    }
    const aim = state.aimResult;
    // Replan at 20Hz, immediately when a new projectile appears. Steering and
    // the final trigger are still evaluated at the game's own frame rate.
    const bulletSignature = game.bullets.length;
    if (state.time - state.planAt >= .045 || !state.future || state.bulletSignature !== bulletSignature) {
      const goal = tacticalGoal(tank, target, aim, world, data, cfg, state);
      state.future = plan(tank, world, cfg, goal, aim, models, state, dt);
      state.planAt = state.time; state.bulletSignature = bulletSignature;
    }
    const input = Object.assign({}, state.future.input);
    // Avoid overshooting an exact firing angle with cached fractional steering.
    const da = diff(tank.angle, aim.angle);
    if (aim.value > .18 && Math.sign(input.steer) === Math.sign(da) &&
        Math.abs(da) < Math.abs(input.steer) * cfg.tank.turn * dt) {
      input.steer = da / (cfg.tank.turn * dt);
    }
    let live = 0;
    for (const b of game.bullets) if (!b.dead && b.owner === tank.index) live++;
    if (tank.cooldown <= dt + 1e-6 && live < cfg.cap) {
      const p = safeShot(tank, state.future, input, world, cfg, dt);
      if (p) {
        const score = shotValue(shotSegments(p, p.angle, world.map, cfg, 1.65), models, cfg);
        input.shoot = score.value >= .25;
      }
    }
    state.lastInput = input;
    return { drive: clamp(input.drive, -1, 1), steer: clamp(input.steer, -1, 1), shoot: !!input.shoot };
  }

  updateBot.updateBot = updateBot;
  updateBot.update = updateBot;
  updateBot.policyName = 'GPT Sentinel';
  return updateBot;
});
