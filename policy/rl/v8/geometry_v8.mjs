/* Schema V3 firing-solution solver.
   ===========================================================
   v2/geometry.mjs 的 traceShot 有一处自伤判定缺陷，V4–V7 全程继承：

     const x = me.x + Math.cos(angle)*offset        // 枪口按「候选方向」算
     const tvx = Math.cos(tank.angle)*tank.speed    // 自机运动按「当前朝向」外推

   枪口用的是对齐后的方向 angle，自机运动外推用的是对齐前的朝向 me.angle，两者相差
   err —— 而 err 正是伺服要闭合的量。所有分歧发生在误差 42.8°–155.0° 的未对齐候选上
   （实测 3321 帧 / 45445 候选：v2 判安全而 v8 判自伤占 v2 安全判定的 2.34%，
   has_safe 翻转 1.26% 的帧）。真实危害是伺服选错转向目标、白转 0.6 秒，损失节奏。

   本文件的修正：自机沿它「真正会走的轨迹」外推 —— 先以最大转速 3.3 rad/s 转向候选
   方向（entities.js:53），对齐后沿该方向直行。圆弧段位置解析可解：
     圆弧 (t ≤ tAlign): x(t) = x0 + (v/w)(sin(θ0+wt) − sinθ0)
                        y(t) = y0 − (v/w)(cos(θ0+wt) − cosθ0)
     直行 (t > tAlign): 沿 angle 匀速
   自机路径不再是直线，故自机碰撞改为 1/60s 采样圆重叠（子弹每帧 10.3px、命中半径
   19.2px，不会漏判）。其他坦克仍用匀速解析求交 —— 不知道它们的指令，匀速是唯一
   诚实假设。

   --- 性能 ---
   js/map.js 的 raycast 对每面被命中的墙都分配一个 {t,nx,ny}，而一条 1509px 的射线
   会命中数十面墙，实测 45μs/次；采集中每帧约 694 次 raycast（64 lidar + 105 flight
   内的反弹），单帧 31ms 全耗在这里。下面给出逐位等价的无分配实现（rayBoxFast 复刻
   U.rayBox 的 slab 语义，包括 tmin 从 0 起算），以及用它重写的 flightV8。
   等价性由 tests 校验，不靠假设。

   只用于离线生成标签，不在推理路径上。
*/

const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/* ---- 无分配 rayBox：复刻 js/util.js:168 U.rayBox 的语义 ---- */
let RB_T = 0, RB_NX = 0, RB_NY = 0;
function rayBoxFast(ox, oy, dx, dy, minx, miny, maxx, maxy) {
  let tmin = 0, tmax = 1, nx = 0, ny = 0;
  if (dx > -1e-9 && dx < 1e-9) {
    if (ox < minx || ox > maxx) return false;
  } else {
    const inv = 1 / dx;
    let t1 = (minx - ox) * inv, t2 = (maxx - ox) * inv, s = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; s = 1; }
    if (t1 > tmin) { tmin = t1; nx = s; ny = 0; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  if (dy > -1e-9 && dy < 1e-9) {
    if (oy < miny || oy > maxy) return false;
  } else {
    const inv = 1 / dy;
    let t1 = (miny - oy) * inv, t2 = (maxy - oy) * inv, s = -1;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; s = 1; }
    if (t1 > tmin) { tmin = t1; nx = 0; ny = s; }
    if (t2 < tmax) tmax = t2;
    if (tmin > tmax) return false;
  }
  RB_T = tmin; RB_NX = nx; RB_NY = ny;
  return true;
}

/* ---- 无分配 raycast：命中时结果写入 RC_*（ES module 活绑定，导入方读到的就是最新值） ---- */
export let RC_T = 0, RC_NX = 0, RC_NY = 0;
export function raycastInto(map, x0, y0, x1, y1, radius = 0) {
  const dx = x1 - x0, dy = y1 - y0;
  if (dx > -1e-9 && dx < 1e-9 && dy > -1e-9 && dy < 1e-9) return false;
  const walls = map.walls;
  let bestT = 2, bestNX = 0, bestNY = 0, found = false;
  for (let i = 0; i < walls.length; i++) {
    const w = walls[i];
    if (rayBoxFast(x0, y0, dx, dy, w.x - radius, w.y - radius,
                   w.x + w.w + radius, w.y + w.h + radius) && RB_T < bestT) {
      bestT = RB_T; bestNX = RB_NX; bestNY = RB_NY; found = true;
    }
  }
  if (found) { RC_T = bestT; RC_NX = bestNX; RC_NY = bestNY; }
  return found;
}

/* ---- 用无分配 raycast 重写的 flight，与 v2 的 flight 逐位等价 ---- */
export function flightV8(game, x, y, angle, speed, seconds, bounces = 0) {
  const cfg = game.__cfg, segments = [];
  let remaining = speed * seconds, time = 0;
  for (let guard = 0; remaining > .02 && guard < 8; guard++) {
    let vx = Math.cos(angle), vy = Math.sin(angle);
    const hit = raycastInto(game.map, x, y, x + vx * remaining, y + vy * remaining, cfg.BULLET.radius);
    const length = remaining * (hit ? clamp(RC_T, 0, 1) : 1), duration = length / speed;
    segments.push({ x, y, vx: vx * speed, vy: vy * speed, time, duration });
    x += vx * length; y += vy * length; remaining -= length; time += duration;
    if (!hit) break;
    const nx = RC_NX || 0, ny = RC_NY || 0;
    if ((!nx && !ny) || ++bounces > cfg.BULLET.bounces) break;
    const dot = vx * nx + vy * ny;
    if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; }
    angle = Math.atan2(vy, vx); x += nx * .8; y += ny * .8; remaining -= .7;
  }
  return segments;
}

/* 自机在 t 秒后的位姿：先以最大转速转向 targetAngle，对齐后直行。 */
export function selfPose(me, targetAngle, t, turnRate) {
  const e0 = wrap(targetAngle - me.angle);
  const w = Math.sign(e0) * turnRate;
  const v = me.speed;
  if (Math.abs(e0) < 1e-6 || Math.abs(w) < 1e-9) {
    return { x: me.x + Math.cos(me.angle) * v * t, y: me.y + Math.sin(me.angle) * v * t, angle: me.angle };
  }
  const tAlign = Math.abs(e0) / turnRate;
  const thA = me.angle + w * tAlign;
  const ax = (v / w) * (Math.sin(thA) - Math.sin(me.angle));
  const ay = -(v / w) * (Math.cos(thA) - Math.cos(me.angle));
  if (t <= tAlign) {
    const th = me.angle + w * t;
    return { x: me.x + (v / w) * (Math.sin(th) - Math.sin(me.angle)),
             y: me.y - (v / w) * (Math.cos(th) - Math.cos(me.angle)), angle: th };
  }
  return { x: me.x + ax + Math.cos(targetAngle) * v * (t - tAlign),
           y: me.y + ay + Math.sin(targetAngle) * v * (t - tAlign), angle: targetAngle };
}

/* 预采样自机轨迹。只在自伤判定窗口内需要。 */
function selfTrack(me, targetAngle, horizon, turnRate, dt) {
  const n = Math.max(1, Math.ceil(horizon / dt));
  const out = new Array(n + 1);
  for (let i = 0; i <= n; i++) out[i] = selfPose(me, targetAngle, i * dt, turnRate);
  return out;
}

/* 修正版 traceShot：签名与 v2 版一致，多返回 selfTime 便于诊断。 */
export function traceShotV8(game, me, angle, opts = {}) {
  const cfg = game.__cfg;
  const horizon = opts.selfHorizon ?? 2.0;
  const dt = 1 / 60;
  const offset = me.radius + 12;
  const x = me.x + Math.cos(angle) * offset, y = me.y + Math.sin(angle) * offset;
  if (raycastInto(game.map, me.x, me.y, x, y, cfg.BULLET.radius)) {
    return { hit: false, self: false, time: 0, blocked: true, selfTime: Infinity };
  }
  const segments = flightV8(game, x, y, angle, cfg.BULLET.speed, cfg.BULLET.life);
  let earliest = Infinity, victim = null, selfTime = Infinity;

  const track = selfTrack(me, angle, horizon, cfg.TANK.turn, dt);
  const hitRadius = me.radius + cfg.BULLET.radius;

  for (const segment of segments) {
    /* --- 其他坦克：匀速解析求交 --- */
    for (const tank of game.tanks) {
      if (!tank.alive || tank.index === me.index) continue;
      const tvx = Math.cos(tank.angle) * tank.speed, tvy = Math.sin(tank.angle) * tank.speed;
      const rx = segment.x - tank.x - tvx * segment.time;
      const ry = segment.y - tank.y - tvy * segment.time;
      const t = game.__util.segCircle(rx, ry, rx + (segment.vx - tvx) * segment.duration,
                                      ry + (segment.vy - tvy) * segment.duration,
                                      0, 0, tank.radius + cfg.BULLET.radius);
      if (t >= 0 && segment.time + t * segment.duration < earliest) {
        earliest = segment.time + t * segment.duration; victim = tank;
      }
    }
    /* --- 自机：沿「转向后直行」的真实轨迹采样 --- */
    if (me.alive && segment.time < horizon) {
      const lo = Math.max(0, cfg.BULLET.spawnGrace - segment.time);
      const steps = Math.ceil(segment.duration / dt);
      for (let i = 0; i <= steps; i++) {
        const s = lo + i * dt;
        if (s > segment.duration) break;
        const when = segment.time + s;
        if (when >= earliest) break;
        const bx = segment.x + segment.vx * s, by = segment.y + segment.vy * s;
        const k = Math.min(track.length - 1, Math.round(when / dt));
        const p = track[k];
        const dxb = bx - p.x, dyb = by - p.y;
        if (dxb * dxb + dyb * dyb <= hitRadius * hitRadius && when < selfTime) {
          selfTime = when;
          if (when < earliest) { earliest = when; victim = me; }
        }
      }
    }
  }
  return { hit: !!victim && victim.index !== me.index, self: victim?.index === me.index,
           time: Number.isFinite(earliest) ? earliest : cfg.BULLET.life,
           blocked: false, selfTime };
}

/* 固定角度基上的瞄准真值。基是等间隔自我中心方向，不含任何求解器偏好，
   所以网络必须自己找出哪个方向好 —— 这是「求解器只当标签预言机」的落地点。 */
export function aimTruthBasis(game, me, count = 64) {
  const out = { hit_enemy: new Float32Array(count), hit_self: new Float32Array(count),
                ttc: new Float32Array(count) };
  if (!me.alive) return out;
  const life = game.__cfg.BULLET.life;
  for (let i = 0; i < count; i++) {
    const phi = me.angle + i * (2 * Math.PI / count);
    const r = traceShotV8(game, me, phi);
    out.hit_enemy[i] = r.hit ? 1 : 0;
    out.hit_self[i] = r.self ? 1 : 0;
    out.ttc[i] = Math.min(1, (r.blocked ? 0 : r.time) / life);
  }
  return out;
}

/* 精确伺服目标：用 v2 的候选生成（含 intercept 提前量解与镜像反弹解）但换成修正版
   自伤判定。返回 V7 同款控制律所需的四个量。 */
export function servoTarget(game, me, candidates) {
  const best = { err: NaN, cone: NaN, has_safe: false };
  if (!me.alive) return best;
  for (const c of candidates) {
    if (!c.hit || c.blocked) continue;
    const r = traceShotV8(game, me, c.angle);
    if (r.self || !r.hit) continue;
    const err = wrap(c.angle - me.angle);
    if (!best.has_safe || Math.abs(err) < Math.abs(best.err)) {
      best.err = err;
      best.cone = Math.atan2(c.target.radius + game.__cfg.BULLET.radius, Math.max(1, c.distance));
      best.has_safe = true;
    }
  }
  return best;
}
