/* Schema V3 observation encoder + label generator.
   ===========================================================
   输入 = 物理状态 + 静态地图的传感器读数。求解器输出一律不出现在输入里，
   只作为训练期标签（见 DESIGN.md §1/§2、公理 1）。

   observe(tracker) 在「step 之前」调用，返回 608 维每帧输入块。
   labels(tracker, trueAction) 在「step 之后」调用，返回 333 维标签块。
   mapVector(game) 每回合调用一次，返回 346 维回合级块。

   对手动作通道用「相邻帧物理状态差」恢复，不用重放的真值 —— 因为浏览器里没有
   真值，两条路径必须完全一致。三个量都要求精确：
     steer_k  角度只被 entities.js:53 改变，所以 Δθ/(turn·dt) 就是指令本身
     shoot_k  cooldown 只被 entities.js:46 衰减、entities.js:93 置为 fireCooldown，
              且置位在哑火检查之前，所以 Δcooldown > 0 精确等价于「本帧尝试开火」
     dspeed_k 用加速度本身（物理可观测量）而不是离散 drive 指令，避开
              「满速饱和 Δspeed=0」与「撞墙 speed*=0.55」两种歧义
*/
import { aimCandidates } from '../v2/geometry.mjs';
import {
  traceShotV8, servoTarget, raycastInto, flightV8, RC_T, RC_NX, RC_NY,
} from './geometry_v8.mjs';

const DT = 1 / 60;
const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

export const SCHEMA_V3 = Object.freeze({
  version: 3,
  frame: 608, map: 346, labels: 333, record: 608 + 333 + 0,

  SELF: 0, SELF_ACTION: 8,
  ENEMY: 11, ENEMY_STRIDE: 9,
  ENEMY_ACTION: 74, ENEMY_ACTION_STRIDE: 3,
  BULLET: 95, BULLET_STRIDE: 8,
  LIDAR: 415, LIDAR_N: 64, LIDAR_STRIDE: 3,
  CLOCK: 607,

  L_TEACHER: 0, L_SERVO: 3, L_AIM: 7, L_AIM_N: 64, L_AIM_STRIDE: 3,
  L_THREAT: 199, L_THREAT_N: 40, L_THREAT_STRIDE: 3,
  L_INTENT: 319, L_INTENT_N: 7, L_INTENT_STRIDE: 2,

  ENEMY_SLOTS: 7, BULLET_SLOTS: 40,
  MAX_AMMO: 5, TANK_SPEED: 178, TURN_RATE: 3.3, FIRE_COOLDOWN: 0.42,
  MAX_BOUNCES: 5, BULLET_LIFE: 6, ROUND_SECONDS: 60,
});

const S = SCHEMA_V3;

export function createTracker(game, seat) {
  return {
    game, seat,
    prevState: null,        // 上一帧观测时刻的全场状态快照
    prevBearing: null,      // 上一帧各对手的相对方位/距离
    prevAction: null,       // 上一帧自机动作
  };
}

/* 在 step 之前调用：把「本帧观测时刻」的全场状态存成快照，供下一帧差分。 */
export function snapshotTracker(tracker) {
  const game = tracker.game;
  tracker.prevState = game.tanks.map(t => ({
    x: t.x, y: t.y, angle: t.angle, speed: t.speed, cooldown: t.cooldown, alive: t.alive,
  }));
  const me = game.tanks[tracker.seat];
  tracker.prevBearing = game.tanks.map(t => t.index === tracker.seat ? null : {
    bearing: Math.atan2(t.y - me.y, t.x - me.x),
    range: Math.hypot(t.x - me.x, t.y - me.y),
  });
}

/* 在 step 之后调用：记录实际执行的指令（自机）。 */
export function recordAction(tracker, action) {
  tracker.prevAction = { drive: action.drive, steer: action.steer, shoot: action.shoot };
}

/* ---------------- 回合级块：346 维墙格 ---------------- */
export function mapVector(game) {
  const out = new Float32Array(S.map);
  let at = 0;
  for (let r = 0; r < 10; r++) for (let c = 0; c < 17; c++) out[at++] = game.map.vAt(r, c);
  for (let r = 0; r < 11; r++) for (let c = 0; c < 16; c++) out[at++] = game.map.hAt(r, c);
  return out;
}

/* ---------------- 每帧输入块：608 维 ---------------- */
export function observe(tracker) {
  const game = tracker.game, seat = tracker.seat, cfg = game.__cfg;
  const me = game.tanks[seat];
  const out = new Float32Array(S.frame);
  const W = cfg.W, H = cfg.H, diagonal = Math.hypot(W, H);
  const cos = Math.cos(me.angle), sin = Math.sin(me.angle);
  const body = (x, y) => [x * cos + y * sin, -x * sin + y * cos];

  const live = game.bullets.filter(b => !b.dead);
  const ammo = Array(8).fill(0);
  for (const b of live) ammo[b.owner]++;

  /* --- 自机 8 --- */
  out[0] = me.x / W; out[1] = me.y / H;
  out[2] = cos; out[3] = sin;
  out[4] = me.speed / S.TANK_SPEED;
  out[5] = me.cooldown / S.FIRE_COOLDOWN;
  out[6] = +me.alive;
  out[7] = Math.min(S.MAX_AMMO, ammo[me.index]) / S.MAX_AMMO;

  /* --- 自机上帧动作 3 --- */
  const pa = tracker.prevAction || { drive: 0, steer: 0, shoot: false };
  out[S.SELF_ACTION] = pa.drive;
  out[S.SELF_ACTION + 1] = pa.steer;
  out[S.SELF_ACTION + 2] = +pa.shoot;

  /* --- 对手 7×9 --- */
  const prev = tracker.prevState;
  let slot = 0;
  for (const t of game.tanks) {
    if (t.index === seat) continue;
    if (slot >= S.ENEMY_SLOTS) break;
    const base = S.ENEMY + slot * S.ENEMY_STRIDE;
    const dx = t.x - me.x, dy = t.y - me.y;
    const [bx, by] = body(dx, dy);
    out[base] = t.x / W; out[base + 1] = t.y / H;
    out[base + 2] = Math.cos(t.angle); out[base + 3] = Math.sin(t.angle);
    out[base + 4] = t.speed / S.TANK_SPEED;
    out[base + 5] = t.cooldown / S.FIRE_COOLDOWN;
    out[base + 6] = +t.alive;
    out[base + 7] = 1;                                   // presence
    out[base + 8] = Math.min(S.MAX_AMMO, ammo[t.index]) / S.MAX_AMMO;

    /* --- 对手动作 3：由相邻帧状态差精确恢复 --- */
    const ab = S.ENEMY_ACTION + slot * S.ENEMY_ACTION_STRIDE;
    const p = prev ? prev[t.index] : null;
    if (p && p.alive && t.alive) {
      out[ab] = Math.max(-1, Math.min(1, wrap(t.angle - p.angle) / (S.TURN_RATE * DT)));
      out[ab + 1] = +((t.cooldown - p.cooldown) > 1e-6);
      out[ab + 2] = Math.max(-2, Math.min(2, (t.speed - p.speed) / (920 * DT)));
    }
    slot++;
  }

  /* --- 子弹 40×8 --- */
  let bs = 0;
  for (const b of live) {
    if (bs >= S.BULLET_SLOTS) break;
    const base = S.BULLET + bs * S.BULLET_STRIDE;
    out[base] = b.x / W; out[base + 1] = b.y / H;
    out[base + 2] = Math.cos(b.angle); out[base + 3] = Math.sin(b.angle);
    out[base + 4] = b.life / S.BULLET_LIFE;
    out[base + 5] = b.bounces / S.MAX_BOUNCES;
    out[base + 6] = b.owner / 7;
    out[base + 7] = 1;
    bs++;
  }

  /* --- 自我中心 lidar 64×3：距离 + body-frame 墙面法向 --- */
  /* 探针半径必须为 0：raycast 用 (w - radius) 膨胀墙体且 rayBox 的 tmin 从 0 起算，
     坦克中心距墙 16px 时（贴墙行驶的常态）起点已落在膨胀区内，会直接返回 t=0。
     radius=0 得到的是「中心到墙面」的真实距离，网络可自行与自身半径比较。 */
  if (me.alive) {
    for (let i = 0; i < S.LIDAR_N; i++) {
      const a = me.angle + i * (2 * Math.PI / S.LIDAR_N);
      const base = S.LIDAR + i * S.LIDAR_STRIDE;
      if (raycastInto(game.map, me.x, me.y, me.x + Math.cos(a) * diagonal,
                      me.y + Math.sin(a) * diagonal, 0)) {
        out[base] = Math.max(0, Math.min(1, RC_T));
        const [nx, ny] = body(RC_NX || 0, RC_NY || 0);
        out[base + 1] = nx; out[base + 2] = ny;
      } else {
        out[base] = 1;
      }
    }
  }

  out[S.CLOCK] = Math.max(0, 1 - game.roundTime / S.ROUND_SECONDS);
  return out;
}

/* 每帧威胁摘要：[最近敌方弹归一化距离, 必中敌方弹数, ttc, 自机弹必中数]。
   只做 RL 奖励（躲避/自击），不进入 608 维输入。 */
export function threatInfo(game, seat) {
  const me = game.tanks[seat], diagonal = Math.hypot(1280, 800);
  let best = 1, willHit = 0, ttc = 1.2, ownWill = 0;
  for (const b of game.bullets) {
    if (b.dead) continue;
    const segments = flightV8(game, b.x, b.y, b.angle, b.speed,
                              Math.min(1.2, b.life), b.bounces);
    const d = closest(segments, me, b.owner === me.index ? b.grace : 0);
    if (b.owner === me.index) {
      if (d.distance < me.radius + b.radius) ownWill++;
    } else {
      if (d.distance / diagonal < best) { best = d.distance / diagonal; ttc = d.time; }
      if (d.distance < me.radius + b.radius) willHit++;
    }
  }
  return [best, willHit, ttc, ownWill];
}

/* 当前炮口方向开一枪的结果：[是否命中敌人, 是否自击, 炮口是否通畅]。
   用于 L0 的“可命中锥内开火”奖励，不做输入。 */
export function shotInfo(game, seat) {
  const me = game.tanks[seat];
  if (!me.alive) return [0, 0, 0];
  const r = traceShotV8(game, me, me.angle);
  return [r.hit ? 1 : 0, r.self ? 1 : 0, r.blocked ? 0 : 1];
}

/* ---------------- 标签块：333 维 ---------------- */

/* 每发子弹的最近接近时刻与距离（v2 geometry.mjs 的 closest 未导出，此处等价实现）。 */
function closest(segments, tank, grace = 0) {
  const tvx = Math.cos(tank.angle) * tank.speed, tvy = Math.sin(tank.angle) * tank.speed;
  let best = Infinity, when = 1.2;
  for (const s of segments) {
    const lo = Math.max(0, grace - s.time);
    if (lo > s.duration) continue;
    const rx = s.x - tank.x - tvx * s.time, ry = s.y - tank.y - tvy * s.time;
    const vx = s.vx - tvx, vy = s.vy - tvy;
    const t = Math.max(lo, Math.min(s.duration, -(rx * vx + ry * vy) / (vx * vx + vy * vy + 1e-9)));
    const d = Math.hypot(rx + vx * t, ry + vy * t);
    if (d < best) { best = d; when = s.time + t; }
  }
  return { distance: Number.isFinite(best) ? best : Math.hypot(1280, 800), time: when };
}

export function labels(tracker, trueAction) {
  const game = tracker.game, seat = tracker.seat, cfg = game.__cfg;
  const me = game.tanks[seat];
  const out = new Float32Array(S.labels);

  /* --- 教师动作 3（原始指令，不是派生量） --- */
  out[0] = trueAction ? trueAction.drive : 0;
  out[1] = trueAction ? trueAction.steer : 0;
  out[2] = trueAction && trueAction.shoot ? 1 : 0;

  if (!me.alive) {
    /* 死亡帧：servo 标记为无解，其余标签置零 —— 训练时按 alive 掩码 */
    out[S.L_SERVO + 2] = 0;
    return out;
  }

  /* --- 精确伺服目标 4 --- */
  const cands = aimCandidates(game, me);
  const srv = servoTarget(game, me, cands);
  if (srv.has_safe) {
    out[S.L_SERVO] = Math.sin(srv.err);
    out[S.L_SERVO + 1] = Math.cos(srv.err);
    out[S.L_SERVO + 2] = 1;
    out[S.L_SERVO + 3] = Math.min(1, srv.cone / Math.PI);
  }

  /* --- 固定角度基瞄准真值 64×3 --- */
  const life = cfg.BULLET.life;
  for (let i = 0; i < S.L_AIM_N; i++) {
    const phi = me.angle + i * (2 * Math.PI / S.L_AIM_N);
    const r = traceShotV8(game, me, phi);
    const base = S.L_AIM + i * S.L_AIM_STRIDE;
    out[base] = r.hit ? 1 : 0;
    out[base + 1] = r.self ? 1 : 0;
    out[base + 2] = Math.min(1, (r.blocked ? 0 : r.time) / life);
  }

  /* --- 子弹威胁 40×3 --- */
  const live = game.bullets.filter(b => !b.dead);
  let bs = 0;
  const diagonal = Math.hypot(cfg.W, cfg.H);
  for (const b of live) {
    if (bs >= S.THREAT_N) break;
    const segments = flightV8(game, b.x, b.y, b.angle, b.speed,
                              Math.min(1.2, b.life), b.bounces);
    const d = closest(segments, me, b.owner === me.index ? b.grace : 0);
    const base = S.L_THREAT + bs * S.L_THREAT_STRIDE;
    out[base] = Math.min(1, d.time / 1.2);
    out[base + 1] = Math.min(1, d.distance / diagonal);
    out[base + 2] = +(d.distance < me.radius + b.radius);
    bs++;
  }

  /* --- 对手意图 7×2：相对方位角速度 + 是否在跟踪我 --- */
  const prevB = tracker.prevBearing;
  let slot = 0;
  for (const t of game.tanks) {
    if (t.index === seat) continue;
    if (slot >= S.L_INTENT_N) break;
    const base = S.L_INTENT + slot * S.L_INTENT_STRIDE;
    const bearing = Math.atan2(t.y - me.y, t.x - me.x);
    const range = Math.hypot(t.x - me.x, t.y - me.y);
    const p = prevB ? prevB[t.index] : null;
    const rate = p ? wrap(bearing - p.bearing) / DT : 0;   // rad/s
    out[base] = Math.max(-1, Math.min(1, rate / Math.PI));
    out[base + 1] = p && Math.abs(rate) < 0.35 && range < p.range ? 1 : 0;
    slot++;
  }
  return out;
}
