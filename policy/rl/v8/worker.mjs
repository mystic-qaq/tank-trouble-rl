/* V8 collection worker: records Schema V3 input + label blocks for the hy and gpt seats.
   ===========================================================
   与 v3/worker.mjs 的差别：单遍采集。v3 需要两遍是因为它先跑完再看谁赢/谁击杀多，
   才知道该录哪些席位；V8 固定录 hy 与 gpt 两席，所以边跑边录即可，引擎工作量减半。

   每帧顺序（顺序不可交换）：
     observe(tracker)        用上一帧快照做差分，得到 608 维输入
     labels(tracker, action) 用 step 前的状态生成 333 维标签
     snapshotTracker(track)  把当前状态存成下一帧的差分基准
     step(action)
     recordAction(tracker)   存下本帧实际执行的指令

   packed 布局：总帧数 × 941（608 输入 + 333 标签）+ 346（地图，每世界一份）。
   每个轨迹的 offset 以「行」计数。
*/
import { TankEnvV2 } from '../v2/env.mjs';
import { skillReference } from '../v2/skills.mjs';
import { SCHEMA_V3, createTracker, snapshotTracker, recordAction, observe, labels, mapVector, threatInfo, shotInfo } from './observe_v3.mjs';

const S = SCHEMA_V3;
const STRIDE = S.frame + S.labels;      // 941
const RECORD_SEATS = ['hy', 'gpt'];
const idle = () => ({ drive: 0, steer: 0, shoot: false });

let buffer = Buffer.alloc(0), queue = Promise.resolve();
const slots = new Map();

function send(meta, data = new Float32Array()) {
  const json = Buffer.from(JSON.stringify(meta));
  const bytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  const head = Buffer.alloc(8);
  head.writeUInt32LE(json.length); head.writeUInt32LE(bytes.length, 4);
  process.stdout.write(Buffer.concat([head, json, bytes]));
}

function packWorld(game, recorded, rows, header) {
  const total = rows.reduce((a, r) => a + r.length, 0);
  const packed = new Float32Array(total * STRIDE + S.map);
  let row = 0;
  const trajectories = [];
  recorded.forEach((seat, i) => {
    const list = rows[i];
    for (const rec of list) {
      packed.set(rec.input, row * STRIDE);
      packed.set(rec.label, row * STRIDE + S.frame);
      row++;
    }
    trajectories.push({ seat, teacher: game.tanks[seat] ? header.players[seat] : 'unknown',
                        length: list.length, offset: (row - list.length) });
  });
  packed.set(mapVector(game), total * STRIDE);
  return { packed, trajectories, total, mapOffset: total * STRIDE };
}

async function collectWorld(config) {
  const e = new TankEnvV2({ objective: 'frag' });
  e.skipFeatures = true;
  await e.reset(config);

  const original = e.policyAction.bind(e);
  const recorded = [];
  e.game.tanks.forEach((t, i) => { if (RECORD_SEATS.includes(config.players[i])) recorded.push(i); });
  if (!recorded.length) throw new Error('World has no hy/gpt seat: ' + JSON.stringify(config.players));

  const trackers = recorded.map(seat => createTracker(e.game, seat));
  const rows = recorded.map(() => []);
  let frames = 0;

  while (!e.done) {
    const actions = e.game.tanks.map(t => t.alive ? original(config.players[t.index], t) : idle());

    // 1. observe + 2. labels，都用 step 前的世界状态
    trackers.forEach((tr, k) => {
      if (!e.game.tanks[tr.seat].alive) return;
      const a = actions[tr.seat];
      rows[k].push({ input: observe(tr), label: labels(tr, a) });
    });
    // 3. 差分基准 = 本帧观测时刻的状态
    trackers.forEach(snapshotTracker);
    // 4. 推进世界
    e.step(actions[0]);
    // 5. 记录本帧实际指令
    trackers.forEach((tr, k) => recordAction(tr, actions[tr.seat]));
    frames++;
  }

  const s = e.summary();
  const { packed, trajectories, total, mapOffset } = packWorld(e.game, recorded, rows, config);
  const worldScores = recorded.map(seat => ({
    seat, teacher: config.players[seat],
    win: +(e.game.roundWinner === seat),
    kills: e.game.kills[seat],
    net: e.ledger ? e.ledger.counts[seat].hits - e.ledger.counts[seat].self_hits : undefined,
  }));
  return send({ ok: true, config, summary: { ...s, frames },
                trajectories: trajectories.map((t, i) => ({ ...t, ...worldScores[i] })),
                mapOffset, mapSize: S.map, stride: STRIDE, schema: S.version,
                world_scores: worldScores }, packed);
}

async function collectSkill(config) {
  const e = new TankEnvV2({ objective: 'frag' });
  e.skipFeatures = true;
  await e.reset(config);
  const tr = createTracker(e.game, e.seat);
  const rows = [];
  while (!e.done) {
    const me = e.game.tanks[e.seat];
    if (!me.alive) break;
    const a = skillReference(e);
    rows.push({ input: observe(tr), label: labels(tr, a) });
    snapshotTracker(tr);
    e.step(a);
    recordAction(tr, a);
  }
  const s = e.summary();
  const total = rows.length;
  const packed = new Float32Array(total * STRIDE + S.map);
  rows.forEach((rec, i) => {
    packed.set(rec.input, i * STRIDE);
    packed.set(rec.label, i * STRIDE + S.frame);
  });
  packed.set(mapVector(e.game), total * STRIDE);
  return send({ ok: true, config, summary: { ...s, frames: total },
                trajectories: [{ seat: e.seat, teacher: 'reference', length: total, offset: 0,
                                 win: s.win, kills: e.kills, net: e.kills - e.suicides, skill: config.skill }],
                mapOffset: total * STRIDE, mapSize: S.map, stride: STRIDE, schema: S.version,
                world_scores: [] }, packed);
}

/* ---- 评测模式：策略驱动自机席位，其余席位由各自的教师策略驱动 ----
   open 返回 [map 346 | obs_v3 608]；act 返回下一帧 obs_v3（在 step 之前观测，
   与训练时 observe→snapshot→step→recordAction 的顺序一致）。 */
const evalSlots = new Map();

async function evalOpen(config, seat, slot, early = false) {
  const e = new TankEnvV2({ objective: config.objective || 'frag', early_death: early,
                            frozen_bots: !!config.frozen_bots,
                            time_limit: Number(config.time_limit) || 0 });
  e.skipFeatures = true;
  await e.reset(config);
  const tr = createTracker(e.game, seat);
  evalSlots.set(slot, { e, tr });
  const packed = new Float32Array(S.map + S.frame);
  packed.set(mapVector(e.game), 0);
  packed.set(observe(tr), S.map);
  return send({ ok: true, mapOffset: 0, obsOffset: S.map }, packed);
}

/* ---- RL 模式：与 eval 相同的观测/步进顺序，额外返回本帧可开火标志与自机事件 ----
   can_fire 在 step 前计算：引擎在 Tank.update 里先 cooldown -= dt 再判 shoot，
   所以「本帧能开火」= cooldown - dt <= 0 且存活弹数 < 上限。 */
const RL_EVENTS = new Set(['emission', 'hit', 'self_hit', 'despawn', 'fizzle']);

async function rlOpen(config, seat, slot) {
  return evalOpen(config, seat, slot, !!config.early_death);
}

async function rlAct(slot, action) {
  const { e, tr } = evalSlots.get(slot);
  if (!e) throw new Error('rl_act on missing slot ' + slot);
  const me = e.game.tanks[tr.seat];
  const obs = observe(tr);
  let live = 0;
  for (const b of e.game.bullets) if (!b.dead && b.owner === me.index) live++;
  const canFire = me.alive && me.cooldown - 1 / 60 <= 0 &&
    live < e.game.__cfg.MAX_BULLETS_PER_TANK;
  const alive = me.alive;
  const threat = threatInfo(e.game, tr.seat);
  const shot = shotInfo(e.game, tr.seat);
  snapshotTracker(tr);
  const r = e.step(action);
  recordAction(tr, action);
  const events = (r.events || []).filter(ev =>
    (ev.owner === tr.seat && RL_EVENTS.has(ev.kind)) ||
    (ev.kind === 'hit' && ev.victim === tr.seat));
  return send({ ok: true, done: e.done, active: r.active, alive: +me.alive,
                can_fire: +canFire, events, threat, shot,
                info: e.done ? { ...e.summary(), active_frames: alive ? 1 : 0 } : null }, obs);
}

async function evalAct(slot, action) {
  const { e, tr } = evalSlots.get(slot);
  const obs = observe(tr);
  snapshotTracker(tr);
  const before = { ...e.game.tanks[tr.seat] };
  const r = e.step(action);
  recordAction(tr, action);
  const done = e.done;
  return send({ ok: true, done, active: r.active,
                info: done ? { ...e.summary(),
                               active_frames: before.alive ? 1 : 0 } : null }, obs);
}

async function handle(m) {
  if (m.op === 'collect_world') return collectWorld(m.config);
  if (m.op === 'collect_skill') return collectSkill(m.config);
  if (m.op === 'eval_open') return evalOpen(m.config, m.seat, m.slot);
  if (m.op === 'eval_act') return evalAct(m.slot, m.action);
  if (m.op === 'eval_close') { evalSlots.delete(m.slot); return send({ ok: true }); }
  if (m.op === 'rl_open') return rlOpen(m.config, m.seat, m.slot);
  if (m.op === 'rl_act') return rlAct(m.slot, m.action);
  if (m.op === 'rl_close') { evalSlots.delete(m.slot); return send({ ok: true }); }
  if (m.op === 'reset') {
    const e = new TankEnvV2(m.options);
    slots.set(m.slot, e);
    await e.reset(m.config);
    return send({ ok: true }, new Float32Array(S.frame));
  }
  if (m.op === 'step') {
    const e = slots.get(m.slot);
    const r = e.step(m.action);
    return send({ ok: true, done: e.done, info: e.done ? e.summary() : null }, new Float32Array(S.frame));
  }
  throw new Error('Unknown V8 operation ' + m.op);
}

process.stdin.on('data', chunk => {
  buffer = Buffer.concat([buffer, chunk]);
  while (buffer.length >= 8) {
    const n = buffer.readUInt32LE(), m = buffer.readUInt32LE(4);
    if (buffer.length < 8 + n + m) break;
    const message = JSON.parse(buffer.subarray(8, 8 + n));
    buffer = buffer.subarray(8 + n + m);
    queue = queue.then(() => handle(message)).catch(error => send({ ok: false, error: error.stack }));
  }
});
