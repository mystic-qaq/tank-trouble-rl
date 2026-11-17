/* hardbot.mjs — 假想强敌：内置 bot 的「认真版」
   用途：只靠打赢内置 bot 不足以说明策略强。这个对手做到：
   - 完美提前量（用真实速度矢量，不抖）
   - 主动追近到 ~180px 再打（近距离命中率高）
   - 有子弹飞来就垂直于弹道侧移
   - 用满 5 发在场子弹，冷却一好就扣扳机
   它仍是简单规则型 AI，但明显强于游戏自带的 bot。
   只在 sim/ 下，不参与 Arena 提交。 */

export function makeHardBot(opts = {}) {
  const P = Object.assign({ closeTo: 180, maxLive: 5, fireArc: 0.05, dodge: true, lead: 1.0 }, opts);
  const mem = new WeakMap();

  function state(tank) {
    let s = mem.get(tank);
    if (!s) {
      s = { jit: 0, stuck: 0, lastX: tank.x, lastY: tank.y, lastT: 0, dir: 1, hist: {}, reverse: 0, path: null, pathT: 0 };
      mem.set(tank, s);
    }
    return s;
  }

  return function update(tank, game, dt) {
    const r = { bs: 620, tr: tank.radius, br: 3.2, turn: 3.3, vmax: 178 };
    const s = state(tank);
    s.jit += (Math.random() - 0.5) * 0.02;
    s.jit = Math.max(-0.06, Math.min(0.06, s.jit));
    const inp = { drive: 0, steer: 0, shoot: false };

    /* 目标：最近的活人 */
    let tgt = null, bd = Infinity;
    for (const o of game.tanks) {
      if (o === tank || !o.alive) continue;
      const d = (o.x - tank.x) ** 2 + (o.y - tank.y) ** 2;
      if (d < bd) { bd = d; tgt = o; }
    }
    if (!tgt) return inp;
    const dist = Math.sqrt(bd);

    /* 速度记忆 */
    let h = s.hist[tgt.index];
    if (!h) { h = s.hist[tgt.index] = { x: tgt.x, y: tgt.y, t: 0, vx: 0, vy: 0, lt: 0, lx: tgt.x, ly: tgt.y }; }
    h.t += dt;
    const ddt = h.t - h.lt;
    if (ddt > 0.1) {
      h.vx = (tgt.x - h.lx) / ddt; h.vy = (tgt.y - h.ly) / ddt;
      h.lt = h.t; h.lx = tgt.x; h.ly = tgt.y;
    }

    /* 侧移：有子弹冲我来就垂直让 */
    if (P.dodge) {
      let worst = null, wd = 1e9;
      for (const b of game.bullets) {
        if (b.dead || b.owner === tank.index) continue;
        const vx = Math.cos(b.angle), vy = Math.sin(b.angle);
        const rx = tank.x - b.x, ry = tank.y - b.y;
        const along = rx * vx + ry * vy;
        if (along < 0 || along > 420) continue;
        const miss = Math.abs(rx * vy - ry * vx);
        if (miss < 60 && along < wd) { wd = along; worst = { vx, vy, miss, along }; }
      }
      if (worst && worst.along / r.bs < 0.7) {
        const side = (tank.x - worst.vx * worst.along) * worst.vy - (tank.y - worst.vy * worst.along) * worst.vx >= 0 ? 1 : -1;
        const want = Math.atan2(worst.vy, worst.vx) + side * Math.PI / 2;
        const diff = Math.atan2(Math.sin(want - tank.angle), Math.cos(want - tank.angle));
        inp.steer = Math.abs(diff) > 0.06 ? (diff < 0 ? -1 : 1) : 0;
        inp.drive = 1;
        /* 顺手回一枪 */
        if (tank.cooldown <= 0 && game.map.hasLOS(tank.x, tank.y, tgt.x, tgt.y)) {
          const tt = dist / r.bs;
          const aa = Math.atan2(tgt.y + h.vy * tt * P.lead - tank.y, tgt.x + h.vx * tt * P.lead - tank.x);
          if (Math.abs(Math.atan2(Math.sin(aa - tank.angle), Math.cos(aa - tank.angle))) < 0.12) inp.shoot = true;
        }
        return inp;
      }
    }

    const los = game.map.hasLOS(tank.x, tank.y, tgt.x, tgt.y);

    if (los) {
      /* 完美提前量 */
      let tt = dist / r.bs, aa = Math.atan2(tgt.y - tank.y, tgt.x - tank.x);
      for (let i = 0; i < 3; i++) {
        const px = tgt.x + h.vx * tt, py = tgt.y + h.vy * tt;
        aa = Math.atan2(py - tank.y, px - tank.x);
        tt = Math.hypot(px - tank.x, py - tank.y) / r.bs;
      }
      aa += s.jit;
      const diff = Math.atan2(Math.sin(aa - tank.angle), Math.cos(aa - tank.angle));
      if (Math.abs(diff) > 0.02) inp.steer = diff < 0 ? -1 : 1;

      /* 追近到理想距离 */
      const front = game.map.solidAt(tank.x + Math.cos(tank.angle) * (r.tr + 18), tank.y + Math.sin(tank.angle) * (r.tr + 18));
      const back = game.map.solidAt(tank.x - Math.cos(tank.angle) * (r.tr + 18), tank.y - Math.sin(tank.angle) * (r.tr + 18));
      if (dist > P.closeTo) inp.drive = front ? 0 : 1;
      else if (dist < P.closeTo * 0.7) inp.drive = back ? 0 : -0.8;
      else inp.drive = 0.3;

      let live = 0;
      for (const b of game.bullets) if (!b.dead && b.owner === tank.index) live++;
      if (live < P.maxLive && Math.abs(diff) < P.fireArc && tank.cooldown <= 0) {
        const tipx = tank.x + Math.cos(tank.angle) * (r.tr + 14);
        const tipy = tank.y + Math.sin(tank.angle) * (r.tr + 14);
        if (!game.map.solidAt(tipx, tipy)) inp.shoot = true;
      }
    } else {
      /* 追击：BFS 路径 */
      s.pathT -= dt;
      if (s.pathT <= 0 || !s.path) {
        s.path = game.map.findPath(tank.x, tank.y, tgt.x, tgt.y);
        s.pathT = 0.4;
      }
      let wp = tgt;
      if (s.path && s.path.length) {
        while (s.path.length > 1 && Math.hypot(s.path[0].x - tank.x, s.path[0].y - tank.y) < 40) s.path.shift();
        wp = s.path[Math.min(s.path.length - 1, 1)];
      }
      const want = Math.atan2(wp.y - tank.y, wp.x - tank.x);
      const diff = Math.atan2(Math.sin(want - tank.angle), Math.cos(want - tank.angle));
      if (Math.abs(diff) > 0.04) inp.steer = diff < 0 ? -1 : 1;
      inp.drive = Math.abs(diff) > 2.0 ? -0.9 : (Math.abs(diff) > 0.8 ? 0.5 : 1);
      if (inp.drive > 0 && game.map.solidAt(tank.x + Math.cos(tank.angle) * (r.tr + 16), tank.y + Math.sin(tank.angle) * (r.tr + 16))) {
        inp.drive = 0.2; inp.steer = s.dir;
      }
    }

    /* 脱困 */
    if (s.lastT === 0) s.lastT = 1;
    s.lastT += dt;
    if (s.lastT > 0.5) {
      if (Math.hypot(tank.x - s.lastX, tank.y - s.lastY) < 10) { s.stuck += 0.5; s.dir = Math.random() < 0.5 ? -1 : 1; }
      else s.stuck = 0;
      s.lastX = tank.x; s.lastY = tank.y; s.lastT = 0;
      if (s.stuck > 1.0) { s.stuck = 0; s.reverse = 0.5; s.path = null; }
    }
    if (s.reverse > 0) { s.reverse -= dt; inp.drive = -1; inp.steer = s.dir; inp.shoot = false; }
    return inp;
  };
}
