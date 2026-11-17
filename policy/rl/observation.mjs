export const SCHEMA = Object.freeze({ version:1, self:16, map:346, enemies:7, bullets:40, entity:12, size:926 });

export function observe(game, seat, previous = { drive:0, steer:0, shoot:false }) {
  const { W, H, TANK:T, BULLET:B, ROUND_TIME_LIMIT:limit } = game.__cfg;
  const me = game.tanks.find(t => t.index === seat);
  const bullets = game.bullets.filter(b => !b.dead);
  const ammo = Array(8).fill(0);
  bullets.forEach(b => ammo[b.owner]++);
  const out = new Float32Array(SCHEMA.size);
  out.set([me.x/W, me.y/H, Math.cos(me.angle), Math.sin(me.angle), me.speed/T.speed,
    me.cooldown/T.fireCooldown, +me.alive, ammo[seat]/5, game.tanks.length/8,
    game.tanks.filter(t => t.alive).length/8, Math.max(0, 1-game.roundTime/limit),
    previous.drive, previous.steer, +previous.shoot, me.radius/80, Math.min(me.lastShotAge,6)/6]);
  let at = SCHEMA.self;
  out.set(game.map.vFlag,at); at += game.map.vFlag.length;
  out.set(game.map.hFlag,at); at += game.map.hFlag.length;
  for (const t of game.tanks.filter(t => t.index !== seat)) {
    out.set([(t.x-me.x)/W, (t.y-me.y)/H, Math.cos(t.angle), Math.sin(t.angle),
      t.speed/T.speed, t.cooldown/T.fireCooldown, +t.alive, ammo[t.index]/5,
      t.x/W,t.y/H,t.radius/80,1],at);
    at += SCHEMA.entity;
  }
  at = SCHEMA.self + SCHEMA.map + SCHEMA.enemies * SCHEMA.entity;
  if (bullets.length > SCHEMA.bullets) throw new Error('Observation bullet capacity exceeded');
  for (const b of bullets) {
    out.set([(b.x-me.x)/W,(b.y-me.y)/H,Math.cos(b.angle)*b.speed/B.speed,
      Math.sin(b.angle)*b.speed/B.speed,+(b.owner===seat),b.life/B.life,b.bounces/B.bounces,
      b.grace/B.spawnGrace,b.x/W,b.y/H,b.radius/80,1],at);
    at += SCHEMA.entity;
  }
  return out;
}
