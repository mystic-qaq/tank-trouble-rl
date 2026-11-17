/* ===========================================================
   entities.js — 坦克、子弹、特效（原版扁平卡通风格）
   =========================================================== */

(function () {
  const CFG = TT.CFG;
  const U = TT.util;
  const P = TT.PALETTE;
  const TAU = U.TAU;
  const INK = P.ink;

  /* =========================================================
     坦克
     ========================================================= */

  TT.Tank = class Tank {
    constructor(index, spawn, palette, isBot) {
      this.index = index;
      this.x = spawn.x;
      this.y = spawn.y;
      this.angle = spawn.a;
      this.palette = palette;
      this.color = palette.main;
      this.isBot = !!isBot;

      this.radius = CFG.TANK.radius;
      this.speed = 0;
      this.cooldown = 0;
      this.alive = true;
      this.recoil = 0;
      this.lastShotAge = 99;
      this.ai = null;
      this.name = palette.name;
    }

    barrelTip() {
      return {
        x: this.x + Math.cos(this.angle) * (this.radius + 12),
        y: this.y + Math.sin(this.angle) * (this.radius + 12),
      };
    }

    update(dt, input, game) {
      if (!this.alive) return;

      this.cooldown = Math.max(0, this.cooldown - dt);
      this.recoil = Math.max(0, this.recoil - dt * 6);
      this.lastShotAge += dt;

      const T = CFG.TANK;

      if (input.steer) {
        this.angle += input.steer * T.turn * dt;
        if (this.angle > Math.PI) this.angle -= TAU;
        if (this.angle < -Math.PI) this.angle += TAU;
      }

      const max = T.speed * (input.drive < 0 ? T.reverseRatio : 1);
      if (input.drive !== 0) {
        const target = input.drive > 0 ? max : -max;
        const rate = (this.speed * input.drive < 0) ? T.brake : T.accel;
        if (this.speed < target) this.speed = Math.min(target, this.speed + rate * dt);
        else if (this.speed > target) this.speed = Math.max(target, this.speed - rate * dt);
      } else {
        const drop = T.brake * 0.75 * dt;
        if (Math.abs(this.speed) <= drop) this.speed = 0;
        else this.speed -= Math.sign(this.speed) * drop;
      }

      const nx = this.x + Math.cos(this.angle) * this.speed * dt;
      const ny = this.y + Math.sin(this.angle) * this.speed * dt;

      this.x = nx;
      this.y = ny;
      const bumped = game.map.resolveCircle(this, this.radius);

      if (bumped && Math.abs(this.speed) > 60) {
        this.speed *= 0.55;
        if (Math.abs(this.speed) > 100) game.shake(1.4);
      }

      if (input.shoot && this.cooldown <= 0) this.fire(game);
    }

    fire(game) {
      let live = 0;
      for (let i = 0; i < game.bullets.length; i++) {
        const b = game.bullets[i];
        if (!b.dead && b.owner === this.index) live++;
      }
      if (live >= CFG.MAX_BULLETS_PER_TANK) return;

      this.cooldown = CFG.TANK.fireCooldown;
      this.recoil = 1;
      this.lastShotAge = 0;
      this.speed -= 16;

      const bx = this.x + Math.cos(this.angle) * (this.radius + 12);
      const by = this.y + Math.sin(this.angle) * (this.radius + 12);
      /* 炮管比车体长，炮口可能已经伸到薄墙另一侧。检查整段炮管。 */
      const blocked = game.map.raycast(this.x, this.y, bx, by, CFG.BULLET.radius);
      if (blocked) {
        game.fx.fizzle(this.x + (bx - this.x) * blocked.t, this.y + (by - this.y) * blocked.t);
        return;
      }
      game.bullets.push(new TT.Bullet(bx, by, this.angle, this.index, this.color));
      game.fx.muzzle(bx, by, this.angle);
      game.shake(2.8);
      TT.Audio.shoot();
    }

    draw(ctx) {
      if (!this.alive) return;
      const p = this.palette;

      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(this.angle);
      ctx.lineJoin = 'round';

      const back = this.recoil * 3.6;

      /* 炮管（先画，炮塔压在上面） */
      ctx.fillStyle = p.main;
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2.6;
      U.rrect(ctx, 3 - back, -3.8, 24, 7.6, 1.2);
      ctx.fill();
      ctx.stroke();

      /* 车体 */
      ctx.fillStyle = p.body;
      U.rrect(ctx, -17, -11.8, 34, 23.6, 2.8);
      ctx.fill();
      ctx.stroke();

      /* 履带 */
      ctx.fillStyle = p.tread;
      ctx.lineWidth = 2.2;
      U.rrect(ctx, -17.5, -12.4, 35, 4.5, 1.2);
      ctx.fill();
      ctx.stroke();
      U.rrect(ctx, -17.5, 7.9, 35, 4.5, 1.2);
      ctx.fill();
      ctx.stroke();

      /* 炮塔 */
      ctx.fillStyle = p.main;
      ctx.lineWidth = 2.6;
      ctx.beginPath();
      ctx.arc(0, 0, 8.3, 0, TAU);
      ctx.fill();
      ctx.stroke();

      /* 炮塔上的小高光 */
      ctx.fillStyle = 'rgba(255,255,255,.28)';
      ctx.beginPath();
      ctx.arc(-2.6, -3, 2.6, 0, TAU);
      ctx.fill();

      ctx.restore();
    }
  };

  /* =========================================================
     子弹
     ========================================================= */

  TT.Bullet = class Bullet {
    constructor(x, y, angle, owner, color) {
      this.x = x; this.y = y;
      this.px = x; this.py = y;
      this.angle = angle;
      this.speed = CFG.BULLET.speed;
      this.owner = owner;
      this.color = color;
      this.bounces = 0;
      this.life = CFG.BULLET.life;
      this.radius = CFG.BULLET.radius;
      this.dead = false;
      this.grace = CFG.BULLET.spawnGrace;
      this.trail = [];
    }

    update(dt, game) {
      this.life -= dt;
      this.grace = Math.max(0, this.grace - dt);
      if (this.life <= 0) { this.dead = true; return; }

      this.px = this.x;
      this.py = this.y;
      this.trail.push({ x: this.x, y: this.y });
      if (this.trail.length > 9) this.trail.shift();

      let remaining = this.speed * dt;
      let guard = 0;

      while (remaining > 0.02 && !this.dead && guard++ < 8) {
        const dx = Math.cos(this.angle) * remaining;
        const dy = Math.sin(this.angle) * remaining;
        const hit = game.map.raycast(this.x, this.y, this.x + dx, this.y + dy, this.radius);

        /* 先命中的物体才生效；反弹后的每一段分别计算，不能用首尾连线。 */
        let victim = null;
        let tankT = Infinity;
        if (game.state === 'playing') {
          for (const tank of game.tanks) {
            if (!tank.alive || (tank.index === this.owner && this.grace > 0)) continue;
            const t = U.segCircle(this.x, this.y, this.x + dx, this.y + dy,
              tank.x, tank.y, tank.radius + this.radius);
            if (t >= 0 && t < tankT && (!hit || t < hit.t)) {
              victim = tank;
              tankT = t;
            }
          }
        }
        if (victim) {
          this.x += dx * tankT;
          this.y += dy * tankT;
          game.killTank(victim, this);
          this.dead = true;
          break;
        }

        if (!hit) {
          this.x += dx;
          this.y += dy;
          break;
        }

        const t = U.clamp(hit.t, 0, 1);
        this.x += dx * t;
        this.y += dy * t;
        remaining -= remaining * t;

        const nx = hit.nx || 0, ny = hit.ny || 0;
        if (nx === 0 && ny === 0) {
          /* 炮口顶在墙里：子弹直接哑火 */
          this.dead = true;
          game.fx.fizzle(this.x, this.y);
          break;
        }

        let vx = Math.cos(this.angle), vy = Math.sin(this.angle);
        const dot = vx * nx + vy * ny;
        if (dot < 0) { vx -= 2 * dot * nx; vy -= 2 * dot * ny; }
        this.angle = Math.atan2(vy, vx);

        this.x += nx * 0.8;
        this.y += ny * 0.8;

        this.bounces++;
        if (this.bounces > CFG.BULLET.bounces) {
          this.dead = true;
          game.fx.fizzle(this.x, this.y);
        } else {
          game.fx.bounceSpark(this.x, this.y, nx, ny);
          TT.Audio.bounce(1);
          game.shake(1);
        }
        remaining -= 0.7;
      }
    }

    draw(ctx) {
      /* 原版子弹：一串淡淡的小圆点 + 实心黑点 */
      for (let i = 0; i < this.trail.length; i++) {
        const p = this.trail[i];
        const k = (i + 1) / this.trail.length;
        ctx.fillStyle = 'rgba(40,40,40,' + (k * 0.16) + ')';
        ctx.beginPath();
        ctx.arc(p.x, p.y, this.radius * (0.35 + k * 0.5), 0, TAU);
        ctx.fill();
      }

      ctx.fillStyle = P.bullet;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, TAU);
      ctx.fill();
    }
  };

  /* =========================================================
     特效
     ========================================================= */

  TT.FX = class FX {
    constructor(game) {
      this.game = game;
      this.parts = [];
      this.rings = [];
      this.flashes = [];
      this.smoke = [];
      this.shakePower = 0;
      this.shakeX = 0;
      this.shakeY = 0;
    }

    clear() {
      this.parts.length = 0;
      this.rings.length = 0;
      this.flashes.length = 0;
      this.smoke.length = 0;
      this.shakePower = 0;
    }

    shake(power) {
      if (power > this.shakePower) this.shakePower = Math.min(16, power);
    }

    puff(x, y, vx, vy, life, r, grow, dark) {
      this.smoke.push({
        x, y, vx, vy, life, max: life, r, grow: grow, dark: dark || 0,
      });
    }

    muzzle(x, y, angle) {
      for (let i = 0; i < 4; i++) {
        const a = angle + (Math.random() - 0.5) * 1.1;
        const sp = 60 + Math.random() * 130;
        this.parts.push({
          x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
          life: 0.12, max: 0.12, size: 2 + Math.random() * 2,
          color: '#2a2a2a', drag: 3,
        });
      }
      this.flashes.push({ x, y, r: 5, max: 16, life: 0.07, t: 0.07 });
    }

    bounceSpark(x, y, nx, ny) {
      const a0 = Math.atan2(ny, nx);
      for (let i = 0; i < 3; i++) {
        const a = a0 + (Math.random() - 0.5) * 2;
        const sp = 30 + Math.random() * 70;
        this.parts.push({
          x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
          life: 0.12, max: 0.12, size: 1.6 + Math.random() * 1.4,
          color: '#2a2a2a', drag: 3.4,
        });
      }
    }

    fizzle(x, y) {
      this.puff(x, y, 0, 0, 0.28, 3, 14, 0.3);
    }

    explosion(x, y, palette) {
      /* 烟团：原版爆炸就是一大团灰烟 */
      for (let i = 0; i < 17; i++) {
        const a = Math.random() * TAU;
        const sp = 12 + Math.random() * 95;
        const r = 9 + Math.random() * 11;
        this.puff(
          x + (Math.random() - 0.5) * 16,
          y + (Math.random() - 0.5) * 16,
          Math.cos(a) * sp, Math.sin(a) * sp,
          0.6 + Math.random() * 0.6,
          r, 42 + Math.random() * 34,
          0.55 + Math.random() * 0.4
        );
      }

      /* 几块深色碎片 */
      for (let i = 0; i < 9; i++) {
        const a = Math.random() * TAU;
        const sp = 90 + Math.random() * 220;
        this.parts.push({
          x, y, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp,
          life: 0.34 + Math.random() * 0.3, max: 0.64,
          size: 2 + Math.random() * 3,
          color: i % 3 === 0 ? palette.body : '#1e1e1e',
          drag: 2.4, rot: Math.random() * TAU, spin: (Math.random() - 0.5) * 14, shard: true,
        });
      }

      this.flashes.push({ x, y, r: 12, max: 62, life: 0.16, t: 0.16 });
      this.rings.push({ x, y, r: 8, max: 74, life: 0.32, t: 0.32, width: 3.5 });
      this.shake(13);
    }

    update(dt) {
      for (let i = this.parts.length - 1; i >= 0; i--) {
        const p = this.parts[i];
        p.life -= dt;
        if (p.life <= 0) { this.parts.splice(i, 1); continue; }
        const d = Math.exp(-p.drag * dt);
        p.vx *= d; p.vy *= d;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        if (p.shard) p.rot += p.spin * dt;
      }

      for (let i = this.rings.length - 1; i >= 0; i--) {
        const r = this.rings[i];
        r.t -= dt;
        if (r.t <= 0) { this.rings.splice(i, 1); continue; }
        const k = 1 - r.t / r.life;
        r.r = r.max * (1 - Math.pow(1 - k, 2.4));
      }

      for (let i = this.flashes.length - 1; i >= 0; i--) {
        const f = this.flashes[i];
        f.t -= dt;
        if (f.t <= 0) { this.flashes.splice(i, 1); continue; }
        f.r = f.max * (1 - Math.pow(1 - (1 - f.t / f.life), 2));
      }

      for (let i = this.smoke.length - 1; i >= 0; i--) {
        const s = this.smoke[i];
        s.life -= dt;
        if (s.life <= 0) { this.smoke.splice(i, 1); continue; }
        const d = Math.exp(-1.9 * dt);
        s.vx *= d; s.vy *= d;
        s.x += s.vx * dt;
        s.y += s.vy * dt;
        s.r += s.grow * dt;
      }

      if (this.shakePower > 0.05) {
        this.shakePower *= Math.pow(0.0016, dt);
        const a = Math.random() * TAU;
        this.shakeX = Math.cos(a) * this.shakePower;
        this.shakeY = Math.sin(a) * this.shakePower;
      } else {
        this.shakePower = 0;
        this.shakeX = 0;
        this.shakeY = 0;
      }
    }

    drawSmoke(ctx) {
      for (let i = 0; i < this.smoke.length; i++) {
        const s = this.smoke[i];
        const k = s.life / s.max;
        const a = Math.min(0.62, k * 0.56) * s.dark;
        const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r);
        g.addColorStop(0, 'rgba(48,48,48,' + a + ')');
        g.addColorStop(0.55, 'rgba(62,62,62,' + (a * 0.62) + ')');
        g.addColorStop(1, 'rgba(78,78,78,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, TAU);
        ctx.fill();
      }
    }

    draw(ctx) {
      for (let i = 0; i < this.rings.length; i++) {
        const r = this.rings[i];
        const a = Math.max(0, r.t / r.life);
        ctx.strokeStyle = 'rgba(40,40,40,' + (a * 0.35) + ')';
        ctx.lineWidth = r.width * a + 0.4;
        ctx.beginPath();
        ctx.arc(r.x, r.y, r.r, 0, TAU);
        ctx.stroke();
      }

      for (let i = 0; i < this.flashes.length; i++) {
        const f = this.flashes[i];
        const a = Math.max(0, f.t / f.life);
        const g = ctx.createRadialGradient(f.x, f.y, 0, f.x, f.y, f.r);
        g.addColorStop(0, 'rgba(255,255,255,' + (a * 0.9) + ')');
        g.addColorStop(0.55, 'rgba(235,235,235,' + (a * 0.4) + ')');
        g.addColorStop(1, 'rgba(220,220,220,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(f.x, f.y, f.r, 0, TAU);
        ctx.fill();
      }

      for (let i = 0; i < this.parts.length; i++) {
        const p = this.parts[i];
        const k = p.life / p.max;
        if (p.shard) {
          ctx.save();
          ctx.translate(p.x, p.y);
          ctx.rotate(p.rot);
          ctx.fillStyle = U.rgba(p.color === '#1e1e1e' ? '#1e1e1e' : p.color, Math.min(1, k * 1.8));
          ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.8);
          ctx.restore();
        } else {
          ctx.fillStyle = U.rgba('#2a2a2a', Math.min(1, k * 1.6));
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * (0.4 + k * 0.6), 0, TAU);
          ctx.fill();
        }
      }
    }
  };
})();
