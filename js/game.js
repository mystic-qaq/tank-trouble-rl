/* ===========================================================
   game.js — 主循环、回合状态机、碰撞与渲染
   =========================================================== */

(function () {
  const CFG = TT.CFG;
  const U = TT.util;
  const TAU = U.TAU;

  const NOOP = function () {};

  TT.Game = class Game {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');

      this.mode = 'demo';           // 'demo' | 'match'
      this.state = 'idle';          // 'intro' | 'playing' | 'over' | 'matchover'
      this.tanks = [];
      this.bullets = [];
      this.slots = ['bot', 'bot', 'bot', 'bot', 'off', 'off', 'off', 'off'];
      this.scores = [0, 0, 0, 0, 0, 0, 0, 0];
      this.kills = [0, 0, 0, 0, 0, 0, 0, 0];
      this.round = 0;
      this.roundsToWin = 5;
      this.map = null;
      this.staticLayer = null;

      this.fx = new TT.FX(this);
      this.time = 0;
      this.stateTime = 0;
      this.last = 0;
      this.running = false;
      this.paused = false;
      this.dpr = 1;
      this.displayScale = 1;
      this.matchWinner = null;
      this.roundWinner = -1;
      this.roundTime = 0;
      this.roundTimedOut = false;
      this.roundSerial = 0;

      this.hooks = {
        banner: NOOP,
        hud: NOOP,
        result: NOOP,
        pause: NOOP,
        roundOver: NOOP,
      };

      this._tick = this._tick.bind(this);
      this.resize();
    }

    /* ---------------- 画布尺寸 ---------------- */

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round((rect.width || CFG.W) * dpr));
      const h = Math.max(1, Math.round((rect.height || CFG.H) * dpr));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
      }
      this.displayScale = this.canvas.width / CFG.W;
      this.dpr = dpr;
    }

    /* ---------------- 生命周期 ---------------- */

    start() {
      if (this.running) return;
      this.running = true;
      this.last = 0;
      requestAnimationFrame(this._tick);
    }

    _tick(ts) {
      if (!this.running) return;
      const now = ts / 1000;
      let dt = this.last ? now - this.last : 0;
      this.last = now;
      if (dt > 1 / 30) dt = 1 / 30;
      if (dt < 0) dt = 0;

      if (!this.paused) this.update(dt);
      this.render();

      requestAnimationFrame(this._tick);
    }

    /* ---------------- 模式 ---------------- */

    startDemo() {
      if (this.mode === 'demo' && this.tanks.length) return;
      this.mode = 'demo';
      this.slots = ['bot', 'bot', 'bot', 'bot', 'off', 'off', 'off', 'off'];
      this.scores = [0, 0, 0, 0, 0, 0, 0, 0];
      this.kills = [0, 0, 0, 0, 0, 0, 0, 0];
      this.matchWinner = null;
      this.round = 0;
      this.roundsToWin = 99;
      this.paused = false;
      this.startRound();
      this.hooks.hud();
    }

    startMatch(slots, roundsToWin) {
      this.mode = 'match';
      this.slots = slots.slice();
      this.scores = [0, 0, 0, 0, 0, 0, 0, 0];
      this.kills = [0, 0, 0, 0, 0, 0, 0, 0];
      this.round = 0;
      this.roundsToWin = roundsToWin || 5;
      this.matchWinner = null;
      this.paused = false;
      this.hooks.hud();
      this.startRound();
    }

    restartMatch() {
      this.startMatch(this.slots, this.roundsToWin);
      this.hooks.pause(false);
    }

    quitToMenu() {
      this.paused = false;
      this.hooks.pause(false);
      this.mode = 'demo';
      this.tanks = [];
      this.bullets = [];
      this.startDemo();
    }

    pause() {
      if (this.mode !== 'match' || this.paused) return;
      if (this.state === 'matchover') return;
      this.paused = true;
      TT.Input.clear();
      this.hooks.pause(true);
    }

    resume() {
      if (!this.paused) return;
      this.paused = false;
      this.last = 0;
      this.hooks.pause(false);
    }

    togglePause() {
      if (this.paused) this.resume();
      else this.pause();
    }

    /* ---------------- 回合 ---------------- */

    startRound() {
      this.round++;
      this.roundSerial++;
      this.roundWinner = -1;
      this.roundTime = 0;
      this.roundTimedOut = false;

      const seeds = CFG.MAP_SEEDS;
      const seed = seeds[(this.round - 1) % seeds.length];
      this.map = new TT.GameMap(seed);
      this.staticLayer = this.map.renderStatic(2);

      this.fx.clear();
      this.bullets = [];
      this.tanks = [];

      const active = [];
      for (let i = 0; i < 8; i++) {
        if (this.slots[i] !== 'off') active.push(i);
      }
      const spawns = this.map.pickSpawns(active.length, this.spawnRandom || Math.random);

      for (let k = 0; k < active.length; k++) {
        const i = active[k];
        const tank = new TT.Tank(i, spawns[k], TT.COLORS[i], this.slots[i] === 'bot');
        this.tanks.push(tank);
      }

      this.state = 'intro';
      this.stateTime = CFG.ROUND_INTRO;

      this.hooks.hud();
      if (this.mode === 'match') {
        this.hooks.banner('ROUND ' + this.round, '#ffffff', 950);
        TT.Audio.roundStart();
      }
    }

    endRound(winner, reason) {
      if (this.state !== 'playing') return;
      this.roundWinner = winner;
      this.roundTimedOut = this.roundTimedOut || reason === 'timeout';
      this.state = 'over';
      this.stateTime = this.mode === 'demo' ? 1.2 : CFG.ROUND_OUTRO;

      if (winner >= 0) this.scores[winner]++;

      if (this.mode === 'demo') return;

      this.hooks.hud();
      if (winner >= 0) {
        const c = TT.COLORS[winner];
        this.hooks.banner(c.name + ' 赢下本回合', c.main, 1700);
        TT.Audio.win();
      } else {
        this.hooks.banner(this.roundTimedOut ? CFG.ROUND_TIME_LIMIT + ' 秒超时 · 平局' : '同归于尽', '#ffffff', 1700);
        TT.Audio.win();
      }
    }

    finishMatch() {
      this.state = 'matchover';
      if (this.mode !== 'match') return;
      this.hooks.result(this.matchWinner, this.scores.slice(), this.kills.slice(), this.slots.slice());
    }

    /* ---------------- 更新 ---------------- */

    update(dt) {
      this.time += dt;

      switch (this.state) {
        case 'intro':
          this.stateTime -= dt;
          this.fx.update(dt);
          if (this.stateTime <= 0) {
            this.state = 'playing';
            this.stateTime = 0;
          }
          break;

        case 'playing':
          {
            const before = Math.floor(this.roundTime);
            const step = Math.min(dt, Math.max(0, CFG.ROUND_TIME_LIMIT - this.roundTime));
            this.roundTime += step;
            this.updateWorld(step);
            if (this.state === 'playing' && this.roundTime >= CFG.ROUND_TIME_LIMIT - 1e-8) {
              this.roundTime = CFG.ROUND_TIME_LIMIT;
              this.roundTimedOut = true;
              this.endRound(-1, 'timeout');
            }
            if (Math.floor(this.roundTime) !== before) this.hooks.hud();
          }
          break;

        case 'over':
          this.stateTime -= dt;
          this.updateBullets(dt);
          this.fx.update(dt);
          if (this.stateTime <= 0) {
            if (this.mode === 'demo') {
              this.startRound();
            } else if (this.roundWinner >= 0 && this.scores[this.roundWinner] >= this.roundsToWin) {
              this.matchWinner = this.roundWinner;
              this.finishMatch();
            } else {
              this.startRound();
            }
          }
          break;

        case 'matchover':
          this.fx.update(dt);
          break;

        default:
          break;
      }
    }

    updateWorld(dt) {
      const dtc = dt;

      /* 输入 → 坦克 */
      for (let i = 0; i < this.tanks.length; i++) {
        const tank = this.tanks[i];
        if (!tank.alive) continue;
        const input = tank.isBot
          ? TT.updateBot(tank, this, dtc)
          : TT.Input.get(tank.index);
        tank.update(dtc, input, this);
      }

      this.resolveTankCollisions();

      this.updateBullets(dt);

      /* 每段弹道在 Bullet.update 内按先后次序处理墙体和坦克碰撞。 */

      this.bullets = this.bullets.filter((b) => !b.dead);
      this.fx.update(dt);

      /* 回合结束判定 */
      let alive = 0, last = null;
      for (let i = 0; i < this.tanks.length; i++) {
        if (this.tanks[i].alive) { alive++; last = this.tanks[i]; }
      }
      if (alive <= 1) this.endRound(alive === 1 ? last.index : -1);
    }

    updateBullets(dt) {
      for (let i = 0; i < this.bullets.length; i++) {
        const b = this.bullets[i];
        if (!b.dead) b.update(dt, this);
      }
    }

    resolveTankCollisions() {
      for (let i = 0; i < this.tanks.length; i++) {
        for (let j = i + 1; j < this.tanks.length; j++) {
          const a = this.tanks[i], b = this.tanks[j];
          if (!a.alive || !b.alive) continue;
          const dx = b.x - a.x, dy = b.y - a.y;
          const minD = a.radius + b.radius;
          const d2 = dx * dx + dy * dy;
          if (d2 >= minD * minD || d2 < 1e-6) continue;
          const d = Math.sqrt(d2);
          const push = (minD - d) / 2;
          const nx = dx / d, ny = dy / d;
          a.x -= nx * push; a.y -= ny * push;
          b.x += nx * push; b.y += ny * push;
          this.map.resolveCircle(a, a.radius);
          this.map.resolveCircle(b, b.radius);
          a.speed *= 0.72;
          b.speed *= 0.72;
        }
      }
    }

    killTank(tank, bullet) {
      if (!tank.alive) return;
      tank.alive = false;

      if (bullet && bullet.owner != null) {
        if (bullet.owner === tank.index) this.kills[tank.index] -= 1;
        else this.kills[bullet.owner] += 1;
      }
      this.fx.explosion(tank.x, tank.y, tank.palette);
      this.shake(14);
      TT.Audio.explode();
      this.hooks.hud();
    }

    /* ---------------- 地面贴花 ---------------- */


    shake(power) { this.fx.shake(power); }

    /* ---------------- 渲染 ---------------- */

    render() {
      const ctx = this.ctx;
      const s = this.displayScale;
      ctx.setTransform(s, 0, 0, s, 0, 0);
      ctx.clearRect(0, 0, CFG.W, CFG.H);

      if (!this.map) return;

      ctx.save();
      ctx.translate(this.fx.shakeX, this.fx.shakeY);

      if (this.staticLayer) ctx.drawImage(this.staticLayer, 0, 0, CFG.W, CFG.H);

      this.fx.drawSmoke(ctx);

      for (let i = 0; i < this.tanks.length; i++) {
        this.tanks[i].draw(ctx, this.time);
      }

      for (let i = 0; i < this.bullets.length; i++) {
        this.bullets[i].draw(ctx);
      }

      this.fx.draw(ctx);

      ctx.restore();

      /* 边角极淡的暗角，让画面聚焦 */
      const vg = ctx.createRadialGradient(
        CFG.W / 2, CFG.H / 2, CFG.H * 0.45,
        CFG.W / 2, CFG.H / 2, CFG.W * 0.78
      );
      vg.addColorStop(0, 'rgba(16,24,40,0)');
      vg.addColorStop(1, 'rgba(16,24,40,.10)');
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, CFG.W, CFG.H);
    }
  };
})();
