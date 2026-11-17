/* ===========================================================
   ui.js — 菜单、设置、右侧击杀榜、结算
   =========================================================== */

(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.prototype.slice.call(document.querySelectorAll(sel));

  const KEY_LABEL = {
    Space: '空格', Enter: 'Enter', ShiftRight: 'R-Shift', Slash: '/',
    ArrowUp: '↑', ArrowDown: '↓', ArrowLeft: '←', ArrowRight: '→',
    Numpad0: 'Num0', Numpad4: 'Num4', Numpad5: 'Num5',
    Numpad6: 'Num6', Numpad8: 'Num8', NumpadEnter: 'NumEnter',
    NumpadDecimal: 'Num.',
  };

  function keyLabel(code) {
    if (KEY_LABEL[code]) return KEY_LABEL[code];
    if (code.indexOf('Key') === 0) return code.slice(3);
    if (code.indexOf('Digit') === 0) return code.slice(5);
    return code;
  }

  function kbd(code) {
    const arrow = code.indexOf('Arrow') === 0 ? ' class="arrow"' : '';
    return '<kbd' + arrow + '>' + keyLabel(code) + '</kbd>';
  }

  const PRESETS = {
    '2': ['human', 'human', 'off', 'off', 'off', 'off', 'off', 'off'],
    '4': ['human', 'human', 'bot', 'bot', 'off', 'off', 'off', 'off'],
    '6': ['human', 'human', 'bot', 'bot', 'bot', 'bot', 'off', 'off'],
    '8': ['human', 'human', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'],
    'solo': ['human', 'bot', 'bot', 'bot', 'bot', 'off', 'off', 'off'],
    'arena': ['bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot', 'bot'],
  };

  const UI = TT.UI = {
    game: null,
    screen: 'menu',
    prevScreen: 'menu',
    slots: PRESETS['4'].slice(),
    roundsToWin: 5,
    bannerTimer: 0,

    init(game) {
      this.game = game;
      this.bannerEl = $('#banner');
      this.overlayEl = $('#overlay');
      this.keyhintsEl = $('#keyhints');
      this.pauseBtn = $('#btn-pause');
      this.boardEl = $('#board');
      this.boardListEl = $('#board-list');
      this.stageEl = $('#stage');

      game.hooks.banner = (text, color, ms) => this.showBanner(text, color, ms);
      game.hooks.hud = () => this.updateBoard();
      game.hooks.result = (w, s, k, sl) => this.showResult(w, s, k, sl);
      game.hooks.pause = (on) => {
        if (on) this.showScreen('pause');
        else if (this.screen === 'pause') this.hideOverlay();
      };

      this.buildSlots();
      this.buildControls();
      this.bind();
      this.applySlotUI();
      this.updateRoundsSeg();

      this.showScreen('menu');
      this.updateChrome(false);
      this.renderKeyHints('menu');
      this.restoreSound();
      this.layout();

      window.addEventListener('resize', () => this.layout());
    },

    /* ---------------- 布局 ---------------- */

    layout() {
      const main = $('#main');
      if (!main) return;
      const stage = this.stageEl;
      const board = this.boardEl;
      const stacked = window.innerWidth < 1020;

      let w, h;
      if (stacked) {
        w = Math.max(280, main.clientWidth);
        h = Math.round(w * 800 / 1280);
      } else {
        const bw = board.offsetWidth || 268;
        const gap = 14;
        const availW = main.clientWidth - bw - gap;
        const availH = main.clientHeight;
        const s = Math.min(availW / 1280, availH / 800);
        w = Math.round(1280 * s);
        h = Math.round(800 * s);
      }

      stage.style.width = w + 'px';
      stage.style.height = h + 'px';
      this.game.resize();
    },

    /* ---------------- 屏幕切换 ---------------- */

    showScreen(name) {
      this.screen = name;
      this.overlayEl.classList.remove('hidden');
      $$('.screen').forEach((el) => {
        el.classList.toggle('active', el.dataset.screen === name);
      });
    },

    hideOverlay() {
      this.screen = null;
      this.overlayEl.classList.add('hidden');
      const el = document.activeElement;
      if (el && el.blur) el.blur();
    },

    updateChrome(inMatch) {
      this.pauseBtn.hidden = !inMatch;
    },

    /* ---------------- 横幅 ---------------- */

    showBanner(text, color, ms) {
      const el = this.bannerEl;
      const c = color || '#ffffff';
      const light = (c === '#ffffff' || c === '#fff' || c === 'white');
      el.style.setProperty('--glow', light ? 'rgba(20,20,20,.20)' : c);
      el.innerHTML = '<span>' + text + '</span>';
      el.classList.remove('hide');
      el.classList.remove('show');
      void el.offsetWidth;
      el.classList.add('show');

      clearTimeout(this.bannerTimer);
      this.bannerTimer = setTimeout(() => {
        el.classList.remove('show');
        el.classList.add('hide');
      }, ms || 1400);
    },

    /* ---------------- 设置界面 ---------------- */

    buildSlots() {
      const wrap = $('#slot-list');
      const keyHint = ['W A S D', '方向键'];
      let html = '';
      for (let i = 0; i < 8; i++) {
        const c = TT.COLORS[i];
        const humanAllowed = i < 2;
        html +=
          '<div class="slot" data-slot="' + i + '" style="--c:' + c.main + '">' +
            '<div class="slot-dot">' + (i + 1) + '</div>' +
            '<div class="slot-name">' + c.name +
              '<small>' + (humanAllowed ? keyHint[i] : 'AI 对手') + '</small>' +
            '</div>' +
            '<div class="segmented" data-seg="' + i + '">' +
              (humanAllowed ? '<button data-mode="human">人类</button>' : '') +
              '<button data-mode="bot">电脑</button>' +
              '<button data-mode="off">关闭</button>' +
            '</div>' +
          '</div>';
      }
      wrap.innerHTML = html;
    },

    applySlotUI() {
      const slots = $$('.slot');
      for (let i = 0; i < slots.length; i++) {
        const mode = this.slots[i];
        slots[i].classList.toggle('active', mode !== 'off');
        slots[i].classList.toggle('off', mode === 'off');
        const seg = slots[i].querySelector('.segmented');
        Array.prototype.forEach.call(seg.children, (b) => {
          b.classList.toggle('active', b.dataset.mode === mode);
        });
      }

      const sig = this.slots.join(',');
      let matched = '';
      for (const key in PRESETS) {
        if (PRESETS[key].join(',') === sig) { matched = key; break; }
      }
      $$('[data-preset]').forEach((el) => {
        el.classList.toggle('active', el.dataset.preset === matched);
      });

      this.validate();
    },

    validate() {
      let active = 0, humans = 0;
      for (let i = 0; i < 8; i++) {
        if (this.slots[i] !== 'off') active++;
        if (this.slots[i] === 'human') humans++;
      }
      /* 0 个真人 = 纯 AI 观战（Arena），也允许开局 */
      const ok = active >= 2 && humans <= 2;
      const btn = $('[data-action="start-match"]');
      if (btn) {
        btn.disabled = !ok;
        btn.style.opacity = ok ? '' : '.45';
        btn.style.cursor = ok ? '' : 'not-allowed';
        btn.textContent = humans === 0 ? '开始观战' : '进入战场';
      }
      return ok;
    },

    updateRoundsSeg() {
      $$('#rounds-seg button').forEach((b) => {
        b.classList.toggle('active', Number(b.dataset.rounds) === this.roundsToWin);
      });
    },

    /* ---------------- 操作说明 ---------------- */

    buildControls() {
      const wrap = $('#controls-grid');
      let html = '';
      for (let i = 0; i < 2; i++) {
        const c = TT.COLORS[i];
        const b = TT.Input.bindings[i];
        html +=
          '<div class="ctrl-card" style="--c:' + c.main + '">' +
            '<h4><i></i>' + c.name + '</h4>' +
            '<div class="ctrl-line"><span>前进 / 倒车</span><span class="keys">' +
              kbd(b.up[0]) + kbd(b.down[0]) + '</span></div>' +
            '<div class="ctrl-line"><span>左转 / 右转</span><span class="keys">' +
              kbd(b.left[0]) + kbd(b.right[0]) + '</span></div>' +
            '<div class="ctrl-line"><span>开火</span><span class="keys">' +
              kbd(b.fire[0]) + '</span></div>' +
          '</div>';
      }
      wrap.innerHTML = html;
    },

    /* ---------------- 底部按键提示 ---------------- */

    renderKeyHints(mode) {
      if (mode !== 'match') { this.keyhintsEl.innerHTML = ''; return; }
      let html = '';
      let humans = 0;
      for (let i = 0; i < 8; i++) {
        if (this.game.slots[i] !== 'human') continue;
        humans++;
        const c = TT.COLORS[i];
        const b = TT.Input.bindings[i];
        html +=
          '<span class="hint" style="--c:' + c.main + '"><i></i>' + c.name +
            '<span class="keys">' + kbd(b.up[0]) + kbd(b.down[0]) +
            kbd(b.left[0]) + kbd(b.right[0]) + kbd(b.fire[0]) + '</span></span>';
      }
      if (!humans) {
        html = '<span class="hint spectator"><i></i>观战模式 · 电脑对战中' +
          '<span class="keys"><kbd>Esc</kbd>暂停</span></span>';
      }
      this.keyhintsEl.innerHTML = html;
    },

    /* ---------------- 右侧击杀榜 ---------------- */

    updateBoard() {
      const g = this.game;
      const active = [];
      for (let i = 0; i < 8; i++) if (g.slots[i] !== 'off') active.push(i);

      $('#board-round').textContent = g.round;
      $('#board-goal').innerHTML = g.mode === 'match'
        ? '先到 <b>' + g.roundsToWin + '</b> 胜 · 剩余 <b>' + Math.max(0, Math.ceil(TT.CFG.ROUND_TIME_LIMIT - g.roundTime)) + '</b>s'
        : '演示对局';

      const rows = active.map((i) => {
        const tank = g.tanks.filter((t) => t.index === i)[0];
        return {
          i: i,
          kills: g.kills[i],
          wins: g.scores[i],
          bot: g.slots[i] === 'bot',
          alive: tank ? tank.alive : true,
        };
      });

      rows.sort((a, b) => {
        if (b.kills !== a.kills) return b.kills - a.kills;
        if (b.wins !== a.wins) return b.wins - a.wins;
        return a.i - b.i;
      });

      let html = '';
      for (let k = 0; k < rows.length; k++) {
        const r = rows[k];
        const c = TT.COLORS[r.i];
        const cls = 'board-row' + (r.alive ? '' : ' dead');
        html +=
          '<div class="' + cls + '" style="--c:' + c.main + '">' +
            '<span class="b-rank">' + (k + 1) + '</span>' +
            '<span class="b-dot"></span>' +
            '<span class="b-name">' + c.name +
              (r.bot ? '<em>BOT</em>' : '') +
            '</span>' +
            '<span class="b-kill">' + r.kills + '</span>' +
            '<span class="b-win">' + r.wins + '</span>' +
          '</div>';
      }
      this.boardListEl.innerHTML = html;
      this.boardEl.classList.toggle('dim', this.screen !== null && this.screen !== 'result');
    },

    /* ---------------- 结算 ---------------- */

    showResult(winner, scores, kills, slots) {
      const crown = $('#result-crown');
      const title = $('#result-title');
      const sub = $('#result-sub');
      const table = $('#result-table');

      if (winner >= 0) {
        const c = TT.COLORS[winner];
        crown.textContent = '👑';
        crown.style.filter = 'drop-shadow(0 6px 12px ' + U.rgba(c.main, 0.4) + ')';
        title.textContent = c.name + ' 获胜！';
        title.style.color = c.main;
      } else {
        crown.textContent = '💥';
        crown.style.filter = '';
        title.textContent = '平局';
        title.style.color = '#141414';
      }
      sub.textContent = '先到 ' + this.game.roundsToWin + ' 胜';

      const list = [];
      for (let i = 0; i < 8; i++) {
        if (slots[i] === 'off') continue;
        list.push({ i: i, wins: scores[i], kills: kills[i], bot: slots[i] === 'bot' });
      }
      list.sort((a, b) => (b.wins - a.wins) || (b.kills - a.kills) || (a.i - b.i));

      let html = '<div class="rt-row rt-head">' +
        '<span class="rt-rank">#</span>' +
        '<span class="rt-name">坦克</span>' +
        '<span class="rt-win">胜场</span>' +
        '<span class="rt-kill">击杀</span>' +
      '</div>';
      for (let k = 0; k < list.length; k++) {
        const r = list[k];
        const c = TT.COLORS[r.i];
        const cls = 'rt-row' + (r.i === winner ? ' winner' : '');
        html +=
          '<div class="' + cls + '" style="--c:' + c.main + '">' +
            '<span class="rt-rank">' + (k + 1) + '</span>' +
            '<span class="rt-name"><i></i>' + c.name + (r.bot ? '<em>BOT</em>' : '') + '</span>' +
            '<span class="rt-win">' + r.wins + '</span>' +
            '<span class="rt-kill">' + r.kills + '</span>' +
          '</div>';
      }
      table.innerHTML = html;

      this.keyhintsEl.innerHTML = '';
      this.updateChrome(false);
      this.showScreen('result');
      this.updateBoard();
    },

    /* ---------------- 声音 ---------------- */

    restoreSound() {
      let muted = false;
      try { muted = localStorage.getItem('tt_muted') === '1'; } catch (e) { /* ignore */ }
      TT.Audio.muted = muted;
      this.syncSoundBtn();
    },

    syncSoundBtn() {
      const btn = $('#btn-sound');
      btn.textContent = TT.Audio.muted ? '🔇' : '🔊';
      btn.classList.toggle('muted', TT.Audio.muted);
    },

    toggleSound() {
      TT.Audio.init();
      TT.Audio.setMuted(!TT.Audio.muted);
      try { localStorage.setItem('tt_muted', TT.Audio.muted ? '1' : '0'); } catch (e) { /* ignore */ }
      this.syncSoundBtn();
    },

    /* ---------------- 事件 ---------------- */

    bind() {
      document.addEventListener('click', (e) => {
        const btn = e.target.closest('button');
        if (!btn) return;

        TT.Audio.init();
        TT.Audio.resume();

        if (btn.dataset.preset) {
          TT.Audio.click();
          const p = PRESETS[btn.dataset.preset];
          if (p) this.slots = p.slice();
          this.applySlotUI();
          return;
        }

        if (btn.dataset.mode && btn.parentElement.dataset.seg != null) {
          TT.Audio.click();
          const i = Number(btn.parentElement.dataset.seg);
          this.slots[i] = btn.dataset.mode;
          this.applySlotUI();
          return;
        }

        if (btn.dataset.rounds) {
          TT.Audio.click();
          this.roundsToWin = Number(btn.dataset.rounds);
          this.updateRoundsSeg();
          return;
        }

        const action = btn.dataset.action;
        if (!action) return;

        switch (action) {
          case 'open-setup':
            TT.Audio.click();
            this.prevScreen = 'menu';
            this.showScreen('setup');
            this.updateBoard();
            break;

          case 'start-arena':
            TT.Audio.click();
            this.slots = PRESETS['arena'].slice();
            this.applySlotUI();
            this.hideOverlay();
            this.game.startMatch(this.slots.slice(), this.roundsToWin);
            this.updateChrome(true);
            this.renderKeyHints('match');
            this.updateBoard();
            break;

          case 'open-controls':
            TT.Audio.click();
            this.prevScreen = this.screen || 'menu';
            this.showScreen('controls');
            break;

          case 'back-menu':
            TT.Audio.click();
            if (this.prevScreen === 'pause') this.showScreen('pause');
            else this.showScreen('menu');
            this.updateBoard();
            break;

          case 'start-match':
            if (!this.validate()) return;
            TT.Audio.click();
            this.hideOverlay();
            this.game.startMatch(this.slots.slice(), this.roundsToWin);
            this.updateChrome(true);
            this.renderKeyHints('match');
            this.updateBoard();
            break;

          case 'resume':
            TT.Audio.click();
            this.game.resume();
            break;

          case 'restart-match':
            TT.Audio.click();
            this.game.restartMatch();
            this.renderKeyHints('match');
            this.updateBoard();
            break;

          case 'quit-match':
            TT.Audio.click();
            this.game.quitToMenu();
            this.showScreen('menu');
            this.updateChrome(false);
            this.renderKeyHints('menu');
            this.updateBoard();
            break;

          case 'rematch':
            TT.Audio.click();
            this.hideOverlay();
            this.game.startMatch(this.game.slots.slice(), this.game.roundsToWin);
            this.updateChrome(true);
            this.renderKeyHints('match');
            this.updateBoard();
            break;

          default:
            break;
        }
      });

      $('#btn-pause').addEventListener('click', () => {
        TT.Audio.click();
        this.game.pause();
      });

      $('#btn-help').addEventListener('click', () => {
        TT.Audio.click();
        if (this.game.mode === 'match' && !this.game.paused && this.screen === null) {
          this.game.pause();
          this.prevScreen = 'pause';
          this.showScreen('controls');
        } else {
          this.prevScreen = this.screen || 'menu';
          this.showScreen('controls');
        }
      });

      $('#btn-sound').addEventListener('click', () => this.toggleSound());

      $('#btn-full').addEventListener('click', () => {
        TT.Audio.click();
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else if (document.documentElement.requestFullscreen) {
          document.documentElement.requestFullscreen().catch(() => {});
        }
      });

      TT.Input.on('down', (code) => {
        if (code === 'Escape') {
          if (this.game.mode === 'match' && this.screen === null) this.game.pause();
          else if (this.screen === 'pause') this.game.resume();
          else if (this.screen === 'controls') this.showScreen(this.prevScreen || 'menu');
        }
        if (code === 'KeyM') this.toggleSound();
      });
    },
  };
})();
