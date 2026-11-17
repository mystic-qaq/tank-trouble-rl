/* ===========================================================
   input.js — 键盘输入与按键绑定
   =========================================================== */

(function () {
  const Input = TT.Input = {
    down: Object.create(null),
    listeners: [],

    bindings: [
      {
        up: ['KeyW'], down: ['KeyS'], left: ['KeyA'], right: ['KeyD'],
        fire: ['Space', 'KeyF'],
      },
      {
        up: ['ArrowUp'], down: ['ArrowDown'], left: ['ArrowLeft'], right: ['ArrowRight'],
        fire: ['Enter', 'ShiftRight', 'Slash', 'NumpadEnter'],
      },
      {
        up: ['KeyI'], down: ['KeyK'], left: ['KeyJ'], right: ['KeyL'],
        fire: ['KeyU', 'KeyO'],
      },
      {
        up: ['Numpad8'], down: ['Numpad5'], left: ['Numpad4'], right: ['Numpad6'],
        fire: ['Numpad0', 'NumpadDecimal'],
      },
    ],

    /* 需要阻止默认行为的按键（避免页面滚动 / 空格触发按钮） */
    blocked: new Set([
      'Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
      'Enter', 'Slash', 'NumpadEnter', 'Numpad0', 'NumpadDecimal',
      'Numpad4', 'Numpad5', 'Numpad6', 'Numpad8', 'ShiftRight',
    ]),

    isDown(code) { return !!this.down[code]; },

    anyDown(list) {
      for (let i = 0; i < list.length; i++) if (this.down[list[i]]) return true;
      return false;
    },

    get(index) {
      const b = this.bindings[index];
      if (!b) return { drive: 0, steer: 0, shoot: false };
      let drive = 0;
      if (this.anyDown(b.up)) drive += 1;
      if (this.anyDown(b.down)) drive -= 1;
      let steer = 0;
      if (this.anyDown(b.left)) steer -= 1;
      if (this.anyDown(b.right)) steer += 1;
      return { drive: drive, steer: steer, shoot: this.anyDown(b.fire) };
    },

    on(type, fn) { this.listeners.push([type, fn]); },

    emit(type, code) {
      for (let i = 0; i < this.listeners.length; i++) {
        if (this.listeners[i][0] === type) this.listeners[i][1](code);
      }
    },

    clear() { this.down = Object.create(null); },

    init() {
      window.addEventListener('keydown', (e) => {
        if (e.repeat) {
          if (this.blocked.has(e.code)) e.preventDefault();
          return;
        }
        const tag = (e.target && e.target.tagName) || '';
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;

        if (this.blocked.has(e.code)) e.preventDefault();
        this.down[e.code] = true;
        this.emit('down', e.code);
      }, { passive: false });

      window.addEventListener('keyup', (e) => {
        this.down[e.code] = false;
        this.emit('up', e.code);
      });

      window.addEventListener('blur', () => this.clear());
      document.addEventListener('visibilitychange', () => {
        if (document.hidden) this.clear();
      });
    },
  };
})();
