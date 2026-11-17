/* ===========================================================
   audio.js — 纯 WebAudio 合成音效（无外部素材）
   =========================================================== */

(function () {
  const A = TT.Audio = {
    ctx: null,
    master: null,
    noiseBuf: null,
    ready: false,
    muted: false,
    volume: 0.55,

    init() {
      if (this.ctx) { this.resume(); return; }
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      try { this.ctx = new AC(); } catch (e) { return; }

      const comp = this.ctx.createDynamicsCompressor();
      comp.threshold.value = -14;
      comp.knee.value = 22;
      comp.ratio.value = 10;
      comp.attack.value = 0.003;
      comp.release.value = 0.22;
      comp.connect(this.ctx.destination);

      this.master = this.ctx.createGain();
      this.master.gain.value = this.muted ? 0 : this.volume;
      this.master.connect(comp);

      const len = Math.floor(this.ctx.sampleRate * 0.9);
      const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const data = buf.getChannelData(0);
      for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
      this.noiseBuf = buf;
      this.ready = true;
    },

    resume() {
      if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
    },

    setMuted(v) {
      this.muted = !!v;
      if (this.master) {
        const t = this.ctx.currentTime;
        this.master.gain.cancelScheduledValues(t);
        this.master.gain.setTargetAtTime(this.muted ? 0 : this.volume, t, 0.02);
      }
    },

    get t() { return this.ctx ? this.ctx.currentTime : 0; },

    /* 通用：一个带包络的振荡器 */
    tone(opt) {
      if (!this.ready || this.muted) return;
      const c = this.ctx, t0 = c.currentTime + (opt.delay || 0);
      const osc = c.createOscillator();
      const g = c.createGain();
      osc.type = opt.type || 'sine';
      osc.frequency.setValueAtTime(opt.f0, t0);
      if (opt.f1 && opt.f1 !== opt.f0) {
        osc.frequency.exponentialRampToValueAtTime(Math.max(20, opt.f1), t0 + opt.dur);
      }
      const peak = opt.gain == null ? 0.2 : opt.gain;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(peak, t0 + (opt.attack || 0.006));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + opt.dur);

      let node = osc;
      if (opt.filter) {
        const f = c.createBiquadFilter();
        f.type = opt.filter;
        f.frequency.setValueAtTime(opt.fc0 || 2000, t0);
        if (opt.fc1) f.frequency.exponentialRampToValueAtTime(Math.max(40, opt.fc1), t0 + opt.dur);
        f.Q.value = opt.q || 1;
        node.connect(f);
        node = f;
      }
      node.connect(g);
      g.connect(this.master);
      osc.start(t0);
      osc.stop(t0 + opt.dur + 0.03);
    },

    /* 噪声爆发 */
    noise(opt) {
      if (!this.ready || this.muted || !this.noiseBuf) return;
      const c = this.ctx, t0 = c.currentTime + (opt.delay || 0);
      const src = c.createBufferSource();
      src.buffer = this.noiseBuf;
      src.loop = true;
      src.playbackRate.value = opt.rate || 1;

      const f = c.createBiquadFilter();
      f.type = opt.filter || 'lowpass';
      f.frequency.setValueAtTime(opt.fc0 || 1600, t0);
      if (opt.fc1) f.frequency.exponentialRampToValueAtTime(Math.max(40, opt.fc1), t0 + opt.dur);
      f.Q.value = opt.q || 0.8;

      const g = c.createGain();
      const peak = opt.gain == null ? 0.3 : opt.gain;
      g.gain.setValueAtTime(0.0001, t0);
      g.gain.exponentialRampToValueAtTime(peak, t0 + (opt.attack || 0.005));
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + opt.dur);

      src.connect(f); f.connect(g); g.connect(this.master);
      src.start(t0);
      src.stop(t0 + opt.dur + 0.05);
    },

    shoot() {
      this.tone({ type: 'sawtooth', f0: 340, f1: 70, dur: 0.13, gain: 0.16, filter: 'lowpass', fc0: 2600, fc1: 500 });
      this.noise({ dur: 0.075, gain: 0.13, fc0: 3200, fc1: 700, filter: 'bandpass', q: 1.1 });
    },

    bounce(power) {
      const p = U.clamp(power == null ? 1 : power, 0.2, 1);
      this.tone({ type: 'triangle', f0: 1250 * (0.8 + p * 0.4), f1: 720, dur: 0.055, gain: 0.035 + 0.03 * p });
    },

    explode() {
      this.noise({ dur: 0.62, gain: 0.5, fc0: 1800, fc1: 70, filter: 'lowpass', q: 0.7 });
      this.tone({ type: 'sine', f0: 150, f1: 36, dur: 0.45, gain: 0.4 });
      this.noise({ dur: 0.12, gain: 0.28, fc0: 5200, fc1: 900, filter: 'highpass' });
    },

    beep(freq, dur, gain, delay) {
      this.tone({ type: 'square', f0: freq, f1: freq, dur: dur || 0.09, gain: gain || 0.06, delay: delay || 0, filter: 'lowpass', fc0: 3200 });
    },

    roundStart() {
      this.beep(392, 0.1, 0.05, 0);
      this.beep(587, 0.14, 0.055, 0.11);
    },

    win() {
      const notes = [523, 659, 784, 1046];
      for (let i = 0; i < notes.length; i++) {
        this.tone({ type: 'triangle', f0: notes[i], f1: notes[i], dur: 0.19, gain: 0.09, delay: i * 0.085 });
      }
    },

    click() {
      this.tone({ type: 'square', f0: 720, f1: 520, dur: 0.045, gain: 0.028, filter: 'lowpass', fc0: 2600 });
    },
  };
})();
