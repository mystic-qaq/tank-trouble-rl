import vm from 'node:vm';
import { createArena, mulberry32, roundSeed, STRATEGY_IDS, MAP_SEEDS } from '../../arena/arena.mjs';
import { observe } from './observation.mjs';

const IDLE = Object.freeze({drive:0,steer:0,shoot:false});
export function normalizeAction(action) {
  const value = v => Number.isFinite(Number(v)) ? Number(v) : 0;
  return { drive:Math.sign(value(action.drive)), steer:Math.max(-1,Math.min(1,value(action.steer))), shoot:!!action.shoot };
}

export class TankEnv {
  constructor(options = {}) {
    this.options = { objective:'win', seed:20261106, min_n:2, max_n:8, ...options };
    if (!['win','frag'].includes(this.options.objective)) throw new Error('Unknown objective');
    this.episode = 0;
  }
  async reset(config = {}) {
    const seed = config.seed ?? (roundSeed(this.options.seed,this.episode++) & 0x3fffffff);
    const random = mulberry32(seed);
    const n = config.players?.length ?? config.n ?? (this.options.min_n + Math.floor(random()*(this.options.max_n-this.options.min_n+1)));
    if (!Number.isInteger(n) || n < 2 || n > 8) throw new Error('n must be 2..8');
    this.seat = config.seat ?? Math.floor(random()*n);
    if (!Number.isInteger(this.seat) || this.seat < 0 || this.seat >= n) throw new Error('Invalid learner seat');
    this.players = config.players?.slice() ?? Array.from({length:n},() => STRATEGY_IDS[Math.floor(random()*STRATEGY_IDS.length)]);
    if (this.players.some(id => !STRATEGY_IDS.includes(id))) throw new Error('Unknown opponent policy');
    this.seed = seed;
    this.mapSeed = config.mapSeed ?? MAP_SEEDS[Math.floor(random()*MAP_SEEDS.length)];
    this.spawnSeed = config.spawnSeed ?? roundSeed(seed,11);
    const env = this.env = await createArena();
    const { TT, sandbox, ctx } = env;
    sandbox.__rnd = mulberry32(roundSeed(seed,21));
    vm.runInContext('Math.random = function () { return __rnd(); };',ctx);
    this.randomStreams = Array.from({length:n},(_,i) => mulberry32(roundSeed(seed,30+i)));
    TT.CFG.MAP_SEEDS = [this.mapSeed];
    const game = this.game = new TT.Game(env.canvas);
    game.__cfg = TT.CFG;
    game.mode = 'match';
    game.slots = Array.from({length:8},(_,i) => i<n ? 'bot' : 'off');
    game.spawnRandom = TT.util.rng(this.spawnSeed);
    // Effects consume no training time or randomness; physics is the actual game implementation.
    game.fx = { clear(){}, update(){}, shake(){}, muzzle(){}, fizzle(){}, bounceSpark(){}, explosion(){} };
    game.startRound();
    game.state = 'playing';
    game.stateTime = 0;
    this.action = {...IDLE}; this.previous = {...IDLE};
    this.kills = 0; this.suicides = 0; this.return = 0; this.frames = 0; this.done = false;
    this.errors = 0;
    const originalKill = game.killTank.bind(game);
    game.killTank = (tank, bullet) => {
      if (tank.alive && bullet?.owner === this.seat) {
        if (tank.index === this.seat) this.suicides++;
        else this.kills++;
      }
      originalKill(tank,bullet);
    };
    TT.updateBot = (tank,g,dt) => tank.index === this.seat ? this.action
      : (this.options.frozen_bots ? IDLE : this.policyAction(this.players[tank.index],tank,dt));
    return this.observation();
  }
  policyAction(id, tank, dt = 1/60) {
    const previous = this.env.sandbox.__rnd;
    this.env.sandbox.__rnd = this.randomStreams[tank.index];
    try { return normalizeAction(this.env.adapters[id](tank,this.game,dt) || IDLE); }
    catch (e) { this.errors++; throw new Error('Policy '+id+': '+e.message,{cause:e}); }
    finally { this.env.sandbox.__rnd = previous; }
  }
  observation() { return observe(this.game,this.seat,this.previous); }
  step(action, teacher = null) {
    if (this.done) throw new Error('reset required after termination');
    const me = this.game.tanks[this.seat];
    const active = me.alive;
    this.action = active ? (teacher ? this.policyAction(teacher,me) : normalizeAction(action)) : {...IDLE};
    const before = this.kills-this.suicides;
    this.game.update(1/60);
    this.frames++;
    this.previous = this.action;
    this.done = this.game.state === 'over';
    // Training-only agent termination is exact once no future reward is possible.
    // Evaluation always finishes the whole world. Frag keeps all owned bullets alive.
    if (!this.done && this.options.early_death && !me.alive &&
        (this.options.objective === 'win' || !this.game.bullets.some(b => !b.dead && b.owner === this.seat))) {
      this.done = true;
    }
    // Curriculum time limit (e.g. L0 target range): terminate at N simulated seconds.
    if (!this.done && this.options.time_limit && this.frames >= this.options.time_limit * 60) {
      this.done = true;
    }
    const reward = this.options.objective === 'frag' ? this.kills-this.suicides-before
      : +(this.done && this.game.roundWinner === this.seat);
    this.return += reward;
    return { observation:this.observation(), reward, terminated:this.done, truncated:false,
      active:+active, action:this.action, info:this.done ? this.summary() : null };
  }
  summary() {
    return { seed:this.seed, spawnSeed:this.spawnSeed, mapSeed:this.mapSeed, n:this.players.length,
      seat:this.seat, opponents:this.players.slice(), frames:this.frames, duration:this.game.roundTime,
      win:+(this.game.roundWinner===this.seat), timeout:this.game.roundTimedOut, round_complete:this.game.state==='over',
      kills:this.kills, suicides:this.suicides, netKills:this.kills-this.suicides,
      return:this.return, errors:this.errors };
  }
}
