import assert from 'node:assert/strict';
import { TankEnv } from './env.mjs';
import { SCHEMA } from './observation.mjs';

const idle={drive:0,steer:0,shoot:false};
for (let n=2;n<=8;n++) {
  const env=new TankEnv({objective:'win'});
  const obs=await env.reset({n,seat:n-1,seed:100+n,players:Array(n).fill('builtin')});
  assert.equal(obs.length,SCHEMA.size);assert.ok(obs.every(Number.isFinite));
  assert.equal(obs[10],1);
  const enemy=Array.from({length:7},(_,i)=>obs[362+i*12+11]);
  assert.equal(enemy.reduce((a,b)=>a+b,0),n-1);
  env.env.TT.updateBot=()=>idle;
  for (let i=0;i<3599;i++) assert.equal(env.step(idle).terminated,false);
  const result=env.step(idle);
  assert.equal(result.terminated,true);assert.equal(result.truncated,false);
  assert.equal(result.info.timeout,true);assert.equal(result.info.duration,60);
  assert.equal(result.info.frames,3600);assert.equal(result.reward,0);
  assert.equal(result.info.return,0);assert.ok(env.game.scores.every(s=>s===0));
  assert.throws(()=>env.step(idle),/reset required/);
  await env.reset({n,seat:0,seed:100+n});assert.equal(env.game.roundTime,0);
}

// Same actual engine, seed and controls reproduce observations and rewards.
const a=new TankEnv({objective:'frag'}),b=new TankEnv({objective:'frag'});
const config={seed:424242,n:4,seat:1,players:['gpt','builtin','hy','glm']};
assert.deepEqual(await a.reset(config),await b.reset(config));
for(let i=0;i<180&&!a.done;i++) {
  const ra=a.step(idle,'gpt'),rb=b.step(idle,'gpt');
  assert.deepEqual(ra.observation,rb.observation);assert.equal(ra.reward,rb.reward);
}

// Death is not episode termination; an actual owned projectile can score afterwards.
const ghost=new TankEnv({objective:'frag'});
await ghost.reset({seed:7,n:4,seat:0,players:['builtin','builtin','builtin','builtin']});
ghost.env.TT.updateBot=()=>idle;
ghost.game.map.walls=[];
ghost.game.tanks.forEach((t,i)=>{t.x=100+100*i;t.y=100;});
ghost.game.killTank(ghost.game.tanks[0],{owner:1});
ghost.game.bullets.push(new ghost.env.TT.Bullet(180,100,0,0,'#fff'));
const posthumous=ghost.step(idle);
assert.equal(posthumous.active,0);assert.equal(posthumous.terminated,false);
assert.equal(posthumous.reward,1);assert.equal(ghost.kills,1);assert.equal(ghost.return,1);
assert.equal(ghost.game.kills[0],1);

// A natural win yields exactly one terminal point, whereas suicides subtract once.
const win=new TankEnv({objective:'win'});
await win.reset({seed:10,n:2,seat:0});win.env.TT.updateBot=()=>idle;
win.game.killTank(win.game.tanks[1],{owner:0});
const victory=win.step(idle);assert.equal(victory.reward,1);assert.equal(victory.info.win,1);
const suicide=new TankEnv({objective:'frag'});
await suicide.reset({seed:11,n:2,seat:0});suicide.env.TT.updateBot=()=>idle;
suicide.game.map.walls=[];
const me=suicide.game.tanks[0];const own=new suicide.env.TT.Bullet(me.x,me.y,0,0,'#fff');own.grace=0;
suicide.game.bullets.push(own);
const death=suicide.step(idle);assert.equal(death.reward,-1);assert.equal(death.info.netKills,-1);
assert.equal(death.info.return,-1);assert.equal(death.info.errors,0);

// The training shortcut only terminates when this agent's exact return is fixed.
const shortcut=new TankEnv({objective:'frag',early_death:true});
await shortcut.reset({seed:12,n:4,seat:0,players:Array(4).fill('builtin')});
shortcut.env.TT.updateBot=()=>idle;shortcut.game.map.walls=[];
shortcut.game.tanks.forEach((t,i)=>{t.x=100+100*i;t.y=100;});
shortcut.game.killTank(shortcut.game.tanks[0],{owner:1});
shortcut.game.bullets.push(new shortcut.env.TT.Bullet(160,100,0,0,'#fff'));
assert.equal(shortcut.step(idle).terminated,false);
assert.equal(shortcut.step(idle).terminated,false);
const delayed=shortcut.step(idle);
assert.equal(delayed.reward,1);assert.equal(delayed.terminated,true);
assert.equal(delayed.info.round_complete,false);assert.equal(delayed.info.netKills,1);
const deadWinner=new TankEnv({objective:'win',early_death:true});
await deadWinner.reset({seed:13,n:4,seat:0});deadWinner.env.TT.updateBot=()=>idle;
deadWinner.game.killTank(deadWinner.game.tanks[0],{owner:1});
assert.equal(deadWinner.step(idle).terminated,true);
const lastTick=new TankEnv({objective:'win'});
await lastTick.reset({seed:14,n:2,seat:0});lastTick.env.TT.updateBot=()=>idle;
lastTick.game.map.walls=[];
lastTick.game.tanks[0].x=100;lastTick.game.tanks[0].y=100;
lastTick.game.tanks[1].x=200;lastTick.game.tanks[1].y=100;
lastTick.game.roundTime=60-1/60;
lastTick.game.bullets.push(new lastTick.env.TT.Bullet(180,100,0,0,'#fff'));
const atDeadline=lastTick.step(idle);
assert.equal(atDeadline.reward,1);assert.equal(atDeadline.info.timeout,false);
assert.equal(atDeadline.info.duration,60);
console.log('✓ 60s exact termination, 2–8 agents, duplicate policies, masks, deterministic replay, posthumous kills, win and net-kill rewards');
