import assert from 'node:assert/strict';
import { TankEnv } from '../env.mjs';
import { TankEnvV2 } from './env.mjs';
import { SCHEMA,DT } from './geometry.mjs';
import { SKILLS } from './skills.mjs';
const idle={drive:0,steer:0,shoot:false};
const state=e=>JSON.stringify({time:e.game.roundTime,tanks:e.game.tanks.map(t=>[t.x,t.y,t.angle,t.speed,t.cooldown,t.alive]),
  bullets:e.game.bullets.map(b=>[b.x,b.y,b.angle,b.bounces,b.life,b.owner,b.dead]),scores:e.game.scores,kills:e.game.kills});
const old=new TankEnv({objective:'frag'}),next=new TankEnvV2({objective:'frag'});
const config={seed:9123,n:4,seat:1,players:['hy','builtin','gpt','builtin']};
await old.reset(config);const observation=await next.reset(config);
assert.equal(observation.length,SCHEMA.size);assert.ok(observation.every(Number.isFinite));
for(let i=0;i<240&&!old.done;i++){
  const a=old.step(idle,'gpt'),b=next.step(idle,'gpt');
  assert.equal(state(old),state(next),'Instrumentation or features changed engine behavior');
  assert.equal(a.reward,b.reward);
}
assert.equal(next.ledger.counts[next.seat].hits,next.kills);
assert.equal(next.ledger.counts[next.seat].self_hits,next.suicides);

// A request during cooldown, at the exact ready boundary, against a wall, and at the ammo limit.
const e=new TankEnvV2();await e.reset({seed:32,n:2,seat:0,skill:'muzzle'});
const me=e.game.tanks[0];me.cooldown=DT+1e-5;e.invalidate();
assert.equal(e.observation()[16],0);
let result=e.step({...idle,shoot:true});assert.equal(e.ledger.counts[0].cooldown,1);
assert.equal(e.ledger.counts[0].emissions,0);
me.cooldown=DT;e.invalidate();assert.equal(e.observation()[16],1);
result=e.step({...idle,shoot:true});assert.equal(e.ledger.counts[0].fizzles,1);assert.equal(e.ledger.counts[0].emissions,0);
me.cooldown=0;
for(let i=0;i<5;i++){const b=new e.env.TT.Bullet(800+i*10,700,0,0,'#fff');e.ledger.attach(b);e.game.bullets.push(b);}
e.invalidate();assert.equal(e.observation()[16],0);
e.step({...idle,shoot:true});assert.equal(e.ledger.counts[0].ammo,1);

const results=[];
for(const skill of SKILLS){
  const env=new TankEnvV2({objective:'frag'});await env.reset({seed:723,n:2,seat:0,skill,players:['builtin','builtin']});
  while(!env.done)env.step(idle,'reference');
  const summary=env.summary();results.push({skill,success:summary.skill_success,frames:summary.frames,events:summary.events});
  assert.equal(summary.skill_success,true,'Microtask reference failed: '+skill);
  assert.ok(summary.duration<=60);
}
console.log(JSON.stringify({gate:'G0',passed:true,skills:results},null,2));
