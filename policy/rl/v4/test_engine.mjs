import assert from 'node:assert/strict';
import {TankEnv} from '../env.mjs';
import {TankEnvV2} from '../v2/env.mjs';
import {skillReference,SKILLS} from '../v2/skills.mjs';
import {teacherLabel} from './worker.mjs';
const idle={drive:0,steer:0,shoot:false};
const state=e=>JSON.stringify({time:e.game.roundTime,tanks:e.game.tanks.map(t=>[t.x,t.y,t.angle,t.speed,t.cooldown,t.alive]),
  bullets:e.game.bullets.map(b=>[b.x,b.y,b.angle,b.bounces,b.life,b.owner,b.dead]),kills:e.game.kills});
const config={seed:123456,n:4,seat:1,players:['hy','builtin','gpt','builtin']};
const a=new TankEnv({objective:'frag'}),b=new TankEnvV2({objective:'frag'});
await a.reset(config);await b.reset(config);
for(let i=0;i<240&&!a.done;i++) {
  a.step(idle,'gpt');b.step(idle,'gpt');assert.equal(state(a),state(b));
}
let labels=0,successes=[];
for(const skill of SKILLS) {
  const e=new TankEnvV2({objective:'frag'});await e.reset({seed:723,n:2,seat:0,skill,players:['builtin','builtin']});
  while(!e.done) {
    const label=teacherLabel(e,'reference'),action=skillReference(e),r=e.step(action);
    assert.equal(label[2],+r.events.some(x=>x.owner===e.seat&&x.kind==='emission'));
    labels++;
  }
  assert.equal(e.summary().skill_success,true,'Analytic skill baseline invalid: '+skill);
  successes.push(skill);
}
console.log(JSON.stringify({passed:true,nativePhysicsUnchanged:true,counterfactualLabelsChecked:labels,skills:successes}));
