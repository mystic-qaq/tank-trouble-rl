import assert from 'node:assert/strict';
import {TankEnvV2} from '../policy/rl/v2/env.mjs';
import {observeV2} from '../policy/rl/v2/geometry.mjs';
import {createTrackers,afterFrame} from './rl-controller.mjs';
import './schedule.js';
let checked=0;
for(const learnedId of ['rl_win','rl_frag','rl_v7'])for(const seat of [0,3]){
  const e=new TankEnvV2({objective:'frag'});await e.reset({seed:45291+seat,n:4,seat});
  globalThis.TT=e.env.TT;
  const names=Array(4).fill('builtin');names[seat]=learnedId;
  const [tracker]=createTrackers(e.game,names);
  assert.deepEqual(observeV2(tracker),e.observation());
  for(let frame=0;frame<240&&!e.done;frame++){
    const tank=e.game.tanks[seat];
    const action={drive:[1,0,-1][Math.floor(frame/40)%3],steer:Math.sin(frame*.11),shoot:frame%15===0};
    tracker.before={x:tank.x,y:tank.y,angle:tank.angle};
    tracker.action=tank.alive?action:{drive:0,steer:0,shoot:false};
    e.step(action);afterFrame([tracker]);
    assert.deepEqual(observeV2(tracker),e.observation(),learnedId+' browser action history differs from training at frame '+frame);
    checked++;
  }
}
const ids=['rl_win','rl_frag','rl_v7','minimax'];
for(let block=0;block<2;block++){
  const plans=Array.from({length:4},(_,i)=>TTArenaSchedule.plan(block*4+i,ids));
  assert.equal(new Set(plans.map(p=>p.spawnSeed)).size,1);
  assert.equal(new Set(plans.map(p=>p.mapSeed)).size,1);
  for(const id of ids)assert.equal(new Set(plans.map(p=>p.seats.indexOf(id))).size,4);
}
assert.notEqual(TTArenaSchedule.plan(0,ids).mapSeed,TTArenaSchedule.plan(4,ids).mapSeed);
console.log(JSON.stringify({passed:true,training_browser_feature_frames_identical:checked,rl_roster_complete_seat_map_cycles:true}));
