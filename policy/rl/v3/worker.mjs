import {TankEnvV2} from '../v2/env.mjs';
import {observeV2} from '../v2/geometry.mjs';
import {skillReference} from '../v2/skills.mjs';
const idle=()=>({drive:0,steer:0,shoot:false});
let buffer=Buffer.alloc(0),queue=Promise.resolve();
const slots=new Map();
function send(meta,data=new Float32Array()){
  const json=Buffer.from(JSON.stringify(meta)),bytes=Buffer.from(data.buffer,data.byteOffset,data.byteLength),head=Buffer.alloc(8);
  head.writeUInt32LE(json.length);head.writeUInt32LE(bytes.length,4);process.stdout.write(Buffer.concat([head,json,bytes]));
}
const state=e=>JSON.stringify({time:e.game.roundTime,tanks:e.game.tanks.map(t=>[t.x,t.y,t.angle,t.speed,t.cooldown,t.alive]),
  bullets:e.game.bullets.map(b=>[b.x,b.y,b.angle,b.life,b.bounces,b.owner,b.dead]),kills:e.game.kills,winner:e.game.roundWinner});
function tracker(e,seat){return {game:e.game,seat,previous:idle(),history:new Float32Array(24),lastAngle:e.game.tanks[seat].angle,lastDisplacement:0,idleFrames:0};}
function updateTracker(t,action,before){
  const me=t.game.tanks[t.seat];t.lastAngle=before.angle;t.lastDisplacement=Math.hypot(me.x-before.x,me.y-before.y);
  t.idleFrames=t.lastDisplacement<.05?t.idleFrames+1:0;t.previous=action;
  t.history.copyWithin(0,3);t.history.set([action.drive,action.steer,+action.shoot],21);
}
function shadow(e,id){
  const me=e.game.tanks[e.seat];if(!me.alive)return null;
  const a=id==='reference'?skillReference(e):e.policyAction(id,me);
  const clone=new e.env.TT.Tank(me.index,{x:me.x,y:me.y,a:me.angle},me.palette,true);clone.speed=me.speed;clone.cooldown=me.cooldown;
  clone.update(1/60,{...a,shoot:false},{map:e.game.map,shake(){}});
  const tip=clone.barrelTip(),cfg=e.game.__cfg;
  const legal=clone.cooldown<=0&&e.game.bullets.filter(b=>!b.dead&&b.owner===me.index).length<5&&
    !e.game.map.raycast(clone.x,clone.y,tip.x,tip.y,cfg.BULLET.radius);
  return [a.drive,a.steer,+(a.shoot&&legal)];
}
function measure(e,events){
  const me=e.game.tanks[e.seat],m=e.measure;
  if(me.alive){
    m.active++;m.frames++;m.radius=Math.max(m.radius,Math.hypot(me.x-m.x,me.y-m.y));
    m.turn+=Math.abs(Math.atan2(Math.sin(me.angle-m.angle),Math.cos(me.angle-m.angle)));m.angle=me.angle;
    m.shots+=events.filter(x=>x.owner===e.seat&&x.kind==='emission').length;
    if(m.frames===180){if(m.radius<24&&m.turn>5.5&&m.shots===0)m.spin+=180;
      m.frames=0;m.x=me.x;m.y=me.y;m.turn=0;m.radius=0;m.shots=0;}
  }
}
function summary(e){return {...e.summary(),spin_frames:e.measure?.spin??0,active_frames:e.measure?.active??0};}
async function collectWorld(config){
  // All teachers decide from the same frame boundary, matching the deployed actors.
  const e=new TankEnvV2({objective:'frag'});e.skipFeatures=true;await e.reset(config);
  const original=e.policyAction.bind(e),actions=[];let current;
  e.policyAction=(id,t)=>current[t.index];
  while(!e.done){
    current=e.game.tanks.map(t=>t.alive?original(e.players[t.index],t):idle());
    actions.push(current.map(a=>({...a})));e.step(current[0]);
  }
  const expected=state(e),winner=e.game.roundWinner;
  const scores=e.game.tanks.map(t=>({seat:t.index,teacher:e.players[t.index],win:+(winner===t.index),
    net:e.ledger.counts[t.index].hits-e.ledger.counts[t.index].self_hits,...e.ledger.counts[t.index]}));
  const chosen=[];
  if(winner>=0)chosen.push(winner);
  const attackers=scores.filter(s=>s.net>0&&!chosen.includes(s.seat)).sort((a,b)=>b.net-a.net||b.hits-a.hits);
  if(attackers.length)chosen.push(attackers[0].seat);
  if(!chosen.length){const active=scores.filter(s=>s.emissions>=2).sort((a,b)=>b.hits-a.hits||b.emissions-a.emissions);if(active.length)chosen.push(active[0].seat);}
  const r=new TankEnvV2({objective:'frag'});r.skipFeatures=true;await r.reset(config);
  const tracks=chosen.map(s=>tracker(r,s)),samples=chosen.map(()=>[]),labels=chosen.map(()=>[]);
  r.policyAction=(id,t)=>current[t.index];
  for(const frame of actions){
    current=frame;const before=tracks.map(t=>({...r.game.tanks[t.seat]})),active=tracks.map(t=>r.game.tanks[t.seat].alive);
    tracks.forEach((t,i)=>{if(active[i])samples[i].push(observeV2(t));});
    const result=r.step(frame[0]);
    tracks.forEach((t,i)=>{
      const a=active[i]?frame[t.seat]:idle();
      if(active[i])labels[i].push([a.drive,a.steer,+result.events.some(x=>x.owner===t.seat&&x.kind==='emission')]);
      updateTracker(t,a,before[i]);
    });
  }
  if(state(r)!==expected)throw new Error('Selected-trajectory replay changed native physics');
  const trajectories=[];let total=0;
  chosen.forEach((seat,i)=>{
    const score=scores[seat],length=samples[i].length;
    trajectories.push({...score,length,offset:total,quality:score.win||score.net>0?1+.25*Math.min(4,Math.max(0,score.net)):.2});total+=length;
  });
  const packed=new Float32Array(total*1969);let row=0;
  samples.forEach((list,i)=>list.forEach((obs,j)=>{packed.set(obs,row*1969);packed.set(labels[i][j],row*1969+1966);row++;}));
  return send({ok:true,config,summary:e.summary(),trajectories,world_scores:scores,replay_identical:true},packed);
}
async function collectSkill(config){
  const e=new TankEnvV2({objective:'frag'});await e.reset(config);const rows=[];
  while(!e.done){
    const me=e.game.tanks[e.seat];if(!me.alive)break;
    const obs=e.observation(),a=skillReference(e),r=e.step(a),row=new Float32Array(1969);
    row.set(obs);row.set([a.drive,a.steer,+r.events.some(x=>x.owner===e.seat&&x.kind==='emission')],1966);rows.push(row);
  }
  const packed=new Float32Array(rows.length*1969);rows.forEach((r,i)=>packed.set(r,i*1969));
  return send({ok:true,config,summary:e.summary(),trajectories:[{seat:0,teacher:'reference',length:rows.length,offset:0,quality:1,skill:config.skill,win:e.summary().win,net:e.kills-e.suicides}],replay_identical:true},packed);
}
async function handle(m){
  if(m.op==='collect_world')return collectWorld(m.config);
  if(m.op==='collect_skill')return collectSkill(m.config);
  if(m.op==='reset'){
    const e=new TankEnvV2(m.options);slots.set(m.slot,e);const obs=await e.reset(m.config),me=e.game.tanks[e.seat];
    e.measure={active:0,spin:0,frames:0,x:me.x,y:me.y,angle:me.angle,turn:0,radius:0,shots:0};return send({ok:true},obs);
  }
  if(m.op==='step'){
    const e=slots.get(m.slot);e.skipFeatures=!e.game.tanks[e.seat].alive;
    const label=m.label_teacher?shadow(e,m.label_teacher):null,r=e.step(m.action,m.teacher);measure(e,r.events);
    return send({ok:true,done:e.done,reward:r.reward,active:r.active,action:r.action,label,events:r.events,info:e.done?summary(e):null},r.observation);
  }
  throw new Error('Unknown V3 operation '+m.op);
}
process.stdin.on('data',chunk=>{
  buffer=Buffer.concat([buffer,chunk]);
  while(buffer.length>=8){const n=buffer.readUInt32LE(),m=buffer.readUInt32LE(4);if(buffer.length<8+n+m)break;
    const message=JSON.parse(buffer.subarray(8,8+n));buffer=buffer.subarray(8+n+m);
    queue=queue.then(()=>handle(message)).catch(error=>send({ok:false,error:error.stack}));}
});
