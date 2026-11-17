// Native 60 Hz engine, no action repeat and no new tactical safety controller.
import {TankEnvV2} from '../v2/env.mjs';
import {skillReference} from '../v2/skills.mjs';
const idle=()=>({drive:0,steer:0,shoot:false});
const slots=new Map();
let buffer=Buffer.alloc(0),queue=Promise.resolve();
function send(meta,data=new Float32Array()) {
  const json=Buffer.from(JSON.stringify(meta)),bytes=Buffer.from(data.buffer,data.byteOffset,data.byteLength),head=Buffer.alloc(8);
  head.writeUInt32LE(json.length);head.writeUInt32LE(bytes.length,4);process.stdout.write(Buffer.concat([head,json,bytes]));
}
export function teacherLabel(e,id) {
  const me=e.game.tanks[e.seat];if(!me.alive)return null;
  const a=id==='reference'?skillReference(e):e.policyAction(id,me);
  const clone=new e.env.TT.Tank(me.index,{x:me.x,y:me.y,a:me.angle},me.palette,true);
  clone.speed=me.speed;clone.cooldown=me.cooldown;
  clone.update(1/60,{...a,shoot:false},{map:e.game.map,shake(){}});
  const tip=clone.barrelTip(),cfg=e.game.__cfg;
  const legal=clone.cooldown<=0&&e.game.bullets.filter(b=>!b.dead&&b.owner===me.index).length<cfg.MAX_BULLETS_PER_TANK&&
    !e.game.map.raycast(clone.x,clone.y,tip.x,tip.y,cfg.BULLET.radius);
  return [a.drive,a.steer,+(a.shoot&&legal)];
}
function measure(e,action,before,events) {
  const me=e.game.tanks[e.seat],m=e.measure;
  if(!before.alive)return;
  m.active++;m.frames++;m.radius=Math.max(m.radius,Math.hypot(me.x-m.x,me.y-m.y));
  m.turn+=Math.abs(Math.atan2(Math.sin(me.angle-m.angle),Math.cos(me.angle-m.angle)));m.angle=me.angle;
  m.driving+=+(action.drive!==0);m.shots+=events.filter(x=>x.owner===e.seat&&x.kind==='emission').length;
  if(m.frames===180) {
    if(m.radius<24&&m.shots===0) {
      if(m.turn>5.5)m.spin+=180;
      if(m.driving>90)m.stuck+=180;
    }
    m.frames=0;m.x=me.x;m.y=me.y;m.turn=0;m.radius=0;m.shots=0;m.driving=0;
  }
}
async function handle(m) {
  if(m.op==='reset') {
    const e=new TankEnvV2(m.options);slots.set(m.slot,e);const obs=await e.reset(m.config),me=e.game.tanks[e.seat];
    e.measure={active:0,spin:0,stuck:0,frames:0,x:me.x,y:me.y,angle:me.angle,turn:0,radius:0,shots:0,driving:0};
    return send({ok:true},obs);
  }
  if(m.op==='step') {
    const e=slots.get(m.slot),before={...e.game.tanks[e.seat]};
    const killsBefore=e.kills,suicidesBefore=e.suicides;
    e.skipFeatures=!before.alive;
    const label=m.label_teacher?teacherLabel(e,m.label_teacher):null;
    const r=m.teacher==='reference'?e.step(skillReference(e)):e.step(m.action,m.teacher);
    measure(e,r.action,before,r.events);
    const info=e.done?{...e.summary(),spin_frames:e.measure.spin,stuck_frames:e.measure.stuck,active_frames:e.measure.active}:null;
    return send({ok:true,done:e.done,reward:r.reward,active:r.active,label,events:r.events,info,
      kills_delta:e.kills-killsBefore,suicides_delta:e.suicides-suicidesBefore},r.observation);
  }
  throw new Error('Unknown V4 operation '+m.op);
}
// Importing teacherLabel in the engine tests must not start a server.
if(process.argv[1]&&new URL(import.meta.url).pathname.replace(/^\/(\w:)/,'$1')===process.argv[1].replaceAll('\\','/')) {
  process.stdin.on('data',chunk=>{
    buffer=Buffer.concat([buffer,chunk]);
    while(buffer.length>=8) {
      const n=buffer.readUInt32LE(),m=buffer.readUInt32LE(4);if(buffer.length<8+n+m)break;
      const message=JSON.parse(buffer.subarray(8,8+n));buffer=buffer.subarray(8+n+m);
      queue=queue.then(()=>handle(message)).catch(error=>send({ok:false,error:error.stack}));
    }
  });
}
