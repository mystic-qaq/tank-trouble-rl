// Framed binary IPC: two uint32 LE lengths, JSON metadata, raw float32 payload.
import { TankEnvV2 } from './env.mjs';
import { SCHEMA } from './geometry.mjs';
import { skillReference } from './skills.mjs';
import { mulberry32,roundSeed } from '../../../arena/arena.mjs';
const idle={drive:0,steer:0,shoot:false};
let input=Buffer.alloc(0),chain=Promise.resolve();
const slots=new Map();
function send(meta,data=new Float32Array()){
  const json=Buffer.from(JSON.stringify(meta)),bytes=Buffer.from(data.buffer,data.byteOffset,data.byteLength);
  const prefix=Buffer.alloc(8);prefix.writeUInt32LE(json.length);prefix.writeUInt32LE(bytes.length,4);
  process.stdout.write(Buffer.concat([prefix,json,bytes]));
}
async function handle(msg){
  if(msg.op==='ping')return send({ok:true,schema:SCHEMA});
  if(msg.op==='reset'){
    const env=new TankEnvV2(msg.options);slots.set(msg.slot,env);
    const obs=await env.reset(msg.config);return send({ok:true,slot:msg.slot},obs);
  }
  if(msg.op==='step'){
    const env=slots.get(msg.slot);env.skipFeatures=!env.game.tanks[env.seat].alive;
    let label=null;
    if(msg.label_teacher&&env.game.tanks[env.seat].alive){
      const me=env.game.tanks[env.seat],action=msg.label_teacher==='reference'?skillReference(env):env.policyAction(msg.label_teacher,me);
      // Counterfactual movement is resolved with the native Tank update, on a fresh clone.
      // The teacher fire label belongs to this teacher movement, never to the student's movement.
      const clone=new env.env.TT.Tank(me.index,{x:me.x,y:me.y,a:me.angle},me.palette,true);
      clone.speed=me.speed;clone.cooldown=me.cooldown;
      clone.update(1/60,{...action,shoot:false},{map:env.game.map,shake(){}});
      const tip=clone.barrelTip(),cfg=env.game.__cfg;
      const legal=clone.cooldown<=0&&env.game.bullets.filter(b=>!b.dead&&b.owner===me.index).length<5&&
        !env.game.map.raycast(clone.x,clone.y,tip.x,tip.y,cfg.BULLET.radius);
      label=[action.drive,action.steer,+(action.shoot&&legal)];
    }
    const r=env.step(msg.action,msg.teacher);
    return send({ok:true,slot:msg.slot,done:env.done,reward:r.reward,active:r.active,action:r.action,info:r.info,events:r.events,label},r.observation);
  }
  if(msg.op==='collect'){
    const env=new TankEnvV2({objective:msg.teacher==='hy'?'win':'frag',early_death:true});
    env.skipFeatures=true;await env.reset(msg.config);
    const rng=mulberry32(roundSeed(msg.config.seed,91)),observations=[],labels=[],rows=[],events=[];
    while(!env.done){
      const me=env.game.tanks[env.seat],active=me.alive;
      const action=active?(msg.teacher==='reference'?(await import('./skills.mjs')).skillReference(env):env.policyAction(msg.teacher,me)):idle;
      const own=env.game.bullets.filter(b=>!b.dead&&b.owner===env.seat).length;
      const eligible=active&&me.cooldown<=1/60+1e-10&&own<5;
      const probability=action.shoot?1:eligible?.12:.025;
      const keep=active&&rng()<probability;
      let obs;
      if(keep){env.skipFeatures=false;env.invalidate();obs=env.observation();env.skipFeatures=true;}
      const r=env.step(action);
      for(const event of r.events)events.push(event);
      if(keep){
        observations.push(obs);const emitted=r.events.some(e=>e.owner===env.seat&&e.kind==='emission');
        labels.push(action.drive,action.steer,+emitted);
        rows.push({frame:env.frames-1,p:probability,eligible:+eligible,requested:+action.shoot,emitted:+emitted});
      }
    }
    const packed=new Float32Array(observations.length*(SCHEMA.size+3));
    observations.forEach((o,i)=>{packed.set(o,i*(SCHEMA.size+3));packed.set(labels.slice(i*3,i*3+3),i*(SCHEMA.size+3)+SCHEMA.size);});
    return send({ok:true,summary:env.summary(),rows,events},packed);
  }
  throw new Error('Unknown operation '+msg.op);
}
process.stdin.on('data',chunk=>{
  input=Buffer.concat([input,chunk]);
  while(input.length>=8){
    const n=input.readUInt32LE(),m=input.readUInt32LE(4);if(input.length<8+n+m)break;
    const msg=JSON.parse(input.subarray(8,8+n));input=input.subarray(8+n+m);
    chain=chain.then(()=>handle(msg)).catch(error=>{send({ok:false,error:error.stack});});
  }
});
