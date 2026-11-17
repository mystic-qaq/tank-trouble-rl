import { observeV2 } from '../policy/rl/v2/geometry.mjs';
export const RL_IDS=['rl_win','rl_frag','rl_v7'];
export const isLearned=id=>RL_IDS.includes(id);
const idle=()=>({drive:0,steer:0,shoot:false});
const files={rl_win:'v2-win',rl_frag:'v2-frag',rl_v7:'v7-distill1'};
let worker,serial=0;
const pending=new Map(),metadata=new Map();
function request(message,transfer=[]){
  if(!worker){
    worker=new Worker(new URL('./rl-inference-worker.js',import.meta.url));
    worker.onmessage=({data})=>{
      const ticket=pending.get(data.request);if(!ticket)return;
      pending.delete(data.request);clearTimeout(ticket.timeout);
      data.error?ticket.reject(new Error(data.error)):ticket.resolve(data.result);
    };
    worker.onerror=error=>{
      for(const ticket of pending.values()){clearTimeout(ticket.timeout);ticket.reject(new Error(error.message));}pending.clear();
    };
  }
  return new Promise((resolve,reject)=>{
    const id=++serial;
    const timeout=setTimeout(()=>{pending.delete(id);reject(new Error('本机模型推理超时，请重载页面。'));},60000);
    pending.set(id,{resolve,reject,timeout});worker.postMessage({...message,request:id},transfer);
  });
}
export async function loadModels(ids){
  const models=[];
  for(const id of ids.filter(isLearned)){
    if(!metadata.has(id)){
      const response=await fetch(new URL('./models/'+files[id]+'.json',import.meta.url));
      if(!response.ok)throw new Error('缺少模型说明：'+id);
      const meta=await response.json();
      if(meta.observation_version!==2||!meta.deterministic||meta.export_bias_adjustment!==0)throw new Error('不支持的模型版本');
      metadata.set(id,meta);
    }
    models.push({id,url:new URL('./models/'+files[id]+'.onnx',import.meta.url).href,sha256:metadata.get(id).onnx_sha256});
  }
  if(models.length)await request({op:'load',models});
  return models.map(m=>metadata.get(m.id));
}
export async function inferObservations(inputs){
  if(!inputs.length)return {outputs:[],milliseconds:0};
  return request({op:'infer',inputs},inputs.map(i=>i.observation.buffer));
}
export function createTrackers(game,seats){
  game.__cfg=TT.CFG;game.__util=TT.util;
  return seats.flatMap((id,seat)=>isLearned(id)?[{id,seat,game,previous:idle(),history:new Float32Array(24),
    lastAngle:game.tanks[seat].angle,lastDisplacement:0,idleFrames:0,action:idle()}]:[]);
}
export async function beforeFrame(trackers){
  const active=trackers.filter(t=>t.game.tanks[t.seat].alive);
  // Every actor sees the same pre-update world, as in training. Never observe mid-seat updates.
  const inputs=active.map(t=>({id:t.id,observation:observeV2(t)}));
  const result=await inferObservations(inputs);
  for(const t of trackers){
    const tank=t.game.tanks[t.seat];t.before={x:tank.x,y:tank.y,angle:tank.angle};t.action=idle();
    const output=result.outputs.find(o=>o.id===t.id);
    if(output)t.action={drive:output.controls[0],steer:output.controls[1],shoot:!!output.controls[2]};
  }
  return result.milliseconds;
}
export function afterFrame(trackers){
  for(const t of trackers){
    const tank=t.game.tanks[t.seat];t.lastAngle=t.before.angle;
    t.lastDisplacement=Math.hypot(tank.x-t.before.x,tank.y-t.before.y);
    t.idleFrames=t.lastDisplacement<.05?t.idleFrames+1:0;
    t.previous=t.action;t.history.copyWithin(0,3);t.history.set([t.action.drive,t.action.steer,+t.action.shoot],21);
  }
}
