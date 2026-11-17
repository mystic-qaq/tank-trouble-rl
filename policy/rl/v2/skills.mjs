import { mulberry32 } from '../../../arena/arena.mjs';
import { aimCandidates,flight,DT } from './geometry.mjs';
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export const SKILLS=['static','moving','bank','muzzle','dodge','volley'];

export function configureSkill(env,kind){
  if(!SKILLS.includes(kind))throw new Error('Unknown skill '+kind);
  const {game}=env,{map}=game,cfg=game.__cfg,rng=mulberry32(env.seed^0x5a18c903);
  map.vFlag.fill(0);map.hFlag.fill(0);
  for(let r=0;r<10;r++){map.vFlag[r*17]=1;map.vFlag[r*17+16]=1;}
  for(let c=0;c<16;c++){map.hFlag[c]=1;map.hFlag[10*16+c]=1;}
  const me=game.tanks[env.seat],target=game.tanks.find(t=>t.index!==env.seat);
  for(const tank of game.tanks){tank.speed=0;tank.cooldown=0;}
  me.x=250+rng()*150;me.y=200+rng()*400;me.angle=(rng()*2-1)*Math.PI;
  target.x=850+rng()*200;target.y=150+rng()*500;target.angle=0;
  if(kind==='moving'){target.angle=rng()<.5?Math.PI/2:-Math.PI/2;target.speed=cfg.TANK.speed;}
  if(kind==='bank'){
    me.x=320;me.y=300+rng()*200;target.x=960;target.y=300+rng()*200;
    for(let r=2;r<8;r++)map.vFlag[r*17+8]=1;
  }
  if(kind==='muzzle'){
    me.x=299;me.y=160+rng()*480;me.angle=(rng()-.5)*.16;target.x=430;target.y=me.y;
    for(let r=0;r<10;r++)map.vFlag[r*17+4]=1;
  }
  if(kind==='dodge'||kind==='volley'){
    me.x=460+rng()*360;me.y=280+rng()*240;
    env.incomingAngle=rng()*Math.PI*2;
    me.angle=wrap(env.incomingAngle+Math.PI/2+(rng()-.5)*.6);target.x=1100;target.y=700;
  }
  map.buildWalls();
  game.staticLayer=null;
  env.skill=kind;env.skillLimit=kind==='muzzle'?60:['dodge','volley'].includes(kind)?120:480;
  const dispatch=env.env.TT.updateBot;
  env.env.TT.updateBot=(tank,g,dt)=>{
    if(tank.index===env.seat)return dispatch(tank,g,dt);
    if(kind!=='moving')return {drive:0,steer:0,shoot:false};
    if(tank.y<100||tank.y>700){
      const goal=tank.y<400?Math.PI/2:-Math.PI/2;
      return {drive:1,steer:clamp(wrap(goal-tank.angle)/(cfg.TANK.turn*dt),-1,1),shoot:false};
    }
    return {drive:1,steer:0,shoot:false};
  };
  if(kind==='dodge'||kind==='volley'){
    for(const dy of kind==='volley'?[-24,0,24]:[0]){
      const a=env.incomingAngle;
      const b=new env.env.TT.Bullet(me.x+260*Math.cos(a)-dy*Math.sin(a),me.y+260*Math.sin(a)+dy*Math.cos(a),wrap(a+Math.PI),target.index,target.color);
      b.life=1.5;env.ledger.attach(b);game.bullets.push(b);
    }
  }
  env.lastAngle=me.angle;env.invalidate();
}

function dodgeReference(env){
  const {game}=env,me=game.tanks[env.seat],cfg=game.__cfg;
  const paths=game.bullets.filter(b=>!b.dead).map(b=>({b,segments:flight(game,b.x,b.y,b.angle,b.speed,Math.min(.8,b.life),b.bounces)}));
  let best=-Infinity,pick={drive:0,steer:0,shoot:false};
  for(const drive of [-1,0,1])for(const steer of [-1,0,1]){
    const clone=new env.env.TT.Tank(me.index,{x:me.x,y:me.y,a:me.angle},me.palette,true);
    Object.assign(clone,me);
    let clearance=1000,hit=false,bumped=0;
    const g={...game,bullets:[],shake(){}};
    for(let step=1;step<=42;step++){
      const before={x:clone.x,y:clone.y};clone.update(DT,{drive,steer,shoot:false},g);
      if(Math.hypot(clone.x-before.x,clone.y-before.y)<.001&&drive)bumped++;
      const t=step*DT;
      for(const {b,segments} of paths){
        if(t>b.life||(b.owner===me.index&&t<b.grace))continue;
        const segment=segments.find(s=>s.time+s.duration>=t)??segments.at(-1);
        if(!segment)continue;
        const dt=clamp(t-segment.time,0,segment.duration);
        const x=segment.x+segment.vx*dt,y=segment.y+segment.vy*dt;
        const distance=Math.hypot(x-clone.x,y-clone.y)-clone.radius-b.radius;
        clearance=Math.min(clearance,distance);if(distance<0)hit=true;
      }
    }
    const score=(hit?-10000:0)+clearance-bumped*.5;
    if(score>best){best=score;pick={drive,steer,shoot:false};}
  }
  return pick;
}

// Analytic reference is only a microtask label generator/test baseline, never a deployed learned policy.
export function skillReference(env){
  if(['dodge','volley'].includes(env.skill))return dodgeReference(env);
  if(env.skill==='muzzle')return {drive:0,steer:0,shoot:false};
  const me=env.game.tanks[env.seat],cfg=env.game.__cfg;
  const candidates=aimCandidates(env.game,me).filter(c=>c.hit&&!c.self&&!c.blocked);
  candidates.sort((a,b)=>Math.abs(wrap(a.angle-me.angle))-Math.abs(wrap(b.angle-me.angle)));
  const candidate=candidates[0];
  if(!candidate)return {drive:0,steer:0,shoot:false};
  const error=wrap(candidate.angle-me.angle),steer=clamp(error/(cfg.TANK.turn*DT),-1,1);
  return {drive:0,steer,shoot:Math.abs(error-steer*cfg.TANK.turn*DT)<.004&&me.cooldown<=DT+1e-10};
}
