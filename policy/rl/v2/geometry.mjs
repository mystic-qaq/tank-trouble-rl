export const DT=1/60;
export const SCHEMA=Object.freeze({version:2,size:1966,self:32,rays:24,history:24,map:374,
  enemy_count:7,enemy_size:24,enemy_offset:454,bullet_count:40,bullet_size:24,bullet_offset:622,
  aim_count:32,aim_size:12,aim_offset:1582,ready_index:16});
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const wrap=a=>Math.atan2(Math.sin(a),Math.cos(a));
const paths=new Map();

function pathsFor(map){
  const key=map.seed+':'+Array.from(map.vFlag).join('')+':'+Array.from(map.hFlag).join('');
  if(paths.has(key))return paths.get(key);
  const total=map.cols*map.rows,distance=new Int16Array(total*total).fill(-1),first=new Int16Array(total*total).fill(-1);
  for(let source=0;source<total;source++){
    const queue=[source],base=source*total;distance[base+source]=0;first[base+source]=source;
    for(let head=0;head<queue.length;head++){
      const at=queue[head],r=Math.floor(at/map.cols),c=at%map.cols;
      const neighbors=[];
      if(c+1<map.cols&&!map.vAt(r,c+1))neighbors.push(at+1);
      if(c>0&&!map.vAt(r,c))neighbors.push(at-1);
      if(r+1<map.rows&&!map.hAt(r+1,c))neighbors.push(at+map.cols);
      if(r>0&&!map.hAt(r,c))neighbors.push(at-map.cols);
      for(const next of neighbors)if(distance[base+next]<0){
        distance[base+next]=distance[base+at]+1;first[base+next]=at===source?next:first[base+at];queue.push(next);
      }
    }
  }
  const result={distance,first,total};paths.set(key,result);
  if(paths.size>32)paths.delete(paths.keys().next().value);
  return result;
}

export function intercept(me,target,speed,offset=28){
  const rx=target.x-me.x,ry=target.y-me.y;
  const vx=Math.cos(target.angle)*target.speed,vy=Math.sin(target.angle)*target.speed;
  const a=vx*vx+vy*vy-speed*speed,b=2*(rx*vx+ry*vy-offset*speed),c=rx*rx+ry*ry-offset*offset;
  let t=0;
  const disc=b*b-4*a*c;
  if(Math.abs(a)>1e-8&&disc>=0){
    const roots=[(-b-Math.sqrt(disc))/(2*a),(-b+Math.sqrt(disc))/(2*a)].filter(x=>x>=0);
    if(roots.length)t=Math.min(...roots);
  }
  return {angle:Math.atan2(ry+vy*t,rx+vx*t),time:t,distance:Math.hypot(rx+vx*t,ry+vy*t)};
}

// Wall trajectories use the game's radius-aware raycast. They are predictions, never damage authority.
export function flight(game,x,y,angle,speed,seconds,bounces=0){
  const cfg=game.__cfg,segments=[];let remaining=speed*seconds,time=0;
  for(let guard=0;remaining>.02&&guard<8;guard++){
    let vx=Math.cos(angle),vy=Math.sin(angle);
    const hit=game.map.raycast(x,y,x+vx*remaining,y+vy*remaining,cfg.BULLET.radius);
    const length=remaining*(hit?clamp(hit.t,0,1):1),duration=length/speed;
    segments.push({x,y,vx:vx*speed,vy:vy*speed,time,duration});
    x+=vx*length;y+=vy*length;remaining-=length;time+=duration;
    if(!hit)break;
    const nx=hit.nx||0,ny=hit.ny||0;
    if((!nx&&!ny)||++bounces>cfg.BULLET.bounces)break;
    const dot=vx*nx+vy*ny;if(dot<0){vx-=2*dot*nx;vy-=2*dot*ny;}
    angle=Math.atan2(vy,vx);x+=nx*.8;y+=ny*.8;remaining-=.7;
  }
  return segments;
}

function closest(segments,tank,grace=0){
  const tvx=Math.cos(tank.angle)*tank.speed,tvy=Math.sin(tank.angle)*tank.speed;
  let best=Infinity,when=1.2;
  for(const s of segments){
    const lo=Math.max(0,grace-s.time);if(lo>s.duration)continue;
    const rx=s.x-tank.x-tvx*s.time,ry=s.y-tank.y-tvy*s.time;
    const vx=s.vx-tvx,vy=s.vy-tvy;
    const t=clamp(-(rx*vx+ry*vy)/(vx*vx+vy*vy+1e-9),lo,s.duration);
    const d=Math.hypot(rx+vx*t,ry+vy*t);
    if(d<best){best=d;when=s.time+t;}
  }
  return {distance:Number.isFinite(best)?best:Math.hypot(1280,800),time:when};
}

export function traceShot(game,me,angle){
  const cfg=game.__cfg,offset=me.radius+12;
  const x=me.x+Math.cos(angle)*offset,y=me.y+Math.sin(angle)*offset;
  if(game.map.raycast(me.x,me.y,x,y,cfg.BULLET.radius))return {hit:false,self:false,time:0,blocked:true};
  const segments=flight(game,x,y,angle,cfg.BULLET.speed,cfg.BULLET.life);
  let earliest=Infinity,victim=null;
  for(const segment of segments)for(const tank of game.tanks){
    if(!tank.alive)continue;
    const lo=tank.index===me.index?Math.max(0,cfg.BULLET.spawnGrace-segment.time):0;
    if(lo>segment.duration)continue;
    const tvx=Math.cos(tank.angle)*tank.speed,tvy=Math.sin(tank.angle)*tank.speed;
    const rx=segment.x+segment.vx*lo-tank.x-tvx*(segment.time+lo);
    const ry=segment.y+segment.vy*lo-tank.y-tvy*(segment.time+lo);
    const duration=segment.duration-lo;
    const t=game.__util.segCircle(rx,ry,rx+(segment.vx-tvx)*duration,ry+(segment.vy-tvy)*duration,0,0,tank.radius+cfg.BULLET.radius);
    if(t>=0&&segment.time+lo+t*duration<earliest){earliest=segment.time+lo+t*duration;victim=tank;}
  }
  return {hit:!!victim&&victim.index!==me.index,self:victim?.index===me.index,time:Number.isFinite(earliest)?earliest:cfg.BULLET.life,blocked:false};
}

export function aimCandidates(game,me){
  const cfg=game.__cfg,enemies=game.tanks.filter(t=>t.index!==me.index&&t.alive);
  const direct=[],bank=[];
  const walls=game.map.walls.slice().sort((a,b)=>{
    const dist=w=>Math.hypot(clamp(me.x,w.x,w.x+w.w)-me.x,clamp(me.y,w.y,w.y+w.h)-me.y);
    return dist(a)-dist(b);
  }).slice(0,12);
  for(const enemy of enemies){
    direct.push({...intercept(me,enemy,cfg.BULLET.speed),bounces:0,target:enemy});
    direct.push({angle:Math.atan2(enemy.y-me.y,enemy.x-me.x),distance:Math.hypot(enemy.x-me.x,enemy.y-me.y),time:0,bounces:0,target:enemy});
    for(const wall of walls){
      const vertical=wall.h>wall.w,axis=vertical?'x':'y';
      const plane=me[axis]<wall[axis]+(vertical?wall.w:wall.h)/2?wall[axis]-cfg.BULLET.radius:wall[axis]+(vertical?wall.w:wall.h)+cfg.BULLET.radius;
      if((me[axis]-plane)*(enemy[axis]-plane)<=0)continue;
      const vx=Math.cos(enemy.angle)*enemy.speed,vy=Math.sin(enemy.angle)*enemy.speed;
      const target={...enemy,[axis]:2*plane-enemy[axis],angle:Math.atan2(vertical?vy:-vy,vertical?-vx:vx)};
      const candidate=intercept(me,target,cfg.BULLET.speed);
      const dx=Math.cos(candidate.angle),dy=Math.sin(candidate.angle);
      const travel=(plane-me[axis])/(vertical?dx:dy);
      if(travel<=0||!Number.isFinite(travel))continue;
      const cross=vertical?me.y+dy*travel:me.x+dx*travel;
      const start=vertical?wall.y:wall.x,end=start+(vertical?wall.h:wall.w);
      if(cross<start-cfg.BULLET.radius||cross>end+cfg.BULLET.radius)continue;
      bank.push({...candidate,bounces:1,target:enemy});
    }
  }
  bank.sort((a,b)=>Math.abs(wrap(a.angle-me.angle))-Math.abs(wrap(b.angle-me.angle)));
  return [...direct,...bank.slice(0,SCHEMA.aim_count-direct.length)].slice(0,SCHEMA.aim_count).map(c=>({...c,...traceShot(game,me,c.angle)}));
}

export function observeV2(env){
  const game=env.game,cfg=game.__cfg,me=game.tanks[env.seat],out=new Float32Array(SCHEMA.size);
  const {W,H,TANK:T,BULLET:B}=cfg,diagonal=Math.hypot(W,H),cos=Math.cos(me.angle),sin=Math.sin(me.angle);
  const body=(x,y)=>[x*cos+y*sin,-x*sin+y*cos];
  const bullets=game.bullets.filter(b=>!b.dead),ammo=Array(8).fill(0);bullets.forEach(b=>ammo[b.owner]++);
  const muzzle=me.barrelTip(),ready=me.alive&&me.cooldown<=DT+1e-10&&ammo[me.index]<cfg.MAX_BULLETS_PER_TANK;
  const rays=[];
  for(let i=0;i<24;i++){
    const a=me.angle+i*Math.PI/12,h=game.map.raycast(me.x,me.y,me.x+Math.cos(a)*diagonal,me.y+Math.sin(a)*diagonal,me.radius);
    rays.push(h?Math.max(0,h.t):1);
  }
  out.set([me.x/W,me.y/H,cos,sin,me.speed/T.speed,me.cooldown/T.fireCooldown,+me.alive,ammo[me.index]/5,
    game.tanks.length/8,game.tanks.filter(t=>t.alive).length/8,Math.max(0,1-game.roundTime/cfg.ROUND_TIME_LIMIT),
    env.previous.drive,env.previous.steer,+env.previous.shoot,me.radius/80,Math.min(me.lastShotAge,6)/6,
    +ready,+!!game.map.raycast(me.x,me.y,muzzle.x,muzzle.y,B.radius),rays[0],rays[12],cos*me.speed/T.speed,sin*me.speed/T.speed,
    Math.sin(me.angle-env.lastAngle),Math.cos(me.angle-env.lastAngle),game.kills[me.index]/7,
    Math.min(6,...bullets.filter(b=>b.owner===me.index).map(b=>b.life))/6,1,1.2,
    T.turn*DT,B.speed/T.speed,env.idleFrames/3600,+(env.lastDisplacement<.05)],0);
  out.set(rays,32);out.set(env.history,56);
  let at=80;
  for(let r=0;r<11;r++)for(let c=0;c<17;c++)out[at++]=r<10?game.map.vAt(r,c):0;
  for(let r=0;r<11;r++)for(let c=0;c<17;c++)out[at++]=c<16?game.map.hAt(r,c):0;
  const path=pathsFor(game.map),cell=p=>clamp(Math.floor(p.y/80),0,9)*16+clamp(Math.floor(p.x/80),0,15);
  const selfCell=cell(me),enemies=game.tanks.filter(t=>t.index!==me.index);
  at=SCHEMA.enemy_offset;
  for(const tank of enemies){
    const [dx,dy]=body(tank.x-me.x,tank.y-me.y),[vx,vy]=body(Math.cos(tank.angle)*tank.speed-cos*me.speed,Math.sin(tank.angle)*tank.speed-sin*me.speed);
    const lead=intercept(me,tank,B.speed),error=wrap(lead.angle-me.angle),targetCell=cell(tank);
    const next=path.first[selfCell*path.total+targetCell],steps=path.distance[selfCell*path.total+targetCell];
    const waypoint=next<0?{x:me.x,y:me.y}:{x:(next%16)*80+40,y:Math.floor(next/16)*80+40};
    const [wx,wy]=body(waypoint.x-me.x,waypoint.y-me.y);
    out.set([dx/diagonal,dy/diagonal,Math.cos(tank.angle-me.angle),Math.sin(tank.angle-me.angle),tank.speed/T.speed,tank.cooldown/T.fireCooldown,
      +tank.alive,ammo[tank.index]/5,tank.x/W,tank.y/H,tank.radius/80,vx/T.speed,vy/T.speed,
      Math.hypot(dx,dy)/diagonal,Math.sin(error),Math.cos(error),+!game.map.raycast(me.x,me.y,tank.x,tank.y,B.radius),
      lead.time/6,Math.atan2(tank.radius+B.radius,Math.max(1,lead.distance))/Math.PI,steps<0?1:steps/160,
      wx/80,wy/80,Math.cos(Math.atan2(dy,dx)),1],at);at+=24;
  }
  at=SCHEMA.bullet_offset;let nearest=Infinity,nearestTime=1.2;
  for(const bullet of bullets){
    const [dx,dy]=body(bullet.x-me.x,bullet.y-me.y),[vx,vy]=body(Math.cos(bullet.angle)*bullet.speed,Math.sin(bullet.angle)*bullet.speed);
    const segments=flight(game,bullet.x,bullet.y,bullet.angle,bullet.speed,Math.min(1.2,bullet.life),bullet.bounces);
    const danger=closest(segments,me,bullet.owner===me.index?bullet.grace:0);
    if(danger.distance<nearest){nearest=danger.distance;nearestTime=danger.time;}
    const projections=[];
    for(const t of [.12,.35,.8]){
      const segment=segments.find(s=>s.time+s.duration>=t)??segments.at(-1);
      const dt=segment?clamp(t-segment.time,0,segment.duration):0;
      const [px,py]=body((segment?segment.x+segment.vx*dt:bullet.x)-me.x,(segment?segment.y+segment.vy*dt:bullet.y)-me.y);
      projections.push(px/diagonal,py/diagonal);
    }
    out.set([dx/diagonal,dy/diagonal,vx/B.speed,vy/B.speed,+(bullet.owner===me.index),bullet.life/B.life,bullet.bounces/B.bounces,
      bullet.grace/B.spawnGrace,bullet.x/W,bullet.y/H,bullet.radius/80,danger.distance/diagonal,danger.time/1.2,
      +(danger.distance<me.radius+bullet.radius),...projections,Math.hypot(dx,dy)/diagonal,
      Math.cos(bullet.angle-me.angle),Math.sin(bullet.angle-me.angle),1],at);at+=24;
  }
  out[26]=Number.isFinite(nearest)?nearest/diagonal:1;out[27]=nearestTime/1.2;
  const candidates=aimCandidates(game,me);at=SCHEMA.aim_offset;
  for(const c of candidates){
    const error=wrap(c.angle-me.angle),[tx,ty]=body(c.target.x-me.x,c.target.y-me.y);
    out.set([Math.sin(error),Math.cos(error),c.distance/diagonal,c.time/6,c.bounces/5,+c.hit,+c.self,
      c.target.speed/T.speed,tx/diagonal,ty/diagonal,Math.atan2(c.target.radius+B.radius,Math.max(1,c.distance))/Math.PI,1],at);at+=12;
  }
  if(!out.every(Number.isFinite))throw new Error('Nonfinite V2 observation');
  return out;
}
