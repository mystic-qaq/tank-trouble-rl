// Observer-only instrumentation. The original engine still decides all physics and damage.
export class EventLedger {
  constructor(env) {
    this.env=env;this.pending=[];this.serial=0;this.activeBullet=null;
    this.counts=Array.from({length:8},()=>({requests:0,emissions:0,hits:0,self_hits:0,fizzles:0,cooldown:0,ammo:0,bounces:0}));
    this.shots=new Map();
    const game=env.game, dispatch=env.env.TT.updateBot;
    env.env.TT.updateBot=(tank,g,dt)=>{
      const action=dispatch(tank,g,dt);
      if(action.shoot){
        this.emit('request',tank.index);
        if(tank.cooldown>dt+1e-10)this.emit('cooldown',tank.index);
      }
      return action;
    };
    for(const tank of game.tanks){
      const fire=tank.fire.bind(tank);
      tank.fire=g=>{
        const before=g.bullets.length;
        const capped=g.bullets.filter(b=>!b.dead&&b.owner===tank.index).length>=g.__cfg.MAX_BULLETS_PER_TANK;
        fire(g);
        if(capped)this.emit('ammo',tank.index);
        else if(g.bullets.length===before)this.emit('fizzle',tank.index,{x:tank.x,y:tank.y});
        for(let i=before;i<g.bullets.length;i++)this.attach(g.bullets[i],true);
      };
    }
    const kill=game.killTank.bind(game);
    game.killTank=(tank,bullet)=>{
      const alive=tank.alive;kill(tank,bullet);
      if(alive&&bullet){
        const self=tank.index===bullet.owner;
        this.emit(self?'self_hit':'hit',bullet.owner,{shot_id:bullet.shotId??null,victim:tank.index,x:tank.x,y:tank.y});
        const shot=this.shots.get(bullet.shotId);if(shot)shot.cause=self?'self_hit':'hit';
      }
    };
    const spark=game.fx.bounceSpark.bind(game.fx);
    game.fx.bounceSpark=(x,y,nx,ny)=>{
      if(this.activeBullet)this.emit('bounce',this.activeBullet.owner,{shot_id:this.activeBullet.shotId,x,y,nx,ny});
      spark(x,y,nx,ny);
    };
  }
  emit(kind,owner,details={}){
    const key={request:'requests',emission:'emissions',hit:'hits',self_hit:'self_hits',fizzle:'fizzles',cooldown:'cooldown',ammo:'ammo',bounce:'bounces'}[kind];
    if(key&&this.counts[owner])this.counts[owner][key]++;
    this.pending.push({kind,owner,frame:Math.round(this.env.game.roundTime*60),...details});
  }
  attach(bullet,emitted=false){
    if(bullet.shotId)return bullet.shotId;
    bullet.shotId=`${this.env.seed}:${this.env.seat}:${bullet.owner}:${this.serial++}`;
    const shot={owner:bullet.owner,cause:null};this.shots.set(bullet.shotId,shot);
    if(emitted)this.emit('emission',bullet.owner,{shot_id:bullet.shotId,x:bullet.x,y:bullet.y,angle:bullet.angle});
    const update=bullet.update.bind(bullet);
    bullet.update=(dt,g)=>{
      const dead=bullet.dead;this.activeBullet=bullet;
      try{update(dt,g);}finally{this.activeBullet=null;}
      if(!dead&&bullet.dead){
        const cause=shot.cause??(bullet.life<=0?'expired':bullet.bounces>g.__cfg.BULLET.bounces?'bounce_limit':'wall');
        this.emit('despawn',bullet.owner,{shot_id:bullet.shotId,cause,x:bullet.x,y:bullet.y});
      }
    };
    return bullet.shotId;
  }
  drain(){const pending=this.pending;this.pending=[];return pending;}
}
