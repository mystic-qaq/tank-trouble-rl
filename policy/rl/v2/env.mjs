import { TankEnv } from '../env.mjs';
import { EventLedger } from './events.mjs';
import { observeV2,SCHEMA } from './geometry.mjs';
import { configureSkill,skillReference } from './skills.mjs';

export class TankEnvV2 extends TankEnv {
  async reset(config={}){
    this.v2ready=false;this.cachedObservation=null;
    await super.reset(config);
    this.game.__util=this.env.TT.util;
    this.history=new Float32Array(24);this.lastAngle=this.game.tanks[this.seat].angle;
    this.lastDisplacement=0;this.idleFrames=0;
    this.ledger=new EventLedger(this);
    this.v2ready=true;
    this.skill=null;
    if(config.skill)configureSkill(this,config.skill);
    return this.observation();
  }
  observation(){
    if(!this.v2ready||this.skipFeatures)return new Float32Array(SCHEMA.size);
    if(!this.cachedObservation)this.cachedObservation=observeV2(this);
    return this.cachedObservation;
  }
  invalidate(){this.cachedObservation=null;}
  step(action,teacher=null){
    const me=this.game.tanks[this.seat],x=me.x,y=me.y,angle=me.angle;
    this.cachedObservation=null;
    // Base returns an observation through this override; update history only after the physical frame.
    this.v2ready=false;
    const reference=teacher==='reference';
    const result=super.step(reference?skillReference(this):action,reference?null:teacher);
    if(this.skill&&this.frames>=this.skillLimit&&!this.done){this.done=true;result.terminated=true;}
    this.lastAngle=angle;this.lastDisplacement=Math.hypot(me.x-x,me.y-y);
    this.idleFrames=this.lastDisplacement<.05?this.idleFrames+1:0;
    this.history.copyWithin(0,3);this.history.set([this.previous.drive,this.previous.steer,+this.previous.shoot],21);
    this.v2ready=true;this.cachedObservation=null;
    result.observation=this.observation();result.events=this.ledger.drain();
    if(this.done)result.info=this.summary();
    return result;
  }
  summary(){
    const me=this.game.tanks[this.seat];
    const success=this.skill?(['muzzle','dodge','volley'].includes(this.skill)?me.alive&&this.ledger.counts[this.seat].fizzles===0:me.alive&&this.game.tanks.filter(t=>t.index!==this.seat).every(t=>!t.alive)):null;
    return {...super.summary(),observation_version:2,skill:this.skill,skill_success:success,events:this.ledger?{...this.ledger.counts[this.seat]}:null};
  }
}
