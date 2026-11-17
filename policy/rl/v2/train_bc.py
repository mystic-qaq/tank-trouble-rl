"""Bounded V2 BC. Fire sees the recorded teacher motion, with no export bias adjustment."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from .model import Policy,checkpoint
from .collect import atomic_json

def load_data(paths,require_both=True):
    groups={False:[],True:[]}
    for directory in paths:
        for file in sorted(directory.glob('episode-*.npz')):
            with np.load(file) as d:
                if not len(d['obs']):continue
                skill=str(d['skill']);n=len(d['obs'])
                # Analytic aiming tasks teach steering/fire, without teaching global inactivity.
                drive_weight=0. if skill in ['static','moving','bank','muzzle'] else 1.
                weights=np.minimum(40,1/d['probability'])
                groups[bool(d['validation'])].append((d['obs'],d['actions'],weights,
                    np.full(n,drive_weight,dtype=np.float32),np.full(n,skill!='combat',dtype=bool)))
    output={}
    for key,rows in groups.items():
        if not rows:
            if require_both:raise ValueError('Both complete-layout train and validation sets required')
            continue
        output[key]=tuple(torch.from_numpy(np.concatenate([row[i] for row in rows])) for i in range(5))
    return output

def weighted(values,weights):return (values*weights).sum()/weights.sum().clamp(min=1)

def validation(model,data,device,batch=2048):
    obs,actions,weights,drive_weight,skills=data
    combat=~skills
    if combat.any():obs,actions,weights=obs[combat],actions[combat],weights[combat]
    tp=fp=fn=0.;drive_ok=turn_ok=total=mae=0.;ready_count=0.;positive=0.
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(obs),batch):
            x=obs[start:start+batch].to(device);a=actions[start:start+batch].to(device);w=weights[start:start+batch].to(device)
            h,d,t,mag=model.motion(x);logits=model.fire_logits(x,h,a[:,:2]);pred=logits>=0;truth=a[:,2]>.5
            # Natural-distribution fire metrics: undo event-focused sampling.
            eligible=x[:,16]>.5;wf=w*eligible
            tp+=float((wf*pred*truth).sum());fp+=float((wf*pred*~truth).sum());fn+=float((wf*~pred*truth).sum())
            ready_count+=float(wf.sum());positive+=float((wf*truth).sum())
            drive_ok+=float((w*(d.argmax(-1)==(a[:,0]+1))).sum())
            turn_ok+=float((w*(t.argmax(-1)==(a[:,1].sign()+1))).sum())
            mae+=float((w*(mag.sigmoid()-a[:,1].abs()).abs()*(a[:,1].abs()>1e-4)).sum());total+=float(w.sum())
    model.train()
    return dict(fire_precision=tp/max(1,tp+fp),fire_recall=tp/max(1,tp+fn),fire_positive=positive,ready=ready_count,
        fire_predicted=tp+fp,drive_accuracy=drive_ok/max(1,total),turn_sign_accuracy=turn_ok/max(1,total),turn_magnitude_mae=mae/max(1,total))

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=3600);p.add_argument('--steps',type=int,default=30000);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--seed',type=int,default=11);p.add_argument('--fire-weight',type=float,default=4);p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',type=Path)
    p.add_argument('--validation-data',type=Path,nargs='+')
    p.add_argument('--dagger-data',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(a.seed)
    start=time.monotonic();data=load_data(a.data);train=data[False];val=data[True];n=len(train[0])
    if a.validation_data:val=load_data(a.validation_data)[True]
    dagger=load_data([a.dagger_data],require_both=False)[False] if a.dagger_data else None
    if not torch.isfinite(train[0]).all():raise ValueError('Nonfinite training input')
    model=Policy().to(a.device)
    if a.resume:model.load_state_dict(torch.load(a.resume,map_location=a.device,weights_only=False)['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);writer=SummaryWriter(str(a.out/'tb'))
    skill_indices=torch.where(train[4])[0];combat_indices=torch.where(~train[4])[0]
    best=-1.;last_metrics={};step=0
    while step<a.steps and time.monotonic()-start<a.seconds:
        # Equal task/combat batches prevent the large teacher corpus erasing microtask skills.
        parts=[];base_batch=a.batch-a.batch//4 if dagger else a.batch
        skill_count=base_batch//2 if len(skill_indices) and len(combat_indices) else base_batch if len(skill_indices) else 0
        for indices,count in [(skill_indices,skill_count),(combat_indices,base_batch-skill_count)]:
            if len(indices):parts.append(indices[torch.randint(len(indices),(count,))])
        index=torch.cat(parts);x,actions,w,dw,_=[v[index].to(a.device) for v in train]
        if dagger:
            di=torch.randint(len(dagger[0]),(a.batch//4,));dx,da,dweights,ddw,_=[v[di].to(a.device) for v in dagger]
            x=torch.cat([x,dx]);actions=torch.cat([actions,da]);w=torch.cat([w,dweights]);dw=torch.cat([dw,ddw])
        h,drive,turn,mag=model.motion(x);sign=actions[:,1].sign().long()+1;nonzero=actions[:,1].abs()>1e-4
        motion_loss=weighted(F.cross_entropy(drive,(actions[:,0]+1).long(),reduction='none'),w*dw)
        turn_loss=weighted(F.cross_entropy(turn,sign,reduction='none'),w)
        magnitude_loss=weighted(F.binary_cross_entropy_with_logits(mag,actions[:,1].abs(),reduction='none'),w*nonzero)
        logits=model.fire_logits(x,h,actions[:,:2]);eligible=x[:,16]>.5
        fire_loss=weighted(F.binary_cross_entropy_with_logits(logits,actions[:,2],reduction='none',pos_weight=torch.tensor(a.fire_weight,device=a.device)),w*eligible)
        loss=motion_loss+turn_loss+magnitude_loss+fire_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite BC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();step+=1
        if step%50==0:
            for name,value in [('drive_loss',motion_loss),('turn_sign_loss',turn_loss),('turn_magnitude_loss',magnitude_loss),('fire_loss',fire_loss)]:writer.add_scalar('train/'+name,float(value.detach()),step)
        if step%500==0 or step==a.steps:
            metrics=validation(model,val,a.device);last_metrics=metrics
            # Selection metric is explicit; the combat/microtask gates still decide advancement.
            quality=2*metrics['fire_precision']*metrics['fire_recall']/max(1e-8,metrics['fire_precision']+metrics['fire_recall'])+.1*metrics['turn_sign_accuracy']
            payload=checkpoint(model,step=step,seed=a.seed,fire_weight=a.fire_weight,validation=metrics,export_bias_adjustment=0)
            torch.save(payload,a.out/'last.pt')
            if quality>best:best=quality;torch.save(payload,a.out/'best.pt')
            for name,value in metrics.items():writer.add_scalar('validation/'+name,value,step)
            status=dict(stage='bc',status='running',step=step,elapsed_seconds=time.monotonic()-start,train_rows=n,validation_rows=len(val[0]),**metrics)
            atomic_json(a.out/'status.json',status);print(json.dumps(status),flush=True);writer.flush()
    metrics=validation(model,val,a.device);torch.save(checkpoint(model,step=step,seed=a.seed,validation=metrics,fire_weight=a.fire_weight,export_bias_adjustment=0),a.out/'last.pt')
    if not (a.out/'best.pt').exists():torch.save(checkpoint(model,step=step,seed=a.seed,validation=metrics,fire_weight=a.fire_weight,export_bias_adjustment=0),a.out/'best.pt')
    atomic_json(a.out/'status.json',dict(stage='bc',status='complete',step=step,elapsed_seconds=time.monotonic()-start,train_rows=n,validation_rows=len(val[0]),**metrics));writer.close()

if __name__=='__main__':main()
