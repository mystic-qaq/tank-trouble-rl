"""V2 bounded PPO: complete agent episodes, undiscounted MC, separate actor/critic."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from .model import load_policy,Critic,checkpoint
from .evaluate import run_evaluation,IDS,MAPS
from .collect import atomic_json
from .train_bc import load_data,weighted

def monte_carlo(records):
    rows=[]
    for episode in records.values():
        total=0.;backward=[]
        for obs,action,lp,reward,active,label in reversed(episode):
            total+=reward
            if active:backward.append((obs,action,lp,total))
        rows.extend(reversed(backward))
    if not rows:raise ValueError('No active transitions')
    return tuple(torch.from_numpy(np.asarray([r[i] for r in rows],dtype=np.float32)) for i in range(4))

def fresh_tasks(seed,version,count=64):
    rng=np.random.default_rng(np.random.SeedSequence([seed,version & 0xffffffff]));tasks=[]
    for index in range(count):
        n=2+index%7
        tasks.append(dict(seed=int(rng.integers(1,2**30)),spawnSeed=int(rng.integers(1,2**30)),mapSeed=MAPS[index%2],
            n=n,seat=int(rng.integers(n)),players=[IDS[int(rng.integers(len(IDS)))] for _ in range(n)],case=index,block=f'{version}:{index}'))
    return tasks

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--gates',type=Path,required=True)
    p.add_argument('--objective',choices=['win','frag'],required=True);p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--frames',type=int,default=200000);p.add_argument('--seconds',type=int,default=3600)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();gate=json.loads(a.gates.read_text());digest=hashlib.sha256(a.model.read_bytes()).hexdigest()
    if not gate['ppo_allowed'] or gate['model_sha256']!=digest:raise ValueError('G0–G3 not passed for this exact checkpoint')
    if gate['objective']!=a.objective:raise ValueError('Objective/gate mismatch')
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'last.pt').exists():raise ValueError('Fresh short-trial directory required')
    torch.set_num_threads(1);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);start=time.monotonic()
    actor,_=load_policy(a.model,a.device);critic=Critic().to(a.device)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=5e-5);critic_opt=torch.optim.Adam(critic.parameters(),lr=3e-4)
    teacher_data=load_data(a.data)[False];writer=SummaryWriter(str(a.out/'tb'));frames=0;version=0
    # Warm up on separate current-student episodes. These transitions never enter the PPO update.
    actor.eval()
    warm_count=min(32,a.frames//7200)
    if warm_count<1:raise ValueError('Trial budget must allow complete warmup episodes')
    warm_rows,warm_records,_=run_evaluation(fresh_tasks(a.seed,-1,warm_count),actor,workers=a.workers,device=a.device,
        early_death=True,trajectory=True,objective=a.objective)
    frames=sum(r['frames'] for r in warm_rows);warmup_frames=frames
    warm=monte_carlo(warm_records);warm_x,warm_y=warm[0],warm[3]
    for _ in range(200):
        index=torch.randint(len(warm_x),(min(1024,len(warm_x)),));pred=critic(warm_x[index].to(a.device))
        loss=F.mse_loss(pred,warm_y[index].to(a.device));critic_opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),1);critic_opt.step()
    del warm_records,warm,warm_x,warm_y
    metrics=dict(stage='critic_warmup',status='complete',frames=frames,warmup_frames=warmup_frames,elapsed_seconds=time.monotonic()-start)
    atomic_json(a.out/'status.json',metrics)
    while a.frames-frames>=3600 and time.monotonic()-start<a.seconds:
        actor.eval();version+=1
        episode_count=min(64,(a.frames-frames)//3600)
        rows,records,_=run_evaluation(fresh_tasks(a.seed,version,episode_count),actor,workers=a.workers,device=a.device,
            early_death=True,trajectory=True,objective=a.objective)
        x,actions,old_lp,returns=monte_carlo(records);del records
        frames+=sum(row['frames'] for row in rows)
        # Exact policy-version barrier: nothing updates while any rollout is outstanding.
        old_value=[];max_error=0.
        with torch.inference_mode():
            for offset in range(0,len(x),1024):
                bx=x[offset:offset+1024].to(a.device);ba=actions[offset:offset+1024].to(a.device)
                replay=actor.act(bx,ba)[1];max_error=max(max_error,float((replay.cpu()-old_lp[offset:offset+1024]).abs().max()))
                old_value.append(critic(bx).cpu())
        if max_error>2e-4:raise RuntimeError('Stale or inconsistent old log-probabilities '+str(max_error))
        advantage=returns-torch.cat(old_value);advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp(min=1e-6)
        actor.train();critic.train();kl=0.;updates=0
        for epoch in range(2):
            for index in torch.randperm(len(x)).split(1024):
                bx=x[index].to(a.device);ba=actions[index].to(a.device);adv=advantage[index].to(a.device)
                _,lp,entropy,_=actor.act(bx,ba);ratio=(lp-old_lp[index].to(a.device)).exp()
                actor_loss=-torch.minimum(ratio*adv,ratio.clamp(.95,1.05)*adv).mean()
                ent_bonus=(entropy*torch.tensor([.001,.001,.0001,.001],device=a.device)).sum(-1).mean()
                teacher_index=torch.randint(len(teacher_data[0]),(256,));tx,ta,tw,tdw,_=[t[teacher_index].to(a.device) for t in teacher_data]
                h,d,t,mag=actor.motion(tx)
                auxiliary=weighted(F.cross_entropy(d,(ta[:,0]+1).long(),reduction='none'),tw*tdw)
                auxiliary+=weighted(F.cross_entropy(t,(ta[:,1].sign()+1).long(),reduction='none'),tw)
                auxiliary+=weighted(F.binary_cross_entropy_with_logits(mag,ta[:,1].abs(),reduction='none'),tw*(ta[:,1].abs()>1e-4))
                auxiliary+=weighted(F.binary_cross_entropy_with_logits(actor.fire_logits(tx,h,ta[:,:2]),ta[:,2],reduction='none'),tw*(tx[:,16]>.5))
                total=actor_loss-ent_bonus+.05*auxiliary
                if not torch.isfinite(total):raise RuntimeError('Nonfinite PPO objective')
                actor_opt.zero_grad(set_to_none=True);total.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);actor_opt.step()
                value_loss=F.mse_loss(critic(bx),returns[index].to(a.device))
                critic_opt.zero_grad(set_to_none=True);value_loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1);critic_opt.step()
                with torch.no_grad():batch_kl=float(((ratio-1)-(lp-old_lp[index].to(a.device))).mean())
                kl+=batch_kl;updates+=1
                if batch_kl>.015:break
            if batch_kl>.015:break
        metrics=dict(stage='ppo_short_trial',status='running',objective=a.objective,frames=frames,version=version,active_samples=len(x),
            policy_replay_max_error=max_error,warmup_frames=warmup_frames,mean_return=float(np.mean([r['return'] for r in rows])),
            actual_emissions=sum(r['events']['emissions'] for r in rows),hits=sum(r['events']['hits'] for r in rows),suicides=sum(r['suicides'] for r in rows),
            approximate_kl=kl/max(1,updates),value_loss=float(value_loss.detach()),elapsed_seconds=time.monotonic()-start)
        for name in ['mean_return','actual_emissions','hits','suicides','approximate_kl','value_loss']:writer.add_scalar('train/'+name,metrics[name],frames)
        torch.save(checkpoint(actor,frames=frames,version=version,seed=a.seed,objective=a.objective,critic=critic.state_dict(),
            gamma=1.,gae_lambda=1.,actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),source_bc_sha256=digest),a.out/'last.pt')
        atomic_json(a.out/'status.json',metrics);writer.flush();print(json.dumps(metrics),flush=True)
        if metrics['actual_emissions']==0 or metrics['hits']==0:
            metrics.update(status='stopped_no_effective_attack');atomic_json(a.out/'status.json',metrics);break
    if a.frames-frames<3600 or time.monotonic()-start>=a.seconds:
        metrics.update(status='budget_complete');atomic_json(a.out/'status.json',metrics)
    writer.close()

if __name__=='__main__':main()
