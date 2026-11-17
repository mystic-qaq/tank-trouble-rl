"""On-policy PPO, retaining ghost trajectories for posthumous projectile rewards."""
import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from model import Policy,load_policy,checkpoint
from vec_env import VecEnv
from evaluate import evaluate

def advantages(rewards,values,dones,last_value,lam=.98):
    result=torch.zeros_like(rewards);carry=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        following=last_value if t==len(rewards)-1 else values[t+1]
        continuing=1-dones[t]
        delta=rewards[t]+following*continuing-values[t]
        carry=delta+lam*continuing*carry
        result[t]=carry
    return result,result+values

def atomic_save(payload,path):
    temp=path.with_suffix(".tmp");torch.save(payload,temp);temp.replace(path)

def restore_rng(payload):
    if "torch_rng" in payload:torch.set_rng_state(payload["torch_rng"].cpu())
    if "numpy_rng" in payload:np.random.set_state(payload["numpy_rng"])
    if "python_rng" in payload:random.setstate(payload["python_rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        # torch.load(map_location='cuda') also moves the CPU ByteTensor RNG states.
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--objective",choices=["win","frag"],required=True)
    p.add_argument("--init",type=Path);p.add_argument("--resume",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1048576)
    p.add_argument("--envs",type=int,default=16);p.add_argument("--workers",type=int,default=16)
    p.add_argument("--rollout",type=int,default=256);p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--batch",type=int,default=512);p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--eval-every",type=int,default=262144);p.add_argument("--eval-episodes",type=int,default=28)
    p.add_argument("--device",default="cuda");p.add_argument("--seed",type=int,default=20261106)
    args=p.parse_args();torch.set_num_threads(4)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device);payload={}
    model,payload=load_policy(args.resume or args.init,device) if args.resume or args.init else (Policy().to(device),{})
    if args.resume and payload.get("objective")!=args.objective:raise ValueError("Resume objective mismatch")
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,eps=1e-5)
    if args.resume:optimizer.load_state_dict(payload["optimizer"])
    reference=copy.deepcopy(model).eval()
    if args.resume and 'reference_model' in payload:reference.load_state_dict(payload['reference_model'])
    for param in reference.parameters():param.requires_grad_(False)
    args.out.mkdir(parents=True,exist_ok=True)
    completed=int(payload.get("steps",0)) if args.resume else 0
    best=float(payload.get("best_score",-float("inf"))) if args.resume else -float("inf")
    start=time.monotonic();start_steps=completed;next_eval=completed+args.eval_every
    episode_log=open(args.out/"episodes.jsonl","a",buffering=1)
    # Resumes keep model/optimizer/RNG; restarted environments intentionally begin fresh episodes.
    if args.resume:
        restore_rng(payload)
    def save(name,metrics=None):
        data=checkpoint(model,stage="ppo",objective=args.objective,steps=completed,seed=args.seed,
            optimizer=optimizer.state_dict(),best_score=best,config=vars(args),metrics=metrics,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,reference_model=reference.state_dict())
        atomic_save(data,args.out/name)
    with VecEnv(args.envs,args.workers,args.objective,args.seed+completed,early_death=True) as env:
        recent=[]
        while completed<args.steps:
            T=min(args.rollout,max(1,(args.steps-completed+args.envs-1)//args.envs));N=args.envs
            observations=torch.empty((T,N,926),device=device)
            actions=torch.empty((T,N,3),device=device)
            logs=torch.empty((T,N),device=device);values=torch.empty_like(logs)
            rewards=torch.empty_like(logs);dones=torch.empty_like(logs);masks=torch.empty_like(logs)
            model.eval()
            for t in range(T):
                x=torch.from_numpy(env.observations).to(device)
                with torch.no_grad():a,lp,_,v,control=model.act(x)
                observations[t]=x;actions[t]=a;logs[t]=lp;values[t]=v
                _,results=env.step(control.cpu().numpy())
                rewards[t]=torch.tensor([r["reward"] for r in results],device=device)
                dones[t]=torch.tensor([r["done"] for r in results],device=device)
                masks[t]=torch.tensor([r["active"] for r in results],device=device)
                for r in results:
                    if r["info"]:
                        recent.append(r["info"]);episode_log.write(json.dumps(dict(steps=completed+(t+1)*N,**r["info"]))+"\n")
                recent=recent[-100:]
            with torch.no_grad():
                last=model(torch.from_numpy(env.observations).to(device))[3]
                adv,returns=advantages(rewards,values,dones,last)
            x=observations.reshape(-1,926);a=actions.reshape(-1,3)
            old_lp=logs.flatten();old_values=values.flatten();target=returns.flatten()
            active=masks.flatten();adv=adv.flatten();valid=active>0
            if valid.any():adv=(adv-adv[valid].mean())/(adv[valid].std(unbiased=False)+1e-8)
            completed+=T*N;model.train();kl_value=0.;loss_value=0.;clip_fraction=0.;stopped=False
            imitation_weight=.03*max(0.,1-completed/262144)
            for epoch in range(args.epochs):
                indices=torch.randperm(T*N,device=device)
                for offset in range(0,T*N,args.batch):
                    ix=indices[offset:offset+args.batch];mask=active[ix];denominator=mask.sum().clamp(min=1)
                    _,lp,entropy,new_value,_=model.act(x[ix],a[ix])
                    logratio=lp-old_lp[ix];ratio=logratio.exp()
                    kl=((ratio-1-logratio)*mask).sum()/denominator
                    kl_value=kl.item()
                    if kl_value>.03:stopped=True;break
                    policy_loss=(torch.maximum(-adv[ix]*ratio,-adv[ix]*ratio.clamp(.8,1.2))*mask).sum()/denominator
                    clipped_value=old_values[ix]+(new_value-old_values[ix]).clamp(-.2,.2)
                    value_loss=.5*torch.maximum((new_value-target[ix]).square(),(clipped_value-target[ix]).square()).mean()
                    entropy_loss=(entropy*mask).sum()/denominator
                    loss=policy_loss+.5*value_loss-.003*entropy_loss
                    if imitation_weight:
                        with torch.no_grad():rd,rs,rf,_=reference(x[ix])
                        md,ms,mf,_=model(x[ix])
                        imitation=(nn.functional.kl_div(md.log_softmax(-1),rd.softmax(-1),reduction="none").sum(-1)
                            +(ms.tanh()-rs.tanh()).square()+nn.functional.binary_cross_entropy_with_logits(mf,rf.sigmoid(),reduction="none"))
                        loss+=imitation_weight*(imitation*mask).sum()/denominator
                    if not torch.isfinite(loss):raise RuntimeError("Nonfinite PPO loss")
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                    loss_value=loss.item();clip_fraction=(((ratio-1).abs()>.2)*mask).sum().item()/denominator.item()
                if stopped:break
            status=dict(stage="ppo",objective=args.objective,steps=completed,loss=loss_value,kl=kl_value,
                clip_fraction=clip_fraction,active_fraction=active.mean().item(),recent_episodes=len(recent),
                recent_win_rate=float(np.mean([r["win"] for r in recent])) if recent else None,
                recent_net_kills=float(np.mean([r["netKills"] for r in recent])) if recent else None,
                frames_per_second=(completed-start_steps)/(time.monotonic()-start),elapsed=time.monotonic()-start)
            save("latest.pt",status)
            if args.eval_every>0 and (completed>=next_eval or completed>=args.steps):
                model.eval();result=evaluate(model,args.objective,args.eval_episodes,min(8,args.envs),device)
                score=result["win_rate"] if args.objective=="win" else result["net_kills"]
                (args.out/f"validation-{completed:09d}.json").write_text(json.dumps(result,indent=2))
                status["validation_score"]=score
                if score>best:best=score;save("best.pt",status)
                next_eval=completed+args.eval_every
                save("latest.pt",status)
            (args.out/"status.json").write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
    episode_log.close();save("final.pt",status)

if __name__=="__main__":main()
