"""Developer validation on explicit held-out seeds; shared scenarios for model/teacher."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model import load_policy
from vec_env import VecEnv

def evaluate(model, objective, episodes=28, envs=8, device="cuda", seed=0x40000000, teacher=None):
    records=[]
    with VecEnv(envs,envs,objective,seed=12345) as env:
        for start in range(0,episodes,envs):
            size=min(envs,episodes-start)
            configs=[dict(seed=seed+start+i,n=2+(start+i)%7,seat=(start+i)//7%(2+(start+i)%7)) for i in range(envs)]
            env.reset(configs)
            complete=[i>=size for i in range(envs)]
            for _ in range(3601):
                if all(complete):break
                if teacher is None:
                    with torch.no_grad():
                        controls=model.act(torch.from_numpy(env.observations).to(device),deterministic=True)[4].cpu().numpy()
                else:controls=None
                _,results=env.step(controls,teacher=teacher,autoreset=False,paused=complete)
                for i,r in enumerate(results):
                    if not complete[i] and r["done"]:
                        records.append(r["info"]);complete[i]=True
            if not all(complete):raise RuntimeError("Evaluation exceeded engine time limit")
    by_n={}
    for n in range(2,9):
        rows=[r for r in records if r["n"]==n]
        if rows:by_n[n]=dict(rounds=len(rows),win_rate=float(np.mean([r["win"] for r in rows])),net_kills=float(np.mean([r["netKills"] for r in rows])))
    return dict(episodes=len(records),win_rate=float(np.mean([v["win_rate"] for v in by_n.values()])),
        net_kills=float(np.mean([v["net_kills"] for v in by_n.values()])),
        timeout_rate=float(np.mean([r["timeout"] for r in records])),by_n=by_n,records=records)

def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path)
    p.add_argument("--objective",choices=["win","frag"],required=True)
    p.add_argument("--teacher",choices=["hy","gpt"])
    p.add_argument("--episodes",type=int,default=28);p.add_argument("--envs",type=int,default=8)
    p.add_argument("--device",default="cuda");p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    if (args.checkpoint is None)==(args.teacher is None):raise ValueError("Specify one model or teacher")
    model,_=load_policy(args.checkpoint,args.device) if args.checkpoint else (None,None)
    if model:model.eval()
    result=evaluate(model,args.objective,args.episodes,args.envs,args.device,teacher=args.teacher)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="records"}),flush=True)

if __name__=="__main__":main()
