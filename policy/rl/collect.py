"""Stream teacher demonstrations, split by complete episode seed."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from vec_env import VecEnv

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--teacher",choices=["hy","gpt"],required=True)
    parser.add_argument("--samples",type=int,default=200000)
    parser.add_argument("--envs",type=int,default=16)
    parser.add_argument("--workers",type=int,default=16)
    parser.add_argument("--seed",type=int,default=20261106)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if list(args.out.glob("shard-*.npz")):
        raise RuntimeError("Use a fresh demonstration output directory")
    obs_buffer=[]; action_buffer=[]; validation_buffer=[]
    count=0; frames=0; shard=0; episodes=0; start=time.monotonic()
    def flush():
        nonlocal shard
        if not obs_buffer: return
        np.savez_compressed(args.out/f"shard-{shard:05d}.npz",obs=np.stack(obs_buffer),
            actions=np.stack(action_buffer),validation=np.asarray(validation_buffer,dtype=bool))
        shard+=1;obs_buffer.clear();action_buffer.clear();validation_buffer.clear()
    with VecEnv(args.envs,args.workers,"win" if args.teacher=="hy" else "frag",args.seed,early_death=True) as env:
        while count<args.samples:
            before=env.observations.copy()
            _,results=env.step(teacher=args.teacher)
            frames+=args.envs
            for i,r in enumerate(results):
                episodes+=int(r["done"])
                if r["active"] and count<args.samples:
                    obs_buffer.append(before[i]); action_buffer.append(r["action"])
                    validation_buffer.append(r["seed"]%10==0)
                    count+=1
            if len(obs_buffer)>=8192 or count>=args.samples:
                flush()
                status=dict(stage="collect",teacher=args.teacher,samples=count,frames=frames,episodes=episodes,
                    elapsed=time.monotonic()-start,samples_per_second=count/(time.monotonic()-start))
                (args.out/"status.json").write_text(json.dumps(status,indent=2))
                print(json.dumps(status),flush=True)
    (args.out/"manifest.json").write_text(json.dumps(dict(teacher=args.teacher,samples=count,seed=args.seed,
        observation_size=926,round_limit=60,split="episode_seed_mod_10",shards=shard),indent=2))

if __name__=="__main__":main()
