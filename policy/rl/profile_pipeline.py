"""Short separate probe; leaves existing experiments and their RNG/checkpoints untouched."""
import argparse
import json
from pathlib import Path
import time
import torch
import numpy as np
from model import load_policy
from vec_env import VecEnv

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--envs',type=int,default=16);p.add_argument('--batches',type=int,default=64)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    torch.set_num_threads(4);model,_=load_policy(args.checkpoint,'cuda');model.eval()
    with VecEnv(args.envs,args.envs,'win',20261109,early_death=True) as env:
        x=torch.from_numpy(env.observations).cuda()
        with torch.no_grad():
            for _ in range(20):model.act(x)
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(200):model.act(x)[4].cpu().numpy()
            torch.cuda.synchronize();inference=(time.perf_counter()-start)/200
        control=np.zeros((args.envs,3),dtype=np.float32)
        start=time.perf_counter()
        for _ in range(args.batches):env.step(control)
        environment=(time.perf_counter()-start)/args.batches
        x=torch.from_numpy(np.tile(env.observations,(32,1))).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(32):
            _,lp,entropy,value,_=model.act(x)
            loss=-lp.mean()+.1*value.square().mean()-.003*entropy.mean()
            opt.zero_grad();loss.backward();opt.step()
        torch.cuda.synchronize();optimizer=(time.perf_counter()-start)/32
    result={'parameters':sum(p.numel() for p in model.parameters()),'num_envs':args.envs,
        'inference_and_action_copy_ms':inference*1000,'environment_and_transport_ms':environment*1000,
        'optimizer_minibatch_ms':optimizer*1000,'optimizer_minibatch_size':32*args.envs,
        'environment_fraction_of_sampling_probe':environment/(environment+inference),
        'note':'Separate short probe under current server contention; not exact production phase percentages.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
