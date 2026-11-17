"""Replay trained recurrent policies, rather than only weakly initialized test networks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import load_policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);actor,_=load_policy(a.model,'cuda:0');actor.eval();sequences=[]
    for line in (a.data/'trajectories.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['length']>=96 and r['skill']=='combat':sequences.append(np.array(np.load(a.data/r['file'],mmap_mode='r')[:96,:1966]))
        if len(sequences)==4:break
    x=torch.from_numpy(np.stack(sequences)).cuda();torch.manual_seed(8);h=actor.initial(len(x));actions=[];probs=[];hidden=[]
    with torch.inference_mode():
        for t in range(96):
            hidden.append(h.clone());action,lp,_,_,h=actor.act(x[:,t],h);actions.append(action);probs.append(lp)
        stored=torch.stack(actions,1);expected=torch.stack(probs,1);error=0.
        for t in range(0,96,32):
            replay=actor.evaluate_actions(x[:,t:t+32],hidden[t],stored[:,t:t+32])[0]
            error=max(error,float((replay-expected[:,t:t+32]).abs().max()))
    if error>1e-3:raise RuntimeError('Trained-policy replay still inconsistent: '+str(error))
    result=dict(passed=True,model=str(a.model),sequences=len(x),frames_per_sequence=96,max_logprob_error=error,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,guard_threshold_unchanged=1e-3)
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
