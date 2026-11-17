"""Behavioral initialization, with an episode-level held-out validation split."""
import argparse
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from model import Policy,checkpoint

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch",type=int,default=1024)
    p.add_argument("--device",default="cuda")
    p.add_argument("--seed",type=int,default=20261106)
    args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    device=torch.device(args.device)
    shards=[np.load(f) for f in sorted(args.data.glob("shard-*.npz"))]
    if not shards:raise RuntimeError("No demonstrations")
    obs=np.concatenate([s["obs"] for s in shards]);actions=np.concatenate([s["actions"] for s in shards])
    validation=np.concatenate([s["validation"] for s in shards])
    if not validation.any() or validation.all():raise RuntimeError("Collect more episodes for a disjoint validation split")
    manifest=json.loads((args.data/"manifest.json").read_text())
    model=Policy().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    indices=np.flatnonzero(~validation); val_indices=np.flatnonzero(validation)
    args.out.mkdir(parents=True,exist_ok=True)
    best=float("inf")
    def loss(batch):
        x=torch.from_numpy(obs[batch]).to(device);a=torch.from_numpy(actions[batch]).float().to(device)
        drive,steer,shoot,_=model(x)
        # Balanced binary classification preserves the rare shooting examples.
        weights=torch.where(a[:,2]>0,shoot_weight,1.0)
        return nn.functional.cross_entropy(drive,(a[:,0]+1).long())+3*nn.functional.mse_loss(steer.tanh(),a[:,1])+nn.functional.binary_cross_entropy_with_logits(shoot,a[:,2],weight=weights)
    positive=float(actions[indices,2].mean())
    shoot_weight=min(20.,max(1.,(1-positive)/max(positive,1e-4)))
    for epoch in range(args.epochs):
        model.train();np.random.shuffle(indices);total=0;seen=0
        for start in range(0,len(indices),args.batch):
            batch=indices[start:start+args.batch];l=loss(batch)
            optimizer.zero_grad();l.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=l.item()*len(batch);seen+=len(batch)
        model.eval()
        with torch.no_grad():
            val=sum(loss(val_indices[i:i+args.batch]).item()*len(val_indices[i:i+args.batch]) for i in range(0,len(val_indices),args.batch))/len(val_indices)
        record=dict(stage="bc",epoch=epoch+1,loss=total/seen,validation_loss=val,teacher=manifest["teacher"],samples=len(obs))
        print(json.dumps(record),flush=True)
        (args.out/"status.json").write_text(json.dumps(record,indent=2))
        payload=checkpoint(model,stage="bc",teacher=manifest["teacher"],seed=args.seed,epoch=epoch+1,optimizer=optimizer.state_dict(),metrics=record)
        # Correct the class-weighted log odds before using the actor as a probability model.
        payload['model']={k:v.detach().clone() for k,v in payload['model'].items()}
        payload['model']['shoot.bias']-=math.log(shoot_weight)
        payload['shoot_positive_weight']=shoot_weight
        torch.save(payload,args.out/"latest.pt")
        if val<best:best=val;torch.save(payload,args.out/"best.pt")

if __name__=="__main__":main()
