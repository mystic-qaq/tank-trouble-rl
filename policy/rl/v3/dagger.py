import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import load_policy
from .rollout import run
from ..v2.evaluate import combat_tasks,skill_tasks
from ..v2.collect import atomic_json,is_validation

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--teacher',choices=['hy','gpt'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=32)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--skill-count',type=int,default=12);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'trajectories.jsonl').exists():raise ValueError('Fresh DAgger directory required')
    actor,_=load_policy(a.model,a.device);actor.eval();saved=frames=positives=0
    tasks=combat_tasks(10,a.seed)+skill_tasks(a.skill_count,a.seed+900001)
    def progress(row,count,total):atomic_json(a.out/'status.json',dict(stage='dagger_collect',status='running',complete=count,total=total))
    rows,records,elapsed=run(tasks,actor,a.workers,a.device,trajectory=True,teacher=a.teacher,on_result=progress,early_death=True)
    with open(a.out/'trajectories.jsonl','w') as file:
        for i,summary in enumerate(rows):
            active=[r for r in records[i] if r['active'] and r['label'] is not None]
            if not active:continue
            matrix=np.concatenate([np.stack([r['obs'] for r in active]),np.asarray([r['label'] for r in active],dtype=np.float32)],axis=1)
            name=f'trajectory-{saved:05d}.npy';np.save(a.out/name,matrix,allow_pickle=False)
            r=dict(file=name,length=len(matrix),teacher='reference' if summary.get('skill') else a.teacher,quality=1,source='dagger',
                seed=summary['seed'],n=summary['n'],skill=summary.get('skill') or 'combat',validation=is_validation(summary['seed']),physics_hz=60)
            file.write(json.dumps(r)+'\n');saved+=1;frames+=len(active);positives+=int(matrix[:,-1].sum())
    atomic_json(a.out/'combat_observation.json',dict(rows=rows,warning='training states and early agent termination; not independent full-round acceptance'))
    status=dict(stage='dagger_collect',status='complete',trajectories=saved,frames=frames,positive_counterfactual_labels=positives,elapsed_seconds=elapsed)
    atomic_json(a.out/'status.json',status);atomic_json(a.out/'manifest.json',status);print(json.dumps(status),flush=True)

if __name__=='__main__':main()
