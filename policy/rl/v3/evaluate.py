import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .model import load_policy
from .rollout import run
from ..v2.evaluate import combat_tasks,skill_tasks,aggregate
from ..v2.collect import atomic_json

def score(rows,objective):
    metric='win' if objective=='win' else 'netKills'
    counts=sorted({r['n'] for r in rows});return float(np.mean([np.mean([r[metric] for r in rows if r['n']==n]) for n in counts]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workers',type=int,default=16);p.add_argument('--count',type=int,default=1)
    p.add_argument('--skills',action='store_true');p.add_argument('--objective',choices=['win','frag'],default='win');p.add_argument('--limit-tasks',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);a.out.mkdir(parents=True,exist_ok=True)
    actor,_=load_policy(a.model,a.device);actor.eval();tasks=skill_tasks(a.count) if a.skills else combat_tasks(a.count)
    if a.limit_tasks:tasks=tasks[:a.limit_tasks]
    writer=SummaryWriter(str(a.out/'tb'));observed=[]
    def progress(row,index,total):
        observed.append(row);atomic_json(a.out/'status.json',dict(stage='combat_eval' if not a.skills else 'skill_eval',status='running',complete=index,total=total))
        if not a.skills:
            for objective in ['win','frag']:writer.add_scalar('eval/'+objective,score(observed,objective),index)
            writer.add_scalar('eval/spin_fraction',sum(r['spin_frames'] for r in observed)/max(1,sum(r['active_frames'] for r in observed)),index)
    rows,_,elapsed=run(tasks,actor,a.workers,a.device,objective=a.objective,on_result=progress)
    report=dict(rows=rows,aggregate=aggregate(rows),elapsed_seconds=elapsed,win=score(rows,'win'),frag=score(rows,'frag'),
        spin_fraction=sum(r['spin_frames'] for r in rows)/max(1,sum(r['active_frames'] for r in rows)),
        emissions=sum(r['events']['emissions'] for r in rows),hits=sum(r['events']['hits'] for r in rows),suicides=sum(r['suicides'] for r in rows),
        checkpoint=str(a.model),rounds=len(rows))
    atomic_json(a.out/'results.json',report);atomic_json(a.out/'status.json',dict(stage='evaluate',status='complete',complete=len(rows),total=len(rows),elapsed_seconds=elapsed))
    writer.close();print(json.dumps({k:v for k,v in report.items() if k not in ['rows','aggregate']}),flush=True)

if __name__=='__main__':main()
