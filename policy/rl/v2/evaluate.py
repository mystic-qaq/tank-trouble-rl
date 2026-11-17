"""Asynchronous CPU environments and centrally batched inference; whole-round paired evaluations."""
import argparse,concurrent.futures as cf,hashlib,json,math,time
from collections import deque
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .model import load_policy
from .transport import Worker
from .collect import atomic_json

IDS=['deepseek','glm','gpt','hy','kimi','minimax','qwen','builtin']
SKILLS=['static','moving','bank','muzzle','dodge','volley']
MAPS=[20260101,88888888]

def combat_tasks(rounds_per_count=100,seed=970001001):
    tasks=[]
    for n in range(2,9):
        for block in range(math.ceil(rounds_per_count/(2*n))):
            rng=np.random.default_rng(seed+n*100003+block)
            players=[IDS[int(rng.integers(len(IDS)))] for _ in range(n)]
            layout=int(rng.integers(1,2**30));spawn=int(rng.integers(1,2**30))
            for map_seed in MAPS:
                for seat in range(n):
                    tasks.append(dict(seed=layout,spawnSeed=spawn,mapSeed=map_seed,n=n,seat=seat,players=players,
                        block=f'{n}:{block}',case=len(tasks)))
    return tasks

def skill_tasks(count=50,seed=870002001):
    return [dict(seed=seed+i*7919,n=2,seat=0,players=['builtin','builtin'],skill=s,case=i,block=f'{s}:{i}')
        for s in SKILLS for i in range(count)]

def run_evaluation(tasks,model=None,teacher=None,workers=16,device='cpu',on_result=None,early_death=False,trajectory=False,label_teacher=None,objective='win',stochastic=None):
    """No global frame barrier. A worker receives its next action as soon as its state is ready."""
    if (model is None)==(teacher is None):raise ValueError('Provide one policy source')
    processes=[Worker() for _ in range(workers)];pending={};ready=deque();todo=iter(enumerate(tasks));results={};records={}
    start=time.monotonic();deadline=None
    def reset(pool,w):
        try:index,task=next(todo)
        except StopIteration:return
        records[index]=[]
        f=pool.submit(w.request,op='reset',slot=0,options={'early_death':early_death,'objective':objective},config=task)
        pending[f]=(w,index,'reset',None)
    try:
        with cf.ThreadPoolExecutor(workers) as pool:
            for w in processes:reset(pool,w)
            while pending or ready:
                if pending:
                    wait=max(0,.002-(time.monotonic()-deadline)) if deadline else None
                    finished,_=cf.wait(pending,timeout=wait,return_when=cf.FIRST_COMPLETED)
                    # Drain every currently completed result into the same inference batch.
                    finished|={f for f in pending if f.done()}
                    for f in finished:
                        w,index,op,transition=pending.pop(f);meta,obs=f.result()
                        if trajectory and op=='step':
                            records[index].append((*transition,float(meta['reward']),bool(meta['active']),meta.get('label')))
                        if meta.get('done'):
                            summary={**tasks[index],**meta['info']};results[index]=summary
                            if on_result:on_result(summary,len(results),len(tasks))
                            reset(pool,w)
                        else:
                            ready.append((w,index,obs))
                            if deadline is None:deadline=time.monotonic()
                if ready and (not pending or len(ready)>=32 or time.monotonic()-deadline>=.002):
                    batch=[ready.popleft() for _ in range(min(128,len(ready)))];deadline=time.monotonic() if ready else None
                    if model is not None:
                        observations=np.stack([o for _,_,o in batch])
                        with torch.inference_mode():
                            stored,lp,_,controls=model.act(torch.from_numpy(observations).to(device),deterministic=not (trajectory if stochastic is None else stochastic))
                        controls=controls.cpu().numpy();stored=stored.cpu().numpy();lp=lp.cpu().numpy()
                    for k,(w,index,obs) in enumerate(batch):
                        action={'drive':int(controls[k,0]),'steer':float(controls[k,1]),'shoot':bool(controls[k,2])} if model is not None else {'drive':0,'steer':0,'shoot':False}
                        transition=(obs,stored[k],float(lp[k])) if trajectory else None
                        f=pool.submit(w.request,op='step',slot=0,action=action,teacher=teacher,label_teacher=label_teacher)
                        pending[f]=(w,index,'step',transition)
        return [results[i] for i in range(len(tasks))],records,time.monotonic()-start
    finally:
        for w in processes:w.close()

def aggregate(rows):
    summary={}
    for group in sorted({r.get('skill') or str(r['n']) for r in rows}):
        part=[r for r in rows if (r.get('skill') or str(r['n']))==group]
        summary[group]={k:float(np.mean([r[k] for r in part])) for k in ['win','netKills','kills','suicides']}
        summary[group].update(rounds=len(part),emissions=float(np.mean([r['events']['emissions'] for r in part])),
            fizzles=float(np.mean([r['events']['fizzles'] for r in part])))
        if part[0].get('skill'):summary[group]['success']=float(np.mean([r['skill_success'] for r in part]))
    return summary

def paired_gate(student,reference,objective,seed=1106):
    if len(student)!=len(reference):raise ValueError('Mismatched paired results')
    metric='win' if objective=='win' else 'netKills';clusters={};counts={}
    for s,t in zip(student,reference):
        for field in ['case','seed','spawnSeed','mapSeed','seat','opponents']:
            if s[field]!=t[field]:raise ValueError('Unpaired evaluation '+field)
        n=s['n'];block=s['block'];clusters.setdefault(n,{}).setdefault(block,[]).append(float(s[metric]-.8*t[metric]))
        counts[n]=counts.get(n,0)+1
    rng=np.random.default_rng(seed);values=[]
    for _ in range(4000):
        by_count=[]
        for n,blocks in clusters.items():
            means=np.array([np.mean(v) for v in blocks.values()]);by_count.append(float(rng.choice(means,len(means),replace=True).mean()))
        values.append(np.mean(by_count))
    low,high=np.quantile(values,[.025,.975]);reference_mean=np.mean([np.mean([r[metric] for r in reference if r['n']==n]) for n in range(2,9)])
    student_mean=np.mean([np.mean([r[metric] for r in student if r['n']==n]) for n in range(2,9)])
    if objective=='frag' and reference_mean<=0:low=-1 # No relative gate against a nonpositive teacher baseline.
    return dict(passed=bool(low>=0 and min(counts.values())>=100 and sum(r['events']['hits'] for r in student)>0),
        metric=metric,student_mean=student_mean,teacher_mean=reference_mean,student_minus_point8_teacher_ci95=[float(low),float(high)],counts=counts,
        bootstrap_unit='paired_layout_block_uniform_player_count',nonpositive_reference_requires_absolute_review=reference_mean<=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path);p.add_argument('--teacher',choices=['hy','gpt','reference'])
    p.add_argument('--suite',choices=['skills','combat'],required=True);p.add_argument('--count',type=int,default=50)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    model,_=load_policy(a.model,a.device) if a.model else (None,None)
    if model:model.eval()
    tasks=skill_tasks(a.count) if a.suite=='skills' else combat_tasks(a.count);writer=SummaryWriter(str(a.out/'tb'));completed=[]
    def progress(row,count,total):
        completed.append(row);atomic_json(a.out/'status.json',dict(stage='evaluate_'+a.suite,status='running',complete=count,total=total))
        for name in ['win','netKills','kills','suicides']:writer.add_scalar('eval/'+name,float(np.mean([r[name] for r in completed])),count)
        writer.add_scalar('eval/actual_emissions',float(np.mean([r['events']['emissions'] for r in completed])),count)
    rows,_,seconds=run_evaluation(tasks,model,a.teacher,a.workers,a.device,progress)
    atomic_json(a.out/'results.json',dict(rows=rows,aggregate=aggregate(rows),elapsed_seconds=seconds,suite=a.suite,
        model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest() if a.model else None,teacher=a.teacher))
    atomic_json(a.out/'status.json',dict(stage='evaluate_'+a.suite,status='complete',complete=len(rows),total=len(rows),elapsed_seconds=seconds));writer.close()
    print(json.dumps(aggregate(rows)),flush=True)

if __name__=='__main__':main()
