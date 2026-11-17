"""Asynchronous native-engine episodes with explicit recurrent state and policy-version ownership."""
import concurrent.futures as cf,time
from collections import deque
import numpy as np
import torch
from .transport import Worker
from .model import HIDDEN

def run(tasks,model,workers=32,device='cuda:0',objective='win',stochastic=False,trajectory=False,teacher=None,on_result=None,early_death=False):
    processes=[Worker() for _ in range(workers)];pending={};ready=deque();todo=iter(enumerate(tasks));results={};records={};hidden={}
    start=time.monotonic();deadline=None
    def reset(pool,w):
        try:index,task=next(todo)
        except StopIteration:return
        hidden[index]=np.zeros(HIDDEN,dtype=np.float32);records[index]=[]
        f=pool.submit(w.request,op='reset',slot=0,options={'early_death':early_death,'objective':objective},config=task)
        pending[f]=(w,index,'reset',None)
    try:
        with cf.ThreadPoolExecutor(workers) as pool:
            for w in processes:reset(pool,w)
            while pending or ready:
                if pending:
                    timeout=max(0,.002-(time.monotonic()-deadline)) if deadline else None
                    finished,_=cf.wait(pending,timeout=timeout,return_when=cf.FIRST_COMPLETED);finished|={f for f in pending if f.done()}
                    for f in finished:
                        w,index,op,transition=pending.pop(f);meta,obs=f.result()
                        if trajectory and op=='step':records[index].append(dict(**transition,reward=float(meta['reward']),active=bool(meta['active']),label=meta.get('label')))
                        if meta.get('done'):
                            summary={**tasks[index],**meta['info']};results[index]=summary
                            if on_result:on_result(summary,len(results),len(tasks))
                            hidden.pop(index);reset(pool,w)
                        else:
                            ready.append((w,index,obs))
                            if deadline is None:deadline=time.monotonic()
                if ready and (not pending or len(ready)>=32 or time.monotonic()-deadline>=.002):
                    batch=[ready.popleft() for _ in range(min(128,len(ready)))];deadline=time.monotonic() if ready else None
                    x=np.stack([o for _,_,o in batch]);h=np.stack([hidden[i] for _,i,_ in batch])[None]
                    with torch.inference_mode():
                        stored,lp,_,controls,last=model.act(torch.from_numpy(x).to(device),torch.from_numpy(h).to(device),deterministic=not stochastic)
                    controls=controls.cpu().numpy();stored=stored.cpu().numpy();lp=lp.cpu().numpy();last=last[0].cpu().numpy()
                    for k,(w,index,obs) in enumerate(batch):
                        before=hidden[index];hidden[index]=last[k].copy()
                        action=dict(drive=int(controls[k,0]),steer=float(controls[k,1]),shoot=bool(controls[k,2]))
                        transition=dict(obs=obs,hidden=before,action=stored[k].copy(),logprob=float(lp[k])) if trajectory else None
                        label_teacher=('reference' if tasks[index].get('skill') else teacher) if teacher else None
                        f=pool.submit(w.request,op='step',slot=0,action=action,label_teacher=label_teacher)
                        pending[f]=(w,index,'step',transition)
        return [results[i] for i in range(len(tasks))],records,time.monotonic()-start
    finally:
        for w in processes:w.close()

def active_episodes(records):
    """MC return includes reward earned by owned bullets after the tank dies."""
    output=[]
    for rows in records.values():
        total=0.;active=[]
        for row in reversed(rows):
            total+=row['reward']
            if row['active']:active.append({**row,'return':total})
        if active:output.append(list(reversed(active)))
    return output
