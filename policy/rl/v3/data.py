"""Memory-mapped full-rate sequences. Burn-in frames never masquerade as shuffled time."""
import json,math
from collections import OrderedDict,Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class Sequences(Dataset):
    def __init__(self,directories,validation=False,learn=64,burn=128,length=1000000,seed=11):
        self.records=[];self.learn=learn;self.burn=burn;self.length=length;self.seed=seed;self.cache=OrderedDict();self.rng=None
        for directory in directories:
            directory=Path(directory)
            with open(directory/'trajectories.jsonl') as file:
                for line in file:
                    r=json.loads(line)
                    if bool(r['validation'])==validation and r['length']>=2:self.records.append({**r,'path':directory/r['file']})
        if not self.records:raise ValueError('No continuous sequences for split')
        teacher_counts=Counter(r['teacher'] for r in self.records);count_counts=Counter(r['n'] for r in self.records)
        weights=[]
        for r in self.records:
            q=float(r.get('quality',1))
            if r.get('source')=='skill':q*=.4
            if r.get('win') and r.get('emissions',1)==0:q*=.2
            # Diversity without making rare weak teachers as dominant as successful strong teachers.
            weights.append(q/math.sqrt(teacher_counts[r['teacher']])/math.sqrt(count_counts[r['n']]))
        self.probability=np.asarray(weights)/sum(weights)
        sources={r.get('source','multi_teacher') for r in self.records}
        masses={'multi_teacher':.25,'skill':.15,'dagger':.60} if 'dagger' in sources else {'multi_teacher':.8,'skill':.2}
        probability=np.zeros(len(weights),dtype=np.float64)
        for source in sources:
            selected=np.array([r.get('source','multi_teacher')==source for r in self.records])
            probability[selected]=self.probability[selected]/self.probability[selected].sum()*masses.get(source,.2)
        self.probability=probability/probability.sum()
    def __len__(self):return self.length
    def array(self,index):
        if index in self.cache:self.cache.move_to_end(index);return self.cache[index]
        data=np.load(self.records[index]['path'],mmap_mode='r');self.cache[index]=data
        if len(self.cache)>32:self.cache.popitem(last=False)
        return data
    def __getitem__(self,index):
        if self.rng is None:
            worker=torch.utils.data.get_worker_info();self.rng=np.random.default_rng(self.seed+(worker.id+1)*100003 if worker else self.seed)
        chosen=int(self.rng.choice(len(self.records),p=self.probability));r=self.records[chosen];array=self.array(chosen)
        start=int(self.rng.integers(max(1,len(array)-self.learn+1)));lo=max(0,start-self.burn);hi=min(len(array),start+self.learn)
        time=self.burn+self.learn;obs=np.zeros((time,1966),dtype=np.float32);actions=np.zeros((time,3),dtype=np.float32);mask=np.zeros(time,dtype=np.float32)
        part=array[lo:hi];obs[:len(part)]=part[:,:1966];actions[:len(part)]=part[:,1966:];mask[start-lo:len(part)]=1
        return torch.from_numpy(obs),torch.from_numpy(actions),torch.from_numpy(mask),torch.tensor(1.,dtype=torch.float32)

def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
