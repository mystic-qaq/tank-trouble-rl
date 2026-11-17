"""Full-rate, continuous, multi-teacher quality trajectories. No architecture comparisons."""
import argparse,concurrent.futures as cf,json,time
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .transport import Worker
from ..v2.collect import atomic_json,is_validation
from ..v2.evaluate import IDS,MAPS,SKILLS

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=80)
    p.add_argument('--worlds',type=int,default=6000);p.add_argument('--skills',action='store_true');p.add_argument('--seconds',type=int,default=5400)
    p.add_argument('--seed',type=int,default=710261106);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'trajectories.jsonl').exists():raise ValueError('Fresh data directory required')
    writer=SummaryWriter(str(a.out/'tb'));workers=[Worker() for _ in range(a.workers)];pending={};submitted=done=frames=trajectories=bytes_written=0
    start=time.monotonic();by_teacher={};shots=0;stop=False
    def submit(pool,w):
        nonlocal submitted
        index=submitted;submitted+=1;seed=a.seed+index;n=2+index%7
        rng=np.random.default_rng(seed)
        config=dict(seed=seed,n=n,seat=0,mapSeed=MAPS[(index//7)%2],players=[IDS[int(rng.integers(len(IDS)))] for _ in range(n)])
        if a.skills:config.update(n=2,seat=0,skill=SKILLS[index%6],players=['builtin','builtin'])
        future=pool.submit(w.request,op='collect_skill' if a.skills else 'collect_world',config=config);pending[future]=(w,index,config)
    try:
        with open(a.out/'trajectories.jsonl','w') as manifest,open(a.out/'worlds.jsonl','w') as worlds,cf.ThreadPoolExecutor(a.workers) as pool:
            for w in workers:
                if submitted<a.worlds:submit(pool,w)
            while pending:
                ready,_=cf.wait(pending,timeout=2,return_when=cf.FIRST_COMPLETED)
                for f in ready:
                    w,index,config=pending.pop(f);header,data=f.result();matrix=data.reshape(-1,1969)
                    if not np.isfinite(matrix).all():raise ValueError('Nonfinite continuous data')
                    for j,t in enumerate(header['trajectories']):
                        name=f'world-{index:06d}-seat-{t["seat"]}.npy';part=matrix[t['offset']:t['offset']+t['length']]
                        if not len(part):continue
                        np.save(a.out/name,part,allow_pickle=False);bytes_written+=(a.out/name).stat().st_size
                        record={**t,'file':name,'seed':config['seed'],'n':config['n'],'mapSeed':config['mapSeed'],'validation':is_validation(config['seed']),
                            'skill':config.get('skill','combat'),'source':'skill' if a.skills else 'multi_teacher','physics_hz':60}
                        manifest.write(json.dumps(record)+'\n');trajectories+=1;by_teacher[t['teacher']]=by_teacher.get(t['teacher'],0)+1
                        shots+=int(part[:,-1].sum())
                    manifest.flush();worlds.write(json.dumps(dict(index=index,**header['summary'],selected=len(header['trajectories']),replay_identical=header['replay_identical']))+'\n');worlds.flush()
                    done+=1;frames+=header['summary']['frames']
                    if submitted<a.worlds and not stop:submit(pool,w)
                elapsed=time.monotonic()-start;stop=elapsed>a.seconds or bytes_written>140*1024**3
                status=dict(stage='continuous_collection',status='running',worlds=done,trajectories=trajectories,world_frames=frames,
                    actual_selected_emissions=shots,bytes=bytes_written,by_teacher=by_teacher,elapsed_seconds=elapsed,frames_per_second=frames/max(1,elapsed))
                atomic_json(a.out/'status.json',status)
                for k in ['worlds','trajectories','actual_selected_emissions','frames_per_second']:writer.add_scalar('collect/'+k,status[k],frames)
                writer.flush()
        status.update(status='complete' if done>=a.worlds else 'budget_exhausted');atomic_json(a.out/'status.json',status)
        atomic_json(a.out/'manifest.json',dict(**status,observation_version=2,continuous=True,split='whole_world_blake2b',seed=a.seed));print(json.dumps(status),flush=True)
    finally:
        for w in workers:w.close()
        writer.close()

if __name__=='__main__':main()
