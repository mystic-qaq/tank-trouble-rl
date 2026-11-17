"""Budgeted, event-aware demonstrations. Whole-layout split and exact inclusion probabilities."""
import argparse,concurrent.futures as cf,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from .transport import Worker

def atomic_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def is_validation(seed):
    return int.from_bytes(hashlib.blake2b(str(seed).encode(),digest_size=8).digest(),'little')%10==0

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',choices=['hy','gpt','reference'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--episodes',type=int,default=100000);p.add_argument('--min-layouts',type=int,default=1000)
    p.add_argument('--min-shots',type=int,default=20000);p.add_argument('--seconds',type=int,default=7200)
    p.add_argument('--seed',type=int,default=620261106);p.add_argument('--skills',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if list(a.out.glob('episode-*.npz')):raise RuntimeError('Fresh output directory required')
    if a.skills and a.teacher!='reference':raise ValueError('Skill data requires reference teacher')
    start=time.monotonic();writer=SummaryWriter(str(a.out/'tb'));workers=[Worker() for _ in range(a.workers)]
    pending={};index=0;completed=0;samples=0;frames=0;shots=0;hits=0;suicides=0;fizzles=0
    skills=['static','moving','bank','muzzle','dodge','volley'];coverage={s:0 for s in skills};firepos=0;valpos=0
    status={};stopping=False
    def submit(pool,w):
        nonlocal index
        # The seed, not a frame or randomly shuffled row, owns the split.
        seed=a.seed+index;config={'seed':seed,'n':2+(index%7),'seat':(index//7)%(2+index%7)}
        if a.skills:config.update(n=2,seat=0,skill=skills[index%6],players=['builtin','builtin'])
        future=pool.submit(w.request,op='collect',teacher=a.teacher,config=config)
        pending[future]=(w,index,config);index+=1
    try:
        with gzip.open(a.out/'events.jsonl.gz','wt') as eventfile,open(a.out/'episodes.jsonl','w') as summaries,cf.ThreadPoolExecutor(a.workers) as pool:
            for w in workers:
                if index<a.episodes:submit(pool,w)
            while pending:
                finished,_=cf.wait(pending,timeout=2,return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    w,episode,config=pending.pop(future);meta,data=future.result();summary=meta['summary'];rows=meta['rows']
                    matrix=data.reshape(-1,1969);validation=is_validation(config['seed'])
                    np.savez_compressed(a.out/f'episode-{episode:06d}.npz',obs=matrix[:,:1966],actions=matrix[:,1966:],
                        probability=np.array([r['p'] for r in rows],dtype=np.float32),eligible=np.array([r['eligible'] for r in rows],dtype=bool),
                        validation=np.array(validation),seed=np.array(config['seed']),skill=np.array(config.get('skill','combat')))
                    for event in meta['events']:eventfile.write(json.dumps({'episode':episode,'seed':config['seed'],**event})+'\n')
                    summaries.write(json.dumps({'episode':episode,'validation':validation,**summary})+'\n');summaries.flush()
                    completed+=1;samples+=len(rows);frames+=summary['frames'];ev=summary['events']
                    shots+=ev['emissions'];hits+=ev['hits'];suicides+=ev['self_hits'];fizzles+=ev['fizzles']
                    firepos+=int(matrix[:,-1].sum());valpos+=int(matrix[:,-1].sum())*validation
                    if a.skills:coverage[config['skill']]+=1
                    if index<a.episodes and not stopping:submit(pool,w)
                elapsed=time.monotonic()-start
                target=completed>=a.min_layouts and shots>=a.min_shots
                stopping=stopping or elapsed>=a.seconds or target
                status=dict(stage='collect',teacher=a.teacher,episodes=completed,unique_layouts=completed,samples=samples,frames=frames,
                    actual_emissions=shots,hits=hits,suicides=suicides,fizzles=fizzles,fire_positive_samples=firepos,validation_fire_positive_samples=valpos,
                    elapsed_seconds=elapsed,frames_per_second=frames/max(1,elapsed),skill_coverage=coverage,budget_seconds=a.seconds,
                    coverage_target_met=target,status='draining' if stopping else 'running')
                atomic_json(a.out/'status.json',status)
                for name in ['episodes','samples','actual_emissions','hits','suicides','fizzles','validation_fire_positive_samples','frames_per_second']:
                    writer.add_scalar('collect/'+name,status[name],frames)
                writer.flush()
        status.update(status='complete' if target else 'budget_exhausted',elapsed_seconds=time.monotonic()-start)
        atomic_json(a.out/'status.json',status)
        atomic_json(a.out/'manifest.json',dict(**status,observation_version=2,split='whole_layout_blake2b_mod_10',seed=a.seed,
            sampling='all_requests_12pct_eligible_negatives_2.5pct_other',max_round_seconds=60))
        print(json.dumps(status),flush=True)
    finally:
        for w in workers:w.close()
        writer.close()

if __name__=='__main__':main()
