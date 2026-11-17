"""Single mainline: four-GPU shared BC -> two three-round DAgger paths -> reward training."""
import argparse,concurrent.futures as cf,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from ..v2.collect import atomic_json
from .data import atomic_save
from .selection import select_initialization
from ..v2.pipeline import run
import torch

def command(module,*args):return [sys.executable,'-m','policy.rl.v3.'+module,*args]
def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--hours',type=float,default=8);a=p.parse_args()
    a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'pipeline.json').exists():raise ValueError('Fresh V3 pipeline required')
    start=time.time();deadline=time.monotonic()+a.hours*3600
    status=dict(version=3,status='running',stage='contracts',started_unix=start,total_wallclock_hours=a.hours,
        architecture='joint_motion_gru_v3',architecture_comparisons=False,ddp_gpu_limit=4,total_active_gpu_limit=4,
        mainline='quality_multi_teacher_sequences_shared_BC_then_three_DAgger_rounds_then_reward_PPO',resume_v1_v2=False)
    atomic_json(a.run/'pipeline.json',status)
    def remaining(reserve=0):return max(1,deadline-time.monotonic()-reserve)
    try:
        run(['node','policy/verify-frozen.mjs'],a.run/'logs/contracts.log',60)
        run(['node','--experimental-vm-modules','--disable-warning=ExperimentalWarning','policy/rl/v2/test_engine.mjs'],a.run/'logs/contracts.log',60)
        run([sys.executable,'-m','pytest','policy/rl/v3/test_contracts.py','-q','-p','no:cacheprovider'],a.run/'logs/contracts.log',180)
        atomic_json(a.run/'contracts.json',dict(passed=True,recurrent_probability_replay=True,posthumous_reward=True,actor_critic_independent=True,
            source_snapshot=str(Path.cwd()),gpu_update_probe=str(a.run/'smoke/distributed-four'),note='No V2 gates are reclassified as passed'))
        status.update(stage='waiting_for_continuous_data');atomic_json(a.run/'pipeline.json',status)
        for source in ['multi','skills']:
            while not (a.run/'data'/source/'manifest.json').exists():
                if remaining()<3600:raise TimeoutError('Data did not finish within the mainline window')
                time.sleep(5)
            manifest=read(a.run/'data'/source/'manifest.json')
            if manifest['status']!='complete':raise RuntimeError('Incomplete continuous dataset '+source)
        status.update(stage='shared_four_gpu_pretrain');atomic_json(a.run/'pipeline.json',status)
        run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4','-m','policy.rl.v3.pretrain',
            '--data',a.run/'data/multi',a.run/'data/skills','--out',a.run/'shared','--steps','30000','--seconds','5400',
            '--batch','8','--eval-device','cuda:0','--eval-workers','16','--evaluate-every','10000'],a.run/'logs/shared-pretrain.log',min(6300,remaining(3600)))
        status.update(stage='two_objective_training');atomic_json(a.run/'pipeline.json',status)
        def objective(job):
            name,teacher,gpu,evalgpu,seed=job;out=a.run/name;out.mkdir(exist_ok=True);source=a.run/'shared/best.pt';data=[a.run/'data/multi',a.run/'data/skills']
            def execute(module,*args,cap=3600,reserve=900):run(command(module,*args),a.run/'logs'/f'{name}.log',min(cap,remaining(reserve)))
            history=[]
            try:
                for iteration in range(1,4):
                    dagger=out/f'dagger-{iteration}'
                    execute('dagger','--model',source,'--teacher',teacher,'--out',dagger,'--device',f'cuda:{gpu}','--workers','32',
                        '--seed',str(750261106+seed*10000+iteration*100003))
                    data.append(dagger);training=out/f'adapt-{iteration}'
                    execute('pretrain','--resume',source,'--data',*data,'--out',training,'--steps','5000','--seconds','600','--batch','8',
                        '--device',f'cuda:{gpu}','--eval-device',f'cuda:{evalgpu}','--eval-workers','16','--evaluate-every','5000','--objective',name,'--seed',str(seed+iteration),cap=1800)
                    source=training/'best.pt';report=read(training/'status.json');history.append(report)
                    atomic_json(out/'status.json',dict(stage='dagger_adaptation',status='running',round=iteration,total_rounds=3,history=history))
                chosen,candidates=select_initialization(a.run,name)
                atomic_json(out/'initialization.json',dict(usable_for_reward_training=chosen is not None,selected=chosen,candidates=candidates,
                    criterion='best measured usable checkpoint across the curriculum, not necessarily its final round'))
                if chosen is None:
                    atomic_json(out/'status.json',dict(stage='initialization',status='stopped_no_usable_behavior',history=history));return dict(objective=name,status='stopped_no_usable_behavior')
                source=Path(chosen['path'])
                execute('train_rl','--model',source,'--data',*data,'--out',out/'rl','--objective',name,'--device',f'cuda:{gpu}',
                    '--eval-device',f'cuda:{evalgpu}','--workers','32','--eval-workers','16','--frames','16000000',
                    '--seconds',str(int(min(21600,remaining(1800)))),'--seed',str(seed),'--eval-every','500000',cap=remaining(900))
                execute('evaluate','--model',out/'rl/best.pt','--out',out/'acceptance','--objective',name,'--device',f'cuda:{gpu}',
                    '--workers','16','--count','100',cap=remaining(),reserve=0)
                report=read(out/'acceptance/results.json');reference=read(Path('/home/xy/rl_for_fun/runs/v2b/references')/name/'results.json')
                # Same native engine/scenarios. Resample whole paired layout blocks, not frames.
                metric='win' if name=='win' else 'netKills';baseline={(r['n'],r['block'],r['mapSeed'],r['seat']):r for r in reference['rows']}
                clusters={}
                for r in report['rows']:
                    t=baseline[(r['n'],r['block'],r['mapSeed'],r['seat'])]
                    for field in ['seed','spawnSeed','opponents']:
                        if r[field]!=t[field]:raise RuntimeError('Acceptance pairing mismatch')
                    clusters.setdefault(r['n'],{}).setdefault(r['block'],[]).append(r[metric]-t[metric])
                import numpy as np
                rng=np.random.default_rng(seed);samples=[]
                for _ in range(4000):
                    means=[]
                    for n,blocks in clusters.items():
                        values=np.array([np.mean(v) for v in blocks.values()]);means.append(rng.choice(values,len(values),replace=True).mean())
                    samples.append(np.mean(means))
                low,high=np.quantile(samples,[.025,.975]);difference=float(np.mean([np.mean([np.mean(v) for v in blocks.values()]) for blocks in clusters.values()]))
                threshold=.03 if name=='win' else .10
                acceptance=dict(objective=name,score=report[name],teacher_difference=difference,difference_ci95=[float(low),float(high)],
                    target_margin=threshold,beats_reference_on_this_dev_set=bool(difference>=threshold and low>0),
                    final_independent_test_pending=True,multiple_training_seeds_pending=True,model_published=False)
                atomic_json(out/'acceptance.json',acceptance);atomic_json(out/'status.json',dict(stage='complete',status='complete',acceptance=acceptance))
                return acceptance
            except BaseException as error:
                atomic_json(out/'status.json',dict(stage='failed_stopped',status='failed_stopped',error=repr(error)));return dict(objective=name,status='failed_stopped',error=repr(error))
        with cf.ThreadPoolExecutor(2) as pool:results=list(pool.map(objective,[('win','hy',0,2,41),('frag','gpt',1,3,73)]))
        status.update(stage='mainline_finished',status='complete',results=results,elapsed_seconds=time.time()-start);atomic_json(a.run/'pipeline.json',status)
    except BaseException as error:
        status.update(status='failed_stopped',error=repr(error),elapsed_seconds=time.time()-start);atomic_json(a.run/'pipeline.json',status);raise

if __name__=='__main__':main()
