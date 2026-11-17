"""Resume failed-before-update reward jobs with strict FP32 and preserved warmup state."""
import argparse,concurrent.futures as cf,json,time,sys
from pathlib import Path
from .pipeline import command
from ..v2.collect import atomic_json
from ..v2.pipeline import run

def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    status=read(root/'pipeline.json');deadline=status['started_unix']+status['total_wallclock_hours']*3600
    for goal in ['win','frag']:
        if (root/goal/'rl-fp32/last.pt').exists():raise ValueError('FP32 continuation already has output')
    status.update(status='running',stage='reward_training_fp32',active_source_snapshot=str(Path.cwd()),
        numerical_fix='TF32 disabled; probability replay threshold unchanged',policy_update_before_guard_failure=False)
    atomic_json(root/'pipeline.json',status)
    def objective(job):
        name,gpu,evalgpu,seed=job;out=root/name
        def remaining(reserve=0):return max(1,deadline-time.time()-reserve)
        try:
            selected=read(out/'initialization.json')['selected'];source=Path(selected['path'])
            resume=out/'rl/last.pt'
            if not resume.exists():resume=out/'rl/warmup.pt'
            if not resume.exists():raise ValueError('No preserved warmup checkpoint')
            data=[root/'data/multi',root/'data/skills',*[out/f'dagger-{i}' for i in range(1,4)]]
            atomic_json(out/'status.json',dict(stage='ppo_fp32',status='resuming',checkpoint=str(resume),selected_initialization=selected))
            run(command('train_rl','--model',source,'--resume',resume,'--data',*data,'--out',out/'rl-fp32','--objective',name,
                '--device',f'cuda:{gpu}','--eval-device',f'cuda:{evalgpu}','--workers','32','--eval-workers','16','--frames','16000000',
                '--seconds',str(int(min(21600,remaining(1800)))),'--seed',str(seed),'--eval-every','500000'),
                root/'logs'/f'{name}-fp32.log',remaining(900))
            run(command('evaluate','--model',out/'rl-fp32/best.pt','--out',out/'acceptance-fp32','--objective',name,
                '--device',f'cuda:{gpu}','--workers','16','--count','100'),root/'logs'/f'{name}-fp32.log',remaining())
            report=read(out/'acceptance-fp32/results.json');baseline=read(Path('/home/xy/rl_for_fun/runs/v2b/references')/name/'results.json')
            metric='win' if name=='win' else 'netKills';reference={(r['n'],r['block'],r['mapSeed'],r['seat']):r for r in baseline['rows']};clusters={}
            for row in report['rows']:
                teacher=reference[(row['n'],row['block'],row['mapSeed'],row['seat'])]
                if any(row[k]!=teacher[k] for k in ['seed','spawnSeed','opponents']):raise ValueError('Unpaired acceptance')
                clusters.setdefault(row['n'],{}).setdefault(row['block'],[]).append(row[metric]-teacher[metric])
            import numpy as np
            rng=np.random.default_rng(seed);samples=[]
            for _ in range(4000):
                means=[]
                for blocks in clusters.values():
                    values=np.array([np.mean(v) for v in blocks.values()]);means.append(rng.choice(values,len(values),replace=True).mean())
                samples.append(np.mean(means))
            low,high=np.quantile(samples,[.025,.975]);difference=float(np.mean([np.mean([np.mean(v) for v in b.values()]) for b in clusters.values()]))
            margin=.03 if name=='win' else .1
            acceptance=dict(objective=name,score=report[name],teacher_difference=difference,difference_ci95=[float(low),float(high)],target_margin=margin,
                beats_reference_on_this_dev_set=bool(difference>=margin and low>0),final_independent_test_pending=True,multiple_training_seeds_pending=True,model_published=False,
                precision='ieee_fp32_no_tf32')
            atomic_json(out/'acceptance.json',acceptance);atomic_json(out/'status.json',dict(stage='complete',status='complete',acceptance=acceptance));return acceptance
        except BaseException as error:
            atomic_json(out/'status.json',dict(stage='failed_stopped',status='failed_stopped',error=repr(error)));return dict(objective=name,status='failed_stopped',error=repr(error))
    with cf.ThreadPoolExecutor(2) as pool:results=list(pool.map(objective,[('win',0,2,41),('frag',1,3,73)]))
    status.update(stage='mainline_finished',status='complete',results=results,elapsed_seconds=time.time()-status['started_unix']);atomic_json(root/'pipeline.json',status)

if __name__=='__main__':main()
