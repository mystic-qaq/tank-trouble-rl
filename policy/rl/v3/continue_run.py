"""Stage-boundary handoff: retain completed work, select best initialization, continue PPO."""
import argparse,concurrent.futures as cf,json,os,signal,time,sys
from pathlib import Path
from .selection import select_initialization
from .pipeline import command
from ..v2.collect import atomic_json
from ..v2.pipeline import run

def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--old-coordinator',type=int,required=True);a=p.parse_args()
    root=a.run;status=read(root/'pipeline.json');deadline=status['started_unix']+status['total_wallclock_hours']*3600
    handoff=root/'audits/handoff-complete.json'
    if handoff.exists():raise ValueError('Handoff already executed')
    old=Path(f'/proc/{a.old_coordinator}');cmd=(old/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'policy.rl.v3.pipeline --run '+str(root) not in cmd:raise ValueError('Wrong coordinator PID')
    if (old/'stat').read_text().split()[2]!='T':raise ValueError('Old coordinator must be stopped at boundary')
    status.update(stage='waiting_for_dagger_boundary',coordinator_handoff_pending=True);atomic_json(root/'pipeline.json',status)
    # Current trainers/evaluators retain their own process groups and run to completion.
    while True:
        files=[root/g/'adapt-3/status.json' for g in ['win','frag']]
        if all(f.exists() and read(f).get('status')=='complete' for f in files):break
        if time.time()>deadline-1800:raise TimeoutError('No remaining window for safe continuation')
        time.sleep(3)
    # A complete status is written just before interpreter cleanup. Do not kill live trainers.
    cleanup_deadline=time.time()+120
    while True:
        children=set()
        for task in (old/'task').iterdir():
            try:children.update(int(v) for v in (task/'children').read_text().split())
            except FileNotFoundError:pass
        live=[]
        for child in children:
            try:
                state=Path(f'/proc/{child}/stat').read_text().split(') ',1)[1].split()[0]
                if state!='Z':live.append(child)
            except FileNotFoundError:pass
        if not live:break
        if time.time()>cleanup_deadline:raise RuntimeError('Completed trainers have not exited: '+str(live))
        time.sleep(2)
    os.kill(a.old_coordinator,signal.SIGKILL) # Only the verified, stopped coordinator. Completed children are preserved.
    atomic_json(handoff,dict(old_pid=a.old_coordinator,new_pid=os.getpid(),source=str(Path.cwd()),time_unix=time.time(),
        deadline_unix=deadline,training_steps_restarted=0,changes=['best_across_curriculum_rounds','CPU_RNG_restore_fix','additional_read_only_stall_diagnostics']))
    status.update(stage='reward_training',coordinator_handoff_pending=False,active_source_snapshot=str(Path.cwd()));atomic_json(root/'pipeline.json',status)
    def objective(job):
        name,gpu,evalgpu,seed=job;out=root/name
        def remaining(reserve=0):return max(1,deadline-time.time()-reserve)
        try:
            chosen,candidates=select_initialization(root,name)
            atomic_json(out/'initialization.json',dict(usable_for_reward_training=chosen is not None,selected=chosen,candidates=candidates,
                selection='maximum primary score among measured usable checkpoints; no teacher-F1 gate'))
            if chosen is None:
                atomic_json(out/'status.json',dict(stage='initialization',status='stopped_no_usable_behavior'));return dict(objective=name,status='stopped_no_usable_behavior')
            source=Path(chosen['path']);data=[root/'data/multi',root/'data/skills',*[out/f'dagger-{i}' for i in range(1,4)]]
            atomic_json(out/'status.json',dict(stage='ppo',status='starting',selected_initialization=chosen))
            if (out/'rl/last.pt').exists():raise ValueError('Unexpected existing PPO output; refuse to overwrite')
            args=command('train_rl','--model',source,'--data',*data,'--out',out/'rl','--objective',name,'--device',f'cuda:{gpu}',
                '--eval-device',f'cuda:{evalgpu}','--workers','32','--eval-workers','16','--frames','16000000',
                '--seconds',str(int(min(21600,remaining(1800)))),'--seed',str(seed),'--eval-every','500000')
            run(args,root/'logs'/f'{name}-continued.log',remaining(900))
            run(command('evaluate','--model',out/'rl/best.pt','--out',out/'acceptance','--objective',name,'--device',f'cuda:{gpu}',
                '--workers','16','--count','100'),root/'logs'/f'{name}-continued.log',remaining())
            report=read(out/'acceptance/results.json')
            baseline=read(Path('/home/xy/rl_for_fun/runs/v2b/references')/name/'results.json')
            metric='win' if name=='win' else 'netKills';reference={(r['n'],r['block'],r['mapSeed'],r['seat']):r for r in baseline['rows']};clusters={}
            for r in report['rows']:
                t=reference[(r['n'],r['block'],r['mapSeed'],r['seat'])]
                if any(r[f]!=t[f] for f in ['seed','spawnSeed','opponents']):raise ValueError('Unpaired acceptance')
                clusters.setdefault(r['n'],{}).setdefault(r['block'],[]).append(r[metric]-t[metric])
            import numpy as np
            rng=np.random.default_rng(seed);samples=[]
            for _ in range(4000):
                scores=[]
                for blocks in clusters.values():
                    values=np.array([np.mean(v) for v in blocks.values()]);scores.append(rng.choice(values,len(values),replace=True).mean())
                samples.append(np.mean(scores))
            lo,hi=np.quantile(samples,[.025,.975]);difference=float(np.mean([np.mean([np.mean(v) for v in b.values()]) for b in clusters.values()]))
            margin=.03 if name=='win' else .1
            acceptance=dict(objective=name,score=report[name],teacher_difference=difference,difference_ci95=[float(lo),float(hi)],target_margin=margin,
                beats_reference_on_this_dev_set=bool(difference>=margin and lo>0),final_independent_test_pending=True,multiple_training_seeds_pending=True,model_published=False)
            atomic_json(out/'acceptance.json',acceptance);atomic_json(out/'status.json',dict(stage='complete',status='complete',acceptance=acceptance));return acceptance
        except BaseException as error:
            atomic_json(out/'status.json',dict(stage='failed_stopped',status='failed_stopped',error=repr(error)));return dict(objective=name,status='failed_stopped',error=repr(error))
    with cf.ThreadPoolExecutor(2) as pool:results=list(pool.map(objective,[('win',0,2,41),('frag',1,3,73)]))
    status.update(stage='mainline_finished',status='complete',results=results,elapsed_seconds=time.time()-status['started_unix']);atomic_json(root/'pipeline.json',status)

if __name__=='__main__':main()
