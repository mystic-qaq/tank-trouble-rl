"""Finite V2 initialization experiment. Every phase and candidate has an explicit cap."""
import argparse,concurrent.futures as cf,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from .collect import atomic_json
from .gates import assess

def run(args,log,timeout=None):
    log.parent.mkdir(parents=True,exist_ok=True)
    with open(log,'a') as out:
        out.write('\nCOMMAND '+json.dumps([str(x) for x in args])+'\n');out.flush()
        process=subprocess.Popen([str(x) for x in args],stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=timeout)
            if code:raise RuntimeError(f'Command failed ({code}), see {log}')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise

def module(name,*args):return [sys.executable,'-m','policy.rl.v2.'+name,*args]
def read(path):return json.loads(path.read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data-seconds',type=int,default=7200)
    p.add_argument('--candidate-seconds',type=int,default=3600);a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'pipeline.json').exists():raise ValueError('Fresh experiment root required; V1 is never resumed')
    start=time.time();status=dict(version=2,status='running',stage='G0',started_unix=start,run=str(a.run),resume_v1=False,
        data_seconds=a.data_seconds,candidate_seconds=a.candidate_seconds,candidates=4,max_cpu_workers=64,max_gpus=4)
    atomic_json(a.run/'pipeline.json',status)
    try:
        run(['node','policy/verify-frozen.mjs'],a.run/'G0-frozen.log',60)
        run(['node','--experimental-vm-modules','--disable-warning=ExperimentalWarning','policy/rl/test-env.mjs'],a.run/'G0-base-engine.log',180)
        run(['node','--experimental-vm-modules','--disable-warning=ExperimentalWarning','policy/rl/v2/test_engine.mjs'],a.run/'G0-v2-engine.log',60)
        run([sys.executable,'-m','pytest','policy/rl/v2/test_network.py','policy/rl/v2/test_training.py','-q','-p','no:cacheprovider'],a.run/'G0-network.log',180)
        source={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['js','arena','policy'] for f in Path(folder).rglob('*')
            if f.is_file() and '__pycache__' not in f.parts and f.suffix in ['.mjs','.js','.py','.json']}
        atomic_json(a.run/'source-manifest.json',source)
        atomic_json(a.run/'G0.json',dict(passed=True,source_manifest_sha256=hashlib.sha256((a.run/'source-manifest.json').read_bytes()).hexdigest(),
            contracts=['frozen_7_policies_74_files','60s_engine_reward_and_replay','observer_physics_identity','event_ledger_boundaries',
                'six_microtask_references','binary_ipc_shadow_joint_action','log_probability_masks_padding','actor_save_load_decode_identity'],
            production_browser_export='separate_predeployment_acceptance_required'))
        status.update(stage='collect');atomic_json(a.run/'pipeline.json',status)
        jobs=[('hy',24,620261106),('gpt',24,620261106),('reference',16,630261106)]
        def collect(job):
            teacher,workers,seed=job;extra=['--skills','--episodes','2400','--min-layouts','2400','--min-shots','0'] if teacher=='reference' else []
            run(module('collect','--teacher',teacher,'--workers',workers,'--seed',seed,'--seconds',a.data_seconds,
                '--out',a.run/'data'/teacher,*extra),a.run/'logs'/f'collect-{teacher}.log',a.data_seconds+180)
        with cf.ThreadPoolExecutor(3) as pool:list(pool.map(collect,jobs))
        coverage={teacher:read(a.run/'data'/teacher/'manifest.json') for teacher in ['hy','gpt','reference']}
        if not all(m['coverage_target_met'] for m in coverage.values()):
            status.update(status='stopped_coverage_gate',stage='coverage',coverage=coverage);atomic_json(a.run/'pipeline.json',status);return
        status.update(stage='references');atomic_json(a.run/'pipeline.json',status)
        refs=a.run/'references'
        def reference(job):
            target,teacher,suite,count,workers=job
            run(module('evaluate','--teacher',teacher,'--suite',suite,'--count',count,'--workers',workers,'--out',refs/target),a.run/'logs'/f'reference-{target}.log',1800)
        with cf.ThreadPoolExecutor(3) as pool:list(pool.map(reference,[('skills','reference','skills',50,16),('win','hy','combat',100,16),('frag','gpt','combat',100,16)]))
        reference_skill=read(refs/'skills/results.json')['aggregate']
        if any(v['success']<.9 for v in reference_skill.values()):raise RuntimeError('Skill reference does not establish a reliable task baseline')
        status.update(stage='bounded_bc_candidates');atomic_json(a.run/'pipeline.json',status)
        def candidate(job):
            objective,teacher,weight,seed,gpu=job;directory=a.run/f'{objective}-fw{weight}';directory.mkdir()
            deadline=time.monotonic()+a.candidate_seconds;log=a.run/'logs'/f'{directory.name}.log';device=f'cuda:{gpu}'
            base_data=[a.run/'data'/teacher,a.run/'data/reference']
            def remaining():
                value=deadline-time.monotonic()
                if value<5:raise TimeoutError('Candidate budget exhausted')
                return value
            def execute(command):run(command,log,remaining())
            def skills():execute(module('evaluate','--model',directory/'bc/best.pt','--suite','skills','--count',50,'--workers',8,'--device',device,'--out',directory/'skills'))
            try:
                execute(module('train_bc','--data',*base_data,'--validation-data',a.run/'data'/teacher,'--out',directory/'bc',
                    '--seconds',min(1200,int(remaining()-5)),'--steps',20000,'--seed',seed,'--fire-weight',weight,'--device',device))
                skills();report=assess(directory,refs,a.run/'data'/teacher,objective);atomic_json(directory/'gates.json',report)
                if not (report['gates']['G1']['passed'] and report['gates']['G2']['passed']):
                    shutil.copytree(directory/'bc',directory/'bc-initial');shutil.copytree(directory/'skills',directory/'skills-initial')
                    execute(module('dagger','--model',directory/'bc/best.pt','--teacher',teacher,'--out',directory/'dagger','--seed',730001001+seed*10000,'--workers',8,'--device',device))
                    execute(module('train_bc','--data',*base_data,'--dagger-data',directory/'dagger','--validation-data',a.run/'data'/teacher,
                        '--resume',directory/'bc/best.pt','--out',directory/'bc','--seconds',min(600,int(remaining()-5)),
                        '--steps',10000,'--seed',seed+100,'--fire-weight',weight,'--device',device))
                    skills();report=assess(directory,refs,a.run/'data'/teacher,objective);atomic_json(directory/'gates.json',report)
                if report['gates']['G1']['passed'] and report['gates']['G2']['passed']:
                    execute(module('evaluate','--model',directory/'bc/best.pt','--suite','combat','--count',100,'--workers',8,'--device',device,'--out',directory/'combat'))
                    report=assess(directory,refs,a.run/'data'/teacher,objective);atomic_json(directory/'gates.json',report)
                # A successful initialization produces evidence; the short-RL phase has its own frame budget.
                atomic_json(directory/'candidate.json',dict(status='initialization_passed' if report['ppo_allowed'] else 'stopped_at_gate',objective=objective,
                    fire_weight=weight,seed=seed,seconds_used=a.candidate_seconds-remaining(),ppo_started=False))
                return dict(candidate=directory.name,ppo_allowed=report['ppo_allowed'])
            except (TimeoutError,subprocess.TimeoutExpired) as error:
                atomic_json(directory/'candidate.json',dict(status='budget_exhausted',error=str(error),ppo_started=False));return dict(candidate=directory.name,ppo_allowed=False,status='budget_exhausted')
            except Exception as error:
                atomic_json(directory/'candidate.json',dict(status='failed_stopped',error=repr(error),ppo_started=False));return dict(candidate=directory.name,ppo_allowed=False,status='failed_stopped')
        candidates=[('win','hy',1,11,0),('win','hy',4,29,1),('frag','gpt',1,11,2),('frag','gpt',4,29,3)]
        with cf.ThreadPoolExecutor(4) as pool:results=list(pool.map(candidate,candidates))
        status.update(stage='initialization_acceptance',status='complete',results=results,elapsed_seconds=time.time()-start,
            next_phase='bounded_200k_frame_two_seed_ppo_only_for_exact_G0_G3_approved_checkpoints',ppo_started=False)
        atomic_json(a.run/'pipeline.json',status)
    except BaseException as error:
        status.update(status='failed_stopped',error=repr(error),elapsed_seconds=time.time()-start);atomic_json(a.run/'pipeline.json',status);raise

if __name__=='__main__':main()
