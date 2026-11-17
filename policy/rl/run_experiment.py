"""Bounded teacher -> BC -> baseline validation -> PPO experiment, suitable for tmux."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--objective",choices=["win","frag"],required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--samples",type=int,default=65536)
    p.add_argument("--steps",type=int,default=1048576)
    p.add_argument("--envs",type=int,default=16)
    p.add_argument("--seed",type=int,default=20261106)
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    lock=open(args.out/"experiment.lock","w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    teacher="hy" if args.objective=="win" else "gpt"
    state=dict(objective=args.objective,teacher=teacher,round_limit=60,samples=args.samples,steps=args.steps,
        envs=args.envs,seed=args.seed,started_at=time.time(),status="running")
    state['source_root']=str(ROOT)
    if (ROOT/'source-manifest.json').exists():
        (args.out/'source-manifest.json').write_bytes((ROOT/'source-manifest.json').read_bytes())
    def stage(name,script,arguments):
        state.update(stage=name,stage_started_at=time.time())
        (args.out/"experiment.json").write_text(json.dumps(state,indent=2))
        print(json.dumps(dict(event="stage_start",**state)),flush=True)
        subprocess.run([sys.executable,str(ROOT/"policy/rl"/script),*map(str,arguments)],cwd=ROOT,check=True)
    try:
        data=args.out/"demonstrations";bc=args.out/"bc";ppo=args.out/"ppo"
        if not (data/"manifest.json").exists():
            stage("collect","collect.py",["--teacher",teacher,"--samples",args.samples,"--envs",args.envs,"--workers",args.envs,"--seed",args.seed,"--out",data])
        if not (bc/"best.pt").exists():
            stage("bc","train_bc.py",["--data",data,"--out",bc,"--epochs",30,"--seed",args.seed])
        for label,extra in [("teacher",["--teacher",teacher]),("bc",["--checkpoint",bc/"best.pt"])]:
            result=args.out/f"validation-{label}.json"
            if not result.exists():
                stage("validate_"+label,"evaluate.py",["--objective",args.objective,"--episodes",28,"--envs",8,"--out",result,*extra])
        if not (ppo/"final.pt").exists():
            initial=["--resume",ppo/"latest.pt"] if (ppo/"latest.pt").exists() else ["--init",bc/"best.pt"]
            stage("ppo","train_ppo.py",["--objective",args.objective,"--out",ppo,"--steps",args.steps,"--envs",args.envs,"--workers",args.envs,"--seed",args.seed,*initial])
        state.update(status="complete",stage="complete",finished_at=time.time())
    except BaseException as error:
        state.update(status="failed",error=str(error),failed_at=time.time());raise
    finally:
        (args.out/"experiment.json").write_text(json.dumps(state,indent=2))

if __name__=="__main__":main()
