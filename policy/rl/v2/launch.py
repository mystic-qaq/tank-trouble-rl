"""Linux launcher: isolated source, free GPU UUIDs, finite tmux experiment and local-only dashboard."""
import argparse,csv,io,json,shlex,socket,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    a=p.parse_args();root=a.source.resolve();a.run=a.run.resolve();home=Path('/home/xy/rl_for_fun');python=home/'.venv/bin/python'
    if (a.run/'pipeline.json').exists():raise ValueError('Existing experiment cannot be relaunched')
    for name in ['tank-v2b-initialization','tank-v2b-board']:
        if subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0:raise ValueError('Session already exists: '+name)
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
    occupied={row.strip() for row in processes.splitlines()}
    available=[]
    for index,uuid,memory,util in csv.reader(io.StringIO(output),skipinitialspace=True):
        if uuid not in occupied and int(memory)<256 and int(util)<10:available.append(dict(index=int(index),uuid=uuid))
    if len(available)<4:raise RuntimeError('Four currently free GPUs are required for this predefined candidate batch')
    selected=available[:4];a.run.mkdir(parents=True,exist_ok=True)
    (a.run/'gpu-allocation.json').write_text(json.dumps(dict(cuda_logical_to_physical=selected,allocated_unix=time.time()),indent=2))
    visible=','.join(g['uuid'] for g in selected)
    environment=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
        'CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+visible]
    pipeline=environment+[str(python),'-m','policy.rl.v2.pipeline','--run',str(a.run)]
    subprocess.run(['tmux','new-session','-d','-s','tank-v2b-initialization','-c',str(root),
        shlex.join(pipeline)+' > '+shlex.quote(str(a.run/'pipeline.log'))+' 2>&1'],check=True)
    port=None
    for candidate in range(6006,6017):
        with socket.socket() as s:
            try:s.bind(('127.0.0.1',candidate));port=candidate;break
            except OSError:continue
    if port is None:raise RuntimeError('Experiment started, but no dashboard port is free')
    board=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1',str(home/'.venv/bin/tensorboard'),
        '--logdir',str(a.run),'--host','127.0.0.1','--port',str(port),'--reload_interval','5','--load_fast=false']
    subprocess.run(['tmux','new-session','-d','-s','tank-v2b-board','-c',str(root),shlex.join(board)+' > '+shlex.quote(str(a.run/'tensorboard.log'))+' 2>&1'],check=True)
    report=dict(source=str(root),run=str(a.run),gpu_mapping=selected,dashboard_remote_port=port,pipeline_session='tank-v2b-initialization',
        dashboard_session='tank-v2b-board',started_unix=time.time())
    (a.run/'launch.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
