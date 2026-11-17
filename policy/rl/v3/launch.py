import argparse,csv,io,json,shlex,socket,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    a.run=a.run.resolve();source=a.source.resolve();home=Path('/home/xy/rl_for_fun')
    if (a.run/'pipeline.json').exists():raise ValueError('V3 already started')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    gpus=[dict(index=int(i),uuid=u) for i,u,m,p in csv.reader(io.StringIO(raw),skipinitialspace=True) if int(m)<256 and int(p)<10]
    if len(gpus)<4:raise RuntimeError('Four free GPUs required; no other jobs will be stopped')
    selected=gpus[:4];visible=','.join(g['uuid'] for g in selected)
    env=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1',
        'CUDA_DEVICE_ORDER=PCI_BUS_ID','CUDA_VISIBLE_DEVICES='+visible]
    command=env+[str(home/'.venv/bin/python'),'-m','policy.rl.v3.pipeline','--run',str(a.run)]
    subprocess.run(['tmux','new-session','-d','-s','tank-v3-main','-c',str(source),shlex.join(command)+' > '+shlex.quote(str(a.run/'logs/pipeline.log'))+' 2>&1'],check=True)
    port=6007
    with socket.socket() as s:s.bind(('127.0.0.1',port))
    logdirs=','.join(name+':'+str(a.run/path) for name,path in [('multi_teacher','data/multi'),('skills','data/skills'),('shared','shared'),('win','win'),('frag','frag')])
    board=['env','CUDA_VISIBLE_DEVICES=',str(home/'.venv/bin/tensorboard'),'--logdir_spec',logdirs,'--host','127.0.0.1','--port',str(port),'--reload_interval','5','--load_fast=false']
    subprocess.run(['tmux','new-session','-d','-s','tank-v3-board','-c',str(source),shlex.join(board)+' > '+shlex.quote(str(a.run/'logs/tensorboard.log'))+' 2>&1'],check=True)
    report=dict(source=str(source),run=str(a.run),gpus=selected,ddp_gpu_limit=4,total_gpu_limit=4,dashboard_port=port,started_unix=time.time())
    (a.run/'launch.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
