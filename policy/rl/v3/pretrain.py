"""One shared recurrent foundation, parallel GPUs for speed, selected by closed-loop outcomes."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .model import Policy,checkpoint,load_policy
from .data import Sequences,atomic_save
from ..v2.collect import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=30000);p.add_argument('--seconds',type=int,default=5400);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',type=Path);p.add_argument('--eval-device',default='cuda:0');p.add_argument('--eval-workers',type=int,default=16)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--evaluate-every',type=int,default=10000);p.add_argument('--objective',choices=['shared','win','frag'],default='shared')
    a=p.parse_args();rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(a.seed)
    if world>1:
        torch.cuda.set_device(local);dist.init_process_group('nccl');device=f'cuda:{local}'
    else:device=a.device
    a.out.mkdir(parents=True,exist_ok=True);actor=Policy().to(device)
    if a.resume:actor,_=load_policy(a.resume,device)
    actor.log_std.requires_grad_(False)
    model=DDP(actor,device_ids=[local]) if world>1 else actor
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    dataset=Sequences(a.data,seed=a.seed+rank*1000003)
    loader=DataLoader(dataset,batch_size=a.batch,num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn')
    batches=iter(loader);writer=SummaryWriter(str(a.out/'tb')) if rank==0 else None
    start=time.monotonic();evaluations=[];step=0
    def save_and_evaluate(step):
        path=a.out/f'step-{step:06d}.pt';atomic_save(checkpoint(actor,step=step,stage='common_bc' if a.objective=='shared' else 'dagger_bc',seed=a.seed),path)
        atomic_save(checkpoint(actor,step=step,seed=a.seed),a.out/'last.pt')
        log=open(a.out/f'eval-{step}.log','w')
        command=[sys.executable,'-m','policy.rl.v3.evaluate','--model',str(path),'--out',str(a.out/f'eval-{step}'),
            '--device',a.eval_device,'--workers',str(a.eval_workers),'--count','1']
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);evaluations.append((step,path,process,log))
    for step in range(1,a.steps+1):
        obs,actions,mask,quality=[x.to(device,non_blocking=True) for x in next(batches)]
        loss,parts=model(obs,actions,mask,quality)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite recurrent BC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        if rank==0 and step%50==0:
            writer.add_scalar('train/loss',float(loss.detach()),step)
            for k,v in zip(['joint_motion','turn_magnitude','conditional_fire'],parts):writer.add_scalar('train/'+k,float(v),step)
            writer.flush()
            atomic_json(a.out/'status.json',dict(stage='shared_pretrain' if a.objective=='shared' else 'dagger_bc',status='running',step=step,
                elapsed_seconds=time.monotonic()-start,parameters=sum(p.numel() for p in actor.parameters()),gpus=world,trajectories=len(dataset.records),loss=float(loss.detach())))
        if step%a.evaluate_every==0 and rank==0:save_and_evaluate(step)
        if step%100==0:
            stop=torch.tensor(int(time.monotonic()-start>a.seconds),device=device)
            if world>1:dist.broadcast(stop,src=0)
            if stop.item():break
    if rank==0:
        if not evaluations or evaluations[-1][0]!=step:save_and_evaluate(step)
        candidates=[]
        for n,path,process,log in evaluations:
            code=process.wait();log.close()
            if code:raise RuntimeError('Closed-loop checkpoint evaluation failed')
            report=json.loads((a.out/f'eval-{n}'/'results.json').read_text())
            quality=(report['win']/.3704064+report['frag']/.8718579)/2 if a.objective=='shared' else report[a.objective]
            candidates.append((quality,path,report))
        best=max(candidates,key=lambda x:x[0]);data=torch.load(best[1],map_location='cpu',weights_only=False);data['closed_loop']=best[2];atomic_save(data,a.out/'best.pt')
        atomic_json(a.out/'status.json',dict(stage='shared_pretrain' if a.objective=='shared' else 'dagger_bc',status='complete',step=step,
            elapsed_seconds=time.monotonic()-start,best_checkpoint=str(best[1]),best_win=best[2]['win'],best_net=best[2]['frag'],spin_fraction=best[2]['spin_fraction'],gpus=world))
        writer.close()
    if world>1:dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
