"""Complete-episode recurrent PPO, independent critic, persistent checkpoints and real evals."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .model import Policy,Critic,load_policy,checkpoint,HIDDEN
from .rollout import run,active_episodes
from .data import Sequences,atomic_save
from ..v2.collect import atomic_json
from ..v2.train_ppo import fresh_tasks

def make_chunks(episodes,critic,device,size=32):
    chunks=[]
    with torch.inference_mode():
        for episode in episodes:
            hidden=torch.zeros(1,1,HIDDEN,device=device)
            for offset in range(0,len(episode),size):
                part=episode[offset:offset+size];x=np.stack([r['obs'] for r in part]);initial=hidden[0,0].cpu().numpy().copy()
                values,hidden=critic(torch.from_numpy(x[None]).to(device),hidden)
                chunk=dict(obs=np.zeros((size,1966),np.float32),actions=np.zeros((size,3),np.float32),lp=np.zeros(size,np.float32),
                    returns=np.zeros(size,np.float32),values=np.zeros(size,np.float32),mask=np.zeros(size,np.float32),
                    actor_hidden=part[0]['hidden'],critic_hidden=initial)
                length=len(part);chunk['obs'][:length]=x;chunk['actions'][:length]=np.stack([r['action'] for r in part])
                chunk['lp'][:length]=[r['logprob'] for r in part];chunk['returns'][:length]=[r['return'] for r in part]
                chunk['values'][:length]=values[0].cpu().numpy();chunk['mask'][:length]=1;chunks.append(chunk)
    advantages=np.concatenate([(c['returns']-c['values'])[c['mask']>0] for c in chunks]);mean,std=advantages.mean(),max(1e-6,advantages.std())
    for c in chunks:c['advantage']=(c['returns']-c['values']-mean)/std
    return chunks

def tensors(chunks,indices,device):
    keys=['obs','actions','lp','returns','mask','advantage','actor_hidden','critic_hidden']
    return {key:torch.from_numpy(np.stack([chunks[int(i)][key] for i in indices])).to(device) for key in keys}

def weighted(x,mask):return (x*mask).sum()/mask.sum().clamp(min=1)

def restore_rng_state(metadata,device):
    # load_policy(map_location='cuda:...') also relocates serialized RNG tensors.
    # Both generator setters require CPU ByteTensors, independent of model placement.
    torch.set_rng_state(metadata['cpu_rng'].cpu())
    torch.cuda.set_rng_state(metadata['cuda_rng'].cpu(),device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--objective',choices=['win','frag'],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--eval-device',default='cuda:2');p.add_argument('--workers',type=int,default=32)
    p.add_argument('--frames',type=int,default=8000000);p.add_argument('--seconds',type=int,default=21600);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--resume',type=Path);p.add_argument('--eval-every',type=int,default=500000);p.add_argument('--episodes',type=int,default=64)
    p.add_argument('--warmup-episodes',type=int,default=32);p.add_argument('--eval-count',type=int,default=1);p.add_argument('--eval-workers',type=int,default=16)
    p.add_argument('--eval-limit',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(a.seed);torch.backends.mha.set_fastpath_enabled(False)
    actor,metadata=load_policy(a.resume or a.model,a.device);critic=Critic().to(a.device)
    critic.encoder.load_state_dict(actor.encoder.state_dict());critic.memory.load_state_dict(actor.memory.state_dict())
    actor_opt=torch.optim.Adam(actor.parameters(),lr=7.5e-5);critic_opt=torch.optim.Adam(critic.parameters(),lr=2.5e-4)
    frames=version=0;aux_weight=.05
    if a.resume:
        if metadata['objective']!=a.objective:raise ValueError('Resume objective mismatch')
        critic.load_state_dict(metadata['critic']);actor_opt.load_state_dict(metadata['actor_optimizer']);critic_opt.load_state_dict(metadata['critic_optimizer'])
        frames=metadata['frames'];version=metadata['version'];aux_weight=metadata.get('aux_weight',.05)
        restore_rng_state(metadata,a.device)
    writer=SummaryWriter(str(a.out/'tb'));start=time.monotonic();deadline=start+a.seconds
    teacher=iter(DataLoader(Sequences(a.data,learn=32,burn=64,seed=a.seed),batch_size=4,num_workers=2,pin_memory=True,persistent_workers=True,multiprocessing_context='spawn'))
    evals=[];best_score=-float('inf');best_path=a.model;last_eval=frames;collapse=0;regressions=0
    def evaluate(path,marker,count=1):
        directory=a.out/f'eval-{marker}';log=open(a.out/f'eval-{marker}.log','w')
        process=subprocess.Popen([sys.executable,'-m','policy.rl.v3.evaluate','--model',str(path),'--out',str(directory),'--device',a.eval_device,
            '--workers',str(a.eval_workers),'--count',str(a.eval_count),'--limit-tasks',str(a.eval_limit),'--objective',a.objective],stdout=log,stderr=subprocess.STDOUT)
        evals.append((process,log,path,directory,marker))
    def save(path):
        atomic_save(checkpoint(actor,objective=a.objective,frames=frames,version=version,seed=a.seed,critic=critic.state_dict(),
            actor_optimizer=actor_opt.state_dict(),critic_optimizer=critic_opt.state_dict(),aux_weight=aux_weight,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(a.device),gamma=1.,return_target='complete_agent_MC'),path)
    evaluate(a.model,'initial')
    # Fresh current-policy episodes for the critic only; never replayed as a PPO batch.
    if not a.resume:
        actor.eval();warm_rows,warm_records,_=run(fresh_tasks(a.seed,-1,a.warmup_episodes),actor,a.workers,a.device,a.objective,
            stochastic=True,trajectory=True,early_death=True)
        frames+=sum(r['frames'] for r in warm_rows);warm_chunks=make_chunks(active_episodes(warm_records),critic,a.device);del warm_records
        for step in range(200):
            index=np.random.default_rng(a.seed+step).integers(len(warm_chunks),size=min(16,len(warm_chunks)));batch=tensors(warm_chunks,index,a.device)
            values,_=critic(batch['obs'],batch['critic_hidden'][None]);loss=weighted(F.smooth_l1_loss(values,batch['returns'],reduction='none'),batch['mask'])
            critic_opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1);critic_opt.step()
        del warm_chunks;save(a.out/'warmup.pt')
    status=dict(stage='ppo',status='running',objective=a.objective,frames=frames)
    stop_reason=None
    while a.frames-frames>=3600 and time.monotonic()<deadline:
        actor.eval();version+=1;episode_count=min(a.episodes,(a.frames-frames)//3600)
        rows,records,seconds=run(fresh_tasks(a.seed,version,episode_count),actor,a.workers,a.device,a.objective,
            stochastic=True,trajectory=True,early_death=True)
        frames+=sum(r['frames'] for r in rows);episodes=active_episodes(records);del records
        chunks=make_chunks(episodes,critic,a.device);del episodes
        replay_error=0.
        with torch.inference_mode():
            for offset in range(0,len(chunks),16):
                b=tensors(chunks,range(offset,min(offset+16,len(chunks))),a.device)
                lp,_,_=actor.evaluate_actions(b['obs'],b['actor_hidden'][None],b['actions'])
                replay_error=max(replay_error,float(((lp-b['lp']).abs()*b['mask']).max()))
        if replay_error>1e-3:raise RuntimeError('Recurrent behavior probability replay mismatch '+str(replay_error))
        actor.train();critic.train();kl_total=0.;updates=0
        for epoch in range(3):
            for index in torch.randperm(len(chunks)).split(16):
                b=tensors(chunks,index,a.device);lp,entropy,_=actor.evaluate_actions(b['obs'],b['actor_hidden'][None],b['actions'])
                delta=lp-b['lp'];ratio=delta.exp();surrogate=torch.minimum(ratio*b['advantage'],ratio.clamp(.9,1.1)*b['advantage'])
                loss=-weighted(surrogate,b['mask'])-weighted((entropy*entropy.new_tensor([.003,.0001,.001])).sum(-1),b['mask'])
                tx,ta,tm,tq=[x.to(a.device,non_blocking=True) for x in next(teacher)];auxiliary,_=actor(tx,ta,tm,tq)
                loss=loss+aux_weight*auxiliary
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                actor_opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),.5);actor_opt.step()
                values,_=critic(b['obs'],b['critic_hidden'][None]);value_loss=weighted(F.smooth_l1_loss(values,b['returns'],reduction='none'),b['mask'])
                critic_opt.zero_grad(set_to_none=True);value_loss.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1);critic_opt.step()
                kl=float(weighted((ratio-1)-delta,b['mask']).detach());kl_total+=kl;updates+=1
                if kl>.02:break
            if kl>.02:break
        emissions=sum(r['events']['emissions'] for r in rows);hits=sum(r['events']['hits'] for r in rows)
        collapse=collapse+1 if (emissions==0 or hits==0) and np.mean([r['return'] for r in rows])<=0 else 0
        status=dict(stage='ppo',status='running',objective=a.objective,frames=frames,version=version,active_chunks=len(chunks),
            policy_replay_error=replay_error,mean_return=float(np.mean([r['return'] for r in rows])),actual_emissions=emissions,hits=hits,
            suicides=sum(r['suicides'] for r in rows),spin_fraction=sum(r['spin_frames'] for r in rows)/max(1,sum(r['active_frames'] for r in rows)),
            approximate_kl=kl_total/max(1,updates),value_loss=float(value_loss.detach()),aux_weight=aux_weight,
            elapsed_seconds=time.monotonic()-start,collection_fps=sum(r['frames'] for r in rows)/max(.001,seconds))
        save(a.out/'last.pt');atomic_json(a.out/'status.json',status);print(json.dumps(status),flush=True)
        for key in ['mean_return','actual_emissions','hits','suicides','spin_fraction','approximate_kl','value_loss','collection_fps']:writer.add_scalar('train/'+key,status[key],frames)
        writer.flush()
        if frames-last_eval>=a.eval_every and not any(p.poll() is None for p,_,_,_,_ in evals):
            path=a.out/f'frames-{frames:09d}.pt';save(path);evaluate(path,frames);last_eval=frames
        for item in list(evals):
            process,log,path,directory,marker=item
            if process.poll() is None:continue
            log.close();evals.remove(item)
            if process.returncode:raise RuntimeError('Concurrent fixed evaluation failed')
            result=json.loads((directory/'results.json').read_text());score=result[a.objective]
            writer.add_scalar('fixed_eval/'+a.objective,score,int(marker) if isinstance(marker,int) else 0)
            if score>best_score:
                improved=best_score>-float('inf');best_score=score;best_path=path;atomic_save(torch.load(path,map_location='cpu',weights_only=False),a.out/'best.pt');regressions=0
                if improved:aux_weight=max(.01,aux_weight*.8)
            elif score<best_score-(.10 if a.objective=='win' else .25):regressions+=1
            if result['spin_fraction']>.8 and result['emissions']<7 and score<best_score:regressions+=1
        if collapse>=3:stop_reason='stopped_no_effective_attack';break
        if regressions>=3:stop_reason='stopped_validated_regression';break
    final=a.out/f'final-{frames:09d}.pt';save(final)
    # Finish queued evals and select a measured checkpoint before the full acceptance run.
    evaluate(final,'final-'+str(frames))
    for process,log,path,directory,marker in evals:
        if process.wait():raise RuntimeError('Final queued evaluation failed')
        log.close();result=json.loads((directory/'results.json').read_text())
        if result[a.objective]>best_score:best_score=result[a.objective];best_path=path
    atomic_save(torch.load(best_path,map_location='cpu',weights_only=False),a.out/'best.pt')
    atomic_json(a.out/'status.json',dict(**{k:v for k,v in status.items() if k!='status'},status=stop_reason or 'budget_complete',best_quick_score=best_score,best_checkpoint=str(best_path)))
    writer.close()

if __name__=='__main__':main()
