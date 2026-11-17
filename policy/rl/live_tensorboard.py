"""Read existing logs into live TensorBoard events; never imports or changes training checkpoints."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter


def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def records(path):
    try:
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):continue
                try:yield json.loads(line)
                except json.JSONDecodeError:continue
    except FileNotFoundError:return


TRAIN_TAGS={
    'recent_win_rate':'Training/mixed_recent_win_rate',
    'recent_net_kills':'Training/mixed_recent_net_kills',
    'loss':'Optimizer/combined_loss','kl':'Optimizer/approx_kl',
    'clip_fraction':'Optimizer/clip_fraction','active_fraction':'System/active_fraction',
    'frames_per_second':'System/aggregate_frames_per_second',
}


class Importer:
    def __init__(self, root, logdir):
        self.root=root;self.logdir=logdir;logdir.mkdir(parents=True,exist_ok=True)
        self.state_path=logdir/'import-state.json'
        self.state=read_json(self.state_path) or {}
        self.writers={}
    def writer(self,name):
        if name not in self.writers:self.writers[name]=SummaryWriter(str(self.logdir/name),flush_secs=2)
        return self.writers[name]
    @staticmethod
    def scalar(writer,tag,value,step,walltime=None):
        if isinstance(value,(int,float)) and math.isfinite(value):
            writer.add_scalar(tag,value,step,walltime=walltime)
    def update(self):
        imported=0
        for objective,teacher in [('win','HY_baseline'),('frag','GPT_baseline')]:
            experiment_root=self.root/'runs/initial60'/objective
            state=self.state.setdefault(objective,{'step':-1,'bc_epoch':0,'validations':[],'baseline_step':-1})
            writer=self.writer(objective)
            experiment=read_json(experiment_root/'experiment.json') or {}
            latest_step=state['step']
            for record in records(self.root/'runs/initial60'/f'{objective}.log'):
                if record.get('event'):continue
                if record.get('stage')=='bc' and record['epoch']>state['bc_epoch']:
                    epoch=record['epoch']
                    for key in ['loss','validation_loss']:self.scalar(writer,'BC/'+key,record.get(key),epoch)
                    state['bc_epoch']=epoch;imported+=1
                elif record.get('stage')=='ppo' and record.get('steps',-1)>state['step']:
                    step=record['steps']
                    walltime=experiment.get('stage_started_at',time.time())+record.get('elapsed',0)
                    for key,tag in TRAIN_TAGS.items():self.scalar(writer,tag,record.get(key),step,walltime)
                    state['step']=step;latest_step=step;imported+=1
            # Actual rollout outcomes: recent within each N, separate from fixed validation.
            episodes=list(records(experiment_root/'ppo/episodes.jsonl'))
            if latest_step>state.get('episode_step',-1):
                for n in range(2,9):
                    rows=[e for e in episodes if e.get('n')==n and e.get('steps',0)<=latest_step][-100:]
                    if rows:
                        self.scalar(writer,f'Training_by_N/N{n}/win_rate',sum(e['win'] for e in rows)/len(rows),latest_step)
                        self.scalar(writer,f'Training_by_N/N{n}/net_kills',sum(e['netKills'] for e in rows)/len(rows),latest_step)
                        self.scalar(writer,f'Training_by_N/N{n}/episodes_in_window',len(rows),latest_step)
                state['episode_step']=latest_step
            validation_files=[(0,experiment_root/'validation-bc.json')]
            validation_files.extend((int(p.stem.split('-')[-1]),p) for p in sorted((experiment_root/'ppo').glob('validation-*.json')))
            for step,path in validation_files:
                key=path.name
                if key in state['validations']:continue
                result=read_json(path)
                if not result:continue
                for field in ['win_rate','net_kills','timeout_rate']:
                    self.scalar(writer,'Evaluation/'+field,result.get(field),step,path.stat().st_mtime)
                for n,row in result.get('by_n',{}).items():
                    for field in ['win_rate','net_kills']:
                        self.scalar(writer,f'Evaluation_by_N/N{n}/'+field,row.get(field),step)
                state['validations'].append(key);imported+=1
            baseline=read_json(experiment_root/'validation-teacher.json')
            baseline_step=max(0,latest_step)
            if baseline and baseline_step>state['baseline_step']:
                baseline_writer=self.writer(teacher)
                points=[0,baseline_step] if state['baseline_step']<0 and baseline_step>0 else [baseline_step]
                for step in points:
                    for field in ['win_rate','net_kills','timeout_rate']:
                        self.scalar(baseline_writer,'Evaluation/'+field,baseline.get(field),step)
                state['baseline_step']=baseline_step
        for writer in self.writers.values():writer.flush()
        temp=self.state_path.with_suffix('.tmp');temp.write_text(json.dumps(self.state,indent=2));temp.replace(self.state_path)
        return imported
    def close(self):
        for writer in self.writers.values():writer.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/xy/rl_for_fun'))
    p.add_argument('--logdir',type=Path);p.add_argument('--interval',type=float,default=5)
    p.add_argument('--once',action='store_true');args=p.parse_args()
    logdir=args.logdir or args.root/'runs/initial60/tensorboard'
    logdir.mkdir(parents=True,exist_ok=True)
    lock=(logdir/'writer.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('TensorBoard log importer is already running; leaving it unchanged.',flush=True)
        return
    importer=Importer(args.root,logdir)
    try:
        while True:
            count=importer.update()
            if count:print(json.dumps({'time':time.time(),'new_records':count,'state':importer.state}),flush=True)
            if args.once:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass
    finally:importer.close()


if __name__=='__main__':main()
