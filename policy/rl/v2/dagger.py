"""Query shadow teachers on student states; retain each full teacher joint action."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .model import load_policy
from .evaluate import run_evaluation,combat_tasks,skill_tasks
from .collect import atomic_json,is_validation

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--teacher',choices=['hy','gpt'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=730001001);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda:0');p.add_argument('--combat-limit',type=int,default=0);p.add_argument('--skill-count',type=int,default=8)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    model,_=load_policy(a.model,a.device);model.eval();rng=np.random.default_rng(a.seed);saved=0;positives=0;frames=0
    if list(a.out.glob('episode-*.npz')):raise ValueError('Fresh DAgger directory required')
    combat=combat_tasks(1,a.seed)
    if a.combat_limit:combat=combat[:a.combat_limit]
    for suite,tasks,teacher in [('combat',combat,a.teacher),('skills',skill_tasks(a.skill_count,a.seed+900001),'reference')]:
        rows,records,_=run_evaluation(tasks,model,workers=a.workers,device=a.device,early_death=True,trajectory=True,label_teacher=teacher,stochastic=False)
        for index,row in enumerate(rows):
            observations=[];labels=[];probabilities=[]
            for obs,action,logprob,reward,active,label in records[index]:
                frames+=1
                if not active or label is None:continue
                probability=1. if label[2] else .12 if obs[16]>.5 else .025
                if rng.random()>=probability:continue
                observations.append(obs);labels.append(label);probabilities.append(probability)
            if not observations:continue
            labels=np.asarray(labels,dtype=np.float32);positives+=int(labels[:,2].sum())
            np.savez_compressed(a.out/f'episode-{saved:06d}.npz',obs=np.stack(observations),actions=labels,
                probability=np.asarray(probabilities,dtype=np.float32),eligible=np.array([x[16]>.5 for x in observations]),
                validation=np.array(is_validation(row['seed'])),seed=np.array(row['seed']),skill=np.array(row.get('skill') or 'combat'))
            saved+=1
        atomic_json(a.out/'status.json',dict(stage='dagger',status='running',trajectories=saved,positive_counterfactual_labels=positives,frames=frames))
    atomic_json(a.out/'manifest.json',dict(stage='dagger',status='complete',trajectories=saved,positive_counterfactual_labels=positives,frames=frames,
        label='teacher_joint_action_on_student_state_fire_resolved_after_teacher_motion',source='counterfactual_not_actual_emission',seed=a.seed))

if __name__=='__main__':main()
