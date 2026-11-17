"""Read-only audit of demonstration imitation, learned actions, and fixed validation history."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model import load_policy


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def behavior(model,x,y,device):
    values=[];drives=[];steers=[];shoots=[]
    with torch.no_grad():
        for start in range(0,len(x),1024):
            d,s,f,v=model(torch.from_numpy(x[start:start+1024]).to(device))
            drives.append(d.argmax(-1).cpu().numpy()-1)
            steers.append(s.tanh().cpu().numpy());shoots.append(f.sigmoid().cpu().numpy());values.append(v.cpu().numpy())
    drive=np.concatenate(drives);steer=np.concatenate(steers);shoot=np.concatenate(shoots);value=np.concatenate(values)
    positive=y[:,2]>.5;predicted=shoot>=.5;moving=np.abs(y[:,1])>.2
    ready=(x[:,5]<=1e-5)&(x[:,7]<.999)
    thresholds={}
    for threshold in [.5,.2,.1,.05,1/21,.02,.01]:
        selected=shoot>=threshold
        thresholds[str(round(threshold,6))]=dict(rate=float(selected.mean()),
            recall=float(selected[positive].mean()) if positive.any() else None,
            precision=float(positive[selected].mean()) if selected.any() else None)
    return dict(drive_accuracy=float(np.mean(drive==y[:,0])),drive_stop_rate=float(np.mean(drive==0)),
        steer_mae=float(np.mean(np.abs(steer-y[:,1]))),steer_rmse=float(np.sqrt(np.mean((steer-y[:,1])**2))),
        steer_abs_mean=float(np.abs(steer).mean()),
        steer_sign_accuracy_on_turns=float(np.mean(np.sign(steer[moving])==np.sign(y[moving,1]))) if moving.any() else None,
        shoot_threshold_rate=float(predicted.mean()),shoot_recall=float(np.mean(predicted[positive])) if positive.any() else None,
        shoot_precision=float(np.mean(positive[predicted])) if predicted.any() else None,
        shoot_probability_on_teacher_shots=float(shoot[positive].mean()) if positive.any() else None,
        shoot_probability_on_teacher_nonshots=float(shoot[~positive].mean()) if (~positive).any() else None,
        shoot_probability_when_ready=float(shoot[ready].mean()) if ready.any() else None,
        shoot_max_probability=float(shoot.max()),shoot_threshold_sweep=thresholds,
        value_mean=float(value.mean()),value_std=float(value.std()),
        steer_latent_std=float(model.log_std.clamp(-4,0).exp().item()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/xy/rl_for_fun'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4);report={}
    for objective in ['win','frag']:
        root=args.root/'runs/initial60'/objective
        shards=[np.load(f) for f in sorted((root/'demonstrations').glob('shard-*.npz'))]
        x=np.concatenate([s['obs'][s['validation']] for s in shards])
        y=np.concatenate([s['actions'][s['validation']] for s in shards])
        teacher=dict(samples=len(x),positive_shots=int(y[:,2].sum()),
            train_samples=sum(int((~s['validation']).sum()) for s in shards),
            train_positive_shots=sum(int(s['actions'][~s['validation'],2].sum()) for s in shards),
            shoot_rate=float(y[:,2].mean()),drive_stop_rate=float(np.mean(y[:,0]==0)),
            steer_abs_mean=float(np.abs(y[:,1]).mean()))
        result={'teacher_holdout':teacher,'checkpoints':{},'validations':[]}
        for label,path in [('bc',root/'bc/best.pt'),('ppo',root/'ppo/latest.pt')]:
            model,ckpt=load_policy(path,args.device);model.eval()
            result['checkpoints'][label]=dict(steps=ckpt.get('steps',0),epoch=ckpt.get('epoch'),
                shoot_positive_weight=ckpt.get('shoot_positive_weight'),behavior=behavior(model,x,y,args.device))
        for path in [root/'validation-teacher.json',root/'validation-bc.json',*sorted((root/'ppo').glob('validation-*.json'))]:
            validation=read(path)
            if validation:
                rows=validation['records']
                result['validations'].append(dict(file=path.name,episodes=len(rows),win_rate=validation['win_rate'],
                    net_kills=validation['net_kills'],kills=float(np.mean([r['kills'] for r in rows])),
                    suicides=float(np.mean([r['suicides'] for r in rows])),timeout_rate=validation['timeout_rate'],
                    by_n=validation['by_n']))
        episodes=[json.loads(line) for line in (root/'ppo/episodes.jsonl').read_text().splitlines() if line.strip()]
        result['training_windows']=[]
        for threshold in range(100000,1100000,100000):
            rows=[r for r in episodes if threshold-100000<r['steps']<=threshold]
            if rows:
                result['training_windows'].append(dict(end_step=threshold,episodes=len(rows),win_rate=float(np.mean([r['win'] for r in rows])),
                    kills=float(np.mean([r['kills'] for r in rows])),suicides=float(np.mean([r['suicides'] for r in rows])),
                    net_kills=float(np.mean([r['netKills'] for r in rows])),duration=float(np.mean([r['duration'] for r in rows])),
                    by_n={n:sum(r['n']==n for r in rows) for n in range(2,9)}))
        report[objective]=result
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
