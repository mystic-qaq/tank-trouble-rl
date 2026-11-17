"""Read-only behavioral audit of existing trajectories; no policy execution or training."""
import argparse,json,time
from pathlib import Path
import numpy as np

def motion_audit(path):
    counts=np.zeros(3,dtype=np.int64);frames=windows=stationary=0
    for line in (path/'trajectories.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r.get('skill')!='combat':continue
        data=np.load(path/r['file'],mmap_mode='r');x=data[:,:1966];actions=x[1:,11:14];frames+=len(actions)
        for i,value in enumerate([-1,0,1]):counts[i]+=int((actions[:,0]==value).sum())
        xy=x[:,:2]*np.array([1280,800])
        for start in range(0,len(x)-180,180):
            windows+=1
            if np.linalg.norm(xy[start:start+180]-xy[start],axis=1).max()<24 and x[start:start+180,13].sum()==0:stationary+=1
    return dict(frames=frames,drive_reverse_idle_forward=counts.tolist(),three_second_windows=windows,
        stationary_no_request_windows=stationary,stationary_no_request_fraction=stationary/max(1,windows),
        interpretation='position stays within 24px and no requested fire; diagnostic, not proof that waiting is tactically wrong')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('/home/xy/rl_for_fun/runs/v3'));p.add_argument('--out',type=Path);p.add_argument('--tensorboard',action='store_true');a=p.parse_args()
    report=dict(reviewed_unix=time.time(),read_only=True,evaluations={},motion={})
    for parent in ['shared','win','frag']:
        base=a.run/parent
        for file in sorted(base.glob('**/results.json')):
            if 'acceptance' in file.parts:continue
            data=json.loads(file.read_text())
            report['evaluations'][str(file.relative_to(a.run))]={k:data[k] for k in ['rounds','win','frag','emissions','hits','suicides','spin_fraction'] if k in data}
    for path in sorted(a.run.glob('*/dagger-*')):
        if (path/'manifest.json').exists():report['motion'][str(path.relative_to(a.run))]=motion_audit(path)
    if a.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        for goal in ['win','frag']:
            with SummaryWriter(str(a.run/goal/'audit/tb')) as writer:
                for key,value in report['motion'].items():
                    if not key.startswith(goal+'/'):continue
                    step=int(key.rsplit('-',1)[1])-1
                    writer.add_scalar('diagnostics/stationary_no_fire_window_fraction',value['stationary_no_request_fraction'],step)
                    writer.add_scalar('diagnostics/drive_forward_fraction',value['drive_reverse_idle_forward'][2]/max(1,value['frames']),step)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
