"""Read-only overview; no training, restart, or checkpoint changes."""
import argparse,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('/home/xy/rl_for_fun/runs/v2b'));a=p.parse_args()
    for filename in ['pipeline.json','G0.json','gpu-allocation.json']:
        path=a.run/filename
        if path.exists():print(filename,json.dumps(json.loads(path.read_text()),ensure_ascii=False))
    for path in sorted(a.run.rglob('status.json')):
        if 'bc-initial' in path.parts or 'skills-initial' in path.parts:continue
        value=json.loads(path.read_text());value['last_update_seconds_ago']=round(time.time()-path.stat().st_mtime,1)
        print(str(path.relative_to(a.run)),json.dumps(value,ensure_ascii=False))
    for path in sorted(a.run.glob('*/gates.json')):
        value=json.loads(path.read_text());print(str(path.relative_to(a.run)),json.dumps({
            'ppo_allowed':value['ppo_allowed'],'gates':{k:v['passed'] for k,v in value['gates'].items()}}))

if __name__=='__main__':main()
