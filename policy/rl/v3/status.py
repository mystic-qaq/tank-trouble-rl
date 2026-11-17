import argparse,json,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=Path('/home/xy/rl_for_fun/runs/v3'));a=p.parse_args()
    for name in ['pipeline.json','launch.json','contracts.json']:
        file=a.run/name
        if file.exists():print(name,json.dumps(json.loads(file.read_text()),ensure_ascii=False))
    for file in sorted(a.run.rglob('status.json')):
        if any(part in ['skills-failed','smoke'] for part in file.parts):continue
        value=json.loads(file.read_text());value['last_update_seconds_ago']=round(time.time()-file.stat().st_mtime,1)
        print(str(file.relative_to(a.run)),json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
