"""Read-only progress summary, independent of PyTorch and the training process."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/xy/rl_for_fun'))
    args=p.parse_args();result={'observed_utc':datetime.now(timezone.utc).isoformat(),'experiments':{}}
    for objective in ['win','frag']:
        root=args.root/'runs/initial60'/objective
        experiment=read_json(root/'experiment.json')
        stages={name:read_json(root/name/'status.json') for name in ['demonstrations','bc','ppo']}
        validations={name:read_json(root/('validation-'+name+'.json')) for name in ['teacher','bc']}
        validations={name:{k:v for k,v in value.items() if k!='records'} for name,value in validations.items() if value}
        result['experiments'][objective]={'experiment':experiment,'stages':stages,'validations':validations}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
