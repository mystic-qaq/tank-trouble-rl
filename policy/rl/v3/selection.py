"""Keep the best measured usable initialization across all completed curriculum rounds."""
from pathlib import Path
import hashlib
import torch
from .model import ARCHITECTURE

def select_initialization(run,objective):
    run=Path(run);candidates=[]
    for path in [run/'shared/best.pt',*sorted((run/objective).glob('adapt-*/best.pt'))]:
        if not path.exists():continue
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data.get('architecture')!=ARCHITECTURE:raise ValueError('Initialization architecture mismatch')
        report=data.get('closed_loop')
        if not report:continue
        usable=report.get('rounds',0)>=70 and report['hits']>=3 and report['emissions']>=7 and report['spin_fraction']<.8
        candidates.append(dict(path=str(path),usable=usable,score=report[objective],win=report['win'],net=report['frag'],
            hits=report['hits'],emissions=report['emissions'],spin_fraction=report['spin_fraction'],step=data['step']))
    eligible=[c for c in candidates if c['usable']]
    if not eligible:return None,candidates
    best=max(eligible,key=lambda c:(c['score'],-c['spin_fraction'],c['net'] if objective=='win' else c['win']))
    best={**best,'sha256':hashlib.sha256(Path(best['path']).read_bytes()).hexdigest()}
    return best,candidates
