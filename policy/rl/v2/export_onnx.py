"""Export the exact deterministic V2 actor and validate its browser deployment contract."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from .model import load_policy

def atan2_symbolic(g,y,x):
    # The legacy exporter divides y/x and yields NaN for padded (0, 0) tokens.
    # Preserve PyTorch atan2(0,0)==0 with an explicit zero-axis branch.
    constant=lambda v:g.op('Constant',value_t=torch.tensor(v,dtype=torch.float32))
    zero=constant(0.);one=constant(1.);pi=constant(np.pi);half=constant(np.pi/2)
    xzero=g.op('Equal',x,zero);safe_x=g.op('Where',xzero,one,x)
    angle=g.op('Atan',g.op('Div',y,safe_x))
    adjusted=g.op('Where',g.op('Greater',x,zero),angle,g.op('Where',g.op('GreaterOrEqual',y,zero),g.op('Add',angle,pi),g.op('Sub',angle,pi)))
    axis=g.op('Where',g.op('Greater',y,zero),half,g.op('Where',g.op('Less',y,zero),g.op('Neg',half),zero))
    return g.op('Where',xzero,axis,adjusted)

class InferenceActor(torch.nn.Module):
    def __init__(self,actor):super().__init__();self.actor=actor
    def forward(self,obs):
        h,drive_logits,turn_logits,magnitude=self.actor.motion(obs)
        drive=drive_logits.argmax(-1).to(obs.dtype)-1
        steer=(turn_logits.argmax(-1).to(obs.dtype)-1)*magnitude.sigmoid()
        movement=torch.stack([drive,steer],dim=-1)
        fire=self.actor.fire_logits(obs,h,movement)
        shoot=((fire>=0)&(obs[:,16]>.5)).to(obs.dtype)
        return torch.stack([drive,steer,shoot],dim=-1),torch.cat([drive_logits,turn_logits,magnitude[:,None],fire[:,None]],dim=-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--id',choices=['rl_win','rl_frag'],required=True)
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    actor,metadata=load_policy(a.model);actor.eval();wrapper=InferenceActor(actor).eval()
    states=[]
    for file in sorted(a.data.glob('episode-*.npz')):
        with np.load(file) as d:
            if bool(d['validation']):states.extend(d['obs'][::max(1,len(d['obs'])//16)][:16])
        if len(states)>=256:break
    states=np.stack(states[:256]+[np.zeros(1966,dtype=np.float32),np.ones(1966,dtype=np.float32)*0])
    # Exercise a mechanically ready empty state too.
    states[-1,6]=1;states[-1,16]=1;states[-1,28]=3.3/60
    expected=[];logits=[]
    with torch.inference_mode():
        for obs in states:
            x=torch.from_numpy(obs[None]);control,raw=wrapper(x)
            assert torch.equal(control,actor.act(x,deterministic=True)[3])
            expected.append(control[0].numpy());logits.append(raw[0].numpy())
    expected=np.stack(expected);logits=np.stack(logits)
    # Export primitive attention operators, without changing weights or decoding.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.register_custom_op_symbolic('aten::atan2',atan2_symbolic,17)
    with torch.no_grad():
        torch.onnx.export(wrapper,torch.from_numpy(states[:1]),str(a.out),input_names=['obs'],output_names=['controls','logits'],
            opset_version=17,dynamo=False,do_constant_folding=True)
    onnx.checker.check_model(onnx.load(a.out))
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(a.out),options,providers=['CPUExecutionProvider'])
    decoded=[];raw=[]
    for state in states:
        c,l=session.run(None,{'obs':state[None]});decoded.append(c[0]);raw.append(l[0])
    decoded=np.stack(decoded);raw=np.stack(raw)
    if not np.isfinite(decoded).all() or not np.isfinite(raw).all():raise RuntimeError('Nonfinite ONNX output')
    if not np.array_equal(decoded[:,[0,2]],expected[:,[0,2]]):raise RuntimeError('Export changed discrete actions')
    error=float(np.max(np.abs(decoded-expected)));logit_error=float(np.max(np.abs(raw-logits)))
    if error>1e-4 or logit_error>2e-4:raise RuntimeError(f'Export mismatch {error}, {logit_error}')
    # Portable fixtures are replayed by the browser and by the original Node engine.
    states.tofile(a.out.with_suffix('.observations.f32'))
    fixture=dict(rows=len(states),observation_size=1966,controls=expected.tolist(),logits=logits.tolist())
    a.out.with_suffix('.parity.json').write_text(json.dumps(fixture))
    report=dict(id=a.id,source_checkpoint=str(a.model),source_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        onnx_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),architecture=metadata['architecture'],observation_version=2,
        training='BC + one DAgger round; PPO not started',candidate=a.model.parents[1].name,checkpoint_step=metadata['step'],
        deterministic=True,policy_hz=60,round_seconds=60,export_bias_adjustment=0,verification_rows=len(states),
        max_control_error=error,max_logit_error=logit_error,discrete_actions_identical=True)
    a.out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
