"""V2: relational geometry encoder, signed continuous turning, motion-conditioned fire."""
import math
import torch
from torch import nn
from torch.distributions import Categorical,Normal,Bernoulli

OBS_SIZE=1966
ARCHITECTURE='relational_motion_fire_v2'

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_encoder=nn.Sequential(nn.Linear(80,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU())
        self.map_encoder=nn.Sequential(nn.Conv2d(2,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,32,3,padding=1),nn.SiLU(),nn.Flatten(),nn.Linear(32*11*17,128),nn.SiLU())
        self.enemy=nn.Sequential(nn.Linear(24,96),nn.SiLU(),nn.Linear(96,96))
        self.bullet=nn.Sequential(nn.Linear(24,96),nn.SiLU(),nn.Linear(96,96))
        self.aim=nn.Sequential(nn.Linear(12,96),nn.SiLU(),nn.Linear(96,96))
        self.query=nn.Linear(256,96)
        self.attention=nn.ModuleList([nn.MultiheadAttention(96,4,batch_first=True) for _ in range(3)])
        self.trunk=nn.Sequential(nn.Linear(544,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
    @staticmethod
    def attend(query,entities,encoder,attention):
        valid=entities[...,-1]>0
        # A dummy zero token makes an empty entity set well-defined without pretending padding is real.
        tokens=encoder(entities)
        tokens=torch.cat([tokens,torch.zeros_like(tokens[:,:1])],dim=1)
        mask=torch.cat([~valid,valid.any(dim=1,keepdim=True)],dim=1)
        return attention(query,tokens,tokens,key_padding_mask=mask,need_weights=False)[0].squeeze(1)
    def forward(self,x):
        self_features=self.self_encoder(x[:,:80])
        map_features=self.map_encoder(x[:,80:454].reshape(-1,2,11,17))
        base=torch.cat([self_features,map_features],dim=-1)
        query=self.query(base).unsqueeze(1)
        enemy=self.attend(query,x[:,454:622].reshape(-1,7,24),self.enemy,self.attention[0])
        bullet=self.attend(query,x[:,622:1582].reshape(-1,40,24),self.bullet,self.attention[1])
        aim=self.attend(query,x[:,1582:].reshape(-1,32,12),self.aim,self.attention[2])
        return self.trunk(torch.cat([base,enemy,bullet,aim],dim=-1))

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.drive=nn.Linear(128,3)
        self.turn=nn.Linear(128,3)
        self.magnitude=nn.Linear(128,1)
        self.log_std=nn.Parameter(torch.tensor([-2.5]))
        self.fire_candidate=nn.Sequential(nn.Linear(16,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.fire_query=nn.Linear(132,64)
        self.fire=nn.Sequential(nn.Linear(201,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        for head in [self.drive,self.turn,self.magnitude]:
            nn.init.normal_(head.weight,std=.01);nn.init.zeros_(head.bias)
    def motion(self,x):
        h=self.encoder(x)
        return h,self.drive(h),self.turn(h),self.magnitude(h).squeeze(-1)
    def fire_logits(self,x,h,controls):
        drive,steer=controls[:,0],controls[:,1]
        delta=steer*x[:,28]
        speed=x[:,4]*178
        target=torch.where(drive>0,torch.full_like(speed,178),torch.full_like(speed,-178*.66))
        rate=torch.where(speed*drive<0,torch.full_like(speed,1150),torch.full_like(speed,920))/60
        advancing=speed+torch.clamp(target-speed,-rate,rate)
        braking=speed.sign()*(speed.abs()-1150*.75/60).clamp(min=0)
        next_speed=torch.where(drive==0,braking,advancing)
        movement=next_speed/60
        aims=x[:,1582:].reshape(-1,32,12)
        alpha=torch.atan2(aims[:,:,0],aims[:,:,1])
        length=aims[:,:,2]*math.hypot(1280,800)
        rx=length*alpha.cos()-movement[:,None]*delta.cos()[:,None]
        ry=length*alpha.sin()-movement[:,None]*delta.sin()[:,None]
        error=torch.atan2(torch.sin(torch.atan2(ry,rx)-delta[:,None]),torch.cos(torch.atan2(ry,rx)-delta[:,None]))
        cone=(aims[:,:,10]*math.pi).clamp(min=.003)
        normalized=(error/cone).clamp(-12,12)
        augmented=torch.cat([aims,normalized.unsqueeze(-1),error.sin().unsqueeze(-1),error.cos().unsqueeze(-1),(normalized.abs()<1).float().unsqueeze(-1)],dim=-1)
        encoded=self.fire_candidate(augmented)
        motion=torch.stack([drive,steer,delta,next_speed/178],dim=-1)
        query=self.fire_query(torch.cat([h,motion],dim=-1))
        valid=aims[:,:,-1]>0
        score=(encoded*query[:,None]).sum(-1)/8
        score=score.masked_fill(~valid,-1e9)
        weights=score.softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp(min=1e-6)
        pooled=(weights[:,:,None]*encoded).sum(1)
        useful=valid&(aims[:,:,5]>.5)&(aims[:,:,6]<.5)
        minimum=normalized.abs().masked_fill(~useful,12).amin(1)/12
        aligned=(useful&(normalized.abs()<1)).float().sum(1)/7
        danger=(valid&(aims[:,:,6]>.5)&(normalized.abs()<1)).float().sum(1)/7
        summary=torch.stack([minimum,aligned,danger,x[:,5],x[:,7]],dim=-1)
        return self.fire(torch.cat([h,motion,pooled,summary],dim=-1)).squeeze(-1)
    def act(self,x,actions=None,deterministic=False):
        h,drive_logits,turn_logits,mean=self.motion(x)
        d=Categorical(logits=drive_logits,validate_args=False)
        t=Categorical(logits=turn_logits,validate_args=False)
        n=Normal(mean,self.log_std.clamp(-5,-.5).exp().expand_as(mean),validate_args=False)
        if actions is None:
            di=drive_logits.argmax(-1) if deterministic else d.sample()
            ti=turn_logits.argmax(-1) if deterministic else t.sample()
            z=mean if deterministic else n.sample()
        else:di,ti,z,_=actions.unbind(-1)
        direction=ti-1
        steer=direction*z.sigmoid()
        controls=torch.stack([di-1,steer],dim=-1)
        fire_logits=self.fire_logits(x,h,controls)
        f=Bernoulli(logits=fire_logits,validate_args=False)
        ready=x[:,16]>.5
        fi=((fire_logits>=0).float() if deterministic else f.sample()) if actions is None else actions[:,3]
        fi=fi*ready
        stored=torch.stack([di.float(),ti.float(),z,fi],dim=-1)
        # sign is a discrete branch; magnitude density is only used on non-zero branches.
        jac=-nn.functional.softplus(-z)-nn.functional.softplus(z)
        moving=(direction!=0).float()
        logprob=d.log_prob(di.long())+t.log_prob(ti.long())+moving*(n.log_prob(z)-jac)+ready*f.log_prob(fi)
        sampled=n.rsample()
        magnitude_entropy=n.entropy()-nn.functional.softplus(-sampled)-nn.functional.softplus(sampled)
        entropy=torch.stack([d.entropy(),t.entropy(),(1-t.probs[:,1])*magnitude_entropy,ready*f.entropy()],dim=-1)
        controls=torch.cat([controls,fi[:,None]],dim=-1)
        return stored,logprob,entropy,controls

class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.value=nn.Linear(128,1)
    def forward(self,x):return self.value(self.encoder(x)).squeeze(-1)

def checkpoint(model,**metadata):
    return dict(architecture=ARCHITECTURE,observation_version=2,round_limit=60,model=model.state_dict(),**metadata)

def load_policy(path,device='cpu'):
    data=torch.load(path,map_location=device,weights_only=False)
    if data.get('architecture')!=ARCHITECTURE:raise ValueError('Wrong V2 architecture')
    model=Policy().to(device);model.load_state_dict(data['model']);return model,data
