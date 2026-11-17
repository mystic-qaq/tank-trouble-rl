"""One production architecture: ~5M recurrent actor, joint motion, conditional fire."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical,Normal,Bernoulli
from ..v2.model import Encoder as SmallEncoder,Policy as FireGeometry

OBS_SIZE=1966
HIDDEN=256
ARCHITECTURE='joint_motion_gru_v3'

def configure_precision():
    # cuDNN may choose different TF32 kernels for one-step and sequence batches.
    # PPO must evaluate the same policy numerically in both paths.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)

class Encoder(SmallEncoder):
    def __init__(self):
        super().__init__()
        self.self_encoder=nn.Sequential(nn.Linear(80,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.map_encoder=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,64,3,padding=1),nn.SiLU(),nn.Flatten(),nn.Linear(64*11*17,256),nn.SiLU())
        self.enemy=nn.Sequential(nn.Linear(24,128),nn.SiLU(),nn.Linear(128,128))
        self.bullet=nn.Sequential(nn.Linear(24,128),nn.SiLU(),nn.Linear(128,128))
        self.aim=nn.Sequential(nn.Linear(12,128),nn.SiLU(),nn.Linear(128,128))
        self.query=nn.Linear(512,128)
        self.attention=nn.ModuleList([nn.MultiheadAttention(128,4,batch_first=True) for _ in range(3)])
        self.trunk=nn.Sequential(nn.Linear(896,512),nn.SiLU(),nn.Linear(512,HIDDEN),nn.SiLU())

class Policy(FireGeometry):
    def __init__(self):
        configure_precision()
        super().__init__();self.encoder=Encoder();self.memory=nn.GRU(HIDDEN,HIDDEN,batch_first=True)
        self.project=nn.Sequential(nn.Linear(HIDDEN,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        del self.drive,self.turn,self.magnitude,self.log_std
        self.joint=nn.Linear(128,9);self.magnitudes=nn.Linear(128,9);self.log_std=nn.Parameter(torch.full((9,),-2.0))
        nn.init.normal_(self.joint.weight,std=.01);nn.init.zeros_(self.joint.bias)
        nn.init.normal_(self.magnitudes.weight,std=.01);nn.init.zeros_(self.magnitudes.bias)
    def initial(self,batch,device=None):return torch.zeros(1,batch,HIDDEN,device=device or next(self.parameters()).device)
    def features(self,obs,hidden=None):
        if obs.ndim==2:obs=obs[:,None,:]
        batch,time,_=obs.shape
        encoded=self.encoder(obs.reshape(-1,OBS_SIZE)).reshape(batch,time,HIDDEN)
        sequence,last=self.memory(encoded,hidden)
        return self.project(sequence),last
    def distribution(self,obs,h,actions=None,deterministic=False):
        joint_logits=self.joint(h);means=self.magnitudes(h)
        category=Categorical(logits=joint_logits,validate_args=False)
        choice=(joint_logits.argmax(-1) if deterministic else category.sample()) if actions is None else actions[:,0].long()
        mean=means.gather(1,choice[:,None]).squeeze(1)
        normal=Normal(mean,self.log_std.clamp(-4,-.5)[choice].exp(),validate_args=False)
        z=(mean if deterministic else normal.sample()) if actions is None else actions[:,1]
        drive=choice.div(3,rounding_mode='floor')-1;direction=choice.remainder(3)-1
        steer=direction*z.sigmoid();motion=torch.stack([drive,steer],-1)
        logits=self.fire_logits(obs,h,motion);fire=Bernoulli(logits=logits,validate_args=False);ready=obs[:,16]>.5
        shoot=((logits>=0).float() if deterministic else fire.sample()) if actions is None else actions[:,2]
        shoot=shoot*ready
        stored=torch.stack([choice.float(),z,shoot],-1)
        jac=-F.softplus(-z)-F.softplus(z)
        lp=category.log_prob(choice)+(direction!=0)*(normal.log_prob(z)-jac)+ready*fire.log_prob(shoot)
        # Analytic joint entropy, bounded-turn differential entropy, Bernoulli entropy.
        all_normal=Normal(means,self.log_std.clamp(-4,-.5).exp()[None].expand_as(means),validate_args=False)
        sample=all_normal.rsample();mag_entropy=all_normal.entropy()-F.softplus(-sample)-F.softplus(sample)
        branch=(torch.arange(9,device=obs.device)%3!=1)
        magnitude_entropy=(category.probs*branch*mag_entropy).sum(-1)
        entropy=torch.stack([category.entropy(),magnitude_entropy,ready*fire.entropy()],-1)
        return stored,lp,entropy,torch.stack([drive,steer,shoot],-1)
    def act(self,obs,hidden=None,actions=None,deterministic=False):
        h,last=self.features(obs,hidden)
        stored,lp,entropy,controls=self.distribution(obs,h[:,0],actions,deterministic)
        return stored,lp,entropy,controls,last
    def evaluate_actions(self,obs,hidden,actions):
        h,last=self.features(obs,hidden);batch,time,_=obs.shape
        _,lp,entropy,_=self.distribution(obs.reshape(-1,OBS_SIZE),h.reshape(-1,128),actions.reshape(-1,3))
        return lp.reshape(batch,time),entropy.reshape(batch,time,3),last
    def forward(self,obs,actions,mask,quality):
        """Sequence behavior-cloning loss, also used as the DDP forward entrypoint."""
        h,_=self.features(obs);batch,time,_=obs.shape;x=obs.reshape(-1,OBS_SIZE);a=actions.reshape(-1,3);h=h.reshape(-1,128)
        weights=mask.reshape(-1)*quality[:,None].expand(batch,time).reshape(-1)
        sign=torch.where(a[:,1].abs()>1e-4,a[:,1].sign(),torch.zeros_like(a[:,1])).long()
        target=(a[:,0].long()+1)*3+(sign+1)
        motion_weight=weights*(1+1.5*(a[:,0]!=0))
        joint_loss=(F.cross_entropy(self.joint(h),target,reduction='none',label_smoothing=.01)*motion_weight).sum()/motion_weight.sum().clamp(min=1)
        magnitude=self.magnitudes(h).gather(1,target[:,None]).squeeze(1);turn_weight=weights*(a[:,1].abs()>1e-4)
        magnitude_loss=(F.binary_cross_entropy_with_logits(magnitude,a[:,1].abs(),reduction='none')*turn_weight).sum()/turn_weight.sum().clamp(min=1)
        fire_logits=self.fire_logits(x,h,a[:,:2]);fire_weight=weights*(x[:,16]>.5)
        fire_loss=(F.binary_cross_entropy_with_logits(fire_logits,a[:,2],reduction='none',pos_weight=x.new_tensor(8.))*fire_weight).sum()/fire_weight.sum().clamp(min=1)
        return joint_loss+magnitude_loss+fire_loss,torch.stack([joint_loss,magnitude_loss,fire_loss]).detach()

class Critic(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=Encoder();self.memory=nn.GRU(HIDDEN,HIDDEN,batch_first=True)
        self.value=nn.Sequential(nn.Linear(HIDDEN,128),nn.SiLU(),nn.Linear(128,1))
    def forward(self,obs,hidden=None):
        batch,time,_=obs.shape;encoded=self.encoder(obs.reshape(-1,OBS_SIZE)).reshape(batch,time,HIDDEN)
        seq,last=self.memory(encoded,hidden);return self.value(seq).squeeze(-1),last

def checkpoint(model,**meta):
    return dict(architecture=ARCHITECTURE,observation_version=2,hidden_size=HIDDEN,round_limit=60,precision_mode='ieee_fp32_no_tf32',model=model.state_dict(),**meta)

def load_policy(path,device='cpu'):
    data=torch.load(path,map_location=device,weights_only=False)
    if data.get('architecture')!=ARCHITECTURE:raise ValueError('Not a V3 recurrent checkpoint')
    model=Policy().to(device);model.load_state_dict(data['model']);return model,data
