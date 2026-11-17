"""Shared architecture, independently trained weights for win and frag objectives."""
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal

OBS_SIZE = 926
ARCHITECTURE = "entity_pool_v1"

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_encoder = nn.Sequential(nn.Linear(16,64),nn.Tanh())
        self.map_encoder = nn.Sequential(nn.Linear(346,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.enemy_encoder = nn.Sequential(nn.Linear(12,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
        self.bullet_encoder = nn.Sequential(nn.Linear(12,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh())
        self.trunk = nn.Sequential(nn.Linear(256,256),nn.Tanh(),nn.Linear(256,128),nn.Tanh())
        self.drive = nn.Linear(128,3)
        self.steer = nn.Linear(128,1)
        self.shoot = nn.Linear(128,1)
        self.value = nn.Linear(128,1)
        self.log_std = nn.Parameter(torch.tensor([-0.7]))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,2**0.5)
                nn.init.zeros_(layer.bias)
        for layer in [self.drive,self.steer,self.shoot]:
            nn.init.orthogonal_(layer.weight,0.01)
        nn.init.orthogonal_(self.value.weight,1.0)

    @staticmethod
    def pool(entities, encoder):
        mask = entities[...,-1:] > 0
        encoded = encoder(entities)
        mean = (encoded*mask).sum(1)/mask.sum(1).clamp(min=1)
        maximum = encoded.masked_fill(~mask,-1e9).amax(1)
        maximum = torch.where(mask.any(1),maximum,torch.zeros_like(maximum))
        return torch.cat([mean,maximum],dim=-1)

    def forward(self, obs):
        enemy = obs[:,362:446].reshape(-1,7,12)
        bullet = obs[:,446:].reshape(-1,40,12)
        features = torch.cat([self.self_encoder(obs[:,:16]),self.map_encoder(obs[:,16:362]),
            self.pool(enemy,self.enemy_encoder),self.pool(bullet,self.bullet_encoder)],dim=-1)
        h = self.trunk(features)
        return self.drive(h),self.steer(h).squeeze(-1),self.shoot(h).squeeze(-1),self.value(h).squeeze(-1)

    def act(self, obs, actions=None, deterministic=False):
        drive,mean,shoot,value = self(obs)
        d = Categorical(logits=drive)
        s = Normal(mean,self.log_std.clamp(-4,0).exp().expand_as(mean))
        f = Bernoulli(logits=shoot)
        if actions is None:
            di = drive.argmax(-1) if deterministic else d.sample()
            z = mean if deterministic else s.sample()
            fi = (shoot>=0).float() if deterministic else f.sample()
            actions = torch.stack([di.float(),z,fi],dim=-1)
        di,z,fi = actions.unbind(-1)
        steer = z.tanh()
        # Stable log(1-tanh(z)^2); all stored PPO actions include the original latent z.
        jacobian = 2*(torch.log(torch.tensor(2.,device=z.device))-z-nn.functional.softplus(-2*z))
        logprob = d.log_prob(di.long())+s.log_prob(z)-jacobian+f.log_prob(fi)
        entropy = d.entropy()+s.entropy()+f.entropy()
        controls = torch.stack([di-1,steer,fi],dim=-1)
        return actions,logprob,entropy,value,controls

def load_policy(path, device="cpu"):
    checkpoint = torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported architecture")
    model = Policy().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint

def checkpoint(model, **metadata):
    return dict(architecture=ARCHITECTURE,observation_version=1,round_limit=60,model=model.state_dict(),**metadata)
