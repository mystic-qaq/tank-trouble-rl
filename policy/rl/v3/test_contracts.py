import numpy as np
import torch
import pytest
from policy.rl.v3.model import Policy,Critic,HIDDEN
from policy.rl.v3.rollout import active_episodes

def observations(batch=2,time=5):
    x=torch.zeros(batch,time,1966);x[:,:,6]=1;x[:,:,16]=1;x[:,:,28]=3.3/60
    x[:,:,1582:1594]=torch.tensor([.05,.99,.3,.5,0,1,0,0,.3,0,.02,1])
    return x

def test_recurrent_rollout_and_sequence_replay_agree():
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False);torch.manual_seed(5)
    m=Policy().eval();x=observations();hidden=m.initial(2);actions=[];probabilities=[]
    for step in range(x.shape[1]):
        a,lp,entropy,control,hidden=m.act(x[:,step],hidden)
        assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()
        assert control[:,1].abs().max()<=1
        actions.append(a);probabilities.append(lp)
    replay=m.evaluate_actions(x,m.initial(2),torch.stack(actions,1))[0]
    assert torch.allclose(replay,torch.stack(probabilities,1),atol=2e-4)

def test_hidden_state_has_effect_and_resets_per_episode():
    m=Policy().eval();x=observations()[:,0]
    with torch.no_grad():
        _,_,_,a,h=m.act(x,m.initial(2),deterministic=True)
        _,_,_,_,h2=m.act(x,h,deterministic=True)
        assert not torch.allclose(h,h2)
        assert torch.equal(a,m.act(x,m.initial(2),deterministic=True)[3])

def test_training_and_empty_entity_gradients_are_finite():
    m=Policy();x=observations();x[:,:,1582:]=0
    actions=torch.zeros(2,5,3);actions[:,:,0]=1;actions[:,:,1]=.4
    loss,_=m(x,actions,torch.ones(2,5),torch.ones(2));loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    critic=Critic()
    assert not ({id(p) for p in m.parameters()}&{id(p) for p in critic.parameters()})

def test_returns_include_inactive_bullet_tail_and_no_episode_leak():
    records={0:[dict(reward=0.,active=True),dict(reward=-1.,active=True),dict(reward=2.,active=False)],1:[dict(reward=-1.,active=True)]}
    episodes=active_episodes(records)
    assert [r['return'] for r in episodes[0]]==[1.,1.]
    assert episodes[1][0]['return']==-1.

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA RNG relocation regression')
def test_restore_rng_after_checkpoint_mapped_to_cuda():
    from policy.rl.v3.train_rl import restore_rng_state
    device='cuda:0';cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(device).clone()
    expected_cpu=torch.rand(5);expected_gpu=torch.rand(5,device=device)
    relocated={'cpu_rng':cpu.to(device),'cuda_rng':gpu.to(device)}
    restore_rng_state(relocated,device)
    assert torch.equal(torch.rand(5),expected_cpu)
    assert torch.equal(torch.rand(5,device=device),expected_gpu)

def test_best_initialization_survives_later_dagger_regression(tmp_path):
    from policy.rl.v3.selection import select_initialization
    from policy.rl.v3.model import ARCHITECTURE
    reports=[('shared',.03,0),('frag/adapt-1',.096,50),('frag/adapt-2',-.114,100),('frag/adapt-3',-.05,90)]
    for directory,score,shots in reports:
        path=tmp_path/directory;path.mkdir(parents=True)
        torch.save(dict(architecture=ARCHITECTURE,step=5000,closed_loop=dict(rounds=70,win=.1,frag=score,hits=12,emissions=shots,spin_fraction=0.)),path/'best.pt')
    selected,candidates=select_initialization(tmp_path,'frag')
    assert selected['path']==str(tmp_path/'frag/adapt-1/best.pt')
    assert len(candidates)==4 and not candidates[0]['usable']

def test_policy_forces_consistent_fp32_before_rollout():
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    model=Policy()
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
