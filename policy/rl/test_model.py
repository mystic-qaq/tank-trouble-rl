import torch
from model import Policy
from train_ppo import advantages,restore_rng

def test_probability_and_bounded_actions():
    torch.manual_seed(42);model=Policy();obs=torch.zeros(16,926)
    action,logprob,_,value,controls=model.act(obs)
    assert torch.isfinite(logprob).all()
    assert torch.allclose(logprob,model.act(obs,action)[1],atol=1e-6)
    assert (controls[:,1].abs()<=1).all()
    assert set(controls[:,0].tolist())<={-1.,0.,1.}
    loss=logprob.mean()+value.mean();loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

def test_terminal_boundaries_and_rollout_bootstrap():
    rewards=torch.tensor([[1.],[2.]])
    zero=torch.zeros_like(rewards)
    advantage,returns=advantages(rewards,zero,torch.tensor([[0.],[1.]]),torch.tensor([100.]),lam=1)
    assert torch.equal(returns,torch.tensor([[3.],[2.]]))
    _,returns=advantages(rewards,zero,torch.ones_like(rewards),torch.tensor([100.]),lam=1)
    assert torch.equal(returns,rewards)
    _,returns=advantages(rewards,zero,zero,torch.tensor([10.]),lam=1)
    assert torch.equal(returns,torch.tensor([[13.],[12.]]))

def test_padding_is_ignored():
    torch.manual_seed(42);model=Policy();obs=torch.zeros(1,926)
    modified=obs.clone();modified[:,362:]=torch.randn_like(modified[:,362:])*10
    modified[:,362:446].reshape(1,7,12)[...,-1]=0
    modified[:,446:].reshape(1,40,12)[...,-1]=0
    for original,changed in zip(model(obs),model(modified)):
        assert torch.allclose(original,changed,atol=1e-6)

def test_cuda_rng_resume_after_device_mapping():
    import pytest
    if not torch.cuda.is_available():pytest.skip('CUDA only')
    torch.manual_seed(123)
    cpu_state=torch.get_rng_state()
    cuda_states=torch.cuda.get_rng_state_all()
    expected=torch.rand(3,device='cuda')
    restore_rng(dict(torch_rng=cpu_state.to('cuda'),cuda_rng=[s.to('cuda') for s in cuda_states]))
    assert torch.equal(expected,torch.rand(3,device='cuda'))
