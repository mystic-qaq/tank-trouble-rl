import torch
from policy.rl.v2.model import Policy,Critic
from policy.rl.v2.model import checkpoint,load_policy

def state(batch=8):
    x=torch.zeros(batch,1966);x[:,6]=1;x[:,16]=1;x[:,28]=3.3/60
    x[:,1582:1594]=torch.tensor([0,1,.3,.5,0,1,0,0,.3,0,.02,1])
    return x

def test_probability_replay_and_bounds():
    torch.manual_seed(3);m=Policy();x=state()
    actions,lp,entropy,controls=m.act(x)
    assert torch.allclose(lp,m.act(x,actions)[1],atol=1e-5)
    assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()
    assert controls[:,1].abs().max()<=1
    (lp.mean()+entropy.mean()*.001).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

def test_empty_entities_and_exact_fire_mask():
    m=Policy();x=torch.zeros(8,1966)
    a,lp,_,control=m.act(x)
    assert (control[:,2]==0).all() and torch.isfinite(lp).all()
    assert torch.isfinite(Critic()(x)).all()

def test_fire_depends_on_recorded_motion_and_no_critic_shared_parameters():
    m=Policy();x=state();h,*_=m.motion(x)
    left=m.fire_logits(x,h,torch.tensor([[1.,-1.]]*8))
    right=m.fire_logits(x,h,torch.tensor([[-1.,1.]]*8))
    assert not torch.allclose(left,right)
    critic=Critic()
    assert not ({id(p) for p in m.parameters()}&{id(p) for p in critic.parameters()})

def test_padding_features_do_not_change_encoder():
    m=Policy();x=state();y=x.clone()
    for start,count,width in [(454,7,24),(622,40,24)]:
        block=y[:,start:start+count*width].reshape(8,count,width)
        block[:,:,:-1]=torch.randn_like(block[:,:,:-1])*10
        block[:,:,-1]=0
    assert torch.allclose(m.encoder(x),m.encoder(y),atol=1e-5)

def test_serialized_actor_keeps_exact_action_decoding(tmp_path):
    m=Policy().eval();x=state();path=tmp_path/'actor.pt'
    torch.save(checkpoint(m,export_bias_adjustment=0),path)
    restored,metadata=load_policy(path);restored.eval()
    with torch.inference_mode():
        original=m.act(x,deterministic=True);decoded=restored.act(x,deterministic=True)
    assert metadata['export_bias_adjustment']==0
    assert torch.equal(original[3],decoded[3])
    assert torch.equal(original[1],decoded[1])

def test_binary_environment_and_shadow_teacher_use_whole_joint_action():
    from policy.rl.v2.transport import Worker
    with Worker() as w:
        header,x=w.request(op='reset',slot=0,options={},config=dict(seed=912,n=2,seat=0,skill='static'))
        assert len(x)==1966 and x[16]==1
        result,after=w.request(op='step',slot=0,action=dict(drive=1,steer=-1,shoot=False),label_teacher='reference')
        assert len(result['label'])==3 and result['label'][0]==0
        assert result['action']['drive']==1 # labels did not replace the learner's executed movement
        assert after.shape==(1966,) and __import__('numpy').isfinite(after).all()
