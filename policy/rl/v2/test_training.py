import numpy as np
import torch
from policy.rl.v2.train_ppo import monte_carlo,fresh_tasks
from policy.rl.v2.collect import is_validation
from policy.rl.v2.evaluate import combat_tasks

def test_full_agent_return_includes_posthumous_bullet_reward():
    obs=np.zeros(1966,dtype=np.float32);action=np.zeros(4,dtype=np.float32)
    episode=[(obs,action,0.,0.,True,None),(obs,action,0.,-1.,True,None),(obs,action,0.,2.,False,None)]
    x,actions,lp,returns=monte_carlo({0:episode})
    assert len(x)==2
    assert torch.equal(returns,torch.tensor([1.,1.]))
    # A different episode must never leak its return into this one.
    assert torch.equal(monte_carlo({0:episode,1:[(obs,action,0.,-1.,True,None)]})[3],torch.tensor([1.,1.,-1.]))

def test_dev_scenarios_have_complete_seat_and_map_cycles():
    tasks=combat_tasks(100)
    for n in range(2,9):
        subset=[t for t in tasks if t['n']==n]
        assert len(subset)>=100 and len(subset)%(2*n)==0
        for block in {t['block'] for t in subset}:
            rows=[t for t in subset if t['block']==block]
            assert len({(t['mapSeed'],t['seat']) for t in rows})==2*n

def test_hashed_layout_split_covers_every_skill():
    for skill in range(6):
        values=[is_validation(630261106+i) for i in range(skill,2400,6)]
        assert 15<sum(values)<80 and not all(values)

def test_ppo_training_scenarios_are_fresh_versions():
    a=fresh_tasks(11,1);b=fresh_tasks(11,2)
    assert not ({t['seed'] for t in a}&{t['seed'] for t in b})
    warm=fresh_tasks(11,-1)
    assert all(t['seed']>=0 for t in warm)
    assert not ({t['seed'] for t in warm}&{t['seed'] for t in a})

def test_good_offline_metrics_cannot_open_ppo_without_combat_evidence(tmp_path):
    import json
    from policy.rl.v2.gates import assess
    candidate=tmp_path/'win-fw1';reference=tmp_path/'references';data=tmp_path/'data'
    for path in [candidate/'skills',candidate/'bc',reference/'skills',data]:path.mkdir(parents=True)
    skills={s:dict(rounds=50,success=1.,suicides=0.,fizzles=0.) for s in ['static','moving','bank','muzzle','dodge','volley']}
    (candidate/'skills/results.json').write_text(json.dumps({'aggregate':skills}));(reference/'skills/results.json').write_text(json.dumps({'aggregate':skills}))
    torch.save({'validation':dict(fire_precision=.9,fire_recall=.9,fire_positive=1000)},candidate/'bc/best.pt')
    (data/'manifest.json').write_text(json.dumps(dict(unique_layouts=1000,actual_emissions=20000)))
    (tmp_path/'G0.json').write_text(json.dumps({'passed':True}))
    result=assess(candidate,reference,data,'win')
    assert result['gates']['G1']['passed'] and result['gates']['G2']['passed']
    assert not result['gates']['G3']['passed'] and not result['ppo_allowed']
