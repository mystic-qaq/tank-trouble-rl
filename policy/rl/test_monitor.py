import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from live_tensorboard import Importer

def test_import_skips_stage_budget_and_is_idempotent(tmp_path):
    root=tmp_path/'project';base=root/'runs/initial60';run=base/'win'
    (run/'ppo').mkdir(parents=True)
    events=[{'event':'stage_start','stage':'bc','steps':1048576},
        {'stage':'bc','epoch':1,'loss':2.,'validation_loss':3.},
        {'event':'stage_start','stage':'ppo','steps':1048576},
        {'stage':'ppo','steps':4096,'recent_win_rate':.1,'recent_net_kills':-.2,'elapsed':10.}]
    (base/'win.log').write_text(''.join(json.dumps(e)+'\n' for e in events)+'{"stage":')
    validation={'win_rate':.2,'net_kills':.5,'timeout_rate':.1,'by_n':{}}
    (run/'validation-bc.json').write_text(json.dumps(validation))
    (run/'validation-teacher.json').write_text(json.dumps(validation))
    importer=Importer(root,tmp_path/'events')
    importer.update();importer.update();importer.close()
    reader=EventAccumulator(str(tmp_path/'events/win')).Reload()
    points=reader.Scalars('Training/mixed_recent_win_rate')
    assert len(points)==1 and points[0].step==4096
    assert len(reader.Scalars('BC/loss'))==1
    assert len(reader.Scalars('Evaluation/win_rate'))==1
    restarted=Importer(root,tmp_path/'events');assert restarted.update()==0;restarted.close()
