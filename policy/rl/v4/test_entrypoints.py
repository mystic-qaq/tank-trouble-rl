import json
from pathlib import Path
import pytest
from policy.rl.v4 import train
from policy.rl.v4.common import GateFailure


def test_no_dagger_or_ppo_entry_with_an_invalid_predecessor(tmp_path, monkeypatch):
    certificate = tmp_path / 'fake.json'
    certificate.write_text(json.dumps(dict(passed=True, receipt_sha256='not-a-receipt')))
    def forbidden(*args, **kwargs):
        raise AssertionError('Training setup ran before predecessor verification')
    monkeypatch.setattr(train, 'setup', forbidden)
    config = Path(train.__file__).with_name('config.json')
    args = (config, [], tmp_path / 'data.json', certificate, tmp_path / 'model.pt', tmp_path / 'out', 'win', 'cpu', 1)
    with pytest.raises(GateFailure):
        train.dagger(*args)
    with pytest.raises(GateFailure):
        train.reward(*args, pilot=True)
    with pytest.raises(GateFailure):
        train.reward(*args, pilot=False)


def test_pretrain_requires_the_same_data_rules_as_its_g0_receipt(tmp_path, monkeypatch):
    config = json.loads(Path(train.__file__).with_name('config.json').read_text())
    monkeypatch.setattr(train, 'verify_data', lambda _: dict(directories=[], rules={'different': True}))
    monkeypatch.setattr(train, 'setup', lambda *_: (config, tmp_path, 0, None))
    with pytest.raises(GateFailure, match='rules differ'):
        train.pretrain(tmp_path / 'config.json', [], tmp_path / 'data.json', tmp_path / 'out', 'cpu', 1)


def test_branch_gpu_groups_are_disjoint_and_never_exceed_four():
    from policy.rl.v4.pipeline import gpu_groups
    for count in range(1, 9):
        groups = gpu_groups(','.join(str(i) for i in range(count)))
        if count == 1:
            assert not groups
        else:
            assert len(groups) == 2
            assert not set(groups[0]) & set(groups[1])
            assert sum(map(len, groups)) <= 4
    with pytest.raises(GateFailure):
        gpu_groups('0,0,1,2')
