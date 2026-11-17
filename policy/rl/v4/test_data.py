import json
from pathlib import Path
import numpy as np
import pytest
from policy.rl.v4.common import atomic_json, source_digest, GateFailure
from policy.rl.v4.data import Frames, audit, validation, verify_data


def fixture_data(root, length=100, fires=5):
    root.mkdir()
    matrix = np.zeros((length, 1969), dtype=np.float32)
    matrix[:, 6] = matrix[:, 16] = 1
    matrix[:fires, -1] = 1
    np.save(root / 'data.npy', matrix)
    seed = next(i for i in range(100) if not validation(i))
    r = dict(file='data.npy', length=length, teacher='hy', source='multi_teacher', quality=1.,
             n=2, seed=seed, validation=False, physics_hz=60)
    (root / 'trajectories.jsonl').write_text(json.dumps(r) + '\n')
    atomic_json(root / 'manifest.json', dict(status='complete', continuous=True))
    (root / 'worlds.jsonl').write_text(json.dumps(dict(replay_identical=True)) + '\n')
    contract = root.parent / 'contracts.json'
    atomic_json(contract, dict(passed=True, source_sha256=source_digest()))
    return contract


def test_fire_oversampling_keeps_a_learnable_class_balance(tmp_path):
    fixture_data(tmp_path / 'data')
    frames = Frames([tmp_path / 'data'], seed=19)
    _, _, actions, weights = frames.batch(4096, 'cpu')
    fire = (actions[:, 2] > .5).float()
    share = float(fire.mean())
    # Balanced batches, not an inverse-proposal correction back to 0.6% fire rate.
    assert .05 < share < .30
    assert abs(float(weights.mean()) - 1.) < .02
    assert float(weights.min()) > 0.
    contribution = float((fire * weights).sum() / weights.sum())
    assert contribution >= .08, 'fire class must retain an effective gradient'
    assert float(weights.max()) / float(weights.min()) < 10., 'no 199-1000x suppression'


def test_long_sparse_trajectory_does_not_suppress_fire_gradient(tmp_path):
    # Real data has a 0.6% fire rate; the old inverse-proposal weight made such
    # trajectories suppress the fire class by ~1000x.
    fixture_data(tmp_path / 'data', length=3000, fires=1)
    frames = Frames([tmp_path / 'data'], seed=7)
    _, _, actions, weights = frames.batch(2048, 'cpu')
    fire = (actions[:, 2] > .5).float()
    assert .05 < float(fire.mean()) < .30
    contribution = float((fire * weights).sum() / weights.sum())
    assert contribution >= .08
    assert float(weights.max()) / float(weights.min()) < 10.


def test_unbiased_validation_is_not_the_training_distribution(tmp_path):
    # The training batch is deliberately class-balanced. Unbiased estimates come
    # from offline()'s uniform validation sampling, never from a scalar weight.
    fixture_data(tmp_path / 'data', length=3000, fires=1)
    frames = Frames([tmp_path / 'data'], seed=7)
    _, _, actions, _ = frames.batch(2048, 'cpu')
    assert float((actions[:, 2] > .5).float().mean()) > 10 * (1 / 3000)


def test_data_audit_detects_mutation_and_impossible_fire_labels(tmp_path):
    root = tmp_path / 'data'
    contract = fixture_data(root)
    rules = dict(min_train_trajectories_per_teacher=0, min_validation_trajectories_per_teacher=0,
                 min_train_fires_per_teacher=0, min_skill_trajectories=0)
    receipt = tmp_path / 'receipt.json'
    audit([root], rules, receipt, contracts=contract)
    verified = verify_data(receipt)
    assert verified['files'][str((root / 'data.npy').resolve())]['fire_rows'] == [0, 1, 2, 3, 4]
    frames = Frames([root], fire_index=verified['files'])
    assert frames.array(0)[1].tolist() == [0, 1, 2, 3, 4]
    matrix = np.load(root / 'data.npy')
    matrix[0, 16] = 0
    np.save(root / 'data.npy', matrix)
    with pytest.raises(GateFailure, match='data changed'):
        verify_data(receipt)
    with pytest.raises(GateFailure, match='unavailable fire'):
        audit([root], rules, tmp_path / 'receipt2.json', contracts=contract)
