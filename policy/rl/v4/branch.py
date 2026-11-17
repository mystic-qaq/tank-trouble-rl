"""One isolated objective; every handoff still goes through the original gates."""
import argparse
import json
import time
from pathlib import Path
from . import train
from .common import atomic_json, BudgetExhausted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--data-receipt', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--predecessor', type=Path, required=True)
    p.add_argument('--objective', choices=['win', 'frag'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + a.seconds
    state = dict(objective=a.objective, status='running', stage='G2_dagger')

    def stage(name):
        state['stage'] = name
        atomic_json(a.out / 'branch.json', state)

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise BudgetExhausted('Branch total budget exhausted; no next stage')
        return value

    try:
        stage('G2_dagger')
        initial, cert = train.dagger(a.config, a.data, a.data_receipt, a.predecessor, a.initial, a.out / 'dagger', a.objective, 'cuda:0', remaining())
        data = [Path(p) for p in json.loads((a.out / 'dagger/training-data.json').read_text())]
        stage('G3_pilot')
        pilot, pilot_cert = train.reward(a.config, data, a.data_receipt, cert, initial, a.out / 'pilot', a.objective, 'cuda:0', remaining(), pilot=True)
        stage('G4_reward_and_final')
        model, final_cert = train.reward(a.config, data, a.data_receipt, pilot_cert, pilot, a.out / 'reward', a.objective, 'cuda:0', remaining(), pilot=False)
        state.update(status='accepted', stage='complete', checkpoint=str(model), certificate=str(final_cert))
        atomic_json(a.out / 'branch.json', state)
    except BaseException as e:
        state.update(status='stopped_not_accepted', error=repr(e), next_stage_allowed=False)
        atomic_json(a.out / 'branch.json', state)
        raise


if __name__ == '__main__':
    main()
