"""Plan by default. --execute explicitly starts the fail-closed training pipeline."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .common import ROOT, atomic_json, digest, source_digest, GateFailure, BudgetExhausted


def preflight(out, deadline):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    commands = [
        ['node', 'policy/verify-frozen.mjs'],
        ['node', '--experimental-vm-modules', '--disable-warning=ExperimentalWarning', 'policy/rl/v4/test_engine.mjs'],
        [sys.executable, '-m', 'pytest', '-q', 'policy/rl/v4'],
    ]
    records = []
    for i, cmd in enumerate(commands):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExhausted('Contract checks incomplete')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        log = out / f'contracts-{i}.log'
        with open(log, 'w') as f:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=min(180, remaining))
        records.append(dict(command=cmd, returncode=result.returncode, log=str(log.resolve()), log_sha256=digest(log)))
        if result.returncode:
            atomic_json(out / 'contracts.json', dict(passed=False, source_sha256=source_digest(), checks=records))
            raise GateFailure('Contract checks failed; see ' + str(log))
    path = out / 'contracts.json'
    atomic_json(path, dict(passed=True, source_sha256=source_digest(), checks=records, cpu_only=True))
    return path


def plan(config):
    return dict(training_started=False, architecture='relational entity attention, motion-conditional fire head, shared kill pretraining then fork',
                gates=['P0 engine/model/data contracts', 'P1 diagnostic BC bootstrap (never a gate)',
                       'P2 shared netKills pretraining with difficulty curriculum',
                       'P3 win/frag fork with dense kill auxiliary and exact KL trust region',
                       'P4 independent dev acceptance: 70% teacher, unchanged suicide ceiling'],
                combat_rounds_per_gate_policy=70 * config['suites']['gate_blocks'],
                skill_cases_per_gate_policy=6 * config['suites']['skill_cases'],
                modes=['greedy deployment', 'same sampled distribution used by PPO'],
                bc_gpu_limit=1, parallel_objective_branches=2,
                reward_and_evaluation_gpus_per_branch=1, maximum_authorized_gpus=min(2, config['gpu_limit']),
                native_workers_per_branch=config['workers'],
                on_failure='stop; preserve artifacts; no next stage, no threshold reduction, no automatic restart')


def gpu_groups(visible, limit=4):
    cards = [v.strip() for v in visible.split(',') if v.strip()][:min(4, limit)]
    if len(cards) != len(set(cards)):
        raise GateFailure('Duplicate GPU assignments')
    if len(cards) < 2:
        return []
    pivot = (len(cards) + 1) // 2
    return [cards[:pivot], cards[pivot:]]


def parallel_branches(out, state, config_path, data, data_receipt, shared, shared_cert, remaining):
    groups = gpu_groups(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    if not groups:
        return False
    jobs = []
    try:
        for objective, cards in zip(['win', 'frag'], groups):
            log = open(out / (objective + '-branch.log'), 'w')
            cmd = [sys.executable, '-m', 'policy.rl.v4.branch', '--config', str(config_path), '--data', *map(str, data),
                   '--data-receipt', str(data_receipt), '--initial', str(shared), '--predecessor', str(shared_cert),
                   '--objective', objective, '--out', str(out / objective), '--seconds', str(remaining())]
            process = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(cards)},
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((objective, process, log, cards))
        state.update(stage='independent_branches_after_accepted_G1', branches={})
        while any(p.poll() is None for _, p, _, _ in jobs):
            remaining()
            for objective, p, _, cards in jobs:
                path = out / objective / 'branch.json'
                detail = json.loads(path.read_text()) if path.exists() else dict(status='starting')
                state['branches'][objective] = dict(**detail, gpu_ids=cards, pid=p.pid, returncode=p.poll())
            atomic_json(out / 'pipeline.json', state)
            time.sleep(5)
        for objective, process, log, cards in jobs:
            log.close()
            path = out / objective / 'branch.json'
            detail = json.loads(path.read_text()) if path.exists() else dict(status='failed_no_report')
            state['branches'][objective] = dict(**detail, gpu_ids=cards, returncode=process.returncode)
        accepted = {o: v for o, v in state['branches'].items() if v['status'] == 'accepted'}
        state.update(status='accepted_both_objectives' if len(accepted) == 2 else 'partial_accepted' if accepted else 'stopped_not_accepted',
                     accepted=accepted, automatic_deployment=False, stage='complete', automatic_restart=False)
        atomic_json(out / 'pipeline.json', state)
        return True
    finally:
        for _, p, log, _ in jobs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
            if not log.closed:
                log.close()


def execute(a):
    # Lazy imports let users inspect the plan without torch/CUDA installed locally.
    from . import train
    from .data import audit
    c = json.loads(a.config.read_text())
    out = a.out.resolve()
    if (ROOT / 'policy') == out or (ROOT / 'policy') in out.parents:
        raise GateFailure('Run output must be outside policy source directories')
    out.mkdir(parents=True, exist_ok=False)
    config_path = out / 'config.json'
    shutil.copyfile(a.config, config_path)
    started = time.time()
    deadline = time.monotonic() + a.seconds
    state = dict(status='running', started_unix=started, deadline_unix=started + a.seconds,
                 source_sha256=source_digest(), config_sha256=digest(config_path), plan=plan(c), stage='G0')
    atomic_json(out / 'pipeline.json', state)

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise BudgetExhausted('Total budget exhausted; next stage forbidden')
        return value

    def stage(name):
        state.update(stage=name)
        atomic_json(out / 'pipeline.json', state)

    try:
        contracts = preflight(out / 'g0', deadline)
        data = [p.resolve() for p in a.data] if a.data else []
        if not data:
            for name, count, skill in [('multi', 6000, False), ('skills', 1200, True)]:
                directory = out / 'data' / name
                cmd = [sys.executable, '-m', 'policy.rl.v3.collect', '--out', str(directory), '--worlds', str(count),
                       '--workers', str(min(c['workers'], 80)), '--seed', str(c['seed'] + (100000 if skill else 0)), '--seconds', str(int(remaining()))]
                if skill:
                    cmd.append('--skills')
                with open(out / f'collect-{name}.log', 'w') as log:
                    result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining())
                if result.returncode:
                    raise GateFailure('Raw data collection failed')
                data.append(directory)
        data_receipt = out / 'g0' / 'data-acceptance.json'
        audit(data, c['data'], data_receipt, deadline, contracts)
        if a.prepare_only:
            state.update(status='prepared_no_training', stage='G0_accepted', next_stage_started=False)
            atomic_json(out / 'pipeline.json', state)
            return
        stage('G1_pretrain')
        shared, shared_cert = train.pretrain(config_path, data, data_receipt, out / 'pretrain', a.device, remaining())
        if str(a.device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
            if parallel_branches(out, state, config_path, data, data_receipt, shared, shared_cert, remaining):
                return
        for objective in ['win', 'frag']:
            branch = out / objective
            stage('G2_dagger_' + objective)
            initial, certificate = train.dagger(config_path, data, data_receipt, shared_cert, shared, branch / 'dagger', objective, a.device, remaining())
            branch_data = [Path(p) for p in json.loads((branch / 'dagger' / 'training-data.json').read_text())]
            stage('G3_pilot_' + objective)
            pilot, pilot_cert = train.reward(config_path, branch_data, data_receipt, certificate, initial, branch / 'pilot', objective, a.device, remaining(), pilot=True)
            stage('G4_reward_and_final_' + objective)
            trained, final_cert = train.reward(config_path, branch_data, data_receipt, pilot_cert, pilot, branch / 'reward', objective, a.device, remaining(), pilot=False)
            state.setdefault('accepted', {})[objective] = dict(checkpoint=str(trained), certificate=str(final_cert))
        state.update(status='accepted_both_objectives', stage='complete', automatic_deployment=False)
        atomic_json(out / 'pipeline.json', state)
    except BaseException as e:
        state.update(status='budget_exhausted_not_accepted' if isinstance(e, (BudgetExhausted, subprocess.TimeoutExpired)) else 'stopped_not_accepted',
                     error=repr(e), next_stage_allowed=False, automatic_restart=False)
        atomic_json(out / 'pipeline.json', state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--out', type=Path)
    p.add_argument('--data', type=Path, nargs='+')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=int, default=28800)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--prepare-only', action='store_true', help='Run G0 only; never train')
    a = p.parse_args()
    if not (a.execute or a.prepare_only):
        print(json.dumps(plan(json.loads(a.config.read_text())), ensure_ascii=False, indent=2))
        return
    if not a.out:
        p.error('--out is required for execution')
    execute(a)


if __name__ == '__main__':
    main()
