"""Stage entrypoints with mandatory predecessor checks and bounded work cycles."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from .common import TEACHERS, STYLES, atomic_json, GateFailure, BudgetExhausted
from .model import Policy, Critic, save_policy, load_policy
from .data import Frames, verify_data, save_dagger
from .gates import verify_certificate, certify
from .evaluate import Evaluator, offline
from .tasks import combat, skills, training
from .rollout import run, monte_carlo
from .ppo import transactional_actor_update


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExhausted('Budget exhausted: stage NOT accepted; no automatic next stage')


def setup(config_path, out, device, seconds):
    config = json.loads(Path(config_path).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    if config['gpu_limit'] > 2 or config['workers'] > 96:
        raise GateFailure('Resource limits exceed the authorized ceiling')
    deadline = time.monotonic() + seconds
    return config, out, deadline, Evaluator(out, config_path, device, deadline)


def ready(dev, audits, config, fraction):
    for objective, result in dev.items():
        minimum = config['gates']['min_teacher_win' if objective == 'win' else 'min_teacher_net']
        if result['teacher'] < minimum or min(result['greedy'], result['sampled']) < fraction * result['teacher']:
            return False
        if result['suicide_sampled'] > result['teacher_suicides'] + config['gates']['suicide_margin']:
            return False
        a = audits[objective]
        if a['fire_positives'] < config['offline']['min_fire_positives']:
            return False
        if any(a[k] < config['offline'][k] for k in ['motion_accuracy', 'fire_precision', 'fire_recall']):
            return False
    return True


def fit(actor, frames, steps, out, config, evaluator, objectives, fraction, writer, device, *, require_ready):
    # A 14 MB model on a 2048 batch does not benefit from DataParallel: parameter
    # broadcast / scatter / gather costs more than the compute, and four CUDA
    # contexts at <10% utilisation merely occupy cards other users could use.
    # G1 runs on one card; each independent G2+ branch gets its own card.
    opt = torch.optim.AdamW(actor.parameters(), lr=config['bc']['lr'], weight_decay=1e-4)
    best_key, stale, qualified = (-1, -float('inf')), 0, False
    data_ms = compute_ms = 0.
    for step in range(1, steps + 1):
        check_budget(evaluator.deadline)
        actor.train()
        t0 = time.monotonic()
        x, style, actions, importance = frames.batch(config['bc']['batch'], device)
        t1 = time.monotonic()
        loss = actor.bc_loss(x, style, actions, importance)
        if not torch.isfinite(loss):
            raise GateFailure('Nonfinite behavior cloning loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        if not torch.isfinite(norm):
            raise GateFailure('Nonfinite behavior cloning gradient')
        opt.step()
        t2 = time.monotonic()
        data_ms, compute_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        if step % 50 == 0:
            writer.add_scalar('bc/loss', float(loss.detach()), step)
            writer.add_scalar('io/data_ms', data_ms, step)
            writer.add_scalar('io/compute_ms', compute_ms, step)
            atomic_json(out / 'status.json', dict(status='training_not_accepted', step=step, loss=float(loss.detach()),
                                                  gpus=1, data_ms=round(data_ms, 2), compute_ms=round(compute_ms, 2),
                                                  steps_per_sec=round(1000 / max(1e-6, data_ms + compute_ms), 2)))
        if step % config['bc']['eval_every'] and step != steps:
            continue
        actor.eval()
        dev = {o: evaluator.dev(actor, o) for o in objectives}
        audits = {o: offline(actor, evaluator_data(frames), TEACHERS[o], config['offline'], device, evaluator.deadline,
                             cache=evaluator.offline_cache) for o in objectives}
        qualified = ready(dev, audits, config, fraction)
        quality = min(min(v['greedy'], v['sampled']) / max(.01, v['teacher']) for v in dev.values())
        key = (int(qualified), quality)
        for objective, values in dev.items():
            for name in ['greedy', 'sampled', 'suicide_greedy', 'suicide_sampled']:
                writer.add_scalar(f'dev/{objective}/{name}', values[name], step)
        atomic_json(out / f'dev-{step:06d}.json', dict(step=step, dev=dev, offline=audits, screening_ready=qualified,
                                                     warning='Development screening only; independent promotion still mandatory'))
        if key > best_key:
            best_key, stale = key, 0
            save_policy(actor, out / 'candidate.pt', stage='bc_candidate', step=step, dev=dev, offline=audits)
        else:
            stale += 1
        writer.flush()
        if qualified:
            break
        if stale >= config['bc']['patience']:
            break
    if require_ready and not best_key[0]:
        raise GateFailure('Pretraining development/offline criteria not met; DAgger is forbidden')
    if not (out / 'candidate.pt').exists():
        raise GateFailure('No validated candidate')
    return out / 'candidate.pt', bool(best_key[0]), best_key[1]


def evaluator_data(frames):
    return sorted({str(Path(r['path']).parent) for r in frames.records})


def frame_options(config):
    bc = config['bc']
    return dict(fire_oversample=bc['fire_oversample'], fire_weight=bc['fire_weight'], style_boost=bc['style_boost'])


def promote(stage, checkpoint, data, config_path, out, evaluator, data_receipt, objective=None, predecessor=None, updates=None):
    check_budget(evaluator.deadline)
    previous = None
    if predecessor:
        cert = json.loads(Path(predecessor).read_text())
        previous = cert['checkpoint']
    evidence = out / 'acceptance-evidence.json'
    evaluator.evidence(stage, checkpoint, data, evidence, objective, previous, updates, data_receipt)
    certificate = out / 'acceptance.json'
    certify(stage, checkpoint, evidence, config_path, certificate, data_receipt=data_receipt, predecessor=predecessor)
    atomic_json(out / 'status.json', dict(status='accepted', stage=stage, certificate=str(certificate), checkpoint=str(checkpoint)))
    return checkpoint, certificate


def pretrain(config_path, data, data_receipt, out, device, seconds):
    receipt = verify_data(data_receipt)
    if {str(Path(p).resolve()) for p in data} != set(receipt['directories']):
        raise GateFailure('Pretraining data differs from audited G0 data')
    c, out, deadline, evaluator = setup(config_path, out, device, seconds)
    if receipt.get('rules') != c['data']:
        raise GateFailure('G0 data coverage rules differ from this configuration')
    frames = Frames(data, c['seed'], fire_index=receipt['files'], **frame_options(c))
    actor = Policy(c['temperature']).to(device)
    writer = SummaryWriter(str(out / 'tb'))
    try:
        candidate, _, _ = fit(actor, frames, c['bc']['max_steps'], out, c, evaluator, ['win', 'frag'],
                               c['gates']['pretrain_teacher_fraction'], writer, device, require_ready=True)
        return promote('pretrain', candidate, data, config_path, out, evaluator, data_receipt)
    finally:
        writer.close()


def dagger(config_path, data, data_receipt, predecessor, initial, out, objective, device, seconds):
    prior = verify_certificate(predecessor, 'pretrain', objective, config_path, initial)
    accepted_data = json.loads(Path(prior['evidence']).read_text())['training_data']
    if {str(Path(p).resolve()) for p in data} != set(accepted_data) or str(Path(data_receipt).resolve()) != prior['data_receipt']:
        raise GateFailure('DAgger input data differs from accepted pretraining data')
    c, out, deadline, evaluator = setup(config_path, out, device, seconds)
    actor, _ = load_policy(initial, device)
    data = list(data)
    best, best_score, stale = None, -float('inf'), 0
    for cycle in range(c['dagger']['max_rounds']):
        check_budget(deadline)
        tasks = combat('dagger-training-' + str(cycle), c['dagger']['blocks_per_round'], c['seed']) + skills('dagger-training-' + str(cycle), 12, c['seed'])
        actor.eval()
        for sampled in [False, True]:
            _, records, _ = run(tasks, actor, style=TEACHERS[objective], workers=c['workers'], device=device, stochastic=sampled,
                                trajectory=True, label_teacher=TEACHERS[objective], objective=objective, deadline=deadline, early_death=True)
            directory = save_dagger(out / f'labels-{cycle}-' / ('sampled' if sampled else 'greedy'), tasks, records, TEACHERS[objective])
            data.append(directory)
            del records
        cycle_out = out / f'cycle-{cycle}'
        cycle_out.mkdir()
        frames = Frames(data, c['seed'] + cycle, fire_index=verify_data(data_receipt)['files'], **frame_options(c))
        writer = SummaryWriter(str(cycle_out / 'tb'))
        try:
            path, qualified, score = fit(actor, frames, c['dagger']['steps_per_round'], cycle_out, c, evaluator, [objective],
                                          c['gates']['dagger_teacher_fraction'], writer, device, require_ready=False)
        finally:
            writer.close()
        if best is None or score > best_score:
            best, best_score, stale = path, score, 0
        else:
            stale += 1
        # Continue collecting from the best measured policy, not an automatically selected last round.
        actor, _ = load_policy(best, device)
        if qualified:
            best = path
            break
        if stale >= c['dagger']['patience']:
            break
    if not qualified:
        raise GateFailure('DAgger development criteria not met; PPO is forbidden')
    atomic_json(out / 'training-data.json', [str(Path(p).resolve()) for p in data])
    return promote('dagger', best, data, config_path, out, evaluator, data_receipt, objective, predecessor)


def reward(config_path, data, data_receipt, predecessor, initial, out, objective, device, seconds, *, pilot):
    prior = verify_certificate(predecessor, 'dagger' if pilot else 'pilot', objective, config_path, initial)
    accepted_data = json.loads(Path(prior['evidence']).read_text())['training_data']
    if {str(Path(p).resolve()) for p in data} != set(accepted_data) or str(Path(data_receipt).resolve()) != prior['data_receipt']:
        raise GateFailure('Reward input data differs from accepted predecessor')
    c, out, deadline, evaluator = setup(config_path, out, device, seconds)
    actor, meta = load_policy(initial, device)
    critic = Critic().to(device)
    critic.encoder.load_state_dict(actor.encoder.state_dict())
    # A zero baseline is valid for complete MC policy gradients; no unchecked critic pretraining stage.
    torch.nn.init.zeros_(critic.head.weight)
    torch.nn.init.zeros_(critic.head.bias)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=c['ppo']['actor_lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=c['ppo']['critic_lr'])
    version = frames_count = 0
    if not pilot:
        for name in ['critic', 'actor_optimizer', 'critic_optimizer', 'cpu_rng', 'version']:
            if name not in meta:
                raise GateFailure('Accepted pilot lacks complete continuation state')
        critic.load_state_dict(meta['critic'])
        actor_opt.load_state_dict(meta['actor_optimizer'])
        critic_opt.load_state_dict(meta['critic_optimizer'])
        torch.set_rng_state(meta['cpu_rng'].cpu())
        if str(device).startswith('cuda'):
            torch.cuda.set_rng_state(meta['cuda_rng'].cpu(), device)
        version, frames_count = meta['version'], meta['frames']
    teacher = Frames(data, c['seed'], fire_index=verify_data(data_receipt)['files'], **frame_options(c))
    if not pilot and 'teacher_rng' in meta:
        teacher.rng.bit_generator.state = meta['teacher_rng']
    writer = SummaryWriter(str(out / 'tb'))
    updates, best_score, best, stale = [], -float('inf'), None, 0

    def save(path):
        save_policy(actor, path, stage='ppo_pilot' if pilot else 'ppo_reward', objective=objective, version=version, frames=frames_count,
                    critic=critic.state_dict(), actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(),
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
                    teacher_rng=teacher.rng.bit_generator.state, updates=updates)

    try:
        for iteration in range(c['ppo']['pilot_iterations'] if pilot else c['ppo']['max_iterations']):
            check_budget(deadline)
            version += 1
            actor.eval()
            rows, records, seconds_used = run(training(c['seed'], version, c['ppo']['episodes']), actor, style=TEACHERS[objective],
                                               workers=c['workers'], device=device, stochastic=True, trajectory=True,
                                               objective=objective, deadline=deadline, early_death=True)
            frames_count += sum(r['frames'] for r in rows)
            if sum(r['suicides'] for r in rows) / max(1, len(rows)) > c['ppo']['max_batch_suicide_rate']:
                atomic_json(out / 'rejected-rollout.json', dict(rows=rows, reason='catastrophic sampled suicide rate; actor not updated'))
                raise GateFailure('Sampled rollout suicide rate exceeded the predeclared emergency ceiling before update')
            active = monte_carlo(records)
            del records
            x = torch.tensor(np.stack([r['obs'] for r in active]), device=device)
            actions = torch.tensor([r['action'] for r in active], device=device, dtype=torch.long)
            old_lp = torch.tensor([r['logprob'] for r in active], device=device)
            returns = torch.tensor([r['return'] for r in active], device=device)
            style = torch.full((len(x),), STYLES.index(TEACHERS[objective]), device=device, dtype=torch.long)
            del active
            with torch.no_grad():
                values = torch.cat([critic(x[i:i + c['ppo']['batch']], style[i:i + c['ppo']['batch']]) for i in range(0, len(x), c['ppo']['batch'])])
                advantages = returns - values
                advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-6)
            check_budget(deadline)
            update = transactional_actor_update(actor, actor_opt, x, style, actions, old_lp, advantages, c['ppo'], teacher, deadline)
            updates.append(update)
            for idx in torch.randperm(len(x), device=device).split(c['ppo']['batch']):
                check_budget(deadline)
                value_loss = F.smooth_l1_loss(critic(x[idx], style[idx]), returns[idx])
                if not torch.isfinite(value_loss):
                    raise GateFailure('Nonfinite critic loss')
                critic_opt.zero_grad(set_to_none=True)
                value_loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise GateFailure('Nonfinite critic gradient')
                critic_opt.step()
            summary = dict(status='training_not_accepted', objective=objective, version=version, frames=frames_count,
                           mean_return=float(np.mean([r['return'] for r in rows])), suicides=sum(r['suicides'] for r in rows),
                           hits=sum(r['events']['hits'] for r in rows), emissions=sum(r['events']['emissions'] for r in rows),
                           collection_fps=sum(r['frames'] for r in rows) / max(.01, seconds_used), **update)
            atomic_json(out / 'status.json', summary)
            for key in ['mean_return', 'suicides', 'hits', 'emissions', 'mean_kl', 'p99_kl']:
                writer.add_scalar('ppo/' + key, summary[key], frames_count)
            writer.flush()
            save(out / 'last-unaccepted.pt')
            del x, actions, old_lp, returns, style, advantages, values
            if not pilot and ((iteration + 1) % c['ppo']['eval_every'] == 0 or iteration + 1 == c['ppo']['max_iterations']):
                result = evaluator.dev(actor, objective)
                score = min(result['greedy'], result['sampled'])
                atomic_json(out / f'dev-{version:06d}.json', result)
                if score > best_score:
                    best_score, stale, best = score, 0, out / 'candidate.pt'
                    save(best)
                else:
                    stale += 1
                if stale >= c['ppo']['patience']:
                    break
        if pilot:
            path = out / 'candidate.pt'
            save(path)
            return promote('pilot', path, data, config_path, out, evaluator, data_receipt, objective, predecessor, updates)
        if best is None:
            raise GateFailure('No complete development evaluation; final acceptance is forbidden')
        return promote('final', best, data, config_path, out, evaluator, data_receipt, objective, predecessor, updates)
    finally:
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['pretrain', 'dagger', 'pilot', 'reward'])
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--data-receipt', type=Path, required=True)
    p.add_argument('--initial', type=Path)
    p.add_argument('--predecessor', type=Path)
    p.add_argument('--objective', choices=['win', 'frag'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=int, default=21600)
    a = p.parse_args()
    try:
        if a.stage == 'pretrain':
            pretrain(a.config, a.data, a.data_receipt, a.out, a.device, a.seconds)
        else:
            if not all([a.initial, a.predecessor, a.objective]):
                p.error('This stage requires --initial, --predecessor and --objective')
            fn = dagger if a.stage == 'dagger' else reward
            kwargs = {} if a.stage == 'dagger' else {'pilot': a.stage == 'pilot'}
            fn(a.config, a.data, a.data_receipt, a.predecessor, a.initial, a.out, a.objective, a.device, a.seconds, **kwargs)
    except Exception as e:
        atomic_json(a.out / 'failure.json', dict(status='stopped_not_accepted', error=str(e), next_stage_allowed=False))
        raise SystemExit(str(e))


if __name__ == '__main__':
    main()
