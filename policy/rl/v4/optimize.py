"""V6 direct combat-objective optimisation: shared kill pretraining, then fork.

The V4 G1 frame-imitation gate and the V5 two-independent-runs layout are both
replaced. The win objective is too sparse to bootstrap from scratch, while the
frag objective supplies a dense kills-minus-suicides signal that produces the
same combat skill. Stages:

  P0  inherited G0 engine/model/data contracts and audited data receipt
  P1  behaviour-prior bootstrap (BC, diagnostic only, never a hard gate)
  P2  SHARED phase: one process optimises netKills with the curriculum, and
      writes best.pt as the shared combat checkpoint (no acceptance claimed)
  P3  FORK phase: two processes initialise from that checkpoint; win adds a
      dense kill-auxiliary shaping term on top of the sparse win reward, frag
      keeps optimising netKills
  P4  independent acceptance per fork: min(greedy, sampled) >= 70% of the
      teacher on a development split, with a suicide-rate ceiling

The 70% teacher standard and the suicide ceiling are inherited unchanged from
G1. Training shaping never changes the measured combat objective. The total
wall-clock budget is split explicitly so a short smoke can never consume the
time reserved for certifying a gate.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from .common import TEACHERS, STYLES, atomic_json, BudgetExhausted, GateFailure
from .data import Frames, verify_data
from .evaluate import Evaluator
from .model import Policy, load_policy, save_policy
from .rollout import run
from .tasks import training

METRIC = {'win': 'win', 'frag': 'netKills'}


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else 0.


def plan_deadlines(started, seconds, acceptance_seconds, skip_acceptance):
    """Split the wall-clock budget so training cannot consume the gate's reserve."""
    if seconds < 1 or acceptance_seconds < 1:
        raise GateFailure('Invalid wall-clock budget')
    if not skip_acceptance and acceptance_seconds >= seconds:
        raise GateFailure('Acceptance reserve must be smaller than the total budget')
    final = started + seconds
    train = started + (seconds if skip_acceptance else max(60, seconds - acceptance_seconds))
    return min(train, final), final


def bootstrap(config, data, receipt, out, device, writer):
    """Behaviour prior only. Offline imitation metrics are diagnostics here."""
    rules = config['optimize']
    frames = Frames(data, config['seed'], fire_index=receipt['files'],
                    fire_oversample=rules['bootstrap_fire_oversample'], fire_weight=1.,
                    style_boost=config['bc']['style_boost'])
    actor = Policy(config['temperature']).to(device)
    optimizer = torch.optim.AdamW(actor.parameters(), lr=rules['bootstrap_lr'], weight_decay=1e-4)
    steps = rules['bootstrap_steps']
    start = time.monotonic()
    for step in range(1, steps + 1):
        x, style, actions, weights = frames.batch(config['bc']['batch'], device)
        loss = actor.bc_loss(x, style, actions, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.)
        optimizer.step()
        if step % 500 == 0:
            writer.add_scalar('bootstrap/loss', float(loss.detach()), step)
    save_policy(actor, out / 'bootstrap.pt', stage='bootstrap', step=steps)
    atomic_json(out / 'bootstrap.json',
                dict(status='diagnostic_only_not_a_gate', step=steps,
                     seconds=time.monotonic() - start,
                     note='Frame imitation is an initialiser, not an acceptance criterion'))
    return actor, frames


def shaped_returns(records, objective, rules):
    """Monte-Carlo returns with training-only dense shaping.

    The win objective otherwise emits a single +1 at the end of a round. A small
    per-frame survival bonus, a terminal death penalty, and (in the fork phase)
    a kill-delta auxiliary give the same optimal policy a learnable signal.
    None of these terms enter the measured win / netKills metric.
    """
    gamma = rules['gamma']
    kill_aux = float(rules.get('kill_aux', 0.))
    rows = []
    for episode in records.values():
        rewards = []
        for row in episode:
            reward = float(row['reward'])
            if objective == 'win':
                reward += rules['survival_bonus'] * float(bool(row['active']))
                if kill_aux:
                    reward += kill_aux * (float(row.get('kills_delta', 0.))
                                         - float(row.get('suicides_delta', 0.)))
            rewards.append(reward)
        if objective == 'win' and not any(float(r['reward']) > 0 for r in episode):
            alive = [i for i, r in enumerate(episode) if r['active']]
            if alive:
                rewards[alive[-1]] -= rules['death_penalty']
        total, backward = 0., []
        for row, reward in zip(reversed(episode), reversed(rewards)):
            total = reward + gamma * total
            if row['active']:
                backward.append((row, total))
        rows.extend(reversed(backward))
    if not rows:
        raise GateFailure('No active policy frames in rollout')
    return rows


def certification_start(curriculum):
    """First iteration index whose training fights have full-strength rosters."""
    return int(curriculum.get('ramp_iterations', 0)) if curriculum and curriculum.get('enabled') else 0


def curriculum_tasks(config, iteration, rules):
    """Ramp tank counts and opponent strength from measured weak policies.

    The arena leaderboard (8-player, 200 rounds) puts hy/gpt/glm/deepseek far
    ahead of qwen/kimi/minimax/builtin. Training a bootstrap policy against the
    full roster emits zero wins, so the win objective has no positive signal at
    all. This only changes the *training* distribution; the acceptance split is
    the untouched, full-strength development set.
    """
    curriculum = rules.get('curriculum') or {}
    count = rules['episodes']
    if not curriculum.get('enabled'):
        return training(config['seed'], iteration, count)
    ramp = max(1, int(curriculum['ramp_iterations']))
    fraction = min(1., (iteration - 1) / ramp)
    start, end = int(curriculum['start_n']), int(curriculum['end_n'])
    max_n = int(round(start + (end - start) * fraction))
    weak = curriculum.get('weak_opponents') or []
    pool = training(config['seed'], iteration, count * 16)
    chosen = []
    for task in pool:
        if task['n'] > max_n:
            continue
        row = dict(task)
        keep_strong = (len(chosen) / count) < fraction
        if weak and not keep_strong:
            players = list(row['players'])
            for i in range(len(players)):
                if i != row['seat']:
                    players[i] = weak[(i + len(chosen)) % len(weak)]
            row['players'] = players
        chosen.append(row)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise GateFailure('Curriculum pool exhausted before filling the episode budget')
    return chosen


def teacher_targets(actor, rows, device):
    """Teacher action targets on the student's own visited states (never a gate)."""
    labels = [r.get('label') for r in rows]
    if any(label is None for label in labels):
        return None
    controls = torch.tensor(np.asarray(labels, dtype=np.float32), device=device)
    return actor.targets(controls)


def update(actor, optimizer, frames, batch, device, rules):
    rows = [r for r, _ in batch]
    x = torch.from_numpy(np.stack([r['obs'] for r in rows])).to(device)
    actions = torch.tensor([r['action'] for r in rows], device=device, dtype=torch.long)
    old_lp = torch.tensor([r['logprob'] for r in rows], device=device)
    returns = torch.tensor([v for _, v in batch], device=device, dtype=torch.float32)
    style = torch.full((len(x),), STYLES.index(rules['policy_style']), device=device, dtype=torch.long)
    advantages = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
    teacher = teacher_targets(actor, rows, device) if rules.get('teacher_aux', 0) > 0 else None
    clip, size = rules['clip'], rules['batch']
    sample = slice(None, None, max(1, len(x) // 2048))
    with torch.no_grad():
        before = actor.logits(x[sample], style[sample]).softmax(-1)
    actor.train()
    last = float('nan')
    for _ in range(rules['epochs']):
        for idx in torch.randperm(len(x), device=device).split(size):
            logits = actor.logits(x[idx], style[idx])
            distribution = Categorical(logits=logits, validate_args=False)
            logp = distribution.log_prob(actions[idx])
            ratio = (logp - old_lp[idx]).exp()
            surrogate = torch.minimum(ratio * advantages[idx],
                                      ratio.clamp(1 - clip, 1 + clip) * advantages[idx])
            loss = rules.get('ppo_weight', 1.) * -surrogate.mean() - rules['entropy'] * distribution.entropy().mean()
            if teacher is not None:
                # Teacher imitation on visited states is a dense prior for a
                # sparse combat reward, never an acceptance criterion.
                logp_teacher = logits.log_softmax(-1).gather(1, teacher[idx][:, None]).squeeze(1)
                valid = (logp_teacher > -1e8).float()
                if rules.get('teacher_weight_mode', 'uniform') == 'advantage':
                    weights = advantages[idx].clamp(-2., 2.).exp() * valid
                else:
                    weights = valid
                loss = loss + rules['teacher_aux'] * (-(logp_teacher * weights).sum() / weights.sum().clamp_min(1e-8))
            if frames is not None:
                bx, bs, ba, bw = frames.batch(min(size, 512), device)
                loss = loss + rules['bc_aux'] * actor.bc_loss(bx, bs, ba, bw)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            optimizer.step()
            last = float(loss.detach())
    actor.eval()
    with torch.no_grad():
        after = actor.logits(x[sample], style[sample]).log_softmax(-1)
        kl = float((before * (before.clamp_min(1e-9).log() - after)).sum(-1).mean())
    return dict(loss=last, mean_kl=kl, frames=len(x))


def acceptance(actor, config, config_path, out, device, deadline, objective, rules):
    """Independent development split; the 70% teacher standard is unchanged."""
    evaluator = Evaluator(out, config_path, device, deadline)
    dev = evaluator.dev(actor, objective)
    student = min(dev['greedy'], dev['sampled'])
    teacher = dev['teacher']
    suicides = max(dev['suicide_greedy'], dev['suicide_sampled'])
    passed = (teacher >= config['gates']['min_teacher_' + ('win' if objective == 'win' else 'net')]
              and student >= rules['accept_fraction'] * teacher
              and suicides <= dev['teacher_suicides'] + rules['suicide_margin'])
    report = dict(objective=objective, dev=dev, student=student, teacher=teacher,
                  required=rules['accept_fraction'] * teacher, suicides=suicides,
                  suicide_ceiling=dev['teacher_suicides'] + rules['suicide_margin'], accepted=bool(passed))
    atomic_json(out / 'acceptance.json', report)
    return report


def optimize(config, config_path, data, receipt, out, device, seconds, objective,
             acceptance_seconds=3600, skip_acceptance=False, init=None, phase='fork'):
    rules = {**config['optimize'], 'objective': objective,
             'policy_style': config.get('policy_style', TEACHERS[objective])}
    started = time.monotonic()
    train_deadline, final_deadline = plan_deadlines(started, seconds, acceptance_seconds, skip_acceptance)
    writer = SummaryWriter(str(out / 'tb'))
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    stop_reason = 'iterations_complete'
    try:
        if init is None:
            actor, frames = bootstrap(config, data, receipt, out, device, writer)
        else:
            # Controlled within-pipeline transfer from the shared kill-pretraining
            # checkpoint: same architecture and observation contract, with an
            # auditable lineage. V1-V5 weights are still rejected by load_policy.
            actor, meta = load_policy(init, device)
            actor.eval()
            frames = Frames(data, config['seed'], fire_index=receipt['files'],
                            fire_oversample=rules['bootstrap_fire_oversample'], fire_weight=1.,
                            style_boost=config['bc']['style_boost'])
            atomic_json(out / 'init.json',
                        dict(source=str(init), architecture=meta.get('architecture'),
                             from_iteration=meta.get('iteration'), from_score=meta.get('score'),
                             from_stage=meta.get('stage'), phase=phase,
                             note='shared kill-pretraining checkpoint, same V6 architecture'))
        optimizer = torch.optim.Adam(actor.parameters(), lr=rules['actor_lr'])
        best, stale, history = -1e9, 0, []
        ramp = certification_start(rules.get('curriculum'))
        # The initial checkpoint is always a valid fallback.
        save_policy(actor, out / 'best.pt', stage='initial', objective=objective, iteration=0, score=0., fitness=0.)
        for iteration in range(1, rules['iterations'] + 1):
            if time.monotonic() >= train_deadline:
                stop_reason = 'training_budget_ended'
                break
            actor.eval()
            tasks = curriculum_tasks(config, iteration, rules)
            try:
                rows, records, elapsed = run(
                    tasks, actor, style=rules['policy_style'], workers=config['workers'],
                    device=device, stochastic=True, trajectory=True, objective=objective,
                    deadline=train_deadline, early_death=True,
                    label_teacher=TEACHERS[objective] if rules.get('teacher_aux', 0) > 0 else None)
            except BudgetExhausted:
                # A truncated rollout must never be used for an update or a gate.
                stop_reason = 'training_budget_ended_during_rollout'
                break
            score = mean(rows, METRIC[objective])
            kills, suicides = mean(rows, 'kills'), mean(rows, 'suicides')
            batch = shaped_returns(records, objective, rules)
            t0 = time.monotonic()
            snapshot = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            info = update(actor, optimizer, frames, batch, device, rules)
            update_seconds = time.monotonic() - t0
            # An exact trust region: discard any update whose measured KL leaves
            # the bound, and drop the optimiser momentum that produced it.
            reverted = info['mean_kl'] > rules.get('max_kl', float('inf'))
            if reverted:
                actor.load_state_dict(snapshot)
                optimizer = torch.optim.Adam(actor.parameters(), lr=rules['actor_lr'])
            del snapshot
            fitness = score - rules.get('tiebreak_suicide', 0.) * suicides
            status = dict(status='optimising_not_accepted', objective=objective, phase=phase,
                          iteration=iteration, score=score, fitness=fitness, kills=kills, suicides=suicides,
                          max_tanks=max(t['n'] for t in tasks), reverted=reverted,
                          collection_seconds=elapsed, update_seconds=update_seconds, **info)
            # Scores from the easy curriculum phase are not comparable with
            # full-strength scores. Only full-strength iterations may certify a
            # checkpoint or consume the early-stopping patience.
            certified = iteration > ramp
            status['certified'] = certified
            history.append(dict(iteration=iteration, score=score, fitness=fitness,
                                kills=kills, suicides=suicides, mean_kl=info['mean_kl'],
                                certified=certified))
            atomic_json(out / 'status.json', status)
            for key in ('score', 'fitness', 'kills', 'suicides', 'mean_kl', 'loss', 'frames'):
                writer.add_scalar('optimize/' + key, status[key], iteration)
            writer.flush()
            improved = certified and not reverted and fitness > best + 1e-9
            if improved:
                best, stale = fitness, 0
                save_policy(actor, out / 'best.pt', stage=phase, objective=objective,
                            iteration=iteration, score=score, fitness=fitness)
            elif certified:
                stale += 1
            print(f'iteration={iteration} {METRIC[objective]}={score:.4f} fitness={fitness:.4f} '
                  f'kills={kills:.3f} suicides={suicides:.3f} kl={info["mean_kl"]:.5f} '
                  f'n<={status["max_tanks"]} {"full" if certified else "ramp"} '
                  f'{"REVERTED" if reverted else ""} '
                  f'frames={info["frames"]} collect={elapsed:.0f}s update={update_seconds:.0f}s', flush=True)
            del records, batch, rows
            if certified and stale >= rules['patience'] and iteration >= rules.get('min_iterations', 0):
                stop_reason = 'early_stop_no_improvement'
                break
        best_actor, meta = load_policy(out / 'best.pt', device)
        best_actor.eval()
        common = dict(objective=objective, phase=phase, stop_reason=stop_reason,
                      iterations=len(history), best_fitness=best,
                      best_checkpoint_iteration=meta.get('iteration'), history=history,
                      checkpoint=str(out / 'best.pt'),
                      initialised_from=None if init is None else str(init),
                      automatic_restart=False)
        if skip_acceptance:
            result = dict(status='training_complete_acceptance_skipped', acceptance=None, **common)
        else:
            report = acceptance(best_actor, config, config_path, out, device, final_deadline, objective, rules)
            result = dict(status='accepted' if report['accepted'] else 'stopped_not_accepted',
                          acceptance=report, **common)
        atomic_json(out / 'result.json', result)
        print(('ACCEPTED' if result['status'] == 'accepted' else result['status']),
              json.dumps(result.get('acceptance') or {}, sort_keys=True), flush=True)
        return result
    except BaseException as e:
        atomic_json(out / 'failure.json',
                    dict(status='stopped_not_accepted', error=repr(e), next_stage_allowed=False,
                         automatic_restart=False))
        raise
    finally:
        writer.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    p.add_argument('--data', type=Path, nargs='+', required=True)
    p.add_argument('--data-receipt', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--objective', choices=['win', 'frag'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seconds', type=int, default=28800)
    p.add_argument('--acceptance-seconds', type=int, default=3600,
                   help='wall-clock tail reserved for the independent dev acceptance')
    p.add_argument('--skip-acceptance', action='store_true',
                   help='smoke and shared phase only: never claim a passed gate')
    p.add_argument('--init', type=Path, default=None,
                   help='V6 shared kill-pretraining checkpoint to fork from')
    p.add_argument('--phase', choices=['shared', 'fork'], default='fork')
    a = p.parse_args()
    receipt = verify_data(a.data_receipt)
    if {str(p.resolve()) for p in a.data} != set(receipt['directories']):
        raise GateFailure('Optimisation data differs from audited G0 data')
    a.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(a.config.read_text())
    if config['gpu_limit'] > 2:
        raise GateFailure('Resource ceiling is two GPUs')
    optimize(config, a.config, a.data, receipt, a.out, a.device, a.seconds, a.objective,
             acceptance_seconds=a.acceptance_seconds, skip_acceptance=a.skip_acceptance,
             init=a.init, phase=a.phase)


if __name__ == '__main__':
    main()
