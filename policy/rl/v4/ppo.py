"""Transactional PPO: evaluate exact categorical KL before committing any update."""
import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from .common import GateFailure, BudgetExhausted


@torch.no_grad()
def distribution_logits(actor, x, style, batch=512):
    return torch.cat([actor.logits(x[i:i + batch], style[i:i + batch]).cpu() for i in range(0, len(x), batch)])


def exact_kl(old_logits, new_logits):
    old_lp, new_lp = old_logits.double().log_softmax(-1), new_logits.double().log_softmax(-1)
    # Impossible actions have negligible probability; their finite -1e9 mask cancels.
    values = (old_lp.exp() * (old_lp - new_lp)).sum(-1).clamp(min=0)
    return float(values.mean()), float(torch.quantile(values, .99))


def transactional_actor_update(actor, optimizer, x, style, actions, old_lp, advantage, rules, teacher=None, deadline=float('inf')):
    actor.eval()  # Same network computation in collection and optimization.
    with torch.no_grad():
        old_logits = distribution_logits(actor, x, style, rules['batch'])
        replay = old_logits.log_softmax(-1).gather(1, actions.cpu()[:, None]).squeeze(1)
        error = float((replay - old_lp.cpu()).abs().max())
    if not np.isfinite(error) or error > rules['replay_tolerance']:
        raise GateFailure('Old probability replay failed before any actor update: ' + str(error))
    saved_model = copy.deepcopy(actor.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    base_lr = optimizer.param_groups[0]['lr']
    try:
        for attempt in range(rules['backtracks']):
            actor.load_state_dict(saved_model)
            # Adam load_state_dict can retain references to moment tensors. Each
            # attempt needs a fresh copy or the supposed rollback snapshot mutates.
            optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            for group in optimizer.param_groups:
                group['lr'] = base_lr * (.5 ** attempt)
            steps = 0
            for epoch in range(rules['epochs']):
                stop = False
                for idx in torch.randperm(len(x), device=x.device).split(rules['batch']):
                    if time.monotonic() >= deadline:
                        raise BudgetExhausted('Budget exhausted inside actor transaction; rolled back')
                    logits = actor.logits(x[idx], style[idx])
                    distribution = torch.distributions.Categorical(logits=logits)
                    lp = distribution.log_prob(actions[idx])
                    delta = lp - old_lp[idx]
                    # Check the current policy BEFORE optimizer.step, then check the whole
                    # rollout AFTER all steps. No minibatch approximation can certify it.
                    with torch.no_grad():
                        before, _ = exact_kl(old_logits[idx.cpu()], logits.detach().cpu())
                    if before > rules['mean_kl']:
                        stop = True
                        break
                    ratio = delta.exp()
                    surrogate = torch.minimum(ratio * advantage[idx], ratio.clamp(1 - rules['clip'], 1 + rules['clip']) * advantage[idx])
                    loss = -surrogate.mean()
                    # No entropy bonus that forces firing/diverse motion on every physical frame.
                    if teacher is not None:
                        tx, ts, ta, tw = teacher.batch(min(128, rules['batch']), x.device)
                        loss = loss + rules['bc_aux'] * actor.bc_loss(tx, ts, ta, tw)
                    if not torch.isfinite(loss):
                        raise GateFailure('Nonfinite actor loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
                    if not torch.isfinite(norm):
                        raise GateFailure('Nonfinite actor gradient')
                    optimizer.step()
                    steps += 1
                if stop:
                    break
            with torch.no_grad():
                new_logits = distribution_logits(actor, x, style, rules['batch'])
                average, p99 = exact_kl(old_logits, new_logits)
            if time.monotonic() >= deadline:
                raise BudgetExhausted('Budget exhausted before actor commit; rolled back')
            if steps and np.isfinite([average, p99]).all() and average <= rules['mean_kl'] and p99 <= rules['p99_kl']:
                return dict(accepted=True, mean_kl=average, p99_kl=p99, replay_error=error, optimizer_steps=steps,
                            backtracks=attempt, lr=optimizer.param_groups[0]['lr'])
        raise GateFailure('PPO trust bound failed after bounded backtracking; actor and optimizer rolled back')
    except BaseException:
        actor.load_state_dict(saved_model)
        optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        raise
