"""V8 Stage 2a: fire-decision RL on a frozen Stage 1 base.

The Stage 1 trunk, drive/steer heads, base fire head and aux heads stay frozen. Only a
zero-initialised safety adapter (self state + raw lidar + clock -> 64 -> 1) is trained, so
the base's movement and its learned trigger behaviour are preserved and the new head can
only add a state-dependent veto/boost to the fire logit.

Reward per learner decision:

    shot hits an enemy            +1.0
    shot hits self                -1.0
    shot resolves without a hit    0.0
    hold while able to fire       -0.12  per decision frame
    muzzle blocked (fizzle)       -0.05

The hold term is the user's "not firing" disadvantage, calibrated between the two
degeneracies measured in this round: with no hold cost the policy collapses to 0.17 shots
per round (kills -> 0), while rewarding the empty shot teaches it to farm walls. At
h = 0.12 the average fired shot is still slightly negative (break-even is ~0.165 given
p_hit 0.025 / p_self 0.19), so suicides fall, but a safe empty shot is still a small
advantage over holding and the fire rate survives.

The one-step advantage reward - V(s) is used unchanged: centring only the fired subset
would hide that the average fired shot is negative and make waste shots look good.

Engine rollouts run in Node workers; the PyTorch model is batched across workers and
carries the per-episode map features and GRU memory. A small value head on the frozen
trunk feature z is fitted to the per-decision reward first, then used as the baseline.

Why this is the first Stage 2 step: the base self-destructs in 79.5% of rounds (0.795
suicides, 0.234 self-hits per shot), which costs far more than its 0.09 kills per round.
The Stage 1 aim_safe_recall was 0.004, i.e. the 512-d z does not make "will this shot come
back at me" linearly separable. The adapter sees the forward wall distance and normal
directly, which is the physical cause of the close-range self-hits.

Usage (server):
  python -m policy.rl.v8.rl --init .../s1a/last.pt --out .../v8-runs/rl-frag-c \
      --style gpt --device cuda:0 --workers 96 --episodes 512 --updates 40
"""
import argparse
import concurrent.futures as cf
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from ..v4.common import STYLES, atomic_json
from ..v4.gates import mean as macro
from ..v4.tasks import combat, training
from .model import TRUNK, fire_features, load_policy
from .transport import Worker

TEACHER_WIN = 0.30187074829931976
TEACHER_SUICIDES = 0.12304421768707482
ACCEPT_FRACTION = 0.7
SUICIDE_MARGIN = 0.03

# hold is the user's "not firing" disadvantage. It sits below the measured break-even
# (~0.165), so the average fire is still slightly negative and suicides fall, but the
# never-fire collapse (hold=0) is avoided. waste=0 gives a safe empty shot a small
# advantage over holding without making wall spam profitable.
REWARD = dict(hit=1.0, self=-1.0, waste=0.0, hold=0.12, fizzle=0.05, unresolved=0.05)


class ValueHead(nn.Module):
    """Baseline on the frozen trunk feature. Trained on the current batch, never exported."""

    def __init__(self, width=TRUNK, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class Rollout:
    """One learner episode: observation stream, decisions and per-shot reward ledger."""

    def __init__(self, task, style_index):
        self.task = task
        self.style_index = style_index
        self.map_feat = None
        self.memory = None
        self.obs = None
        self.z = []
        self.fire = []
        self.logp = []
        self.prob = []
        self.input = []
        self.reward = []
        self.can = []
        self.pending = {}
        self.hold_reward = 0.0
        self.shot_reward = 0.0
        self.info = None

    def record(self, z, fire, logp, prob, observation):
        self.z.append(z)
        self.fire.append(fire)
        self.logp.append(logp)
        self.prob.append(prob)
        self.input.append(observation)
        self.reward.append(0.0)
        self.can.append(0)

    def on_act(self, header, data, mode):
        self.obs = np.asarray(data, np.float32)
        if mode == 'train':
            t = len(self.z) - 1
            can = int(header.get('can_fire', 0))
            self.can[t] = can
            if can and self.fire[t] == 0:
                self.reward[t] -= REWARD['hold']
                self.hold_reward -= REWARD['hold']
            for event in header.get('events') or []:
                kind = event.get('kind')
                shot = event.get('shot_id')
                if kind == 'emission':
                    self.pending[shot] = t
                elif kind in ('hit', 'self_hit'):
                    frame = self.pending.pop(shot, t)
                    value = REWARD['hit'] if kind == 'hit' else REWARD['self']
                    self.reward[frame] += value
                    self.shot_reward += value
                elif kind == 'despawn':
                    if event.get('cause') in ('hit', 'self_hit'):
                        self.pending.pop(shot, None)
                    else:
                        frame = self.pending.pop(shot, t)
                        self.reward[frame] += REWARD['waste']
                        self.shot_reward += REWARD['waste']
                elif kind == 'fizzle':
                    self.reward[t] -= REWARD['fizzle']
                    self.shot_reward -= REWARD['fizzle']
        if header.get('done'):
            self.info = header.get('info') or {}

    def finish(self, mode):
        out = dict(task=self.task, info=self.info or {}, frames=len(self.z),
                   hold_reward=self.hold_reward, shot_reward=self.shot_reward)
        if mode == 'train':
            unresolved = len(self.pending)
            if unresolved:
                for frame in self.pending.values():
                    self.reward[frame] -= REWARD['unresolved']
                    self.shot_reward -= REWARD['unresolved']
            out.update(z=np.stack(self.z).astype(np.float32),
                       fire=np.asarray(self.fire, np.int64),
                       logp=np.asarray(self.logp, np.float32),
                       prob=np.asarray(self.prob, np.float32),
                       input=np.stack(self.input).astype(np.float32),
                       reward=np.asarray(self.reward, np.float32),
                       can=np.asarray(self.can, np.int8),
                       unresolved=unresolved)
        return out


class Pool:
    """Batched engine rollouts. One Node process per worker; one outstanding request each.

    The main loop is single-threaded (the GPU forward), while worker IPC runs in threads.
    A worker's next action is sent as soon as its observation is ready; up to 128 ready
    workers are batched into one model call.
    """

    def __init__(self, model, device, workers, style_index, mode='train'):
        self.model = model
        self.device = device
        self.count = workers
        self.style_index = style_index
        self.mode = mode
        self.workers = [Worker() for _ in range(workers)]
        self.executor = cf.ThreadPoolExecutor(workers)

    def run(self, tasks):
        self.todo = iter(tasks)
        self.pending = {}
        self.ready = deque()
        self.states = [None] * self.count
        self.closed = set()
        self.episodes = []
        self.last_flush = time.monotonic()
        try:
            for wid in range(self.count):
                self._start(wid)
            while self.pending or self.ready:
                if self.pending:
                    done, _ = cf.wait(self.pending, timeout=0.002,
                                      return_when=cf.FIRST_COMPLETED)
                    done |= {f for f in self.pending if f.done()}
                    for future in done:
                        wid, kind = self.pending.pop(future)
                        header, data = future.result()
                        if not header.get('ok'):
                            raise RuntimeError(header.get('error'))
                        state = self.states[wid]
                        if kind == 'open':
                            map_vec = torch.from_numpy(
                                np.asarray(data[:346], np.float32)).unsqueeze(0).to(self.device)
                            with torch.inference_mode():
                                state.map_feat = self.model.encode_map(map_vec)
                            state.memory = self.model.zero_memory(1, self.device)
                            state.obs = np.asarray(data[346:], np.float32).copy()
                            self.ready.append(wid)
                        else:
                            state.on_act(header, data, self.mode)
                            if header.get('done'):
                                self.episodes.append(state.finish(self.mode))
                                self.states[wid] = None
                                self._start(wid)
                            else:
                                self.ready.append(wid)
                if self.ready and (not self.pending or len(self.ready) >= 32
                                   or time.monotonic() - self.last_flush >= 0.002):
                    self._flush()
        finally:
            self.executor.shutdown(wait=True)
            for worker in self.workers:
                worker.close()
        return self.episodes

    def _start(self, wid):
        task = next(self.todo, None)
        if task is None:
            self.states[wid] = None
            if wid not in self.closed:
                self.closed.add(wid)
                self.workers[wid].close()
            return
        state = Rollout(task, self.style_index)
        self.states[wid] = state
        future = self.executor.submit(self.workers[wid].request, op='rl_open', slot=0,
                                      config=task, seat=task['seat'])
        self.pending[future] = (wid, 'open')

    def _flush(self):
        self.last_flush = time.monotonic()
        batch = [self.ready.popleft() for _ in range(min(128, len(self.ready)))]
        states = [self.states[wid] for wid in batch]
        obs = torch.from_numpy(np.stack([s.obs for s in states])).to(self.device)
        with torch.inference_mode():
            memory = (torch.cat([s.memory[0] for s in states], 0),
                      torch.cat([s.memory[1] for s in states], 0))
            map_feat = (torch.cat([s.map_feat[0] for s in states], 0),
                        torch.cat([s.map_feat[1] for s in states], 0))
            style = torch.full((len(batch),), self.style_index, dtype=torch.long,
                               device=self.device)
            memory = self.model.advance_memory(obs, memory)
            out = self.model.step(obs, style, map_feat, memory)
            slices = [(memory[0][k:k + 1].clone(), memory[1][k:k + 1].clone())
                      for k in range(len(batch))]
            logits = out['fire']
            mu = out['steer_mu'].squeeze(-1)
            prob = torch.sigmoid(logits)
            u = torch.rand(len(batch), 2, device=self.device)
            if self.mode == 'sampled':
                cdf = out['drive'].softmax(-1).cumsum(-1)
                drive = torch.searchsorted(cdf.contiguous(),
                                           (u[:, 0].clamp(0, 1 - 1e-6))[:, None].contiguous(),
                                           right=True).squeeze(1).clamp(0, 2)
                dist = Normal(mu, out['steer_logstd'].squeeze(-1).clamp(-4.0, 1.0).exp())
                steer = torch.tanh(dist.icdf(u[:, 1].clamp(1e-6, 1 - 1e-6)))
                fire = (torch.rand(len(batch), device=self.device) < prob).float()
                logp = None
            else:
                drive = out['drive'].argmax(-1)
                steer = mu
                if self.mode == 'train':
                    fire = (u[:, 0] < prob).float()
                    # log pi(a|s), not the BCE loss: the PPO ratio below must be
                    # exp(logpi_new - logpi_old). Storing -logpi inverts the gradient.
                    logp = -F.binary_cross_entropy_with_logits(logits, fire, reduction='none')
                else:
                    fire = (logits > 0).float()
                    logp = None
            z = out['z']
        drive_np = (drive.float() - 1.0).cpu().numpy()
        steer_np = steer.cpu().numpy()
        fire_np = fire.cpu().numpy()
        z_np = z.cpu().numpy() if self.mode == 'train' else None
        logp_np = logp.cpu().numpy() if logp is not None else None
        prob_np = prob.cpu().numpy() if self.mode == 'train' else None
        for k, wid in enumerate(batch):
            state = states[k]
            state.memory = slices[k]
            if self.mode == 'train':
                state.record(z_np[k].copy(), int(fire_np[k]), float(logp_np[k]),
                             float(prob_np[k]), np.array(state.obs, np.float32))
            action = dict(drive=int(drive_np[k]), steer=float(steer_np[k]),
                          shoot=bool(fire_np[k] > 0.5))
            future = self.executor.submit(self.workers[wid].request, op='rl_act', slot=0,
                                          action=action)
            self.pending[future] = (wid, 'act')


def eval_rows(episodes):
    return [{**ep['task'], **(ep['info'] or {})} for ep in episodes]


def summarise(rows):
    ev = lambda k: float(np.mean([(r.get('events') or {}).get(k, 0) for r in rows]))
    shots = max(ev('emissions'), 1e-9)
    return dict(episodes=len(rows),
                win_macro=macro(rows, 'win'),
                win_plain=float(np.mean([r['win'] for r in rows])),
                netKills_macro=macro(rows, 'netKills'),
                kills=float(np.mean([r['kills'] for r in rows])),
                suicides_macro=macro(rows, 'suicides'),
                timeout=float(np.mean([r['timeout'] for r in rows])),
                shots=shots, hit_per_shot=ev('hits') / shots,
                self_per_shot=ev('self_hits') / shots)


def run_eval(model, device, style_index, blocks, modes, workers):
    tasks = combat('development', blocks, 20261107)
    out = {}
    for mode in modes:
        pool = Pool(model, device, min(workers, max(1, len(tasks))), style_index, mode=mode)
        episodes = pool.run(tasks)
        rows = eval_rows(episodes)
        stats = summarise(rows)
        stats['pass_win'] = stats['win_macro'] >= ACCEPT_FRACTION * TEACHER_WIN
        stats['pass_suicide'] = stats['suicides_macro'] <= TEACHER_SUICIDES + SUICIDE_MARGIN
        out[mode] = stats
    return out


def update_stats(episodes):
    total = lambda key: sum((ep['info'] or {}).get('events', {}).get(key, 0) for ep in episodes)
    shots = total('emissions')
    hits = total('hits')
    self_hits = total('self_hits')
    fizzles = total('fizzles')
    requests = total('requests')
    fire = np.concatenate([ep['fire'] for ep in episodes])
    can = np.concatenate([ep['can'] for ep in episodes])
    prob = np.concatenate([ep['prob'] for ep in episodes])
    mask = can > 0
    return dict(episodes=len(episodes),
                frames=int(sum(ep['frames'] for ep in episodes)),
                shots=int(shots), hits=int(hits), self_hits=int(self_hits),
                fizzles=int(fizzles), requests=int(requests),
                kills=float(np.mean([(ep['info'] or {}).get('kills', 0) for ep in episodes])),
                suicides=float(np.mean([(ep['info'] or {}).get('suicides', 0) for ep in episodes])),
                netKills=float(np.mean([(ep['info'] or {}).get('netKills', 0) for ep in episodes])),
                wins=float(np.mean([(ep['info'] or {}).get('win', 0) for ep in episodes])),
                timeout=float(np.mean([(ep['info'] or {}).get('timeout', 0) for ep in episodes])),
                self_per_shot=self_hits / max(shots, 1),
                hit_per_shot=hits / max(shots, 1),
                fire_rate=float(fire[mask].mean()) if mask.any() else 0.0,
                mean_prob=float(prob[mask].mean()) if mask.any() else 0.0,
                hold_reward=float(sum(ep['hold_reward'] for ep in episodes)),
                shot_reward=float(sum(ep['shot_reward'] for ep in episodes)),
                unresolved=int(sum(ep['unresolved'] for ep in episodes)))


def fit_value(value_head, opt, z, reward, args):
    z = np.ascontiguousarray(z)
    for _ in range(args.value_steps):
        index = np.random.randint(0, len(z), min(args.value_batch, len(z)))
        zb = torch.from_numpy(z[index]).to(args.device)
        rb = torch.from_numpy(reward[index]).to(args.device)
        loss = F.mse_loss(value_head(zb), rb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
        opt.step()
    return float(loss.detach())


def ppo_update(model, value_head, episodes, opt_policy, opt_value, args):
    z = np.concatenate([ep['z'] for ep in episodes])
    fire = np.concatenate([ep['fire'] for ep in episodes])
    logp = np.concatenate([ep['logp'] for ep in episodes])
    reward = np.concatenate([ep['reward'] for ep in episodes])
    can = np.concatenate([ep['can'] for ep in episodes])
    raw = np.concatenate([ep['input'] for ep in episodes])
    keep = can > 0
    z, fire, logp, reward, raw = z[keep], fire[keep], logp[keep], reward[keep], raw[keep]

    # Critic first, on a uniform sample of all decisions (holds dominate, so the baseline
    # stays close to the hold cost and does not memorise the rare shot outcomes).
    sample = np.random.choice(len(z), min(args.value_rows, len(z)), replace=False)
    value_loss = fit_value(value_head, opt_value, z[sample], reward[sample], args)

    positive = np.flatnonzero(fire > 0.5)
    negative = np.flatnonzero(fire <= 0.5)
    if len(positive) == 0:
        return dict(policy_loss=0.0, value_loss=value_loss, kl=0.0, entropy=0.0,
                    positives=0, balanced=0)
    # Balanced policy batch: every fired decision plus an equal number of holds, so the
    # rare trigger pulls are not diluted by ~500x more holds.
    npos = len(positive)
    nneg = min(len(negative), max(npos, args.batch - npos))
    balanced = np.concatenate([positive,
                               np.random.choice(negative, nneg, replace=len(negative) < nneg)])
    np.random.shuffle(balanced)

    zb = torch.from_numpy(np.ascontiguousarray(z[balanced])).to(args.device)
    ob = torch.from_numpy(np.ascontiguousarray(raw[balanced])).to(args.device)
    with torch.no_grad():
        baseline = value_head(zb).cpu().numpy()
    # No within-fire normalisation. Centering the fired subset at zero hides the fact that
    # the average fired shot is negative (self-hits outnumber hits ~20:1) and makes the
    # neutral waste states look better than they are. The one-step advantage must stay on
    # the same scale as holding, so a self-hit lowers the trigger rate globally instead of
    # merely redistributing it among fire states.
    adv = torch.from_numpy((reward[balanced] - baseline).astype(np.float32)).to(args.device)
    fire_t = torch.from_numpy(fire[balanced].astype(np.float32)).to(args.device)
    old_t = torch.from_numpy(logp[balanced]).to(args.device)

    losses, kls, ents = [], [], []
    for _ in range(args.epochs):
        order = np.random.permutation(len(balanced))
        for chunk in np.array_split(order, max(1, len(order) // args.batch)):
            index = torch.from_numpy(chunk).to(args.device)
            zc = zb[index]
            logits = (model.fire_head(zc) + model.fire_adapter(fire_features(ob[index]))).squeeze(-1)
            new = -F.binary_cross_entropy_with_logits(logits, fire_t[index], reduction='none')
            ratio = (new - old_t[index]).exp()
            surrogate = torch.minimum(ratio * adv[index],
                                      ratio.clamp(1 - args.clip, 1 + args.clip) * adv[index])
            entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
            loss = -surrogate.mean() - args.entropy * entropy
            opt_policy.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt_policy.step()
            kl = float((((ratio - 1) - (new - old_t[index])).mean()).detach())
            losses.append(float(loss.detach()))
            kls.append(kl)
            ents.append(float(entropy.detach()))
            if kl > args.max_kl:
                break
        if kls and kls[-1] > args.max_kl:
            break
    return dict(policy_loss=float(np.mean(losses)) if losses else 0.0,
                value_loss=value_loss,
                kl=float(np.mean(kls)) if kls else 0.0,
                entropy=float(np.mean(ents)) if ents else 0.0,
                positives=int(len(positive)), balanced=int(len(balanced)))


def save_checkpoint(model, value_head, path, **metadata):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(dict(architecture='perception_factored_memory_v8',
                    temperature=model.temperature,
                    model=model.state_dict(),
                    observation='schema_v3',
                    round_limit=60,
                    value=value_head.state_dict(),
                    **metadata), temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--style', default='gpt')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--seconds', type=int, default=10800)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--vlr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--entropy', type=float, default=0.005)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--max-kl', type=float, default=0.02)
    p.add_argument('--value-rows', type=int, default=100000)
    p.add_argument('--value-batch', type=int, default=4096)
    p.add_argument('--value-steps', type=int, default=150)
    p.add_argument('--eval-every', type=int, default=8)
    p.add_argument('--eval-blocks', type=int, default=1)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--eval-modes', default='greedy,sampled')
    a = p.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model, meta = load_policy(a.init, a.device)
    model.eval()
    style_index = STYLES.index(a.style)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.eval_only:
        result = run_eval(model, a.device, style_index, a.eval_blocks,
                          a.eval_modes.split(','), a.workers)
        atomic_json(out / 'fast-eval.json', dict(checkpoint=a.init, style=a.style,
                                                 blocks=a.eval_blocks, runs=result))
        for mode, stats in result.items():
            print('%-8s win %.4f  netK %+.3f  suicides %.4f  shots %.2f  hit/shot %.3f  '
                  'self/shot %.3f' % (mode, stats['win_macro'], stats['netKills_macro'],
                                      stats['suicides_macro'], stats['shots'],
                                      stats['hit_per_shot'], stats['self_per_shot']), flush=True)
        print('WROTE', out / 'fast-eval.json')
        return

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.fire_adapter.parameters():
        param.requires_grad_(True)
    value_head = ValueHead().to(a.device)
    if 'value' in meta:
        try:
            value_head.load_state_dict(meta['value'])
        except Exception:
            pass
    opt_policy = torch.optim.Adam(model.fire_adapter.parameters(), lr=a.lr)
    opt_value = torch.optim.Adam(value_head.parameters(), lr=a.vlr)

    history = []
    best = None
    start = time.monotonic()
    for update in range(1, a.updates + 1):
        if time.monotonic() - start >= a.seconds:
            print('time budget reached before update', update, flush=True)
            break
        tasks = training(a.seed, update, a.episodes)
        for task in tasks:
            task['early_death'] = True
        pool = Pool(model, a.device, min(a.workers, len(tasks)), style_index, mode='train')
        rollout_start = time.monotonic()
        episodes = pool.run(tasks)
        rollout_seconds = time.monotonic() - rollout_start
        stats = update_stats(episodes)
        update_start = time.monotonic()
        losses = ppo_update(model, value_head, episodes, opt_policy, opt_value, a)
        update_seconds = time.monotonic() - update_start
        del episodes
        row = dict(update=update, rollout_seconds=round(rollout_seconds, 1),
                   update_seconds=round(update_seconds, 1), **stats, **losses,
                   reward=REWARD, style=a.style, base=a.init)
        save_checkpoint(model, value_head, out / 'last.pt', rl=row)
        if a.eval_blocks > 0 and a.eval_every > 0 and (
                update % a.eval_every == 0 or update == a.updates):
            evals = run_eval(model, a.device, style_index, a.eval_blocks,
                             ['greedy', 'sampled'], a.workers)
            row['eval'] = evals
            greedy = evals['greedy']
            if greedy['shots'] >= 1.0 and (best is None or
                                           greedy['suicides_macro'] < best['suicides_macro']):
                best = dict(update=update, **greedy)
                save_checkpoint(model, value_head, out / 'best.pt', rl=row)
            print('  eval   win %.4f  netK %+.3f  suicides %.4f  shots %.2f  hit/shot %.3f  '
                  'self/shot %.3f' % (greedy['win_macro'], greedy['netKills_macro'],
                                      greedy['suicides_macro'], greedy['shots'],
                                      greedy['hit_per_shot'], greedy['self_per_shot']), flush=True)
        history.append(row)
        atomic_json(out / 'history.json', history)
        print('[u%02d] eps %d  frames %d  shots %d  self/shot %.3f  hit/shot %.3f  '
              'suicides %.3f  kills %.3f  netK %.3f  fire %.3f  p %.3f  pg %.3f  v %.3f  '
              'kl %.4f  %ds+%ds'
              % (update, stats['episodes'], stats['frames'], stats['shots'],
                 stats['self_per_shot'], stats['hit_per_shot'], stats['suicides'],
                 stats['kills'], stats['netKills'], stats['fire_rate'], stats['mean_prob'],
                 losses['policy_loss'], losses['value_loss'], losses['kl'],
                 rollout_seconds, update_seconds), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
