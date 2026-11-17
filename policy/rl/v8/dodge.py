"""V8 Stage 2b-dodge: learn to survive incoming fire by moving, while firing.

The Stage 1 trunk, base drive/steer/fire heads and aux heads stay frozen. Only a
zero-initialised MoveAdapter on the raw bullets + ego lidar + clock + seat is trained, so
the base's behaviour is reproduced exactly at initialisation.

Curriculum: 1v1 (n=2, one fixed-strategy opponent) first. The objective is survival, not
kills. Reward per learner decision:

    alive this frame                 +0.01
    incoming bullet on a will-hit
      course that then misses        +0.5 per bullet
    an incoming hit (the death)      -2.0, plus -5.0 for dying
    a self-hit                       -1.0
    surviving to the round end       +2.0 (+1.0 for the win)
    entering a new map cell          +0.02
    moving slower than 40 px/s       -0.005 per frame
    staying in one cell > 1.5 s      -0.005 per frame

The coverage / displacement / camping terms are the anti-hide shaping the user asked for:
without them the optimal survival policy is to sit in a corner and micro-move. The dev
eval reports survival time, hits taken, self-hits, coverage cells and mean speed
alongside shots, so a low-survival-but-fast-circling policy is visible.

The per-frame threat (nearest incoming distance, will-hit count, own will-hit count) comes
from the JS solver label and is reward-only; it is not an input. The policy sees raw bullet
positions/velocities, the ego lidar and the seat. The seat is needed to distinguish own
bullets from incoming ones; Schema V3 carries the owner id but not the learner index.

Usage (server):
  python -m policy.rl.v8.dodge --init .../s1a/last.pt --out .../v8-runs/dodge-a \
      --device cuda:0 --workers 96 --episodes 512 --updates 40
"""
import argparse
import concurrent.futures as cf
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

from ..v4.common import IDS, STYLES, atomic_json
from ..v4.tasks import number
from .model import Policy, load_policy
from .rl import ValueHead, fit_value, save_checkpoint
from .transport import Worker

EPS = 1e-6
DT = 1 / 60

REWARD = dict(survive=0.01, death=5.0, survive_bonus=2.0, win=1.0, incoming=2.0,
              self_hit=1.0, dodge=0.5, new_cell=0.02, low_move=0.005, camp=0.005)
LOW_MOVE_PX = 40.0
CAMP_FRAMES = 90


def dodge_tasks(seed, version, count):
    """1v1 tasks; the learner seat plays one fixed strategy. objective=win gives exact
    early death (no waiting for the learner's bullets after it dies)."""
    rng = np.random.default_rng(number(seed, 'dodge', version))
    rows = []
    for i in range(count):
        seat = int(rng.integers(2))
        opponent = IDS[int(rng.integers(len(IDS)))]
        map_seed = [20260101, 88888888][i % 2] if i % 3 else number(seed, 'dodge-map', version, i)
        rows.append(dict(seed=number(seed, 'dodge-layout', version, i),
                         spawnSeed=number(seed, 'dodge-spawn', version, i),
                         mapSeed=map_seed, n=2, seat=seat, players=[opponent, opponent],
                         block='dodge:%d' % version, case=i, objective='win',
                         early_death=True, opponent=opponent))
    return rows


def eval_tasks(count=70):
    return dodge_tasks(975310, 0, count)


class Rollout:
    def __init__(self, task, style_index):
        self.task = task
        self.seat = int(task['seat'])
        self.style = style_index
        self.map_feat = None
        self.memory = None
        self.obs = None
        self.z = []
        self.obs_in = []
        self.drive = []
        self.steer = []
        self.logp = []
        self.value = []
        self.reward = []
        self.flags = []
        self.cells = set()
        self.prev_cell = None
        self.camp = 0
        self.last_xy = None
        self.speeds = []
        self.path = 0.0
        self.prev_will = 0
        self.incoming = 0
        self.self_hits = 0
        self.dodges = 0
        self.deaths = 0
        self.wins = 0
        self.timeout = 0
        self.info = None

    def track(self, observation):
        ox, oy = float(observation[0]), float(observation[1])
        cell = (min(15, int(ox * 16)), min(9, int(oy * 10)))
        new = cell not in self.cells
        self.cells.add(cell)
        if cell == self.prev_cell:
            self.camp += 1
        else:
            self.camp = 0
        camped = self.camp > CAMP_FRAMES
        low = False
        if self.last_xy is not None:
            d = float(np.hypot((ox - self.last_xy[0]) * 1280.0,
                               (oy - self.last_xy[1]) * 800.0))
            self.path += d
            self.speeds.append(d * 60.0)
            low = d < LOW_MOVE_PX * DT
        self.last_xy = (ox, oy)
        self.prev_cell = cell
        return new, low, camped

    def record(self, z, observation, drive, steer, logp, value):
        self.z.append(z)
        self.obs_in.append(observation)
        self.drive.append(drive)
        self.steer.append(steer)
        self.logp.append(logp)
        self.value.append(value)
        self.reward.append(0.0)
        self.flags.append(self.track(observation))

    def on_act(self, header, data, mode):
        self.obs = np.asarray(data, np.float32)
        t = len(self.z) - 1
        incoming_now = 0
        self_now = 0
        for event in header.get('events') or []:
            if (event.get('kind') == 'hit' and event.get('victim') == self.seat
                    and event.get('owner') != self.seat):
                incoming_now += 1
            elif event.get('kind') == 'self_hit':
                self_now += 1
        self.incoming += incoming_now
        self.self_hits += self_now
        if header.get('done'):
            self.info = header.get('info') or {}
            if int(header.get('alive', 0)):
                if int(self.info.get('win', 0)):
                    self.wins += 1
                if int(self.info.get('timeout', 0)):
                    self.timeout += 1
            else:
                self.deaths += 1
        if mode != 'train':
            return
        active = int(header.get('active', 0))
        threat = header.get('threat') or [1.0, 0, 1.2, 0]
        will = int(threat[1])
        reward = 0.0
        new_cell, low, camped = self.flags[t]
        if active:
            reward += REWARD['survive']
            if new_cell:
                reward += REWARD['new_cell']
            if low:
                reward -= REWARD['low_move']
            if camped:
                reward -= REWARD['camp']
            dodged = max(0, self.prev_will - will - incoming_now)
            if dodged:
                reward += REWARD['dodge'] * dodged
                self.dodges += dodged
        self.prev_will = will
        if incoming_now:
            reward -= REWARD['incoming'] * incoming_now
        if self_now:
            reward -= REWARD['self_hit'] * self_now
        if header.get('done'):
            if int(header.get('alive', 0)):
                reward += REWARD['survive_bonus']
                if int(self.info.get('win', 0)):
                    reward += REWARD['win']
            else:
                reward -= REWARD['death']
        self.reward[t] += reward

    def finish(self, mode):
        info = self.info or {}
        out = dict(task=self.task, info=info, seat=self.seat,
                   frames=int(info.get('frames', len(self.z))),
                   coverage=len(self.cells),
                   displacement=float(np.mean(self.speeds)) if self.speeds else 0.0,
                   path=float(self.path), incoming=self.incoming, self_hits=self.self_hits,
                   dodges=self.dodges, deaths=self.deaths, wins=self.wins,
                   timeout=self.timeout)
        if mode == 'train':
            out.update(z=np.stack(self.z).astype(np.float32),
                       obs_in=np.stack(self.obs_in).astype(np.float32),
                       drive=np.asarray(self.drive, np.int64),
                       steer=np.asarray(self.steer, np.float32),
                       logp=np.asarray(self.logp, np.float32),
                       value=np.asarray(self.value, np.float32),
                       reward=np.asarray(self.reward, np.float32))
        return out


class Pool:
    def __init__(self, model, value_head, device, workers, style_index, mode='train'):
        self.model = model
        self.value_head = value_head
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
        seat = torch.tensor([s.seat for s in states], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            memory = (torch.cat([s.memory[0] for s in states], 0),
                      torch.cat([s.memory[1] for s in states], 0))
            map_feat = (torch.cat([s.map_feat[0] for s in states], 0),
                        torch.cat([s.map_feat[1] for s in states], 0))
            style = torch.full((len(batch),), self.style_index, dtype=torch.long,
                               device=self.device)
            memory = self.model.advance_memory(obs, memory)
            out = self.model.step(obs, style, map_feat, memory, seat)
            slices = [(memory[0][k:k + 1].clone(), memory[1][k:k + 1].clone())
                      for k in range(len(batch))]
            logits = out['drive']
            mu = out['steer_mu'].squeeze(-1)
            logstd = out['steer_logstd'].squeeze(-1)
            fire_logit = out['fire']
            u = torch.rand(len(batch), 3, device=self.device)
            if self.mode == 'greedy':
                drive = logits.argmax(-1)
                steer = mu
                fire = (fire_logit > 0).float()
                logp = None
            else:
                cdf = logits.softmax(-1).cumsum(-1)
                drive = torch.searchsorted(cdf.contiguous(),
                                           (u[:, 0].clamp(0, 1 - EPS))[:, None].contiguous(),
                                           right=True).squeeze(1).clamp(0, 2)
                steer = torch.tanh(Normal(mu, logstd.exp()).icdf(
                    u[:, 1].clamp(EPS, 1 - EPS)))
                fire = (u[:, 2] < fire_logit.sigmoid()).float()
                logp = ((Categorical(logits=logits).log_prob(drive)
                         + Policy._squashed_logp(mu, logstd, steer))
                        if self.mode == 'train' else None)
            z = out['z']
            value = self.value_head(z).reshape(-1)
        drive_np = (drive.float() - 1.0).cpu().numpy()
        steer_np = steer.cpu().numpy()
        fire_np = fire.cpu().numpy()
        z_np = z.cpu().numpy() if self.mode == 'train' else None
        logp_np = logp.cpu().numpy() if logp is not None else None
        value_np = value.cpu().numpy() if value is not None else None
        for k, wid in enumerate(batch):
            state = states[k]
            state.memory = slices[k]
            if self.mode == 'train':
                state.record(z_np[k].copy(), np.array(state.obs, np.float32),
                             int(drive_np[k]) + 1, float(steer_np[k]),
                             float(logp_np[k]), float(value_np[k]))
            else:
                state.track(np.asarray(state.obs, np.float32))
            action = dict(drive=int(drive_np[k]), steer=float(steer_np[k]),
                          shoot=bool(fire_np[k] > 0.5))
            future = self.executor.submit(self.workers[wid].request, op='rl_act', slot=0,
                                          action=action)
            self.pending[future] = (wid, 'act')


def eval_rows(episodes):
    rows = []
    for ep in episodes:
        info = ep['info'] or {}
        rows.append({**ep['task'], **info,
                     'survival': float(ep['frames']) / 60.0,
                     'coverage': ep['coverage'], 'displacement': ep['displacement'],
                     'incoming': ep['incoming'], 'self_hits': ep['self_hits'],
                     'dodges': ep['dodges'],
                     'shots': (info.get('events') or {}).get('emissions', 0),
                     'kills': info.get('kills', 0), 'win': info.get('win', 0),
                     'timeout': info.get('timeout', 0)})
    return rows


def summarise(rows):
    fields = ['survival', 'win', 'timeout', 'incoming', 'self_hits', 'dodges',
              'coverage', 'displacement', 'shots', 'kills']
    out = dict(episodes=len(rows))
    for field in fields:
        out[field] = float(np.mean([r.get(field, 0) for r in rows]))
    return out


def run_eval(model, value_head, device, style_index, modes, workers):
    tasks = eval_tasks()
    out = {}
    for mode in modes:
        pool = Pool(model, value_head, device, min(workers, len(tasks)), style_index, mode=mode)
        episodes = pool.run(tasks)
        out[mode] = summarise(eval_rows(episodes))
    return out


def compute_gae(episode, gamma, lam):
    reward = episode['reward']
    value = episode['value']
    n = len(reward)
    adv = np.zeros(n, np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_value = float(value[t + 1]) if t + 1 < n else 0.0
        nonterminal = 1.0 if t + 1 < n else 0.0
        delta = float(reward[t]) + gamma * next_value * nonterminal - float(value[t])
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    episode['adv'] = adv
    episode['ret'] = adv + value


def update_stats(episodes):
    frames = int(sum(ep['frames'] for ep in episodes))
    return dict(episodes=len(episodes), frames=frames,
                survival=float(np.mean([ep['frames'] for ep in episodes])) / 60.0,
                coverage=float(np.mean([ep['coverage'] for ep in episodes])),
                displacement=float(np.mean([ep['displacement'] for ep in episodes])),
                incoming=float(np.mean([ep['incoming'] for ep in episodes])),
                self_hits=float(np.mean([ep['self_hits'] for ep in episodes])),
                dodges=float(np.mean([ep['dodges'] for ep in episodes])),
                deaths=float(np.mean([ep['deaths'] for ep in episodes])),
                wins=float(np.mean([ep['wins'] for ep in episodes])),
                timeout=float(np.mean([ep['timeout'] for ep in episodes])))


def ppo_update(model, value_head, episodes, opt_policy, opt_value, args):
    z = np.concatenate([ep['z'] for ep in episodes])
    obs = np.concatenate([ep['obs_in'] for ep in episodes])
    drive = np.concatenate([ep['drive'] for ep in episodes])
    steer = np.concatenate([ep['steer'] for ep in episodes])
    logp = np.concatenate([ep['logp'] for ep in episodes])
    ret = np.concatenate([ep['ret'] for ep in episodes])
    adv = np.concatenate([ep['adv'] for ep in episodes])
    seat = np.concatenate([np.full(len(ep['z']), ep['seat'], np.int64) for ep in episodes])

    sample = np.random.choice(len(z), min(args.value_rows, len(z)), replace=False)
    value_loss = fit_value(value_head, opt_value, z[sample], ret[sample], args)

    z_all = torch.from_numpy(np.ascontiguousarray(z)).to(args.device)
    with torch.no_grad():
        baseline = value_head(z_all).cpu().numpy()
    del z_all
    advantage = ret - baseline
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    adv = torch.from_numpy(advantage.astype(np.float32)).to(args.device)
    ret_t = torch.from_numpy(ret.astype(np.float32)).to(args.device)
    drive_t = torch.from_numpy(drive.astype(np.int64)).to(args.device)
    steer_t = torch.from_numpy(steer.astype(np.float32)).to(args.device)
    old_t = torch.from_numpy(logp.astype(np.float32)).to(args.device)
    seat_t = torch.from_numpy(seat.astype(np.int64)).to(args.device)

    losses, kls, ents = [], [], []
    for _ in range(args.epochs):
        order = np.random.permutation(len(z))
        for chunk in np.array_split(order, max(1, len(order) // args.batch)):
            index = torch.from_numpy(chunk).to(args.device)
            zb = torch.from_numpy(z[chunk]).to(args.device)
            ob = torch.from_numpy(obs[chunk]).to(args.device)
            sb = seat_t[index]
            move = model.move_adapter(ob, sb)
            logits = model.drive_head(zb) + move[:, :3]
            raw = model.steer_head(zb)
            mu = torch.tanh(raw[:, :1] + move[:, 3:4]).squeeze(-1)
            logstd = (raw[:, 1:2] + move[:, 4:5]).clamp(-4.0, 1.0).squeeze(-1)
            new = (Categorical(logits=logits).log_prob(drive_t[index])
                   + Policy._squashed_logp(mu, logstd, steer_t[index]))
            ratio = (new - old_t[index]).exp()
            surrogate = torch.minimum(ratio * adv[index],
                                      ratio.clamp(1 - args.clip, 1 + args.clip) * adv[index])
            entropy = (Categorical(logits=logits).entropy()
                       + Normal(mu, logstd.exp()).entropy()).mean()
            value = value_head(zb)
            loss = (-surrogate.mean() - args.entropy * entropy
                    + args.vf * F.mse_loss(value, ret_t[index]))
            opt_policy.zero_grad(set_to_none=True)
            opt_value.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.move_adapter.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 1.0)
            opt_policy.step()
            opt_value.step()
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
                samples=int(len(z)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--style', default='hy')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--episodes', type=int, default=512)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--seconds', type=int, default=10800)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--vlr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--entropy', type=float, default=0.01)
    p.add_argument('--vf', type=float, default=0.5)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--max-kl', type=float, default=0.02)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--value-rows', type=int, default=100000)
    p.add_argument('--value-batch', type=int, default=4096)
    p.add_argument('--value-steps', type=int, default=50)
    p.add_argument('--eval-every', type=int, default=8)
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--eval-modes', default='greedy,sampled')
    a = p.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model, meta = load_policy(a.init, a.device)
    model.eval()
    style_index = STYLES.index(a.style)
    value_head = ValueHead().to(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.eval_only:
        result = run_eval(model, value_head, a.device, style_index,
                          a.eval_modes.split(','), a.workers)
        atomic_json(out / 'dodge-eval.json', dict(checkpoint=a.init, style=a.style,
                                                  runs=result))
        for mode, stats in result.items():
            print('%-8s survival %.2fs  wins %.3f  timeout %.3f  incoming %.3f  self %.3f  '
                  'dodges %.2f  cover %.1f  speed %.1f  shots %.2f  kills %.3f'
                  % (mode, stats['survival'], stats['win'], stats['timeout'],
                     stats['incoming'], stats['self_hits'], stats['dodges'],
                     stats['coverage'], stats['displacement'], stats['shots'],
                     stats['kills']), flush=True)
        print('WROTE', out / 'dodge-eval.json')
        return

    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.move_adapter.parameters():
        param.requires_grad_(True)
    if 'value' in meta:
        try:
            value_head.load_state_dict(meta['value'])
        except Exception:
            pass
    opt_policy = torch.optim.Adam(model.move_adapter.parameters(), lr=a.lr)
    opt_value = torch.optim.Adam(value_head.parameters(), lr=a.vlr)

    history = []
    best = None
    start = time.monotonic()
    for update in range(1, a.updates + 1):
        if time.monotonic() - start >= a.seconds:
            print('time budget reached before update', update, flush=True)
            break
        tasks = dodge_tasks(a.seed, update, a.episodes)
        pool = Pool(model, value_head, a.device, min(a.workers, len(tasks)),
                    style_index, mode='train')
        rollout_start = time.monotonic()
        episodes = pool.run(tasks)
        rollout_seconds = time.monotonic() - rollout_start
        stats = update_stats(episodes)
        for ep in episodes:
            compute_gae(ep, a.gamma, a.lam)
        update_start = time.monotonic()
        losses = ppo_update(model, value_head, episodes, opt_policy, opt_value, a)
        update_seconds = time.monotonic() - update_start
        del episodes
        row = dict(update=update, rollout_seconds=round(rollout_seconds, 1),
                   update_seconds=round(update_seconds, 1), **stats, **losses,
                   reward=REWARD, style=a.style, base=a.init)
        save_checkpoint(model, value_head, out / 'last.pt', rl=row)
        if a.eval_every > 0 and (update % a.eval_every == 0 or update == a.updates):
            evals = run_eval(model, value_head, a.device, style_index,
                             ['greedy', 'sampled'], a.workers)
            row['eval'] = evals
            greedy = evals['greedy']
            if (greedy['coverage'] >= 15.0 and greedy['displacement'] >= 30.0
                    and (best is None or greedy['survival'] > best['survival'])):
                best = dict(update=update, **greedy)
                save_checkpoint(model, value_head, out / 'best.pt', rl=row)
            print('  eval greedy survival %.2fs cover %.1f speed %.1f incoming %.3f '
                  'self %.3f shots %.2f | sampled survival %.2fs cover %.1f speed %.1f'
                  % (greedy['survival'], greedy['coverage'], greedy['displacement'],
                     greedy['incoming'], greedy['self_hits'], greedy['shots'],
                     evals['sampled']['survival'], evals['sampled']['coverage'],
                     evals['sampled']['displacement']), flush=True)
        history.append(row)
        atomic_json(out / 'history.json', history)
        print('[u%02d] eps %d frames %d  survival %.2fs  cover %.1f  speed %.1f  '
              'incoming %.3f  self %.3f  dodges %.2f  wins %.3f  pg %.3f  v %.3f  kl %.4f  '
              '%ds+%ds'
              % (update, stats['episodes'], stats['frames'], stats['survival'],
                 stats['coverage'], stats['displacement'], stats['incoming'],
                 stats['self_hits'], stats['dodges'], stats['wins'],
                 losses['policy_loss'], losses['value_loss'], losses['kl'],
                 rollout_seconds, update_seconds), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
