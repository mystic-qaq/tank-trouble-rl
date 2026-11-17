"""V8 L0: static target range — find the aim, then everything else.

Learner vs 1-3 frozen bots at their spawn points. The bots never move and never fire,
so the only skills that matter are turning the hull onto a target, firing when the shot
is actually on, and approaching around walls. Episode capped at 20 simulated seconds.

This is the first curriculum level. It exists because the frozen base hits only ~2.8% of
its shots (teachers ~14.6%) and both earlier RL attempts failed on that: a fire-only veto
collapsed to never firing, and a survival-only dodge learned to circle. L0 makes the kill
the objective from the first frame.

Reward per learner decision (user-specified shape):

    kill (hit == kill, one-hit-kill)        +2.0
    fire while the current shell would hit  +0.2
    aim progress                            +0.5 * (F(s') - F(s))
    self-hit                                -1.0
    time                                    -0.02 per second
    all targets dead                        +2.0

F(s) = -|bearing error to nearest enemy| / pi is a potential: the shaping term is dense,
telescopes to 0 while the policy holds an aim, and cannot be farmed by oscillation or by
idling. The first version paid only on "in-cone fire", which the base never reached
(in-cone rate 0.005), so the only live gradient was the self-hit penalty and the policy
learned to never fire (greedy 0 shots). The potential makes turning toward a target
worth a gradient before any shot can land.

The whole 20 s timeout costs -0.4, strictly less than one self-hit, so suiciding at the
start is never better than timing out. There is no separate death penalty: in L0 the only
possible death is a self-hit, already counted once above.

Trainable: the zero-initialised MoveAdapter (drive/steer) and the zero-initialised fire
adapter. The Stage 1 trunk, base heads and aux heads stay frozen. Same 96-worker rollout,
per-frame GAE and clipped PPO as the dodge stage.

Usage (server):
  python -m policy.rl.v8.l0 --init .../s1a/last.pt --out .../v8-runs/l0-a \
      --device cuda:0 --workers 96 --episodes 512 --updates 40
"""
import argparse
import concurrent.futures as cf
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Bernoulli, Categorical, Normal
from torch.nn import functional as F

from ..v4.common import STYLES, atomic_json
from ..v4.tasks import number
from .model import Policy, fire_features, load_policy
from .rl import ValueHead, fit_value, save_checkpoint
from .transport import Worker

EPS = 1e-6
DT = 1 / 60

REWARD = dict(kill=2.0, in_cone=0.2, self_hit=1.0, time=0.02, win=2.0, aim=0.5)
TIME_LIMIT = 20


def aim_potential(obs):
    """Potential F(s) = -|bearing error to nearest enemy| / pi, in [-1, 0].

    Potential-based shaping (F(s') - F(s)) is dense and telescopes: staying aimed gives
    zero, oscillating gives zero net, so it cannot be farmed. It only makes turning
    toward a target worth something before any shot can land.
    """
    ox, oy = float(obs[0]), float(obs[1])
    me_angle = math.atan2(float(obs[3]), float(obs[2]))
    best_bearing, best_d2 = None, 1e18
    for k in range(7):
        b = 11 + k * 9
        if obs[b + 7] <= 0.5:
            continue
        dx, dy = (float(obs[b]) - ox) * 1280.0, (float(obs[b + 1]) - oy) * 800.0
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_bearing = math.atan2(dy, dx)
    if best_bearing is None:
        return 0.0
    error = math.atan2(math.sin(best_bearing - me_angle), math.cos(best_bearing - me_angle))
    return -abs(error) / math.pi


def l0_tasks(seed, version, count):
    """1-3 frozen bots at random spawns; 20 s to clear."""
    rng = np.random.default_rng(number(seed, 'l0', version))
    rows = []
    for i in range(count):
        n = 2 + int(rng.integers(0, 3))          # 2..4 tanks = learner + 1..3 bots
        seat = int(rng.integers(n))
        map_seed = [20260101, 88888888][i % 2] if i % 3 else number(seed, 'l0-map', version, i)
        rows.append(dict(seed=number(seed, 'l0-layout', version, i),
                         spawnSeed=number(seed, 'l0-spawn', version, i),
                         mapSeed=map_seed, n=n, seat=seat, players=['builtin'] * n,
                         block='l0:%d' % version, case=i, objective='win',
                         early_death=True, frozen_bots=True, time_limit=TIME_LIMIT))
    return rows


def eval_tasks(count=70):
    return l0_tasks(975311, 0, count)


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
        self.fire = []
        self.logp = []
        self.value = []
        self.reward = []
        self.potential = []
        self.emissions = 0
        self.in_cone = 0
        self.hits = 0
        self.self_hits = 0
        self.wins = 0
        self.timeouts = 0
        self.info = None

    def record(self, z, observation, drive, steer, fire, logp, value):
        self.z.append(z)
        self.obs_in.append(observation)
        self.drive.append(drive)
        self.steer.append(steer)
        self.fire.append(fire)
        self.logp.append(logp)
        self.value.append(value)
        self.reward.append(0.0)
        self.potential.append(aim_potential(observation))

    def on_act(self, header, data, mode):
        self.obs = np.asarray(data, np.float32)
        t = len(self.z) - 1
        shot = header.get('shot') or [0, 0, 0]
        kills_now = 0
        self_now = 0
        emitted = False
        for event in header.get('events') or []:
            if event.get('kind') == 'emission':
                emitted = True
            elif event.get('kind') == 'hit' and event.get('owner') == self.seat:
                kills_now += 1
            elif event.get('kind') == 'self_hit':
                self_now += 1
        self.emissions += 1 if emitted else 0
        if emitted and int(shot[0]):
            self.in_cone += 1
        self.hits += kills_now
        self.self_hits += self_now
        if header.get('done'):
            self.info = header.get('info') or {}
            if int(self.info.get('win', 0)):
                self.wins += 1
            elif int(header.get('alive', 0)):
                self.timeouts += 1
        if mode != 'train':
            return
        reward = 0.0
        if int(header.get('active', 0)):
            reward -= REWARD['time'] * DT
        reward += REWARD['aim'] * (aim_potential(self.obs) - self.potential[t])
        if emitted and int(shot[0]):
            reward += REWARD['in_cone']
        reward += REWARD['kill'] * kills_now
        reward -= REWARD['self_hit'] * self_now
        if header.get('done') and int(header.get('alive', 0)) and int(self.info.get('win', 0)):
            reward += REWARD['win']
        self.reward[t] += reward

    def finish(self, mode):
        info = self.info or {}
        out = dict(task=self.task, info=info, seat=self.seat,
                   frames=int(info.get('frames', len(self.z))),
                   duration=float(info.get('duration', len(self.z) * DT)),
                   wins=self.wins, timeouts=self.timeouts,
                   emissions=self.emissions, in_cone=self.in_cone,
                   hits=self.hits, self_hits=self.self_hits,
                   kills=int(info.get('kills', 0)))
        if mode == 'train':
            out.update(z=np.stack(self.z).astype(np.float32),
                       obs_in=np.stack(self.obs_in).astype(np.float32),
                       drive=np.asarray(self.drive, np.int64),
                       steer=np.asarray(self.steer, np.float32),
                       fire=np.asarray(self.fire, np.float32),
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
                         + Policy._squashed_logp(mu, logstd, steer)
                         - F.binary_cross_entropy_with_logits(fire_logit, fire, reduction='none'))
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
                             int(drive_np[k]) + 1, float(steer_np[k]), int(fire_np[k]),
                             float(logp_np[k]), float(value_np[k]))
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
                     'duration': ep['duration'], 'win': info.get('win', 0),
                     'timeout': info.get('timeout', 0),
                     'shots': ep['emissions'], 'in_cone': ep['in_cone'],
                     'hits': ep['hits'], 'self_hits': ep['self_hits'],
                     'kills': ep['kills']})
    return rows


def summarise(rows):
    wins = [r for r in rows if r['win']]
    shots = max(sum(r['shots'] for r in rows), 1e-9)
    return dict(episodes=len(rows),
                win_rate=float(np.mean([r['win'] for r in rows])),
                timeout_rate=float(np.mean([r['timeout'] for r in rows])),
                clear_time=float(np.mean([r['duration'] for r in wins])) if wins else 0.0,
                shots=float(np.mean([r['shots'] for r in rows])),
                hit_per_shot=sum(r['hits'] for r in rows) / shots,
                self_per_shot=sum(r['self_hits'] for r in rows) / shots,
                in_cone_rate=sum(r['in_cone'] for r in rows) / shots,
                kills=float(np.mean([r['kills'] for r in rows])))


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
    wins = [ep['duration'] for ep in episodes if ep['wins']]
    return dict(episodes=len(episodes),
                frames=int(sum(ep['frames'] for ep in episodes)),
                win_rate=float(np.mean([ep['wins'] for ep in episodes])),
                timeout_rate=float(np.mean([ep['timeouts'] for ep in episodes])),
                clear_time=float(np.mean(wins)) if wins else 0.0,
                shots=float(np.mean([ep['emissions'] for ep in episodes])),
                hits=float(np.mean([ep['hits'] for ep in episodes])),
                self_hits=float(np.mean([ep['self_hits'] for ep in episodes])),
                in_cone_rate=float(sum(ep['in_cone'] for ep in episodes) /
                                   max(sum(ep['emissions'] for ep in episodes), 1)))


def ppo_update(model, value_head, episodes, opt_policy, opt_value, args):
    z = np.concatenate([ep['z'] for ep in episodes])
    obs = np.concatenate([ep['obs_in'] for ep in episodes])
    drive = np.concatenate([ep['drive'] for ep in episodes])
    steer = np.concatenate([ep['steer'] for ep in episodes])
    fire = np.concatenate([ep['fire'] for ep in episodes])
    logp = np.concatenate([ep['logp'] for ep in episodes])
    ret = np.concatenate([ep['ret'] for ep in episodes])
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
    fire_t = torch.from_numpy(fire.astype(np.float32)).to(args.device)
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
            fire_logit = (model.fire_head(zb).squeeze(-1)
                          + model.fire_adapter(fire_features(ob)).squeeze(-1))
            new = (Categorical(logits=logits).log_prob(drive_t[index])
                   + Policy._squashed_logp(mu, logstd, steer_t[index])
                   - F.binary_cross_entropy_with_logits(fire_logit, fire_t[index], reduction='none'))
            ratio = (new - old_t[index]).exp()
            surrogate = torch.minimum(ratio * adv[index],
                                      ratio.clamp(1 - args.clip, 1 + args.clip) * adv[index])
            entropy = (Categorical(logits=logits).entropy()
                       + Normal(mu, logstd.exp()).entropy()
                       + Bernoulli(logits=fire_logit).entropy()).mean()
            value = value_head(zb)
            loss = (-surrogate.mean() - args.entropy * entropy
                    + args.vf * F.mse_loss(value, ret_t[index]))
            opt_policy.zero_grad(set_to_none=True)
            opt_value.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
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
    p.add_argument('--seed', type=int, default=20261112)
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
        atomic_json(out / 'l0-eval.json', dict(checkpoint=a.init, style=a.style, runs=result))
        for mode, stats in result.items():
            print('%-8s win %.3f  timeout %.3f  clear %.2fs  shots %.2f  hit/shot %.3f  '
                  'self/shot %.3f  in-cone %.3f  kills %.2f'
                  % (mode, stats['win_rate'], stats['timeout_rate'], stats['clear_time'],
                     stats['shots'], stats['hit_per_shot'], stats['self_per_shot'],
                     stats['in_cone_rate'], stats['kills']), flush=True)
        print('WROTE', out / 'l0-eval.json')
        return

    for param in model.parameters():
        param.requires_grad_(False)
    for module in (model.move_adapter, model.fire_adapter):
        for param in module.parameters():
            param.requires_grad_(True)
    if 'value' in meta:
        try:
            value_head.load_state_dict(meta['value'])
        except Exception:
            pass
    opt_policy = torch.optim.Adam(
        list(model.move_adapter.parameters()) + list(model.fire_adapter.parameters()), lr=a.lr)
    opt_value = torch.optim.Adam(value_head.parameters(), lr=a.vlr)

    history = []
    best = None
    start = time.monotonic()
    for update in range(1, a.updates + 1):
        if time.monotonic() - start >= a.seconds:
            print('time budget reached before update', update, flush=True)
            break
        tasks = l0_tasks(a.seed, update, a.episodes)
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
            if best is None or greedy['win_rate'] > best['win_rate']:
                best = dict(update=update, **greedy)
                save_checkpoint(model, value_head, out / 'best.pt', rl=row)
            print('  eval greedy win %.3f clear %.2fs hit/shot %.3f self/shot %.3f '
                  'in-cone %.3f | sampled win %.3f clear %.2fs hit/shot %.3f self/shot %.3f'
                  % (greedy['win_rate'], greedy['clear_time'], greedy['hit_per_shot'],
                     greedy['self_per_shot'], greedy['in_cone_rate'],
                     evals['sampled']['win_rate'], evals['sampled']['clear_time'],
                     evals['sampled']['hit_per_shot'], evals['sampled']['self_per_shot']),
                  flush=True)
        history.append(row)
        atomic_json(out / 'history.json', history)
        print('[u%02d] eps %d frames %d  win %.3f  clear %.2fs  shots %.2f  hits %.2f  '
              'self %.2f  in-cone %.3f  pg %.3f  v %.3f  kl %.4f  %ds+%ds'
              % (update, stats['episodes'], stats['frames'], stats['win_rate'],
                 stats['clear_time'], stats['shots'], stats['hits'], stats['self_hits'],
                 stats['in_cone_rate'], losses['policy_loss'], losses['value_loss'],
                 losses['kl'], rollout_seconds, update_seconds), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
