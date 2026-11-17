"""V8 Stage 1: multi-task supervised pretraining on Schema V3.

One model, two teachers (hy -> win style, gpt -> frag style), five losses. No reward, no
rollout: aim/threat/intent labels are functions of the state, generated offline by the
solver (policy/rl/v8/geometry_v8.mjs) and stored next to each frame.

  L = 1.0*L_action + 0.3*L_aim + 0.2*L_threat + 0.2*L_intent + 0.1*L_forward

Phase A trains single frames with zero memory: perception + aim, cheap and fast.
Phase B trains sequences so the per-opponent GRU learns from (state, action) history.

Usage:
  python -m policy.rl.v8.train --data /home/xy/rl_for_fun/data-v3/multi \
      --out /home/xy/rl_for_fun/v8-runs/s1a --phase a --steps 20000 --batch 1024
"""
import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..v4.common import STYLES
from .model import (Policy, OBS, ENEMY, ENEMY_SLOTS, ENEMY_STRIDE, BULLET, BULLET_SLOTS,
                    BULLET_STRIDE, AIM_N, INTENT_N, ACTION_DIM, save_policy)

FRAME, LABEL = 608, 333
STRIDE = FRAME + LABEL
L_TEACHER, L_SERVO, L_AIM, L_THREAT, L_INTENT = 0, 3, 7, 199, 319
STYLE_OF = {name: STYLES.index(name) for name in STYLES}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


class Corpus:
    """Random access over many .npy trajectories with a bounded file-handle cache."""

    def __init__(self, root, lru=96, split=None):
        root = Path(root)
        self.root = root
        self.records = []
        for line in (root / 'trajectories.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if split == 'train' and r['validation']:
                continue
            if split == 'validation' and not r['validation']:
                continue
            if r['length'] < 3:
                continue
            self.records.append(r)
        # `offset` in the manifest is a row index *within that world's* packed matrix,
        # not a global offset, so build the global prefix sum from the lengths instead.
        self.lengths = np.array([r['length'] for r in self.records], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.styles = np.array([STYLE_OF.get(r['teacher'], STYLE_OF['gpt']) for r in self.records],
                               dtype=np.int64)
        self.total = int(self.lengths.sum())
        self.lru = lru
        self.cache = OrderedDict()
        self.map_cache = OrderedDict()

    def file(self, fa):
        key = self.records[fa]['file']
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        array = np.load(self.root / key, mmap_mode='r')
        self.cache[key] = array
        if len(self.cache) > self.lru:
            self.cache.popitem(last=False)
        return array

    def map_of(self, fa):
        """346-dim wall grid, stored once per world rather than per trajectory."""
        key = self.records[fa]['map_file']
        if key in self.map_cache:
            self.map_cache.move_to_end(key)
            return self.map_cache[key]
        array = np.load(self.root / key).astype(np.float32)
        self.map_cache[key] = array
        if len(self.map_cache) > self.lru:
            self.map_cache.popitem(last=False)
        return array

    def sample_rows(self, batch, rng, with_next=False):
        idx = rng.integers(0, self.total, batch)
        fa = np.searchsorted(self.starts, idx, side='right') - 1
        row = idx - self.starts[fa]
        out = np.empty((batch, STRIDE), dtype=np.float32)
        nxt = np.empty((batch, STRIDE), dtype=np.float32) if with_next else None
        maps = np.empty((batch, 346), dtype=np.float32)
        for k in range(batch):
            a = self.file(fa[k])
            out[k] = a[row[k]]
            if with_next:
                nxt[k] = a[min(row[k] + 1, len(a) - 1)]
            maps[k] = self.map_of(fa[k])
        return out, nxt, self.styles[fa], maps

    def sample_blocks(self, batch, block, rng):
        """Contiguous runs from batch/blk trajectories.

        Per-row sampling costs 0.343 s/step against 0.048 s of forward pass: 1024 Python
        level mmap reads dominate. Slicing one run per trajectory is ~30x cheaper. The
        cost is temporal correlation inside a block, which is harmless here and gives the
        forward model genuine adjacent pairs for free.
        """
        nfiles = max(1, batch // block)
        fa = rng.integers(0, len(self.records), nfiles)
        out = np.empty((nfiles, block, STRIDE), dtype=np.float32)
        maps = np.empty((nfiles, block, 346), dtype=np.float32)
        for i, f in enumerate(fa):
            a = self.file(f)
            n = len(a)
            span = min(block, n)
            start = 0 if n <= span else int(rng.integers(0, n - span))
            out[i, :span] = a[start:start + span]
            if span < block:
                out[i, span:] = a[-1]
            maps[i] = self.map_of(f)
        return (out.reshape(nfiles * block, STRIDE),
                np.repeat(self.styles[fa], block),
                maps.reshape(nfiles * block, 346), block)

    def sample_sequences(self, batch, length, rng):
        ok = np.nonzero(self.lengths > length + 1)[0]
        fa = ok[rng.integers(0, len(ok), batch)]
        start = np.array([rng.integers(0, self.lengths[f] - length) for f in fa])
        out = np.empty((batch, length, STRIDE), dtype=np.float32)
        maps = np.empty((batch, 346), dtype=np.float32)
        for k, f in enumerate(fa):
            out[k] = self.file(f)[start[k]:start[k] + length]
            maps[k] = self.map_of(f)
        return out, self.styles[fa], maps


def to_tensors(block, device):
    x = torch.from_numpy(np.ascontiguousarray(block)).to(device)
    return x[..., :FRAME], x[..., FRAME:]


def action_targets(lab):
    """Teacher action plus the analytic servo label -> three factored targets."""
    drive = lab[..., L_TEACHER].round().clamp(-1, 1).long() + 1
    steer_teacher = lab[..., L_TEACHER + 1].clamp(-0.999, 0.999)
    has_safe = lab[..., L_SERVO + 2] > 0.5
    err = torch.atan2(lab[..., L_SERVO], lab[..., L_SERVO + 1])
    cone = (lab[..., L_SERVO + 3] * np.pi).clamp_min(1e-6)
    steer = torch.where(has_safe, (err / (3.3 / 60)).clamp(-0.999, 0.999), steer_teacher)
    fire = (has_safe & (err.abs() <= cone)) | (lab[..., L_TEACHER + 2] > 0.5)
    return drive, steer, fire.float()


def masks(obs):
    b = len(obs)
    enemy = obs[:, ENEMY:ENEMY + ENEMY_SLOTS * ENEMY_STRIDE].reshape(b, ENEMY_SLOTS, ENEMY_STRIDE)
    bullet = obs[:, BULLET:BULLET + BULLET_SLOTS * BULLET_STRIDE].reshape(b, BULLET_SLOTS, BULLET_STRIDE)
    return (enemy[..., 7] > 0.5).float(), (bullet[..., 7] > 0.5).float()


def multi_task_loss(model, out, lab, obs, next_obs, action, weights):
    b = len(lab)
    drive_t, steer_t, fire_t = action_targets(lab)

    l_drive = F.cross_entropy(out['drive'], drive_t)
    # The steer target is saturated at +/-1 on 68.5% of frames (the servo demanding full
    # turn), so a unimodal Gaussian NLL fights the bimodality. Fit mu with SmoothL1 and fit
    # sigma to the residual, with mu detached so the two do not fight.
    mu, logstd = out['steer_mu'].squeeze(-1), out['steer_logstd'].squeeze(-1)
    std = logstd.exp()
    l_steer = (F.smooth_l1_loss(mu, steer_t)
               + 0.1 * (logstd + 0.5 * ((steer_t - mu.detach()) / std) ** 2).mean())
    l_fire = F.binary_cross_entropy_with_logits(
        out['fire'], fire_t,
        pos_weight=torch.tensor(weights['fire_pos'], device=fire_t.device))

    # hit_enemy and hit_self are perfectly exclusive in the label (verified: BOTH = 0.000000),
    # so this is one 3-way classification, not two independent BCEs. Two BCEs cannot express
    # the exclusion and collapsed safe-direction recall to 0.0002 in the first attempt.
    aim = lab[..., L_AIM:L_AIM + AIM_N * 3].reshape(b, AIM_N, 3)
    cls = torch.where(aim[..., 0] > 0.5, 1, torch.where(aim[..., 1] > 0.5, 2, 0))
    p = out['aim']
    l_aim = (F.cross_entropy(p[..., :3].reshape(-1, 3), cls.reshape(-1))
             + F.smooth_l1_loss(torch.sigmoid(p[..., 3]), aim[..., 2]))

    threat = lab[..., L_THREAT:L_THREAT + BULLET_SLOTS * 3].reshape(b, BULLET_SLOTS, 3)
    t = out['threat']
    bm = (obs[:, BULLET + 7:BULLET + BULLET_SLOTS * BULLET_STRIDE:BULLET_STRIDE] > 0.5).float()
    per = (F.smooth_l1_loss(torch.sigmoid(t[..., 0]), threat[..., 0], reduction='none')
           + F.smooth_l1_loss(torch.sigmoid(t[..., 1]), threat[..., 1], reduction='none')
           + F.binary_cross_entropy_with_logits(t[..., 2], threat[..., 2], reduction='none'))
    l_threat = (per * bm).sum() / bm.sum().clamp_min(1.0)

    intent = lab[..., L_INTENT:L_INTENT + INTENT_N * 2].reshape(b, INTENT_N, 2)
    q = out['intent']
    em = (obs[:, ENEMY + 7:ENEMY + ENEMY_SLOTS * ENEMY_STRIDE:ENEMY_STRIDE] > 0.5).float()
    per_i = (F.binary_cross_entropy_with_logits(q[..., 1], intent[..., 1], reduction='none')
             + F.smooth_l1_loss(torch.sigmoid(q[..., 0]), intent[..., 0], reduction='none'))
    l_intent = (per_i * em).sum() / em.sum().clamp_min(1.0)

    l_forward = F.smooth_l1_loss(model.forward_head(torch.cat([out['z'], action], -1)), next_obs)

    total = (weights['action'] * (l_drive + l_steer + l_fire)
             + weights['aim'] * l_aim + weights['threat'] * l_threat
             + weights['intent'] * l_intent + weights['forward'] * l_forward)
    parts = dict(drive=float(l_drive), steer=float(l_steer), fire=float(l_fire),
                 aim=float(l_aim), threat=float(l_threat), intent=float(l_intent),
                 forward=float(l_forward))
    return total, parts


@torch.no_grad()
def evaluate(model, corpus, device, rng, batch=4096, buckets=6, style_name='hy'):
    """Offline metrics. Stage 1 exit criteria are read from here (see report.py)."""
    was_training = model.training
    model.eval()
    tp = fp = fn = tp_a = fp_a = fn_a = 0
    hit_e = hit_s = n_aim = steer_ok = cls_ok = total = 0
    steer_err, cones, drive_ok = [], [], 0
    for _ in range(buckets):
        arr, _, maps, _ = corpus.sample_blocks(batch, 32, rng)
        obs, lab = to_tensors(arr, device)
        map_feat = model.encode_map(torch.from_numpy(maps).to(device))
        style = torch.full((len(obs),), STYLE_OF[style_name], dtype=torch.long, device=device)
        memory = model.advance_memory(obs, model.zero_memory(len(obs), device))
        out = model.step(obs, style, map_feat, memory)
        drive_t, steer_t, fire_t = action_targets(lab)

        drive_ok += int((out['drive'].argmax(-1) == drive_t).sum())
        se = (out['steer_mu'].squeeze(-1) - steer_t).abs()
        steer_err.append(se)
        steer_ok += int((se <= 0.125).sum())       # engine command tolerance used by V7
        has_safe = lab[..., L_SERVO + 2] > 0.5
        cones.append(lab[..., L_SERVO + 3][has_safe])

        pred, true = out['fire'] > 0, fire_t > 0.5
        tp += int((pred & true).sum()); fp += int((pred & ~true).sum()); fn += int((~pred & true).sum())

        aim = lab[..., L_AIM:L_AIM + AIM_N * 3].reshape(-1, AIM_N, 3)
        pred_cls = out['aim'][..., :3].argmax(-1)
        true_cls = torch.where(aim[..., 0] > 0.5, 1, torch.where(aim[..., 1] > 0.5, 2, 0))
        ps, ts = pred_cls == 1, true_cls == 1
        tp_a += int((ps & ts).sum()); fp_a += int((ps & ~ts).sum()); fn_a += int((~ps & ts).sum())
        cls_ok += int((pred_cls == true_cls).sum())
        n_aim += aim[..., 0].numel()
        hit_e += int(aim[..., 0].sum()); hit_s += int(aim[..., 1].sum())
        total += len(obs)
    if was_training:
        model.train()

    s = torch.cat(steer_err)
    cn = torch.cat(cones) if sum(len(c) for c in cones) else torch.tensor([0.176], device=device)
    return dict(samples=total,
                drive_accuracy=drive_ok / max(1, total),
                steer_within_command=steer_ok / max(1, total),
                steer_abs_median=float(s.median()),
                steer_error_in_cones=float(s.median() / cn.median().clamp_min(1e-6)),
                fire_precision=tp / max(1, tp + fp),
                fire_recall=tp / max(1, tp + fn),
                aim_safe_precision=tp_a / max(1, tp_a + fp_a),
                aim_safe_recall=tp_a / max(1, tp_a + fn_a),
                aim_class_accuracy=cls_ok / max(1, n_aim),
                aim_hit_enemy_rate=hit_e / max(1, n_aim),
                aim_hit_self_rate=hit_s / max(1, n_aim))


@torch.no_grad()
def style_separation(model, corpus, device, rng, batch=4096):
    """Does the style embedding actually condition behaviour? KL between drive distributions."""
    block, _, _, maps = corpus.sample_rows(batch, rng)
    obs, _ = to_tensors(block, device)
    map_feat = model.encode_map(torch.from_numpy(maps).to(device))
    memory = model.advance_memory(obs, model.zero_memory(len(obs), device))
    was_training = model.training
    model.eval()
    out = {}
    for name in ('hy', 'gpt'):
        style = torch.full((len(obs),), STYLE_OF[name], dtype=torch.long, device=device)
        p = model.step(obs, style, map_feat, memory)['drive'].softmax(-1)
        out[name] = p.mean(0)
    if was_training:
        model.train()
    mid = 0.5 * (out['hy'] + out['gpt'])
    kl = float((out['hy'] * (out['hy'] / mid).clamp_min(1e-9).log()).sum()
               + (out['gpt'] * (out['gpt'] / mid).clamp_min(1e-9).log()).sum())
    return dict(drive_kl=kl,
                hy=[round(float(v), 4) for v in out['hy']],
                gpt=[round(float(v), 4) for v in out['gpt']])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='/home/xy/rl_for_fun/data-v3/multi')
    p.add_argument('--out', default='/home/xy/rl_for_fun/v8-runs/stage1')
    p.add_argument('--phase', choices=['a', 'b'], default='a')
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--seq', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--seconds', type=int, default=36000)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--init', default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--w-aim', type=float, default=0.3)
    p.add_argument('--w-threat', type=float, default=0.2)
    p.add_argument('--w-intent', type=float, default=0.2)
    p.add_argument('--w-forward', type=float, default=0.1)
    p.add_argument('--fire-pos', type=float, default=16.0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20261110)
    rng = np.random.default_rng(20261110)
    corpus = Corpus(a.data, split='train')
    print('corpus: %d trajectories, %d rows, %.1f GB'
          % (len(corpus.records), corpus.total, corpus.total * STRIDE * 4 / 1e9), flush=True)
    atomic_json(out / 'config.json', vars(a))

    model = Policy().to(a.device)
    if a.init:
        meta = torch.load(a.init, map_location='cpu', weights_only=False)
        model.load_state_dict(meta['model'])
        print('resumed from %s (step %s)' % (a.init, meta.get('step')), flush=True)
    print('parameters %.3fM' % (sum(q.numel() for q in model.parameters()) / 1e6), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps,
                                                pct_start=0.1, anneal_strategy='cos')
    w = dict(action=1.0, aim=a.w_aim, threat=a.w_threat, intent=a.w_intent,
             forward=a.w_forward, fire_pos=a.fire_pos)
    history, started, best, step = [], time.time(), -1e9, 0
    model.train()

    for step in range(1, a.steps + 1):
        if time.time() - started > a.seconds:
            print('time budget reached at step %d' % step, flush=True)
            break
        arr, sty, maps, blk = corpus.sample_blocks(a.batch, a.seq, rng)
        nfiles = len(arr) // blk
        obs_flat, lab_flat = to_tensors(arr, a.device)
        o = obs_flat.reshape(nfiles, blk, FRAME)
        nxt = torch.cat([o[:, 1:], o[:, -1:]], 1)
        if a.phase == 'a':
            obs, lab, next_obs = obs_flat, lab_flat, nxt.reshape(-1, FRAME)
            steps = 1
            map_rows = torch.from_numpy(maps).to(a.device)
        else:
            obs, lab, next_obs = o, lab_flat.reshape(nfiles, blk, LABEL), nxt
            steps = blk
            map_rows = torch.from_numpy(maps.reshape(nfiles, blk, 346)[:, 0]).to(a.device)

        b = len(map_rows)
        style = torch.from_numpy(sty).to(a.device)
        if a.phase == 'b':
            style = style.reshape(nfiles, blk)[:, 0]
        map_feat = model.encode_map(map_rows)
        memory = model.zero_memory(b, a.device)
        agg, total = {}, 0.0
        for k in range(steps):
            obs_k = obs[:, k] if a.phase == 'b' else obs
            lab_k = lab[:, k] if a.phase == 'b' else lab
            nxt_k = next_obs[:, k] if a.phase == 'b' else next_obs
            memory = model.advance_memory(obs_k, memory)
            surf = model.step(obs_k, style, map_feat, memory)
            drive_t, steer_t, fire_t = action_targets(lab_k)
            action = torch.stack([drive_t.float() - 1.0, steer_t, fire_t], -1)
            loss, parts = multi_task_loss(model, surf, lab_k, obs_k, nxt_k, action, w)
            total = total + loss / steps
            for key, value in parts.items():
                agg[key] = agg.get(key, 0.0) + value / steps

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            print('step %6d  loss %.4f  ' % (step, float(total))
                  + ' '.join('%s %.3f' % (k, v) for k, v in agg.items())
                  + '  lr %.2e  %.0fs' % (sched.get_last_lr()[0], time.time() - started),
                  flush=True)

        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, corpus, a.device, np.random.default_rng(7))
            sep = style_separation(model, corpus, a.device, np.random.default_rng(11))
            row = dict(step=step, seconds=round(time.time() - started, 1), loss=float(total),
                       style_kl=sep['drive_kl'], **{k: round(v, 6) for k, v in m.items()})
            history.append(row)
            atomic_json(out / 'history.json', history)
            save_policy(model, out / 'last.pt', step=step, phase=a.phase, metrics=m,
                        style_separation=sep)
            score = m['aim_safe_precision'] + m['aim_safe_recall'] + m['fire_precision'] + m['fire_recall']
            if score > best:
                best = score
                save_policy(model, out / 'best.pt', step=step, phase=a.phase, metrics=m,
                            style_separation=sep)
            print('  [eval %6d] drive %.4f  steer %.3f cones  fire P %.3f R %.3f  '
                  'aimsafe P %.3f R %.3f  styleKL %.4f'
                  % (step, m['drive_accuracy'], m['steer_error_in_cones'], m['fire_precision'],
                     m['fire_recall'], m['aim_safe_precision'], m['aim_safe_recall'],
                     sep['drive_kl']), flush=True)

    save_policy(model, out / 'final.pt', step=step, phase=a.phase)
    print('DONE %d steps in %.0fs' % (step, time.time() - started), flush=True)


if __name__ == '__main__':
    main()
