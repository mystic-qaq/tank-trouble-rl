"""Export the V8 actor to ONNX and prove it matches PyTorch exactly.

V7 could not be exported: its graph contained atan2/argmin/gather from the analytic aim
readout, plus adaptive_avg_pool2d on a non-factor size, plus a broken exporter fallback
chain, plus a wrong baked style index. V8 removes the first (the solver is label-only) and
the third (pool_spatial uses slices), and bakes the style explicitly, so this should be a
plain export. This script proves it rather than assuming it.

The graph is recurrent: slot memory (1,7,128) and global memory (1,128) are explicit
inputs and outputs, one frame per call. Decoding is deterministic (argmax drive, mu steer,
threshold fire) so the graph needs no sampling ops.

Usage: python -m policy.rl.v8.export --checkpoint ... --out DIR
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..v4.common import STYLES
from .model import Policy, load_policy, OBS, ENEMY_SLOTS, MEMORY

OPSET = 17


class Actor(torch.nn.Module):
    """obs, map, style, memory -> (controls[3], slot_memory, global_memory)."""

    def __init__(self, policy, style_index):
        super().__init__()
        self.policy = policy
        self.register_buffer('style', torch.tensor([style_index], dtype=torch.long))

    def forward(self, obs, map_vec, slot_mem, glob_mem):
        map_feat = self.policy.encode_map(map_vec)
        slot_mem, glob_mem = self.policy.advance_memory(obs, (slot_mem, glob_mem))
        out = self.policy.step(obs, self.style, map_feat, (slot_mem, glob_mem))
        drive = out['drive'].argmax(-1).float() - 1.0
        steer = out['steer_mu'].squeeze(-1)
        fire = (out['fire'] > 0).float()
        return torch.stack([drive, steer, fire], -1), slot_mem, glob_mem


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sample_batch(data, count, rng):
    root = Path(data)
    records = [json.loads(l) for l in (root / 'trajectories.jsonl').read_text().splitlines() if l.strip()]
    pick = [records[i] for i in rng.choice(len(records), min(64, len(records)), replace=False)]
    obs, maps = [], []
    for r in pick:
        a = np.load(root / r['file'])
        m = np.load(root / r['map_file'])
        take = min(16, len(a))
        idx = rng.choice(len(a), take, replace=False)
        obs.append(a[idx, :OBS])
        maps.append(np.repeat(m[None], take, 0))
        if sum(len(x) for x in obs) >= count:
            break
    obs = np.concatenate(obs)[:count].astype(np.float32)
    maps = np.concatenate(maps)[:count].astype(np.float32)
    edge = np.zeros((4, OBS), dtype=np.float32)
    edge[0, 6] = 1.0
    edge[1, 6] = 1.0; edge[1, 16] = 1.0
    edge[2, 16] = 1.0
    obs = np.concatenate([obs, edge])
    maps = np.concatenate([maps, np.repeat(maps[:1], 4, 0)])
    return torch.from_numpy(obs), torch.from_numpy(maps), len(obs) - 4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='/home/xy/rl_for_fun/data-v3/multi')
    p.add_argument('--out', default='/home/xy/rl_for_fun/v8-runs/onnx')
    p.add_argument('--id', default='v8-win')
    p.add_argument('--style', default='hy')
    p.add_argument('--states', type=int, default=4000)
    a = p.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.checkpoint).resolve()

    model, meta = load_policy(str(ckpt), 'cpu')
    model.eval()
    style_index = STYLES.index(a.style)
    actor = Actor(model, style_index).eval()

    rng = np.random.default_rng(20261110)
    obs, maps, n_real = sample_batch(a.data, a.states, rng)
    print('parity states %d (%d real + 4 edge)' % (len(obs), n_real), flush=True)

    def reference(o, m, sm, gm):
        o1, m1, sm1, gm1 = o[:1], m[:1], sm, gm
        return actor(o1, m1, sm1, gm1)

    zero_slot = torch.zeros(1, ENEMY_SLOTS, MEMORY)
    zero_glob = torch.zeros(1, MEMORY)
    ref_c, ref_s, ref_g = [], [], []
    slot, glob = zero_slot, zero_glob
    t0 = time.time()
    for i in range(len(obs)):
        c, slot, glob = actor(obs[i:i + 1], maps[i:i + 1], slot, glob)
        ref_c.append(c)
        ref_s.append(slot.clone())
        ref_g.append(glob.clone())
    ctl_ref = torch.cat(ref_c)
    print('pytorch reference %.1fs' % (time.time() - t0), flush=True)

    path = out / ('%s.onnx' % a.id)
    torch.onnx.export(actor, (obs[:1].contiguous(), maps[:1].contiguous(), zero_slot, zero_glob),
                      str(path), input_names=['obs', 'map_vec', 'slot_mem', 'glob_mem'],
                      output_names=['controls', 'slot_mem_out', 'glob_mem_out'],
                      opset_version=OPSET, dynamo=False)
    print('exported %s (%d bytes)' % (path.name, path.stat().st_size), flush=True)

    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(path)))
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    ctl_o, s_o, g_o = [], [], []
    slot_np = np.zeros((1, ENEMY_SLOTS, MEMORY), np.float32)
    glob_np = np.zeros((1, MEMORY), np.float32)
    for i in range(len(obs)):
        c, s, g = sess.run(None, {'obs': obs[i:i + 1].numpy(), 'map_vec': maps[i:i + 1].numpy(),
                                  'slot_mem': slot_np, 'glob_mem': glob_np})
        ctl_o.append(c)
        slot_np, glob_np = s, g
    ctl_o = torch.from_numpy(np.concatenate(ctl_o))

    finite = bool(torch.isfinite(ctl_o).all())
    drive_match = float((ctl_ref[:, 0] == ctl_o[:, 0]).float().mean())
    steer_err = float((ctl_ref[:, 1] - ctl_o[:, 1]).abs().max())
    fire_match = float((ctl_ref[:, 2] == ctl_o[:, 2]).float().mean())
    mismatch = (ctl_ref != ctl_o).any(-1).nonzero().flatten().tolist()
    mem_err = float(max((torch.from_numpy(slot_np) - ref_s[-1]).abs().max(),
                        (torch.from_numpy(glob_np) - ref_g[-1]).abs().max()))

    print('parity: finite=%s drive=%.6f fire=%.6f max_steer_err=%.3e memory_err=%.3e mismatches=%d/%d'
          % (finite, drive_match, fire_match, steer_err, mem_err, len(mismatch), len(obs)), flush=True)

    descriptor = dict(
        id=a.id, source_checkpoint=str(ckpt), source_sha256=sha256(ckpt),
        onnx_sha256=sha256(path), architecture='perception_factored_memory_v8',
        observation='schema_v3', deterministic=True, policy_style=a.style,
        style_index=style_index, opset=OPSET, batch='static 1',
        inputs=['obs(1,608)', 'map_vec(1,346)', 'slot_mem(1,7,128)', 'glob_mem(1,128)'],
        verification_states=int(len(obs)), edge_cases=4,
        outputs_finite=finite, drive_match=drive_match, fire_match=fire_match,
        max_steer_error=steer_err, memory_error=mem_err,
        mismatched_frames=len(mismatch), mismatches=mismatch[:32])
    (out / ('%s.json' % a.id)).write_text(json.dumps(descriptor, indent=2, ensure_ascii=False))
    ok = finite and drive_match == 1.0 and fire_match == 1.0 and not mismatch
    print('EXPORT_OK=%s WROTE %s' % (ok, out / ('%s.json' % a.id)), flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
