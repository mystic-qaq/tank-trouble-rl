"""Sanity-check a Schema V3 collection directory before committing hours of compute.

Checks shape, finiteness, and the label distributions. The point is to catch a broken
label generator cheaply: if aim/hit rates or the servo's has_safe fraction are wildly
off from what the engine implies, the corpus is worthless and no amount of training
will fix it.

Usage: python -m policy.rl.v8.inspect --data /home/xy/rl_for_fun/data-v3/multi
"""
import argparse
import json
from pathlib import Path

import numpy as np

S = dict(frame=608, labels=333, stride=941, map=346,
         SELF_ACTION=8, ENEMY=11, ENEMY_STRIDE=9, ENEMY_ACTION=74, ENEMY_ACTION_STRIDE=3,
         BULLET=95, BULLET_STRIDE=8, LIDAR=415, LIDAR_STRIDE=3, CLOCK=607,
         L_TEACHER=0, L_SERVO=3, L_AIM=7, L_AIM_STRIDE=3, L_THREAT=199,
         L_THREAT_STRIDE=3, L_INTENT=319, L_INTENT_STRIDE=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--files', type=int, default=24)
    a = p.parse_args()

    traj = [json.loads(line) for line in (a.data / 'trajectories.jsonl').read_text().splitlines() if line.strip()]
    print('trajectories %d  teachers %s' % (
        len(traj), {k: sum(1 for t in traj if t['teacher'] == k) for k in {t['teacher'] for t in traj}}))
    lengths = [t['length'] for t in traj]
    print('length frames: min %d  p50 %d  mean %.0f  max %d  total %d'
          % (min(lengths), int(np.median(lengths)), float(np.mean(lengths)), max(lengths), sum(lengths)))

    rng = np.random.default_rng(0)
    pick = [traj[i] for i in rng.choice(len(traj), min(a.files, len(traj)), replace=False)]
    rows = []
    maps = []
    for t in pick:
        arr = np.load(a.data / t['file'])
        assert arr.shape[1] == S['stride'], (t['file'], arr.shape)
        assert np.isfinite(arr).all(), t['file']
        rows.append(arr)
        maps.append(np.load(a.data / t['map_file']))
    X = np.concatenate(rows)
    M = np.stack(maps)
    print('sampled %d files -> %d rows, finite OK' % (len(pick), len(X)))
    print('map: %s  values in {0,1}: %s  mean %.4f' % (
        M.shape, bool(np.isin(M, [0, 1]).all()), float(M.mean())))

    f, l = X[:, :S['frame']], X[:, S['frame']:]

    def frac(v):
        return float(v.mean())

    print('\n--- 输入块 ---')
    print('self alive            %.4f' % frac(f[:, 6]))
    print('self cooldown/0.42    mean %.3f  p90 %.3f' % (f[:, 5].mean(), np.percentile(f[:, 5], 90)))
    print('self ammo/5           mean %.3f' % f[:, 7].mean())
    en = f[:, S['ENEMY']:S['ENEMY'] + 7 * S['ENEMY_STRIDE']].reshape(-1, 7, S['ENEMY_STRIDE'])
    print('enemy presence        %.4f' % en[..., 7].mean())
    ea = f[:, S['ENEMY_ACTION']:S['ENEMY_ACTION'] + 7 * S['ENEMY_ACTION_STRIDE']].reshape(-1, 7, 3)
    present = en[..., 7] > 0
    print('enemy steer  in [-1,1]: %.4f  mean|.| %.4f' % (
        frac((ea[..., 0].ravel()[present.ravel()].__abs__() <= 1.0001)),
        float(np.abs(ea[..., 0][present]).mean())))
    print('enemy shoot  rate     %.4f' % frac(ea[..., 1][present]))
    print('enemy dspeed  mean %.4f  p99|.| %.4f' % (
        float(ea[..., 2][present].mean()), float(np.percentile(np.abs(ea[..., 2][present]), 99))))
    bu = f[:, S['BULLET']:S['BULLET'] + 40 * S['BULLET_STRIDE']].reshape(-1, 40, S['BULLET_STRIDE'])
    print('bullet presence       %.4f  (mean live bullets/row %.2f)' % (
        bu[..., 7].mean(), bu[..., 7].sum(1).mean()))
    print('bullet owner/7 range  [%.3f, %.3f]' % (bu[..., 6][bu[..., 7] > 0].min(), bu[..., 6][bu[..., 7] > 0].max()))
    li = f[:, S['LIDAR']:S['LIDAR'] + 64 * S['LIDAR_STRIDE']].reshape(-1, 64, S['LIDAR_STRIDE'])
    print('lidar distance        mean %.3f  frac<0.99 %.3f' % (li[..., 0].mean(), frac(li[..., 0] < 0.99)))
    print('lidar normal |n|      mean %.3f  (0 means no hit)' % np.linalg.norm(li[..., 1:3], axis=-1).mean())
    print('clock                 mean %.3f' % f[:, S['CLOCK']].mean())

    print('\n--- 标签块 ---')
    print('teacher drive ∈{-1,0,1}     %s' % sorted(set(np.round(l[:, 0], 3).tolist())))
    print('teacher steer  in [-1,1]    %.4f' % frac(np.abs(l[:, 1]) <= 1.0001))
    print('teacher shoot rate          %.4f' % frac(l[:, 2]))
    print('servo has_safe              %.4f' % frac(l[:, S['L_SERVO'] + 2]))
    safe = l[:, S['L_SERVO'] + 2] > 0
    if safe.any():
        err = np.degrees(np.arctan2(l[safe, S['L_SERVO']], l[safe, S['L_SERVO'] + 1]))
        print('servo |err| deg             p50 %.2f  p90 %.2f' % (
            np.percentile(np.abs(err), 50), np.percentile(np.abs(err), 90)))
        print('servo cone/pi               mean %.4f' % l[safe, S['L_SERVO'] + 3].mean())
    aim = l[:, S['L_AIM']:S['L_AIM'] + 64 * S['L_AIM_STRIDE']].reshape(-1, 64, 3)
    print('aim  hit_enemy rate         %.4f  (any per row %.4f)' % (
        frac(aim[..., 0]), frac(aim[..., 0].max(1))))
    print('aim  hit_self  rate         %.4f  (any per row %.4f)' % (
        frac(aim[..., 1]), frac(aim[..., 1].max(1))))
    print('aim  ttc mean %.3f  (nonzero %.4f)' % (
        float(aim[..., 2].mean()), frac(aim[..., 2] > 0)))
    th = l[:, S['L_THREAT']:S['L_THREAT'] + 40 * S['L_THREAT_STRIDE']].reshape(-1, 40, 3)
    live_mask = bu[..., 7] > 0
    print('threat will_hit_me rate     %.4f  (per live bullet %.4f)' % (
        frac(th[..., 2]), frac(th[..., 2][live_mask]) if live_mask.any() else 0))
    print('threat ttc mean (live)      %.4f' % float(th[..., 0][live_mask].mean()) if live_mask.any() else 0)
    it = l[:, S['L_INTENT']:S['L_INTENT'] + 7 * S['L_INTENT_STRIDE']].reshape(-1, 7, 2)
    print('intent tracking-my rate     %.4f' % frac(it[..., 1][present]))
    print('intent bearing_rate |.|mean %.4f' % float(np.abs(it[..., 0][present]).mean()))

    print('\n--- 派生动作标签一致性 ---')
    # steer* = servo when has_safe else teacher steer
    steer_label = np.where(safe, np.arctan2(l[:, S['L_SERVO']], l[:, S['L_SERVO'] + 1])
                           / (3.3 / 60), l[:, 1])
    steer_label = np.clip(steer_label, -1, 1)
    fire_label = ((safe & (np.abs(np.arctan2(l[:, S['L_SERVO']], l[:, S['L_SERVO'] + 1])) <=
                           np.clip(l[:, S['L_SERVO'] + 3] * np.pi, 1e-6, None)))
                  | (l[:, 2] > 0.5)).astype(np.float32)
    print('steer* in [-1,1]            %.4f  mean|.| %.3f' % (
        frac(np.abs(steer_label) <= 1.0001), float(np.abs(steer_label).mean())))
    print('fire*  rate                 %.4f  (teacher %.4f, analytic %.4f)' % (
        frac(fire_label), frac(l[:, 2]), frac(safe & (np.abs(np.arctan2(l[:, S['L_SERVO']], l[:, S['L_SERVO'] + 1]))
                                                   <= np.clip(l[:, S['L_SERVO'] + 3] * np.pi, 1e-6, None)))))
    print('\nOK')


if __name__ == '__main__':
    main()
