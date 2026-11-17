"""Label distribution audit: is the aim target exclusive, and is the steer target learnable?"""
import sys

import numpy as np

sys.path.insert(0, '.')
from policy.rl.v8.train import Corpus, L_TEACHER, L_SERVO, L_AIM, AIM_N  # noqa: E402

c = Corpus('/home/xy/rl_for_fun/data-v3/multi', split='train')
rng = np.random.default_rng(3)
rows, _, _, _ = c.sample_blocks(8192, 32, rng)
lab = rows[:, 608:]
teach = lab[:, L_TEACHER:L_TEACHER + 3]

vals, counts = np.unique(teach[:, 0].round(), return_counts=True)
print('teacher drive dist   ' + str({float(k): round(float(v) / counts.sum(), 4) for k, v in zip(vals, counts)}))
print('teacher shoot rate   %.4f' % teach[:, 2].mean())
print('has_safe             %.4f' % lab[:, L_SERVO + 2].mean())

aim = lab[:, L_AIM:L_AIM + AIM_N * 3].reshape(-1, AIM_N, 3)
he = aim[..., 0] > 0.5
hs = aim[..., 1] > 0.5
print('aim hit_enemy        %.4f' % he.mean())
print('aim hit_self         %.4f' % hs.mean())
print('aim BOTH             %.6f' % (he & hs).mean())
print('aim safe             %.4f' % (he & ~hs).mean())
print('exclusivity both/any %.5f' % (float((he & hs).sum()) / max(1.0, float((he | hs).sum()))))

mask = lab[:, L_SERVO + 2] > 0.5
err = np.arctan2(lab[:, L_SERVO], lab[:, L_SERVO + 1])
srv = np.clip(err / (3.3 / 60), -0.999, 0.999)
st = np.where(mask, srv, lab[:, L_TEACHER + 1])
print('steer target saturated (|>=0.99) %.3f' % (np.abs(st) >= 0.99).mean())
print('steer target near zero  (|<=0.125) %.3f' % (np.abs(st) <= 0.125).mean())
print('steer target median |.| %.3f' % np.median(np.abs(st)))
print('always-zero baseline within 0.125 %.3f' % (np.abs(st) <= 0.125).mean())

# fire label and its dependence on the servo
cone = (lab[:, L_SERVO + 3] * np.pi).clip(1e-6)
fire_analytic = mask & (np.abs(err) <= cone)
fire_teacher = lab[:, L_TEACHER + 2] > 0.5
print('fire analytic %.4f  teacher %.4f  union %.4f'
      % (fire_analytic.mean(), fire_teacher.mean(), (fire_analytic | fire_teacher).mean()))
print('in-cone among has_safe %.4f' % (np.abs(err[mask]) <= cone[mask]).mean())
