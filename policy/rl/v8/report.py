"""Standalone offline report for a V8 checkpoint.

Training-time eval prints as it goes; this recomputes the same metrics from a saved
checkpoint so the numbers can be re-derived (and re-derived after a metric fix) without
retraining. Exit criteria for Stage 1 come from here.

Usage: python -m policy.rl.v8.report --checkpoint .../final.pt --split validation
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import load_policy
from .train import Corpus, evaluate, style_separation

EXIT = dict(drive_accuracy=0.90, steer_within_command=0.85,
            fire_precision=0.85, fire_recall=0.65, aim_safe_recall=0.70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='/home/xy/rl_for_fun/data-v3/multi')
    p.add_argument('--split', default='train', choices=['train', 'validation'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--buckets', type=int, default=16)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--out', default=None)
    a = p.parse_args()

    torch.set_grad_enabled(False)
    model, meta = load_policy(a.checkpoint, a.device)
    model.eval()
    corpus = Corpus(a.data, split=a.split)
    print('checkpoint %s (step %s)  corpus %d rows (%s)'
          % (a.checkpoint, meta.get('step'), corpus.total, a.split), flush=True)

    rng = np.random.default_rng(11)
    metrics = evaluate(model, corpus, a.device, rng, batch=a.batch, buckets=a.buckets)
    sep = style_separation(model, corpus, a.device, rng, batch=a.batch)

    print(json.dumps(metrics, indent=2))
    print('style separation KL %.4f  hy %s  gpt %s'
          % (sep['drive_kl'], sep['hy'], sep['gpt']))
    verdict = {k: bool(metrics[k] >= v) for k, v in EXIT.items()}
    print('exit criteria: ' + '  '.join('%s=%s' % (k, 'PASS' if v else 'FAIL')
                                       for k, v in verdict.items()))
    print('STAGE1_OFFLINE_PASS=%s' % all(verdict.values()))

    if a.out:
        Path(a.out).write_text(json.dumps(dict(checkpoint=a.checkpoint, step=meta.get('step'),
                                               split=a.split, metrics=metrics,
                                               style_separation=sep, exit=EXIT,
                                               verdict=verdict,
                                               pass_all=all(verdict.values())), indent=2))


if __name__ == '__main__':
    main()
