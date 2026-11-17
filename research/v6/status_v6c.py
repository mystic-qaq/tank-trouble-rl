#!/usr/bin/env python3
"""One-shot V6C status: shared segments, win-rate probes, fork gates, resources."""
import datetime
import json
import pathlib
import subprocess

V = pathlib.Path('$RL_ROOT/v6-runs')
print('===', datetime.datetime.now().strftime('%F %T %Z'), '===')
print('-- shared / forks --')
for name in ['shared2', 'shared3-s1', 'shared3-s2', 'shared3-s3', 'shared3-s4',
             'fork3-win', 'fork3-frag']:
    d = V / name
    s = d / 'status.json'
    if s.exists():
        st = json.loads(s.read_text())
        extra = ''
        p = d / 'probe.json'
        if p.exists():
            q = json.loads(p.read_text())
            extra = ' | dev_win teacher=%.3f student=%.3f' % (q['teacher_win'], q['student_win'])
        g = d / 'fork-gate.json'
        if g.exists():
            extra += ' | fork_gate=%s' % json.loads(g.read_text())['accepted']
        a = d / 'acceptance.json'
        if a.exists():
            extra += ' | accepted=%s' % json.loads(a.read_text()).get('accepted')
        print('%-12s iter=%3d %-8s=%+.4f kills=%.3f sui=%.3f kl=%.4f n<=%d collect=%.0fs%s' %
              (name, st['iteration'], 'win' if st['objective'] == 'win' else 'netKills',
               st['score'], st['kills'], st['suicides'], st['mean_kl'],
               st['max_tanks'], st['collection_seconds'], extra))
print('-- pipeline3 tail --')
p = V / 'pipeline3.log'
if p.exists():
    print('\n'.join(p.read_text().splitlines()[-8:]))
print('-- resources --')
out = subprocess.run(
    ['bash', '-lc',
     'free -h | head -2; echo -n "node_workers="; ps -eo cmd | grep -c "[w]orker.mjs"; '
     'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -4'],
    capture_output=True, text=True).stdout
print(out.rstrip())
