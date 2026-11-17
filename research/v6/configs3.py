"""V6C configs. Every output stays in v6-runs/, never inside the release tree,
so source_digest() and the G0 contracts / receipt2.json remain valid.
"""
import json
import pathlib

V = pathlib.Path('$RL_ROOT/v6-runs')

# --- extended shared phase -------------------------------------------------
# Objective is now WIN, not frag. A turtle init (dev win 0.0536 < 0.20 random)
# would get almost no positive win signal against full-strength rosters, so
# re-enable the weak-opponent/n ramp: it is the one change V5/V6B measured as
# effective for getting a combat policy off the ground.
c = json.loads((V / 'shared2-config.json').read_text())
o = c['optimize']
o['curriculum']['enabled'] = True
o['curriculum']['ramp_iterations'] = 15
o['iterations'] = 600          # wall clock is the real limit
o['episodes'] = 64
o['patience'] = 999            # never early-stop: we are waiting for a win rate
o['min_iterations'] = 999
o['teacher_aux'] = 0.0
o['bootstrap_steps'] = 100     # ignored when --init is supplied
(V / 'shared3-base-config.json').write_text(json.dumps(c, indent=2))

# --- win-rate probe: identical dev split / seed as acceptance --------------
p = json.loads((V / 'fork2-config.json').read_text())
p['workers'] = 16              # be considerate on a shared machine
(V / 'probe-config.json').write_text(json.dumps(p, indent=2))

# --- forks: no ramp either, more training time -----------------------------
f = json.loads((V / 'fork2-config.json').read_text())
f['optimize']['curriculum']['enabled'] = False
f['optimize']['iterations'] = 300
(V / 'fork3-config.json').write_text(json.dumps(f, indent=2))

print('shared3-base: curriculum=on ramp=%d iterations=%d patience=%d min_iter=%d '
      'kill_aux=%s survival_bonus=%s death_penalty=%s' %
      (o['curriculum']['ramp_iterations'], o['iterations'], o['patience'],
       o['min_iterations'], o['kill_aux'], o['survival_bonus'], o['death_penalty']))
print('probe: workers=%d dev_blocks=%s seed=%s' %
      (p['workers'], p['suites']['dev_blocks'], p['seed']))
print('fork3: curriculum=off iterations=%d' % f['optimize']['iterations'])
