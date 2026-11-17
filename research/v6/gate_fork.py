"""Fork gate: only a measured development win rate may release the two forks.

Thresholds live in fork-thresholds.json next to this file, so they can be
calibrated against a real baseline without touching any audited source. A gate
that is not met continues the shared phase; it never relaxes itself.
"""
import json
import pathlib
import sys

probe = json.loads(pathlib.Path(sys.argv[1]).read_text())
result = json.loads(pathlib.Path(sys.argv[2]).read_text())
out = pathlib.Path(sys.argv[3])
rules = json.loads((pathlib.Path(__file__).with_name('fork-thresholds.json')).read_text())

frac = float(rules['win_fraction_of_teacher'])
min_abs = float(rules['min_absolute_win'])
max_sui = float(rules['max_suicides'])
min_score = float(rules['min_best_shared_score'])

teacher = float(probe['teacher_win'])
student = float(probe['student_win'])
sui = max(float(probe['suicide_greedy']), float(probe['suicide_sampled']))
history = result.get('history') or []
# Under objective=win the segment's training score IS a win rate; this is a
# trend sanity check, not a second acceptance criterion.
best_score = max([float(x['score']) for x in history] or [-99.0])
required = frac * teacher

checks = {
    'teacher_baseline_positive': teacher > 0,
    'win_fraction_of_teacher': student >= required,
    'win_absolute_floor': student >= min_abs,
    'suicide_ceiling': sui <= max_sui,
    'shared_best_training_score': best_score > min_score,
}
report = dict(
    accepted=all(checks.values()), checks=checks,
    teacher_win=teacher, student_win=student,
    required_win_fraction=required, required_win_absolute=min_abs,
    suicides=sui, max_suicides=max_sui,
    shared_best_training_score=best_score, min_best_shared_score=min_score,
    shared_iterations=result.get('iterations'),
    probe_checkpoint=probe.get('checkpoint'),
    rule='student_win >= fraction*teacher_win AND >= absolute floor AND '
         'suicides <= ceiling AND shared best training score > floor')
out.write_text(json.dumps(report, indent=2))
print(('FORK_GATE_OK' if report['accepted'] else 'FORK_GATE_NOT_MET'),
      'teacher=%.4f student=%.4f need=%.4f/%.4f sui=%.3f shared_best=%.4f checks=%s' %
      (teacher, student, required, min_abs, sui, best_score, json.dumps(checks)), flush=True)
sys.exit(0 if report['accepted'] else 1)
