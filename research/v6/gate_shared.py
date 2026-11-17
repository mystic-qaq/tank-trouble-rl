"""V6B -> V6C handoff gate.

Operator decision: keep the SHARED phase on the dense netKills objective for
much longer, and only fork once a measured win rate appears. This module runs
the FULL, unchanged shared quality check and records the handoff, but ends the
V6B shell with a non-zero status so run_v6b.sh cannot start the two fork2 runs.
Nothing is discarded: shared2/best.pt and result.json stay as the audited
checkpoint that V6C initialises from.

This file lives outside the source_digest roots (js/, arena/, policy/), so the
G0 contracts and receipt2.json remain valid.
"""
import json
import math
import pathlib
import sys

import torch

out = pathlib.Path(sys.argv[1])
r = json.loads((out / "result.json").read_text())
h = r["history"]
best_iter = r.get("best_checkpoint_iteration")
row = next((x for x in h if x["iteration"] == best_iter), None)
ckpt = out / "best.pt"
problems = []

if r.get("status") != "training_complete_acceptance_skipped":
    problems.append("status=" + str(r.get("status")))
if r.get("iterations", 0) < 30:
    problems.append("iterations=%d<30" % r.get("iterations", 0))
if row is None:
    problems.append("best iteration absent from history")
if not all(math.isfinite(x["mean_kl"]) and math.isfinite(x["score"]) for x in h):
    problems.append("non-finite metrics")
if not ckpt.exists():
    problems.append("missing best.pt")
else:
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not str(meta.get("architecture", "")).endswith("_v6"):
        problems.append("wrong architecture")
    if meta.get("stage") == "bootstrap":
        problems.append("bootstrap checkpoint was never replaced")

score = float(row["score"]) if row else -99.0
suicides = float(row["suicides"]) if row else 99.0
if score <= -0.2:
    problems.append("best netKills %.4f <= -0.2" % score)
if suicides >= 0.3:
    problems.append("best suicides %.4f >= 0.3" % suicides)

report = dict(
    accepted=not problems, phase="shared", objective="frag",
    rule="netKills > -0.2 AND suicides < 0.3 AND iterations >= 30",
    iterations=r.get("iterations"), best_iteration=best_iter,
    best_netKills=score, best_suicides=suicides,
    best_fitness=r.get("best_fitness"), problems=problems,
    handoff_to="v6c", fork2_suppressed=True,
    reason="extend shared PPO until a measured win rate appears before forking",
    note="V6B ends here on purpose; V6C resumes from this audited best.pt. "
         "A non-zero exit is used to stop run_v6b.sh before its fork2 stage.")
(out / "shared-gate.json").write_text(json.dumps(report, indent=2))

if problems:
    print("SHARED_GATE_FAILED", json.dumps(problems), flush=True)
else:
    print("SHARED_GATE_OK iters=%s best_iter=%s netKills=%.4f suicides=%.4f" %
          (r.get("iterations"), best_iter, score, suicides), flush=True)
    print("V6B_HANDOFF: fork2 deliberately suppressed; V6C resumes from %s" % ckpt, flush=True)
sys.exit(1)
