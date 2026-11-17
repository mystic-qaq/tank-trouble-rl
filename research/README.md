# Experiment artifacts

This directory contains the **curated evidence** for the research narrative in the main README. It is deliberately small: scripts, metrics, receipts, and logs — not the 80 GB demonstration corpus or the multi-hundred-megabyte rollout dumps.

## Index

| Path | What it is |
|---|---|
| [`results.csv`](results.csv) | Headline numbers on the frozen development split, with their source file |
| [`v5/`](v5/) | Final V5 acceptance reports (`win`, `frag`) |
| [`v6/`](v6/) | The actual V6B/V6C orchestration: shared-phase gate, fork gate, probe, status, run script |
| [`v6/iteration-history.csv`](v6/iteration-history.csv) | All 68 iterations of the shared PPO run used in the README curve |
| [`v6/manifest.json`](v6/manifest.json) | Checkpoint provenance, architecture, dev probe, and the stop decision |
| [`v7/`](v7/) | V7 vs V6 vs random probe logs and the measured random baseline |
| [`tools/make_figures.py`](tools/make_figures.py) | Regenerates every SVG figure in `assets/` with the standard library only |

## Provenance notes

- **Paths were normalised** (`/home/<user>/...` → `$RL_ROOT`, `server3090` → `gpu-host`) so the artifacts do not leak cluster-specific usernames. Numeric fields, hashes, and timestamps were not altered.
- **`source_digest` values in logs refer to the original training snapshot.** This published tree is a documentation snapshot; editing any file under `js/`, `arena/` or `policy/` changes the digest and requires a fresh G0, by design.
- **Model weights are not included** except for the three ONNX files used by the browser arena (`arena/models/`). Training checkpoints (`.pt`) and the 80 GB corpus are excluded by `.gitignore`.
- **The raw Chinese lab logs** remain under `policy/rl/` next to the code that produced them. They are working documents with cluster paths and timestamps; the English notes in `docs/` cross-check their numbers.

## Reproduce the figures

```bash
python research/tools/make_figures.py
```

This writes `assets/training-v6.svg`, `assets/results.svg`, `assets/pipeline.svg` and `assets/architecture-v6.svg`.

## Reproduce the runs

The actual long-run scripts are in [`v6/`](v6/). They assume a Linux GPU host with the release snapshot and the frozen corpus, and are included for auditability rather than as a one-command installer. See [docs/05-reproduction.md](../docs/05-reproduction.md) for the full environment.
