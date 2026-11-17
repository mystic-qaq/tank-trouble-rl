"""V8 corpus collection: Schema V3 records for the hy and gpt seats.

Roster is pinned so that hy and gpt each always hold a seat (order alternates by world,
so neither always spawns in seat 0); remaining seats are drawn from the eight frozen
policies. Every world therefore yields exactly two recorded trajectories, and hy/gpt
face a varied field instead of each other only.

Single-pass: see worker.mjs for why v3's two-pass replay is not needed here.

Usage:
  python -m policy.rl.v8.collect --out /home/xy/rl_for_fun/data-v3/multi --worlds 8000 --workers 96
  python -m policy.rl.v8.collect --out /home/xy/rl_for_fun/data-v3/skills --skills --worlds 1200 --workers 96
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .transport import Worker

IDS = ['deepseek', 'glm', 'gpt', 'hy', 'kimi', 'minimax', 'qwen', 'builtin']
SKILLS = ['static', 'moving', 'bank', 'muzzle', 'dodge', 'volley']
MAPS = [20260101, 88888888]
RECORD_SEATS = ['hy', 'gpt']


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def is_validation(seed):
    return int.from_bytes(hashlib.blake2b(str(seed).encode(), digest_size=8).digest(), 'little') % 10 == 0


def roster(index, n):
    """hy and gpt always seated, exactly once each; the rest drawn from the other six.

    The rest must exclude hy/gpt, otherwise a world records more than two seats and the
    teacher balance stops being 50/50. Seat order alternates so neither teacher always
    spawns in seat 0.
    """
    rng = np.random.default_rng(0x5EED + index)
    pool = [i for i in IDS if i not in RECORD_SEATS]
    rest = [pool[int(rng.integers(len(pool)))] for _ in range(max(0, n - 2))]
    pair = ['hy', 'gpt'] if index % 2 == 0 else ['gpt', 'hy']
    return pair + rest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worlds', type=int, default=8000)
    p.add_argument('--workers', type=int, default=96)
    p.add_argument('--seconds', type=int, default=10800)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--skills', action='store_true')
    p.add_argument('--max-gb', type=float, default=110.0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'trajectories.jsonl').exists():
        raise ValueError('Fresh data directory required: ' + str(a.out))

    workers = [Worker() for _ in range(a.workers)]
    pending = {}
    submitted = done = frames = trajectories = recorded_rows = 0
    written = 0
    by_teacher = {}
    started = time.monotonic()
    stop = False

    def submit(pool, w):
        nonlocal submitted
        index = submitted
        submitted += 1
        seed = a.seed + index
        if a.skills:
            config = dict(seed=seed, n=2, seat=0, mapSeed=MAPS[(index // 7) % 2],
                          skill=SKILLS[index % 6], players=['builtin', 'builtin'])
            op = 'collect_skill'
        else:
            n = 2 + index % 7
            config = dict(seed=seed, n=n, seat=0, mapSeed=MAPS[(index // 7) % 2],
                          players=roster(index, n))
            op = 'collect_world'
        pending[pool.submit(w.request, op=op, config=config)] = (w, index, config)

    counter = 0
    try:
        with open(a.out / 'trajectories.jsonl', 'w') as manifest, \
             open(a.out / 'worlds.jsonl', 'w') as worlds, \
             cf.ThreadPoolExecutor(a.workers) as pool:
            for w in workers:
                if submitted < a.worlds:
                    submit(pool, w)
            while pending:
                ready, _ = cf.wait(pending, timeout=2, return_when=cf.FIRST_COMPLETED)
                for f in ready:
                    w, index, config = pending.pop(f)
                    header, data = f.result()
                    stride = header['stride']
                    if len(data) != header['mapOffset'] + header['mapSize']:
                        raise ValueError('Packed length mismatch')
                    if not np.isfinite(data).all():
                        raise ValueError('Nonfinite data in world %d' % index)
                    matrix = data[:header['mapOffset']].reshape(-1, stride)
                    world_map = data[header['mapOffset']:].copy()
                    map_name = 'world-%06d-map.npy' % index
                    np.save(a.out / map_name, world_map, allow_pickle=False)
                    written += (a.out / map_name).stat().st_size

                    for t in header['trajectories']:
                        length = int(t['length'])
                        if length <= 0:
                            continue
                        name = 'world-%06d-seat-%d.npy' % (index, t['seat'])
                        part = matrix[t['offset']:t['offset'] + length]
                        if len(part) != length:
                            raise ValueError('Trajectory slice out of range')
                        np.save(a.out / name, part, allow_pickle=False)
                        written += (a.out / name).stat().st_size
                        record = dict(t, file=name, map_file=map_name,
                                      seed=config['seed'], n=config['n'],
                                      mapSeed=config['mapSeed'],
                                      validation=is_validation(config['seed']),
                                      skill=config.get('skill', 'combat'),
                                      source='skill' if a.skills else 'multi_teacher',
                                      schema=int(header['schema']), stride=stride,
                                      physics_hz=60)
                        manifest.write(json.dumps(record) + '\n')
                        trajectories += 1
                        recorded_rows += length
                        by_teacher[t['teacher']] = by_teacher.get(t['teacher'], 0) + 1
                    manifest.flush()
                    worlds.write(json.dumps(dict(index=index, players=config['players'],
                                                 **{k: v for k, v in header['summary'].items() if k != 'events'})) + '\n')
                    worlds.flush()

                    done += 1
                    frames += int(header['summary']['frames'])
                    counter += 1
                    if submitted < a.worlds and not stop:
                        submit(pool, w)

                elapsed = time.monotonic() - started
                stop = elapsed > a.seconds or written > a.max_gb * 1024 ** 3
                status = dict(stage='schema_v3_collection',
                              status='running' if done < a.worlds else 'complete',
                              worlds=done, trajectories=trajectories, world_frames=frames,
                              recorded_rows=recorded_rows,
                              bytes=written, gb=round(written / 1024 ** 3, 2),
                              by_teacher=by_teacher, elapsed_seconds=round(elapsed, 1),
                              rows_per_second=round(recorded_rows / max(1, elapsed), 1),
                              frames_per_second=round(frames / max(1, elapsed), 1),
                              worlds_per_minute=round(done / max(1e-9, elapsed) * 60, 2))
                atomic_json(a.out / 'status.json', status)
                if counter % 200 == 0 < counter:
                    print(json.dumps(status), flush=True)

        status.update(status='complete' if done >= a.worlds else 'budget_exhausted')
        atomic_json(a.out / 'status.json', status)
        atomic_json(a.out / 'manifest.json', dict(
            **status, observation_version=3, schema=3, continuous=True,
            split='whole_world_blake2b', seed=a.seed, record_seats=RECORD_SEATS,
            frame_dims=608, label_dims=333, map_dims=346))
        print(json.dumps(status), flush=True)
    finally:
        for w in workers:
            w.close()


if __name__ == '__main__':
    main()
