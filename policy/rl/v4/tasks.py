"""Disjoint map/layout namespaces; every combat block rotates every controlled seat."""
import numpy as np
from .common import IDS, SKILLS, canonical


def number(*parts):
    return int(canonical(parts)[:8], 16) % (2**30 - 1) + 1


def combat(split, blocks, seed=20261107):
    rows = []
    for n in range(2, 9):
        for b in range(blocks):
            rng = np.random.default_rng(number(seed, split, n, b))
            roster = [IDS[int(rng.integers(len(IDS)))] for _ in range(n)]
            for variant in range(2):
                # Unseen generated maps as well as the two actual Arena maps.
                map_seed = [20260101, 88888888][b % 2] if variant == 0 else number(seed, split, 'map', n, b)
                for seat in range(n):
                    rows.append(dict(seed=number(seed, split, 'layout', n, b, variant), spawnSeed=number(seed, split, 'spawn', n, b, variant),
                                     mapSeed=map_seed, n=n, seat=seat, players=roster, block=f'{split}:{n}:{b}', case=len(rows)))
    return rows


def skills(split, count, seed=20261107):
    return [dict(seed=number(seed, split, skill, i), n=2, seat=0, players=['builtin', 'builtin'], skill=skill,
                 block=f'{split}:{skill}:{i}', case=i) for skill in SKILLS for i in range(count)]


def training(seed, version, count):
    rng = np.random.default_rng(number(seed, 'ppo-train', version))
    return [dict(seed=number(seed, 'ppo-train-layout', version, i), spawnSeed=number(seed, 'ppo-train-spawn', version, i),
                 mapSeed=[20260101, 88888888][i % 2] if i % 3 else number(seed, 'ppo-map', version, i),
                 n=(n := 2 + i % 7), seat=int(rng.integers(n)), players=[IDS[int(rng.integers(len(IDS)))] for _ in range(n)],
                 case=i, block=f'train:{version}:{i}') for i in range(count)]
