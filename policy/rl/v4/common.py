"""Small standard-library-only helpers; gate inspection does not require CUDA."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
IDS = ['deepseek', 'glm', 'gpt', 'hy', 'kimi', 'minimax', 'qwen', 'builtin']
STYLES = IDS + ['reference']
TEACHERS = {'win': 'hy', 'frag': 'gpt'}
SKILLS = ['static', 'moving', 'bank', 'muzzle', 'dodge', 'volley']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def source_digest():
    # Includes all imported engine/adapter/data code, not merely the coordinator.
    paths = [p for base in ['js', 'arena', 'policy'] for p in (ROOT / base).rglob('*')
             if p.suffix in {'.py', '.mjs', '.js', '.json'} and '__pycache__' not in p.parts
             and 'models' not in p.parts]
    return canonical({p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)})


class GateFailure(RuntimeError):
    pass


class BudgetExhausted(RuntimeError):
    pass
