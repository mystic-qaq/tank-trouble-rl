"""Verified server-side relocation. No source is removed before destination SHA checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

SOURCE = Path('/home/xy/rl_for_fun/runs')
DESTINATION = Path('/data/xy/rl_for_fun/runs')
OPERATIONS = Path('/data/xy/rl_for_fun/operations')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(dict(source=str(SOURCE), destination=str(DESTINATION), action='copy, verify every destination hash, symlink, remove verified retired copy')))
        return
    if SOURCE.is_symlink() or not SOURCE.is_dir() or SOURCE.resolve() != SOURCE or DESTINATION.exists():
        raise RuntimeError('Unexpected source/destination; refusing overwrite or repeated migration')
    OPERATIONS.mkdir(parents=True, exist_ok=True)
    journal = OPERATIONS / 'runs-migration.json'
    if journal.exists():
        raise RuntimeError('Migration journal already exists; inspect before retry')
    before = shutil.disk_usage(SOURCE).free
    known_path = SOURCE / 'v4-delivery-check-20261107/g0/data-acceptance.json'
    known = json.loads(known_path.read_text())['files'] if known_path.exists() else {}
    files = {}
    for path in SOURCE.rglob('*'):
        if path.is_symlink():
            raise RuntimeError('Unexpected nested link: ' + str(path))
        if path.is_file():
            st = path.stat()
            files[str(path.relative_to(SOURCE))] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    total = sum(v['size'] for v in files.values())
    if shutil.disk_usage(DESTINATION.parent.parent).free < total + (200 << 30):
        raise RuntimeError('Insufficient destination headroom')
    state = dict(status='copying', started_unix=time.time(), source=str(SOURCE), destination=str(DESTINATION), files=len(files), bytes=total)
    write(journal, state)
    DESTINATION.mkdir(parents=True, exist_ok=False)
    subprocess.run(['rsync', '-a', str(SOURCE) + '/', str(DESTINATION) + '/'], check=True)
    destination_files = {str(p.relative_to(DESTINATION)) for p in DESTINATION.rglob('*') if p.is_file()}
    if destination_files != set(files):
        raise RuntimeError('File inventory mismatch; original preserved')
    state.update(status='verifying_sha256', verified_files=0, verified_bytes=0)
    write(journal, state)
    for i, (relative, metadata) in enumerate(files.items(), 1):
        src, dst = SOURCE / relative, DESTINATION / relative
        old = known.get(str(src))
        expected = old['sha256'] if old and old['size'] == metadata['size'] and old['mtime_ns'] == metadata['mtime_ns'] else sha(src)
        if dst.stat().st_size != metadata['size'] or sha(dst) != expected:
            raise RuntimeError('Content mismatch; original preserved: ' + relative)
        metadata['sha256'] = expected
        state['verified_files'] = i
        state['verified_bytes'] += metadata['size']
        if i % 128 == 0:
            write(journal, state)
    current = {str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()}
    if current != set(files):
        raise RuntimeError('Source inventory changed; original preserved')
    for relative, metadata in files.items():
        st = (SOURCE / relative).stat()
        if st.st_size != metadata['size'] or st.st_mtime_ns != metadata['mtime_ns']:
            raise RuntimeError('Source changed during copy; original preserved: ' + relative)
    write(OPERATIONS / 'runs-migration-files.json', files)
    retired = SOURCE.parent / ('.runs-migration-retired-' + str(os.getpid()))
    if retired.exists():
        raise RuntimeError('Retired path already exists')
    SOURCE.rename(retired)
    try:
        SOURCE.symlink_to(DESTINATION, target_is_directory=True)
    except BaseException:
        retired.rename(SOURCE)
        raise
    if SOURCE.resolve() != DESTINATION or retired.is_symlink() or retired.resolve() != SOURCE.parent / retired.name:
        raise RuntimeError('Unexpected resolved paths; retired copy preserved')
    state.update(status='verified_link_installed', retired=str(retired))
    write(journal, state)
    # Only this exact, verified former runs directory is removed, after link validation.
    shutil.rmtree(retired)
    state.update(status='complete', completed_unix=time.time(), home_free_before=before,
                 home_free_after=shutil.disk_usage(SOURCE.parent).free, original_path_is_symlink=SOURCE.is_symlink())
    write(journal, state)
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
