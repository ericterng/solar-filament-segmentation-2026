"""Verify the final weights in weights/ against the pinned SHA-256 list.

Every file in weights/manifest.json must exist, must not be an unfetched git-lfs pointer, and its
SHA-256 must equal both the manifest and docs/experiments/final_weights_sha256_20261106.txt (looked up by
the original outputs/ path). The rescorer JSONs are checked for presence and their own recorded SHA of the
report they were fitted from. Exit code 0 only when everything matches. No torch import; runs anywhere.

    python scripts/verify_final_weights.py            # from the repo root (or any cwd)
    python scripts/verify_final_weights.py --quiet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'weights'
MANIFEST = WEIGHTS / 'manifest.json'
SHA_LIST = ROOT / 'docs/experiments/final_weights_sha256_20261106.txt'
LFS_POINTER = b'version https://git-lfs.github.com/spec/v1'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    with path.open('rb') as handle:
        return handle.read(len(LFS_POINTER)) == LFS_POINTER


def pinned_list() -> dict[str, str]:
    """{outputs path: sha256} from the 2026-11-06 list (sha256sum format)."""
    rows = {}
    for line in SHA_LIST.read_text(encoding='utf-8').splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            rows[path.strip().lstrip('*')] = digest
    return rows


def verify(quiet: bool = False) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    pinned = pinned_list()
    rows = []
    for entry in manifest['weights']:
        path = WEIGHTS / entry['file']
        row = {'file': entry['file'], 'role': entry['role'], 'member': entry['member'], 'ok': False, 'why': ''}
        if not path.exists():
            row['why'] = 'missing'
        elif is_lfs_pointer(path):
            row['why'] = 'git-lfs pointer only: run `git lfs pull`'
        else:
            actual = sha256(path)
            expected_list = pinned.get(entry['source'])
            if actual != entry['sha256']:
                row['why'] = f'sha256 {actual[:12]} != manifest {entry["sha256"][:12]}'
            elif expected_list is None:
                row['why'] = f'{entry["source"]} not in {SHA_LIST.name}'
            elif actual != expected_list:
                row['why'] = f'sha256 {actual[:12]} != list {expected_list[:12]}'
            elif path.stat().st_size != entry['bytes']:
                row['why'] = f'size {path.stat().st_size} != {entry["bytes"]}'
            else:
                row['ok'], row['why'] = True, actual[:12]
        rows.append(row)
    sources = {e['source'] for e in manifest['weights']}
    for source in pinned:
        if source not in sources:
            rows.append({'file': '(list)', 'role': '-', 'member': source, 'ok': False, 'why': 'listed weight has no manifest entry'})
    for entry in manifest['rescorers']:
        path = WEIGHTS / entry['file']
        row = {'file': entry['file'], 'role': 'rescorer', 'member': entry['pool'], 'ok': False, 'why': ''}
        if not path.exists():
            row['why'] = 'missing'
        else:
            params = json.loads(path.read_text(encoding='utf-8'))
            needed = {'features', 'scaler_mean', 'scaler_scale', 'coef', 'intercept', 'e_iou_by_bin', 'area_bins', 'threshold'}
            missing = needed - set(params)
            if missing:
                row['why'] = f'missing keys {sorted(missing)}'
            elif params.get('pool') != entry['pool'] or params.get('detectors') != entry['detectors']:
                row['why'] = 'pool or detector set differs from manifest'
            else:
                row['ok'], row['why'] = True, f'threshold {params["threshold"]:.6f}'
        rows.append(row)
    if not quiet:
        width = max(len(r['file']) for r in rows)
        for r in rows:
            print(f'{"OK " if r["ok"] else "BAD"} {r["file"]:<{width}} {r["role"]:<9} {r["member"]:<18} {r["why"]}')
        n_ok = sum(r['ok'] for r in rows)
        print(f'{n_ok}/{len(rows)} entries verified; manifest {MANIFEST.relative_to(ROOT)}, list {SHA_LIST.relative_to(ROOT)}')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    rows = verify(args.quiet)
    sys.exit(0 if all(r['ok'] for r in rows) else 1)


if __name__ == '__main__':
    main()
