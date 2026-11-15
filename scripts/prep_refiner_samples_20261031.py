"""Probe 28, local step 0: training samples per (candidate, record) and the oracle gate (CPU, no GPU).

From instances_val_s<seed>.json (every .15 candidate with its keep flag and date half): for each
candidate and each annotation record of its image, the record's best-IoU GT; IoU >= .25 makes one
sample (target = that GT mask). Writes samples_s<seed>.pkl + stats (per-half counts, share of IoU >= .5,
target / candidate area-ratio quartiles) and three sample overlays. Any half under 300 samples voids the
protocol (exit 1 after writing the stats).
Oracle (step 0, kept instances only, keep decisions frozen, §B 3-5 conflict rule, per_stem scoring):
near_miss (swap kept instances whose cross-record best IoU is in [.25, .5) for that GT) is the GATE --
mean delta over both U-Net seeds < .005 voids the protocol; full (swap every kept instance with best
IoU >= .25) and per_record (each record scored against its own best-GT swaps) are information.
identity must reproduce the saved C_s<seed>_d43 MAIN cell to 1e-9. Refuses to overwrite.

    .venv-unet/Scripts/python.exe scripts/prep_refiner_samples_20261031.py --unet-seed 0   # then 1, then --gate
"""
import pandas  # noqa: F401  # Windows .venv-unet: pandas imported after cv2 + smp crashes (access violation)
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fusion_probe_unet_crossrecipe_20261020 import KEYS, MAIN, per_stem, total  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, save_json, sha256  # noqa: E402
from solar.evaluation import aggregate, score_instances  # noqa: E402
import refiner_core_20261031 as core  # noqa: E402

OUT = ROOT / 'outputs/refiner_20261031'
CACHE = ROOT / 'outputs/amey_unet_cache'
SAMPLE_FLOOR = 300
OVERLAYS = 3
RNG_SEED = 20261031


def record_targets(store: AnnotationStore, image: dict) -> tuple[list, list[list[tuple[int, float]]]]:
    records = store.ground_truth(image['stem'] + '.jpeg')
    per = core.best_gt_per_record([i['rle'] for i in image['instances']], records)
    return records, per


def build_samples(blob: dict, store: AnnotationStore) -> tuple[list[dict], dict]:
    samples, seed = [], blob['meta']['unet_seed']
    for image in blob['images']:
        records, per = record_targets(store, image)
        for inst, per_rec in zip(image['instances'], per):
            cand_area = int(inst['area'])
            for (rec_id, gt), (idx, iou) in zip(records, per_rec):
                if iou >= core.IOU_MIN:
                    samples.append({'stem': image['stem'], 'half': image['half'], 'unet_seed': seed, 'order': inst['order'],
                                    'kept': inst['kept'], 'record': rec_id, 'gt_index': idx, 'iou': iou,
                                    'cand_rle': inst['rle'], 'gt_rle': core.rle_json(gt[idx]),
                                    'cand_area': cand_area, 'gt_area': int(core.decode(gt[idx]).sum())})
    stats = {}
    for h in 'AB':
        sel = [s for s in samples if s['half'] == h]
        ratio = np.asarray([s['gt_area'] / s['cand_area'] for s in sel]) if sel else np.zeros(0)
        stats[h] = {'samples': len(sel), 'images': len({s['stem'] for s in sel}),
                    'share_iou_ge_05': float(np.mean([s['iou'] >= .5 for s in sel])) if sel else None,
                    'share_kept': float(np.mean([s['kept'] for s in sel])) if sel else None,
                    'area_ratio_quartiles': [float(q) for q in np.percentile(ratio, [25, 50, 75])] if sel else None,
                    'iou_quartiles': [float(q) for q in np.percentile([s['iou'] for s in sel], [25, 50, 75])] if sel else None}
    stats['floor'] = SAMPLE_FLOOR
    stats['floor_pass'] = all(stats[h]['samples'] >= SAMPLE_FLOOR for h in 'AB')
    halves = blob['meta']['halves']
    assert not set(halves['A']) & set(halves['B']) and len(halves['A']) + len(halves['B']) == 141
    assert all(s['stem'] in set(halves[s['half']]) for s in samples)
    return samples, stats


def overlays(samples: list[dict], seed: int, out: Path = OUT) -> list[str]:
    rng = np.random.default_rng(RNG_SEED + seed)
    picks = rng.choice(len(samples), size=min(OVERLAYS, len(samples)), replace=False)
    paths = []
    for n, i in enumerate(picks):
        s = samples[int(i)]
        flat = cv2.imread(str(CACHE / 'flat' / f"{s['stem']}.png"), cv2.IMREAD_GRAYSCALE)
        cand, gt = core.decode(s['cand_rle']), core.decode(s['gt_rle'])
        y0, x0 = core.sample_crop_origin(core.bbox_of(cand), cand.shape, np.random.default_rng(0), jitter=0)
        t = core.TILE
        panel = cv2.cvtColor(flat[y0:y0 + t, x0:x0 + t], cv2.COLOR_GRAY2BGR)
        for mask, color in ((gt[y0:y0 + t, x0:x0 + t], (0, 255, 0)), (cand[y0:y0 + t, x0:x0 + t], (0, 0, 255))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, color, 1)
        cv2.putText(panel, f"{s['stem']} iou {s['iou']:.2f} green=GT red=candidate", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        path = out / 'sample_overlays' / f's{seed}_{n}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), panel)
        paths.append(str(path))
    return paths


def kept_with_targets(image: dict, records: list, per: list) -> tuple[list[dict], list[np.ndarray], list[int], list]:
    kept_idx = [i for i, inst in enumerate(image['instances']) if inst['kept']]
    kept = [image['instances'][i] for i in kept_idx]
    originals = [core.decode(inst['rle']) for inst in kept]
    order = sorted(range(len(kept)), key=lambda i: kept[i]['order'])
    return kept, originals, order, [per[i] for i in kept_idx]


def swapped(originals: list[np.ndarray], swaps: list[np.ndarray | None], order: list[int]) -> list[np.ndarray]:
    """§B 3-5 with the oracle target in place of the network output (region = whole image)."""
    if not originals:
        return []
    any_original = np.zeros(originals[0].shape, bool)
    for o in originals:
        any_original |= o
    refined = [o.copy() if t is None else (t & ~(any_original & ~o)) for o, t in zip(originals, swaps)]
    finals, _ = core.resolve(originals, refined, order)
    return finals


def oracle(blob: dict, store: AnnotationStore, control: dict) -> dict:
    stems = [im['stem'] for im in blob['images']]
    preds = {arm: {} for arm in ('identity', 'near_miss', 'full')}
    per_record_rows, swapped_counts = [], {'near_miss': 0, 'full': 0}
    for image in blob['images']:
        records, per = record_targets(store, image)
        kept, originals, order, per_kept = kept_with_targets(image, records, per)
        best = [core.best_across_records(p) for p in per_kept]
        for arm in ('identity', 'near_miss', 'full'):
            swaps = []
            for (j, idx, iou), o in zip(best, originals):
                take = arm != 'identity' and core.oracle_replace(iou, arm)
                swaps.append(core.decode(records[j][1][idx]) if take else None)
                swapped_counts[arm] = swapped_counts.get(arm, 0) + int(take)
            preds[arm][image['stem']] = [core.encode(m) for m in swapped(originals, swaps, order)]
        for j, (rec_id, gt) in enumerate(records):
            swaps = [core.decode(gt[p[j][0]]) if p[j][1] >= core.IOU_MIN else None for p in per_kept]
            per_record_rows.append(score_instances(gt, [core.encode(m) for m in swapped(originals, swaps, order)]))
    cells = {arm: total(per_stem(store, preds[arm], stems), stems) for arm in preds}
    for k in (*KEYS, 'pq'):
        assert abs(cells['identity'][k] - control[k]) < 1e-9, ('identity parity', k, cells['identity'][k], control[k])
    per_record = aggregate(per_record_rows)
    return {'identity_parity': 'MAIN cell reproduced to 1e-9 on every key',
            'cells': {arm: {k: cells[arm][k] for k in (*KEYS, 'pq')} for arm in cells},
            'per_record': {k: per_record[k] for k in (*KEYS, 'pq')},
            'delta': {arm: cells[arm]['pq'] - control['pq'] for arm in ('near_miss', 'full')},
            'per_record_delta': per_record['pq'] - control['pq'],
            'swapped': {k: v for k, v in swapped_counts.items() if k != 'identity'}}


def run_seed(seed: int) -> None:
    out_pkl, out_oracle = OUT / f'samples_s{seed}.pkl', OUT / f'oracle_s{seed}.json'
    for p in (out_pkl, out_oracle):
        if p.exists():
            raise FileExistsError(p)
    src = OUT / f'instances_val_s{seed}.json'
    blob = json.loads(src.read_text(encoding='utf-8'))
    assert blob['meta']['unet_seed'] == seed
    store = AnnotationStore(DEFAULT_DATA)
    samples, stats = build_samples(blob, store)
    meta = {'probe': 28, 'unet_seed': seed, 'halves': blob['meta']['halves'], 'iou_min': core.IOU_MIN,
            'floor': SAMPLE_FLOOR, 'floor_pass': stats['floor_pass'], 'source_instances_sha256': sha256(src),
            'control_pq': blob['meta']['control_pq'], 'stats': stats}
    out_pkl.write_bytes(pickle.dumps({'meta': meta, 'samples': samples}))
    save_json(OUT / f'samples_s{seed}_stats.json', {**meta, 'overlays': overlays(samples, seed), 'samples_sha256': sha256(out_pkl)})
    print('SAMPLES', json.dumps(stats))
    control = json.loads(Path(blob['meta']['control_json']).read_text())[MAIN]
    report = oracle(blob, store, control)
    save_json(out_oracle, {'unet_seed': seed, 'control_pq': control['pq'], **report})
    print('ORACLE', json.dumps({'delta': report['delta'], 'per_record_delta': report['per_record_delta'], 'swapped': report['swapped']}))
    if not stats['floor_pass']:
        raise SystemExit(f'sample floor {SAMPLE_FLOOR} not met: {stats}')


def gate() -> None:
    out = OUT / 'oracle.json'
    if out.exists():
        raise FileExistsError(out)
    reports = {s: json.loads((OUT / f'oracle_s{s}.json').read_text()) for s in (0, 1)}
    g = core.oracle_gate({s: r['delta']['near_miss'] for s, r in reports.items()})
    save_json(out, {**g, 'full_delta': {s: r['delta']['full'] for s, r in reports.items()},
                    'per_record_delta': {s: r['per_record_delta'] for s, r in reports.items()},
                    'swapped': {s: r['swapped'] for s, r in reports.items()},
                    'rule': 'mean near_miss delta over both U-Net seeds >= .005, else the protocol is void (no GPU)'})
    print('GATE', json.dumps(g))
    if not g['gate_pass']:
        raise SystemExit('oracle gate failed: protocol void, do not train')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unet-seed', type=int, choices=(0, 1))
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    if args.gate:
        gate()
    elif args.unet_seed is not None:
        run_seed(args.unet_seed)
    else:
        parser.error('--unet-seed or --gate')


if __name__ == '__main__':
    main()
