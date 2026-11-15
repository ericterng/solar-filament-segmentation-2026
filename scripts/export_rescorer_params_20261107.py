"""Freeze the test-time rescorer of a final pool into weights/rescorer_<pool>.json (CPU, GPU 0).

The two finals apply probe 7's logistic rescorer refit once on ALL 141 fold1 val candidates of their own
U-Net + detector set (export_combo_20261026 for r43, export_amey_combo_20261104 for A6). Their report.json
stores the standardized coefficients but not the StandardScaler mean/scale, so the rescorer could not be
re-applied from the report alone. This script refits it exactly as the export did and dumps everything the
frozen rescorer in solar.final_pipeline needs: scaler mean/scale, coefficients, intercept, E[IoU] by area
bin, PQ_train and the keep threshold PQ_train / 2.

Gates (any failure -> no file): the refit's summary() must equal the report's `rescorer` block (1e-9 on
every number) and the in-sample PQ of the refit over the 141 val images must equal the report's
`val_insample_rescored_pq_not_a_score` (1e-9); the frozen numpy rescorer must keep exactly the same
candidates as the sklearn one. Refuses to overwrite.

Two ways to get the val candidates:
- `--instances outputs/refiner_20261031/instances_val_s0.json` (r43 only): the candidates with their six
  features and RLEs as dump_combo_instances_20261031 saved them (parity to 1e-9 with the combo probe).
- `--logits <dir with <stem>.npy>` (A6, on the A5000 where the val logits live): recompute the candidates
  from cpstrong_s0's dihedral val logits + the pool's flip_views_val.json, as the export scripts do.

    .venv-unet/Scripts/python.exe scripts/export_rescorer_params_20261107.py --pool r43 \
        --instances outputs/refiner_20261031/instances_val_s0.json
    env/bin/python scripts/export_rescorer_params_20261107.py --pool A6 \
        --logits outputs/unet_cpstrong_20261025/cpstrong_s0/logits_val
"""
import pandas  # noqa: F401  # Windows .venv-unet: pandas imported after cv2 + smp crashes (access violation)
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from rescoring_probe_20261012 import AREA_BINS, FEATURES, OPTION_B_MIN, Rescorer, candidates, pq, records_for, totals  # noqa: E402
from fusion_probe_unet_20261010 import CACHE, load_disk_geometry  # noqa: E402
from fusion_probe_multiscale_20261012 import arrays  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, sha256  # noqa: E402
from solar.final_pipeline import DETECTOR_DIRS, POOLS, FrozenRescorer  # noqa: E402

REPORTS = {'r43': ROOT / 'outputs/combo_cpstrong_s0_cp43_submission_20261026/report.json',
           'A6': ROOT / 'outputs/amey_A6_cpstrong_s0_submission_20261104/report.json'}
TOL = 1e-9


def from_instances(path: Path, members: tuple) -> tuple[list[dict], list[str], dict]:
    blob = json.loads(path.read_text(encoding='utf-8'))
    meta = blob['meta']
    assert list(meta['members']) == list(members) and meta['unet'] == 'cpstrong_s0', meta
    inst = []
    for im in blob['images']:
        for d in sorted(im['instances'], key=lambda d: d['order']):
            counts = d['rle'] if isinstance(d['rle'], str) else d['rle']['counts']
            inst.append({'stem': im['stem'], **{k: d[k] for k in (*FEATURES, 'area')},
                         'rle': {'size': [2048, 2048], 'counts': counts.encode('ascii')}})
    stems = sorted(im['stem'] for im in blob['images'])   # all 141, including images with no candidate
    return inst, stems, {'instances_json': str(path.resolve().relative_to(ROOT)).replace('\\', '/'),
                         'instances_sha256': sha256(path), 'candidates': meta['candidates']}


def from_logits(logits_dir: Path, members: tuple) -> tuple[list[dict], list[str], dict]:
    views = {m: json.loads((DETECTOR_DIRS[m] / 'flip_views_val.json').read_text()) for m in members}
    stems = sorted(views[members[0]])
    assert len(stems) == 141 and all(sorted(v) == stems for v in views.values())
    disks = load_disk_geometry(CACHE / 'disk.json')
    inst = []
    for i, stem in enumerate(stems):
        logits = np.load(logits_dir / f'{stem}.npy').astype(np.float32)
        inst += candidates(stem, logits, disks[stem], [views[m][stem] for m in members])
        print(f'val {i + 1}/141 candidates={len(inst)}', flush=True)
    meta = json.loads((logits_dir / 'cache_meta.json').read_text())
    return inst, stems, {'logits_dir': str(logits_dir), 'logits_cache_meta': meta,
                  'views_sha256': {m: sha256(DETECTOR_DIRS[m] / 'flip_views_val.json') for m in members}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', choices=sorted(POOLS), required=True)
    parser.add_argument('--instances', type=Path)
    parser.add_argument('--logits', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    members = POOLS[args.pool]
    out = args.out or ROOT / f'weights/rescorer_{args.pool}.json'
    if out.exists():
        raise FileExistsError(out)
    report = json.loads(REPORTS[args.pool].read_text(encoding='utf-8'))['checks']
    assert report['detectors'] == list(members), (report['detectors'], members)
    if args.instances:
        inst, stems, provenance = from_instances(args.instances, members)
    elif args.logits:
        inst, stems, provenance = from_logits(args.logits, members)
    else:
        parser.error('--instances or --logits is required')
    assert len(stems) == 141 and {d['stem'] for d in inst} <= set(stems), len(stems)
    store, recs = AnnotationStore(DEFAULT_DATA), []
    for stem in stems:   # every val image, also those without a candidate (their GT count FN in PQ_train)
        idx = [i for i, d in enumerate(inst) if d['stem'] == stem]
        assert idx == list(range(idx[0], idx[0] + len(idx))) if idx else True, stem   # contiguous per image
        recs += records_for(store, stem, idx, [inst[i]['rle'] for i in idx])
    X, area, scores = arrays(inst)
    fused = pq(totals(scores >= OPTION_B_MIN, recs))
    assert abs(fused - report['val_fused_s030_pq']) < TOL, (fused, report['val_fused_s030_pq'])
    rescorer = Rescorer(X, area, scores, recs)
    summary = rescorer.summary()
    saved = report['rescorer']
    for key in ('pq_train', 'threshold', 'intercept', 'pairs', 'pair_hit_rate'):
        assert abs(summary[key] - saved[key]) < TOL, (key, summary[key], saved[key])
    assert np.allclose(summary['e_iou_by_bin'], saved['e_iou_by_bin'], atol=TOL, rtol=0)
    for f in FEATURES:
        assert abs(summary['coef_standardized'][f] - saved['coef_standardized'][f]) < TOL, \
            (f, summary['coef_standardized'][f], saved['coef_standardized'][f])
    keep = rescorer.keep(X, area)
    insample = pq(totals(keep, recs))
    assert abs(insample - report['val_insample_rescored_pq_not_a_score']) < TOL, \
        (insample, report['val_insample_rescored_pq_not_a_score'])
    params = {'pool': args.pool, 'detectors': list(members), 'unet': 'cpstrong_s0', 'features': list(FEATURES),
              'scaler_mean': rescorer.scaler.mean_.tolist(), 'scaler_scale': rescorer.scaler.scale_.tolist(),
              'coef': rescorer.model.coef_[0].tolist(), 'intercept': float(rescorer.model.intercept_[0]),
              'e_iou_by_bin': rescorer.e_iou.tolist(), 'area_bins': list(AREA_BINS),
              'pq_train': float(rescorer.pq_train), 'threshold': float(rescorer.pq_train / 2),
              'candidate_min_score': .15,
              'rule': 'keep iff sigmoid(coef . (x - mean) / scale + intercept) * e_iou[bin(area)] > threshold',
              'fit': {'model': 'sklearn LogisticRegression(C=1, lbfgs, max_iter=1000) on StandardScaler features, '
                               'one row per (candidate, annotator record), target = hit at IoU > .5',
                      'pairs': summary['pairs'], 'pair_hit_rate': summary['pair_hit_rate'], 'val_images': 141,
                      'val_fused_s030_pq': fused, 'val_insample_rescored_pq_not_a_score': insample,
                      'report': str(REPORTS[args.pool].relative_to(ROOT)).replace('\\', '/'),
                      'report_csv_sha256': report['csv_sha256']},
              'provenance': provenance}
    frozen = FrozenRescorer(params)
    assert np.array_equal(frozen.keep(X, area), keep), 'frozen rescorer disagrees with sklearn'
    assert np.abs(frozen.expected(X, area) - rescorer.expected(X, area)).max() < 1e-12
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(params, indent=1), encoding='utf-8')
    print(json.dumps({'out': str(out), 'candidates': len(inst), 'kept_insample': int(keep.sum()),
                      'pq_train': params['pq_train'], 'threshold': params['threshold'], 'insample_pq': insample,
                      'gates': 'summary 1e-9, in-sample PQ 1e-9, frozen == sklearn'}))


if __name__ == '__main__':
    main()
