"""Probe: does the ameyypawar-recipe U-Net improve YOLO fold1 (and YOLO+DINO)?

Run with `.venv-unet`. Needs U-Net logits cached by upstream scripts/cache_logits.py
(dihedral TTA, last epoch) for the 141 author fold1 val observations, which
neither YOLO nor this U-Net trained on.

Pre-registered before any U-Net score was seen (2026-10-10). Fixed settings,
no tuning on these images:
- Y: author YOLO11m+refiner conf.35 exclusive masks (CSV order = descending conf).
- U: U-Net alone, upstream baseline postprocess threshold .6, min_area 400,
  bridge_gap 24, close_radius 3, open_radius 0.
- YU0: upstream final fusion (threshold .5, close 3, grow 4, min_area 200,
  keep_unclaimed 0); Y masks claim U-Net pixels in Y's confidence order.
- YU400: same, but unclaimed U-Net groups >= 400 px (gap 24) are kept as
  extra instances (upstream FusionParams docstring setting).
On 141 val the best of {Y, YU0, YU400} becomes the base. On the 27 images also
clean for DINO, the s.9/a1000 DINO additions (fusion_probe_yolo_dino_20261010)
are re-applied on top of that base and reported.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/ameyypawar-filament-seg'))
sys.path.insert(0, str(ROOT/'src'))
from filament_seg.dataset import load_disk_geometry  # noqa: E402
from filament_seg.fusion import Detection, FusionParams, fuse  # noqa: E402
from filament_seg.postprocess import PostprocessParams, logits_to_instances  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, save_json  # noqa: E402
from solar.evaluation import aggregate, score_instances  # noqa: E402

RUN = ROOT/'outputs/amey_unet_fold1_20261010'
LOGITS = RUN/'logits_val_last_dihedral'
CACHE = ROOT/'outputs/amey_unet_cache'
YOLO_CSV = ROOT/'outputs/author_fold1_submission_20261010/original_dev_conf035.csv'
DINO_RAW = ROOT/'outputs/fusion_probe_yolo_dino_20261010/dino_raw'
U_PARAMS = PostprocessParams(threshold=.6, min_area=400, bridge_gap=24, close_radius=3, open_radius=0)
FUSIONS = {'YU0': FusionParams(threshold=.5, close_radius=3, grow=4, min_score=0., min_area=200, keep_unclaimed=0),
           'YU400': FusionParams(threshold=.5, close_radius=3, grow=4, min_score=0., min_area=200, keep_unclaimed=400)}


def rles_from_labels(labels: np.ndarray) -> list[dict]:
    return [coco.encode(np.asfortranarray((labels == v).astype(np.uint8))) for v in np.unique(labels) if v]


def yolo_masks() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with YOLO_CSV.open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row['filament_id'].rsplit('_', 1)[0], []).append(
                {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')})
    return out


def score(store, preds, stems) -> dict:
    rows = [score_instances(gt, preds.get(s, [])) for s in stems for _, gt in store.ground_truth(s+'.jpeg')]
    return aggregate(rows)


def main() -> None:
    stems = sorted(json.loads((ROOT/'outputs/author_fold1_submission_20261010/dev_replay/raw_boxes.json').read_text()))
    assert len(stems) == 141 and all((LOGITS/(s+'.npy')).exists() for s in stems)
    clean27 = json.loads((ROOT/'outputs/fusion_probe_yolo_dino_20261010/inference.json').read_text())['images']
    store, disks, yolo = AnnotationStore(DEFAULT_DATA), load_disk_geometry(CACHE/'disk.json'), yolo_masks()
    preds = {'Y': yolo, 'U': {}, 'YU0': {}, 'YU400': {}}
    for i, stem in enumerate(stems):
        logits = np.load(LOGITS/(stem+'.npy')).astype(np.float32)
        disk = disks[stem].mask(logits.shape)
        preds['U'][stem] = rles_from_labels(logits_to_instances(logits, disk, U_PARAMS))
        # Y is in descending-confidence order; ranks stand in for scores (min_score is 0).
        dets = [Detection(score=float(len(yolo.get(stem, [])) - k), counts=m['counts'].decode('ascii'))
                for k, m in enumerate(yolo.get(stem, []))]
        for name, params in FUSIONS.items():
            preds[name][stem] = rles_from_labels(fuse(logits, disk, dets, params))
        print(f'{i+1}/141 {stem}', flush=True)
    metrics = {name: score(store, p, stems) for name, p in preds.items()}
    assert abs(metrics['Y']['pq'] - .3704306509) < 1e-6, metrics['Y']['pq']  # replay of formal YOLO conf.35
    base = max(['Y', 'YU0', 'YU400'], key=lambda n: metrics[n]['pq'])

    from fusion_probe_yolo_dino_20261010 import fuse as dino_fuse, to_rle  # noqa: E402
    dino = {s: [(to_rle(r), r['score']) for r in json.loads((DINO_RAW/(s+'.json')).read_text())] for s in clean27}
    on27 = {name: score(store, preds[name], clean27) for name in ['Y', 'U', 'YU0', 'YU400']}
    on27['Y+D'] = score(store, {s: dino_fuse(yolo.get(s, []), dino[s], .9, 1000) for s in clean27}, clean27)
    on27[base+'+D'] = score(store, {s: dino_fuse(preds[base][s], dino[s], .9, 1000) for s in clean27}, clean27)
    report = {'val141': metrics, 'base': base, 'clean27': on27,
              'params': {'U': vars(U_PARAMS), **{k: vars(v) for k, v in FUSIONS.items()}}}
    save_json(RUN/'fusion_probe_report.json', report)
    for name, m in metrics.items():
        print(f"141 {name:6s} PQ {m['pq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']} small {m['small_hit']}/{m['small_gt']}")
    for name, m in on27.items():
        print(f"27  {name:9s} PQ {m['pq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']}")
    print('BASE', base)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT/'scripts'))
    main()
