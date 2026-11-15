"""Probe 3 (Option 1): U-Net pixels claimed by all low-conf YOLO detections.

Hypothesis: the earlier YU0 fusion barely helped because only the 795 conf.35,
gated, ownership-resolved masks could claim U-Net pixels. Upstream claims with
every detection scoring >= .30, using the detector's own masks.

Pre-registered 2026-10-11, before any number below was computed. Run with
`.venv-unet`. No tuning beyond this grid; every cell is reported.
- Detections: export_author_detections_20261011 (author fold1 YOLO, conf >= .10,
  141 val images neither model trained on); score = YOLO conf.
- Fusion: upstream final setting threshold .5, close 3, grow 4, min_area 200,
  keep_unclaimed 0 (third_party/.../notebooks/reproduce.ipynb FUSION).
- Grid (4 cells): claim mask {yolo: native retina mask, ref: refiner mask
  prob > .55 & disk, before gate/ownership, every detection} x min_score
  {.20, .30}.
- Gates: replaying native gate + ownership at conf .35 from the new export
  must score Y = .3704306509; U must score .3864.
- Baselines: U .3864, UY .3894. Decision: the best cell becomes a test
  candidate only if its PQ >= UY + .005 (selection over 4 cells inflates the
  max); otherwise stop and report. Small-object recall and split/merge are
  reported but not gated.
"""
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fusion_probe_unet_20261010 import (CACHE, LOGITS, U_PARAMS, load_disk_geometry,  # noqa: E402
                                        logits_to_instances, rles_from_labels, score)
from filament_seg.fusion import Detection, FusionParams, fuse  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, save_json  # noqa: E402

DETECTIONS = ROOT/'outputs/author_detections_conf010_20261011'
OUT = ROOT/'outputs/fusion_probe_option1_20261011'
UY_PQ = .38935256170435484
MARGIN = .005
MASKS = {'yolo': 'yolo_counts', 'ref': 'ref_counts'}
MIN_SCORES = (.20, .30)


def params(min_score: float) -> FusionParams:
    return FusionParams(threshold=.5, close_radius=3, grow=4, min_score=min_score, min_area=200,
                        keep_unclaimed=0, unclaimed_gap=24)


def native_ownership(dets: list[dict], conf: float = .35, min_area: int = 96) -> list[dict]:
    """Author conf.35 output rebuilt from saved refiner masks (parity gate)."""
    kept = [d for d in dets if d['conf'] >= conf and d['ref_gate']]
    occ = np.zeros((2048, 2048), bool)
    out = []
    for d in sorted(kept, key=lambda z: -z['conf']):
        m = coco.decode({'size': [2048, 2048], 'counts': d['ref_counts'].encode('ascii')}).astype(bool) & ~occ
        if m.sum() >= min_area:
            occ |= m
            out.append(coco.encode(np.asfortranarray(m.astype(np.uint8))))
    return out


def load_detections(split: str) -> dict[str, list[dict]]:
    return json.loads((DETECTIONS/f'detections_{split}.json').read_text())


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    detections = load_detections('val')
    stems = sorted(detections)
    assert len(stems) == 141 and all((LOGITS/(s+'.npy')).exists() for s in stems)
    store, disks = AnnotationStore(DEFAULT_DATA), load_disk_geometry(CACHE/'disk.json')
    cells = [f'{m}_s{s:.2f}' for m in MASKS for s in MIN_SCORES]
    preds = {name: {} for name in ['Y', 'U', *cells]}
    for i, stem in enumerate(stems):
        dets = detections[stem]
        logits = np.load(LOGITS/(stem+'.npy')).astype(np.float32)
        disk = disks[stem].mask(logits.shape)
        preds['Y'][stem] = native_ownership(dets)
        preds['U'][stem] = rles_from_labels(logits_to_instances(logits, disk, U_PARAMS))
        for mask, key in MASKS.items():
            claim = [Detection(score=d['conf'], counts=d[key]) for d in dets]
            for s in MIN_SCORES:
                preds[f'{mask}_s{s:.2f}'][stem] = rles_from_labels(fuse(logits, disk, claim, params(s)))
        print(f'{i+1}/141 {stem} dets={len(dets)}', flush=True)
    metrics = {name: score(store, p, stems) for name, p in preds.items()}
    assert abs(metrics['Y']['pq'] - .3704306509) < 1e-9, metrics['Y']['pq']
    assert abs(metrics['U']['pq'] - .3864) < 5e-5, metrics['U']['pq']
    best = max(cells, key=lambda c: metrics[c]['pq'])
    decision = {'best_cell': best, 'best_pq': metrics[best]['pq'], 'uy_pq': UY_PQ,
                'delta_vs_uy': metrics[best]['pq'] - UY_PQ,
                'export_test': metrics[best]['pq'] >= UY_PQ + MARGIN}
    OUT.mkdir(parents=True)
    save_json(OUT/'report.json', {'val141': metrics, 'decision': decision,
                                  'fusion': {c: vars(params(s)) for c, s in
                                             ((f'{m}_s{s:.2f}', s) for m in MASKS for s in MIN_SCORES)}})
    for name, m in metrics.items():
        print(f"141 {name:10s} PQ {m['pq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']} "
              f"small {m['small_hit']}/{m['small_gt']} split {m['split_gt']} merge {m['merge_pred']}")
    print('DECISION', json.dumps(decision))


if __name__ == '__main__':
    main()
