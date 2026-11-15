"""Probe 5 (Option B): three same-split YOLO detectors claiming U-Net pixels.

Pre-registered 2026-10-11, while the extra detectors were still training (no
m1280 / m1024 output existed). Run with `.venv-unet`.
- Members: det4oof_f1 (yolo11m 1536, outputs/author_flip_views_20261011) and
  run_extra_detectors_20261011 m1280 / m1024 (same fold1 split, size/seed
  only), each with four flip views at its own imgsz.
- Per member: as-is view, or its four flips merged (merge_views IoU .5, mean
  conf over 4). Across members: merge_views IoU .5 again, score = mean over
  the 3 members (a member that missed it counts 0); as upstream
  merge_detections over yolo_predict --tta flips outputs.
- Fusion: unchanged Option 1 setting (threshold .5, close 3, grow 4,
  min_area 200, keep_unclaimed 0), YOLO native masks.
- Grid (4 cells): {ens_asis, ens_tta} x min_score {.20, .30}. Single members
  (as-is, s.30) are reported for information only.
- Gate: det4 as-is at s.30 must reproduce Option 1 .4250201.
- Baseline: Option 1 .4250 (Public .35); flip-TTA .4283 is reported alongside.
  Decision: the best cell becomes a test candidate only if PQ >= .4250 + .005;
  otherwise stop and report.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fusion_probe_unet_20261010 import CACHE, LOGITS, load_disk_geometry, rles_from_labels, score  # noqa: E402
from fusion_probe_option1_20261011 import Detection, fuse, params  # noqa: E402
from filament_seg.fusion import merge_views  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, save_json  # noqa: E402

MEMBERS = {'m1536': ROOT/'outputs/author_flip_views_20261011',
           'm1280': ROOT/'outputs/extra_m1280_20261011',
           'm1024': ROOT/'outputs/extra_m1024_20261011'}
OUT = ROOT/'outputs/fusion_probe_ensemble_20261011'
OPTION1_PQ = .42502008952191084
TTA_PQ = .4282535567639003
MARGIN = .005
MIN_SCORES = (.20, .30)


def as_is(views: list[list[dict]]) -> list[Detection]:
    return [Detection(d['conf'], d['counts']) for d in views[0]]


def flips(views: list[list[dict]]) -> list[Detection]:
    return merge_views([[Detection(d['conf'], d['counts']) for d in v] for v in views], .5)


def ensemble(per_member: list[list[Detection]]) -> list[Detection]:
    return merge_views(per_member, .5)


def load(split: str) -> dict[str, dict[str, list]]:
    return {m: json.loads((d/f'flip_views_{split}.json').read_text()) for m, d in MEMBERS.items()}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    views = load('val')
    stems = sorted(views['m1536'])
    assert len(stems) == 141 and all(sorted(v) == stems and all(len(v[s]) == 4 for s in stems) for v in views.values())
    store, disks = AnnotationStore(DEFAULT_DATA), load_disk_geometry(CACHE/'disk.json')
    cells = [f'ens_{k}_s{s:.2f}' for k in ('asis', 'tta') for s in MIN_SCORES]
    singles = [f'{m}_asis_s0.30' for m in MEMBERS]
    preds = {c: {} for c in singles + cells}
    for i, stem in enumerate(stems):
        logits = np.load(LOGITS/(stem+'.npy')).astype(np.float32)
        disk = disks[stem].mask(logits.shape)
        members = {k: [f(views[m][stem]) for m in MEMBERS] for k, f in (('asis', as_is), ('tta', flips))}
        for m, dets in zip(MEMBERS, members['asis']):
            preds[f'{m}_asis_s0.30'][stem] = rles_from_labels(fuse(logits, disk, dets, params(.30)))
        for k, per in members.items():
            merged = ensemble(per)
            for s in MIN_SCORES:
                preds[f'ens_{k}_s{s:.2f}'][stem] = rles_from_labels(fuse(logits, disk, merged, params(s)))
        print(f'{i+1}/141 {stem}', flush=True)
    metrics = {c: score(store, p, stems) for c, p in preds.items()}
    assert abs(metrics['m1536_asis_s0.30']['pq'] - OPTION1_PQ) < 1e-9, metrics['m1536_asis_s0.30']['pq']
    best = max(cells, key=lambda c: metrics[c]['pq'])
    decision = {'best_cell': best, 'best_pq': metrics[best]['pq'], 'option1_pq': OPTION1_PQ,
                'delta_vs_option1': metrics[best]['pq'] - OPTION1_PQ, 'delta_vs_flip_tta': metrics[best]['pq'] - TTA_PQ,
                'export_test': metrics[best]['pq'] >= OPTION1_PQ + MARGIN}
    OUT.mkdir(parents=True)
    save_json(OUT/'report.json', {'val141': metrics, 'decision': decision})
    for name, m in metrics.items():
        print(f"141 {name:16s} PQ {m['pq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']} "
              f"small {m['small_hit']}/{m['small_gt']} split {m['split_gt']} merge {m['merge_pred']}")
    print('DECISION', json.dumps(decision))


if __name__ == '__main__':
    main()
