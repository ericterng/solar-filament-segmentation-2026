"""Train an extra author-recipe YOLO on the same fold1 split, then save flip views.

Option B (2026-10-11): upstream ameyypawar averaged several detectors trained
on one split (yolo11s 1024, yolo11m 1280, yolo11m 1536). Other folds' detectors
trained on our 141 val images, so they cannot be validated; extra members here
use the identical fold1 566 train / 141 val split and differ from the existing
det4oof_f1 (yolo11m-seg, 1536, seed 43) only in input size and seed:
- m1280: yolo11m-seg, imgsz 1280, seed 44
- m1024: yolo11m-seg, imgsz 1024, seed 45
Everything else is the pinned HeShen source through its detector stage (same
labels, augmentation, 50 epochs, patience 15, batch 2, native best.pt), via
run_author_fold.portable_source. No refiner. Afterwards the native best.pt
predicts 141 val + 180 test on four flipped copies at the member's own imgsz
(conf .10, iou .60, agnostic NMS, retina masks), as export_author_flip_views.
GPU1 only (UUID + PID checks per batch / image). Refuses to overwrite.

    python scripts/run_extra_detectors_20261011.py {m1280,m1024} {smoke,train}
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_author_fold import GPU_UUID, SOURCE, SOURCE_SHA, portable_source, replace_once
from export_author_detections_20261011 import rle

MEMBERS = {'m1280': {'imgsz': 1280, 'seed': 44}, 'm1024': {'imgsz': 1024, 'seed': 45}}
PRETRAINED_SHA = 'eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf'
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member_source(work: Path, name: str, imgsz: int, seed: int, smoke: bool) -> str:
    """Author source through the detector stage, with only size/seed/name changed."""
    source = portable_source(work)
    source = source[:source.index('print("[det] done", flush=True)')] + 'print("[det] done", flush=True)\n'
    source = replace_once(source, 'imgsz=1536, batch=2', f'imgsz={imgsz}, batch=2')
    source = replace_once(source, 'seed=42 + FOLD', f'seed={seed}')
    assert source.count('f"det4oof_f{FOLD}"') == 3
    source = source.replace('f"det4oof_f{FOLD}"', repr(name))
    source = replace_once(source, 'f"det4oof_f{FOLD}_best.pt"', repr(f'{name}_best.pt'))
    if smoke:
        source = replace_once(source, 'for split in ("train", "val"):',
                              'train_stems = train_stems[:8]\nval_stems = val_stems[:2]\nfor split in ("train", "val"):')
        source = replace_once(source, 'epochs=50,', 'epochs=1,')
    return source


def detect(model, source, flip_lr: bool, flip_ud: bool, imgsz: int, torch) -> list[dict]:
    r = model.predict(source, conf=.10, iou=.60, agnostic_nms=True, retina_masks=True, imgsz=imgsz,
                      device=torch.device('cuda:0'), verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        return []
    assert r.masks is not None and len(r.masks.data) == len(r.boxes)
    masks = r.masks.data.cpu().numpy() > .5
    assert masks.shape[1:] == (2048, 2048)
    if flip_ud:
        masks = masks[:, ::-1]
    if flip_lr:
        masks = masks[:, :, ::-1]
    return [{'box': [float(v) for v in box], 'conf': float(cf), 'counts': rle(m)}
            for box, cf, m in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), masks)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('member', choices=list(MEMBERS))
    parser.add_argument('mode', choices=['smoke', 'train'])
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = MEMBERS[args.member]
    name = f'det_{args.member}_f1'
    work = ROOT / 'outputs' / f'extra_{args.member}_{args.mode}_20261011'
    work.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import cv2
    import numpy as np
    import torch
    from solar.yolo_device import assert_gpu_one

    assert digest(ROOT / 'outputs/pretrained/yolo11m-seg.pt') == PRETRAINED_SHA
    formal_manifest = json.loads((ROOT / 'outputs/formal/manifest.json').read_text())

    def dump(file, obj):
        (work / file).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

    def set_stage(stage):
        (work / 'stage').write_text(stage)
        print(f'[stage] {stage}', flush=True)

    def save_manifest(g):
        assert g['train_stems'] == formal_manifest['train'] and g['val_stems'] == formal_manifest['val']
        assert {s: g['medoid_ann'][s] for s in g['stems_all']} == formal_manifest['medoid']
        dump('manifest.json', {'train': g['train_stems'], 'val': g['val_stems'], 'equals_formal_split': True})

    runtime = {'member': args.member, 'mode': args.mode, **cfg, 'source_sha256': SOURCE_SHA,
               'runner_sha256': digest(Path(__file__))}

    def attach_checks(det):
        def started(trainer):
            trainer.args.device = torch.device('cuda:0')
            trainer.validator.args.device = torch.device('cuda:0')
            runtime.update({'gpu': assert_gpu_one(), 'amp_effective': bool(trainer.amp), 'args': vars(trainer.args)})
            dump('runtime.json', runtime)

        def batch(trainer):
            assert_gpu_one()
            assert trainer.batch_size == 2, 'Native detector batch changed'
            assert torch.isfinite(trainer.loss).all(), 'Nonfinite detector loss'

        def ended(trainer):
            runtime['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            dump('runtime.json', runtime)

        det.add_callback('on_pretrain_routine_end', started)
        det.add_callback('on_train_batch_end', batch)
        det.add_callback('on_train_end', ended)

    source = member_source(work, name, cfg['imgsz'], cfg['seed'], args.mode == 'smoke')
    (work / 'executed_source.py').write_text(source, encoding='utf-8')
    env = {'__name__': '__main__', 'save_manifest': save_manifest,
           'attach_checks': attach_checks, 'set_stage': set_stage}
    set_stage('detector')
    runtime['gpu_before_source'] = assert_gpu_one()
    dump('runtime.json', runtime)
    exec(compile(source, str(SOURCE), 'exec'), env)
    best = work / f'{name}_best.pt'
    model = env['det']
    torch.cuda.empty_cache()

    dataset = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
    splits = [('val', formal_manifest['val'], dataset / 'train/train_images'),
              ('test', sorted(p.stem for p in (dataset / 'test/test_images').glob('*.jpeg')), dataset / 'test/test_images')]
    if args.mode == 'smoke':
        splits = [(s, stems[:2], d) for s, stems, d in splits]
    reports = {}
    for split, stems, image_dir in splits:
        set_stage(f'views_{split}')
        views = {}
        for n, stem in enumerate(stems):
            assert_gpu_one()
            path = image_dir / f'{stem}.jpeg'
            image = cv2.imread(str(path))
            per = [detect(model, str(path), False, False, cfg['imgsz'], torch)]
            for lr, ud in FLIPS[1:]:
                view = image[:, ::-1] if lr else image
                per.append(detect(model, np.ascontiguousarray(view[::-1] if ud else view), lr, ud, cfg['imgsz'], torch))
            views[stem] = per
            if (n + 1) % 20 == 0:
                print(f'[views] {split} {n + 1}/{len(stems)}', flush=True)
        out = work / f'flip_views_{split}.json'
        out.write_text(json.dumps(views))
        reports[split] = {'images': len(stems), 'detections_per_view': [sum(len(v[k]) for v in views.values()) for k in range(4)],
                          'json_sha256': digest(out), 'gpu': assert_gpu_one()}
        print(f'[done] {split}: {reports[split]}', flush=True)
    dump('verification.json', {'passed': True, 'member': args.member, **cfg, 'mode': args.mode,
                               'best_sha256': digest(best), 'pretrained_sha256': PRETRAINED_SHA,
                               'source_sha256': SOURCE_SHA, 'runner_sha256': digest(Path(__file__)),
                               'reports': reports})
    set_stage('complete')


if __name__ == '__main__':
    main()
