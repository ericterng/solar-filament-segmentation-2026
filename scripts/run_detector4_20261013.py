"""Train a 4th Option B detector with the upstream (ameyypawar) recipe, then save flip views.

Research report section 4.4 (2026-10-12/13): the three Option B members all use
the HeShen recipe (raw JPEG, one medoid annotator per image, mosaic). This one
follows vendored third_party/ameyypawar-filament-seg scripts/export_yolo.py and
train_detector.py instead, so its errors are more likely to differ:
- Images: the limb-darkening-flattened PNGs the U-Net sees (upstream
  preprocess cache, outputs/amey_flat_20261013 = local outputs/amey_unet_cache/flat).
- Labels: every annotator record of a training image is its own sample;
  largest polygon per filament, normalised (upstream label_lines).
- Train set: the 566 fold1 train images only (outputs/formal/manifest.json); the
  141 val images are never read before inference. Ultralytics' required val set
  is the first 16 train samples and is unused (val=False).
- Fixed in advance: yolo11m-seg (same pretrained SHA as the other members),
  imgsz 1280, seed 46, 60 epochs, cos_lr, no early stop, last.pt; upstream
  augmentation (mosaic 0, flips .5, translate .05, scale .2, hsv_v .2,
  mask_ratio 4); batch 2 (GTX 1080 Ti 11 GB, as the other members), AMP default.
- Inference: last.pt on four flipped copies of each flat PNG for 141 val + 180
  test (conf .10, iou .60, agnostic NMS, retina masks, imgsz 1280), all four
  views predicted from cv2.imread arrays as the training loader reads them.
GPU1 only (UUID / PID checked); refuses to overwrite. smoke = 8 samples,
1 epoch, 2 images per split.
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
from run_author_fold import GPU_UUID
from export_author_detections_20261011 import rle

PRETRAINED_SHA = 'eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf'
FLAT = ROOT / 'outputs/amey_flat_20261013'
DATASET = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
FLIPS = ((False, False), (True, False), (False, True), (True, True))
IMGSZ, SEED, EPOCHS, BATCH = 1280, 46, 60, 2


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polygon_area(polygon: list[float], np) -> float:
    xs, ys = np.asarray(polygon[0::2]), np.asarray(polygon[1::2])
    return 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def label_lines(record: dict, annotations: list[dict], np) -> list[str]:
    lines = []
    for annotation in annotations:
        polygons = [p for p in annotation.get('segmentation') or [] if len(p) >= 6]
        if not polygons:
            continue
        polygon = max(polygons, key=lambda p: polygon_area(p, np))
        xs = np.clip(np.asarray(polygon[0::2], dtype=np.float64) / record['width'], 0.0, 1.0)
        ys = np.clip(np.asarray(polygon[1::2], dtype=np.float64) / record['height'], 0.0, 1.0)
        lines.append('0 ' + ' '.join(f'{x:.6f} {y:.6f}' for x, y in zip(xs, ys)))
    return lines


def build_dataset(out: Path, train_stems: list[str], limit: int, np) -> dict:
    raw = json.loads((DATASET / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text(encoding='utf-8'))
    by_image: dict[int, list[dict]] = {}
    for ann in raw['annotations']:
        by_image.setdefault(ann['image_id'], []).append(ann)
    wanted = set(train_stems)
    records = sorted((r for r in raw['images'] if Path(r['file_name']).stem in wanted), key=lambda r: r['id'])
    assert {Path(r['file_name']).stem for r in records} == wanted
    if limit:
        records = records[:limit]
    subsets = {'train': records, 'val': records[:16]}
    counts = {}
    for subset, recs in subsets.items():
        (out / 'images' / subset).mkdir(parents=True)
        (out / 'labels' / subset).mkdir(parents=True)
        n = 0
        for r in recs:
            os.link(FLAT / f"{Path(r['file_name']).stem}.png", out / 'images' / subset / f"{r['id']}.png")
            lines = label_lines(r, by_image.get(r['id'], []), np)
            (out / 'labels' / subset / f"{r['id']}.txt").write_text('\n'.join(lines) + '\n', encoding='utf-8')
            n += len(lines)
        counts[subset] = {'samples': len(recs), 'filaments': n}
    (out / 'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n',
                                   encoding='utf-8')
    return counts


def detect(model, image, flip_lr: bool, flip_ud: bool, np, torch) -> list[dict]:
    view = image[:, ::-1] if flip_lr else image
    view = np.ascontiguousarray(view[::-1] if flip_ud else view)
    r = model.predict(view, conf=.10, iou=.60, agnostic_nms=True, retina_masks=True, imgsz=IMGSZ,
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
    parser.add_argument('mode', choices=['smoke', 'train'])
    args = parser.parse_args()
    smoke = args.mode == 'smoke'
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from solar.yolo_device import assert_gpu_one

    work = ROOT / 'outputs' / f'det4_amey_{args.mode}_20261013'
    work.mkdir(exist_ok=False)
    stage = lambda s: ((work / 'stage').write_text(s), print(f'[stage] {s}', flush=True))
    pretrained = ROOT / 'outputs/pretrained/yolo11m-seg.pt'
    assert digest(pretrained) == PRETRAINED_SHA
    flat_check = json.loads((FLAT / 'sha256.json').read_text())
    manifest = json.loads((ROOT / 'outputs/formal/manifest.json').read_text())
    assert len(manifest['train']) == 566 and len(manifest['val']) == 141 and not set(manifest['train']) & set(manifest['val'])
    test_stems = sorted(p.stem for p in (DATASET / 'test/test_images').glob('*.jpeg'))
    assert len(test_stems) == 180 and set(flat_check) == set(manifest['train']) | set(manifest['val']) | set(test_stems)
    assert all(digest(FLAT / f'{s}.png') == h for s, h in flat_check.items()), 'flat PNG differs from local cache'

    stage('dataset')
    counts = build_dataset(work / 'yolo', manifest['train'], 8 if smoke else 0, np)
    runtime = {'mode': args.mode, 'imgsz': IMGSZ, 'seed': SEED, 'epochs': 1 if smoke else EPOCHS, 'batch': BATCH,
               'dataset': counts, 'runner_sha256': digest(Path(__file__)), 'gpu_before': assert_gpu_one()}
    (work / 'runtime.json').write_text(json.dumps(runtime, indent=2))
    print(json.dumps(counts), flush=True)

    stage('train')
    model = YOLO(str(pretrained))

    def started(trainer):
        trainer.args.device = torch.device('cuda:0')
        trainer.validator.args.device = torch.device('cuda:0')
        runtime['gpu_train_start'] = assert_gpu_one()

    def batch_end(trainer):
        assert_gpu_one()
        assert torch.isfinite(trainer.loss).all(), 'Nonfinite detector loss'

    def train_end(trainer):
        runtime.update({'amp_effective': bool(trainer.amp), 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        (work / 'runtime.json').write_text(json.dumps(runtime, indent=2, default=str))

    model.add_callback('on_pretrain_routine_end', started)
    model.add_callback('on_train_batch_end', batch_end)
    model.add_callback('on_train_end', train_end)
    model.train(data=str(work / 'yolo/data.yaml'), imgsz=IMGSZ, epochs=runtime['epochs'], batch=BATCH,
                device=torch.device('cuda:0'), workers=4, patience=1000, project=str(work), name='train', exist_ok=True,
                seed=SEED, deterministic=False, single_cls=True, cos_lr=True, plots=False, val=False,
                mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0, fliplr=0.5, flipud=0.5, degrees=0.0,
                shear=0.0, perspective=0.0, translate=0.05, scale=0.2, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2,
                overlap_mask=True, mask_ratio=4)
    last = work / 'train/weights/last.pt'
    assert last.exists()
    del model
    torch.cuda.empty_cache()
    # Ultralytics' end-of-training validation rewrites the env string; CUDA was already
    # initialised on the UUID, so restore the string and re-check the real PID -> UUID mapping.
    runtime['visible_after_train'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    runtime['gpu_after_train'] = assert_gpu_one()

    model = YOLO(str(last))
    reports = {}
    for split, stems in (('val', manifest['val']), ('test', test_stems)):
        stems = stems[:2] if smoke else stems
        stage(f'views_{split}')
        views = {}
        for n, stem in enumerate(stems):
            assert_gpu_one()
            image = cv2.imread(str(FLAT / f'{stem}.png'))
            assert image is not None and image.shape == (2048, 2048, 3)
            views[stem] = [detect(model, image, lr, ud, np, torch) for lr, ud in FLIPS]
            if (n + 1) % 20 == 0:
                print(f'[views] {split} {n + 1}/{len(stems)}', flush=True)
        path = work / f'flip_views_{split}.json'
        path.write_text(json.dumps(views))
        reports[split] = {'images': len(stems), 'detections_per_view': [sum(len(v[k]) for v in views.values()) for k in range(4)],
                          'json_sha256': digest(path)}
        print(f'[done] {split}: {reports[split]}', flush=True)
    (work / 'verification.json').write_text(json.dumps(
        {'passed': True, **runtime, 'last_sha256': digest(last), 'pretrained_sha256': PRETRAINED_SHA,
         'gpu_after': assert_gpu_one(), 'reports': reports}, indent=2, default=str))
    stage('complete')


if __name__ == '__main__':
    main()
