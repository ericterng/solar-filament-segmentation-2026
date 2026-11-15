"""Probe 18 (2026-10-24): YOLO11x-seg 1536 in place of / next to Option B's m1536 member.

Protocol: docs/experiments/yolo11x_detector_20261024.md, fixed before any smoke run.
Template: run_yolo26_detector_20261020.py. The member is the pinned HeShen source through
its detector stage, exactly as Option B's m1536 (det4oof_f1): raw JPEGs, one medoid
annotator per image, imgsz 1536, batch 2, AMP, 50 epochs, patience 15, mosaic .25,
close_mosaic 5, flipud .5, workers 2, native best.pt, ultralytics 8.3.221 (the existing
detector_amey_20261015 environment). The ONLY edits after run_author_fold.portable_source
are the pretrained weights (yolo11m-seg.pt -> yolo11x-seg.pt) and the seed (43 or 143).
After training, best.pt predicts 141 val + 180 test on four flipped copies at imgsz
1536 (conf .10, iou .60, agnostic NMS, retina masks) -> flip_views_{val,test}.json with
the other members' format.
A5000 shared-host conventions: the UUID comes from --gpu-uuid; PID -> UUID is
verified through nvidia-smi before the source, at train start, every 50 batches, every
epoch, after training and per image. Refuses to overwrite.

    python scripts/run_yolo11x_detector_20261024.py {s43,s143} {smoke,train} --gpu-uuid GPU-...
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
from run_author_fold import SOURCE, SOURCE_SHA, portable_source, replace_once  # noqa: E402

MEMBERS = {'s43': 43, 's143': 143}
OLD_PRETRAINED = 'outputs/pretrained/yolo11m-seg.pt'
NEW_PRETRAINED = 'outputs/pretrained/yolo11x-seg.pt'
#: Downloaded 2026-10-24 from ultralytics/assets release v8.3.0 (125,090,821 bytes).
PRETRAINED_SHA = '4e53a5f5fd3ee2ae3361c62169c6bb3ed4ae251dd0e57606e230955aa52d919c'
AMP_CHECK_SHA = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'  # yolo11n.pt, AMP check only
ANNOTATIONS_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
MANIFEST_SHA = '3cf138309bbd992199edd2e29199c56da6b927abd600e6b679de4155077ebaa2'
ULTRALYTICS = '8.3.221'
REFERENCE_ARGS = ROOT / 'docs/experiments/yolo26_m1536_reference_args_20261020.yaml'
REFERENCE_ARGS_SHA = '51ab729bfb3ac6d90ed9133281f83357dba22de2186640632544f84029d0b087'
#: Keys that differ by construction (paths, run directory) and are not recipe settings.
PATH_KEYS = ('model', 'data', 'project', 'save_dir', 'device')
IMGSZ, BATCH, CHECK_EVERY = 1536, 2, 50
PREDICT = dict(conf=.10, iou=.60, agnostic_nms=True, retina_masks=True, imgsz=IMGSZ, verbose=False)
FLIPS = ((False, False), (True, False), (False, True), (True, True))
DATASET = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def m1536_source(work: Path) -> str:
    """Option B's m1536 member: the portable author source through its detector stage."""
    source = portable_source(work)
    return source[:source.index('print("[det] done", flush=True)')] + 'print("[det] done", flush=True)\n'


def member_source(work: Path, seed: int, smoke: bool) -> str:
    """m1536_source with only the pretrained weights and the seed changed (smoke also shrinks data / epochs)."""
    source = m1536_source(work)
    source = replace_once(source, repr(str(ROOT / OLD_PRETRAINED)), repr(str(ROOT / NEW_PRETRAINED)))
    source = replace_once(source, 'seed=42 + FOLD', f'seed={seed}')
    if smoke:
        source = replace_once(source, 'for split in ("train", "val"):',
                              'train_stems = train_stems[:8]\nval_stems = val_stems[:2]\nfor split in ("train", "val"):')
        source = replace_once(source, 'epochs=50,', 'epochs=1,')
    return source


def cfg_diff(actual: dict, reference: dict) -> dict:
    """Every trainer arg that differs from the m1536 run's args.yaml, path keys reported separately."""
    out = {'changed': {}, 'only_new': {}, 'only_reference': {}, 'paths': {}}
    for key in sorted(set(actual) | set(reference)):
        if key not in reference:
            out['only_new'][key] = actual[key]
        elif key not in actual:
            out['only_reference'][key] = reference[key]
        elif str(actual[key]) != str(reference[key]):
            bucket = 'paths' if key in PATH_KEYS else 'changed'
            out[bucket][key] = {'reference': reference[key], 'new': actual[key]}
    return out


def rle(mask) -> str:
    from pycocotools import mask as mask_util
    import numpy as np
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('ascii')


def detect(model, source, flip_lr: bool, flip_ud: bool, torch) -> list[dict]:
    r = model.predict(source, device=torch.device('cuda:0'), **PREDICT)[0]
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
    parser.add_argument('--gpu-uuid', default=os.environ.get('DETECTOR_GPU_UUID'),
                        help='physical GPU UUID to use (default: $DETECTOR_GPU_UUID)')
    args = parser.parse_args()
    if not args.gpu_uuid:
        parser.error('--gpu-uuid (or DETECTOR_GPU_UUID) is required')
    os.chdir(ROOT)
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    seed, smoke = MEMBERS[args.member], args.mode == 'smoke'
    assert digest(ROOT / NEW_PRETRAINED) == PRETRAINED_SHA
    # ultralytics 8.3 check_amp loads ./yolo11n.pt (cwd = ROOT) and would download it if missing.
    assert digest(ROOT / 'yolo11n.pt') == AMP_CHECK_SHA
    assert digest(REFERENCE_ARGS) == REFERENCE_ARGS_SHA
    assert digest(DATASET / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json') == ANNOTATIONS_SHA
    assert digest(ROOT / 'outputs/formal/manifest.json') == MANIFEST_SHA
    from solar.gpu_uuid_pin import assert_pinned, pin_uuid
    physical = pin_uuid(args.gpu_uuid)
    import cv2
    import numpy as np
    import torch
    import ultralytics
    import yaml
    assert ultralytics.__version__ == ULTRALYTICS, ultralytics.__version__
    check = lambda: assert_pinned(args.gpu_uuid)

    work = ROOT / 'outputs' / (f'yolo11x_{args.member}_smoke_20261024' if smoke else f'yolo11x_{args.member}_20261024')
    work.mkdir(parents=True, exist_ok=False)
    formal_manifest = json.loads((ROOT / 'outputs/formal/manifest.json').read_text())
    reference = yaml.safe_load(REFERENCE_ARGS.read_text())

    def dump(file, obj):
        (work / file).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

    def set_stage(stage):
        (work / 'stage').write_text(stage)
        print(f'[stage] {stage}', flush=True)

    def save_manifest(g):
        assert g['train_stems'] == formal_manifest['train'] and g['val_stems'] == formal_manifest['val']
        assert {s: g['medoid_ann'][s] for s in g['stems_all']} == formal_manifest['medoid']
        dump('manifest.json', {'train': g['train_stems'], 'val': g['val_stems'], 'equals_formal_split': True})

    runtime = {'member': args.member, 'mode': args.mode, 'imgsz': IMGSZ, 'seed': seed, 'source_sha256': SOURCE_SHA,
               'runner_sha256': digest(Path(__file__)), 'pretrained_sha256': PRETRAINED_SHA,
               'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'physical_index': physical,
               'amp_check_sha256': AMP_CHECK_SHA}
    batches = {'n': 0}

    def attach_checks(det):
        def started(trainer):
            # Native trainer overwrites args.device with an environment string; keep the object.
            trainer.args.device = torch.device('cuda:0')
            trainer.validator.args.device = torch.device('cuda:0')
            actual = vars(trainer.args)
            diff = cfg_diff(actual, reference)
            runtime.update({'gpu_train_start': check(), 'amp_requested': trainer.args.amp,
                            'amp_effective': bool(trainer.amp), 'args': actual,
                            'optimizer_class': type(trainer.optimizer).__name__ if trainer.optimizer else None})
            dump('runtime.json', runtime)
            dump('cfg_diff.json', diff)
            print('[cfg_diff]', json.dumps(diff, default=str), flush=True)
            print('[optimizer]', runtime['optimizer_class'], flush=True)

        def batch(trainer):
            assert trainer.batch_size == BATCH, 'Native detector batch changed'
            assert torch.isfinite(trainer.loss).all(), 'Nonfinite detector loss'
            batches['n'] += 1
            if batches['n'] % CHECK_EVERY == 0:
                check()

        def epoch_end(trainer):
            check()
            runtime['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            dump('runtime.json', runtime)

        def ended(trainer):
            runtime.update({'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'train_batches': batches['n'],
                            'optimizer_class_end': type(trainer.optimizer).__name__})
            dump('runtime.json', runtime)

        det.add_callback('on_pretrain_routine_end', started)
        det.add_callback('on_train_batch_end', batch)
        det.add_callback('on_train_epoch_end', epoch_end)
        det.add_callback('on_train_end', ended)

    source = member_source(work, seed, smoke)
    (work / 'executed_source.py').write_text(source, encoding='utf-8')
    env = {'__name__': '__main__', 'save_manifest': save_manifest,
           'attach_checks': attach_checks, 'set_stage': set_stage}
    set_stage('detector')
    runtime['gpu_before_source'] = check()
    dump('runtime.json', runtime)
    exec(compile(source, str(SOURCE), 'exec'), env)
    best = work / 'det4oof_f1_best.pt'
    assert best.exists()
    model = env['det']
    torch.cuda.empty_cache()
    # The end-of-training validation may rewrite the env string; CUDA was initialised on the
    # UUID already, so restore the string and re-check the real PID -> UUID mapping.
    runtime['visible_after_train'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    runtime['gpu_after_train'] = check()

    splits = [('val', formal_manifest['val'], DATASET / 'train/train_images'),
              ('test', sorted(p.stem for p in (DATASET / 'test/test_images').glob('*.jpeg')), DATASET / 'test/test_images')]
    assert len(splits[0][1]) == 141 and len(splits[1][1]) == 180
    if smoke:
        splits = [(s, stems[:2], d) for s, stems, d in splits]
    reports = {}
    for split, stems, image_dir in splits:
        set_stage(f'views_{split}')
        views = {}
        for n, stem in enumerate(stems):
            check()
            path = image_dir / f'{stem}.jpeg'
            image = cv2.imread(str(path))
            assert image is not None and image.shape[:2] == (2048, 2048)
            per = [detect(model, str(path), False, False, torch)]
            for lr, ud in FLIPS[1:]:
                view = image[:, ::-1] if lr else image
                per.append(detect(model, np.ascontiguousarray(view[::-1] if ud else view), lr, ud, torch))
            views[stem] = per
            if (n + 1) % 20 == 0:
                print(f'[views] {split} {n + 1}/{len(stems)}', flush=True)
        out = work / f'flip_views_{split}.json'
        out.write_text(json.dumps(views))
        reports[split] = {'images': len(stems), 'detections_per_view': [sum(len(v[k]) for v in views.values()) for k in range(4)],
                          'json_sha256': digest(out)}
        print(f'[done] {split}: {reports[split]}', flush=True)
    runtime['peak_allocated_bytes_total'] = torch.cuda.max_memory_allocated()
    dump('runtime.json', runtime)
    dump('verification.json', {'passed': True, 'member': args.member, 'seed': seed, 'imgsz': IMGSZ, 'mode': args.mode,
                               'best_sha256': digest(best), 'pretrained_sha256': PRETRAINED_SHA,
                               'source_sha256': SOURCE_SHA, 'runner_sha256': digest(Path(__file__)),
                               'ultralytics': ultralytics.__version__, 'amp_effective': runtime.get('amp_effective'),
                               'gpu_after': check(), 'reports': reports})
    set_stage('complete')


if __name__ == '__main__':
    main()
