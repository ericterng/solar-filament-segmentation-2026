"""Probe 22 (2026-10-25): Option B's m1536 member retrained with ultralytics copy_paste 0.0 -> 0.5.

Protocol: docs/experiments/yolo_copypaste_detector_20261025.md, fixed before any smoke run.
Template: run_yolo11x_detector_20261024.py (probe 18), whose helpers are imported unchanged.
The member is the pinned HeShen source through its detector stage, exactly as Option B's
m1536 (det4oof_f1): yolo11m-seg.pt, raw JPEGs, one medoid annotator per image, imgsz 1536,
batch 2, AMP, 50 epochs, patience 15, mosaic .25, close_mosaic 5, flipud .5, workers 2,
native best.pt, ultralytics 8.3.221 (the existing detector_amey_20261015 environment).
The ONLY edits after run_author_fold.portable_source are the seed (43 or 143) and
`copy_paste=0.5` appended to det.train(...). copy_paste_mode stays the default "flip":
ultralytics 8.3.221 CopyPaste mirrors the image left-right and pastes round(.5 * n) of the n
mirrored instances whose box overlaps no existing box by IoA >= .30 (deterministic, every
training image, segment polygons required); close_mosaic turns it off for the last 5 epochs.
At train start the runner asserts the trainer args, finds the CopyPaste transform in the
train dataset, and counts pasted instances on the first SELF_CHECK training images (fails if 0).
After training, best.pt predicts 141 val + 180 test on four flipped copies (probe 18 detect()).
A5000 shared-host conventions as probe 18: --gpu-uuid, PID -> UUID checks, refuses to overwrite.

    python scripts/run_yolo_copypaste_20261025.py {s43,s143} {smoke,train} --gpu-uuid GPU-...
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
from run_author_fold import SOURCE, SOURCE_SHA, replace_once  # noqa: E402
from run_yolo11x_detector_20261024 import (AMP_CHECK_SHA, ANNOTATIONS_SHA, BATCH, CHECK_EVERY, DATASET,  # noqa: E402
                                           FLIPS, IMGSZ, MANIFEST_SHA, REFERENCE_ARGS, REFERENCE_ARGS_SHA,
                                           ULTRALYTICS, cfg_diff, detect, m1536_source)

MEMBERS = {'s43': 43, 's143': 143}
COPY_PASTE, COPY_PASTE_MODE = 0.5, 'flip'
PRETRAINED = 'outputs/pretrained/yolo11m-seg.pt'
PRETRAINED_SHA = 'eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf'  # Option B m1536 start
TRAIN_CALL_END = 'flipud=0.5, mosaic=0.25, close_mosaic=5)'
SELF_CHECK = 50
#: Recipe keys allowed to differ from m1536's args.yaml (copy_paste by design; warmup_bias_lr is set to
#: 0.0 by optimizer=auto at runtime in every probe-18 / YOLO26 smoke, the saved args.yaml has 0.1).
EXPECTED_CHANGED = {'copy_paste', 'warmup_bias_lr', 'seed', 'epochs'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member_source(work: Path, seed: int, smoke: bool) -> str:
    """m1536_source with only the seed and copy_paste changed (smoke also shrinks data / epochs)."""
    source = m1536_source(work)
    source = replace_once(source, 'seed=42 + FOLD', f'seed={seed}')
    source = replace_once(source, TRAIN_CALL_END, TRAIN_CALL_END[:-1] + f', copy_paste={COPY_PASTE})')
    if smoke:
        source = replace_once(source, 'for split in ("train", "val"):',
                              'train_stems = train_stems[:8]\nval_stems = val_stems[:2]\nfor split in ("train", "val"):')
        source = replace_once(source, 'epochs=50,', 'epochs=1,')
    return source


def copypaste_transforms(transform) -> list:
    """Every distinct CopyPaste object reachable from a dataset's transform tree."""
    from ultralytics.data.augment import CopyPaste
    found, seen, stack = [], set(), [transform]
    while stack:
        t = stack.pop()
        if t is None or id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, CopyPaste):
            found.append(t)
        stack.extend(getattr(t, 'transforms', None) or [])
        stack.append(getattr(t, 'pre_transform', None))
    return found


def paste_count(labels: dict, p: float = COPY_PASTE) -> tuple[int, int]:
    """(instances before, instances pasted) for one labels dict under ultralytics' flip-mode CopyPaste."""
    from ultralytics.data.augment import CopyPaste
    before = len(labels['instances'])
    out = CopyPaste(p=p, mode=COPY_PASTE_MODE)(labels)
    return before, len(out['instances']) - before


def self_check(dataset, n: int) -> dict:
    """Paste count on the first n training images, in the main process (flip mode draws no random numbers)."""
    before = pasted = hit = 0
    for i in range(min(n, len(dataset))):
        b, p = paste_count(dataset.get_image_and_label(i))
        before, pasted, hit = before + b, pasted + p, hit + (p > 0)
    return {'images': min(n, len(dataset)), 'instances': before, 'pasted': pasted, 'images_with_paste': hit}


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
    assert digest(ROOT / PRETRAINED) == PRETRAINED_SHA
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

    work = ROOT / 'outputs' / (f'yolo_cp_{args.member}_smoke_20261025' if smoke else f'yolo_cp_{args.member}_20261025')
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
               'runner_sha256': digest(Path(__file__)),
               'probe18_runner_sha256': digest(ROOT / 'scripts/run_yolo11x_detector_20261024.py'),
               'pretrained_sha256': PRETRAINED_SHA, 'copy_paste': COPY_PASTE, 'copy_paste_mode': COPY_PASTE_MODE,
               'ultralytics': ultralytics.__version__, 'torch': torch.__version__, 'physical_index': physical,
               'amp_check_sha256': AMP_CHECK_SHA, 'copypaste_by_epoch': []}
    batches = {'n': 0}

    def active_copypaste(trainer) -> list[float]:
        return [t.p for t in copypaste_transforms(trainer.train_loader.dataset.transforms)]

    def attach_checks(det):
        def started(trainer):
            # Native trainer overwrites args.device with an environment string; keep the object.
            trainer.args.device = torch.device('cuda:0')
            trainer.validator.args.device = torch.device('cuda:0')
            assert trainer.args.copy_paste == COPY_PASTE and trainer.args.copy_paste_mode == COPY_PASTE_MODE, \
                (trainer.args.copy_paste, trainer.args.copy_paste_mode)
            assert trainer.args.task == 'segment' and trainer.train_loader.dataset.use_segments
            assert active_copypaste(trainer) == [COPY_PASTE], active_copypaste(trainer)
            check_result = self_check(trainer.train_loader.dataset, SELF_CHECK)
            assert check_result['pasted'] > 0, check_result
            actual = vars(trainer.args)
            diff = cfg_diff(actual, reference)
            unexpected = set(diff['changed']) - EXPECTED_CHANGED
            runtime.update({'gpu_train_start': check(), 'amp_requested': trainer.args.amp,
                            'amp_effective': bool(trainer.amp), 'args': actual, 'copypaste_self_check': check_result,
                            'cfg_unexpected_changes': sorted(unexpected),
                            'optimizer_class': type(trainer.optimizer).__name__ if trainer.optimizer else None})
            dump('runtime.json', runtime)
            dump('cfg_diff.json', diff)
            print('[cfg_diff]', json.dumps(diff, default=str), flush=True)
            print('[copypaste_self_check]', json.dumps(check_result), flush=True)
            assert not unexpected and not diff['only_new'] and not diff['only_reference'], diff

        def batch(trainer):
            assert trainer.batch_size == BATCH, 'Native detector batch changed'
            assert torch.isfinite(trainer.loss).all(), 'Nonfinite detector loss'
            batches['n'] += 1
            if batches['n'] % CHECK_EVERY == 0:
                check()

        def epoch_end(trainer):
            check()
            runtime['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            runtime['copypaste_by_epoch'].append([trainer.epoch + 1, active_copypaste(trainer)])
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
                               'copy_paste': COPY_PASTE, 'copy_paste_mode': COPY_PASTE_MODE,
                               'copypaste_self_check': runtime['copypaste_self_check'],
                               'copypaste_by_epoch': runtime['copypaste_by_epoch'],
                               'best_sha256': digest(best), 'pretrained_sha256': PRETRAINED_SHA,
                               'source_sha256': SOURCE_SHA, 'runner_sha256': digest(Path(__file__)),
                               'ultralytics': ultralytics.__version__, 'amp_effective': runtime.get('amp_effective'),
                               'gpu_after': check(), 'reports': reports})
    set_stage('complete')


if __name__ == '__main__':
    main()
