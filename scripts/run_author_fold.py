"""Run pinned HeShen fold1 source with explicit portability edits only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SOURCE = ROOT / 'src/solar/vendor/heshen/train.py'
SOURCE_SHA = '40347ae60cb818ca28798d8e2c57f3f8d412eb4a45b8974778dea32e50f660ee'
GPU_UUID = 'GPU-39364e20-0c22-554a-7b47-7159faa17d2c'


def replace_once(source: str, old: str, new: str) -> str:
    """Refuse source drift instead of applying approximate patches."""
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def portable_source(work: Path) -> str:
    """Preserve author recipes, including native targets and best selection."""
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    source = SOURCE.read_text(encoding='utf-8')
    source = replace_once(source, 'INPUT_ROOT = Path("/kaggle/input")', f'INPUT_ROOT = Path({str(ROOT / "data")!r})')
    source = replace_once(source, 'WORK = Path("/kaggle/working")', f'WORK = Path({str(work)!r})')
    source = replace_once(source, 'YOLO_DIR = Path("/tmp/yolo")', f'YOLO_DIR = Path({str(work / "yolo")!r})')
    source = replace_once(source, 'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)', '# Environment frozen before execution; no unpinned pip mutation.')
    source = replace_once(source, 'det = YOLO("yolo11m-seg.pt")', f'det = YOLO({str(ROOT / "outputs/pretrained/yolo11m-seg.pt")!r})\nattach_checks(det)')
    assert source.count('device=0') == 2
    source = source.replace('device=0', 'device=torch.device("cuda:0")')
    source = replace_once(source, '# ================= Stage 1', 'save_manifest(globals())\n# ================= Stage 1')
    source = replace_once(source, 'refiner = train_refiner()', 'set_stage("refiner")\nrefiner = train_refiner()')
    source = replace_once(source, 'opt.zero_grad()', 'assert torch.isfinite(loss), "Nonfinite refiner loss"\n            opt.zero_grad()')
    source = replace_once(source, 'sch.step()', 'assert nb > 0, "Refiner has no training batches"\n        sch.step()')
    source = replace_once(source, 'THR, GATE = 0.55, 0.10', 'set_stage("evaluation")\nTHR, GATE = 0.55, 0.10')
    source = replace_once(source, 'pq, tp, fp, fn, miou = pq_score(GT, pdf)', 'pdf.to_csv(WORK / f"pred_conf{cf}.csv", index=False)\n    pq, tp, fp, fn, miou = pq_score(GT, pdf)')
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'smoke', 'train'])
    args = parser.parse_args()
    os.chdir(ROOT)
    work = ROOT / 'outputs' / ('formal' if args.mode == 'train' else args.mode)
    work.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import torch
    from solar.yolo_device import assert_gpu_one

    def dump(name, obj):
        (work / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

    def set_stage(stage):
        (work / 'stage').write_text(stage)
        print(f'[stage] {stage}', flush=True)

    def save_manifest(g):
        train, val = g['train_stems'], g['val_stems']
        assert not {s[:8] for s in train} & {s[:8] for s in val}
        assert len(train) + len(val) == 707
        images = {s: hashlib.sha256((g['train_images_dir'] / f'{s}.jpeg').read_bytes()).hexdigest() for s in g['stems_all']}
        result = {'source_sha256': SOURCE_SHA, 'train': train, 'val': val,
             'fold_of': g['fold_of'], 'medoid': g['medoid_ann'], 'images_sha256': images,
             'annotation_sha256': hashlib.sha256(g['ann_json'].read_bytes()).hexdigest(),
             'selected_instances': {s: len(g['records'][s][g['medoid_ann'][s]]) for s in g['stems_all']}}
        if args.mode == 'train':
            assert result == json.loads((ROOT / 'outputs/prepare/manifest.json').read_text())
        dump('manifest.json', result)

    runtime = {'mode': args.mode, 'source_sha256': SOURCE_SHA,
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def attach_checks(det):
        def started(trainer):
            # Native trainer overwrites args.device with an environment string.
            # Its final validator lowercases UUID strings; retain the object.
            trainer.args.device = torch.device('cuda:0')
            trainer.validator.args.device = torch.device('cuda:0')
            runtime.update({'gpu': assert_gpu_one(), 'amp_requested': trainer.args.amp,
                            'amp_effective': bool(trainer.amp), 'args': vars(trainer.args)})
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

    source = portable_source(work)
    env = {'__name__': '__main__', 'save_manifest': save_manifest,
           'attach_checks': attach_checks, 'set_stage': set_stage}
    if args.mode == 'prepare':
        source = source.split('from ultralytics import YOLO')[0]
    elif args.mode == 'smoke':
        source = replace_once(source, 'for split in ("train", "val"):',
                              'train_stems = train_stems[:8]\nval_stems = val_stems[:2]\nfor split in ("train", "val"):')
        source = replace_once(source, 'epochs=50,', 'epochs=1,')
        source = source.split('set_stage("refiner")')[0]
    dump('environment.json', {'python': sys.version, 'packages': sorted(
        f'{d.metadata["Name"]}=={d.version}' for d in __import__('importlib.metadata', fromlist=['distributions']).distributions())})
    (work / 'executed_source.py').write_text(source, encoding='utf-8')
    set_stage('preparation' if args.mode == 'prepare' else 'detector')
    if args.mode != 'prepare':
        runtime['gpu_before_source'] = assert_gpu_one()
        dump('runtime.json', runtime)
    exec(compile(source, str(SOURCE), 'exec'), env)
    if args.mode == 'smoke':
        # Native predict after a fresh best.pt load; smoke-only short refiner run.
        torch.cuda.empty_cache()
        env['det'].predict(str(env['train_images_dir'] / f'{env["val_stems"][0]}.jpeg'),
                           imgsz=1536, device=torch.device('cuda:0'), verbose=False)
        runtime['gpu_after_detector_reload_predict'] = assert_gpu_one()
        dump('runtime.json', runtime)
        env['train_stems'] = json.loads((work / 'manifest.json').read_text())['train'][:32]
        env['EPOCHS'] = 1
        set_stage('refiner_smoke')
        model = env['train_refiner']()
        saved = torch.load(work / 'ref_fold1.pt', map_location='cpu', weights_only=False)
        reloaded = env['Refiner']().to('cuda')
        reloaded.load_state_dict(saved['model'])
        model.eval(); reloaded.eval()
        xb, _ = env['DS'](env['build_items']()[:1])[0]
        with torch.no_grad(), torch.autocast('cuda'):
            before = model(xb[None].to('cuda'))
            after = reloaded(xb[None].to('cuda'))
        assert torch.isfinite(after).all() and torch.equal(before, after)
        dump('smoke_passed.json', {'passed': True, 'gpu': assert_gpu_one(),
             'detector_best_reload_predict': True, 'refiner_batch': 32,
             'refiner_reload_equal': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()})
    set_stage('complete')


if __name__ == '__main__':
    main()
