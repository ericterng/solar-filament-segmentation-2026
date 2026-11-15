"""Replay native author decoding, then export supplied test images at conf .35."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_author_fold import GPU_UUID, SOURCE, SOURCE_SHA


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_definitions(source: str) -> str:
    """Extract original imports/constants and functions without training calls."""
    return (source[:source.index('INPUT_ROOT =')]
            + source[source.index('CROP, CONTEXT'):source.index('refiner = train_refiner()')]
            + source[source.index('def disk_mask'):source.index('gt_rows = []')])


def native_loop(source: str) -> str:
    loop = source[source.index('for n, stem in enumerate(val_stems)'):source.index('json.dump(raw10')]
    assert loop.count('device=0') == 1
    loop = loop.replace('device=0', 'device=torch.device("cuda:0")')
    return loop.replace('    g = gray_of(stem)', '    assert_gpu_one()\n    g = gray_of(stem)', 1)


def main() -> None:
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import torch
    import pandas as pd
    from ultralytics import YOLO
    from solar.yolo_device import assert_gpu_one
    from solar.submission import write_submission, validate_submission

    formal = ROOT / 'outputs/formal'
    output = ROOT / 'outputs/submission_conf035'
    output.mkdir(exist_ok=False)
    assert digest(SOURCE) == SOURCE_SHA
    assert json.loads((formal / 'independent_verification.json').read_text())['passed']
    protected = [formal / name for name in ('det4oof_f1_best.pt', 'ref_fold1.pt',
                 'manifest.json', 'pred_conf0.35.csv', 'pq_report_fold1.json')]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before[str(protected[0].relative_to(ROOT))] == '88cfa497ee5a70fcc899f141d8cb74cd4559ec2cf02eb67f2eeb61db60277fb4'
    assert before[str(protected[1].relative_to(ROOT))] == '787c6db03aa5b868125f2c76fd38f7642f69e936e9312907a73a981171e319a8'
    gpu = assert_gpu_one()
    source = SOURCE.read_text(encoding='utf-8')
    definitions, loop = native_definitions(source), native_loop(source)
    (output / 'native_definitions.py').write_text(definitions)
    (output / 'native_loop.py').write_text(loop)
    env = {'assert_gpu_one': assert_gpu_one}
    exec(compile(definitions, str(SOURCE), 'exec'), env)
    env['refiner'] = env['Refiner']().to('cuda')
    checkpoint = torch.load(formal / 'ref_fold1.pt', map_location='cpu', weights_only=False)
    assert checkpoint['fold_excluded'] == 1
    env['refiner'].load_state_dict(checkpoint['model'])
    env['refiner'].eval()
    env['det'] = YOLO(str(formal / 'det4oof_f1_best.pt'))
    manifest = json.loads((formal / 'manifest.json').read_text())
    dataset = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
    reports = {}
    for split, stems, image_dir in (
        ('dev_replay', manifest['val'], dataset / 'train/train_images'),
        ('test', sorted(p.stem for p in (dataset / 'test/test_images').glob('*.jpeg')), dataset / 'test/test_images'),
    ):
        print(f'[stage] {split}', flush=True)
        (output / 'stage').write_text(split)
        if split == 'test':
            assert len(stems) == 180 and len(set(stems)) == 180
        image_hashes = {s: digest(image_dir / f'{s}.jpeg') for s in stems}
        env.update({'train_images_dir': image_dir, 'val_stems': stems, 'GRAY_CACHE': {},
                    'THR': .55, 'GATE': .10, 'CONFS': (.35,), 'rles': {.35: []}, 'raw10': {},
                    't0': env['time'].time()})
        exec(compile(loop, str(SOURCE), 'exec'), env)
        predictions = {s: [] for s in stems}
        for stem, counts in env['rles'][.35]:
            predictions[stem].append({'size': [2048, 2048], 'counts': counts.encode()})
        assert set(env['raw10']) == set(stems)
        split_dir = output / split
        split_dir.mkdir()
        if split == 'dev_replay':
            original = pd.read_csv(formal / 'pred_conf0.35.csv')
            cached = {s: [] for s in stems}
            for row in original.itertuples():
                cached[row.filament_id.rsplit('_', 1)[0]].append(row.segmentation_rle)
            replayed = {s: [r['counts'].decode() for r in rs] for s, rs in predictions.items()}
            assert replayed == cached, 'Native fresh-process dev RLE differs from validated run'
        path = split_dir / 'submission.csv'
        write_submission(path, predictions)
        expected = {s: len(rs) for s, rs in predictions.items()}
        checks = validate_submission(path, expected)
        reports[split] = {'passed': True, 'csv_sha256': digest(path), 'bytes': path.stat().st_size,
                          'validation': checks, 'counts': expected, 'images_sha256': image_hashes,
                          'native_dev_rle_equal': split == 'dev_replay', 'gpu': assert_gpu_one()}
        (split_dir / 'manifest.json').write_text(json.dumps(reports[split], indent=2))
        (split_dir / 'raw_boxes.json').write_text(json.dumps(env['raw10']))
        print(f'[done] {split}: {checks}', flush=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before == after
    result = {'passed': True, 'conf': .35, 'source_sha256': SOURCE_SHA,
              'exporter_sha256': digest(Path(__file__)), 'protected_hashes': before,
              'gpu': gpu, 'reports': reports}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    (output / 'stage').write_text('complete')


if __name__ == '__main__':
    main()
