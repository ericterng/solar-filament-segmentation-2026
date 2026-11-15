"""Save every author fold1 detection (conf >= .10) with both of its masks.

Option 1 groundwork (2026-10-11): fusing the U-Net with only the conf.35,
gated, ownership-resolved YOLO masks left most U-Net pixels unclaimed. This
re-runs the unchanged author inference (no training) on the 141 fold1 val and
180 test images and keeps, per detection: conf, box, the YOLO native mask
(retina_masks, > .5) and the refiner mask (prob > THR .55 & disk, before the
area gate and ownership), plus whether it passes the native gate.

Parity gate: replaying the native gate + confidence ownership at conf .35 from
the saved refiner masks must reproduce the earlier exports RLE for RLE (dev:
formal pred_conf0.35.csv; test: submission_conf035/test), and boxes must equal
the earlier raw_boxes.json. Runs on physical GPU1 only; refuses to overwrite.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_author_fold import GPU_UUID, SOURCE, SOURCE_SHA
from export_author_submission import digest, native_definitions

THR, GATE, CONF_FLOOR, PARITY_CONF = .55, .10, .10, .35


def rle(mask) -> str:
    from pycocotools import mask as mask_util
    import numpy as np
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('ascii')


def ownership(dets: list[dict], masks: list, conf: float, min_area: int) -> list[str]:
    """Native conf-ordered exclusive ownership over gate-passing refiner masks."""
    import numpy as np
    kept = [(d['conf'], m) for d, m in zip(dets, masks) if d['conf'] >= conf and d['ref_gate']]
    occ = np.zeros((2048, 2048), bool)
    out = []
    for _, m in sorted(kept, key=lambda z: -z[0]):
        m2 = m.astype(bool) & ~occ
        if m2.sum() >= min_area:
            occ |= m2
            out.append(rle(m2))
    return out


def capture(env: dict, stems: list[str], image_dir: Path, torch) -> tuple[dict, dict]:
    import numpy as np
    env.update({'train_images_dir': image_dir, 'GRAY_CACHE': {}})
    detections, replay = {}, {}
    for n, stem in enumerate(stems):
        env['assert_gpu_one']()
        g = env['gray_of'](stem)
        disk = env['disk_mask'](g)
        r = env['det'].predict(str(image_dir / f'{stem}.jpeg'), conf=CONF_FLOOR, iou=0.60,
                               agnostic_nms=True, retina_masks=True, imgsz=1536,
                               device=torch.device('cuda:0'), verbose=False)[0]
        dets, ref_masks = [], []
        if r.boxes is not None and len(r.boxes):
            assert r.masks is not None and len(r.masks.data) == len(r.boxes)
            yolo = r.masks.data.cpu().numpy() > .5
            assert yolo.shape[1:] == (2048, 2048)
            for k, ((a, b, c, d), cf) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())):
                box = [float(a), float(b), float(c), float(d)]
                bside = max(box[2] - box[0], box[3] - box[1])
                m = (env['refine_prob'](g, *box) > THR).astype(np.uint8) & disk
                area = int(m.sum())
                gate = area >= env['MIN_AREA'] and area >= GATE * max(96.0, bside) ** 2 * 0.05
                dets.append({'box': box, 'conf': float(cf), 'yolo_counts': rle(yolo[k]),
                             'yolo_area': int(yolo[k].sum()), 'ref_counts': rle(m),
                             'ref_area': area, 'ref_gate': bool(gate)})
                ref_masks.append(m)
        detections[stem] = dets
        replay[stem] = ownership(dets, ref_masks, PARITY_CONF, env['MIN_AREA'])
        if (n + 1) % 20 == 0:
            print(f'[capture] {n + 1}/{len(stems)}', flush=True)
    return detections, replay


def main() -> None:
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_UUID
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    import torch
    import pandas as pd
    from ultralytics import YOLO
    from solar.yolo_device import assert_gpu_one

    formal = ROOT / 'outputs/formal'
    previous = ROOT / 'outputs/submission_conf035'
    output = ROOT / 'outputs/detections_conf010_20261011'
    output.mkdir(exist_ok=False)
    assert digest(SOURCE) == SOURCE_SHA
    protected = [formal / 'det4oof_f1_best.pt', formal / 'ref_fold1.pt', formal / 'manifest.json',
                 formal / 'pred_conf0.35.csv', previous / 'test/submission.csv',
                 previous / 'dev_replay/raw_boxes.json', previous / 'test/raw_boxes.json']
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before['outputs/formal/det4oof_f1_best.pt'] == '88cfa497ee5a70fcc899f141d8cb74cd4559ec2cf02eb67f2eeb61db60277fb4'
    assert before['outputs/formal/ref_fold1.pt'] == '787c6db03aa5b868125f2c76fd38f7642f69e936e9312907a73a981171e319a8'
    assert before['outputs/formal/pred_conf0.35.csv'] == '1684931589acb8c37130f445c0d57f7b47c75e3c03d5c2d7f14db7e69102a742'
    assert before['outputs/submission_conf035/test/submission.csv'] == 'a93771cc89f6ba9e5f3d5793e49581d7638c912645efbcaacabf62560dd34671'
    gpu = assert_gpu_one()
    env = {'assert_gpu_one': assert_gpu_one}
    exec(compile(native_definitions(SOURCE.read_text(encoding='utf-8')), str(SOURCE), 'exec'), env)
    env['refiner'] = env['Refiner']().to('cuda')
    checkpoint = torch.load(formal / 'ref_fold1.pt', map_location='cpu', weights_only=False)
    assert checkpoint['fold_excluded'] == 1
    env['refiner'].load_state_dict(checkpoint['model'])
    env['refiner'].eval()
    env['det'] = YOLO(str(formal / 'det4oof_f1_best.pt'))
    manifest = json.loads((formal / 'manifest.json').read_text())
    dataset = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
    reports = {}
    for split, stems, image_dir, reference in (
        ('val', manifest['val'], dataset / 'train/train_images', formal / 'pred_conf0.35.csv'),
        ('test', sorted(p.stem for p in (dataset / 'test/test_images').glob('*.jpeg')),
         dataset / 'test/test_images', previous / 'test/submission.csv'),
    ):
        print(f'[stage] {split}', flush=True)
        (output / 'stage').write_text(split)
        assert len(stems) == {'val': 141, 'test': 180}[split] and len(set(stems)) == len(stems)
        detections, replay = capture(env, stems, image_dir, torch)
        expected = {s: [] for s in stems}
        for row in pd.read_csv(reference).itertuples():
            expected[row.filament_id.rsplit('_', 1)[0]].append(row.segmentation_rle)
        assert replay == expected, f'{split}: conf.35 ownership replay differs from earlier export'
        boxes = json.loads((previous / ('dev_replay' if split == 'val' else 'test') / 'raw_boxes.json').read_text())
        assert {s: [d['box'] + [d['conf']] for d in ds] for s, ds in detections.items()} == boxes, split
        path = output / f'detections_{split}.json'
        path.write_text(json.dumps(detections))
        reports[split] = {'images': len(stems), 'detections': sum(map(len, detections.values())),
                          'conf_ge_030': sum(d['conf'] >= .30 for ds in detections.values() for d in ds),
                          'ref_gate_pass': sum(d['ref_gate'] for ds in detections.values() for d in ds),
                          'conf035_replay_instances': sum(map(len, replay.values())),
                          'conf035_replay_equal': True, 'boxes_equal': True,
                          'json_sha256': digest(path), 'gpu': assert_gpu_one()}
        print(f'[done] {split}: {reports[split]}', flush=True)
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before == after
    result = {'passed': True, 'thr': THR, 'gate': GATE, 'conf_floor': CONF_FLOOR,
              'yolo_mask': 'retina_masks data > .5', 'ref_mask': 'prob > THR & disk, pre-gate/ownership',
              'source_sha256': SOURCE_SHA, 'exporter_sha256': digest(Path(__file__)),
              'protected_hashes': before, 'gpu': gpu, 'reports': reports}
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    (output / 'stage').write_text('complete')


if __name__ == '__main__':
    main()
