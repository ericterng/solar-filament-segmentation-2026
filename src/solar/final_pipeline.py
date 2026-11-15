"""Final inference: repo weights -> the two final CSVs (notebooks/final_inference.ipynb calls only this module).

One pipeline, one switch. Both finals share the U-Net (weights/unet_effb4_cpstrong_s0.pt), the fusion
rules, the rescorer form and the instance refiner (weights/refiner_effb4_r0.pt); they differ only in the
detector set (`POOLS`) and the rescorer fitted for that set (weights/rescorer_<pool>.json):

- final1_A6 : cp_s43, m1280, m1024, det4, d1024, d1536  (Kaggle Public 0.40, csv sha256 2ba22b86...)
- final2_r0 : cp_s43, m1280, m1024                       (Kaggle Public 0.40, csv sha256 fafef71c...)

Stages, each a function below and timed by the notebook:
1. preprocess      : disk geometry + limb-darkening flattening of every test JPEG (upstream filament_seg.disk),
                     cached as flat PNG (uint8, clip(flat, 0, 2) * 127.5) + disk.json, exactly as upstream
                     scripts/preprocess.py writes them (the U-Net, the upstream-recipe detectors and the
                     refiner all read this cache).
2. unet_logits     : tiled (512, overlap 128, Hann-blended) dihedral-8 TTA logits, cached float16 per image
                     through upstream filament_seg.logit_cache.ensure_logits (the same code path that
                     produced the original cache).
3. detector_views  : every detector predicts four flipped copies (none, lr, ud, lr+ud) at its own imgsz with
                     conf .10, iou .60, agnostic NMS, retina masks; masks are flipped back and RLE-encoded.
                     cp_s43 / m1280 / m1024 read the raw JPEG (view 0 straight from the file path, as the
                     training-time runners did); det4 / d1024 / d1536 read the flat PNG.
4. select          : scripts/rescoring_probe_20261012.candidates (flip-merge, member-merge, fusion: U-Net
                     prob .5, close 3, grow 4 px, min_area 200, candidate floor .15, six features), then the
                     frozen rescorer keeps candidates with p * E[IoU] > PQ_train / 2.
5. refine          : scripts/apply_refiner_20261031.process_image (512 tiles, dihedral 8, prob .5, components
                     touching the original, 200 px fallback, first-come conflict resolution in fusion order).
6. write + compare : solar.submission.write_submission / validate_submission; byte and instance comparison
                     against the reference CSVs when they are available.

Everything numeric is pinned in code; nothing is tuned here. GPU strongly recommended; without one the code falls
back to CPU, which is slow and was not tested end to end.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'third_party/ameyypawar-filament-seg'
for p in (ROOT / 'scripts', UPSTREAM, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from solar.data import DEFAULT_DATA, read_manifest, sha256  # noqa: E402
from solar.submission import parse_prediction_id, validate_submission, write_submission  # noqa: E402

WEIGHTS = ROOT / 'weights'
MANIFEST = WEIGHTS / 'manifest.json'
POOLS = {'r43': ('cp_s43', 'm1280', 'm1024'),
         'A6': ('cp_s43', 'm1280', 'm1024', 'det4', 'd1024', 'd1536')}
FINALS = {'final1_A6': 'A6', 'final2_r0': 'r43'}
#: Where the original flip views live locally (parity checks and the rescorer export only; not needed to run).
DETECTOR_DIRS = {'cp_s43': ROOT / 'outputs/yolo_cp_s43_20261025', 'm1280': ROOT / 'outputs/extra_m1280_20261011',
                 'm1024': ROOT / 'outputs/extra_m1024_20261011', 'det4': ROOT / 'outputs/det4_amey_20261013',
                 'd1024': ROOT / 'outputs/det_amey_d1024_20261015', 'd1536': ROOT / 'outputs/det_amey_d1536_20261015'}
FLIPS = ((False, False), (True, False), (False, True), (True, True))
PREDICT = dict(conf=.10, iou=.60, agnostic_nms=True, retina_masks=True, verbose=False)
TILE, OVERLAP, TTA = 512, 128, 'dihedral'
FLAT_CLIP = 2.0
IMAGE_SHAPE = (2048, 2048)
IMAGE_MANIFEST = ROOT / 'outputs/data_audit/image_manifest.csv'


# ----------------------------------------------------------------------------- manifest / inputs
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def weight_entries() -> dict[str, dict]:
    """{member: manifest entry} for the detectors, plus 'unet' and 'refiner'."""
    out = {}
    for e in manifest()['weights']:
        out[e['member'] if e['role'] == 'detector' else e['role']] = e
    return out


def check_weights() -> dict[str, str]:
    """SHA-256 of every weight file must equal the manifest (also rejects unfetched git-lfs pointers)."""
    from verify_final_weights import verify  # noqa: E402  (scripts/)

    rows = verify(quiet=True)
    bad = [r for r in rows if not r['ok']]
    if bad:
        raise RuntimeError('weights not verified: ' + '; '.join(f"{r['file']}: {r['why']}" for r in bad))
    return {r['file']: r['why'] for r in rows}


def test_stems(data: Path = DEFAULT_DATA, image_manifest: Path = IMAGE_MANIFEST) -> list[str]:
    """The 180 official test images. With the audited manifest present, every file's SHA is checked too."""
    folder = data / 'test/test_images'
    names = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.jpeg')
    if image_manifest.exists():   # local audit file (outputs/, not in git): same file set and SHA-256 per image
        rows = [r for r in read_manifest(image_manifest) if r['split'] == 'test']
        if sorted(r['file_name'] for r in rows) != names:
            raise RuntimeError('test file set differs from the audited manifest')
        for r in rows:
            if sha256(folder / r['file_name']) != r['sha256']:
                raise RuntimeError(f"test image hash changed: {r['file_name']}")
    if len(names) != 180:
        raise RuntimeError(f'expected 180 test images, found {len(names)} in {folder}')
    return [Path(n).stem for n in names]


# ----------------------------------------------------------------------------- 1. preprocess
def _flatten_one(args: tuple[str, str, str]) -> tuple[str, dict]:
    from filament_seg.disk import detect_disk, flatten, read_image

    stem, image_path, flat_path = args
    image = read_image(image_path)
    disk = detect_disk(image)
    flat = flatten(image, disk)
    png = (np.clip(flat, 0.0, FLAT_CLIP) * (255.0 / FLAT_CLIP)).astype(np.uint8)
    cv2.imwrite(flat_path, png)
    return stem, {'cx': disk.cx, 'cy': disk.cy, 'radius': disk.radius}


def preprocess(stems: list[str], image_dir: Path, cache_dir: Path, workers: int = 4, force: bool = False) -> dict:
    """flat/<stem>.png + disk.json under cache_dir (upstream scripts/preprocess.py, test images only)."""
    flat_dir = cache_dir / 'flat'
    flat_dir.mkdir(parents=True, exist_ok=True)
    disk_path = cache_dir / 'disk.json'
    geometry = json.loads(disk_path.read_text(encoding='utf-8')) if disk_path.exists() else {}
    todo = [(s, str(image_dir / f'{s}.jpeg'), str(flat_dir / f'{s}.png')) for s in stems
            if force or s not in geometry or not (flat_dir / f'{s}.png').exists()]
    if todo:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for stem, geo in pool.map(_flatten_one, todo, chunksize=4):
                    geometry[stem] = geo
        else:
            for item in todo:
                stem, geo = _flatten_one(item)
                geometry[stem] = geo
        disk_path.write_text(json.dumps(geometry, indent=2), encoding='utf-8')
    return {'images': len(stems), 'computed': len(todo), 'cache_dir': str(cache_dir)}


# ----------------------------------------------------------------------------- 2. U-Net logits
def unet_logits(stems: list[str], checkpoint: Path, cache_dir: Path, logits_dir: Path, device, force: bool = False) -> dict:
    """Dihedral-8 tiled logits (float16 .npy per stem) through upstream ensure_logits; cache_meta.json records the model."""
    from filament_seg.dataset import load_disk_geometry
    from filament_seg.logit_cache import ensure_logits

    disks = load_disk_geometry(cache_dir / 'disk.json')
    ensure_logits(stems, [str(checkpoint)], disks, cache_dir / 'flat', logits_dir, TILE, OVERLAP, device, force, tta=TTA)
    return {'images': len(stems), 'tta': TTA, 'tile': TILE, 'overlap': OVERLAP, 'dtype': 'float16',
            'checkpoint_sha256': sha256(checkpoint)}


# ----------------------------------------------------------------------------- 3. detector views
def _rle_counts(mask: np.ndarray) -> str:
    return coco.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('ascii')


def detect(model, source, flip_lr: bool, flip_ud: bool, imgsz: int, device) -> list[dict]:
    """One YOLO view: predict, flip the masks back, RLE-encode (identical to the training-time runners)."""
    r = model.predict(source, imgsz=imgsz, device=device, **PREDICT)[0]
    if r.boxes is None or not len(r.boxes):
        return []
    assert r.masks is not None and len(r.masks.data) == len(r.boxes)
    masks = r.masks.data.cpu().numpy() > .5
    assert masks.shape[1:] == IMAGE_SHAPE, masks.shape
    if flip_ud:
        masks = masks[:, ::-1]
    if flip_lr:
        masks = masks[:, :, ::-1]
    return [{'box': [float(v) for v in box], 'conf': float(cf), 'counts': _rle_counts(m)}
            for box, cf, m in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), masks)]


def flip_views(model, stem: str, imgsz: int, input_kind: str, image_dir: Path, flat_dir: Path, device) -> list[list[dict]]:
    """Four views in FLIPS order. 'jpeg': view 0 from the file path, views 1-3 from flipped arrays (as the
    cp_s43 / m1280 / m1024 runners did); 'flat_png': all four from the flat PNG array (det4 / d1024 / d1536)."""
    if input_kind == 'jpeg':
        path = image_dir / f'{stem}.jpeg'
        image = cv2.imread(str(path))
        assert image is not None and image.shape[:2] == IMAGE_SHAPE, path
        per = [detect(model, str(path), False, False, imgsz, device)]
        for lr, ud in FLIPS[1:]:
            view = image[:, ::-1] if lr else image
            per.append(detect(model, np.ascontiguousarray(view[::-1] if ud else view), lr, ud, imgsz, device))
        return per
    image = cv2.imread(str(flat_dir / f'{stem}.png'))
    assert image is not None and image.shape[:2] == IMAGE_SHAPE, stem
    per = []
    for lr, ud in FLIPS:
        view = image[:, ::-1] if lr else image
        per.append(detect(model, np.ascontiguousarray(view[::-1] if ud else view), lr, ud, imgsz, device))
    return per


def detector_views(stems: list[str], members: tuple, image_dir: Path, cache_dir: Path, views_dir: Path, device,
                   force: bool = False, log=print) -> dict:
    """views_dir/<member>_flip_views_test.json per detector (reused when it covers `stems`); returns timings and counts."""
    from ultralytics import YOLO

    entries = weight_entries()
    views_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for m in members:
        out = views_dir / f'{m}_flip_views_test.json'
        e = entries[m]
        if out.exists() and not force:
            views = json.loads(out.read_text())
            if set(stems) <= set(views):   # a cache from a smaller run (e.g. LIMIT = 2) is rebuilt instead
                report[m] = {'cached': True, 'images': len(views)}
                continue
        started = time.perf_counter()
        model = YOLO(str(WEIGHTS / e['file']))
        views = {}
        for n, stem in enumerate(stems, 1):
            views[stem] = flip_views(model, stem, e['imgsz'], e['input'], image_dir, cache_dir / 'flat', device)
            if n % 30 == 0 or n == len(stems):
                log(f'  [{m}] {n}/{len(stems)} {time.perf_counter() - started:.0f}s')
        out.write_text(json.dumps(views))
        report[m] = {'cached': False, 'images': len(views), 'imgsz': e['imgsz'], 'input': e['input'],
                     'seconds': time.perf_counter() - started, 'weight_sha256': e['sha256'],
                     'detections_per_view': [sum(len(v[k]) for v in views.values()) for k in range(4)],
                     'json_sha256': sha256(out)}
        del model
    return report


# ----------------------------------------------------------------------------- 4. select (fusion + frozen rescorer)
class FrozenRescorer:
    """The logistic rescorer, with the fitted numbers frozen in weights/rescorer_<pool>.json (numpy only).

    expected = sigmoid(coef . (x - mean) / scale + intercept) * E[IoU | area bin];  keep iff expected > threshold
    (threshold = PQ_train / 2, the PQ-optimal rule). Equivalent to sklearn's StandardScaler + LogisticRegression
    predict_proba to ~1e-15; scripts/export_rescorer_params_20261107.py asserts the kept sets are identical.
    """

    def __init__(self, params: dict):
        self.params = params
        self.features = tuple(params['features'])
        self.mean = np.asarray(params['scaler_mean'], dtype=np.float64)
        self.scale = np.asarray(params['scaler_scale'], dtype=np.float64)
        self.coef = np.asarray(params['coef'], dtype=np.float64)
        self.intercept = float(params['intercept'])
        self.e_iou = np.asarray(params['e_iou_by_bin'], dtype=np.float64)
        self.area_bins = tuple(params['area_bins'])
        self.threshold = float(params['threshold'])

    @classmethod
    def load(cls, path: Path) -> 'FrozenRescorer':
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def expected(self, X: np.ndarray, area: np.ndarray) -> np.ndarray:
        z = ((np.asarray(X, dtype=np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        p = 1.0 / (1.0 + np.exp(-z))
        return p * self.e_iou[np.digitize(area, self.area_bins)]

    def keep(self, X: np.ndarray, area: np.ndarray) -> np.ndarray:
        return self.expected(X, area) > self.threshold


def select_instances(stems: list[str], members: tuple, logits_dir: Path, views_dir: Path, cache_dir: Path,
                     rescorer: FrozenRescorer, log=print) -> tuple[dict[str, list[dict]], list[dict]]:
    """Fusion candidates per test image, kept by the frozen rescorer; RLEs in fusion order."""
    from filament_seg.dataset import load_disk_geometry
    from rescoring_probe_20261012 import candidates
    from fusion_probe_multiscale_20261012 import arrays

    views = {m: json.loads((views_dir / f'{m}_flip_views_test.json').read_text()) for m in members}
    assert all(set(v) >= set(stems) for v in views.values()), 'views missing for some test images'
    disks = load_disk_geometry(cache_dir / 'disk.json')
    preds, rows = {}, []
    for i, stem in enumerate(stems, 1):
        logits = np.load(logits_dir / f'{stem}.npy').astype(np.float32)
        found = candidates(stem, logits, disks[stem], [views[m][stem] for m in members])
        if found:
            X, area, _ = arrays(found)
            keep = rescorer.keep(X, area)
        else:
            keep = np.zeros(0, bool)
        preds[stem] = [d['rle'] for d, k in zip(found, keep) if k]
        rows.append({'image': stem, 'candidates': len(found), 'kept': len(preds[stem])})
        if i % 30 == 0 or i == len(stems):
            log(f'  select {i}/{len(stems)} candidates={sum(r["candidates"] for r in rows)} kept={sum(r["kept"] for r in rows)}')
    return preds, rows


def instances_blob(preds: dict[str, list[dict]], meta: dict) -> dict:
    """apply_refiner input: CSV row order per image == fusion order; every row kept."""
    import refiner_core_20261031 as core

    images = [{'stem': stem, 'instances': [{'order': i, 'rle': core.rle_json(r), 'kept': True} for i, r in enumerate(masks)]}
              for stem, masks in sorted(preds.items()) if masks]
    return {'meta': {**meta, 'images_with_rows': len(images), 'instances': sum(len(im['instances']) for im in images)},
            'images': images}


# ----------------------------------------------------------------------------- 5. refine
def refine(preds: dict[str, list[dict]], cache_dir: Path, checkpoint: Path, device, batch: int = 16, log=print
           ) -> tuple[dict[str, list[dict]], dict]:
    """Apply the instance refiner to every kept instance (count and order unchanged)."""
    import torch
    from apply_refiner_20261031 import load_refiners, process_image
    from filament_seg.dataset import load_disk_geometry
    import refiner_core_20261031 as core

    model, metas = load_refiners([checkpoint], torch.device(device) if isinstance(device, str) else device)
    assert metas[0]['train_half'] == 'all' and metas[0]['probe'] == 28, metas
    disks = load_disk_geometry(cache_dir / 'disk.json')
    blob = instances_blob(preds, {'source': 'final_pipeline.select_instances'})
    out_images, started = [], time.perf_counter()
    for n, image in enumerate(blob['images'], 1):
        out_images.append(process_image(image, model, cache_dir, disks, torch.device(device) if isinstance(device, str) else device,
                                        batch, identity=False))
        if n % 20 == 0 or n == len(blob['images']):
            log(f'  refine {n}/{len(blob["images"])} {time.perf_counter() - started:.0f}s')
    refined = {s: [] for s in preds}
    for im in out_images:
        refined[im['stem']] = [core.rle_bytes(i['rle']) for i in im['instances']]
    inst = [i for im in out_images for i in im['instances']]
    assert {s: len(m) for s, m in refined.items()} == {s: len(m) for s, m in preds.items()}
    return refined, {'instances': len(inst), 'changed': int(sum(i['changed'] for i in inst)),
                     'fallback': int(sum(i['fallback'] for i in inst)), 'seconds': time.perf_counter() - started,
                     'checkpoint_sha256': sha256(checkpoint), 'refiner_meta': metas[0]}


# ----------------------------------------------------------------------------- 6. write + compare
def write_csv(preds: dict[str, list[dict]], path: Path) -> dict:
    write_submission(path, preds)
    checks = validate_submission(path, {s: len(m) for s, m in preds.items()})
    checks.update({'csv_sha256': sha256(path), 'csv_bytes': path.stat().st_size, 'instances': sum(map(len, preds.values()))})
    return checks


def read_csv_rles(path: Path) -> dict[str, list[dict]]:
    import csv
    out: dict[str, list[dict]] = {}
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for row in csv.DictReader(handle):
            out.setdefault(parse_prediction_id(row['filament_id']), []).append(
                {'size': list(IMAGE_SHAPE), 'counts': row['segmentation_rle'].encode('ascii')})
    return out


def compare_csv(path: Path, reference: Path, stems: list[str] | None = None) -> dict:
    """Byte equality, instance counts, exact-RLE matches and the IoU > .5 match share against a reference CSV.
    `stems` restricts both sides to a subset (smoke runs); byte equality is then reported as None."""
    new, ref = read_csv_rles(path), read_csv_rles(reference)
    if stems is not None:
        keep = set(stems)
        new, ref = {s: m for s, m in new.items() if s in keep}, {s: m for s, m in ref.items() if s in keep}
    exact = matched = 0
    for stem, masks in ref.items():
        cand = new.get(stem, [])
        exact += sum(1 for r in masks if any(r['counts'] == c['counts'] for c in cand))
        if masks and cand:
            iou = np.asarray(coco.iou(masks, cand, [0] * len(cand)))
            matched += int((iou.max(axis=1) > .5).sum())
    n_ref, n_new = sum(map(len, ref.values())), sum(map(len, new.values()))
    return {'reference': str(reference), 'reference_sha256': sha256(reference), 'csv_sha256': sha256(path),
            'byte_equal': (Path(path).read_bytes() == Path(reference).read_bytes()) if stems is None else None,
            'reference_instances': n_ref, 'instances': n_new, 'exact_rle_matches': exact,
            'matched_iou_gt_05': matched, 'match_share': matched / max(n_ref, 1),
            'images_with_different_count': sum(1 for s in set(ref) | set(new) if len(ref.get(s, [])) != len(new.get(s, [])))}


# ----------------------------------------------------------------------------- all stages on any image folder
def predict(stems: list[str], image_dir: Path, work_dir: Path, final: str, device, workers: int = 4,
            refiner_batch: int = 16, log=print) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Stages 1-5 for one final on any folder of 2048 x 2048 H-alpha JPEGs (e.g. the validation images of the
    training set): returns (refined, unrefined) `{stem: [RLE, ...]}`. Intermediate files are cached under
    `work_dir`, so use a directory of its own per image set. The same calls, in the same order, as the notebook's
    test-set cells; it uses the frozen rescorer and the shipped refiner, so on fold1 validation images those two
    stages are in-sample (they were fitted on these images) while the U-Net and the detectors are not."""
    import torch

    members = POOLS[FINALS[final]]
    cache, logits, views = work_dir / 'cache', work_dir / 'logits_unet', work_dir / 'views'
    entries = weight_entries()
    preprocess(stems, image_dir, cache, workers=workers)
    unet_logits(stems, WEIGHTS / entries['unet']['file'], cache, logits, torch.device(device))
    detector_views(stems, members, image_dir, cache, views, device, log=log)
    rescorer = FrozenRescorer.load(WEIGHTS / manifest()['finals'][final]['rescorer'])
    unrefined, _ = select_instances(stems, members, logits, views, cache, rescorer, log=log)
    refined, _ = refine(unrefined, cache, WEIGHTS / entries['refiner']['file'], device, batch=refiner_batch, log=log)
    return refined, unrefined


# ----------------------------------------------------------------------------- environment
def environment(device) -> dict:
    import torch
    info = {'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            'device': str(device), 'cpu_count': os.cpu_count(),
            'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32}
    if str(device).startswith('cuda') and torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    for name in ('numpy', 'cv2', 'pycocotools', 'segmentation_models_pytorch', 'timm', 'ultralytics', 'albumentations', 'sklearn'):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, '__version__', 'present')
        except ImportError:
            info[name] = None
    return info
