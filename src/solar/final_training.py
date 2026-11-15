"""Training stage of the demo notebook (optional): how every file in weights/ was produced, and code to redo it.

The notebook loads weights/ by default. With `TRAIN = True` it calls `retrain`, which runs the recipes below and
writes a new set of weights (same file names) to outputs/retrain/weights/. Steps, each needing the ones above it:

0. inputs      fold1 split by observation date (`fold1_split`, the rule of the pinned upstream source), the
               preprocessing cache (upstream scripts/preprocess.py) and the two public starting points
               (ImageNet EfficientNet-B4, COCO yolo11m-seg; pinned by SHA-256).
1. unet        scripts/train_unet_cpstrong_20261025.py as a subprocess (boundary-weighted BCE + Dice,
               1-3 pasted filaments per crop, 30 epochs, seed 0, last checkpoint).
2. detectors   six yolo11m-seg models in two recipes:
               cp_s43 / m1280 / m1024  the pinned upstream source src/solar/vendor/heshen/train.py through its
                                       detector stage (one medoid annotator per image, raw JPEG, 50 epochs, best.pt);
               det4 / d1024 / d1536    the upstream ameyypawar recipe: scripts/run_detector_amey_20261015.build_dataset
                                       (one sample per annotator, limb-flattened PNG) and its training arguments
                                       (60 epochs, no mosaic, last.pt).
3. candidates  U-Net + detectors on the 141 validation images -> fusion candidates with their six features
               (solar.final_pipeline stages + scripts/rescoring_probe_20261012.candidates).
4. rescorer    logistic match probability x expected IoU, kept above PQ_train / 2
               (scripts/rescoring_probe_20261012.Rescorer), frozen to JSON for solar.final_pipeline.FrozenRescorer.
5. refiner     training pairs (candidate, best annotation with IoU >= .25) from the 3-detector candidates
               (scripts/prep_refiner_samples_20261031.build_samples), then scripts/train_refiner_20261031.py
               (initialised from the U-Net, 16 epochs, all 141 validation images, last checkpoint).

Why this module exists next to the dated runner scripts: the runners that produced the shipped weights are kept in
scripts/ with their original code, but they pin the GPU of the host they ran on,
assert the SHA of host-local files and always continue into validation / test inference. The functions here call the
same training code and pass the same arguments, with the data folder, the work folder and the device as parameters.
tests/test_final_training.py checks that equivalence without a GPU (identical executed source, identical training
arguments, identical split, rescorer and refiner samples reproduced from the recorded candidates).

Retraining does not reproduce the shipped files bit for bit: the U-Net and the ameyypawar detectors train without a
global torch seed / with deterministic=False, cuDNN and AMP are not bit-exact across cards, and the shipped weights
come from two GPU types. Expect scores within the seed noise (about +-.005 PQ).

Smoke run (`retrain(..., smoke=True)`): every step runs once with the original runners' own smoke settings, only to
show that the code path works on a machine; the files it writes are not usable models. U-Net and refiner: the
scripts' `--smoke` flag; HeShen detectors: the runners' smoke edit (8 training / 2 validation images, 1 epoch);
ameyypawar detectors: the runner's smoke mode (8 annotator records, 1 epoch). The validation candidates behind the
rescorer and the refiner come from the shipped first-stage weights (smoke-trained detectors find nothing), so the
smoke rescorer can be compared with the shipped one. Output goes to `SMOKE_OUT`, never to `OUT`.

Requirements: requirements.txt. The U-Net step keeps the original script's guards, so it needs Linux, one
NVIDIA GPU visible through nvidia-smi and the cache / split at the paths that script fixes (`UNET_CACHE`,
`UNET_SPLIT`). The U-Net and the refiner run as subprocesses on the GPU this process may use (the
`gpu_index`-th entry of CUDA_VISIBLE_DEVICES, passed to them by UUID). Recorded cost of the original runs is in `RECIPES`: about 20 GPU-hours on the cards used, 18 of
them for the three detectors trained on a GTX 1080 Ti, plus the U-Net and d1536 whose run time was not recorded.
Measured 2026-11-13: retrain() with the defaults ran all 10 steps on one RTX A5000 in 6.7 h (U-Net 47 min, the six
detectors 29-89 min each, candidates 24 min, refiner 7 min), at most 12.9 GB of GPU memory.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'scripts'
UPSTREAM = ROOT / 'third_party/ameyypawar-filament-seg'
DATA = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
OUT = ROOT / 'outputs/retrain'
SMOKE_OUT = ROOT / 'outputs/retrain_smoke'   # smoke runs only; retrain refuses to mix the two folders
# Fixed by scripts/train_unet_encoder_20261015.py (CACHE, SPLITS); the U-Net script reads these module constants.
UNET_CACHE = ROOT / 'outputs/amey_unet_cache'
UNET_SPLIT = ROOT / 'outputs/amey_unet_fold1_20261010/splits.json'
HF_HOME = ROOT / 'hf_home'
YOLO_PRETRAINED = ROOT / 'outputs/pretrained/yolo11m-seg.pt'
YOLO_PRETRAINED_URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt'
YOLO_PRETRAINED_SHA = 'eb9a06f63e2206c35d68d839b08c362429ebecf933ad54c1ad68b2fd001c17cf'
ENCODER_REPO = ('smp-hub/efficientnet-b4.imagenet', '05cd5dde5dab658f00c463f9b9aa0ced76784f40',
                'c465ce05723692769f0d069312aaf8fcdf9a6f5153e48127f5b7eae68e4ec1dd')   # repo, revision, sha256
FOLD, SPLIT_SEED, N_FOLDS = 1, 42, 5
HALF_SEED = 20261012          # rescoring_probe_20261012.SEED: date permutation behind the two validation halves
REFINER_POOL = 'r43'          # the shipped refiner was trained on the 3-detector candidates and serves both settings

# cp_s43 / m1280 / m1024: what changes in the pinned source's det.train(...) call; everything else is that call.
HESHEN = {'cp_s43': {'imgsz': 1536, 'seed': 43, 'name': 'det4oof_f1', 'copy_paste': 0.5},
          'm1280': {'imgsz': 1280, 'seed': 44, 'name': 'det_m1280_f1', 'copy_paste': None},
          'm1024': {'imgsz': 1024, 'seed': 45, 'name': 'det_m1024_f1', 'copy_paste': None}}
HESHEN_TRAIN_CALL_END = 'flipud=0.5, mosaic=0.25, close_mosaic=5)'
# det4 / d1024 / d1536: the keyword arguments of model.train in scripts/run_detector_amey_20261015.py
# (and, identically, scripts/run_detector4_20261013.py); imgsz and seed per member.
AMEY = {'det4': {'imgsz': 1280, 'seed': 46}, 'd1024': {'imgsz': 1024, 'seed': 47}, 'd1536': {'imgsz': 1536, 'seed': 48}}
AMEY_TRAIN = dict(epochs=60, batch=2, workers=4, patience=1000, exist_ok=True, deterministic=False, single_cls=True,
                  cos_lr=True, plots=False, val=False, mosaic=0.0, close_mosaic=0, mixup=0.0, copy_paste=0.0,
                  fliplr=0.5, flipud=0.5, degrees=0.0, shear=0.0, perspective=0.0, translate=0.05, scale=0.2,
                  hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, overlap_mask=True, mask_ratio=4)
DETECTORS = (*HESHEN, *AMEY)
STEPS = ('inputs', 'unet', *DETECTORS, 'rescorer', 'refiner')
# Smoke settings = the original runners' smoke modes (scripts/run_extra_detectors_20261011.member_source,
# scripts/run_yolo_copypaste_20261025.member_source, scripts/run_detector_amey_20261015.main).
HESHEN_SMOKE_EDITS = (('for split in ("train", "val"):',
                       'train_stems = train_stems[:8]\nval_stems = val_stems[:2]\nfor split in ("train", "val"):'),
                      ('epochs=50,', 'epochs=1,'))
AMEY_SMOKE = {'records': 8, 'epochs': 1}

# What was run for the shipped files. `recorded` quotes the run logs (outputs/<run>/runtime.json, not in git) and
# the experiment logs (not in this repository); None = not recorded.
RECIPES = {
    'unet': {'file': 'unet_effb4_cpstrong_s0.pt', 'script': 'scripts/train_unet_cpstrong_20261025.py --seed 0',
             'data': '566 fold1 training images, one annotator per image per epoch, 512 px crops',
             'settings': 'EfficientNet-B4 U-Net (ImageNet encoder), 30 epochs, batch 16, AdamW 1e-3 cosine, AMP, '
                         'boundary-weighted BCE (band 2 px, weight 4) + Dice, 1-3 pasted filaments per crop, last checkpoint',
             'recorded': {'gpu': 'RTX A5000', 'hours': None, 'peak_gb': 12.2}},
    'cp_s43': {'file': 'yolo11m_seg_cp_s43_1536.pt', 'script': 'scripts/run_yolo_copypaste_20261025.py s43 train',
               'data': '566 training images, medoid annotator, raw JPEG',
               'settings': 'yolo11m-seg (COCO), imgsz 1536, 50 epochs, batch 2, mosaic .25, copy_paste .5, seed 43, best.pt',
               'recorded': {'gpu': 'RTX A5000', 'hours': 1.0, 'peak_gb': 12.1}},
    'm1280': {'file': 'yolo11m_seg_m1280.pt', 'script': 'scripts/run_extra_detectors_20261011.py m1280 train',
              'data': '566 training images, medoid annotator, raw JPEG',
              'settings': 'yolo11m-seg (COCO), imgsz 1280, 50 epochs, batch 2, mosaic .25, seed 44, best.pt',
              'recorded': {'gpu': 'GTX 1080 Ti', 'hours': 5.4, 'peak_gb': 5.8}},
    'm1024': {'file': 'yolo11m_seg_m1024.pt', 'script': 'scripts/run_extra_detectors_20261011.py m1024 train',
              'data': '566 training images, medoid annotator, raw JPEG',
              'settings': 'yolo11m-seg (COCO), imgsz 1024, 50 epochs, batch 2, mosaic .25, seed 45, best.pt',
              'recorded': {'gpu': 'GTX 1080 Ti', 'hours': 3.7, 'peak_gb': 5.7}},
    'det4': {'file': 'yolo11m_seg_det4_1280.pt', 'script': 'scripts/run_detector4_20261013.py train',
             'data': '926 annotator records of the 566 training images, limb-flattened PNG',
             'settings': 'yolo11m-seg (COCO), imgsz 1280, 60 epochs, batch 2, no mosaic, cosine LR, seed 46, last.pt',
             'recorded': {'gpu': 'GTX 1080 Ti', 'hours': 8.8, 'peak_gb': 5.6}},
    'd1024': {'file': 'yolo11m_seg_d1024.pt', 'script': 'scripts/run_detector_amey_20261015.py d1024 train',
              'data': '926 annotator records of the 566 training images, limb-flattened PNG',
              'settings': 'yolo11m-seg (COCO), imgsz 1024, 60 epochs, batch 2, no mosaic, cosine LR, seed 47, last.pt',
              'recorded': {'gpu': 'RTX A5000', 'hours': 0.9, 'peak_gb': 11.9}},
    'd1536': {'file': 'yolo11m_seg_d1536.pt', 'script': 'scripts/run_detector_amey_20261015.py d1536 train',
              'data': '926 annotator records of the 566 training images, limb-flattened PNG',
              'settings': 'yolo11m-seg (COCO), imgsz 1536, 60 epochs, batch 2, no mosaic, cosine LR, seed 48, last.pt',
              'recorded': {'gpu': 'RTX A5000', 'hours': None, 'peak_gb': 12.5}},
    'rescorer': {'file': 'rescorer_A6.json / rescorer_r43.json', 'script': 'scripts/export_rescorer_params_20261107.py',
                 'data': 'all fusion candidates (score >= .15) of the 141 validation images, one row per annotator record',
                 'settings': 'StandardScaler + LogisticRegression(C=1, lbfgs) on six features; E[IoU] per area bin; '
                             'keep iff p x E[IoU] > PQ_train / 2',
                 'recorded': {'gpu': 'CPU', 'hours': 0.0, 'peak_gb': None}},
    'refiner': {'file': 'refiner_effb4_r0.pt', 'script': 'scripts/train_refiner_20261031.py --train-half all --refiner-seed 0',
                'data': '1398 (candidate, annotation) pairs with IoU >= .25 from the 141 validation images (3-detector candidates)',
                'settings': '3-channel EfficientNet-B4 U-Net initialised from the U-Net, 16 epochs, batch 8, AdamW 1e-4 cosine, '
                            'boundary-weighted BCE + Dice, seed 0, last checkpoint',
                'recorded': {'gpu': 'RTX A5000', 'hours': 0.11, 'peak_gb': 6.4}},
}


def _scripts_on_path() -> None:
    for p in (SCRIPTS, UPSTREAM, ROOT / 'src'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def describe() -> list[dict]:
    """One row per trained file for the notebook: what it is trained on, with what, and what the original run cost."""
    rows = []
    for name, r in RECIPES.items():
        rec = r['recorded']
        cost = [rec['gpu'], 'time not recorded' if rec['hours'] is None else '{:g} h'.format(rec['hours'])]
        if rec['peak_gb']:
            cost.append('peak {:g} GB'.format(rec['peak_gb']))
        rows.append({'step': name, 'file': r['file'], 'data': r['data'], 'settings': r['settings'],
                     'original run': ', '.join(cost), 'runner': r['script']})
    return rows


# ----------------------------------------------------------------------------- 0. inputs
def fold1_split(stems) -> tuple[list[str], list[str]]:
    """(train, validation) image stems. Observation dates (first 8 characters) are sorted, shuffled with
    random.Random(42) and dealt into five folds; fold 1 is validation. Every annotator of an observation, and
    every observation of a date, therefore stays on one side. Same rule as src/solar/vendor/heshen/train.py."""
    stems = sorted(set(stems))
    days = sorted({s[:8] for s in stems})
    random.Random(SPLIT_SEED).shuffle(days)
    fold = {day: i % N_FOLDS for i, day in enumerate(days)}
    return [s for s in stems if fold[s[:8]] != FOLD], [s for s in stems if fold[s[:8]] == FOLD]


def annotation_stems(data: Path = DATA) -> list[str]:
    raw = json.loads((data / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text(encoding='utf-8'))
    return sorted({Path(r['file_name']).stem for r in raw['images']})


def unet_split(data: Path = DATA) -> dict:
    """The upstream U-Net's splits.json for fold1: annotator-record ids, every record following its observation
    (the content scripts/prepare_amey_unet_fold1.py wrote)."""
    _scripts_on_path()
    from filament_seg.data import load_annotations
    annotations = load_annotations(data / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    _, val_stems = fold1_split(r.stem for r in annotations.images.values())
    val_stems = set(val_stems)
    train = sorted(i for i, r in annotations.images.items() if r.stem not in val_stems)
    val = sorted(i for i, r in annotations.images.items() if r.stem in val_stems)
    assert not {annotations.images[i].stem[:8] for i in train} & {s[:8] for s in val_stems}, 'date leak'
    return {'group_by': 'heshen_author_fold1_date', 'seed': SPLIT_SEED, 'val_fraction': None, 'n_groups': None,
            'train': train, 'val': val}


def fetch_pretrained(yolo: Path = YOLO_PRETRAINED, hf_home: Path = HF_HOME) -> dict:
    """Download the two public starting points when missing and check their pinned SHA-256."""
    if not yolo.exists():
        import urllib.request
        yolo.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YOLO_PRETRAINED_URL, yolo)
    if sha256(yolo) != YOLO_PRETRAINED_SHA:
        raise RuntimeError(f'{yolo}: sha256 differs from the pinned yolo11m-seg.pt')
    from huggingface_hub import hf_hub_download
    repo, revision, expected = ENCODER_REPO
    # segmentation_models_pytorch reads config.json before model.safetensors (encoders.get_encoder); without it the
    # offline U-Net / refiner fall back to downloading the original .pth from GitHub.
    hf_hub_download(repo, filename='config.json', revision=revision, cache_dir=hf_home / 'hub')
    encoder = Path(hf_hub_download(repo, filename='model.safetensors', revision=revision, cache_dir=hf_home / 'hub'))
    if sha256(encoder) != expected:
        raise RuntimeError(f'{repo}@{revision}: sha256 differs from the pinned encoder')
    return {'yolo11m-seg': str(yolo), 'efficientnet-b4.imagenet': str(encoder)}


def prepare_inputs(data: Path = DATA, workers: int = 4, python: str = sys.executable) -> dict:
    """Preprocessing cache for the 707 training and 180 test images (flat PNG, masks, disk geometry; upstream
    scripts/preprocess.py, CPU) and the fold1 split file, at the paths the U-Net script fixes."""
    subprocess.run([python, 'scripts/preprocess.py', '--train-images', str(data / 'train/train_images'),
                    '--test-images', str(data / 'test/test_images'),
                    '--annotations', str(data / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),
                    '--cache-dir', str(UNET_CACHE), '--workers', str(workers)], cwd=UPSTREAM, check=True)
    split = unet_split(data)
    if UNET_SPLIT.exists():
        assert json.loads(UNET_SPLIT.read_text(encoding='utf-8')) == split, f'{UNET_SPLIT} differs from fold1'
    else:
        UNET_SPLIT.parent.mkdir(parents=True, exist_ok=True)
        UNET_SPLIT.write_text(json.dumps(split, indent=2), encoding='utf-8')
    return {'cache': str(UNET_CACHE), 'split': str(UNET_SPLIT), 'train_records': len(split['train']),
            'val_records': len(split['val'])}


# ----------------------------------------------------------------------------- 1. U-Net
def gpu_table() -> dict[str, str]:
    """{nvidia-smi index: UUID} of the physical GPUs (nvidia-smi ignores CUDA_VISIBLE_DEVICES)."""
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    return {i.strip(): u.strip() for i, u in (line.split(', ', 1) for line in rows.strip().splitlines())}


def gpu_uuid(index: int, visible: str | None = None, table: dict[str, str] | None = None) -> str:
    """UUID of the `index`-th GPU this process may use, i.e. of the device torch calls cuda:<index>: the index-th
    entry of CUDA_VISIBLE_DEVICES (an nvidia-smi index or a UUID) when it is set, else nvidia-smi's GPU <index>.
    Subprocesses get this UUID, so they run on the same physical card as the parent."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    table = gpu_table() if table is None else table
    if visible:
        entry = [v.strip() for v in visible.split(',') if v.strip()][index]
        return entry if entry.startswith('GPU-') else table[entry]
    return table[str(index)]


def unet_command(out: Path, uuid: str, data: Path = DATA, seed: int = 0, hf_home: Path = HF_HOME,
                 python: str = sys.executable, smoke: bool = False) -> tuple[list[str], dict]:
    """Command and environment of the original U-Net run (scripts/launch_unet_cpstrong_20261025.sh), with the GPU
    UUID, data folder and output folder of this machine. `smoke` adds the script's own --smoke flag."""
    env = {'CUDA_VISIBLE_DEVICES': uuid, 'UNET_ENCODER_GPU_UUID': uuid, 'HF_HOME': str(hf_home), 'HF_HUB_OFFLINE': '1',
           'FILAMENT_DATA_ROOT': str(data), 'FILAMENT_OUTPUT_ROOT': str(out.parent), 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4'}
    cmd = [python, 'scripts/train_unet_cpstrong_20261025.py', '--seed', str(seed), '--out', str(out)]
    return (cmd + ['--smoke'] if smoke else cmd), env


def train_unet(out: Path, gpu_index: int = 0, data: Path = DATA, hf_home: Path = HF_HOME, uuid: str | None = None,
               smoke: bool = False) -> Path:
    """Run the original U-Net training script on one NVIDIA GPU (`uuid`, else `gpu_uuid(gpu_index)`); returns its
    last checkpoint (the one shipped)."""
    assert UNET_SPLIT.exists() and (UNET_CACHE / 'disk.json').exists(), 'run prepare_inputs first'
    cmd, env = unet_command(out, uuid or gpu_uuid(gpu_index), data, hf_home=hf_home, smoke=smoke)
    subprocess.run(cmd, cwd=ROOT, env={**os.environ, **env}, check=True)
    return out / 'model_last.pt'


# ----------------------------------------------------------------------------- 2. detectors
def heshen_source(member: str, work: Path, data_root: Path = ROOT / 'data', pretrained: Path = YOLO_PRETRAINED,
                  device: str = 'cuda:0', smoke: bool = False) -> str:
    """The pinned upstream source through its detector stage, with this member's image size, seed, run name and
    copy-paste setting. With the default arguments the text equals what the original runner executed
    (scripts/run_extra_detectors_20261011.member_source, scripts/run_yolo_copypaste_20261025.member_source);
    `smoke` applies those runners' smoke edit (8 training / 2 validation images, 1 epoch)."""
    _scripts_on_path()
    from run_author_fold import portable_source, replace_once
    cfg = HESHEN[member]
    source = portable_source(work)
    source = source[:source.index('print("[det] done", flush=True)')] + 'print("[det] done", flush=True)\n'
    if cfg['imgsz'] != 1536:
        source = replace_once(source, 'imgsz=1536, batch=2', f"imgsz={cfg['imgsz']}, batch=2")
    source = replace_once(source, 'seed=42 + FOLD', f"seed={cfg['seed']}")
    if cfg['name'] != 'det4oof_f1':
        assert source.count('f"det4oof_f{FOLD}"') == 3
        source = source.replace('f"det4oof_f{FOLD}"', repr(cfg['name']))
        source = replace_once(source, 'f"det4oof_f{FOLD}_best.pt"', repr(f"{cfg['name']}_best.pt"))
    if cfg['copy_paste'] is not None:
        source = replace_once(source, HESHEN_TRAIN_CALL_END, HESHEN_TRAIN_CALL_END[:-1] + f", copy_paste={cfg['copy_paste']})")
    if Path(data_root) != ROOT / 'data':
        source = replace_once(source, f'INPUT_ROOT = Path({str(ROOT / "data")!r})', f'INPUT_ROOT = Path({str(data_root)!r})')
    if Path(pretrained) != ROOT / 'outputs/pretrained/yolo11m-seg.pt':
        source = replace_once(source, repr(str(ROOT / 'outputs/pretrained/yolo11m-seg.pt')), repr(str(pretrained)))
    if device != 'cuda:0':
        source = replace_once(source, 'torch.device("cuda:0")', f'torch.device({device!r})')
    if smoke:
        for old, new in HESHEN_SMOKE_EDITS:
            source = replace_once(source, old, new)
    return source


def train_heshen_detector(member: str, work: Path, data_root: Path = ROOT / 'data', pretrained: Path = YOLO_PRETRAINED,
                          device: str = 'cuda:0', smoke: bool = False) -> Path:
    """Build the dataset and train one of cp_s43 / m1280 / m1024 by executing `heshen_source`; returns best.pt.
    `data_root` is any folder that contains the competition's train_images/ and annotation JSON exactly once."""
    import torch
    work.mkdir(parents=True, exist_ok=False)
    source = heshen_source(member, work, data_root, pretrained, device, smoke=smoke)
    (work / 'executed_source.py').write_text(source, encoding='utf-8')

    def save_manifest(g: dict) -> None:
        train, val = fold1_split(g['stems_all'])
        assert (g['train_stems'], g['val_stems']) == (train, val), 'the source split differs from fold1_split'
        (work / 'manifest.json').write_text(json.dumps({'train': train, 'val': val}, indent=2), encoding='utf-8')

    def attach_checks(det) -> None:
        def started(trainer) -> None:   # as the original runner: keep trainer and validator on the requested device
            trainer.args.device = torch.device(device)
            trainer.validator.args.device = torch.device(device)

        def batch(trainer) -> None:
            assert trainer.batch_size == 2 and torch.isfinite(trainer.loss).all()

        det.add_callback('on_pretrain_routine_end', started)
        det.add_callback('on_train_batch_end', batch)

    env = {'__name__': '__main__', 'save_manifest': save_manifest, 'attach_checks': attach_checks}
    exec(compile(source, str(work / 'executed_source.py'), 'exec'), env)
    return work / f"{HESHEN[member]['name']}_best.pt"


def amey_train_arguments(member: str, smoke: bool = False) -> dict:
    """model.train keyword arguments of the ameyypawar runner for this member, except data / device / project / name."""
    arguments = {'imgsz': AMEY[member]['imgsz'], 'seed': AMEY[member]['seed'], **AMEY_TRAIN}
    if smoke:
        arguments['epochs'] = AMEY_SMOKE['epochs']
    return arguments


def train_amey_detector(member: str, work: Path, data: Path = DATA, flat_dir: Path = UNET_CACHE / 'flat',
                        pretrained: Path = YOLO_PRETRAINED, device: str = 'cuda:0', smoke: bool = False) -> Path:
    """Build the dataset and train one of det4 / d1024 / d1536; returns last.pt. `work` must be on the same drive
    as `flat_dir` (the dataset builder hard-links the flattened images)."""
    _scripts_on_path()
    import numpy as np
    import torch
    import run_detector_amey_20261015 as amey
    from ultralytics import YOLO
    work.mkdir(parents=True, exist_ok=False)
    train, _ = fold1_split(annotation_stems(data))
    amey.FLAT, amey.DATASET = Path(flat_dir), Path(data)
    counts = amey.build_dataset(work / 'yolo', train, AMEY_SMOKE['records'] if smoke else 0, np)
    (work / 'dataset.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    model = YOLO(str(pretrained))

    def started(trainer) -> None:   # as the original runner: keep trainer and validator on the requested device
        trainer.args.device = torch.device(device)
        trainer.validator.args.device = torch.device(device)

    model.add_callback('on_pretrain_routine_end', started)
    model.train(data=str(work / 'yolo/data.yaml'), device=torch.device(device), project=str(work), name='train',
                **amey_train_arguments(member, smoke))
    return work / 'train/weights/last.pt'


def train_detector(member: str, work: Path, device: str = 'cuda:0', data: Path = DATA, smoke: bool = False) -> Path:
    if member in HESHEN:
        return train_heshen_detector(member, work, data_root=data, device=device, smoke=smoke)
    return train_amey_detector(member, work, data=data, device=device, smoke=smoke)


# ----------------------------------------------------------------------------- 3. validation candidates
@contextlib.contextmanager
def use_weights(folder: Path | None):
    """Point solar.final_pipeline at another weights folder (same file names as weights/manifest.json)."""
    from solar import final_pipeline as fp
    previous = fp.WEIGHTS
    if folder is not None:
        fp.WEIGHTS = Path(folder)
    try:
        yield fp
    finally:
        fp.WEIGHTS = previous


def validation_candidates(pool: str, work: Path, device: str = 'cuda:0', weights: Path | None = None,
                          data: Path = DATA, workers: int = 4, log=print) -> tuple[list[dict], list[str]]:
    """U-Net and the pool's detectors on the 141 validation images, then every fusion candidate (score >= .15)
    with its six features and mask. Intermediate files are cached under `work`."""
    import numpy as np
    import torch
    with use_weights(weights) as fp:
        from filament_seg.dataset import load_disk_geometry
        from rescoring_probe_20261012 import candidates
        _, stems = fold1_split(annotation_stems(data))
        image_dir, members = data / 'train/train_images', fp.POOLS[pool]
        cache, logits, views = work / 'cache', work / 'logits_unet', work / 'views'
        fp.preprocess(stems, image_dir, cache, workers=workers)
        fp.unet_logits(stems, fp.WEIGHTS / fp.weight_entries()['unet']['file'], cache, logits, torch.device(device))
        fp.detector_views(stems, members, image_dir, cache, views, device, log=log)
        loaded = {m: json.loads((views / f'{m}_flip_views_test.json').read_text()) for m in members}
        disks = load_disk_geometry(cache / 'disk.json')
        inst = []
        for stem in stems:
            inst += candidates(stem, np.load(logits / f'{stem}.npy').astype(np.float32), disks[stem],
                               [loaded[m][stem] for m in members])
    return inst, stems


def _records(inst: list[dict], stems: list[str], store) -> list[dict]:
    from rescoring_probe_20261012 import records_for
    recs = []
    for stem in stems:   # every validation image, also those without a candidate (their annotations count as misses)
        idx = [i for i, d in enumerate(inst) if d['stem'] == stem]
        assert not idx or idx == list(range(idx[0], idx[0] + len(idx))), f'candidates of {stem} are not contiguous'
        recs += records_for(store, stem, idx, [inst[i]['rle'] for i in idx])
    return recs


# ----------------------------------------------------------------------------- 4. rescorer
def fit_rescorer(inst: list[dict], stems: list[str], store, pool: str) -> dict:
    """Fit the rescorer on all validation candidates and freeze it in the format of weights/rescorer_<pool>.json
    (the numbers scripts/export_rescorer_params_20261107.py writes, without its comparison to the recorded report)."""
    _scripts_on_path()
    import numpy as np
    from fusion_probe_multiscale_20261012 import arrays
    from rescoring_probe_20261012 import AREA_BINS, FEATURES, OPTION_B_MIN, Rescorer, pq, totals
    from solar.final_pipeline import POOLS, FrozenRescorer
    recs = _records(inst, stems, store)
    X, area, scores = arrays(inst)
    rescorer = Rescorer(X, area, scores, recs)
    keep = rescorer.keep(X, area)
    params = {'pool': pool, 'detectors': list(POOLS[pool]), 'features': list(FEATURES),
              'scaler_mean': rescorer.scaler.mean_.tolist(), 'scaler_scale': rescorer.scaler.scale_.tolist(),
              'coef': rescorer.model.coef_[0].tolist(), 'intercept': float(rescorer.model.intercept_[0]),
              'e_iou_by_bin': rescorer.e_iou.tolist(), 'area_bins': list(AREA_BINS),
              'pq_train': float(rescorer.pq_train), 'threshold': float(rescorer.pq_train / 2), 'candidate_min_score': .15,
              'rule': 'keep iff sigmoid(coef . (x - mean) / scale + intercept) * e_iou[bin(area)] > threshold',
              'fit': {'model': 'sklearn LogisticRegression(C=1, lbfgs, max_iter=1000) on StandardScaler features, '
                               'one row per (candidate, annotator record), target = hit at IoU > .5',
                      'pairs': rescorer.pairs, 'pair_hit_rate': rescorer.hit_rate, 'val_images': len(stems),
                      'candidates': len(inst), 'kept_insample': int(keep.sum()),
                      'val_fused_s030_pq': float(pq(totals(scores >= OPTION_B_MIN, recs))),
                      'val_insample_rescored_pq_not_a_score': float(pq(totals(keep, recs)))}}
    assert np.array_equal(FrozenRescorer(params).keep(X, area), keep), 'frozen rescorer disagrees with sklearn'
    return params


# ----------------------------------------------------------------------------- 5. refiner
def refiner_instances(inst: list[dict], stems: list[str], store) -> dict:
    """Candidates per validation image with their date half and a cross-fitted keep flag (the rescorer of one half
    decides for the other)."""
    _scripts_on_path()
    import numpy as np
    import refiner_core_20261031 as core
    from fusion_probe_multiscale_20261012 import arrays
    from rescoring_probe_20261012 import Rescorer
    from solar.grouped_cv import observation_date
    fields = ('score', 'members', 'area', 'log_area', 'support', 'mean_prob', 'radius')   # the dump's layout
    dates = sorted({observation_date(s) for s in stems})
    order = np.random.default_rng(HALF_SEED).permutation(len(dates))
    half_a = {dates[i] for i in order[:len(dates) // 2]}
    side = lambda stem: 'A' if observation_date(stem) in half_a else 'B'   # noqa: E731
    recs = _records(inst, stems, store)
    X, area, scores = arrays(inst)
    half = np.asarray([side(d['stem']) for d in inst])
    kept = np.zeros(len(inst), dtype=bool)
    for train, apply in (('A', 'B'), ('B', 'A')):
        rescorer = Rescorer(X, area, scores, [r for r in recs if side(r['stem']) == train])
        kept[half == apply] = rescorer.keep(X[half == apply], area[half == apply])
    by_stem: dict[str, list] = {s: [] for s in stems}
    for d, k in zip(inst, kept):
        by_stem[d['stem']].append({'order': len(by_stem[d['stem']]), 'rle': core.rle_json(d['rle']),
                                   **{f: (int(d[f]) if f in ('members', 'area') else float(d[f])) for f in fields},
                                   'kept': bool(k)})
    halves = {h: [s for s in stems if side(s) == h] for h in 'AB'}
    return {'meta': {'probe': 28, 'unet_seed': 0, 'halves': halves, 'candidates': len(inst), 'kept': int(kept.sum())},
            'images': [{'stem': s, 'half': side(s), 'instances': by_stem[s]} for s in stems]}


def refiner_samples(blob: dict, store, path: Path) -> dict:
    """Write the refiner's training pairs (pickle read by scripts/train_refiner_20261031.py); returns their statistics."""
    _scripts_on_path()
    import refiner_core_20261031 as core
    from prep_refiner_samples_20261031 import build_samples
    samples, stats = build_samples(blob, store)
    meta = {'probe': 28, 'unet_seed': blob['meta']['unet_seed'], 'halves': blob['meta']['halves'],
            'iou_min': core.IOU_MIN, 'stats': stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({'meta': meta, 'samples': samples}))
    return {'samples': len(samples), **stats}


def refiner_command(samples: Path, init: Path, out: Path, cache: Path, cpu: bool = False,
                    python: str = sys.executable, smoke: bool = False) -> list[str]:
    """The original refiner run (scripts/launch_refiner_20261031.sh, the trainALL / seed 0 call). `smoke` adds the
    script's own --smoke flag (40 samples, 1 epoch)."""
    cmd = [python, 'scripts/train_refiner_20261031.py', '--samples', str(samples), '--train-half', 'all',
           '--refiner-seed', '0', '--init-ckpt', str(init), '--cache', str(cache), '--out', str(out)]
    return cmd + ['--cpu'] * cpu + ['--smoke'] * smoke


def refiner_env(data: Path = DATA, uuid: str | None = None, hf_home: Path = HF_HOME) -> dict:
    """Environment the original launcher exported for the refiner (the script checks the cached ImageNet encoder
    offline under HF_HOME), with this machine's data folder and GPU UUID."""
    env = {'FILAMENT_DATA_ROOT': str(data), 'HF_HOME': str(hf_home), 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4'}
    if uuid:
        env['CUDA_VISIBLE_DEVICES'] = uuid
    return env


def train_refiner(samples: Path, unet: Path, out: Path, cache: Path, device: str = 'cuda:0', data: Path = DATA,
                  uuid: str | None = None, smoke: bool = False) -> Path:
    """Train the refiner from the U-Net checkpoint `unet`; returns its last checkpoint. The script requires the
    checkpoint to be named .../cpstrong_s0/model_last.pt, so the file is copied to that name first. On CUDA it runs
    on GPU `uuid` (default: `gpu_uuid` of the device index)."""
    init = out.parent / 'refiner_init/cpstrong_s0/model_last.pt'
    init.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(unet, init)
    if device.startswith('cuda'):
        uuid = uuid or gpu_uuid(int(device.split(':', 1)[1]) if ':' in device else 0)
    env = {**os.environ, **refiner_env(data, uuid if device != 'cpu' else None)}
    subprocess.run(refiner_command(samples, init, out, cache, cpu=device == 'cpu', smoke=smoke), cwd=ROOT, env=env, check=True)
    return out / 'model_last.pt'


# ----------------------------------------------------------------------------- all steps
def out_dir(smoke: bool = False) -> Path:
    """Default output folder: `OUT` for real retraining, `SMOKE_OUT` for smoke runs."""
    return SMOKE_OUT if smoke else OUT


def _check_out(out: Path, smoke: bool) -> None:
    """A smoke run never writes into the real retraining folder and vice versa (also not through retrain.json)."""
    other = OUT if smoke else SMOKE_OUT
    if Path(out).resolve() == other.resolve():
        raise ValueError(f'{out} is the {"real retraining" if smoke else "smoke"} folder; smoke={smoke} must not write there')
    record = Path(out) / 'retrain.json'
    if record.exists():
        previous = json.loads(record.read_text(encoding='utf-8'))
        if bool(previous.get('smoke', False)) != smoke:
            raise ValueError(f'{record} was written with smoke={previous.get("smoke", False)}; use another folder')


class _StepClock:
    """Start / end time, seconds and (for steps run in this process) peak CUDA memory of each step."""

    def __init__(self, log):
        self.log, self.rows = log, {}

    @contextlib.contextmanager
    def __call__(self, step: str, cuda: bool):
        import torch
        cuda = cuda and torch.cuda.is_available()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started, t = time.strftime('%Y-%m-%d %H:%M:%S'), time.perf_counter()
        self.log(f'[retrain] {started} start {step}')
        yield
        row = {'started': started, 'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'seconds': time.perf_counter() - t}
        if cuda:
            row['peak_allocated_gb_this_process'] = torch.cuda.max_memory_allocated() / 2 ** 30
        self.rows[step] = row
        self.log(f"[retrain] {row['finished']} end {step}: {row['seconds'] / 60:.1f} min")


def retrain(steps=STEPS, out: Path | None = None, device: str = 'cuda:0', gpu_index: int = 0, data: Path = DATA,
            finals=('final1_A6',), workers: int = 4, log=print, smoke: bool = False) -> dict:
    """Run the chosen steps in order and collect the results in `out`/weights under the shipped file names.
    Files of steps that are not chosen are copied from weights/, so the result is always a complete set that
    `use_weights(out / 'weights')` (or `solar.final_pipeline.WEIGHTS = ...`) can run. Returns what was produced.

    `out` defaults to `out_dir(smoke)`. `smoke=True` runs every step with the original runners' smoke settings
    (module docstring) into a folder of its own; without it the training settings are those of the shipped files.
    `out`/retrain.json is rewritten after every step and keeps the earlier runs into the same folder."""
    from solar import final_pipeline as fp
    from solar.data import AnnotationStore
    unknown = set(steps) - set(STEPS)
    assert not unknown, f'unknown steps {sorted(unknown)}; choose from {STEPS}'
    out = Path(out) if out is not None else out_dir(smoke)
    _check_out(out, smoke)
    weights = out / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    if smoke:
        (weights / 'SMOKE_RUN_NOT_A_MODEL.txt').write_text('Written by retrain(smoke=True): 1-epoch smoke settings; '
                                                           'these files only show that the code runs.\n', encoding='utf-8')
    entries, produced, clock = fp.weight_entries(), {}, _StepClock(log)
    for file in [e['file'] for e in entries.values()] + [fp.manifest()['finals'][f]['rescorer'] for f in fp.FINALS]:
        if not (weights / file).exists():
            shutil.copyfile(fp.WEIGHTS / file, weights / file)
    # Resolve the card once, before Ultralytics rewrites CUDA_VISIBLE_DEVICES in this process (select_device does).
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    uuid = gpu_uuid(int(device.split(':', 1)[1]) if ':' in device else 0) if device.startswith('cuda') else None
    record = out / 'retrain.json'
    history = json.loads(record.read_text(encoding='utf-8'))['runs'] if record.exists() else []
    run = {'steps': list(steps), 'smoke': smoke, 'device': device, 'gpu_uuid': uuid, 'finals': list(finals),
           'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'produced': produced, 'timings': clock.rows}

    def save(status: str) -> None:
        run['status'] = status
        record.write_text(json.dumps({'smoke': smoke, 'out': str(out), 'runs': history + [run]}, indent=1, default=str),
                          encoding='utf-8')

    def restore_visible() -> None:
        if visible is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = visible

    save('running')
    try:
        if 'inputs' in steps:
            with clock('inputs', cuda=False):
                produced['inputs'] = {**prepare_inputs(data, workers), **fetch_pretrained()}
            log(f"inputs: {produced['inputs']}")
            save('running')
        if 'unet' in steps:
            with clock('unet', cuda=False):
                checkpoint = train_unet(out / 'unet_cpstrong_s0', gpu_index, data, uuid=uuid, smoke=smoke)
            shutil.copyfile(checkpoint, weights / entries['unet']['file'])
            produced['unet'] = sha256(checkpoint)
            save('running')
        for member in DETECTORS:
            if member in steps:
                with clock(member, cuda=True):
                    checkpoint = train_detector(member, out / f'detector_{member}', device, data, smoke=smoke)
                restore_visible()
                shutil.copyfile(checkpoint, weights / entries[member]['file'])
                produced[member] = sha256(checkpoint)
                save('running')
        if 'rescorer' in steps or 'refiner' in steps:
            store = AnnotationStore(data)
            pools = {fp.FINALS[f] for f in finals} | ({REFINER_POOL} if 'refiner' in steps else set())
            # Smoke-trained detectors find (almost) nothing; the smoke run fits on the shipped first stage instead.
            first_stage = fp.WEIGHTS if smoke else weights
            with clock('validation_candidates', cuda=True):
                found = {pool: validation_candidates(pool, out / 'validation', device, first_stage, data, workers, log)
                         for pool in sorted(pools)}
            restore_visible()
            produced['validation_candidates'] = {'first_stage_weights': str(first_stage),
                                                 **{pool: len(inst) for pool, (inst, _) in found.items()}}
            save('running')
            if 'rescorer' in steps:
                with clock('rescorer', cuda=False):
                    for final in finals:
                        pool = fp.FINALS[final]
                        params = fit_rescorer(*found[pool], store, pool)
                        (weights / fp.manifest()['finals'][final]['rescorer']).write_text(json.dumps(params, indent=1), encoding='utf-8')
                        produced[f'rescorer_{pool}'] = params['fit']
                save('running')
            if 'refiner' in steps:
                with clock('refiner', cuda=False):
                    stats = refiner_samples(refiner_instances(*found[REFINER_POOL], store), store, out / 'refiner_samples.pkl')
                    checkpoint = train_refiner(out / 'refiner_samples.pkl', weights / entries['unet']['file'], out / 'refiner_r0',
                                               out / 'validation/cache', device, data, uuid=uuid, smoke=smoke)
                shutil.copyfile(checkpoint, weights / entries['refiner']['file'])
                produced['refiner'] = {'sha256': sha256(checkpoint), 'samples': stats['samples']}
    except BaseException as error:
        run['error'] = repr(error)
        save('failed')
        raise
    run['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save('done')
    return produced
