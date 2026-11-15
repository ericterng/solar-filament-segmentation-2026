"""Probe 28: train one instance refiner (docs/experiments/instance_refiner_20261031.md §A). Runs on the A5000.

A 3-channel EfficientNet-B4 U-Net [flat, radius, candidate mask] initialised from cpstrong_s<unet seed>'s
model_last.pt with a zero third input channel, so at step 0 it is exactly the current U-Net (asserted
on a random input, 1e-5). Trained on the samples of ONE date half of fold1 val (or all 141 for test),
boundary-weighted BCE + Dice (probe 16's loss, cpstrong's), AdamW 1e-4 cosine, 16 epochs, batch 8,
AMP on CUDA, last checkpoint. Never imports `solar`; depends on the upstream package only through
the probe 16 / 19 wrappers that already live on the remote.

Probe 30 (docs/experiments/refiner_oof566_20261102.md, `--init imagenet`): the same network from the pinned
ImageNet weights, trained on the 566 out-of-fold images. An explicit --lr and --val-instances are required;
long boxes may crop on an off-centre apply tile, the flat channel gets cpstrong's brightness/contrast jitter,
and no fold1 stem may reach the samples.

    env/bin/python scripts/train_refiner_20261031.py --samples outputs/refiner_20261031/samples_s0.pkl \
        --train-half A --refiner-seed 0 --init-ckpt outputs/unet_cpstrong_20261025/cpstrong_s0/model_last.pt \
        --out outputs/refiner_20261031/r_s0_trainA_r0
    env/bin/python scripts/train_refiner_20261031.py --samples outputs/refiner_oof_20261102/samples_oof566.pkl \
        --train-half oof566 --refiner-seed 0 --init imagenet --lr 1e-3 \
        --val-instances outputs/refiner_20261031/instances_val_s0.json --out outputs/refiner_oof_20261102/r_oof_r0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_unet_boundary_20261019 import BoundaryDiceBCELoss  # noqa: E402  (puts the upstream package on sys.path)
from train_unet_encoder_20261015 import check_weights  # noqa: E402
from filament_seg.dataset import load_disk_geometry  # noqa: E402
from filament_seg.model import build_model  # noqa: E402
import refiner_core_20261031 as core  # noqa: E402

ENCODER = 'efficientnet-b4'
IMAGE_SHAPE = (2048, 2048)
EQUIVALENCE_TOL = 1e-5
OOF_HALF, OOF_PROBE = 'oof566', 30
APPLY_HALF = {'A': 'B', 'B': 'A', 'all': 'test', OOF_HALF: 'fold1 and test'}
SMOKE_SAMPLES, SMOKE_EPOCHS = 40, 1
BRIGHTNESS_LIMIT, CONTRAST_LIMIT, JITTER_P = .15, .15, .5  # cpstrong's RandomBrightnessContrast
LONG_TILE_P = .5
FOLD1_IMAGES = 141


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_device(cpu: bool = False) -> torch.device:
    """CUDA only when a card is actually visible (CUDA_VISIBLE_DEVICES='' leaves is_available() true on Windows)."""
    if cpu or not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return torch.device('cpu')
    return torch.device('cuda')


def no_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_init(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """The 2-channel cpstrong U-Net exactly as upstream load_trained rebuilds it."""
    payload = torch.load(str(path), map_location=device)
    assert payload.get('encoder', 'resnet34') == ENCODER and payload.get('in_channels', 2) == 2, \
        (payload.get('encoder'), payload.get('in_channels'))
    model = build_model(encoder=ENCODER, in_channels=2, weights=None).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, payload


def expand_first_conv(model: nn.Module) -> str:
    """Give the encoder's first conv a third, all-zero input channel; returns the module name."""
    for name, m in model.encoder.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 2:
            w = m.weight.data
            m.weight = nn.Parameter(torch.cat([w, torch.zeros_like(w[:, :1])], dim=1))
            m.in_channels = 3
            return name
    raise RuntimeError('no 2-channel conv found in the encoder')


def make_refiner(init_ckpt: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    """(3-channel model, init payload, equivalence report); asserts the zero channel changes nothing."""
    model2, payload = load_init(init_ckpt, device)
    model3 = copy.deepcopy(model2)
    conv = expand_first_conv(model3)
    report = equivalence_check(model2, model3, device)
    report['first_conv'] = conv
    assert report['max_abs_diff'] < EQUIVALENCE_TOL, report
    del model2
    return model3, payload, report


def make_imagenet_refiner(device: torch.device) -> tuple[nn.Module, dict]:
    """Probe 30: (3-channel model, pinned weight record); the third channel keeps ImageNet's third-channel weights."""
    weights = check_weights(ENCODER)
    model = build_model(encoder=ENCODER, in_channels=3, weights='imagenet').to(device)
    verify_conv_stem(model, Path(weights['path']))
    return model, weights


def verify_conv_stem(model: nn.Module, weights_path: Path) -> None:
    """A matching file SHA does not prove loading: the first conv must equal the file's stem weights."""
    from safetensors.torch import load_file

    state = load_file(str(weights_path), device='cpu')
    keys = [k for k in state if k.endswith('conv_stem.weight')]
    assert len(keys) == 1, keys
    conv = next(m for m in model.encoder.modules() if isinstance(m, nn.Conv2d))
    assert torch.equal(conv.weight.data.cpu(), state[keys[0]]), 'ImageNet weights were not loaded'


def assert_no_val_stems(train_stems: set[str], val_instances: Path) -> str:
    """Probe 30 leak guard at run time: no fold1 stem may be trained on; returns the fold1 stems' digest."""
    val = sorted(im['stem'] for im in json.loads(val_instances.read_text(encoding='utf-8'))['images'])
    assert len(val) == FOLD1_IMAGES, len(val)
    overlap = train_stems & set(val)
    assert not overlap, f'fold1 stems in the training samples: {sorted(overlap)[:5]}'
    return hashlib.sha256('\n'.join(val).encode('ascii')).hexdigest()


def oof_crop_origin(bbox: tuple[int, int, int, int], shape: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    """Probe 30 §A: boxes over SINGLE_MAX px crop half the time on a random apply tile + jitter, else as probe 28."""
    y0, y1, x0, x1 = bbox
    if max(y1 - y0, x1 - x0) > core.SINGLE_MAX and rng.random() < LONG_TILE_P:
        origins = core.tile_origins(bbox, shape)
        ty, tx = origins[int(rng.integers(len(origins)))]
        dy, dx = rng.integers(-core.JITTER, core.JITTER + 1, 2)
        return core.clip_origin(int(ty + dy), core.TILE, shape[0]), core.clip_origin(int(tx + dx), core.TILE, shape[1])
    return core.sample_crop_origin(bbox, shape, rng)


def jitter_flat(flat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Upstream's RandomBrightnessContrast on a [0, 1] float image, drawn from the sample's own rng."""
    if rng.random() >= JITTER_P:
        return flat
    alpha = 1.0 + rng.uniform(-CONTRAST_LIMIT, CONTRAST_LIMIT)
    beta = rng.uniform(-BRIGHTNESS_LIMIT, BRIGHTNESS_LIMIT)
    return np.clip(flat * alpha + beta, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def equivalence_check(model2: nn.Module, model3: nn.Module, device: torch.device, size: int = 512, seed: int = 0) -> dict:
    no_tf32()
    g = torch.Generator().manual_seed(seed)
    x2 = torch.randn(1, 2, size, size, generator=g).to(device)
    x3 = torch.cat([x2, torch.zeros(1, 1, size, size, device=device)], dim=1)
    x3b = torch.cat([x2, torch.ones(1, 1, size, size, device=device)], dim=1)
    model2.eval(), model3.eval()
    y2, y3, y3b = model2(x2).float(), model3(x3).float(), model3(x3b).float()
    return {'max_abs_diff': float((y2 - y3).abs().max()), 'size': size,
            'ones_channel_max_abs_diff_info': float((y2 - y3b).abs().max())}


class RefinerSamples(torch.utils.data.Dataset):
    """One (candidate, record) sample -> (3x512x512 input, 1x512x512 target), §A crop + dihedral."""

    def __init__(self, samples: list[dict], flats: dict[str, np.ndarray], disks: dict, seed: int, tile: int = core.TILE,
                 oof: bool = False):
        self.samples, self.flats, self.disks, self.seed, self.tile, self.oof = samples, flats, disks, seed, tile, oof
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[i]
        rng = np.random.default_rng([self.seed, self.epoch, i])
        cand, gt = core.decode(s['cand_rle']), core.decode(s['gt_rle'])
        t = self.tile
        crop_origin = oof_crop_origin if self.oof else core.sample_crop_origin
        y0, x0 = crop_origin(core.bbox_of(cand), cand.shape, rng)
        flat = self.flats[s['stem']][y0:y0 + t, x0:x0 + t].astype(np.float32) / 255.0
        if self.oof:
            flat = jitter_flat(flat, rng)
        radius = self.disks[s['stem']].radius_window(y0, x0, t, t).astype(np.float32)
        x = np.stack([flat, radius, cand[y0:y0 + t, x0:x0 + t].astype(np.float32)], axis=0)
        y = gt[y0:y0 + t, x0:x0 + t].astype(np.float32)[None]
        k = int(rng.integers(8))
        return torch.from_numpy(core.orient(x, k)), torch.from_numpy(core.orient(y, k))


def load_flats(cache: Path, stems: list[str]) -> dict[str, np.ndarray]:
    flats = {}
    for s in stems:
        img = cv2.imread(str(cache / 'flat' / f'{s}.png'), cv2.IMREAD_GRAYSCALE)
        assert img is not None and img.shape == IMAGE_SHAPE, (s, None if img is None else img.shape)
        flats[s] = img
    return flats


def select_samples(samples: list[dict], train_half: str, limit: int | None) -> list[dict]:
    sel = samples if train_half in ('all', OOF_HALF) else [s for s in samples if s['half'] == train_half]
    if train_half == OOF_HALF:
        assert all(s['half'] == OOF_HALF for s in sel)
    else:
        assert all(s['half'] in ('A', 'B') for s in sel)
    if train_half in ('A', 'B'):
        assert all(s['half'] == train_half for s in sel)
    return sel[:limit] if limit else sel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--train-half', choices=sorted(APPLY_HALF), required=True)
    parser.add_argument('--refiner-seed', type=int, required=True)
    parser.add_argument('--init', choices=('cpstrong', 'imagenet'), default='cpstrong')
    parser.add_argument('--init-ckpt', type=Path, help='cpstrong init only')
    parser.add_argument('--val-instances', type=Path, help='imagenet init only: fold1 instances json, the leak guard')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=HERE.parent / 'outputs/amey_unet_cache')
    parser.add_argument('--epochs', type=int, default=core.TRAIN_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=core.TRAIN_BATCH)
    parser.add_argument('--lr', type=float, help=f'cpstrong default {core.TRAIN_LR}; imagenet must set it')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true', help=f'{SMOKE_SAMPLES} samples, {SMOKE_EPOCHS} epoch')
    parser.add_argument('--cpu', action='store_true', help='local smoke only')
    args = parser.parse_args()
    imagenet = args.init == 'imagenet'
    if imagenet and (args.lr is None or args.val_instances is None or args.init_ckpt or args.train_half != OOF_HALF):
        parser.error(f'--init imagenet needs --lr, --val-instances and --train-half {OOF_HALF}, and no --init-ckpt')
    if not imagenet and (args.init_ckpt is None or args.train_half == OOF_HALF):
        parser.error('--init cpstrong needs --init-ckpt and a train half of probe 28')
    lr = core.TRAIN_LR if args.lr is None else args.lr
    if args.out.exists():
        raise FileExistsError(args.out)
    epochs, limit = (SMOKE_EPOCHS, SMOKE_SAMPLES) if args.smoke else (args.epochs, None)
    device = pick_device(args.cpu)
    torch.manual_seed(args.refiner_seed)
    np.random.seed(args.refiner_seed)

    blob = pickle.loads(args.samples.read_bytes())
    samples = select_samples(blob['samples'], args.train_half, limit)
    assert samples, 'no samples'
    stems = sorted({s['stem'] for s in samples})
    guard = {}
    if imagenet:
        assert blob['meta']['probe'] == OOF_PROBE, blob['meta']
        unet_seed, init_ckpt_sha = None, None
        train_stems = {s['stem'] for s in blob['samples']}  # every sample, also under --smoke
        guard = {'val_stems_sha256': assert_no_val_stems(train_stems, args.val_instances),
                 'train_stems_sha256': hashlib.sha256('\n'.join(sorted(train_stems)).encode('ascii')).hexdigest()}
        model, init_weights = make_imagenet_refiner(device)
        init_encoder, equivalence = ENCODER, None
    else:
        unet_seed = int(blob['meta']['unet_seed'])
        assert args.init_ckpt.name == 'model_last.pt' and f'cpstrong_s{unet_seed}' in str(args.init_ckpt), \
            (args.init_ckpt, unet_seed)
        halves = blob['meta']['halves']
        if args.train_half != 'all':
            assert not set(stems) & set(halves[APPLY_HALF[args.train_half]]), 'training stems leak into the apply half'
            assert set(stems) <= set(halves[args.train_half])
        model, payload, equivalence = make_refiner(args.init_ckpt, device)
        init_ckpt_sha, init_encoder, init_weights = sha256(args.init_ckpt), payload.get('encoder'), None
    flats = load_flats(args.cache, stems)
    disks = load_disk_geometry(args.cache / 'disk.json')
    dataset = RefinerSamples(samples, flats, disks, seed=args.refiner_seed, oof=imagenet)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                         pin_memory=device.type == 'cuda',
                                         generator=torch.Generator().manual_seed(args.refiner_seed))
    loss_fn = BoundaryDiceBCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args.out.mkdir(parents=True)
    forward_calls, shapes, losses, started = 0, set(), [], time.time()
    for epoch in range(1, epochs + 1):
        dataset.epoch = epoch
        model.train()
        running, n = 0.0, 0
        for x, y in loader:
            shapes.add(tuple(x.shape[1:]))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type='cuda'):
                    logits = model(x)
                loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
            forward_calls += 1
            running += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        scheduler.step()
        losses.append(running / max(n, 1))
        print(f'epoch {epoch}/{epochs} loss {losses[-1]:.4f} forward_calls {forward_calls} '
              f'{time.time() - started:.0f}s', flush=True)
    assert shapes == {(3, core.TILE, core.TILE)}, shapes

    ckpt = args.out / 'model_last.pt'
    refiner_meta = {'probe': OOF_PROBE if imagenet else 28, 'unet_seed': unet_seed, 'refiner_seed': args.refiner_seed,
                    'train_half': args.train_half, 'apply_half': APPLY_HALF[args.train_half], 'smoke': args.smoke,
                    'init': args.init, 'lr': lr, 'epochs': epochs, **guard}
    torch.save({'model': model.state_dict(), 'encoder': ENCODER, 'in_channels': 3, 'refiner': refiner_meta}, ckpt)
    runtime = {**refiner_meta, 'init_ckpt': args.init_ckpt and str(args.init_ckpt), 'init_ckpt_sha256': init_ckpt_sha,
               'init_encoder': init_encoder, 'init_weights': init_weights, 'equivalence': equivalence, 'samples': len(samples),
               'stems': len(stems), 'batch_size': args.batch_size,
               'forward_calls': forward_calls, 'expected_forward_calls': epochs * int(np.ceil(len(samples) / args.batch_size)),
               'loss': {'name': 'boundary_weighted_bce_dice', 'band': 2, 'weight': 4.0}, 'epoch_loss': losses,
               'shapes': sorted(shapes), 'device': str(device), 'amp': use_amp,
               'peak_mem_gb': float(torch.cuda.max_memory_allocated() / 1e9) if use_amp else None,
               'seconds': time.time() - started, 'checkpoint_sha256': sha256(ckpt),
               'script_sha256': sha256(Path(__file__)), 'core_sha256': sha256(HERE / 'refiner_core_20261031.py')}
    assert runtime['forward_calls'] == runtime['expected_forward_calls'], runtime
    (args.out / 'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    print('DONE', json.dumps({k: runtime[k] for k in ('train_half', 'samples', 'forward_calls', 'checkpoint_sha256')}))


if __name__ == '__main__':
    main()
