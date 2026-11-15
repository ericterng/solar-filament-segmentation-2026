"""Probes 19-21: three last training-recipe changes on top of the probe 16 boundary U-Net (pre-registered 2026-10-24).

docs/experiments/unet_last3_20261024.md. Every arm starts from probe 16's recipe
(train_unet_boundary_20261019.py: EfficientNet-B4, boundary-weighted BCE band 2 /
weight 1+4, 30 epochs, batch 16, crop 512, lr 1e-3, last checkpoint, fold1) and
changes ONE thing. The saved bw_s0 / bw_s1 runs are the control (not retrained).
- emaswa: weight averaging. The training itself is unchanged; the wrapper also
  keeps an EMA of all float weights and buffers (decay EMA_DECAY, updated after
  every optimizer step) -> model_ema.pt, and an equal average of the last
  SWA_EPOCHS epoch-end weights and buffers -> model_swa.pt. model_last.pt is the
  same recipe as bw (informational).
- copypaste: with probability CP_PROB per training crop, one ground-truth filament
  from another fold1 training image (donor bank built once from the training
  views) is pasted into the crop before the upstream flips / brightness jitter.
  Only the filament's own pixels are pasted, shifted so the median of a ring
  around it matches the target's ring median; the location must keep every
  pasted pixel inside radius CP_RADIUS_MAX of the disk and CP_GAP px away from
  existing filaments, else the paste is skipped. The mask gets the pasted pixels.
  The paste uses its own RNG, so the base crop sequence is the same as bw's.
- bw8: boundary weight 1+4 -> 1+7 (BW8_WEIGHT). Band unchanged.
All constants are fixed in advance and not swept. Upstream files are not modified:
train.py is loaded as a module and its names DiceBCELoss / FilamentCrops /
build_model / torch are rebound.
Remote (A5000): CUDA_VISIBLE_DEVICES=<UUID> UNET_ENCODER_GPU_UUID=<UUID> HF_HUB_OFFLINE=1 \
    env/bin/python scripts/train_unet_last3_20261024.py --arm ARM --seed N --out DIR [--smoke]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unet_boundary_20261019 as bw  # noqa: E402
import train_unet_encoder_20261015 as base  # noqa: E402
from train_unet_encoder_20261015 import CACHE, SPLITS, UPSTREAM, assert_on_gpu, expected_uuid, sha256, torch  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402
from filament_seg.dataset import _AUGMENT, FilamentCrops  # noqa: E402

ARMS = ('emaswa', 'copypaste', 'bw8')
ENCODER = bw.ENCODER
EMA_DECAY = 0.999
SWA_EPOCHS = 5
CP_PROB, CP_MAX_SIDE, CP_MIN_AREA, CP_RADIUS_MAX, CP_GAP, CP_TRIES = 0.5, 256, 10, 0.95, 3, 10
CP_MARGIN, CP_RING_IN, CP_RING_OUT = 6, 1, 5
BW8_WEIGHT = 7.0


# --- bw8 -----------------------------------------------------------------

class BW8DiceBCELoss(bw.BoundaryDiceBCELoss):
    """Probe 16's loss with the boundary weight raised from 4 to BW8_WEIGHT (band unchanged)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, weight=BW8_WEIGHT, **kwargs)


# --- emaswa ----------------------------------------------------------------

class WeightAverager:
    """EMA (per optimizer step) and SWA (last SWA_EPOCHS epoch ends) of a model's state_dict."""

    def __init__(self, epochs: int, decay: float = EMA_DECAY, swa_epochs: int = SWA_EPOCHS) -> None:
        self.epochs, self.decay, self.swa_epochs = epochs, decay, min(swa_epochs, epochs)
        self.model: torch.nn.Module | None = None
        self.ema: dict[str, torch.Tensor] | None = None
        self.swa_sum: dict[str, torch.Tensor] | None = None
        self.swa_count = self.ema_updates = 0
        self.swa_epochs_used: list[int] = []

    @torch.no_grad()
    def update(self) -> None:
        # state_dict tensors share storage with the live parameters / buffers, which the
        # optimizer and BatchNorm update in place, so the dict is built once.
        if self.ema is None:
            self._state = self.model.state_dict()
            self.ema = {k: v.detach().clone() for k, v in self._state.items()}
        else:
            for k, v in self._state.items():
                if v.is_floating_point():
                    self.ema[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.ema[k].copy_(v)
        self.ema_updates += 1

    @torch.no_grad()
    def on_epoch_end(self, payload: dict, last_path: Path) -> None:
        epoch = payload['epoch']
        if self.ema is None:
            raise RuntimeError('epoch ended before any EMA update: optimizer injection failed')
        ema_cpu = {k: v.detach().cpu().clone() for k, v in self.ema.items()}
        torch.save({**payload, 'model': ema_cpu,
                    'averaging': {'kind': 'ema', 'decay': self.decay, 'updates': self.ema_updates}},
                   last_path.with_name('model_ema.pt'))
        if epoch > self.epochs - self.swa_epochs:
            state = payload['model']
            if self.swa_sum is None:
                self.swa_sum = {k: v.detach().double().cpu().clone() if v.is_floating_point() else v.detach().cpu().clone()
                                for k, v in state.items()}
            else:
                for k, v in state.items():
                    if v.is_floating_point():
                        self.swa_sum[k].add_(v.detach().double().cpu())
                    else:
                        self.swa_sum[k].copy_(v.detach().cpu())
            self.swa_count += 1
            self.swa_epochs_used.append(epoch)
        if epoch == self.epochs:
            ref = payload['model']
            swa = {k: (v / self.swa_count).to(ref[k].dtype) if v.is_floating_point() else v
                   for k, v in self.swa_sum.items()}
            torch.save({**payload, 'model': swa,
                        'averaging': {'kind': 'swa', 'epochs': self.swa_epochs_used}},
                       last_path.with_name('model_swa.pt'))


class _Proxy:
    """Attribute proxy: overrides first, then the wrapped module."""

    def __init__(self, wrapped, **overrides) -> None:
        self.__dict__.update(overrides)
        self._wrapped = wrapped

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def install_averager(train, averager: WeightAverager) -> None:
    """Rebind train.build_model / train.torch so upstream's own loop feeds the averager."""
    original_build = train.build_model

    def build_model(*args, **kwargs):
        averager.model = original_build(*args, **kwargs)
        return averager.model

    class AveragingAdamW(torch.optim.AdamW):
        def step(self, closure=None):
            loss = super().step(closure)
            averager.update()
            return loss

    def save(obj, path, *args, **kwargs):
        torch.save(obj, path, *args, **kwargs)
        if str(path).endswith('_last.pt'):
            averager.on_epoch_end(obj, Path(path))

    train.build_model = build_model
    train.torch = _Proxy(torch, optim=_Proxy(torch.optim, AdamW=AveragingAdamW), save=save)


# --- copypaste -------------------------------------------------------------

def _disk_kernel(radius: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _ring(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (cv2.dilate(m, _disk_kernel(CP_RING_OUT)) > 0) & ~(cv2.dilate(m, _disk_kernel(CP_RING_IN)) > 0)


def donor_patches(flat_u8: np.ndarray, rles: list) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], int]:
    """(flat patch in [0,1], instance mask patch, donor ring median) per usable instance; and #skipped."""
    if not rles:
        return [], 0
    masks = mask_utils.decode(rles).astype(bool)  # H, W, N
    union = masks.any(axis=2)
    height, width = union.shape
    out, skipped = [], 0
    for i, rle in enumerate(rles):
        x, y, w, h = (int(round(v)) for v in mask_utils.toBbox(rle))
        if max(w, h) > CP_MAX_SIDE or int(masks[:, :, i].sum()) < CP_MIN_AREA:
            skipped += 1
            continue
        y0, x0 = max(y - CP_MARGIN, 0), max(x - CP_MARGIN, 0)
        y1, x1 = min(y + h + CP_MARGIN, height), min(x + w + CP_MARGIN, width)
        m = masks[y0:y1, x0:x1, i]
        ring = _ring(m) & ~union[y0:y1, x0:x1]
        if ring.sum() < 8:
            skipped += 1
            continue
        patch = flat_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
        out.append((patch, m, float(np.median(patch[ring]))))
    return out, skipped


def paste_one(rng: random.Random, flat: np.ndarray, radius: np.ndarray, mask: np.ndarray,
              donor: tuple[np.ndarray, np.ndarray, float]) -> tuple[np.ndarray, np.ndarray, bool]:
    """Paste one donor filament into a crop (float flat in [0,1], radius map, 0/1 mask)."""
    patch, m, donor_median = donor
    k, flip = rng.randrange(4), rng.random() < 0.5
    patch, m = np.rot90(patch, k), np.rot90(m, k)
    if flip:
        patch, m = patch[:, ::-1], m[:, ::-1]
    h, w = m.shape
    if h > flat.shape[0] or w > flat.shape[1]:
        return flat, mask, False
    occupied = cv2.dilate((mask > 0).astype(np.uint8), _disk_kernel(CP_GAP)) > 0
    ring = _ring(m)
    for _ in range(CP_TRIES):
        y0, x0 = rng.randrange(flat.shape[0] - h + 1), rng.randrange(flat.shape[1] - w + 1)
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        if occupied[sl][m].any() or (radius[sl][m] > CP_RADIUS_MAX).any():
            continue
        free_ring = ring & ~occupied[sl]
        target_median = float(np.median(flat[sl][free_ring])) if free_ring.sum() >= 8 else donor_median
        flat, mask = flat.copy(), mask.copy()
        flat[sl][m] = np.clip(patch[m] + (target_median - donor_median), 0.0, 1.0)
        mask[sl][m] = 1.0
        return flat, mask, True
    return flat, mask, False


class CopyPasteCrops(FilamentCrops):
    """Upstream FilamentCrops plus one pasted training filament with probability CP_PROB."""

    report_dir: Path | None = None
    report: dict = {}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.target != 'annotator':
            raise ValueError('copy-paste is defined for target=annotator only')
        self.bank: list[tuple[np.ndarray, np.ndarray, float]] = []
        skipped = 0
        for image_id in self.image_ids:
            patches, n_skip = donor_patches(self._read_flat(self.annotations.images[image_id].stem),
                                            self.annotations.gt_rles(image_id))
            self.bank += patches
            skipped += n_skip
        if not self.bank:
            raise RuntimeError('empty copy-paste donor bank')
        self._paste_rngs: dict[int, random.Random] = {}
        CopyPasteCrops.report = {'donor_views': len(self.image_ids), 'donors': len(self.bank),
                                 'skipped_instances': skipped, **self._self_check()}

    def _paste_rng(self) -> random.Random:
        worker = torch.utils.data.get_worker_info()
        key = 0 if worker is None else worker.id + 1
        rng = self._paste_rngs.get(key)
        if rng is None:
            rng = random.Random(1_000_003 * (self.seed + 1) + key)
            self._paste_rngs[key] = rng
        return rng

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = self._item(index)
        return x, y

    def _item(self, index: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
        # Upstream FilamentCrops.__getitem__, verbatim except for the paste step.
        image_id = self.image_ids[index // self.crops_per_image]
        record = self.annotations.images[image_id]
        stem = record.stem
        shape = (record.height, record.width)
        rng = self._rng()

        flat_u8 = self._flat(stem)
        y0, x0 = self._sample_crop_origin(rng, image_id, stem, shape)
        size = self.crop_size

        flat_crop = flat_u8[y0 : y0 + size, x0 : x0 + size].astype(np.float32) / 255.0
        radius_crop = self._disk[stem].radius_window(y0, x0, *flat_crop.shape).astype(np.float32)
        mask_crop = self._load_mask(image_id, stem, shape)[
            y0 : y0 + size, x0 : x0 + size
        ].astype(np.float32)

        pasted = False
        paste_rng = self._paste_rng()
        if paste_rng.random() < CP_PROB:
            donor = self.bank[paste_rng.randrange(len(self.bank))]
            flat_crop, mask_crop, pasted = paste_one(paste_rng, flat_crop, radius_crop, mask_crop, donor)

        if self.augment:
            out = _AUGMENT(image=flat_crop, mask=mask_crop, radius=radius_crop)
            flat_crop, mask_crop, radius_crop = out["image"], out["mask"], out["radius"]

        x = np.stack([flat_crop, radius_crop], axis=0).astype(np.float32)
        y = mask_crop[None, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), pasted

    def _self_check(self, n: int = 200) -> dict:
        """Paste rate on n crops in the main process (key 0, never used by DataLoader workers)."""
        saved_aug, self.augment = self.augment, False
        tries = 0
        pasted = 0
        examples = []
        for i in range(n):
            x, y, ok = self._item((i * 7919) % len(self))
            tries += 1
            pasted += ok
            if ok and len(examples) < 4:
                examples.append((x[0].numpy(), y[0].numpy()))
        self.augment = saved_aug
        if self.report_dir is not None:
            for j, (img, msk) in enumerate(examples):
                rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                edge = (cv2.morphologyEx((msk > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
                rgb[edge] = (0, 0, 255)
                cv2.imwrite(str(self.report_dir / f'copypaste_example_{j}.png'), rgb)
        return {'self_check_crops': tries, 'self_check_pasted': pasted,
                'self_check_paste_rate': pasted / max(tries, 1)}


# --- main -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    uuid = expected_uuid()
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('HF_HUB_OFFLINE=1 is required (weights must be pinned and already cached)')
    weights = base.check_weights(ENCODER)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import segmentation_models_pytorch as smp
    import timm
    loss_cls = BW8DiceBCELoss if args.arm == 'bw8' else bw.BoundaryDiceBCELoss
    weight = BW8_WEIGHT if args.arm == 'bw8' else bw.WEIGHT
    arm_settings = {
        'emaswa': {'ema_decay': EMA_DECAY, 'swa_epochs': SWA_EPOCHS, 'buffers': 'averaged like weights'},
        'copypaste': {'prob': CP_PROB, 'max_side': CP_MAX_SIDE, 'min_area': CP_MIN_AREA,
                      'radius_max': CP_RADIUS_MAX, 'gap_px': CP_GAP, 'tries': CP_TRIES,
                      'ring_px': [CP_RING_IN, CP_RING_OUT]},
        'bw8': {'boundary_weight': BW8_WEIGHT},
    }[args.arm]
    runtime = {'arm': args.arm, 'arm_settings': arm_settings, 'encoder': ENCODER, 'seed': args.seed,
               'smoke': args.smoke,
               'loss': {'name': 'boundary_weighted_bce', 'band_px': bw.BAND, 'weight': weight,
                        'dice': 'upstream, unchanged', 'forward_calls': 0},
               'precision': 'upstream AMP (autocast fp16 + GradScaler) on CUDA',
               'weights': weights, 'split': str(SPLITS), 'cache': str(CACHE),
               'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                            'smp': smp.__version__, 'timm': timm.__version__},
               'gpu_start': assert_on_gpu(uuid)}
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))

    spec = importlib.util.spec_from_file_location('upstream_train', UPSTREAM / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    assert train.DiceBCELoss is bw.DiceBCELoss and train.FilamentCrops is FilamentCrops
    train.DiceBCELoss = loss_cls
    epochs = 1 if args.smoke else 30
    averager = None
    if args.arm == 'emaswa':
        averager = WeightAverager(epochs)
        install_averager(train, averager)
    elif args.arm == 'copypaste':
        CopyPasteCrops.report_dir = out
        train.FilamentCrops = CopyPasteCrops
    epoch_args = ['--epochs', '1', '--limit-val', '2', '--crops-per-image', '1'] if args.smoke else \
                 ['--epochs', '30', '--limit-val', '24']
    sys.argv = ['train.py', *epoch_args, '--encoder', ENCODER, '--batch-size', '16', '--workers', '4',
                '--seed', str(args.seed), '--split', str(SPLITS), '--cache-dir', str(CACHE),
                '--out', str(out / 'model.pt')]
    runtime['argv'] = sys.argv
    train.main()

    runtime['loss']['forward_calls'] = bw.BoundaryDiceBCELoss.calls
    if bw.BoundaryDiceBCELoss.calls == 0:
        raise RuntimeError('boundary loss was never called: loss injection failed')
    checkpoints = {'model_last.pt': sha256(out / 'model_last.pt')}
    if averager is not None:
        if averager.ema_updates == 0 or averager.swa_count != min(SWA_EPOCHS, epochs):
            raise RuntimeError(f'averaging incomplete: {averager.ema_updates} EMA updates, {averager.swa_count} SWA epochs')
        runtime['averaging'] = {'ema_updates': averager.ema_updates, 'swa_epochs': averager.swa_epochs_used}
        checkpoints.update({name: sha256(out / name) for name in ('model_ema.pt', 'model_swa.pt')})
    if args.arm == 'copypaste':
        if CopyPasteCrops.report.get('self_check_pasted', 0) == 0:
            raise RuntimeError(f'copy-paste never pasted in the self-check: {CopyPasteCrops.report}')
        runtime['copypaste'] = CopyPasteCrops.report
    runtime.update({'gpu_end': assert_on_gpu(uuid),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'checkpoints_sha256': checkpoints})
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    main()
