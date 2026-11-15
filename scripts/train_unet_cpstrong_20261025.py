"""Probe 23: stronger copy-paste on top of probe 20's copy-paste U-Net (pre-registered 2026-10-25).

docs/experiments/unet_copypaste_strong_20261025.md. Same recipe as probe 20's
copypaste arm (train_unet_last3_20261024.py: EfficientNet-B4, boundary-weighted
BCE band 2 / weight 1+4, 30 epochs, batch 16, crop 512, lr 1e-3, last checkpoint,
fold1, donor bank and paste rules unchanged) with ONE change, the paste dose:
- probe 20: with probability .5 per crop, try one paste (about .5 filaments per crop).
- probe 23: every crop draws N uniform in {CP_N_MIN..CP_N_MAX} and tries N pastes in
  turn (mean 2 tries per crop, 4x probe 20). Each paste sees the mask updated by the
  earlier ones, so the CP_GAP / disk rules also hold between pasted filaments.
The paste RNG stays separate from the base crop RNG, so the base crop sequence is
the same as bw's and probe 20's. The saved copypaste_s0 / copypaste_s1 runs are the
control (not retrained). Constants are fixed in advance and not swept. Upstream
files and the probe 19-21 wrapper are not modified; the wrapper is imported.
Remote (A5000): CUDA_VISIBLE_DEVICES=<UUID> UNET_ENCODER_GPU_UUID=<UUID> HF_HUB_OFFLINE=1 \
    env/bin/python scripts/train_unet_cpstrong_20261025.py --seed N --out DIR [--smoke]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unet_last3_20261024 as last3  # noqa: E402
from train_unet_last3_20261024 import bw, base, CACHE, SPLITS, UPSTREAM, assert_on_gpu, expected_uuid, sha256, torch  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from filament_seg.dataset import _AUGMENT, FilamentCrops  # noqa: E402

ARM = 'cpstrong'
ENCODER = last3.ENCODER
CP_N_MIN, CP_N_MAX = 1, 3


def paste_many(rng, flat: np.ndarray, radius: np.ndarray, mask: np.ndarray, bank: list,
               n: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Try n pastes in turn (probe 20's paste_one each); return the crop, mask and #pasted."""
    pasted = 0
    for _ in range(n):
        donor = bank[rng.randrange(len(bank))]
        flat, mask, ok = last3.paste_one(rng, flat, radius, mask, donor)
        pasted += ok
    return flat, mask, pasted


class StrongCopyPasteCrops(last3.CopyPasteCrops):
    """Probe 20's CopyPasteCrops with N in {CP_N_MIN..CP_N_MAX} paste tries on every crop."""

    report_dir: Path | None = None

    def _item(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Upstream FilamentCrops.__getitem__, verbatim except for the paste step (as in probe 20).
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

        paste_rng = self._paste_rng()
        n = paste_rng.randint(CP_N_MIN, CP_N_MAX)
        flat_crop, mask_crop, pasted = paste_many(paste_rng, flat_crop, radius_crop, mask_crop, self.bank, n)
        self._last_tries = n

        if self.augment:
            out = _AUGMENT(image=flat_crop, mask=mask_crop, radius=radius_crop)
            flat_crop, mask_crop, radius_crop = out["image"], out["mask"], out["radius"]

        x = np.stack([flat_crop, radius_crop], axis=0).astype(np.float32)
        y = mask_crop[None, :, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), pasted

    def _self_check(self, n: int = 200) -> dict:
        """Tries / pasted per crop on n crops in the main process (key 0, never used by workers)."""
        saved_aug, self.augment = self.augment, False
        tries, pasted, examples = Counter(), Counter(), []
        for i in range(n):
            x, y, k = self._item((i * 7919) % len(self))
            tries[self._last_tries] += 1
            pasted[k] += 1
            if k >= 2 and len(examples) < 4:
                examples.append((x[0].numpy(), y[0].numpy()))
        self.augment = saved_aug
        if self.report_dir is not None:
            for j, (img, msk) in enumerate(examples):
                rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                edge = (cv2.morphologyEx((msk > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)
                rgb[edge] = (0, 0, 255)
                cv2.imwrite(str(self.report_dir / f'cpstrong_example_{j}.png'), rgb)
        total_tries = sum(k * v for k, v in tries.items())
        total_pasted = sum(k * v for k, v in pasted.items())
        return {'self_check_crops': n,
                'self_check_tries_hist': {str(k): tries[k] for k in range(CP_N_MIN, CP_N_MAX + 1)},
                'self_check_pasted_hist': {str(k): pasted[k] for k in range(0, CP_N_MAX + 1)},
                'self_check_tries': total_tries, 'self_check_pasted': total_pasted,
                'self_check_pasted_per_crop': total_pasted / n,
                'self_check_paste_success': total_pasted / max(total_tries, 1),
                'self_check_paste_rate': sum(v for k, v in pasted.items() if k > 0) / n}


def main() -> None:
    parser = argparse.ArgumentParser()
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
    arm_settings = {'n_tries': [CP_N_MIN, CP_N_MAX], 'n_distribution': 'uniform', 'prob': 1.0,
                    'max_side': last3.CP_MAX_SIDE, 'min_area': last3.CP_MIN_AREA,
                    'radius_max': last3.CP_RADIUS_MAX, 'gap_px': last3.CP_GAP, 'tries': last3.CP_TRIES,
                    'ring_px': [last3.CP_RING_IN, last3.CP_RING_OUT]}
    runtime = {'arm': ARM, 'arm_settings': arm_settings, 'encoder': ENCODER, 'seed': args.seed,
               'smoke': args.smoke,
               'loss': {'name': 'boundary_weighted_bce', 'band_px': bw.BAND, 'weight': bw.WEIGHT,
                        'dice': 'upstream, unchanged', 'forward_calls': 0},
               'precision': 'upstream AMP (autocast fp16 + GradScaler) on CUDA',
               'weights': weights, 'split': str(SPLITS), 'cache': str(CACHE),
               'wrapper_sha256': {'cpstrong': sha256(Path(__file__)),
                                  'last3': sha256(Path(last3.__file__))},
               'versions': {'python': sys.version.split()[0], 'torch': torch.__version__,
                            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                            'smp': smp.__version__, 'timm': timm.__version__},
               'gpu_start': assert_on_gpu(uuid)}
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))

    spec = importlib.util.spec_from_file_location('upstream_train', UPSTREAM / 'scripts/train.py')
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    assert train.DiceBCELoss is bw.DiceBCELoss and train.FilamentCrops is FilamentCrops
    train.DiceBCELoss = bw.BoundaryDiceBCELoss
    StrongCopyPasteCrops.report_dir = out
    train.FilamentCrops = StrongCopyPasteCrops
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
    report = last3.CopyPasteCrops.report  # set by CopyPasteCrops.__init__, with this class's self-check
    if report.get('self_check_pasted', 0) == 0 or 'self_check_pasted_hist' not in report:
        raise RuntimeError(f'strong copy-paste self-check missing or never pasted: {report}')
    runtime['copypaste'] = report
    runtime.update({'gpu_end': assert_on_gpu(uuid),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'checkpoints_sha256': {'model_last.pt': sha256(out / 'model_last.pt')}})
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    main()
