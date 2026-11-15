"""Probe 16: EfficientNet-B4 U-Net with boundary-weighted BCE (pre-registered 2026-10-19).

docs/experiments/unet_boundary_20261019.md. The ONLY change relative to the saved
2026-10-15 efficientnet-b4 runs (train_unet_encoder_20261015.py) is the loss: the
BCE term becomes a pixel-weighted BCE whose weight is 1 + WEIGHT inside a band of
BAND px on either side of the filament boundary (band = dilate(y) - erode(y),
computed on the GPU from the competition mask with max_pool2d). The Dice term is
upstream's, verbatim. BAND = 2, WEIGHT = 4 are fixed in advance and not swept.
Injection: upstream scripts/train.py is loaded as a module and its DiceBCELoss
name is rebound to BoundaryDiceBCELoss; the upstream file is not modified. The
wrapper counts loss forward calls and writes them to runtime.json (0 = the
injection did not take effect = failure).
Recipe otherwise identical: 30 epochs, batch 16, workers 4, limit-val 24, crop
512, lr 1e-3, last checkpoint, fold1 splits.json, upstream AMP on CUDA.
Remote (A5000): CUDA_VISIBLE_DEVICES=<UUID> UNET_ENCODER_GPU_UUID=<UUID> HF_HUB_OFFLINE=1 \
    env/bin/python scripts/train_unet_boundary_20261019.py --seed N --out DIR [--smoke]
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_unet_encoder_20261015 as base  # noqa: E402
from train_unet_encoder_20261015 import CACHE, SPLITS, UPSTREAM, assert_on_gpu, expected_uuid, sha256, torch  # noqa: E402

import torch.nn.functional as F  # noqa: E402
from filament_seg.model import DiceBCELoss  # noqa: E402

ENCODER = 'efficientnet-b4'
BAND, WEIGHT = 2, 4.0


def boundary_band(target: torch.Tensor, band: int = BAND) -> torch.Tensor:
    """1 within `band` px (Chebyshev) of the filament boundary, on both sides; 0 elsewhere. N,1,H,W."""
    t = target.float()
    k = 2 * band + 1
    dilated = F.max_pool2d(t, k, stride=1, padding=band)
    eroded = 1.0 - F.max_pool2d(1.0 - t, k, stride=1, padding=band)
    return (dilated - eroded).clamp_(0.0, 1.0)


class BoundaryDiceBCELoss(DiceBCELoss):
    """Upstream DiceBCELoss with the BCE term replaced by a boundary-weighted mean BCE."""

    calls = 0

    def __init__(self, *args, band: int = BAND, weight: float = WEIGHT, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.band, self.weight = band, weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        BoundaryDiceBCELoss.calls += 1
        w = 1.0 + self.weight * boundary_band(target, self.band)
        bce = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none')
        bce_loss = (w * bce).sum() / w.sum()

        # Dice / Tversky term: upstream DiceBCELoss.forward, verbatim.
        probs = torch.sigmoid(logits).flatten(1)
        target_flat = target.flatten(1)
        intersection = (probs * target_flat).sum(dim=1)
        false_pos = probs.sum(dim=1) - intersection
        false_neg = target_flat.sum(dim=1) - intersection
        denominator = (2.0 * intersection + 2.0 * (1.0 - self.fn_weight) * false_pos
                       + 2.0 * self.fn_weight * false_neg)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


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
    runtime = {'encoder': ENCODER, 'seed': args.seed, 'smoke': args.smoke,
               'loss': {'name': 'boundary_weighted_bce', 'band_px': BAND, 'weight': WEIGHT,
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
    assert train.DiceBCELoss is DiceBCELoss
    train.DiceBCELoss = BoundaryDiceBCELoss  # criterion = DiceBCELoss(fn_weight=...) resolves this name at call time
    epochs = ['--epochs', '1', '--limit-val', '2', '--crops-per-image', '1'] if args.smoke else \
             ['--epochs', '30', '--limit-val', '24']
    sys.argv = ['train.py', *epochs, '--encoder', ENCODER, '--batch-size', '16', '--workers', '4',
                '--seed', str(args.seed), '--split', str(SPLITS), '--cache-dir', str(CACHE),
                '--out', str(out / 'model.pt')]
    runtime['argv'] = sys.argv
    train.main()
    runtime['loss']['forward_calls'] = BoundaryDiceBCELoss.calls
    if BoundaryDiceBCELoss.calls == 0:
        raise RuntimeError('BoundaryDiceBCELoss was never called: loss injection failed')
    runtime.update({'gpu_end': assert_on_gpu(uuid),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'model_last_sha256': sha256(out / 'model_last.pt')})
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    main()
