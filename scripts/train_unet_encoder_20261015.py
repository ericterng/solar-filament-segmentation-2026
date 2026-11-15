"""Train the ameyypawar U-Net recipe with a chosen ImageNet encoder (experiment 2, 2026-10-15).

Pre-registered in docs/experiments/unet_encoder_20261015.md before any run.
Vendored upstream scripts/train.py runs unmodified; it already takes --encoder
and passes it to filament_seg.model.build_model(in_channels=2,
weights="imagenet"), and load_trained() reads the encoder back from the
checkpoint, so cache_logits.py needs no change either. This wrapper only:
- refuses to start unless CUDA_VISIBLE_DEVICES is the single GPU UUID given to
  the launcher (UNET_ENCODER_GPU_UUID), and checks through nvidia-smi that this
  PID really runs on that UUID, at the start and at the end;
- refuses to start unless the pinned smp-hub weight file is already in the
  offline Hugging Face cache (HF_HUB_OFFLINE=1), and records its SHA-256;
- writes runtime.json (encoder, seed, precision, versions, GPU, peak memory).
Precision is upstream's default: AMP (torch.autocast + GradScaler) on CUDA, the
same for both encoders. Recipe identical to the local ResNet34 seeds: 30
epochs, batch 16, workers 4, limit-val 24, crop 512, lr 1e-3, last checkpoint,
fold1 splits.json.
Remote (A5000): CUDA_VISIBLE_DEVICES=<UUID> UNET_ENCODER_GPU_UUID=<UUID> HF_HUB_OFFLINE=1 \
    env/bin/python scripts/train_unet_encoder_20261015.py --encoder E --seed N --out DIR [--smoke]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/ameyypawar-filament-seg'
sys.path.insert(0, str(UPSTREAM))

import torch  # noqa: E402

CACHE = ROOT / 'outputs/amey_unet_cache'
SPLITS = ROOT / 'outputs/amey_unet_fold1_20261010/splits.json'
ENCODERS = ('resnet34', 'efficientnet-b4')
#: Pinned by segmentation_models_pytorch==0.5.0 (encoders/resnet.py, encoders/efficientnet.py).
WEIGHTS = {
    'resnet34': ('smp-hub/resnet34.imagenet', '7a57b34f723329ff020b3f8bc41771163c519d0c',
                 '6549ccd46a2d20a0a74fca866cfc5c6f9b3672d468b0a92aa94a58545a61103e'),
    'efficientnet-b4': ('smp-hub/efficientnet-b4.imagenet', '05cd5dde5dab658f00c463f9b9aa0ced76784f40',
                        'c465ce05723692769f0d069312aaf8fcdf9a6f5153e48127f5b7eae68e4ec1dd'),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_weights(encoder: str) -> dict:
    """The pinned weight file must already be cached; never download here."""
    from huggingface_hub import hf_hub_download

    repo, revision, expected = WEIGHTS[encoder]
    path = Path(hf_hub_download(repo, filename='model.safetensors', revision=revision, local_files_only=True))
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f'{repo}@{revision} sha256 {actual} != pinned {expected}')
    return {'repo_id': repo, 'revision': revision, 'sha256': actual, 'path': str(path)}


def expected_uuid() -> str:
    uuid = os.environ.get('UNET_ENCODER_GPU_UUID', '')
    if not uuid.startswith('GPU-') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise RuntimeError('CUDA_VISIBLE_DEVICES must equal UNET_ENCODER_GPU_UUID (one GPU UUID)')
    return uuid


def assert_on_gpu(uuid: str) -> dict:
    """Map this PID to physical GPUs through nvidia-smi, not through device labels."""
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f'expected exactly one visible GPU, found {torch.cuda.device_count()}')
    torch.empty(1, device='cuda:0')
    torch.cuda.synchronize()
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    actual = {line.split(', ', 1)[1].strip() for line in output.strip().splitlines()
              if line.split(', ', 1)[0].strip() == str(os.getpid())}
    if actual != {uuid}:
        raise RuntimeError(f'PID {os.getpid()} runs on {actual}; expected only {uuid}')
    return {'pid': os.getpid(), 'uuid': uuid, 'name': torch.cuda.get_device_name(0),
            'verified_by': 'nvidia-smi compute-apps pid->gpu_uuid'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=ENCODERS, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    uuid = expected_uuid()
    if os.environ.get('HF_HUB_OFFLINE') != '1':
        raise RuntimeError('HF_HUB_OFFLINE=1 is required (weights must be pinned and already cached)')
    weights = check_weights(args.encoder)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    import segmentation_models_pytorch as smp
    import timm
    runtime = {'encoder': args.encoder, 'seed': args.seed, 'smoke': args.smoke,
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
    epochs = ['--epochs', '1', '--limit-val', '2', '--crops-per-image', '1'] if args.smoke else \
             ['--epochs', '30', '--limit-val', '24']
    sys.argv = ['train.py', *epochs, '--encoder', args.encoder, '--batch-size', '16', '--workers', '4',
                '--seed', str(args.seed), '--split', str(SPLITS), '--cache-dir', str(CACHE),
                '--out', str(out / 'model.pt')]
    runtime['argv'] = sys.argv
    train.main()
    runtime.update({'gpu_end': assert_on_gpu(uuid),
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'model_last_sha256': sha256(out / 'model_last.pt')})
    (out / 'runtime.json').write_text(json.dumps(runtime, indent=2))


if __name__ == '__main__':
    main()
