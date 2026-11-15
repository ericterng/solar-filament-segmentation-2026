"""Probe 28: apply a trained refiner to the kept instances of an instances json (§B). Runs on the A5000.

For every image, every kept instance gets its 512-tile(s) (core.tile_origins), the 3-channel input
[flat, radius, this instance's mask] through the refiner with dihedral-8 TTA (upstream order, logits
averaged), prob >= .5 inside the tile union, components touching the original minus every other kept
instance's original (core.refine_in_box), then core.resolve over the image in fusion order. Output
keeps the input order and count; `--identity` skips the network and must reproduce the input exactly.
Val: `--apply-a CKPT_TRAINED_ON_B --apply-b CKPT_TRAINED_ON_A` (checked against each checkpoint's
train_half); test: `--apply-all CKPT_TRAINED_ON_ALL`. Each of the three takes several checkpoints
(probe 29): their logits are averaged before the sigmoid, one checkpoint runs as before.
`--all-candidates` (probe 31) treats every candidate of the json as kept; the output has them all.
`--equivalence-check` runs the untrained 3-channel
model through upstream's tiled dihedral inference on a few val images and compares with the cached
cpstrong logits (the real-data initialisation gate). Never imports `solar`.

    env/bin/python scripts/apply_refiner_20261031.py --instances outputs/refiner_20261031/instances_val_s0.json \
        --apply-a outputs/refiner_20261031/r_s0_trainB_r0/model_last.pt \
        --apply-b outputs/refiner_20261031/r_s0_trainA_r0/model_last.pt --out outputs/refiner_20261031/refined_val_s0_r0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_refiner_20261031 import ENCODER, IMAGE_SHAPE, OOF_HALF, OOF_PROBE, make_refiner, no_tf32, pick_device, sha256  # noqa: E402
from filament_seg.dataset import load_disk_geometry, load_model_input  # noqa: E402
from filament_seg.model import build_model, load_trained, tiled_predict_tta  # noqa: E402
import refiner_core_20261031 as core  # noqa: E402

#: Real-data equivalence gate: 3-channel model with a zero third channel vs the 2-channel original
#: (same run, fp32) and vs the cached float16 logits (different run, float16 rounding).
EQ_VS_MODEL2_MAX, EQ_VS_CACHE_MAX, EQ_VS_CACHE_MEAN = 1e-3, 5e-2, 5e-3


def load_refiner(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(str(path), map_location=device)
    meta = payload['refiner']
    assert payload['encoder'] == ENCODER and payload['in_channels'] == 3 and meta['probe'] in (28, OOF_PROBE), payload.keys()
    model = build_model(encoder=ENCODER, in_channels=3, weights=None).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    return model, meta


class AverageLogits(nn.Module):
    """Probe 29: several refiners behind one forward; logits are averaged, never probabilities."""

    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x).float() for m in self.models]).mean(dim=0)


def load_refiners(paths: list[Path], device: torch.device) -> tuple[nn.Module, list[dict]]:
    """One checkpoint: the plain model. Several: AverageLogits over distinct refiner seeds of one (U-Net seed, half)."""
    loaded = [load_refiner(p, device) for p in paths]
    models, metas = [m for m, _ in loaded], [meta for _, meta in loaded]
    assert len({m['refiner_seed'] for m in metas}) == len(metas), metas
    assert len({(m['unet_seed'], m['train_half']) for m in metas}) == 1, metas
    return (models[0] if len(models) == 1 else AverageLogits(models).to(device).eval()), metas


def check_apply_all(metas: list[dict], instances_meta: dict) -> None:
    """--apply-all leak guard: a json with date halves is fold1, and only probe 30's models never saw fold1."""
    assert all(m['train_half'] in ('all', OOF_HALF) for m in metas), metas
    if 'halves' in instances_meta:
        assert all(m['probe'] == OOF_PROBE and m['train_half'] == OOF_HALF for m in metas), \
            ('a refiner trained on fold1 must not be applied to fold1', metas)


class ZeroThirdChannel(nn.Module):
    """Feed a 2-channel batch to the 3-channel model with an all-zero third channel."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([x, torch.zeros_like(x[:, :1])], dim=1))


@torch.no_grad()
def refine_instance(model: nn.Module, x2: np.ndarray, original: np.ndarray, others: np.ndarray,
                    device: torch.device, batch: int) -> tuple[np.ndarray, int]:
    """§B 1-3 for one instance: returns (refined full-size bool mask, number of tiles)."""
    t = core.TILE
    origins = core.tile_origins(core.bbox_of(original), original.shape)
    by0, by1, bx0, bx1 = box = core.tile_union_box(origins)
    region = core.tile_union_mask(origins, box)
    logit_sum = np.zeros((by1 - by0, bx1 - bx0), np.float32)
    count = np.zeros_like(logit_sum)
    inputs, where = [], []
    for ty, tx in origins:
        x3 = np.concatenate([x2[:, ty:ty + t, tx:tx + t], original[None, ty:ty + t, tx:tx + t].astype(np.float32)])
        for k in range(8):
            inputs.append(core.orient(x3, k))
            where.append((ty, tx, k))
    outputs = []
    for i in range(0, len(inputs), batch):
        xb = torch.from_numpy(np.stack(inputs[i:i + batch])).to(device)
        outputs.append(model(xb)[:, 0].float().cpu().numpy())
    outputs = np.concatenate(outputs)
    per_tile: dict[tuple[int, int], np.ndarray] = {}
    for out, (ty, tx, k) in zip(outputs, where):
        per_tile.setdefault((ty, tx), np.zeros((t, t), np.float32))
        per_tile[(ty, tx)] += core.unorient(out, k) / 8.0
    for (ty, tx), logits in per_tile.items():
        logit_sum[ty - by0:ty - by0 + t, tx - bx0:tx - bx0 + t] += logits
        count[ty - by0:ty - by0 + t, tx - bx0:tx - bx0 + t] += 1
    prob = np.where(count > 0, 1.0 / (1.0 + np.exp(-logit_sum / np.maximum(count, 1))), 0.0).astype(np.float32)
    refined_box = core.refine_in_box(prob, original[by0:by1, bx0:bx1], others[by0:by1, bx0:bx1], region)
    refined = np.zeros(original.shape, bool)
    refined[by0:by1, bx0:bx1] = refined_box
    return refined, len(origins)


def process_image(image: dict, model: nn.Module | None, cache: Path, disks: dict, device: torch.device,
                  batch: int, identity: bool, all_candidates: bool = False) -> dict:
    kept = [inst for inst in image['instances'] if all_candidates or inst.get('kept', True)]
    order = sorted(range(len(kept)), key=lambda i: kept[i]['order'])
    originals = [core.decode(inst['rle']) for inst in kept]
    for o in originals:
        assert o.shape == IMAGE_SHAPE and o.any(), image['stem']
    core.assert_disjoint(originals)
    n_tiles = [0] * len(kept)
    if identity or not kept:
        refined = [o.copy() for o in originals]
    else:
        x2 = load_model_input(cache / 'flat', image['stem'], disks[image['stem']])
        assert x2 is not None and x2.shape == (2, *IMAGE_SHAPE), image['stem']
        any_original = np.zeros(IMAGE_SHAPE, bool)
        for o in originals:
            any_original |= o
        refined = []
        for i, o in enumerate(originals):
            r, n_tiles[i] = refine_instance(model, x2, o, any_original & ~o, device, batch)
            refined.append(r)
    finals, info = core.resolve(originals, refined, order)
    if identity:
        assert all(not inf['changed'] for inf in info), image['stem']
    core.assert_disjoint(finals)
    out_instances = []
    for inst, final, inf, nt in zip(kept, finals, info, n_tiles):
        rle = core.rle_json(core.encode(final))
        if identity:
            assert rle['counts'] == core.rle_json(inst['rle'])['counts'], image['stem']
        out_instances.append({**{k: inst[k] for k in inst if k != 'rle'}, 'original_rle': core.rle_json(inst['rle']),
                              'rle': rle, 'n_tiles': nt, **inf})
    return {**{k: image[k] for k in image if k != 'instances'}, 'instances': out_instances}


def equivalence_on_real_data(init_ckpt: Path, stems: list[str], logits_dir: Path, cache: Path, device: torch.device) -> dict:
    no_tf32()
    disks = load_disk_geometry(cache / 'disk.json')
    model2 = load_trained(str(init_ckpt), device)
    model3, _, random_report = make_refiner(init_ckpt, device)
    wrapped = ZeroThirdChannel(model3).to(device).eval()
    rows = {}
    for stem in stems:
        x2 = load_model_input(cache / 'flat', stem, disks[stem])
        assert x2 is not None, stem
        l2 = tiled_predict_tta(model2, x2, 'dihedral', tile=512, overlap=128, device=device)
        l3 = tiled_predict_tta(wrapped, x2, 'dihedral', tile=512, overlap=128, device=device)
        cached = np.load(logits_dir / f'{stem}.npy').astype(np.float32)
        d32, d3c, d2c = np.abs(l3 - l2), np.abs(l3 - cached), np.abs(l2 - cached)
        rows[stem] = {'max_abs_vs_model2': float(d32.max()), 'max_abs_vs_cache': float(d3c.max()),
                      'mean_abs_vs_cache': float(d3c.mean()), 'model2_max_abs_vs_cache_info': float(d2c.max()),
                      'cache_dtype': str(np.load(logits_dir / f'{stem}.npy').dtype)}
    ok = all(r['max_abs_vs_model2'] <= EQ_VS_MODEL2_MAX and r['max_abs_vs_cache'] <= EQ_VS_CACHE_MAX
             and r['mean_abs_vs_cache'] <= EQ_VS_CACHE_MEAN for r in rows.values())
    return {'random_input': random_report, 'per_stem': rows, 'tolerances': {'vs_model2_max': EQ_VS_MODEL2_MAX,
            'vs_cache_max': EQ_VS_CACHE_MAX, 'vs_cache_mean': EQ_VS_CACHE_MEAN}, 'passed': ok}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cache', type=Path, default=HERE.parent / 'outputs/amey_unet_cache')
    parser.add_argument('--apply-a', type=Path, nargs='+', help='checkpoint(s) applied to half-A images (trained on half B)')
    parser.add_argument('--apply-b', type=Path, nargs='+', help='checkpoint(s) applied to half-B images (trained on half A)')
    parser.add_argument('--apply-all', type=Path, nargs='+', help='checkpoint(s) applied to every image (test; trained on all)')
    parser.add_argument('--identity', action='store_true')
    parser.add_argument('--all-candidates', action='store_true', help='probe 31: refine the dropped candidates too')
    parser.add_argument('--limit', type=int, help='smoke: first N images only')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--equivalence-check', nargs='*', metavar='STEM')
    parser.add_argument('--init-ckpt', type=Path)
    parser.add_argument('--logits-dir', type=Path)
    parser.add_argument('--cpu', action='store_true', help='local smoke only')
    args = parser.parse_args()
    device = pick_device(args.cpu)
    if args.out is None or args.out.exists():
        raise FileExistsError(args.out)

    if args.equivalence_check is not None:
        report = equivalence_on_real_data(args.init_ckpt, args.equivalence_check, args.logits_dir, args.cache, device)
        args.out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('EQUIVALENCE', json.dumps(report))
        if not report['passed']:
            raise SystemExit('real-data equivalence check failed')
        return

    blob = json.loads(args.instances.read_text(encoding='utf-8'))
    images = blob['images'][:args.limit] if args.limit else blob['images']
    disks = load_disk_geometry(args.cache / 'disk.json')
    models: dict[str, tuple[nn.Module, list[dict]]] = {}
    if not args.identity:
        if args.apply_all:
            models['all'] = load_refiners(args.apply_all, device)
            check_apply_all(models['all'][1], blob['meta'])
        else:
            assert args.apply_a and args.apply_b, 'val needs --apply-a and --apply-b'
            models['A'], models['B'] = load_refiners(args.apply_a, device), load_refiners(args.apply_b, device)
            assert all(m['train_half'] == 'B' for m in models['A'][1]) and all(m['train_half'] == 'A' for m in models['B'][1]), \
                (models['A'][1], models['B'][1])  # leak guard: never apply a refiner to the half it was trained on
        for _, metas in models.values():
            for m in metas:
                assert m['unet_seed'] is None or m['unet_seed'] == blob['meta'].get('unet_seed', m['unet_seed']), (m, blob['meta'])
    started, out_images = time.time(), []
    for n, image in enumerate(images, 1):
        if args.identity:
            model = None
        elif 'all' in models:
            model = models['all'][0]
        else:
            model = models[image['half']][0]
        out_images.append(process_image(image, model, args.cache, disks, device, args.batch, args.identity, args.all_candidates))
        if n % 10 == 0 or n == len(images):
            print(f'{n}/{len(images)} {time.time() - started:.0f}s', flush=True)
    inst = [i for im in out_images for i in im['instances']]
    summary = {'images': len(out_images), 'instances': len(inst), 'changed': sum(i['changed'] for i in inst),
               'fallback': sum(i['fallback'] for i in inst),
               'iou_original_final_quartiles': [float(q) for q in np.percentile([i['iou_original_final'] for i in inst], [25, 50, 75])] if inst else None,
               'tiles_hist': {str(k): int(v) for k, v in zip(*np.unique([i['n_tiles'] for i in inst], return_counts=True))} if inst else {},
               'identity': args.identity, 'all_candidates': args.all_candidates, 'limit': args.limit, 'seconds': time.time() - started,
               'models': {k: [{'path': str(p), 'sha256': sha256(p), 'meta': meta} for p, meta in zip(paths, models[k][1])]
                          for k, paths in (('A', args.apply_a), ('B', args.apply_b), ('all', args.apply_all)) if paths},
               'instances_sha256': sha256(args.instances), 'script_sha256': sha256(Path(__file__)),
               'core_sha256': sha256(HERE / 'refiner_core_20261031.py')}
    args.out.write_text(json.dumps({'meta': {**blob.get('meta', {}), 'apply': summary}, 'images': out_images}),
                        encoding='utf-8')
    print('DONE', json.dumps({k: summary[k] for k in ('images', 'instances', 'changed', 'fallback', 'identity')}))


if __name__ == '__main__':
    main()
