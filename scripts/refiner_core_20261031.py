"""Probe 28 (docs/experiments/instance_refiner_20261031.md): the pure-numpy rules shared by every stage.

Runs on the A5000 and locally, so it depends on numpy / cv2 / pycocotools only -- never on torch,
`solar` or the upstream package. Everything a test can pin lives here:
- tile placement (§B 1): one 512 tile centred on the instance box when its longest side is <= 320 px,
  otherwise a 512 / stride-384 grid centred on the box that covers the box grown by 96 px;
- the training crop origin (§A): box centre + uniform +-96 px jitter, clipped into the image;
- the component rule (§B 3): prob >= .5 inside the tile union, 8-connected components touching the
  original, minus every other kept instance's original mask;
- conflict resolution (§B 4-5): only pixels outside every original compete, first-come in fusion
  order; a result under 200 px falls back to the original, which nothing else can ever take;
- the sample / oracle target: per (candidate, annotation record) the record's best-IoU GT, kept when
  IoU >= .25;
- the 2x2 decision rule.
"""
from __future__ import annotations

import math
from itertools import product

import cv2
import numpy as np
from pycocotools import mask as coco

TILE, STRIDE, MARGIN, SINGLE_MAX, JITTER = 512, 384, 96, 320, 96
PROB_THRESHOLD, MIN_AREA = .5, 200
IOU_MIN, NEAR_MISS = .25, (.25, .5)
TRAIN_LR, TRAIN_EPOCHS, TRAIN_BATCH = 1e-4, 16, 8
OLD_MARGIN, UPLOAD_MARGIN = .005, .002
ORACLE_GATE = .005
#: Upstream model.TTA_MODES['dihedral'] order: (flip_lr, flip_ud, transpose).
DIHEDRAL = [(flip_lr, flip_ud, transpose) for transpose in (False, True)
            for flip_lr in (False, True) for flip_ud in (False, True)]


# ----------------------------------------------------------------------------- RLE
def rle_bytes(rle: dict) -> dict:
    c = rle['counts']
    return {'size': list(rle['size']), 'counts': c if isinstance(c, bytes) else c.encode('ascii')}


def rle_json(rle: dict) -> dict:
    c = rle['counts']
    return {'size': list(rle['size']), 'counts': c.decode('ascii') if isinstance(c, bytes) else c}


def decode(rle: dict) -> np.ndarray:
    return coco.decode(rle_bytes(rle)).astype(bool)


def encode(mask: np.ndarray) -> dict:
    return coco.encode(np.asfortranarray(mask.astype(np.uint8)))


def iou_matrix(a: list[dict], b: list[dict]) -> np.ndarray:
    """(len(a), len(b)) IoU, zeros when either side is empty."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(coco.iou([rle_bytes(r) for r in a], [rle_bytes(r) for r in b], [0] * len(b)))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 1.0


# ----------------------------------------------------------------------------- geometry
def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1), half-open, of a non-empty mask."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError('empty mask')
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def clip_origin(origin: int, size: int, limit: int) -> int:
    return int(min(max(origin, 0), max(limit - size, 0)))


def _axis_origins(lo: int, hi: int, limit: int, tile: int, stride: int, margin: int) -> list[int]:
    centre = (lo + hi) / 2
    extent = (hi - lo) + 2 * margin
    n = 1 if extent <= tile else int(math.ceil((extent - tile) / stride)) + 1
    length = tile + (n - 1) * stride
    start = int(round(centre - length / 2))
    return sorted({clip_origin(start + i * stride, tile, limit) for i in range(n)})


def tile_origins(bbox: tuple[int, int, int, int], shape: tuple[int, int], tile: int = TILE, stride: int = STRIDE,
                 margin: int = MARGIN, single_max: int = SINGLE_MAX) -> list[tuple[int, int]]:
    """§B 1: (y0, x0) of every tile for one instance; one centred tile for boxes <= single_max px."""
    y0, y1, x0, x1 = bbox
    height, width = shape
    if max(y1 - y0, x1 - x0) <= single_max:
        return [(clip_origin(int(round((y0 + y1) / 2 - tile / 2)), tile, height),
                 clip_origin(int(round((x0 + x1) / 2 - tile / 2)), tile, width))]
    ys = _axis_origins(y0, y1, height, tile, stride, margin)
    xs = _axis_origins(x0, x1, width, tile, stride, margin)
    return list(product(ys, xs))


def tile_union_box(origins: list[tuple[int, int]], tile: int = TILE) -> tuple[int, int, int, int]:
    """Bounding box (y0, y1, x0, x1) of the tile union."""
    ys = [o[0] for o in origins]
    xs = [o[1] for o in origins]
    return min(ys), max(ys) + tile, min(xs), max(xs) + tile


def tile_union_mask(origins: list[tuple[int, int]], box: tuple[int, int, int, int], tile: int = TILE) -> np.ndarray:
    """Boolean mask of the tile union, in the coordinates of `box`."""
    y0, y1, x0, x1 = box
    out = np.zeros((y1 - y0, x1 - x0), bool)
    for ty, tx in origins:
        out[ty - y0:ty - y0 + tile, tx - x0:tx - x0 + tile] = True
    return out


def sample_crop_origin(bbox: tuple[int, int, int, int], shape: tuple[int, int], rng: np.random.Generator,
                       tile: int = TILE, jitter: int = JITTER) -> tuple[int, int]:
    """§A: training crop = box centre + uniform +-jitter on both axes, clipped into the image."""
    y0, y1, x0, x1 = bbox
    cy = (y0 + y1) / 2 + rng.integers(-jitter, jitter + 1)
    cx = (x0 + x1) / 2 + rng.integers(-jitter, jitter + 1)
    return (clip_origin(int(round(cy - tile / 2)), tile, shape[0]),
            clip_origin(int(round(cx - tile / 2)), tile, shape[1]))


# ----------------------------------------------------------------------------- orientation (upstream TTA order)
def orient(a: np.ndarray, k: int) -> np.ndarray:
    """Upstream model._orient for DIHEDRAL[k], on the last two axes."""
    flip_lr, flip_ud, transpose = DIHEDRAL[k]
    if transpose:
        a = np.swapaxes(a, -1, -2)
    if flip_lr:
        a = a[..., :, ::-1]
    if flip_ud:
        a = a[..., ::-1, :]
    return np.ascontiguousarray(a)


def unorient(a: np.ndarray, k: int) -> np.ndarray:
    """Upstream model._unorient for DIHEDRAL[k], on the last two axes (works for N-D too)."""
    flip_lr, flip_ud, transpose = DIHEDRAL[k]
    if flip_ud:
        a = a[..., ::-1, :]
    if flip_lr:
        a = a[..., :, ::-1]
    if transpose:
        a = np.swapaxes(a, -1, -2)
    return np.ascontiguousarray(a)


# ----------------------------------------------------------------------------- §B 3-5
def refine_in_box(prob: np.ndarray, original: np.ndarray, others: np.ndarray, region: np.ndarray,
                  threshold: float = PROB_THRESHOLD) -> np.ndarray:
    """§B 3 inside one box: components of (prob >= threshold & region) touching `original`, minus `others`.

    All four arrays share the box's shape. `original` must lie inside the box (callers guarantee it:
    the box is the tile union, which always contains the instance box). Returns a bool mask.
    """
    binary = (prob >= threshold) & region
    if not binary.any():
        return np.zeros_like(original, bool)
    n, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    touched = np.unique(labels[original & binary])
    touched = touched[touched > 0]
    if touched.size == 0:
        return np.zeros_like(original, bool)
    keep = np.isin(labels, touched)
    return keep & ~others


def resolve(originals: list[np.ndarray], refined: list[np.ndarray], order: list[int],
            min_area: int = MIN_AREA) -> tuple[list[np.ndarray], list[dict]]:
    """§B 4-5 over one image. `order[j]` is the index of the j-th instance in fusion order.

    Each refined mask must already exclude every other instance's original (§B 3); this is asserted.
    Only pixels outside every original compete, first-come in `order`. A result under `min_area`
    falls back to the original, which by construction nobody else can own. Returns the final masks
    (input index order) and one info dict per instance.
    """
    n = len(originals)
    assert len(refined) == n and sorted(order) == list(range(n)), (len(refined), n, order)
    if n == 0:
        return [], []
    shape = originals[0].shape
    any_original = np.zeros(shape, bool)
    for o in originals:
        any_original |= o
    owner = np.full(shape, -1, np.int32)
    finals: list[np.ndarray | None] = [None] * n
    info: list[dict | None] = [None] * n
    for rank, i in enumerate(order):
        o, r = originals[i], refined[i]
        others = any_original & ~o
        assert not (r & others).any(), f'refined {i} overlaps another original'
        new = r & ~o & (owner < 0)
        final = (r & o) | new
        fallback = int(final.sum()) < min_area
        if fallback:
            final = o.copy()
        owner[final & ~o] = i
        finals[i] = final
        info[i] = {'rank': rank, 'fallback': fallback, 'changed': bool((final != o).any()),
                   'area_original': int(o.sum()), 'area_refined_raw': int(r.sum()), 'area_final': int(final.sum()),
                   'iou_original_final': mask_iou(o, final)}
    return finals, info  # type: ignore[return-value]


def assert_disjoint(masks: list[np.ndarray]) -> None:
    if not masks:
        return
    stacked = np.zeros(masks[0].shape, np.int32)
    for m in masks:
        stacked += m
    assert int(stacked.max(initial=0)) <= 1, 'masks overlap'


def union_arm(originals: list[np.ndarray], finals: list[np.ndarray]) -> list[np.ndarray]:
    """Info arm U: original | final per instance; disjoint by construction (asserted)."""
    out = [o | f for o, f in zip(originals, finals)]
    assert_disjoint(out)
    return out


# ----------------------------------------------------------------------------- targets (§A samples, step 0 oracle)
def best_gt_per_record(cand_rles: list[dict], records: list[tuple[str, list[dict]]]) -> list[list[tuple[int, float]]]:
    """[candidate][record] -> (index of that record's best-IoU GT, IoU); (-1, 0.) when the record is empty."""
    out = [[(-1, 0.0) for _ in records] for _ in cand_rles]
    for j, (_, gt) in enumerate(records):
        if not gt or not cand_rles:
            continue
        iou = iou_matrix(cand_rles, gt)
        best = iou.argmax(axis=1)
        for i in range(len(cand_rles)):
            out[i][j] = (int(best[i]), float(iou[i, best[i]]))
    return out


def best_across_records(per_record: list[tuple[int, float]]) -> tuple[int, int, float]:
    """(record index, GT index, IoU) of the best record for one candidate."""
    j = int(np.argmax([iou for _, iou in per_record])) if per_record else -1
    if j < 0:
        return -1, -1, 0.0
    return j, per_record[j][0], per_record[j][1]


def oracle_replace(best_iou: float, mode: str) -> bool:
    """Which kept instances an oracle arm swaps for their target GT."""
    if mode == 'near_miss':
        return NEAR_MISS[0] <= best_iou < NEAR_MISS[1]
    if mode == 'full':
        return best_iou >= IOU_MIN
    raise ValueError(mode)


# ----------------------------------------------------------------------------- decision
def decision(cells: dict[str, float], control: dict[int, float], refiner_seeds=(0, 1)) -> dict:
    """2x2 main cells X_s{s}_r{r}; delta_s = mean_r PQ(X_s_r) - control_s."""
    deltas = {s: {r: cells[f'X_s{s}_r{r}'] - control[s] for r in refiner_seeds} for s in control}
    per_seed = {s: float(np.mean(list(v.values()))) for s, v in deltas.items()}
    mean = float(np.mean(list(per_seed.values())))
    both = all(v > 0 for v in per_seed.values())
    return {'control_pq': control, 'deltas': deltas, 'per_unet_seed_delta': per_seed, 'mean_delta': mean,
            'old_rule_success': bool(both and mean >= OLD_MARGIN),
            'upload_rule': 'both per-U-Net-seed deltas > 0 and mean >= .002 (fixed 2026-10-26)',
            'upload_ok': bool(both and mean >= UPLOAD_MARGIN)}


def oracle_gate(near_miss_delta: dict[int, float]) -> dict:
    mean = float(np.mean(list(near_miss_delta.values())))
    return {'near_miss_delta': near_miss_delta, 'mean_delta': mean, 'gate': ORACLE_GATE,
            'gate_pass': bool(mean >= ORACLE_GATE)}
