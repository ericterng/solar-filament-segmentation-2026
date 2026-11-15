"""Probe 7: cross-fitted instance rescoring with a PQ-optimal keep rule.

Pre-registered 2026-10-12, before any rescoring number was computed; design
from docs/competition/research/pq_improvement_options_20261012.md section 4.1.
Run with `.venv-unet`. No tuning beyond what is written here.
- Candidates: Option B unchanged (U-Net seed0 dihedral logits; m1536 / m1280 /
  m1024 flip-merged, then member-merged; fusion threshold .5, close 3, grow 4,
  min_area 200, keep_unclaimed 0) except min_score .15. Claims run in
  descending score, so every instance scoring >= .30 is identical to Option B.
- Six features per instance: merged score; members that found it (1-3); log
  claim area; claim area / native detection mask area; mean U-Net probability
  inside it; centroid radius / disk radius.
- Model: logistic regression (sklearn, standardized, C=1, lbfgs) on every
  (instance, annotator record) pair; target = the instance hits a GT of that
  record at IoU > .5. E[IoU | hit] = mean hit IoU per area bin (<800,
  800-3200, >=3200 px) in the training half (all-bin mean if a bin is empty).
  Keep an instance iff p * E[IoU] > PQ_train / 2, PQ_train = Option B PQ on
  the training half (Lipton et al. 2014: optimal F1 threshold = F1* / 2; RQ is
  F1). Dropped pixels are discarded. No threshold sweep.
- Cross-fitting: the 131 observation dates are split in two halves with
  numpy default_rng(20261012); fit on one half, apply to the other, both ways;
  PQ over all 141.
- Gates: tracked fusion and member counting must equal upstream fuse /
  merge_views; candidates filtered to score >= .30 must reproduce Option B
  .4305464235.
- Decision: test candidate only if cross-fitted PQ >= Option B + .005.
  Reported, not gated: date-level paired bootstrap of the delta (2000 draws,
  seed 20261012), small recall, split / merge. Reference only: all candidates
  at .15 unfiltered; one score threshold per area bin chosen from
  {.15, .20, ..., .50} on the training half and applied to the other half.
- If it passes, the model is refit on all 141 and applied to the 180 test
  images by a separate export script.
"""
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fusion_probe_unet_20261010 import CACHE, LOGITS, load_disk_geometry, score  # noqa: E402
from fusion_probe_option1_20261011 import Detection, fuse, params  # noqa: E402
from fusion_probe_ensemble_20261011 import MEMBERS, flips, load  # noqa: E402
from filament_seg.fusion import merge_views, unet_binary  # noqa: E402
from filament_seg.postprocess import _ellipse  # noqa: E402
from filament_seg.rle import counts_to_mask, rle_from_counts  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, save_json  # noqa: E402
from solar.grouped_cv import observation_date  # noqa: E402

OUT = ROOT/'outputs/rescoring_probe_20261012'
OPTION_B_PQ = .43054642350099626
MARGIN = .005
SEED = 20261012
CANDIDATE_MIN = .15
OPTION_B_MIN = .30
AREA_BINS = (800, 3200)
FEATURES = ('score', 'members', 'log_area', 'support', 'mean_prob', 'radius')
REFERENCE_GRID = (.15, .20, .25, .30, .35, .40, .45, .50)


def merge_counted(views: list[list[Detection]], iou_threshold: float = .5) -> list[tuple[Detection, int]]:
    """filament_seg.fusion.merge_views, also returning how many views found each detection."""
    pool = sorted(((det, view) for view, dets in enumerate(views) for det in dets), key=lambda item: -item[0].score)
    if not pool:
        return []
    rles = [rle_from_counts(det.counts) for det, _ in pool]
    ious = np.asarray(coco.iou(rles, rles, [0] * len(rles)))
    used = np.zeros(len(pool), dtype=bool)
    merged = []
    for i, (detection, view) in enumerate(pool):
        if used[i]:
            continue
        used[i] = True
        seen, total = {view}, detection.score
        for j in np.argsort(-ious[i], kind='stable'):
            if ious[i, j] <= iou_threshold:
                break
            if used[j] or pool[j][1] in seen:
                continue
            used[j] = True
            seen.add(pool[j][1])
            total += pool[j][0].score
        merged.append((Detection(total / len(views), detection.counts), len(seen)))
    return merged


def fuse_tracked(logits: np.ndarray, disk: np.ndarray, dets: list[Detection], min_score: float
                 ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """fuse() under params(min_score), also returning (detection index, native area) per label."""
    p = params(min_score)
    height, width = logits.shape
    binary = unet_binary(logits, disk, p.threshold, p.close_radius)
    kernel = _ellipse(p.grow)
    labels = np.zeros(logits.shape, dtype=np.int32)
    claimed = np.zeros(logits.shape, dtype=bool)
    sources = []
    for k in sorted(range(len(dets)), key=lambda k: -dets[k].score):
        if dets[k].score < p.min_score:
            break
        mask = counts_to_mask(dets[k].counts, height, width)
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        y0, y1 = max(int(ys.min()) - p.grow, 0), min(int(ys.max()) + p.grow + 1, height)
        x0, x1 = max(int(xs.min()) - p.grow, 0), min(int(xs.max()) + p.grow + 1, width)
        crop = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), kernel).astype(bool)
        window = (slice(y0, y1), slice(x0, x1))
        pixels = binary[window] & crop & ~claimed[window]
        if int(pixels.sum()) < p.min_area:
            continue
        labels[window][pixels] = len(sources) + 1
        claimed[window] |= pixels
        sources.append((k, int(ys.size)))
    return labels, sources


def candidates(stem: str, logits: np.ndarray, disk, member_views: list[list[list[dict]]]) -> list[dict]:
    """Option B instances at min_score .15, each with its six features and RLE."""
    per_member = [flips(v) for v in member_views]
    counted = merge_counted(per_member, .5)
    merged = [d for d, _ in counted]
    assert [(d.score, d.counts) for d in merged] == [(d.score, d.counts) for d in merge_views(per_member, .5)]
    disk_mask = disk.mask(logits.shape)
    labels, sources = fuse_tracked(logits, disk_mask, merged, CANDIDATE_MIN)
    assert np.array_equal(labels, fuse(logits, disk_mask, merged, params(CANDIDATE_MIN))), stem
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    n = len(sources)
    area = np.bincount(lab, minlength=n + 1)[1:]
    prob = np.bincount(lab, weights=1 / (1 + np.exp(-logits[ys, xs].astype(np.float64))), minlength=n + 1)[1:]
    cy = np.bincount(lab, weights=ys, minlength=n + 1)[1:] / area
    cx = np.bincount(lab, weights=xs, minlength=n + 1)[1:] / area
    out = []
    for i, (k, native) in enumerate(sources):
        det, members = counted[k]
        out.append({'stem': stem, 'score': det.score, 'members': members, 'area': int(area[i]),
                    'log_area': float(np.log(area[i])), 'support': float(area[i] / native),
                    'mean_prob': float(prob[i] / area[i]),
                    'radius': float(np.hypot(cx[i] - disk.cx, cy[i] - disk.cy) / disk.radius),
                    'rle': coco.encode(np.asfortranarray((labels == i + 1).astype(np.uint8)))})
    return out


def records_for(store, stem: str, inst_idx: list[int], rles: list[dict]) -> list[dict]:
    """Per annotator record: hits and hit-IoU sum of each candidate (non-overlapping, so additive)."""
    rows = []
    for rid, gt in store.ground_truth(stem+'.jpeg'):
        iou = np.asarray(coco.iou(rles, gt, [0] * len(gt))) if rles and gt else np.zeros((len(rles), len(gt)))
        hit = iou > .5
        rows.append({'stem': stem, 'record': rid, 'gt': len(gt), 'idx': np.asarray(inst_idx, dtype=int),
                     'hits': hit.sum(axis=1).astype(int), 'siou': (iou * hit).sum(axis=1)})
    return rows


def totals(keep: np.ndarray, recs: list[dict]) -> np.ndarray:
    """(tp, fp, fn, sum_iou) of the kept candidates over these records."""
    t = np.zeros(4)
    for r in recs:
        k = keep[r['idx']]
        h = r['hits'][k]
        t += (h.sum(), (h == 0).sum(), r['gt'] - h.sum(), r['siou'][k].sum())
    return t


def pq(t: np.ndarray) -> float:
    denom = t[..., 0] + .5 * t[..., 1] + .5 * t[..., 2]
    return t[..., 3] / denom


def area_bin(area: np.ndarray) -> np.ndarray:
    return np.digitize(area, AREA_BINS)


class Rescorer:
    """Logistic match probability x per-bin hit IoU, kept above PQ_train / 2."""

    def __init__(self, X: np.ndarray, area: np.ndarray, scores: np.ndarray, recs: list[dict]):
        rows = [(i, h, s) for r in recs for i, h, s in zip(r['idx'], r['hits'], r['siou'])]
        idx = np.asarray([i for i, _, _ in rows], dtype=int)
        hits = np.asarray([h for _, h, _ in rows])
        hit_iou = np.asarray([s / h if h else 0. for _, h, s in rows])
        self.scaler = StandardScaler().fit(X[idx])
        self.model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000).fit(self.scaler.transform(X[idx]), hits > 0)
        bins = area_bin(area[idx])
        overall = hit_iou[hits > 0].mean()
        self.e_iou = np.asarray([hit_iou[(hits > 0) & (bins == b)].mean() if ((hits > 0) & (bins == b)).any()
                                 else overall for b in range(len(AREA_BINS) + 1)])
        self.pq_train = pq(totals(scores >= OPTION_B_MIN, recs))
        self.pairs, self.hit_rate = len(rows), float((hits > 0).mean())

    def expected(self, X: np.ndarray, area: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1] * self.e_iou[area_bin(area)]

    def keep(self, X: np.ndarray, area: np.ndarray) -> np.ndarray:
        return self.expected(X, area) > self.pq_train / 2

    def summary(self) -> dict:
        return {'pq_train': self.pq_train, 'threshold': self.pq_train / 2, 'e_iou_by_bin': self.e_iou.tolist(),
                'pairs': self.pairs, 'pair_hit_rate': self.hit_rate,
                'coef_standardized': dict(zip(FEATURES, self.model.coef_[0].tolist())),
                'intercept': float(self.model.intercept_[0])}


def reference_thresholds(scores: np.ndarray, area: np.ndarray, recs: list[dict]) -> tuple[float, ...]:
    """Reference only: one score threshold per area bin, best training-half PQ (first max wins)."""
    bins = area_bin(area)
    best, best_pq = None, -1.
    for combo in itertools.product(REFERENCE_GRID, repeat=len(AREA_BINS) + 1):
        value = pq(totals(scores >= np.asarray(combo)[bins], recs))
        if value > best_pq:
            best, best_pq = combo, value
    return best


def bootstrap(dates: list[str], recs: list[dict], keep_new: np.ndarray, keep_base: np.ndarray) -> dict:
    by_date = {d: [r for r in recs if observation_date(r['stem']) == d] for d in dates}
    new = np.stack([totals(keep_new, by_date[d]) for d in dates])
    base = np.stack([totals(keep_base, by_date[d]) for d in dates])
    draws = np.random.default_rng(SEED).integers(0, len(dates), size=(2000, len(dates)))
    delta = pq(new[draws].sum(axis=1)) - pq(base[draws].sum(axis=1))
    return {'draws': 2000, 'delta_mean': float(delta.mean()), 'ci95': np.percentile(delta, [2.5, 97.5]).tolist(),
            'p_delta_gt_0': float((delta > 0).mean())}


def calibration(scores: np.ndarray, area: np.ndarray, recs: list[dict]) -> list[dict]:
    """Information only: pair hit rate by score band x area bin over all 141 images."""
    edges = (.15, .20, .25, .30, .40, .50, .70, 1.01)
    pairs = [(i, h) for r in recs for i, h in zip(r['idx'], r['hits'])]
    idx = np.asarray([i for i, _ in pairs], dtype=int)
    hit = np.asarray([h > 0 for _, h in pairs])
    bins = area_bin(area[idx])
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        for b in range(len(AREA_BINS) + 1):
            sel = (scores[idx] >= lo) & (scores[idx] < hi) & (bins == b)
            rows.append({'score': [lo, hi], 'area_bin': b, 'pairs': int(sel.sum()),
                         'hit_rate': float(hit[sel].mean()) if sel.any() else None})
    return rows


def exact(store, inst: list[dict], keep: np.ndarray, stems: list[str]) -> dict:
    preds = {s: [] for s in stems}
    for d, k in zip(inst, keep):
        if k:
            preds[d['stem']].append(d['rle'])
    return score(store, preds, stems)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    views = load('val')
    stems = sorted(views['m1536'])
    assert len(stems) == 141 and all(sorted(v) == stems and all(len(v[s]) == 4 for s in stems) for v in views.values())
    store, disks = AnnotationStore(DEFAULT_DATA), load_disk_geometry(CACHE/'disk.json')
    inst, recs = [], []
    for i, stem in enumerate(stems):
        logits = np.load(LOGITS/(stem+'.npy')).astype(np.float32)
        found = candidates(stem, logits, disks[stem], [views[m][stem] for m in MEMBERS])
        recs += records_for(store, stem, list(range(len(inst), len(inst) + len(found))), [d['rle'] for d in found])
        inst += found
        print(f'{i+1}/141 {stem} candidates={len(found)}', flush=True)
    X = np.asarray([[d[f] for f in FEATURES] for d in inst], dtype=np.float64)
    area = np.asarray([d['area'] for d in inst])
    scores = np.asarray([d['score'] for d in inst])
    base = scores >= OPTION_B_MIN
    assert abs(pq(totals(base, recs)) - OPTION_B_PQ) < 1e-9, pq(totals(base, recs))
    assert abs(exact(store, inst, base, stems)['pq'] - OPTION_B_PQ) < 1e-9

    dates = sorted({observation_date(s) for s in stems})
    order = np.random.default_rng(SEED).permutation(len(dates))
    halves = {'A': {dates[i] for i in order[:len(dates) // 2]}, 'B': {dates[i] for i in order[len(dates) // 2:]}}
    half_of = {s: next(h for h, ds in halves.items() if observation_date(s) in ds) for s in stems}
    inst_half = np.asarray([half_of[d['stem']] for d in inst])
    crossfit, reference, fits = np.zeros(len(inst), bool), np.zeros(len(inst), bool), {}
    for train, apply in (('A', 'B'), ('B', 'A')):
        train_recs = [r for r in recs if half_of[r['stem']] == train]
        rescorer = Rescorer(X, area, scores, train_recs)
        sel = inst_half == apply
        crossfit[sel] = rescorer.keep(X[sel], area[sel])
        combo = reference_thresholds(scores, area, train_recs)
        reference[sel] = scores[sel] >= np.asarray(combo)[area_bin(area[sel])]
        fits[f'fit_{train}_apply_{apply}'] = {**rescorer.summary(), 'reference_thresholds': combo,
                                              'train_images': sum(h == train for h in half_of.values())}
    metrics = {'option_b_s0.30': exact(store, inst, base, stems),
               'candidates_s0.15': exact(store, inst, np.ones(len(inst), bool), stems),
               'rescored_crossfit': exact(store, inst, crossfit, stems),
               'reference_area_thresholds_crossfit': exact(store, inst, reference, stems)}
    assert abs(metrics['rescored_crossfit']['pq'] - pq(totals(crossfit, recs))) < 1e-9
    full = Rescorer(X, area, scores, recs)
    changes = {'dropped_from_option_b': int((base & ~crossfit).sum()), 'added_below_0.30': int((~base & crossfit).sum()),
               'kept': int(crossfit.sum()), 'candidates': len(inst), 'option_b_instances': int(base.sum())}
    decision = {'crossfit_pq': metrics['rescored_crossfit']['pq'], 'option_b_pq': OPTION_B_PQ,
                'delta': metrics['rescored_crossfit']['pq'] - OPTION_B_PQ,
                'export_test': metrics['rescored_crossfit']['pq'] >= OPTION_B_PQ + MARGIN}
    OUT.mkdir(parents=True)
    save_json(OUT/'report.json', {
        'val141': metrics, 'decision': decision, 'changes': changes,
        'bootstrap_crossfit_vs_option_b': bootstrap(dates, recs, crossfit, base),
        'halves': {h: sorted(ds) for h, ds in halves.items()}, 'fits': fits,
        'full_fit_for_test': full.summary(), 'calibration_info_only': calibration(scores, area, recs),
        'fusion': vars(params(CANDIDATE_MIN)), 'features': FEATURES})
    save_json(OUT/'instances.json', [{k: v for k, v in d.items() if k != 'rle'} | {'half': h, 'kept': bool(k)}
                                     for d, h, k in zip(inst, inst_half.tolist(), crossfit)])
    for name, m in metrics.items():
        print(f"141 {name:36s} PQ {m['pq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']} "
              f"small {m['small_hit']}/{m['small_gt']} split {m['split_gt']} merge {m['merge_pred']}")
    print('CHANGES', json.dumps(changes))
    print('DECISION', json.dumps(decision))


if __name__ == '__main__':
    main()
