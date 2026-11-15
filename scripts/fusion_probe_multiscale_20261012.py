"""Probe 8: multi-scale detector TTA under Option B, alone and with instance rescoring.

Pre-registered 2026-10-12, before any large-scale detection existed; design
from docs/competition/research/pq_improvement_options_20261012.md section 4.2,
plus probe 7's rescoring stacked on top. Run with
`.venv-unet` after copying outputs/multiscale_views_20261012 from the remote.
- Views: each member's four saved flip views at its own size plus four at
  round32(1.25 x size) (export_multiscale_views_20261012). Per member, all 8
  views are merged with merge_views (IoU .5, mean score over 8, a missing view
  counts 0); across members unchanged (merge_views IoU .5, mean over 3).
- Fusion: Option B setting unchanged (threshold .5, close 3, grow 4,
  min_area 200, keep_unclaimed 0, YOLO native masks).
- Two cells:
  ms_s0.30           min_score .30, no other change.
  ms_rescored        probe 7's rescoring, unchanged and refit on these
                     candidates (min_score .15, six features, logistic, keep iff
                     p * E[IoU] > PQ_train / 2, PQ_train = this cell's s.30 PQ
                     on the training half), cross-fitted on the same two date
                     halves (numpy default_rng(20261012)).
- Gate: Option B candidates (own-size views only) at s.30 must reproduce
  .4305464235.
- Decision: the better cell becomes a test candidate only if its PQ >= Option B
  + .005 (= .4355); best of two cells, so the max is slightly inflated.
  Reported, not gated: date-level paired bootstrap of the better cell vs
  Option B, small recall, split / merge. Reference only: large-scale views
  alone (four per member) at s.30.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fusion_probe_unet_20261010 import CACHE, LOGITS, load_disk_geometry  # noqa: E402
from fusion_probe_ensemble_20261011 import MEMBERS, load  # noqa: E402
from rescoring_probe_20261012 import (FEATURES, OPTION_B_MIN, OPTION_B_PQ, SEED, Rescorer,  # noqa: E402
                                      candidates, exact, pq, records_for, totals)
from solar.data import AnnotationStore, DEFAULT_DATA, save_json  # noqa: E402
from solar.grouped_cv import observation_date  # noqa: E402

LARGE = ROOT/'outputs/multiscale_views_20261012'
OUT = ROOT/'outputs/fusion_probe_multiscale_20261012'
MARGIN = .005
POOLS = ('option_b', 'ms', 'large_only')


def load_large(split: str) -> dict[str, dict[str, list]]:
    return {m: json.loads((LARGE/f'{m}_flip_views_{split}.json').read_text()) for m in MEMBERS}


def member_views(pool: str, own: dict, big: dict, stem: str) -> list[list[list[dict]]]:
    if pool == 'option_b':
        return [own[m][stem] for m in MEMBERS]
    if pool == 'ms':
        return [own[m][stem] + big[m][stem] for m in MEMBERS]
    return [big[m][stem] for m in MEMBERS]


def arrays(inst: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.asarray([[d[f] for f in FEATURES] for d in inst], dtype=np.float64),
            np.asarray([d['area'] for d in inst]), np.asarray([d['score'] for d in inst]))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    own, big = load('val'), load_large('val')
    stems = sorted(own['m1536'])
    assert len(stems) == 141 and all(sorted(v) == stems and all(len(v[s]) == 4 for s in stems)
                                     for v in [*own.values(), *big.values()])
    store, disks = AnnotationStore(DEFAULT_DATA), load_disk_geometry(CACHE/'disk.json')
    inst = {p: [] for p in POOLS}
    recs = {p: [] for p in POOLS}
    for i, stem in enumerate(stems):
        logits = np.load(LOGITS/(stem+'.npy')).astype(np.float32)
        for p in POOLS:
            found = candidates(stem, logits, disks[stem], member_views(p, own, big, stem))
            recs[p] += records_for(store, stem, list(range(len(inst[p]), len(inst[p]) + len(found))),
                                   [d['rle'] for d in found])
            inst[p] += found
        print(f'{i+1}/141 {stem} ' + ' '.join(f'{p}={len(inst[p])}' for p in POOLS), flush=True)
    data = {p: arrays(inst[p]) for p in POOLS}
    base = data['option_b'][2] >= OPTION_B_MIN
    assert abs(pq(totals(base, recs['option_b'])) - OPTION_B_PQ) < 1e-9

    X, area, scores = data['ms']
    dates = sorted({observation_date(s) for s in stems})
    order = np.random.default_rng(SEED).permutation(len(dates))
    halves = {'A': {dates[i] for i in order[:len(dates) // 2]}, 'B': {dates[i] for i in order[len(dates) // 2:]}}
    half_of = {s: next(h for h, ds in halves.items() if observation_date(s) in ds) for s in stems}
    inst_half = np.asarray([half_of[d['stem']] for d in inst['ms']])
    rescored, fits = np.zeros(len(inst['ms']), bool), {}
    for train, apply in (('A', 'B'), ('B', 'A')):
        rescorer = Rescorer(X, area, scores, [r for r in recs['ms'] if half_of[r['stem']] == train])
        sel = inst_half == apply
        rescored[sel] = rescorer.keep(X[sel], area[sel])
        fits[f'fit_{train}_apply_{apply}'] = rescorer.summary()
    keeps = {'ms_s0.30': ('ms', scores >= OPTION_B_MIN), 'ms_rescored': ('ms', rescored)}
    metrics = {'option_b_s0.30': exact(store, inst['option_b'], base, stems),
               **{c: exact(store, inst[p], k, stems) for c, (p, k) in keeps.items()},
               'large_only_s0.30_reference': exact(store, inst['large_only'], data['large_only'][2] >= OPTION_B_MIN, stems)}
    best = max(keeps, key=lambda c: metrics[c]['pq'])
    # Paired bootstrap needs both arms on one candidate pool: Option B's s.30 set and the
    # winning cell's set are scored record by record through their own pools.
    pool, keep = keeps[best]
    by_date = lambda rs, k: np.stack([totals(k, [r for r in rs if observation_date(r['stem']) == d]) for d in dates])
    new, old = by_date(recs[pool], keep), by_date(recs['option_b'], base)
    draws = np.random.default_rng(SEED).integers(0, len(dates), size=(2000, len(dates)))
    delta = pq(new[draws].sum(axis=1)) - pq(old[draws].sum(axis=1))
    boot = {'cell': best, 'draws': 2000, 'delta_mean': float(delta.mean()),
            'ci95': np.percentile(delta, [2.5, 97.5]).tolist(), 'p_delta_gt_0': float((delta > 0).mean())}
    decision = {'best_cell': best, 'best_pq': metrics[best]['pq'], 'option_b_pq': OPTION_B_PQ,
                'delta': metrics[best]['pq'] - OPTION_B_PQ, 'export_test': metrics[best]['pq'] >= OPTION_B_PQ + MARGIN}
    OUT.mkdir(parents=True)
    save_json(OUT/'report.json', {'val141': metrics, 'decision': decision, 'bootstrap_best_vs_option_b': boot,
                                  'rescoring_fits': fits, 'candidates': {p: len(inst[p]) for p in POOLS},
                                  'halves': {h: sorted(ds) for h, ds in halves.items()}})
    for name, m in metrics.items():
        print(f"141 {name:28s} PQ {m['pq']:.4f} TP {m['tp']} FP {m['fp']} FN {m['fn']} "
              f"small {m['small_hit']}/{m['small_gt']} split {m['split_gt']} merge {m['merge_pred']}")
    print('BOOTSTRAP', json.dumps(boot))
    print('DECISION', json.dumps(decision))


if __name__ == '__main__':
    main()
