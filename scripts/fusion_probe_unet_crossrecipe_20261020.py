"""Cross-recipe U-Net logits ensemble: do {bw, effb4, effb5} averages beat the boundary U-Net alone?

Pre-registered 2026-10-20 in docs/experiments/unet_crossrecipe_ensemble_20261020.md
(protocol from docs/competition/research/beyond_public039_options_20261020.md §13.1),
before any cell was scored. Zero GPU: the six saved fold1 val logits caches are averaged
with equal weights by the upstream scripts/average_logits.py (same as U3, probe 2026-10-12),
then scored by probe 11's pipeline unchanged (three Option B detectors, four flips each,
fusion min_score .15 candidates, fused_s0.30, unet_alone, 6-feature rescorer cross-fitted
on two date halves from default_rng(20261012)). Threshold .5 is not retuned.
- Parity first: re-scoring bw_s0 / bw_s1 alone must reproduce the saved probe 16 results
  (rescored .45377... / .45270...) exactly before any ensemble number counts.
- Cells (fixed): E3_s0 = mean(bw_s0, effb4_s0, effb5_s0), E3_s1 likewise for seed 1,
  E6 = all six, BW2 = mean(bw_s0, bw_s1), E2_s0 / E2_s1 = mean(bw_s, effb5_s).
- Decision (one rule): delta_s = PQ(E3_s) - PQ(bw_s) on rescored_crossfit_info. Success only
  if mean(delta) >= .005 and both deltas > 0. E6 / BW2 / E2 are reported against bw_s0, bw_s1
  and their mean, never gated.
- Also reported: per-image totals, so the 2018-2022 and single-annotator subsets and a
  date-paired bootstrap of each delta come from the same matches.
Results never overwrite an existing file.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fusion_probe_unet_20261010 import CACHE, U_PARAMS, load_disk_geometry, logits_to_instances, rles_from_labels  # noqa: E402
from fusion_probe_ensemble_20261011 import MEMBERS, load  # noqa: E402
from fusion_probe_multiscale_20261012 import arrays  # noqa: E402
from rescoring_probe_20261012 import OPTION_B_MIN, SEED, Rescorer, candidates, records_for  # noqa: E402
from solar.data import AnnotationStore, DEFAULT_DATA, save_json  # noqa: E402
from solar.evaluation import aggregate, score_instances  # noqa: E402
from solar.grouped_cv import observation_date  # noqa: E402

SOURCES = {'bw_s0': ROOT/'outputs/unet_boundary_20261019/bw_s0',
           'bw_s1': ROOT/'outputs/unet_boundary_20261019/bw_s1',
           'effb4_s0': ROOT/'outputs/unet_encoder_20261015/effb4_s0',
           'effb4_s1': ROOT/'outputs/unet_encoder_20261015/effb4_s1',
           'effb5_s0': ROOT/'outputs/unet_encoder_20261017/effb5_s0',
           'effb5_s1': ROOT/'outputs/unet_encoder_20261017/effb5_s1'}
#: SHA-256 of each logits_val directory's files (sorted names, concatenated), checked 2026-10-20.
#: bw_s0 / bw_s1 equal the remote copies recorded in unet_boundary_20261019.md.
SOURCE_SHA = {'bw_s0': '7a646d27ce7bbf40e1accdeb33e1390b0474157802664ec3a78d1fe4e8c08d74',
              'bw_s1': 'f327c80abbf1a5de5e9f92b17ba2b69ddeb2d530e0bec47cac6d54d74cbd1bb0',
              'effb4_s0': '659b63c9651f53d113e04a0f4288fb0ba843161d5b200effe8e1d599564f84ac',
              'effb4_s1': '2213d48d3d722c492b1eb27dfa4e9713e67c3ed2b71dc1848b100c757c262db4',
              'effb5_s0': '5a0edb6ec4384ad07e0bdba53e5f8128a52e2c9b56ee72e60108c179e0bd60b4',
              'effb5_s1': '92d72dad5220867d648bfc96ab0de3de155beadeb65ac8fea1f8d2ab3c64e7e0'}
ENCODER = {'bw': 'efficientnet-b4', 'effb4': 'efficientnet-b4', 'effb5': 'efficientnet-b5'}
CELLS = {'E3_s0': ('bw_s0', 'effb4_s0', 'effb5_s0'),
         'E3_s1': ('bw_s1', 'effb4_s1', 'effb5_s1'),
         'E6': ('bw_s0', 'bw_s1', 'effb4_s0', 'effb4_s1', 'effb5_s0', 'effb5_s1'),
         'BW2': ('bw_s0', 'bw_s1'),
         'E2_s0': ('bw_s0', 'effb5_s0'),
         'E2_s1': ('bw_s1', 'effb5_s1')}
PARITY = ('bw_s0', 'bw_s1')
WORK = ROOT/'outputs/unet_crossrecipe_20261020'
OUT = ROOT/'outputs/fusion_probe_unet_crossrecipe_20261020'
SAVED = ROOT/'outputs/fusion_probe_unet_boundary_20261019'
AVERAGE = ROOT/'third_party/ameyypawar-filament-seg/scripts/average_logits.py'
MAIN = 'rescored_crossfit_info'
CELL_NAMES = ('fused_s0.30', 'unet_alone', MAIN)
MARGIN = .005
BOOT = 2000
KEYS = ('tp', 'fp', 'fn', 'sum_iou', 'gt', 'pred', 'small_gt', 'small_hit', 'split_gt', 'merge_pred')


def logits_dir(name: str) -> Path:
    return SOURCES[name]/'logits_val' if name in SOURCES else WORK/name/'logits_val'


def dir_sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for f in sorted(path.iterdir(), key=lambda p: p.name):
        digest.update(f.read_bytes())
    return digest.hexdigest()


def check_member(name: str) -> dict:
    """Each member's logits must come from its own last checkpoint, encoder and seed, with the pinned SHA."""
    runtime = json.loads((SOURCES[name]/'runtime.json').read_text())
    meta = json.loads((SOURCES[name]/'logits_val'/'cache_meta.json').read_text())
    arm, seed = name.split('_s')
    assert runtime['encoder'] == ENCODER[arm] and runtime['seed'] == int(seed) and not runtime['smoke'], runtime
    if arm == 'bw':
        assert runtime['loss']['name'] == 'boundary_weighted_bce' and runtime['loss']['forward_calls'] > 0, runtime['loss']
    assert (meta['tta'], meta['tile'], meta['overlap']) == ('dihedral', 512, 128) and len(meta['checkpoints']) == 1, meta
    assert meta['checkpoints'][0]['path'].endswith(f'/{name}/model_last.pt'), meta
    sha = dir_sha(SOURCES[name]/'logits_val')
    assert sha == SOURCE_SHA[name], (name, sha)
    return {'encoder': runtime['encoder'], 'seed': runtime['seed'], 'logits_sha256': sha,
            'model_last_sha256': runtime.get('model_last_sha256'), 'checkpoint': meta['checkpoints'][0]['path']}


def check_provenance(name: str) -> dict:
    members = CELLS.get(name, (name,))
    info = {m: check_member(m) for m in members}
    meta = json.loads((logits_dir(name)/'cache_meta.json').read_text())
    assert [c['path'] for c in meta['checkpoints']] == [info[m]['checkpoint'] for m in members], meta
    assert (meta['tta'], meta['tile'], meta['overlap']) == ('dihedral', 512, 128), meta
    out = {'members': list(members), 'member_provenance': info}
    if name in CELLS:
        out['averaged_logits_sha256'] = dir_sha(logits_dir(name))
    return out


def average(cell: str) -> None:
    for m in CELLS[cell]:
        check_member(m)
    out = logits_dir(cell)
    subprocess.run([sys.executable, str(AVERAGE), '--out', str(out), *(str(logits_dir(m)) for m in CELLS[cell])],
                   check=True)
    assert len(list(out.glob('*.npy'))) == 141, out


def per_stem(store, preds: dict, stems: list[str]) -> dict:
    """Totals per image, summed over that image's annotation records (aggregate() sums them the same way)."""
    out = {}
    for s in stems:
        rows = [score_instances(gt, preds.get(s, [])) for _, gt in store.ground_truth(s+'.jpeg')]
        out[s] = {k: sum(r[k] for r in rows) for k in KEYS}
    return out


def total(stem_rows: dict, stems) -> dict:
    return aggregate([stem_rows[s] for s in stems])


def score_run(name: str) -> dict:
    """probe11.score_run with the logits directory swapped and per-image totals kept."""
    provenance = check_provenance(name)
    views = load('val')
    stems = sorted(views['m1536'])
    src = logits_dir(name)
    assert len(stems) == 141 and all((src/(s+'.npy')).exists() for s in stems)
    store, disks = AnnotationStore(DEFAULT_DATA), load_disk_geometry(CACHE/'disk.json')
    dates = sorted({observation_date(s) for s in stems})
    order = np.random.default_rng(SEED).permutation(len(dates))
    half_a = {dates[i] for i in order[:len(dates) // 2]}
    inst, recs, alone = [], [], {}
    for i, stem in enumerate(stems):
        logits = np.load(src/(stem+'.npy')).astype(np.float32)
        disk = disks[stem]
        alone[stem] = rles_from_labels(logits_to_instances(logits, disk.mask(logits.shape), U_PARAMS))
        found = candidates(stem, logits, disk, [views[m][stem] for m in MEMBERS])
        recs += records_for(store, stem, list(range(len(inst), len(inst) + len(found))), [d['rle'] for d in found])
        inst += found
        print(f'{name} {i+1}/141', flush=True)
    X, area, scores = arrays(inst)
    half = np.asarray(['A' if observation_date(d['stem']) in half_a else 'B' for d in inst])
    rescored = np.zeros(len(inst), bool)
    for train, apply in (('A', 'B'), ('B', 'A')):
        rescorer = Rescorer(X, area, scores, [r for r in recs if ('A' if observation_date(r['stem']) in half_a else 'B') == train])
        rescored[half == apply] = rescorer.keep(X[half == apply], area[half == apply])

    def kept(keep):
        preds = {s: [] for s in stems}
        for d, k in zip(inst, keep):
            if k:
                preds[d['stem']].append(d['rle'])
        return preds

    rows = {'fused_s0.30': per_stem(store, kept(scores >= OPTION_B_MIN), stems),
            'unet_alone': per_stem(store, alone, stems),
            MAIN: per_stem(store, kept(rescored), stems)}
    annotators = {s: len(store.ground_truth(s+'.jpeg')) for s in stems}
    result = {'provenance': provenance, **{c: total(rows[c], stems) for c in CELL_NAMES}}
    result['per_image'] = {'annotators': annotators, **rows}
    return result


def subsets(per_image: dict) -> dict:
    stems = sorted(per_image['annotators'])
    groups = {'all': stems,
              'years_2018_2022': [s for s in stems if '2018' <= s[:4] <= '2022'],
              'single_annotator': [s for s in stems if per_image['annotators'][s] == 1]}
    return {g: {'images': len(ss), **{c: total(per_image[c], ss) for c in CELL_NAMES}} for g, ss in groups.items()}


def bootstrap(a: dict, b: dict) -> dict:
    """Date-paired bootstrap of PQ(a) - PQ(b): resample observation dates, sum both arms' per-image totals."""
    stems = sorted(a)
    by_date = {}
    for s in stems:
        by_date.setdefault(observation_date(s), []).append(s)
    dates = sorted(by_date)
    rng = np.random.default_rng(20261020)
    deltas = []
    for _ in range(BOOT):
        pick = [s for i in rng.integers(0, len(dates), len(dates)) for s in by_date[dates[i]]]
        deltas.append(total(a, pick)['pq'] - total(b, pick)['pq'])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {'ci95': [float(lo), float(hi)], 'p_le_0': float(np.mean(np.asarray(deltas) <= 0)), 'dates': len(dates)}


def decide() -> None:
    if (OUT/'report.json').exists():
        raise FileExistsError(OUT/'report.json')
    res = {n: json.loads((OUT/f'{n}.json').read_text()) for n in (*PARITY, *CELLS)}
    for run in PARITY:  # parity: same pipeline, same logits -> identical totals to probe 16
        saved = json.loads((SAVED/f'{run}.json').read_text())
        for c in CELL_NAMES:
            for k in (*KEYS, 'pq'):
                assert abs(res[run][c][k] - saved[c][k]) < 1e-9, (run, c, k, res[run][c][k], saved[c][k])
    ctrl = {s: res[f'bw_s{s}'] for s in (0, 1)}
    pq = {n: {c: res[n][c]['pq'] for c in CELL_NAMES} for n in res}
    deltas = {s: pq[f'E3_s{s}'][MAIN] - pq[f'bw_s{s}'][MAIN] for s in (0, 1)}
    mean_delta = float(np.mean(list(deltas.values())))
    info = {}
    for n in CELLS:
        info[n] = {c: {'vs_bw_s0': pq[n][c] - pq['bw_s0'][c], 'vs_bw_s1': pq[n][c] - pq['bw_s1'][c],
                       'vs_mean_bw': pq[n][c] - (pq['bw_s0'][c] + pq['bw_s1'][c]) / 2} for c in CELL_NAMES}
        info[n]['bootstrap_main'] = {f'vs_bw_s{s}': bootstrap(res[n]['per_image'][MAIN], ctrl[s]['per_image'][MAIN])
                                     for s in (0, 1)}
    decision = {'main_cell': MAIN, 'rule': 'E3_s - bw_s, mean >= .005 and both > 0',
                'deltas': deltas, 'mean_delta': mean_delta, 'margin': MARGIN,
                'success': bool(mean_delta >= MARGIN and all(d > 0 for d in deltas.values())),
                'parity': 'bw_s0 / bw_s1 re-scored totals equal probe 16 saved results (1e-9)'}
    summary = {n: {c: {k: res[n][c][k] for k in ('pq', 'tp', 'fp', 'fn', 'small_hit', 'small_gt', 'split_gt', 'merge_pred')}
                   for c in CELL_NAMES} for n in res}
    report = {'val141': summary, 'subsets': {n: subsets(res[n]['per_image']) for n in res},
              'provenance': {n: res[n]['provenance'] for n in res}, 'info_deltas': info, 'decision': decision}
    save_json(OUT/'report.json', report)
    for n, m in summary.items():
        for c, v in m.items():
            print(f"{n:9s} {c:24s} PQ {v['pq']:.4f} TP {v['tp']} FP {v['fp']} FN {v['fn']} "
                  f"small {v['small_hit']}/{v['small_gt']} split {v['split_gt']} merge {v['merge_pred']}")
    for n, sub in report['subsets'].items():
        print(n, {g: (v['images'], round(v[MAIN]['pq'], 4)) for g, v in sub.items()})
    print('INFO', json.dumps({n: {'main': d[MAIN], 'bootstrap_main': d['bootstrap_main']} for n, d in info.items()}))
    print('DECISION', json.dumps(decision))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--average', choices=CELLS)
    parser.add_argument('--run', choices=(*PARITY, *CELLS))
    parser.add_argument('--decide', action='store_true')
    args = parser.parse_args()
    if args.decide:
        decide()
        return
    if args.average:
        average(args.average)
        return
    if not args.run:
        parser.error('--average, --run or --decide is required')
    path = OUT/f'{args.run}.json'
    if path.exists():
        raise FileExistsError(path)
    result = score_run(args.run)
    save_json(path, result)
    print(args.run, {c: round(result[c]['pq'], 4) for c in CELL_NAMES})


if __name__ == '__main__':
    main()
