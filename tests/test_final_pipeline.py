"""Final inference pipeline (notebooks/final_inference.ipynb): weights manifest, frozen rescorer, flips, preprocessing parity."""
import json
from pathlib import Path

import pandas  # noqa: F401  (.venv-unet on Windows: import pandas before cv2 + smp)
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip('cv2')
from solar import final_pipeline as fp  # noqa: E402


def test_pools_match_the_export_scripts_and_manifest():
    assert fp.POOLS['r43'] == ('cp_s43', 'm1280', 'm1024')
    assert fp.POOLS['A6'] == ('cp_s43', 'm1280', 'm1024', 'det4', 'd1024', 'd1536')
    assert fp.FINALS == {'final1_A6': 'A6', 'final2_r0': 'r43'}
    m = fp.manifest()
    for final, pool in fp.FINALS.items():
        assert tuple(m['finals'][final]['detectors']) == fp.POOLS[pool]
        assert m['finals'][final]['rescorer'] == f'rescorer_{pool}.json'
    entries = fp.weight_entries()
    assert set(entries) == {'unet', 'refiner', *fp.POOLS['A6']}
    assert {entries[d]['input'] for d in ('cp_s43', 'm1280', 'm1024')} == {'jpeg'}
    assert {entries[d]['input'] for d in ('det4', 'd1024', 'd1536')} == {'flat_png'}
    assert {entries[d]['imgsz'] for d in ('cp_s43', 'd1536')} == {1536}
    assert entries['m1280']['imgsz'] == entries['det4']['imgsz'] == 1280 and entries['m1024']['imgsz'] == entries['d1024']['imgsz'] == 1024


def test_manifest_sha_equals_pinned_list():
    pinned = {}
    for line in (ROOT / 'docs/experiments/final_weights_sha256_20261106.txt').read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        pinned[path.strip()] = digest
    for e in fp.manifest()['weights']:
        assert pinned[e['source']] == e['sha256'], e['file']
    assert len(pinned) == 8


@pytest.mark.skipif(not (fp.WEIGHTS / 'unet_effb4_cpstrong_s0.pt').exists(), reason='weights not checked out')
def test_verify_script_passes_on_the_checked_in_weights():
    import importlib.util
    spec = importlib.util.spec_from_file_location('verify_final_weights', ROOT / 'scripts/verify_final_weights.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.verify(quiet=True)
    weights = [r for r in rows if r['role'] != 'rescorer']
    assert len(weights) == 8 and all(r['ok'] for r in weights), [r for r in weights if not r['ok']]


def test_frozen_rescorer_equals_sklearn():
    sklearn = pytest.importorskip('sklearn')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(20261107)
    X = rng.normal(size=(400, 6)) * np.array([.2, 1, 1.5, .3, .2, .3]) + np.array([.3, 2, 7, .7, .8, .5])
    area = rng.integers(100, 6000, size=400)
    y = (X @ np.array([2, .1, .05, .5, 1, -.3]) + rng.normal(size=400)) > 2.5
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1., solver='lbfgs', max_iter=1000).fit(scaler.transform(X), y)
    e_iou = np.array([.67, .68, .70])
    params = {'features': list('abcdef'), 'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
              'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]), 'e_iou_by_bin': e_iou.tolist(),
              'area_bins': [800, 3200], 'threshold': .4573 / 2}
    frozen = fp.FrozenRescorer(params)
    expected = model.predict_proba(scaler.transform(X))[:, 1] * e_iou[np.digitize(area, [800, 3200])]
    assert np.abs(frozen.expected(X, area) - expected).max() < 1e-12
    assert np.array_equal(frozen.keep(X, area), expected > .4573 / 2)
    assert 0 < frozen.keep(X, area).sum() < 400


@pytest.mark.skipif(not (fp.WEIGHTS / 'rescorer_r43.json').exists(), reason='rescorer params not checked out')
def test_r43_rescorer_params_match_the_export_report():
    params = json.loads((fp.WEIGHTS / 'rescorer_r43.json').read_text(encoding='utf-8'))
    report = json.loads((ROOT / 'outputs/combo_cpstrong_s0_cp43_submission_20261026/report.json').read_text(encoding='utf-8'))['checks'] \
        if (ROOT / 'outputs/combo_cpstrong_s0_cp43_submission_20261026/report.json').exists() else None
    assert params['pool'] == 'r43' and params['detectors'] == list(fp.POOLS['r43'])
    assert abs(params['threshold'] - params['pq_train'] / 2) < 1e-15
    assert len(params['scaler_mean']) == len(params['scaler_scale']) == len(params['coef']) == 6
    if report:
        saved = report['rescorer']
        assert abs(params['pq_train'] - saved['pq_train']) < 1e-9
        assert np.allclose(params['coef'], [saved['coef_standardized'][f] for f in params['features']], atol=1e-9, rtol=0)
        assert abs(params['intercept'] - saved['intercept']) < 1e-9
        assert np.allclose(params['e_iou_by_bin'], saved['e_iou_by_bin'], atol=1e-9, rtol=0)


def test_flip_views_undo_matches_the_runners(monkeypatch):
    """detect() flips masks back exactly as run_yolo11x / run_detector_amey did: ud first, then lr."""
    class Boxes:
        def __init__(self, n):
            import torch
            self.xyxy, self.conf = torch.zeros(n, 4), torch.full((n,), .5)

        def __len__(self):
            return len(self.conf)

    class Masks:
        def __init__(self, data):
            import torch
            self.data = torch.from_numpy(data)

    class Result:
        def __init__(self, masks):
            self.boxes, self.masks = Boxes(len(masks)), Masks(masks)

    base = np.zeros((2048, 2048), np.float32)
    base[10:20, 100:130] = 1.0   # a small rectangle near the top-left
    calls = []

    class Model:
        def predict(self, source, **kw):
            calls.append(kw)
            # The model "sees" the flipped view and returns the rectangle where it lies in that view.
            view = source if isinstance(source, np.ndarray) else None
            return [Result(np.stack([base if view is None else _where(view)]))]

    def _where(view):
        # The view is a flipped 3-channel image whose channel 0 encodes the rectangle at 255.
        return (view[..., 0] == 255).astype(np.float32)

    image = np.zeros((2048, 2048, 3), np.uint8)
    image[10:20, 100:130] = 255
    monkeypatch.setattr(fp.cv2, 'imread', lambda path, *a: image.copy())
    views = fp.flip_views(Model(), 'stem', 1536, 'flat_png', ROOT, ROOT, 'cpu')
    assert len(views) == 4 and all(len(v) == 1 for v in views)
    from pycocotools import mask as coco
    for v in views:
        decoded = coco.decode({'size': [2048, 2048], 'counts': v[0]['counts'].encode('ascii')})
        assert np.array_equal(decoded.astype(bool), base.astype(bool))   # every view maps back to the same place
    assert all(c['imgsz'] == 1536 and c['conf'] == .10 and c['iou'] == .60 and c['agnostic_nms'] and c['retina_masks'] for c in calls)


@pytest.mark.skipif(not (ROOT / 'outputs/amey_unet_cache/disk.json').exists() or not (fp.DEFAULT_DATA / 'test/test_images').exists(),
                    reason='local cache or data absent')
def test_preprocess_reproduces_the_cached_flat_png_and_disk(tmp_path):
    """Three test images: the notebook's preprocessing must equal the cache the finals were built from, byte for byte."""
    cache = ROOT / 'outputs/amey_unet_cache'
    disks = json.loads((cache / 'disk.json').read_text())
    stems = sorted(p.stem for p in (fp.DEFAULT_DATA / 'test/test_images').glob('*.jpeg'))[:3]
    info = fp.preprocess(stems, fp.DEFAULT_DATA / 'test/test_images', tmp_path, workers=1)
    assert info['computed'] == 3
    new = json.loads((tmp_path / 'disk.json').read_text())
    for s in stems:
        assert (tmp_path / 'flat' / f'{s}.png').read_bytes() == (cache / 'flat' / f'{s}.png').read_bytes(), s
        assert all(abs(new[s][k] - disks[s][k]) < 1e-9 for k in ('cx', 'cy', 'radius')), s


@pytest.mark.skipif(not (ROOT / 'outputs/refiner_submission_20261031/submission.csv').exists(), reason='final CSVs absent')
def test_compare_csv_on_the_two_finals():
    a = ROOT / 'outputs/amey_A6_cpstrong_s0_submission_20261104/refined/submission.csv'
    b = ROOT / 'outputs/refiner_submission_20261031/submission.csv'
    same = fp.compare_csv(b, b)
    assert same['byte_equal'] and same['exact_rle_matches'] == same['reference_instances'] == 1110 and same['match_share'] == 1.0
    diff = fp.compare_csv(a, b)
    assert not diff['byte_equal'] and diff['instances'] == 1109 and diff['reference_instances'] == 1110
    assert .95 < diff['match_share'] < 1.0
