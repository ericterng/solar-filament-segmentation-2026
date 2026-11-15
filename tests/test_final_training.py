"""Training stage (solar.final_training): the recipes equal what the original runners executed. CPU only, no model.

Tests that need files outside git (competition data, recorded candidate dumps under outputs/) are skipped without them.
"""
import ast
import json
import pickle
import sys
from pathlib import Path

import pandas  # noqa: F401  (.venv-unet on Windows: import pandas before cv2 + smp)
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip('cv2')
from solar import final_pipeline as fp  # noqa: E402
from solar import final_training as ft  # noqa: E402

sys.path.insert(0, str(ROOT / 'scripts'))
ANNOTATIONS = ROOT / ft.DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
needs_data = pytest.mark.skipif(not ANNOTATIONS.exists(), reason='competition annotations not available')
DUMPS = {'r43': ROOT / 'outputs/refiner_20261031/instances_val_s0.json',
         'A6': ROOT / 'outputs/refiner_20261031/amey_20261104/A6/instances_val_s0.json'}
SAMPLES = ROOT / 'outputs/refiner_20261031/samples_s0.pkl'


def train_call(path: Path) -> dict[str, ast.expr]:
    """Keyword arguments of the single `model.train(...)` call in a runner script."""
    calls = [n for n in ast.walk(ast.parse(path.read_text(encoding='utf-8'))) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'train'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'model']
    assert len(calls) == 1, path
    return {k.arg: k.value for k in calls[0].keywords}


def function_dump(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name))


# ----------------------------------------------------------------------------- recipes and pins
def test_recipes_cover_every_shipped_file_and_step():
    entries = fp.weight_entries()
    assert set(ft.RECIPES) == {'unet', 'refiner', 'rescorer', *ft.DETECTORS} and set(ft.DETECTORS) == set(fp.POOLS['A6'])
    assert set(ft.STEPS) == {'inputs', *ft.RECIPES}
    for name in ('unet', 'refiner', *ft.DETECTORS):
        assert ft.RECIPES[name]['file'] == entries[name]['file']
        assert (ROOT / ft.RECIPES[name]['script'].split()[0]).exists()
    for member, cfg in {**ft.HESHEN, **ft.AMEY}.items():
        assert entries[member]['imgsz'] == cfg['imgsz']
        assert f"seed {cfg['seed']}" in entries[member]['training'] and f"seed {cfg['seed']}" in ft.RECIPES[member]['settings']
    assert {entries[m]['input'] for m in ft.HESHEN} == {'jpeg'} and {entries[m]['input'] for m in ft.AMEY} == {'flat_png'}
    assert len(ft.describe()) == len(ft.RECIPES)
    assert ft.REFINER_POOL in fp.POOLS


def test_pretrained_pins_equal_the_runners():
    import run_detector_amey_20261015 as amey
    import run_extra_detectors_20261011 as extra
    assert ft.YOLO_PRETRAINED_SHA == amey.PRETRAINED_SHA == extra.PRETRAINED_SHA
    encoder_script = (ROOT / 'scripts/train_unet_encoder_20261015.py').read_text(encoding='utf-8')
    assert all(repr(value) in encoder_script for value in ft.ENCODER_REPO)
    assert "CACHE = ROOT / 'outputs/amey_unet_cache'" in encoder_script and ft.UNET_CACHE == ROOT / 'outputs/amey_unet_cache'
    assert "SPLITS = ROOT / 'outputs/amey_unet_fold1_20261010/splits.json'" in encoder_script
    assert ft.UNET_SPLIT == ROOT / 'outputs/amey_unet_fold1_20261010/splits.json'
    import rescoring_probe_20261012 as probe
    assert ft.HALF_SEED == probe.SEED


# ----------------------------------------------------------------------------- split
def test_fold1_split_keeps_dates_together_and_is_deterministic():
    stems = [f'201{y}0{m}{d:02d}1200{k}0Bh' for y in range(1, 4) for m in range(1, 5) for d in range(1, 9) for k in range(2)]
    train, val = ft.fold1_split(stems)
    assert sorted(train + val) == sorted(stems) and not {s[:8] for s in train} & {s[:8] for s in val}
    assert (train, val) == ft.fold1_split(reversed(stems))
    assert abs(len(val) / len(stems) - .2) < .02


@needs_data
def test_fold1_split_is_the_shipped_validation_set(monkeypatch):
    monkeypatch.chdir(ROOT)
    train, val = ft.fold1_split(ft.annotation_stems())
    assert (len(train), len(val)) == (566, 141)
    shipped = json.loads((ROOT / 'docs/report/val_predictions/val_fold1_A6_crossfit.json').read_text(encoding='utf-8'))
    assert val == [im['stem'] for im in shipped['images']]


@needs_data
@pytest.mark.skipif(not ft.UNET_SPLIT.exists(), reason='the original splits.json is not on this machine')
def test_unet_split_equals_the_original_file():
    assert ft.unet_split() == json.loads(ft.UNET_SPLIT.read_text(encoding='utf-8'))


# ----------------------------------------------------------------------------- U-Net and refiner commands
def test_unet_command_matches_the_original_launcher(tmp_path):
    cmd, env = ft.unet_command(tmp_path / 'run/cpstrong_s0', 'GPU-00000000-0000-0000-0000-000000000000', python='python')
    assert cmd == ['python', 'scripts/train_unet_cpstrong_20261025.py', '--seed', '0', '--out', str(tmp_path / 'run/cpstrong_s0')]
    assert env['CUDA_VISIBLE_DEVICES'] == env['UNET_ENCODER_GPU_UUID'] and env['HF_HUB_OFFLINE'] == '1'
    launcher = (ROOT / 'scripts/launch_unet_cpstrong_20261025.sh').read_text(encoding='utf-8')
    assert 'scripts/train_unet_cpstrong_20261025.py --seed' in launcher
    for name in env:
        assert name in launcher, name
    wrapper = (ROOT / 'scripts/train_unet_cpstrong_20261025.py').read_text(encoding='utf-8')
    for text in ("'--epochs', '30'", "'--batch-size', '16'", "'--workers', '4'", 'CP_N_MIN', 'BoundaryDiceBCELoss'):
        assert text in wrapper, text


def test_refiner_command_matches_the_original_launcher(tmp_path):
    cmd = ft.refiner_command(tmp_path / 's.pkl', tmp_path / 'cpstrong_s0/model_last.pt', tmp_path / 'out', tmp_path / 'cache', python='python')
    assert cmd[:2] == ['python', 'scripts/train_refiner_20261031.py'] and '--cpu' not in cmd
    options = dict(zip(cmd[2::2], cmd[3::2]))
    assert options['--train-half'] == 'all' and options['--refiner-seed'] == '0'
    launcher = (ROOT / 'scripts/launch_refiner_20261031.sh').read_text(encoding='utf-8')
    assert 'scripts/train_refiner_20261031.py' in launcher and all(o in launcher for o in options)
    assert ft.refiner_command(tmp_path / 's.pkl', tmp_path / 'i', tmp_path / 'o', tmp_path / 'c', cpu=True)[-1] == '--cpu'
    import refiner_core_20261031 as core
    assert (core.TRAIN_EPOCHS, core.TRAIN_BATCH, core.TRAIN_LR, core.IOU_MIN) == (16, 8, 1e-4, .25)


# ----------------------------------------------------------------------------- detectors
@pytest.mark.parametrize('member', ['m1280', 'm1024'])
def test_heshen_source_equals_the_extra_detector_runner(member, tmp_path):
    import run_extra_detectors_20261011 as extra
    cfg = extra.MEMBERS[member]
    assert cfg == {k: ft.HESHEN[member][k] for k in ('imgsz', 'seed')}
    original = extra.member_source(tmp_path, f'det_{member}_f1', cfg['imgsz'], cfg['seed'], False)
    assert ft.heshen_source(member, tmp_path) == original


def test_heshen_source_equals_the_copy_paste_runner(tmp_path):
    import run_yolo_copypaste_20261025 as cp
    assert cp.MEMBERS['s43'] == ft.HESHEN['cp_s43']['seed'] and cp.COPY_PASTE == ft.HESHEN['cp_s43']['copy_paste']
    assert cp.TRAIN_CALL_END == ft.HESHEN_TRAIN_CALL_END
    assert ft.heshen_source('cp_s43', tmp_path) == cp.member_source(tmp_path, 43, False)


def test_heshen_source_changes_only_paths_and_device(tmp_path):
    default = ft.heshen_source('m1280', tmp_path).splitlines()
    moved = ft.heshen_source('m1280', tmp_path, data_root=tmp_path / 'elsewhere', pretrained=tmp_path / 'y.pt',
                             device='cuda:1').splitlines()
    assert len(default) == len(moved)
    changed = [(a, b) for a, b in zip(default, moved) if a != b]
    assert len(changed) == 3
    assert changed[0][1] == f"INPUT_ROOT = Path({str(tmp_path / 'elsewhere')!r})"
    assert changed[1][1] == f"det = YOLO({str(tmp_path / 'y.pt')!r})"
    assert "device=torch.device('cuda:1')" in changed[2][1] and 'imgsz=1280, batch=2' in changed[2][1]
    compile('\n'.join(moved), 'executed_source.py', 'exec')


@pytest.mark.parametrize('runner', ['run_detector_amey_20261015.py', 'run_detector4_20261013.py'])
def test_amey_training_arguments_equal_the_runners(runner):
    module = __import__(runner[:-3])
    literal, named, other = {}, {}, set()
    for key, value in train_call(ROOT / 'scripts' / runner).items():
        if isinstance(value, ast.Constant):
            literal[key] = value.value
        elif isinstance(value, ast.Name) and hasattr(module, value.id):
            named[key] = getattr(module, value.id)
        else:
            other.add(key)
    assert module.EPOCHS == ft.AMEY_TRAIN['epochs']
    fixed = {**literal, **{k: v for k, v in named.items() if k in ('batch', 'workers')}, 'epochs': module.EPOCHS}
    assert {k: v for k, v in fixed.items() if k != 'name'} == ft.AMEY_TRAIN and literal['name'] == 'train'
    assert other | (set(named) - {'batch', 'workers'}) == {'data', 'imgsz', 'epochs', 'device', 'project', 'seed'}


def test_amey_members_and_dataset_builder_are_shared_by_both_runners():
    import run_detector4_20261013 as det4
    import run_detector_amey_20261015 as amey
    assert amey.MEMBERS == {m: ft.AMEY[m] for m in ('d1024', 'd1536')}
    assert {'imgsz': det4.IMGSZ, 'seed': det4.SEED} == ft.AMEY['det4']
    for name in ('polygon_area', 'label_lines', 'build_dataset'):
        assert function_dump(ROOT / 'scripts/run_detector4_20261013.py', name) == \
            function_dump(ROOT / 'scripts/run_detector_amey_20261015.py', name), name


@needs_data
def test_amey_dataset_builder_writes_one_sample_per_annotator(tmp_path, monkeypatch):
    import run_detector_amey_20261015 as amey
    monkeypatch.chdir(ROOT)
    train, _ = ft.fold1_split(ft.annotation_stems())
    raw = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
    counts = {}
    for record in raw['images']:
        counts[Path(record['file_name']).stem] = counts.get(Path(record['file_name']).stem, 0) + 1
    stems = [next(s for s in train if counts[s] == 1), next(s for s in train if counts[s] > 1)]
    flat = tmp_path / 'flat'
    flat.mkdir()
    for stem in stems:
        (flat / f'{stem}.png').write_bytes(b'not a real image; the builder only links it')
    monkeypatch.setattr(amey, 'FLAT', flat)
    monkeypatch.setattr(amey, 'DATASET', ROOT / ft.DATA)
    report = amey.build_dataset(tmp_path / 'yolo', stems, 0, np)
    expected = counts[stems[0]] + counts[stems[1]]
    assert report['train']['samples'] == expected == len(list((tmp_path / 'yolo/images/train').glob('*.png')))
    labels = sorted((tmp_path / 'yolo/labels/train').glob('*.txt'))
    assert len(labels) == expected and report['train']['filaments'] == sum(len(p.read_text().splitlines()) for p in labels)
    first = labels[0].read_text().splitlines()[0].split()
    assert first[0] == '0' and len(first) % 2 == 1 and all(0 <= float(v) <= 1 for v in first[1:])
    assert 'names:\n  0: filament' in (tmp_path / 'yolo/data.yaml').read_text()


# ----------------------------------------------------------------------------- rescorer and refiner samples
def recorded_candidates(pool: str) -> tuple[list[dict], list[str]]:
    from export_rescorer_params_20261107 import from_instances
    inst, stems, _ = from_instances(DUMPS[pool], fp.POOLS[pool])
    return inst, stems


@needs_data
@pytest.mark.parametrize('pool', ['A6', 'r43'])
def test_fit_rescorer_reproduces_the_shipped_parameters(pool, monkeypatch):
    if not DUMPS[pool].exists():
        pytest.skip('recorded validation candidates are not on this machine')
    monkeypatch.chdir(ROOT)
    from solar.data import AnnotationStore
    params = ft.fit_rescorer(*recorded_candidates(pool), AnnotationStore(ft.DATA), pool)
    shipped = json.loads((fp.WEIGHTS / f'rescorer_{pool}.json').read_text(encoding='utf-8'))
    for key in ('scaler_mean', 'scaler_scale', 'coef', 'intercept', 'e_iou_by_bin', 'pq_train', 'threshold'):
        assert np.allclose(params[key], shipped[key], atol=1e-9, rtol=0), key
    for key in ('pool', 'detectors', 'features', 'area_bins', 'candidate_min_score'):
        assert params[key] == shipped[key], key
    for key in ('pairs', 'val_images'):
        assert params['fit'][key] == shipped['fit'][key], key
    for key in ('pair_hit_rate', 'val_fused_s030_pq', 'val_insample_rescored_pq_not_a_score'):
        assert params['fit'][key] == pytest.approx(shipped['fit'][key], abs=1e-9), key
    fp.FrozenRescorer(params)


@needs_data
def test_refiner_instances_and_samples_reproduce_the_recorded_ones(tmp_path, monkeypatch):
    if not DUMPS['r43'].exists() or not SAMPLES.exists():
        pytest.skip('recorded validation candidates / samples are not on this machine')
    monkeypatch.chdir(ROOT)
    from solar.data import AnnotationStore
    store = AnnotationStore(ft.DATA)
    recorded = json.loads(DUMPS['r43'].read_text(encoding='utf-8'))
    blob = ft.refiner_instances(*recorded_candidates('r43'), store)
    assert blob['meta']['halves'] == recorded['meta']['halves']
    assert (blob['meta']['candidates'], blob['meta']['kept']) == (recorded['meta']['candidates'], recorded['meta']['kept'])
    for mine, theirs in zip(blob['images'], recorded['images']):
        assert (mine['stem'], mine['half']) == (theirs['stem'], theirs['half'])
        assert mine['instances'] == theirs['instances'], mine['stem']
    stats = ft.refiner_samples(blob, store, tmp_path / 'samples.pkl')
    mine, theirs = pickle.loads((tmp_path / 'samples.pkl').read_bytes()), pickle.loads(SAMPLES.read_bytes())
    assert stats['samples'] == len(theirs['samples']) == 1398
    assert mine['samples'] == theirs['samples']
    assert mine['meta']['halves'] == theirs['meta']['halves'] and mine['meta']['unet_seed'] == theirs['meta']['unet_seed'] == 0
    assert mine['meta']['probe'] == theirs['meta']['probe'] and mine['meta']['iou_min'] == theirs['meta']['iou_min']


def test_use_weights_restores_the_folder(tmp_path):
    original = fp.WEIGHTS
    with ft.use_weights(tmp_path) as module:
        assert module.WEIGHTS == tmp_path
    assert fp.WEIGHTS == original
    with ft.use_weights(None) as module:
        assert module.WEIGHTS == original


def test_retrain_rejects_unknown_steps(tmp_path):
    with pytest.raises(AssertionError, match='unknown steps'):
        ft.retrain(steps=('unet', 'not_a_step'), out=tmp_path)


# ----------------------------------------------------------------------------- smoke mode and GPU selection
@pytest.mark.parametrize('member', ['m1280', 'm1024'])
def test_heshen_smoke_source_equals_the_extra_detector_runner_smoke(member, tmp_path):
    import run_extra_detectors_20261011 as extra
    cfg = extra.MEMBERS[member]
    assert ft.heshen_source(member, tmp_path, smoke=True) == extra.member_source(tmp_path, f'det_{member}_f1', cfg['imgsz'], cfg['seed'], True)
    assert ft.heshen_source(member, tmp_path, smoke=False) == ft.heshen_source(member, tmp_path)


def test_heshen_smoke_source_equals_the_copy_paste_runner_smoke(tmp_path):
    import run_yolo_copypaste_20261025 as cp
    assert ft.heshen_source('cp_s43', tmp_path, smoke=True) == cp.member_source(tmp_path, 43, True)


def test_amey_smoke_changes_only_epochs_and_records_as_the_runner():
    for member in ft.AMEY:
        default = ft.amey_train_arguments(member)
        assert default == {'imgsz': ft.AMEY[member]['imgsz'], 'seed': ft.AMEY[member]['seed'], **ft.AMEY_TRAIN}
        smoke = ft.amey_train_arguments(member, smoke=True)
        assert {k for k in default if default[k] != smoke[k]} == {'epochs'} and smoke['epochs'] == 1
    runner = (ROOT / 'scripts/run_detector_amey_20261015.py').read_text(encoding='utf-8')
    assert "build_dataset(work / 'yolo', manifest['train'], 8 if smoke else 0, np)" in runner and ft.AMEY_SMOKE['records'] == 8
    assert "'epochs': 1 if smoke else EPOCHS" in runner and ft.AMEY_SMOKE['epochs'] == 1
    assert "trainer.args.device = torch.device('cuda:0')" in runner   # the device callback train_amey_detector repeats


def test_unet_and_refiner_smoke_add_only_the_scripts_own_flag(tmp_path):
    uuid = 'GPU-00000000-0000-0000-0000-000000000000'
    cmd, env = ft.unet_command(tmp_path / 'u', uuid, python='python')
    smoke_cmd, smoke_env = ft.unet_command(tmp_path / 'u', uuid, python='python', smoke=True)
    assert smoke_cmd == cmd + ['--smoke'] and smoke_env == env
    args = (tmp_path / 's.pkl', tmp_path / 'cpstrong_s0/model_last.pt', tmp_path / 'out', tmp_path / 'cache')
    assert ft.refiner_command(*args, python='python', smoke=True) == ft.refiner_command(*args, python='python') + ['--smoke']
    assert ft.refiner_command(*args, python='python', cpu=True, smoke=True)[-2:] == ['--cpu', '--smoke']
    for script in ('scripts/train_unet_cpstrong_20261025.py', 'scripts/train_refiner_20261031.py'):
        assert "add_argument('--smoke', action='store_true'" in (ROOT / script).read_text(encoding='utf-8'), script


def test_refiner_env_is_what_the_original_launcher_exported():
    env = ft.refiner_env(uuid='GPU-x')
    launcher = (ROOT / 'scripts/launch_refiner_20261031.sh').read_text(encoding='utf-8')
    assert 'export HF_HOME=$PWD/hf_home HF_HUB_OFFLINE=1' in launcher and env['HF_HUB_OFFLINE'] == '1'
    assert env['HF_HOME'] == str(ft.HF_HOME) and env['CUDA_VISIBLE_DEVICES'] == 'GPU-x'
    for name in env:
        assert name in launcher, name
    assert 'CUDA_VISIBLE_DEVICES' not in ft.refiner_env()


def test_gpu_uuid_follows_cuda_visible_devices():
    table = {'0': 'GPU-a', '1': 'GPU-b', '3': 'GPU-d'}
    assert ft.gpu_uuid(0, visible='3', table=table) == 'GPU-d'          # one card by index: cuda:0 is physical 3
    assert ft.gpu_uuid(1, visible='3,1', table=table) == 'GPU-b'
    assert ft.gpu_uuid(0, visible='GPU-zz', table=table) == 'GPU-zz'    # one card by UUID
    assert ft.gpu_uuid(1, visible='', table=table) == 'GPU-b'           # nothing set: nvidia-smi order


def test_fetch_pretrained_caches_what_smp_reads_offline(tmp_path, monkeypatch):
    import huggingface_hub
    yolo = tmp_path / 'yolo.pt'
    yolo.write_bytes(b'y')
    encoder = tmp_path / 'model.safetensors'
    encoder.write_bytes(b'e')
    fetched = []

    def download(repo, filename, revision, cache_dir):
        fetched.append((repo, filename, revision, Path(cache_dir)))
        return str(encoder if filename == 'model.safetensors' else tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    monkeypatch.setattr(ft, 'YOLO_PRETRAINED_SHA', ft.sha256(yolo))
    monkeypatch.setattr(ft, 'ENCODER_REPO', (*ft.ENCODER_REPO[:2], ft.sha256(encoder)))
    ft.fetch_pretrained(yolo, tmp_path / 'hf')
    repo, revision, _ = ft.ENCODER_REPO
    assert fetched == [(repo, f, revision, tmp_path / 'hf/hub') for f in ('config.json', 'model.safetensors')]
    smp_source = Path(__import__('segmentation_models_pytorch').__file__).parent / 'encoders/__init__.py'
    if smp_source.exists():   # the order smp itself downloads in
        text = smp_source.read_text(encoding='utf-8')
        assert text.index('filename="config.json"') < text.index('filename="model.safetensors"')


def test_smoke_and_real_outputs_never_mix(tmp_path):
    assert ft.out_dir(True) == ft.SMOKE_OUT != ft.OUT == ft.out_dir(False)
    with pytest.raises(ValueError):
        ft._check_out(ft.OUT, smoke=True)
    with pytest.raises(ValueError):
        ft._check_out(ft.SMOKE_OUT, smoke=False)
    (tmp_path / 'retrain.json').write_text(json.dumps({'smoke': True, 'runs': []}), encoding='utf-8')
    ft._check_out(tmp_path, smoke=True)
    with pytest.raises(ValueError):
        ft._check_out(tmp_path, smoke=False)
    with pytest.raises(ValueError):
        ft.retrain(steps=('unet',), out=ft.OUT, smoke=True)


def fake_steps(monkeypatch, tmp_path, calls: list) -> None:
    """Replace every step of retrain by a recorder that writes a small file, so its wiring can be checked on CPU."""
    import solar.data
    shipped = tmp_path / 'shipped'
    shipped.mkdir()
    for file in [e['file'] for e in fp.weight_entries().values()] + ['rescorer_A6.json', 'rescorer_r43.json']:
        (shipped / file).write_text('shipped')
    monkeypatch.setattr(fp, 'WEIGHTS', shipped)
    monkeypatch.setattr(solar.data, 'AnnotationStore', lambda data: 'store')
    monkeypatch.setattr(ft, 'gpu_uuid', lambda index: f'GPU-{index}')

    def file(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def detector(member, work, device, data, smoke):
        calls.append(('detector', member, smoke))
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # what Ultralytics' select_device does
        return file(work / 'w.pt', member)

    monkeypatch.setattr(ft, 'prepare_inputs', lambda data, workers: calls.append(('prepare_inputs',)) or {'cache': 'c'})
    monkeypatch.setattr(ft, 'fetch_pretrained', lambda: {'yolo': 'y'})
    monkeypatch.setattr(ft, 'train_unet', lambda out, gpu_index, data, uuid, smoke: calls.append(('unet', uuid, smoke)) or file(out / 'm.pt', 'unet'))
    monkeypatch.setattr(ft, 'train_detector', detector)
    monkeypatch.setattr(ft, 'validation_candidates', lambda pool, work, device, weights, data, workers, log:
                        calls.append(('candidates', pool, Path(weights))) or ([{'stem': 's'}], ['s']))
    monkeypatch.setattr(ft, 'fit_rescorer', lambda inst, stems, store, pool: {'fit': {'pool': pool}})
    monkeypatch.setattr(ft, 'refiner_instances', lambda inst, stems, store: {'images': []})
    monkeypatch.setattr(ft, 'refiner_samples', lambda blob, store, path: {'samples': 3})
    monkeypatch.setattr(ft, 'train_refiner', lambda samples, unet, out, cache, device, data, uuid, smoke:
                        calls.append(('refiner', uuid, smoke)) or file(out / 'r.pt', 'refiner'))


@pytest.mark.parametrize('smoke', [False, True])
def test_retrain_wiring_passes_smoke_gpu_and_restores_the_environment(smoke, tmp_path, monkeypatch):
    calls: list = []
    fake_steps(monkeypatch, tmp_path, calls)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-selected')
    out = tmp_path / 'run'
    produced = ft.retrain(out=out, device='cuda:0', smoke=smoke, log=lambda *a: None)
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-selected'
    assert ('unet', 'GPU-0', smoke) in calls and ('refiner', 'GPU-0', smoke) in calls
    assert [c[1] for c in calls if c[0] == 'detector'] == list(ft.DETECTORS) and all(c[2] == smoke for c in calls if c[0] == 'detector')
    first_stage = {c[2] for c in calls if c[0] == 'candidates'}
    assert first_stage == ({fp.WEIGHTS} if smoke else {out / 'weights'})
    assert (out / 'weights/yolo11m_seg_d1536.pt').read_text() == 'd1536' and (out / 'weights/refiner_effb4_r0.pt').read_text() == 'refiner'
    assert (out / 'weights/SMOKE_RUN_NOT_A_MODEL.txt').exists() == smoke
    record = json.loads((out / 'retrain.json').read_text(encoding='utf-8'))
    assert record['smoke'] == smoke and len(record['runs']) == 1 and record['runs'][0]['status'] == 'done'
    assert set(record['runs'][0]['timings']) == {'inputs', 'unet', *ft.DETECTORS, 'validation_candidates', 'rescorer', 'refiner'}
    assert set(produced) >= {'inputs', 'unet', *ft.DETECTORS, 'rescorer_A6', 'refiner'}
    ft.retrain(steps=('rescorer',), out=out, device='cpu', smoke=smoke, log=lambda *a: None)   # a second run is appended
    assert len(json.loads((out / 'retrain.json').read_text(encoding='utf-8'))['runs']) == 2


def test_retrain_records_a_failed_step(tmp_path, monkeypatch):
    calls: list = []
    fake_steps(monkeypatch, tmp_path, calls)

    def broken(*args, **kwargs):
        raise RuntimeError('boom')

    monkeypatch.setattr(ft, 'train_detector', broken)
    with pytest.raises(RuntimeError, match='boom'):
        ft.retrain(steps=('cp_s43',), out=tmp_path / 'run', device='cpu', log=lambda *a: None)
    run = json.loads((tmp_path / 'run/retrain.json').read_text(encoding='utf-8'))['runs'][0]
    assert run['status'] == 'failed' and 'boom' in run['error']
