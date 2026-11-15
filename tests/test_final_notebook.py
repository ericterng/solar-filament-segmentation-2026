"""Demo notebook (notebooks/final_inference.ipynb): structure, defaults, and the cells that need no model."""
import json
from pathlib import Path

import pandas  # noqa: F401  (.venv-unet on Windows: import pandas before cv2 + smp)
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/final_inference.ipynb'
DATA = ROOT / 'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'


def cells() -> list[dict]:
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']


def source(cell: dict) -> str:
    return ''.join(cell['source'])


def test_sections_follow_the_required_order_and_cells_compile():
    headings = [source(c).splitlines()[0] for c in cells() if c['cell_type'] == 'markdown' and source(c).startswith('## ')]
    assert headings == ['## 1. Preprocessing', '## 2. Training (optional)', '## 3. Inference', '## 4. Post-processing',
                        '## 5. Evaluation', '## Summary']
    ids = [c['id'] for c in cells()]
    assert len(ids) == len(set(ids))
    for i, cell in enumerate(cells()):
        if cell['cell_type'] == 'code':
            compile(source(cell), f'cell {i}', 'exec')
            assert cell['outputs'] == [] and cell['execution_count'] is None, 'commit the notebook without outputs'


def test_defaults_load_the_submitted_setting_and_do_not_train():
    scope: dict = {}
    exec(source(cells()[1]), scope)
    assert scope['RUN_FINALS'] == ['final1_A6'] and scope['TRAIN'] is False and scope['EVALUATE_RERUN'] is False
    assert scope['TRAIN_SMOKE'] is False   # smoke training writes to its own folder and is never the default
    from solar import final_training as ft
    assert tuple(scope['TRAIN_STEPS']) == ft.STEPS
    code = '\n'.join(source(c) for c in cells() if c['cell_type'] == 'code')
    assert 'smoke=TRAIN_SMOKE' in code and 'fp.WEIGHTS = ft.out_dir(TRAIN_SMOKE)' in code
    for call in ('fp.check_weights()', 'fp.preprocess(', 'ft.retrain(', 'fp.unet_logits(', 'fp.detector_views(', 'fp.select_instances(',
                 'fp.refine(', 'fp.write_csv(', 'fe.summary(', 'fe.check_recorded(', 'fp.predict('):
        assert call in code, call


@pytest.mark.skipif(not (DATA / 'test/test_images').exists() or not (ROOT / 'weights/unet_effb4_cpstrong_s0.pt').exists(),
                    reason='competition data or weights not available')
def test_setup_training_and_evaluation_cells_run_without_a_model(tmp_path, monkeypatch, capsys):
    pytest.importorskip('torch')
    monkeypatch.chdir(ROOT / 'notebooks')   # as nbconvert runs it; the setup cell moves to the repository root
    by_id = {c['id']: source(c) for c in cells()}
    scope: dict = {}
    exec(by_id['54ed94e0'], scope)                      # parameters
    scope.update({'OUT_DIR': str(tmp_path), 'DEVICE': 'cpu'})
    exec(by_id['ebbd87d9'], scope)                      # setup: imports, SHA check of weights/, test image list
    assert Path.cwd() == ROOT and len(scope['stems']) == 180 and len(scope['report']['weights']) == 10   # 8 weights + 2 rescorers
    assert scope['members'] == scope['fp'].POOLS['A6']
    exec(by_id['c4d58f06'], scope)                      # training: prints the recipes, trains nothing by default
    exec(by_id['0a6c72d5'], scope)                      # evaluation of the recorded cross-fitted predictions
    exec(by_id['1b9e84f3'], scope)                      # optional rerun: off by default
    out = capsys.readouterr().out
    assert 'TRAIN = False' in out and 'EVALUATE_RERUN = False' in out and 'unet      -> unet_effb4_cpstrong_s0.pt' in out
    evaluation = scope['report']['evaluation']
    assert list(evaluation) == ['final1_A6 cross-fitted, before the refiner', 'final1_A6 cross-fitted, after the refiner']
    after = evaluation['final1_A6 cross-fitted, after the refiner']
    assert (after['tp'], after['fp'], after['fn']) == (1057, 411, 588) and abs(after['pq'] - 0.46698114914943156) < 1e-9
    assert 'training' not in scope['report'] and scope['fp'].WEIGHTS == ROOT / 'weights'
    json.dumps(scope['report'], default=str)
