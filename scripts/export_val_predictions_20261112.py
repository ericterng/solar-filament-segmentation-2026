"""Copy the cross-fitted fold1 validation predictions into the repo (docs/report/val_predictions/). CPU, no model.

The evaluation cell of notebooks/final_inference.ipynb and scripts/report_figures_20261111.py read these files, so
the reported validation numbers and figures can be reproduced from a clone of the repo plus the competition data.

Sources (under outputs/, not in git), one per detector setting:
- A6  (6 detectors, the submitted setting): outputs/refiner_20261031/amey_20261104/A6/refined_val_s0_r0.json
- r43 (3 detectors):                        outputs/refiner_20261031/refined_val_s0_r0.json
Both hold the 141 fold1 validation images. The U-Net (cpstrong_s0) and the detectors never saw them; the rescorer
and the refiner (r_s0_train{A,B}_r0) were fitted on one date half and applied to the other.

Only what scoring and the figures need is kept: per image the stem, its date half and, per kept instance, the fusion
score, the mask before the refiner (original_rle) and after it (rle). Run-specific metadata of the source is dropped;
its SHA-256 is recorded instead. Each slim file is re-scored before it is written and must equal the recorded PQ.
Refuses to overwrite unless --force.

    .venv-unet/Scripts/python.exe scripts/export_val_predictions_20261112.py
"""
import pandas  # noqa: F401  # Windows .venv-unet: pandas imported after cv2 + smp crashes (access violation)
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from solar import final_evaluation as fe  # noqa: E402
from solar.data import sha256  # noqa: E402

SOURCES = {'A6': ROOT / 'outputs/refiner_20261031/amey_20261104/A6/refined_val_s0_r0.json',
           'r43': ROOT / 'outputs/refiner_20261031/refined_val_s0_r0.json'}
DETECTORS = {'A6': ['cp_s43', 'm1280', 'm1024', 'det4', 'd1024', 'd1536'], 'r43': ['cp_s43', 'm1280', 'm1024']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    store = fe.AnnotationStore(fe.DEFAULT_DATA)
    for setting, source in SOURCES.items():
        out = fe.VAL_PREDICTIONS[setting]
        if out.exists() and not args.force:
            raise FileExistsError(out)
        blob = json.loads(source.read_text(encoding='utf-8'))
        assert blob['meta']['members'] == DETECTORS[setting] and blob['meta']['unet'] == 'cpstrong_s0', blob['meta']['members']
        assert len(blob['images']) == 141 and all(i['kept'] for im in blob['images'] for i in im['instances'])
        images = [{'stem': im['stem'], 'half': im['half'],
                   'instances': [{'score': i['score'], 'original_rle': i['original_rle'], 'rle': i['rle']} for i in im['instances']]}
                  for im in blob['images']]
        meta = {'setting': setting, 'split': 'fold1 validation, 141 images, grouped by observation date',
                'unet': 'cpstrong_s0 (weights/unet_effb4_cpstrong_s0.pt)', 'detectors': DETECTORS[setting],
                'rescorer': 'six-feature logistic, cross-fitted on the two date halves (field "half")',
                'refiner': 'r_s0_train{A,B}_r0: each half is refined by the model trained on the other half',
                'fields': {'score': 'fusion score', 'original_rle': 'mask before the refiner', 'rle': 'mask after the refiner'},
                'source': str(source.relative_to(ROOT)).replace('\\', '/'), 'source_sha256': sha256(source),
                'recorded': fe.RECORDED[setting]}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'meta': meta, 'images': images}, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
        for key in ('rle', 'original_rle'):
            result = fe.panoptic_quality(fe.load_predictions(out, key), store)
            fe.check_recorded(result, setting, key)
            print(setting, key, {k: result[k] for k in ('tp', 'fp', 'fn', 'pq')})
        print('wrote', out.relative_to(ROOT), out.stat().st_size, 'bytes', sha256(out)[:12])


if __name__ == '__main__':
    main()
