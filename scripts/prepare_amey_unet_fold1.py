"""Write ameyypawar's splits.json using the HeShen author fold1 observations.

The U-Net then shares YOLO fold1's validation set (141 observations, 566 train),
so every fusion score on those 141 images is out-of-sample for both models.
Every annotator view follows its observation. Refuses to overwrite.

    .venv-unet/Scripts/python.exe scripts/prepare_amey_unet_fold1.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT/'third_party/ameyypawar-filament-seg'
sys.path.insert(0, str(VENDOR))
from filament_seg.data import load_annotations  # noqa: E402

DATA = ROOT/'data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
YOLO_VAL = ROOT/'outputs/author_fold1_submission_20261010/dev_replay/raw_boxes.json'
OUT = ROOT/'outputs/amey_unet_fold1_20261010'


def main() -> None:
    path = OUT/'splits.json'
    if path.exists():
        raise FileExistsError(path)
    annotations = load_annotations(DATA/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    val_stems = set(json.loads(YOLO_VAL.read_text()))
    stems = {r.stem for r in annotations.images.values()}
    assert len(val_stems) == 141 and val_stems <= stems and len(stems) == 707
    train = sorted(i for i, r in annotations.images.items() if r.stem not in val_stems)
    val = sorted(i for i, r in annotations.images.items() if r.stem in val_stems)
    train_stems = {annotations.images[i].stem for i in train}
    assert len(train_stems) == 566 and not train_stems & val_stems
    assert not {s[:8] for s in train_stems} & {s[:8] for s in val_stems}, 'date leak'
    OUT.mkdir(parents=True, exist_ok=True)
    split = {'group_by': 'heshen_author_fold1_date', 'seed': 42, 'val_fraction': None,
             'n_groups': None, 'train': train, 'val': val}
    path.write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(f'train {len(train)} views / 566 obs; val {len(val)} views / 141 obs -> {path}')


if __name__ == '__main__':
    main()
