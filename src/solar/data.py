"""Competition data and deterministic date-grouped partitions."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

DEFAULT_DATA = Path('data/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def make_split(rows: list[dict], seed: int = 20261005) -> dict:
    """Shuffle sorted dates with NumPy PCG64, reserve ceil(20%) dates."""
    train = sorted((r for r in rows if r['split'] == 'train'), key=lambda r: r['file_name'])
    dates = sorted({r['date'] for r in train})
    if len(dates) < 2:
        raise ValueError('At least two dates required')
    rng = np.random.default_rng(seed)
    selected = set(rng.permutation(dates)[:int(np.ceil(len(dates) * .2))])
    result = {'seed': seed, 'algorithm': 'sorted dates; NumPy PCG64 permutation; ceil(0.2*n_dates)',
              'train': [r['file_name'] for r in train if r['date'] not in selected],
              'validation': [r['file_name'] for r in train if r['date'] in selected]}
    result['train_dates'] = sorted(set(dates) - selected)
    result['validation_dates'] = sorted(selected)
    for side in ('train', 'validation'):
        subset = [r for r in train if r['file_name'] in set(result[side])]
        result[side + '_distribution'] = {
            'images': len(subset), 'dates': len({r['date'] for r in subset}),
            'stations': dict(Counter(r['station'] for r in subset)),
            'years': dict(Counter(r['date'][:4] for r in subset))}
    result['smoke'] = spaced_sample(result['validation'], 20)
    result['development'] = spaced_sample(result['train'], 24)
    assert not set(result['train']) & set(result['validation'])
    assert not set(result['train_dates']) & set(result['validation_dates'])
    return result


def spaced_sample(names: list[str], count: int) -> list[str]:
    indices = np.linspace(0, len(names) - 1, min(count, len(names)), dtype=int)
    return [names[i] for i in indices]


class AnnotationStore:
    """Keep annotator records separate; never concatenate across annotators."""
    def __init__(self, data: Path):
        self.path = data / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
        raw = json.loads(self.path.read_text(encoding='utf-8'))
        self.records = defaultdict(list)
        self.annotations = defaultdict(list)
        for record in raw['images']:
            self.records[record['file_name']].append(record)
        for ann in raw['annotations']:
            self.annotations[ann['image_id']].append(ann)

    def ground_truth(self, name: str) -> list[tuple[str, list[dict]]]:
        result = []
        for rec in self.records[name]:
            masks = [coco.merge(coco.frPyObjects(a['segmentation'], rec['height'], rec['width']))
                     for a in self.annotations[rec['id']]]
            result.append((rec['id'], masks))
        return result


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))
