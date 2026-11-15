"""Strict CSV/RLE serialization; empty detections recorded in a separate manifest."""
import csv
import re
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

ID_PATTERN = re.compile(r'^(\d{14}[A-Za-z]{2})_([1-9]\d*)$')


def parse_prediction_id(value: str) -> str:
    match = ID_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f'Invalid prediction ID: {value}')
    return match.group(1)


def write_submission(path: Path, predictions: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, masks in sorted(predictions.items()):
            for index, rle in enumerate(masks, 1):
                ident = f'{stem}_{index}'
                parse_prediction_id(ident)
                counts = rle['counts']
                writer.writerow([ident, counts.decode('ascii') if isinstance(counts, bytes) else counts])


def validate_submission(path: Path, expected: dict[str, int], size: tuple[int, int] = (2048, 2048),
                        require_nonoverlap: bool = True) -> dict:
    ids = set()
    unions = {}
    counts = {stem: 0 for stem in expected}
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Unexpected CSV columns')
        for row in reader:
            ident = row['filament_id']
            stem = parse_prediction_id(ident)
            if ident in ids or stem not in expected:
                raise ValueError('Duplicate ID or unknown image')
            ids.add(ident)
            value = row['segmentation_rle']
            rle = {'size': list(size), 'counts': value.encode('ascii')}
            # Round-trip validates counts and dimensions; no empty instance is added.
            mask = coco.decode(rle)
            if mask.shape != size or not np.any(mask) or not np.isin(mask, [0, 1]).all():
                raise ValueError('Invalid mask')
            encoded = coco.encode(np.asfortranarray(mask))['counts'].decode('ascii')
            if encoded != value:
                raise ValueError('Noncanonical or invalid RLE')
            if require_nonoverlap:
                if stem in unions:
                    if coco.iou([rle], [unions[stem]], [0])[0, 0] > 0:
                        raise ValueError(f'Overlapping masks in image {stem}')
                    unions[stem] = coco.merge([unions[stem], rle])
                else:
                    unions[stem] = rle
            counts[stem] += 1
    if counts != expected:
        raise ValueError('CSV counts disagree with completed inference manifest')
    return {'images_completed': len(expected), 'instances': len(ids),
            'zero_detection_images': [stem for stem, n in counts.items() if n == 0],
            'rle_round_trips_verified': len(ids), 'nonoverlap_verified': require_nonoverlap}
