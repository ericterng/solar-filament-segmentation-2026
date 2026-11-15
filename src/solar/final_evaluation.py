"""Evaluation stage of the demo notebook: PQ, IoU / Dice distributions and one-to-many / many-to-one counts.

The four quantities are the ones the competition rubric names. Definitions follow the official
self-evaluation notebook (docs/competition/official_code/self_evaluation.ipynb):

- every annotator's record of an image is scored separately (records are never merged across annotators);
- PQ counts a hit when IoU > .5 (solar.evaluation.score_instances, the memory-efficient RLE adapter);
- the IoU and Dice distributions use every annotation-prediction pair that overlaps (score > 0);
- a "partner" of a mask is any mask of the other side that overlaps it (IoU > 0); an annotation with two or
  more partners is split (one-to-many), a prediction with two or more partners is merged (many-to-one).

Two ways to get predictions to score:

1. `load_predictions(VAL_PREDICTIONS[setting])`: the cross-fitted fold1 validation predictions shipped in
   docs/report/val_predictions/ (141 images that neither the U-Net nor the detectors were trained on; the
   rescorer and the refiner were fitted on one date half and applied to the other). CPU, seconds. `RECORDED`
   holds the scores recorded when they were produced; `check_recorded` stops if a rerun differs.
2. any `{stem: [COCO RLE, ...]}` mapping, e.g. the output of solar.final_pipeline run on validation images, or a
   submission CSV of training images read with `load_predictions`.

    from solar import final_evaluation as fe
    preds = fe.load_predictions(fe.VAL_PREDICTIONS['A6'])
    print(fe.summary(preds, fe.AnnotationStore(fe.DEFAULT_DATA)))
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools import mask as coco

from solar.data import AnnotationStore, DEFAULT_DATA
from solar.evaluation import aggregate, score_instances

ROOT = Path(__file__).resolve().parents[2]
VAL_DIR = ROOT / 'docs/report/val_predictions'
# setting -> shipped cross-fitted fold1 validation predictions (scripts/export_val_predictions_20261112.py)
VAL_PREDICTIONS = {'A6': VAL_DIR / 'val_fold1_A6_crossfit.json', 'r43': VAL_DIR / 'val_fold1_r43_crossfit.json'}
# Scores recorded when the predictions were produced: after the refiner ('rle') and before it ('original_rle').
# A6: outputs/score_amey_val_refiner_20261111/report.json; r43: outputs/fusion_probe_refiner_20261031/report.json.
RECORDED = {'A6': {'rle': {'tp': 1057, 'fp': 411, 'fn': 588, 'pq': 0.46698114914943156},
                   'original_rle': {'tp': 1053, 'fp': 415, 'fn': 592, 'pq': 0.46393793272459877}},
            'r43': {'rle': {'tp': 1056, 'fp': 410, 'fn': 589, 'pq': 0.466807817661896},
                    'original_rle': {'pq': 0.46262466262312496}}}
IMAGE_SHAPE = (2048, 2048)


def _rle(rle: dict) -> dict:
    counts = rle['counts']
    return {'size': list(rle['size']), 'counts': counts if isinstance(counts, bytes) else counts.encode('ascii')}


def load_predictions(path: Path, key: str = 'rle') -> dict[str, list[dict]]:
    """Read `{stem: [RLE, ...]}` from a shipped validation JSON (`key` 'rle' = refined, 'original_rle' = before the
    refiner) or from a submission CSV. Images without predictions are kept with an empty list (JSON only)."""
    path = Path(path)
    if path.suffix == '.csv':
        from solar.final_pipeline import read_csv_rles
        return read_csv_rles(path)
    blob = json.loads(path.read_text(encoding='utf-8'))
    return {im['stem']: [_rle(i[key]) for i in im['instances']] for im in blob['images']}


def annotator_records(store: AnnotationStore, stem: str) -> list[list[dict]]:
    """One list of masks per annotator of this image."""
    return [masks for _, masks in store.ground_truth(stem + '.jpeg')]


def iou_matrix(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """IoU, shape (len(gt), len(pred)); the same pycocotools call as solar.evaluation.score_instances."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    return np.asarray(coco.iou(pred, gt, [0] * len(gt))).T


def panoptic_quality(preds: dict[str, list[dict]], store: AnnotationStore) -> dict:
    """PQ over all annotator records of the images in `preds`, with SQ, RQ and the split / merge counts."""
    rows = [score_instances(gt, masks) for stem, masks in preds.items() for gt in annotator_records(store, stem)]
    total = aggregate(rows)
    denom = total['tp'] + .5 * total['fp'] + .5 * total['fn']
    total.update({'sq': total['sum_iou'] / total['tp'] if total['tp'] else 0., 'rq': total['tp'] / denom if denom else 0.,
                  'images': len(preds), 'annotator_records': len(rows)})
    return total


def overlap_statistics(preds: dict[str, list[dict]], store: AnnotationStore) -> dict:
    """IoU of every overlapping annotation-prediction pair, IoU of every overlapping pair between two annotators
    of the same image, and the partner counts of annotations and predictions (one entry per annotator record)."""
    pairs, human = [], []
    gt_partners, pred_partners = Counter(), Counter()
    multi = 0
    for stem, masks in preds.items():
        records = annotator_records(store, stem)
        multi += len(records) > 1
        for gt in records:
            iou = iou_matrix(gt, masks)
            pairs.append(iou.ravel())
            if not gt:
                pred_partners.update([0] * len(masks))
            elif not masks:
                gt_partners.update([0] * len(gt))
            else:
                gt_partners.update((iou > 0).sum(axis=1).tolist())
                pred_partners.update((iou > 0).sum(axis=0).tolist())
        for a in range(len(records)):
            for b in range(a + 1, len(records)):
                human.append(iou_matrix(records[a], records[b]).ravel())
    pair_iou = np.concatenate(pairs) if pairs else np.zeros(0)
    human_iou = np.concatenate(human) if human else np.zeros(0)
    return {'pair_iou': pair_iou[pair_iou > 0], 'annotator_iou': human_iou[human_iou > 0],
            'images_with_two_or_more_annotators': int(multi),
            'annotation_partners': dict(sorted(gt_partners.items())), 'prediction_partners': dict(sorted(pred_partners.items()))}


def multi_annotator_subset(preds: dict[str, list[dict]], store: AnnotationStore) -> dict[str, list[dict]]:
    """The images of `preds` that two or more annotators drew: exactly the images behind `annotator_iou`.
    Score this subset to compare prediction-vs-annotation with annotator-vs-annotator on the same images."""
    return {stem: masks for stem, masks in preds.items() if len(annotator_records(store, stem)) > 1}


def dice_from_iou(iou: np.ndarray) -> np.ndarray:
    """Dice of one pair of masks is a function of its IoU: 2 IoU / (1 + IoU)."""
    return 2 * iou / (1 + iou)


def summary(preds: dict[str, list[dict]], store: AnnotationStore) -> dict:
    """The rubric's four quantities as plain numbers (JSON-serialisable)."""
    pq, stats = panoptic_quality(preds, store), overlap_statistics(preds, store)

    def describe(iou: np.ndarray) -> dict:
        if not len(iou):
            return {'pairs': 0}
        dice = dice_from_iou(iou)
        return {'pairs': int(len(iou)), 'iou_mean': float(iou.mean()), 'iou_median': float(np.median(iou)),
                'dice_mean': float(dice.mean()), 'dice_median': float(np.median(dice)),
                'iou_quartiles': [float(q) for q in np.percentile(iou, [25, 50, 75])]}

    split = sum(n for k, n in stats['annotation_partners'].items() if k > 1)
    merged = sum(n for k, n in stats['prediction_partners'].items() if k > 1)
    return {'pq': pq['pq'], 'sq': pq['sq'], 'rq': pq['rq'], 'tp': pq['tp'], 'fp': pq['fp'], 'fn': pq['fn'],
            'images': pq['images'], 'annotator_records': pq['annotator_records'],
            'prediction_vs_annotation': describe(stats['pair_iou']),
            'annotator_vs_annotator': {**describe(stats['annotator_iou']), 'images': stats['images_with_two_or_more_annotators']},
            'annotation_partners': stats['annotation_partners'], 'prediction_partners': stats['prediction_partners'],
            'one_to_many_annotations': int(split), 'many_to_one_predictions': int(merged),
            'annotations_without_overlap': int(stats['annotation_partners'].get(0, 0)),
            'predictions_without_overlap': int(stats['prediction_partners'].get(0, 0))}


def check_recorded(result: dict, setting: str, key: str = 'rle') -> None:
    """Stop when a rerun of the shipped validation predictions differs from the recorded score."""
    for name, value in RECORDED[setting][key].items():
        if abs(result[name] - value) > 1e-9:
            raise AssertionError(f'{setting}/{key}: {name} = {result[name]!r}, recorded {value!r}')
