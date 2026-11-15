"""Memory-efficient RLE adapter for official snapshot Version 6 (Apache-2.0).

Same strict IoU>0.5 hit-matrix counts, no assignment. IoU uses pycocotools
instead of full-resolution float32 torch layers. See docs/competition/official_code/README.md.
"""
import numpy as np
from pycocotools import mask as coco


def score_instances(gt: list[dict], pred: list[dict]) -> dict:
    iou = coco.iou(pred, gt, [0] * len(gt)).T if gt and pred else np.zeros((len(gt), len(pred)))
    hit = iou > .5
    areas = np.asarray([int(coco.area(r)) for r in gt])
    matched = hit.any(axis=1)
    small = areas < 800
    return {'tp': int(hit.sum()), 'fp': int((hit.sum(axis=0) == 0).sum()),
            'fn': int((hit.sum(axis=1) == 0).sum()), 'sum_iou': float(iou[hit].sum()),
            'gt': len(gt), 'pred': len(pred), 'small_gt': int(small.sum()),
            'small_hit': int((small & matched).sum()),
            'split_gt': int(((iou > 0).sum(axis=1) > 1).sum()),
            'merge_pred': int(((iou > 0).sum(axis=0) > 1).sum())}


def aggregate(rows: list[dict]) -> dict:
    keys = ('tp', 'fp', 'fn', 'sum_iou', 'gt', 'pred', 'small_gt', 'small_hit', 'split_gt', 'merge_pred')
    totals = {k: sum(r[k] for r in rows) for k in keys}
    denom = totals['tp'] + .5 * totals['fp'] + .5 * totals['fn']
    totals['pq'] = totals['sum_iou'] / denom if denom else 0.
    totals['small_recall'] = totals['small_hit'] / totals['small_gt'] if totals['small_gt'] else None
    return totals
