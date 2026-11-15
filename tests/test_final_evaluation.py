"""Evaluation stage (solar.final_evaluation): PQ, overlap distributions, partner counts, shipped validation predictions."""
import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as coco

from solar import final_evaluation as fe

ROOT = Path(__file__).resolve().parents[1]
HAS_DATA = (ROOT / fe.DEFAULT_DATA / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').exists()


def box(top: int, left: int, height: int, width: int, shape: tuple[int, int] = (64, 64)) -> dict:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[top:top + height, left:left + width] = 1
    return coco.encode(np.asfortranarray(mask))


class Store:
    """Two annotators for 'a', one for 'b'; same interface as solar.data.AnnotationStore.ground_truth."""
    records = {'a.jpeg': [('a1', [box(0, 0, 10, 10), box(30, 0, 10, 40)]), ('a2', [box(0, 0, 10, 12)])],
               'b.jpeg': [('b1', [box(0, 0, 10, 10), box(0, 12, 10, 10)])]}

    def ground_truth(self, name: str) -> list:
        return self.records[name]


PREDS = {'a': [box(0, 0, 10, 10),        # hits a1[0] (IoU 1) and a2[0] (IoU 100/120)
               box(30, 0, 10, 18),       # a1[1] is split: IoU 180/400
               box(30, 20, 10, 20),      # a1[1] is split: IoU 200/400
               box(50, 50, 5, 5)],       # overlaps nothing
         'b': [box(0, 0, 10, 22)]}       # merges b1[0] and b1[1]: IoU 100/220 each


def test_panoptic_quality_scores_every_annotator_record_separately():
    pq = fe.panoptic_quality(PREDS, Store())
    # a1: tp 1, fp 3, fn 1; a2: tp 1, fp 3, fn 0; b1: tp 0, fp 1, fn 2
    assert (pq['tp'], pq['fp'], pq['fn']) == (2, 7, 3)
    assert pq['annotator_records'] == 3 and pq['images'] == 2
    assert pq['sum_iou'] == pytest.approx(1 + 100 / 120)
    assert pq['pq'] == pytest.approx((1 + 100 / 120) / (2 + 3.5 + 1.5))
    assert pq['pq'] == pytest.approx(pq['sq'] * pq['rq'])
    assert pq['split_gt'] == 1 and pq['merge_pred'] == 1


def test_partner_counts_and_overlapping_pairs():
    stats = fe.overlap_statistics(PREDS, Store())
    # annotations: a1[0] 1, a1[1] 2, a2[0] 1, b1[0] 1, b1[1] 1
    assert stats['annotation_partners'] == {1: 4, 2: 1}
    # predictions, once per annotator record: a1 -> 1, 1, 1, 0; a2 -> 1, 0, 0, 0; b1 -> 2
    assert stats['prediction_partners'] == {0: 4, 1: 4, 2: 1}
    assert sorted(np.round(stats['pair_iou'], 6)) == sorted(np.round([1, 180 / 400, 200 / 400, 100 / 120, 100 / 220, 100 / 220], 6))
    # annotators of 'a' overlap in one pair: 10x10 vs 10x12
    assert stats['images_with_two_or_more_annotators'] == 1
    assert stats['annotator_iou'] == pytest.approx([100 / 120])


def test_multi_annotator_subset_keeps_only_the_images_behind_annotator_iou():
    store = Store()
    subset = fe.multi_annotator_subset(PREDS, store)
    assert list(subset) == ['a'] and subset['a'] is PREDS['a']
    stats, full = fe.overlap_statistics(subset, store), fe.overlap_statistics(PREDS, store)
    # the annotator side is unchanged; the prediction side loses the two pairs of 'b'
    assert stats['images_with_two_or_more_annotators'] == full['images_with_two_or_more_annotators'] == 1
    assert stats['annotator_iou'] == pytest.approx(full['annotator_iou'])
    assert sorted(np.round(stats['pair_iou'], 6)) == sorted(np.round([1, 180 / 400, 200 / 400, 100 / 120], 6))
    assert fe.multi_annotator_subset({'b': PREDS['b']}, store) == {}


def test_dice_is_computed_from_iou():
    a, b = coco.decode(box(0, 0, 10, 10)).astype(bool), coco.decode(box(0, 4, 10, 10)).astype(bool)
    direct = 2 * (a & b).sum() / (a.sum() + b.sum())
    iou = (a & b).sum() / (a | b).sum()
    assert fe.dice_from_iou(np.asarray([iou]))[0] == pytest.approx(direct)


def test_empty_sides_do_not_crash():
    store = Store()
    summary = fe.summary({'a': [], 'b': []}, store)
    assert (summary['tp'], summary['fp'], summary['fn']) == (0, 0, 5) and summary['pq'] == 0
    assert summary['annotations_without_overlap'] == 5 and summary['prediction_vs_annotation'] == {'pairs': 0}


def test_summary_counts_split_and_merged():
    summary = fe.summary(PREDS, Store())
    assert summary['one_to_many_annotations'] == 1 and summary['many_to_one_predictions'] == 1
    assert summary['predictions_without_overlap'] == 4 and summary['prediction_vs_annotation']['pairs'] == 6
    json.dumps(summary)   # the notebook writes it to report.json


def test_load_predictions_reads_a_submission_csv(tmp_path):
    from solar.submission import write_submission
    preds = {'20110109104734Ch': [box(100, 100, 30, 40, fe.IMAGE_SHAPE), box(500, 600, 20, 20, fe.IMAGE_SHAPE)]}
    write_submission(tmp_path / 'submission.csv', preds)
    back = fe.load_predictions(tmp_path / 'submission.csv')
    assert list(back) == list(preds)
    assert [int(coco.area(r)) for r in back['20110109104734Ch']] == [1200, 400]


def test_shipped_validation_files_are_complete():
    for setting, path in fe.VAL_PREDICTIONS.items():
        blob = json.loads(path.read_text(encoding='utf-8'))
        assert blob['meta']['setting'] == setting and blob['meta']['recorded'] == fe.RECORDED[setting]
        assert len(blob['images']) == 141 and len({im['stem'] for im in blob['images']}) == 141
        assert {im['half'] for im in blob['images']} == {'A', 'B'}
        assert all(set(i) == {'score', 'original_rle', 'rle'} for im in blob['images'] for i in im['instances'])
    a6, r43 = (json.loads(p.read_text(encoding='utf-8')) for p in fe.VAL_PREDICTIONS.values())
    assert [im['stem'] for im in a6['images']] == [im['stem'] for im in r43['images']]


@pytest.mark.skipif(not HAS_DATA, reason='competition annotations not available')
@pytest.mark.parametrize('setting', ['A6', 'r43'])
def test_shipped_validation_predictions_reproduce_the_recorded_scores(setting, monkeypatch):
    monkeypatch.chdir(ROOT)
    store = fe.AnnotationStore(fe.DEFAULT_DATA)
    for key in ('rle', 'original_rle'):
        fe.check_recorded(fe.panoptic_quality(fe.load_predictions(fe.VAL_PREDICTIONS[setting], key), store), setting, key)


@pytest.mark.skipif(not HAS_DATA, reason='competition annotations not available')
def test_summary_equals_the_reported_validation_numbers(monkeypatch):
    monkeypatch.chdir(ROOT)
    stats = {'model_pairs': 1368, 'model_iou_mean': 0.6039012818176855, 'model_dice_mean': 0.7330755459169981,
             'human_pairs': 663, 'human_images': 61,
             'gt_degree': {'0': 333, '1': 1263, '2': 42, '3': 7}, 'pred_degree': {'0': 164, '1': 1245, '2': 54, '3': 5}}
    summary = fe.summary(fe.load_predictions(fe.VAL_PREDICTIONS['A6']), fe.AnnotationStore(fe.DEFAULT_DATA))
    assert summary['annotator_records'] == 228
    assert summary['prediction_vs_annotation']['pairs'] == stats['model_pairs']
    assert summary['prediction_vs_annotation']['iou_mean'] == pytest.approx(stats['model_iou_mean'], abs=1e-12)
    assert summary['prediction_vs_annotation']['dice_mean'] == pytest.approx(stats['model_dice_mean'], abs=1e-12)
    assert summary['annotator_vs_annotator']['pairs'] == stats['human_pairs']
    assert summary['annotator_vs_annotator']['images'] == stats['human_images']
    assert {str(k): v for k, v in summary['annotation_partners'].items()} == stats['gt_degree']
    assert {str(k): v for k, v in summary['prediction_partners'].items()} == stats['pred_degree']
