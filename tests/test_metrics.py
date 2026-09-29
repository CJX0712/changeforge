"""评测指标测试。

作者: 晨星
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from changeforge.core.errors import EvaluationError, InvalidChangePointsError
from changeforge.eval import (
    adjusted_rand_index,
    evaluate,
    evaluate_many,
    labels_from_change_points,
    match_change_points,
)


def test_perfect_match_f1_is_one():
    report = evaluate([30, 70], [30, 70], n_samples=100, tolerance=0)
    assert report.f1 == pytest.approx(1.0)
    assert report.precision == pytest.approx(1.0)
    assert report.recall == pytest.approx(1.0)
    assert report.mean_abs_error == pytest.approx(0.0)
    assert report.count_error == 0


def test_within_tolerance_counts_as_hit():
    report = evaluate([30, 70], [32, 68], n_samples=100, tolerance=5)
    assert report.matched == 2
    assert report.f1 == pytest.approx(1.0)
    assert report.mean_abs_error == pytest.approx(2.0)


def test_out_of_tolerance_misses():
    report = evaluate([30, 70], [60], n_samples=100, tolerance=5)
    assert report.matched == 0
    assert report.precision == pytest.approx(0.0)
    assert report.recall == pytest.approx(0.0)
    assert math.isnan(report.mean_abs_error)


def test_empty_prediction_and_empty_truth():
    report = evaluate([], [], n_samples=100, tolerance=3)
    assert report.f1 == pytest.approx(1.0)
    assert report.count_error == 0


def test_empty_prediction_with_truth():
    report = evaluate([30], [], n_samples=100, tolerance=3)
    assert report.precision == pytest.approx(0.0)
    assert report.recall == pytest.approx(0.0)
    assert report.count_error == -1


def test_over_segmentation_count_error():
    report = evaluate([30], [10, 30, 50, 90], n_samples=100, tolerance=0)
    assert report.count_error == 3
    assert report.matched == 1


def test_matching_is_one_to_one():
    pairs, unmatched_true, unmatched_pred = match_change_points([30, 31], [30, 31], tolerance=5)
    assert len(pairs) == 2
    assert unmatched_true == []
    assert unmatched_pred == []


def test_labels_from_change_points():
    labels = labels_from_change_points([3, 7], 10)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]


def test_ari_identical_is_one():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_ari_mismatched_shapes_raise():
    with pytest.raises(EvaluationError):
        adjusted_rand_index(np.zeros(5), np.zeros(4))


def test_out_of_range_change_points_raise():
    with pytest.raises(InvalidChangePointsError):
        evaluate([200], [30], n_samples=100, tolerance=1)


def test_negative_tolerance_raises():
    with pytest.raises(EvaluationError):
        evaluate([30], [30], n_samples=100, tolerance=-1)


def test_report_to_dict_is_serialisable():
    payload = evaluate([30], [31], n_samples=100, tolerance=2).to_dict()
    assert payload["matched"] == 1
    assert payload["tolerance"] == 2


def test_evaluate_many():
    reports = evaluate_many([30, 70], [[30, 70], [30]], n_samples=100, tolerance=0)
    assert len(reports) == 2
    assert reports[0].f1 > reports[1].f1


def test_ari_single_sample_is_one():
    assert adjusted_rand_index(np.zeros(1), np.zeros(1)) == pytest.approx(1.0)
