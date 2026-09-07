import numpy as np
import pytest

from denomark.evaluation.metrics import (
    calibrated_scan_scores,
    rank_auc,
    roc_interpolated_tpr,
    score_matrix,
)


def test_score_matrix_uses_string_unit_size_keys():
    records = [
        {"raw_scores_by_unit_size": {"12": 1.0, "13": 2.0}},
        {"raw_scores_by_unit_size": {"12": 3.0, "13": 4.0}},
    ]
    assert score_matrix(records, [12, 13]).tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_calibrated_scan_is_per_size_then_bonferroni():
    calibration = np.asarray([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0]])
    evaluation = np.asarray([[3.0, 0.0], [1.5, 1.5], [-1.0, -1.0]])
    scores = calibrated_scan_scores(evaluation, calibration)
    # Row 0 has min p=1/4, corrected by two sizes to 1/2.
    assert scores[0] == pytest.approx(-np.log(0.5))
    # Row 2 is no stronger than any calibration sample and clips to p=1.
    assert scores[2] == pytest.approx(0.0)


def test_rank_auc_gives_half_credit_to_ties():
    assert rank_auc([1.0, 2.0], [0.0, 2.0]) == pytest.approx(0.625)


def test_roc_interpolation_handles_vertical_tied_segments():
    value = roc_interpolated_tpr([2.0, 2.0], [2.0, 1.0], 0.25)
    assert value == pytest.approx(0.5)


def test_scan_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        calibrated_scan_scores(np.asarray([[np.nan]]), np.asarray([[0.0]]))
