"""Empirical scanning, calibration separation, ties, and ROC interpolation."""

import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from denmark.baselines.semantic_detect import calibrated_scan
from denmark.core.calibration import (
    CalibratedDetectorSuite,
    empirical_right_tail_p,
    empirical_right_tail_p_loo,
)
from denmark.evaluation.metrics import (
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


class FakeRawDetector:
    scan_block_sizes = [12, 13]

    def raw(self, detector, text, token_ids, block_size=None):
        assert detector == "scan_raw"
        return SimpleNamespace(raw=float(token_ids[block_size - 12]))


def test_empirical_loo_excludes_only_self_and_keeps_other_ties():
    pool = [3.0, 3.0, 1.0]
    assert empirical_right_tail_p(3.0, pool) == pytest.approx(3.0 / 4.0)
    assert empirical_right_tail_p_loo(3.0, pool) == pytest.approx(2.0 / 3.0)


def test_suite_uses_loo_only_for_negative_objects():
    negatives = [
        {"text": "n0", "token_ids": [3.0, 3.0]},
        {"text": "n1", "token_ids": [2.0, 2.0]},
        {"text": "n2", "token_ids": [1.0, 1.0]},
    ]
    suite = CalibratedDetectorSuite(FakeRawDetector(), negatives, detectors=["calibrated_scan"])
    negative = suite.score_item(negatives[0], "calibrated_scan")
    positive = suite.score_item({"text": "p", "token_ids": [4.0, 4.0]}, "calibrated_scan")
    assert negative["p_value"] == pytest.approx(2.0 / 3.0)
    assert positive["p_value"] == pytest.approx(0.5)


def test_suite_supports_disjoint_calibration_and_roc_negatives():
    calibration = [
        {"text": "c0", "token_ids": [3.0, 3.0]},
        {"text": "c1", "token_ids": [2.0, 2.0]},
        {"text": "c2", "token_ids": [1.0, 1.0]},
    ]
    heldout = [{"text": "n0", "token_ids": [100.0, 100.0]}]
    suite = CalibratedDetectorSuite(
        FakeRawDetector(),
        heldout,
        detectors=["calibrated_scan"],
        calibration_items=calibration,
    )
    positive = suite.score_item({"text": "p", "token_ids": [4.0, 4.0]}, "calibrated_scan")
    assert positive["p_value"] == pytest.approx(0.5)
    assert suite.neg_scores["calibrated_scan"] == pytest.approx([0.69314718056])


class CalibratedScanTests(unittest.TestCase):
    def test_negative_scores_are_out_of_fold_and_positive_uses_full_calibration(self):
        result = calibrated_scan(
            [25],
            {25: [100.0, 0.0, 100.0, 0.0]},
            {25: [100.0]},
            [0, 1, 0, 1],
        )
        self.assertEqual(
            [record["n_calibration"] for record in result["negative_records"]],
            [2, 2, 2, 2],
        )
        self.assertEqual(result["positive_records"][0]["n_calibration"], 4)
        self.assertEqual(result["negative_fold_ids"], [0, 1, 0, 1])
        # A high held-out negative sees only the two low scores in the other
        # fold, proving it was not calibrated on itself or its prompt group.
        self.assertAlmostEqual(
            result["negative_records"][0]["best_single_size_p"],
            1.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
