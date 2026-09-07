from types import SimpleNamespace

import pytest

from denomark.core.calibration import (
    CalibratedDetectorSuite,
    empirical_right_tail_p,
    empirical_right_tail_p_loo,
)


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
    suite = CalibratedDetectorSuite(
        FakeRawDetector(), negatives, detectors=["calibrated_scan"]
    )
    negative = suite.score_item(negatives[0], "calibrated_scan")
    positive = suite.score_item(
        {"text": "p", "token_ids": [4.0, 4.0]}, "calibrated_scan"
    )
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
    positive = suite.score_item(
        {"text": "p", "token_ids": [4.0, 4.0]}, "calibrated_scan"
    )
    assert positive["p_value"] == pytest.approx(0.5)
    assert suite.neg_scores["calibrated_scan"] == pytest.approx([0.69314718056])
