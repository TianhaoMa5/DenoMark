import unittest

from denomark.baselines.semantic_detect import calibrated_scan


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
