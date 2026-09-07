import random
import unittest

import torch

from denomark.baselines.pmark.model import (
    build_pmark_pivots,
    build_pmark_secret_bits,
    pmark_soft_matches,
    select_pmark_candidate,
)


class PMarkBlockTests(unittest.TestCase):
    def test_pivots_are_deterministic_and_orthogonal(self):
        pivots_a = build_pmark_pivots(16, 4, 42)
        pivots_b = build_pmark_pivots(16, 4, 42)
        self.assertTrue(torch.equal(pivots_a, pivots_b))
        self.assertTrue(torch.allclose(pivots_a @ pivots_a.T, torch.eye(4), atol=1e-5))

    def test_secret_bits_are_deterministic(self):
        bits_a = build_pmark_secret_bits(12, 4, 0)
        bits_b = build_pmark_secret_bits(12, 4, 0)
        self.assertTrue(torch.equal(bits_a, bits_b))
        self.assertEqual(tuple(bits_a.shape), (12, 4))

    def test_prior_filter_applies_channels_sequentially(self):
        projections = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
        random.seed(0)
        selected, medians, survivors = select_pmark_candidate(
            projections,
            torch.tensor([True, False]),
            [True, True, True, True],
            median_method="prior",
        )
        self.assertEqual(selected, 1)
        self.assertEqual(survivors, [1])
        self.assertEqual(medians, [0.0, 0.0])

    def test_soft_detector_matches_official_prior_rule(self):
        projections = torch.tensor([0.01, -0.01, -0.02, 0.02])
        targets = torch.tensor([True, False, True, False])
        scores = pmark_soft_matches(projections, targets, tolerance=0.001, decay=250)
        self.assertEqual(scores[:2].tolist(), [1.0, 1.0])
        self.assertTrue(torch.all(scores[2:] < 1.0))
        self.assertTrue(torch.all(scores[2:] > 0.0))


if __name__ == "__main__":
    unittest.main()
