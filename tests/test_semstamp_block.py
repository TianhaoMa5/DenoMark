import unittest
import torch

from denomark.baselines.semstamp.model import (
    build_lsh_hyperplanes,
    embedding_hashes_and_margins,
    first_accepted_candidate,
    valid_bins_from_previous_hash,
)


class SemStampBlockTests(unittest.TestCase):
    def test_lsh_hyperplanes_and_hashes_are_deterministic(self):
        planes_a = build_lsh_hyperplanes(3, 4, seed=1234)
        planes_b = build_lsh_hyperplanes(3, 4, seed=1234)
        planes_c = build_lsh_hyperplanes(3, 4, seed=1235)
        self.assertTrue(torch.equal(planes_a, planes_b))
        self.assertFalse(torch.equal(planes_a, planes_c))
        self.assertTrue(torch.allclose(planes_a.norm(dim=-1), torch.ones(3)))

        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        hyperplanes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        hashes, margins, projections = embedding_hashes_and_margins(
            embeddings,
            hyperplanes,
        )
        self.assertEqual(hashes.tolist(), [2, 0])
        self.assertEqual(margins.tolist(), [0.0, 0.0])
        self.assertEqual(projections.tolist(), [[1.0, 0.0], [-1.0, 0.0]])

    def test_valid_bins_are_keyed_and_have_semstamp_size(self):
        bins_a = valid_bins_from_previous_hash(
            3, lsh_dim=3, accept_rate=0.25, rng_device="cpu"
        )
        bins_b = valid_bins_from_previous_hash(
            3, lsh_dim=3, accept_rate=0.25, rng_device="cpu"
        )
        bins_c = valid_bins_from_previous_hash(
            4, lsh_dim=3, accept_rate=0.25, rng_device="cpu"
        )
        self.assertEqual(bins_a, bins_b)
        self.assertEqual(len(bins_a), 2)
        self.assertEqual(len(set(bins_a)), 2)
        self.assertTrue(all(0 <= value < 8 for value in bins_a))
        self.assertNotEqual(bins_a, bins_c)

    def test_first_accepted_candidate_uses_sampling_order_not_margin_maximum(self):
        selected = first_accepted_candidate(
            candidate_hashes=[1, 6, 6, 2],
            candidate_margins=[0.9, 0.03, 0.8, 0.7],
            candidate_valid=[True, True, True, True],
            valid_bins=[6, 7],
            margin=0.02,
        )
        self.assertEqual(selected, 1)  # candidate 2 has a larger margin

    def test_first_accepted_candidate_applies_margin_and_text_validity(self):
        selected = first_accepted_candidate(
            candidate_hashes=[6, 6, 7],
            candidate_margins=[0.5, 0.01, 0.03],
            candidate_valid=[False, True, True],
            valid_bins=[6, 7],
            margin=0.02,
        )
        self.assertEqual(selected, 2)
        self.assertIsNone(
            first_accepted_candidate(
                [6, 7], [0.01, 0.019], [True, True], [6, 7], margin=0.02
            )
        )

    def test_valid_bins_reject_invalid_accept_rate(self):
        for accept_rate in (0.0, 1.0, -0.2, 1.2):
            with self.subTest(accept_rate=accept_rate):
                with self.assertRaises(ValueError):
                    valid_bins_from_previous_hash(
                        0,
                        lsh_dim=3,
                        accept_rate=accept_rate,
                        rng_device="cpu",
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA parity test")
    def test_cuda_valid_bins_match_official_semstamp_randperm(self):
        key = 15_485_863
        for previous_hash in range(8):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(key * previous_hash)
            expected = tuple(
                torch.randperm(8, device="cuda", generator=generator)[:2]
                .cpu()
                .tolist()
            )
            actual = valid_bins_from_previous_hash(
                previous_hash,
                lsh_dim=3,
                accept_rate=0.25,
                hash_key=key,
                rng_device="cuda",
            )
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
