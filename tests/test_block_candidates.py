"""Block Best-of-K, PMark and SemStamp candidate selection."""

import random
import unittest
from types import SimpleNamespace

import pytest
import torch

import denmark.baselines.block_best_of_k.model as block_best_of_k
from denmark.baselines.block_best_of_k.model import (
    llada_generate_block_best_of_k,
    select_argmax_candidate,
    semantic_candidate_scores,
)
from denmark.baselines.pmark.model import (
    build_pmark_pivots,
    build_pmark_secret_bits,
    pmark_soft_matches,
    select_pmark_candidate,
)
from denmark.baselines.semstamp.model import (
    build_lsh_hyperplanes,
    embedding_hashes_and_margins,
    first_accepted_candidate,
    valid_bins_from_previous_hash,
)
from denmark.experiments.block_candidates import audit_dataset


class _FakeLLaDA:
    def __init__(self, vocab_size=12, preferred_token=3):
        self.vocab_size = vocab_size
        self.preferred_token = preferred_token
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1
        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )
        logits[..., self.preferred_token] = 10.0
        return SimpleNamespace(logits=logits)


class _FakeTokenizer:
    all_special_ids = []
    bos_token_id = None
    eos_token_id = None
    pad_token_id = None
    unk_token_id = None
    mask_token_id = 9

    def convert_ids_to_tokens(self, token_ids, skip_special_tokens=False):
        return [str(token_id) for token_id in token_ids]

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


def test_semantic_candidate_scores_matches_detector_statistic():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    signs = torch.tensor([1.0, -1.0])

    scalar, per_channel = semantic_candidate_scores(embeddings, directions, signs)

    assert torch.allclose(
        per_channel,
        torch.tensor([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]),
    )
    assert torch.allclose(scalar, torch.tensor([0.5, -0.5, -0.5]))
    assert select_argmax_candidate(scalar) == 0


def test_select_argmax_candidate_uses_first_index_on_tie():
    assert select_argmax_candidate(torch.tensor([0.1, 0.7, 0.7])) == 1
    with pytest.raises(ValueError):
        select_argmax_candidate(torch.empty(0))


def test_block_best_of_k_decodes_full_blocks_and_commits_argmax(monkeypatch):
    def fake_score(candidate_texts, _enc, _enc_tok, dirs, _signs, _device):
        n = len(candidate_texts)
        scores = torch.arange(n, dtype=torch.float32)
        channels = scores.unsqueeze(1).expand(n, dirs.shape[0]).clone()
        return scores, channels, [True] * n

    monkeypatch.setattr(block_best_of_k, "_score_candidate_texts", fake_score)
    model = _FakeLLaDA()
    tokenizer = _FakeTokenizer()
    prompt = torch.tensor([[7]], dtype=torch.long)
    directions = torch.zeros(2, 1, 2)
    signs = torch.ones(2, 1)

    text, token_ids, diagnostics = llada_generate_block_best_of_k(
        prompt,
        model,
        encoder=object(),
        encoder_tokenizer=object(),
        llada_tokenizer=tokenizer,
        directions=directions,
        signs=signs,
        mask_id=9,
        gen_length=4,
        block_size=2,
        steps=4,
        temperature=0.0,
        num_candidates=3,
        candidate_batch_size=2,
        device="cpu",
        record_candidate_texts=True,
    )

    assert token_ids == [3, 3, 3, 3]
    assert text == "3 3 3 3"
    assert model.calls == 8  # 2 blocks x 2 steps x ceil(3 / 2) microbatches
    assert len(diagnostics) == 2
    assert all(diag["selected_candidate_index"] == 2 for diag in diagnostics)
    assert all(diag["candidate_scores"] == [0.0, 1.0, 2.0] for diag in diagnostics)
    assert all(
        diag["selection_reason"] == "argmax_semantic_watermark_score" for diag in diagnostics
    )
    assert all(len(diag["candidate_texts"]) == 3 for diag in diagnostics)


def test_block_best_of_k_commits_selected_tokens_into_next_prefix(monkeypatch):
    seen_canvases = []

    def fake_sampler(base_canvas, _model, _mask, block_start, block_end, *_args):
        seen_canvases.append(base_canvas.detach().cpu().clone())
        block_id = len(seen_canvases) - 1
        first_token = 3 + 3 * block_id
        return torch.tensor(
            [
                [first_token, first_token],
                [first_token + 1, first_token + 1],
                [first_token + 2, first_token + 2],
            ],
            dtype=torch.long,
        )[:, : block_end - block_start]

    def score_by_first_token(candidate_texts, _enc, _enc_tok, dirs, _signs, _device):
        scores = torch.tensor([float(text.split()[0]) for text in candidate_texts])
        return scores, scores.unsqueeze(1).expand(-1, dirs.shape[0]), [True] * len(scores)

    monkeypatch.setattr(block_best_of_k, "_sample_candidate_microbatch", fake_sampler)
    monkeypatch.setattr(block_best_of_k, "_score_candidate_texts", score_by_first_token)
    tokenizer = _FakeTokenizer()
    directions = torch.zeros(2, 1, 2)
    signs = torch.ones(2, 1)

    _text, token_ids, diagnostics = llada_generate_block_best_of_k(
        torch.tensor([[11]], dtype=torch.long),
        object(),
        object(),
        object(),
        tokenizer,
        directions,
        signs,
        9,
        gen_length=4,
        block_size=2,
        steps=4,
        temperature=0.6,
        num_candidates=3,
        candidate_batch_size=3,
        device="cpu",
    )

    assert seen_canvases[0][0, 1:3].tolist() == [9, 9]
    assert seen_canvases[1][0, 1:3].tolist() == [5, 5]
    assert token_ids == [5, 5, 8, 8]
    assert [diag["selected_token_ids"] for diag in diagnostics] == [[5, 5], [8, 8]]


def test_block_best_of_k_rejects_incompatible_shapes():
    with pytest.raises(ValueError, match="prompt"):
        llada_generate_block_best_of_k(
            torch.tensor([1, 2]),
            object(),
            object(),
            object(),
            _FakeTokenizer(),
            torch.zeros(1, 1, 2),
            torch.ones(1, 1),
            9,
            gen_length=2,
            block_size=2,
            device="cpu",
        )


def test_audit_checks_candidate_argmax_and_quality():
    diagnostic = {
        "selection_rule": "argmax_semantic_watermark_score",
        "empty_block_policy": "first_candidate_when_all_semantically_empty",
        "num_candidates": 3,
        "candidate_scores": [-0.1, 0.3, 0.2],
        "candidate_valid": [True, True, True],
        "selected_candidate_index": 1,
        "selected_score": 0.3,
        "selection_reason": "argmax_semantic_watermark_score",
        "selected_token_ids": [3, 3],
    }
    row = {
        "ds_key": "toy",
        "source_idx": 17,
        "selected_position": 0,
        "shard": 0,
        "num_shards": 1,
        "_audit_source_file": "/tmp/generations_shard0.jsonl",
        "generated_token_ids": [3, 3, 3, 3],
        "retokenized_token_ids": [3, 3, 3, 3],
        "block_diagnostics": [
            {**diagnostic, "block_id": 0, "block_start": 0, "block_end": 2},
            {**diagnostic, "block_id": 1, "block_start": 2, "block_end": 4},
        ],
        "text": "a b c d",
        "token_len": 4,
        "rep4": 0.0,
        "passed_quality": True,
    }

    summary = audit_dataset(
        "toy",
        [row],
        expected=1,
        gen_length=4,
        block_size=2,
        num_candidates=3,
        mask_id=9,
        min_response_tokens=4,
        max_rep4=0.2,
        expected_num_shards=1,
    )
    assert summary["valid"] is True

    row["block_diagnostics"][0]["selected_candidate_index"] = 2
    bad_summary = audit_dataset(
        "toy",
        [row],
        expected=1,
        gen_length=4,
        block_size=2,
        num_candidates=3,
        mask_id=9,
        min_response_tokens=4,
        max_rep4=0.2,
        expected_num_shards=1,
    )
    assert bad_summary["valid"] is False
    assert any("argmax=1" in error for error in bad_summary["structural_errors"])


def test_audit_accepts_all_empty_candidate_fallback():
    empty_diagnostic = {
        "selection_rule": "argmax_semantic_watermark_score",
        "empty_block_policy": "first_candidate_when_all_semantically_empty",
        "num_candidates": 3,
        "block_id": 0,
        "block_start": 0,
        "block_end": 2,
        "candidate_scores": [None, None, None],
        "candidate_valid": [False, False, False],
        "selected_candidate_index": 0,
        "selected_score": None,
        "selection_reason": "all_candidates_empty_fallback_first",
        "selected_token_ids": [2, 2],
    }
    row = {
        "ds_key": "toy",
        "source_idx": 4,
        "selected_position": 0,
        "shard": 0,
        "num_shards": 1,
        "generated_token_ids": [3, 3, 2, 2],
        "retokenized_token_ids": [3, 3, 2, 2],
        "block_diagnostics": [
            {
                **empty_diagnostic,
                "candidate_scores": [0.1, 0.3, 0.2],
                "candidate_valid": [True, True, True],
                "selected_candidate_index": 1,
                "selected_score": 0.3,
                "selection_reason": "argmax_semantic_watermark_score",
                "selected_token_ids": [3, 3],
            },
            {**empty_diagnostic, "block_id": 1, "block_start": 2, "block_end": 4},
        ],
        "text": "a b c d",
        "token_len": 4,
        "rep4": 0.0,
        "passed_quality": True,
    }

    summary = audit_dataset(
        "toy",
        [row],
        expected=1,
        gen_length=4,
        block_size=2,
        num_candidates=3,
        mask_id=9,
        min_response_tokens=4,
        max_rep4=0.2,
    )
    assert summary["valid"] is True


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
        bins_a = valid_bins_from_previous_hash(3, lsh_dim=3, accept_rate=0.25, rng_device="cpu")
        bins_b = valid_bins_from_previous_hash(3, lsh_dim=3, accept_rate=0.25, rng_device="cpu")
        bins_c = valid_bins_from_previous_hash(4, lsh_dim=3, accept_rate=0.25, rng_device="cpu")
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
            first_accepted_candidate([6, 7], [0.01, 0.019], [True, True], [6, 7], margin=0.02)
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
                torch.randperm(8, device="cuda", generator=generator)[:2].cpu().tolist()
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
