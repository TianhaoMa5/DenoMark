"""Prompt preparation, quality filtering, deduplication, and metric schemas."""

import unittest

from denmark.data.prepare_waterbench import normalize_prompt
from denmark.data.protocol import (
    canonical_prompt_key,
    extract_user_prompt,
    grouped_crossfit_folds,
    quality_and_dedup_indices,
    resolve_prompt_seed,
)
from denmark.evaluation.collect_metrics import normalize_metrics
from denmark.evaluation.summarize_runtime import token_count


class EvalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.source_rows = [
            {"input": "Question zero?", "raw_prompt": "Please answer. Question zero?"},
            {"input": "Question one?", "raw_prompt": "Please answer. Question one?"},
        ]

    def test_prompt_full_takes_precedence_over_reindexed_source_idx(self):
        row = {
            "source_idx": 0,
            "prompt_full": (
                "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
                "Please answer. Question one?<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        }
        prompt, source = resolve_prompt_seed(row, source_rows=self.source_rows, row_index=7)
        self.assertEqual(prompt, "Please answer. Question one?")
        self.assertEqual(source, "prompt_full_user_span")
        self.assertEqual(
            canonical_prompt_key(prompt, self.source_rows),
            "waterbench-input:Question one?",
        )

    def test_unparseable_serialized_prompt_fails_closed(self):
        prompt_full = "broken<|start_header_id|>assistant<|end_header_id|>completion"
        self.assertEqual(extract_user_prompt(prompt_full), "")
        with self.assertRaises(ValueError):
            resolve_prompt_seed({"prompt_full": prompt_full}, row_index=3)

    def test_truncated_user_turn_without_assistant_is_recoverable(self):
        prompt_full = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nA retained prompt truncated inside the user turn"
        )
        self.assertEqual(
            extract_user_prompt(prompt_full),
            "A retained prompt truncated inside the user turn",
        )

    def test_quality_filter_and_prompt_dedup(self):
        items = [
            {"token_len": 160, "rep4": 0.1, "prompt_key": "a"},
            {"token_len": 170, "rep4": 0.1, "prompt_key": "a"},
            {"token_len": 149, "rep4": 0.1, "prompt_key": "b"},
            {"token_len": 180, "rep4": 0.21, "prompt_key": "c"},
            {"token_len": 200, "rep4": 0.2, "prompt_key": "d"},
        ]
        indices, audit = quality_and_dedup_indices(
            items,
            min_token_len=150,
            max_rep4=0.2,
            deduplicate_prompt=True,
        )
        self.assertEqual(indices, [0, 4])
        self.assertEqual(audit["n_quality"], 3)
        self.assertEqual(audit["n_removed_duplicate_prompt"], 1)

    def test_grouped_crossfit_is_balanced_and_has_no_group_leakage(self):
        keys = ["a", "a", "b", "c", "d", "e", "f", "g"]
        folds = grouped_crossfit_folds(keys, n_folds=3, seed=42)
        self.assertEqual(folds[0], folds[1])
        sizes = [folds.count(index) for index in range(3)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(folds, grouped_crossfit_folds(keys, n_folds=3, seed=42))


if __name__ == "__main__":
    unittest.main()


def test_prompt_builder_removes_answers_and_applies_templates():
    source = {
        "question": "Why does diversification reduce risk?",
        "answer": "This must not be copied.",
        "source_id": "private-row-label",
    }
    row = normalize_prompt(source, "finance_qa")
    assert set(row) == {"prompt", "raw_prompt", "input", "context"}
    assert row["input"] == source["question"]
    assert row["context"] == ""
    assert source["answer"] not in row.values()
    assert source["source_id"] not in row.values()
    assert "financial knowledge within 300 words" in row["prompt"]


def test_runtime_token_count_accepts_all_runner_schemas():
    assert token_count({"token_len": 151}) == 151
    assert token_count({"dgmark_token_len": 152}) == 152
    assert token_count({"hash_distribution_token_len": 153}) == 153
    assert token_count({"generated_token_ids": [1, 2, 3]}) == 3


def test_metric_collector_normalizes_flat_and_nested_tpr_schemas():
    flat = normalize_metrics(
        {
            "n_pos": 300,
            "n_neg": 10000,
            "roc_tpr_at_0_5pct": 0.5,
            "roc_tpr_at_1pct": 0.6,
            "roc_tpr_at_5pct": 0.8,
            "auc": 0.9,
        }
    )
    nested = normalize_metrics(
        {
            "n_positive": 300,
            "n_negative": 10000,
            "tpr": {"0.005": 0.5, "0.01": 0.6, "0.05": 0.8},
            "auc": 0.9,
        }
    )
    assert flat == nested
