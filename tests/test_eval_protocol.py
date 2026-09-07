import unittest

from denomark.data.protocol import (
    canonical_prompt_key,
    extract_user_prompt,
    grouped_crossfit_folds,
    quality_and_dedup_indices,
    resolve_prompt_seed,
)


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
        prompt, source = resolve_prompt_seed(
            row, source_rows=self.source_rows, row_index=7
        )
        self.assertEqual(prompt, "Please answer. Question one?")
        self.assertEqual(source, "prompt_full_user_span")
        self.assertEqual(
            canonical_prompt_key(prompt, self.source_rows),
            "waterbench-input:Question one?",
        )

    def test_unparseable_serialized_prompt_fails_closed(self):
        prompt_full = (
            "broken<|start_header_id|>assistant<|end_header_id|>completion"
        )
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
