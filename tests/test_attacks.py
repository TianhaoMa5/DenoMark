"""Parrot selection and sentence-level GPT prompt construction."""

from types import SimpleNamespace

import pytest

from denmark.attacks.common import sentence_attack_spec, split_sentences
from denmark.attacks.parrot import (
    build_attacker,
    build_bigrams,
    compare_ngram_overlap,
    select_by_bigram_overlap,
)


class DummyTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
            ids.append(self.vocab[token])
        return {"input_ids": ids}


class DictScorer:
    def __init__(self, scores):
        self.scores = scores

    def score(self, original, candidates):
        return [self.scores[candidate] for candidate in candidates]


def test_build_bigrams():
    assert build_bigrams([1, 2, 3, 4]) == [(1, 2), (2, 3), (3, 4)]
    assert build_bigrams([1]) == []


def test_compare_ngram_overlap_counts_candidate_multiplicity():
    original = [(1, 2), (2, 3)]
    candidate = [(1, 2), (1, 2), (9, 9)]
    assert compare_ngram_overlap(original, candidate) == 2


def test_selects_lowest_overlap_after_semantic_filter():
    tokenizer = DummyTokenizer()
    original = "a b c d"
    candidates = [
        "a b c d",
        "w x y z",
        "d c b a",
    ]
    scorer = DictScorer(
        {
            "a b c d": 0.99,
            "w x y z": 0.50,
            "d c b a": 0.98,
        }
    )

    result = select_by_bigram_overlap(
        original,
        candidates,
        tokenizer,
        scorer,
        bert_threshold=0.03,
    )

    assert result.selected == "d c b a"
    assert result.selected_index == 2
    assert result.candidates[1].accepted is False
    assert result.candidates[2].accepted is True


def test_length_filter_rejects_too_long_low_overlap_candidate():
    tokenizer = DummyTokenizer()
    original = "a b c d"
    candidates = [
        "a b c d",
        "w x y z q r s t",
    ]
    scorer = DictScorer({"a b c d": 0.99, "w x y z q r s t": 0.99})

    result = select_by_bigram_overlap(
        original,
        candidates,
        tokenizer,
        scorer,
        bert_threshold=0.03,
        max_length_ratio=1.5,
    )

    assert result.selected == "a b c d"
    assert result.candidates[1].accepted is False


@pytest.mark.parametrize("count", [1, 4, 7, 10])
def test_parrot_configuration(count):
    class FakeParrot:
        def augment(self, *args, **kwargs):
            return ["candidate"] * kwargs["max_return_phrases"]

    au = SimpleNamespace(
        SParrot=FakeParrot,
        ParrotParaphraseConfig=SimpleNamespace,
        ParrotParaphrase=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    attacker = build_attacker(au, count)
    assert attacker.cfg.num_beams == count
    assert attacker.cfg.use_bigram_filter == (count > 1)
    if count > 1:
        assert len(attacker.parrot.augment(max_return_phrases=count)) == count
        assert attacker.candidate_counts == [count]
    attacker.reset_trace()
    assert attacker.candidate_counts == []


def test_sentence_attack_split_and_compression_target_are_stable():
    text = "Dr. Smith measured 10 samples. The result was stable!"
    sentences = split_sentences(text)
    assert sentences == ["Dr. Smith measured 10 samples.", "The result was stable!"]
    prompt, temperature, bounds = sentence_attack_spec("compress_60_70", sentences[0])
    assert temperature == 0.7
    assert bounds == (3, 3)
    assert "about 3 to 3 words" in prompt
