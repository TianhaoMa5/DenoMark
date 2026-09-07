from types import SimpleNamespace

import torch

from denomark.baselines.common import llada_generate_unwatermarked
from denomark.core.scoring import block_decode
from denomark.core.model import get_model_logits, resolve_mask_id, safe_decode


class DummyModel:
    def __call__(self, input_ids):
        batch, length = input_ids.shape
        logits = torch.arange(batch * length * 3, dtype=torch.float32).view(batch, length, 3)
        return SimpleNamespace(logits=logits)


class DummyLLaDA2Model:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
                "use_cache": use_cache,
            }
        )
        batch, length = input_ids.shape
        logits = torch.arange(batch * length * 3, dtype=torch.float32).view(batch, length, 3)
        return SimpleNamespace(logits=logits)


class DummyTokenizer:
    mask_token_id = 4
    all_special_ids = [4]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [str(token_id) for token_id in ids]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in ids)


def test_llada_logits_slice_is_position_aligned():
    model = DummyModel()
    input_ids = torch.tensor([[10, 11, 12, 13]])
    logits = get_model_logits(model, input_ids, "llada", logit_start=1, logit_end=3)
    torch.testing.assert_close(logits, model(input_ids).logits[:, 1:3])


def test_dream_logits_are_shifted_before_slicing():
    model = DummyModel()
    input_ids = torch.tensor([[10, 11, 12, 13]])
    raw = model(input_ids).logits
    shifted = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
    logits = get_model_logits(model, input_ids, "dream", logit_start=1, logit_end=3)
    torch.testing.assert_close(logits, shifted[:, 1:3])


def test_llada2_logits_use_active_block_as_attention_boundary():
    model = DummyLLaDA2Model()
    input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    logits = get_model_logits(model, input_ids, "llada2", logit_start=3, logit_end=6)
    assert logits.shape == (1, 3, 3)
    call = model.calls[0]
    assert torch.isneginf(call["attention_mask"][0, 0, :3, 3:]).all()
    assert (call["attention_mask"][0, 0, 3:, :] == 0).all()


def test_llada2_unwatermarked_generation_uses_each_block_boundary():
    model = DummyLLaDA2Model()
    model.device = torch.device("cpu")
    llada_generate_unwatermarked(
        torch.tensor([[0]]),
        model,
        DummyTokenizer(),
        mask_id=1,
        gen_length=2,
        block_size=1,
        steps=2,
        temperature=0.0,
        remasking="random",
        generator_family="llada2",
    )
    assert len(model.calls) == 2
    assert model.calls[0]["input_ids"].shape[1] == 2
    assert model.calls[1]["input_ids"].shape[1] == 3


def test_llada2_generation_truncates_at_first_eos():
    model = DummyLLaDA2Model()
    model.device = torch.device("cpu")
    tokenizer = DummyTokenizer()
    tokenizer.eos_token_id = 2
    _, tokens = llada_generate_unwatermarked(
        torch.tensor([[0]]),
        model,
        tokenizer,
        mask_id=1,
        gen_length=3,
        block_size=1,
        steps=3,
        temperature=0.0,
        remasking="random",
        generator_family="llada2",
    )
    assert tokens == [2]


def test_special_filter_does_not_assume_zero_one_two_are_special():
    assert block_decode([0, 1, 2, 4], 0, 4, DummyTokenizer()) == "0 1 2"


def test_resolve_mask_id_prefers_explicit_then_tokenizer():
    assert resolve_mask_id(DummyTokenizer(), "llada") == 4
    assert resolve_mask_id(DummyTokenizer(), "llada", mask_id=123) == 123


class MissingTokenTokenizer:
    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [None if token_id == 99 else str(token_id) for token_id in ids]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in ids)


def test_safe_decode_filters_ids_without_string_token():
    assert safe_decode(MissingTokenTokenizer(), [1, 99, 2]) == "1 2"
