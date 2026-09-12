"""Backbone token alignment, EOS handling, and random position budgets."""

from __future__ import annotations

import types
from types import SimpleNamespace

import torch

from denmark.baselines.common import llada_generate_unwatermarked
from denmark.core.model import (
    DreamSemanticUnitDecodeController,
    NativeSemanticWatermarkHook,
    get_model_logits,
    resolve_mask_id,
    safe_decode,
)
from denmark.core.scoring import block_decode


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


class RandomBaselineModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.input_lengths = []

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        use_cache=None,
    ):
        self.input_lengths.append(input_ids.shape[1])
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 3), dtype=torch.float32)
        logits[..., 2] = 1.0
        return SimpleNamespace(logits=logits)


class RandomBaselineTokenizer:
    eos_token_id = None

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [str(token_id) for token_id in ids]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in ids)


def generate(model, generator_family):
    return llada_generate_unwatermarked(
        torch.tensor([[0]]),
        model,
        RandomBaselineTokenizer(),
        mask_id=1,
        gen_length=2,
        block_size=1,
        steps=2,
        temperature=0.0,
        remasking="random",
        generator_family=generator_family,
    )


def test_llada2_random_baseline_forwards_only_through_active_block():
    model = RandomBaselineModel()

    generate(model, "llada2")

    assert model.input_lengths == [2, 3]


def test_legacy_llada_random_baseline_keeps_full_canvas():
    model = RandomBaselineModel()

    generate(model, "llada")

    assert model.input_lengths == [3, 3]


def test_random_global_grouped_selects_exact_global_budget_and_records_groups():
    torch.manual_seed(7)
    prompt_len = 3
    controller = DreamSemanticUnitDecodeController(
        prompt_len=prompt_len,
        gen_length=50,
        block_size=25,
        cand_block_size=8,
        mask_id=99,
        steps=7,
        eps=1e-3,
        mode="random_global_grouped",
    )
    x = torch.full((1, prompt_len + 50), 99, dtype=torch.long)
    x[:, :prompt_len] = 1
    logits = torch.zeros((1, prompt_len + 50, 4))

    assert controller.logits_hook(0, x, logits) is logits
    picked = controller.allowed_mask[0].nonzero(as_tuple=False).flatten()
    assert picked.numel() == 8
    assert bool(((picked >= prompt_len) & (picked < prompt_len + 50)).all())
    diag = controller.diag[-1]
    assert diag["selected_position_count"] == 8
    assert sum(len(group) for group in diag["selected_positions_by_block"].values()) == 8
    assert diag["selected_group_count"] == len(diag["selected_positions_by_block"])


def test_random_global_grouped_uses_remaining_positions_on_tail_step():
    controller = DreamSemanticUnitDecodeController(
        prompt_len=1,
        gen_length=25,
        block_size=25,
        cand_block_size=8,
        mask_id=99,
        steps=4,
        eps=1e-3,
        mode="random_global_grouped",
    )
    x = torch.arange(26).unsqueeze(0)
    x[0, [4, 8, 21]] = 99
    logits = torch.zeros((1, 26, 4))

    controller.logits_hook(3, x, logits)
    picked = controller.allowed_mask[0].nonzero(as_tuple=False).flatten().tolist()
    assert sorted(picked) == [4, 8, 21]


def test_hook_scores_all_semantic_groups_from_same_precommit_canvas():
    mask_id = 99
    hook = NativeSemanticWatermarkHook.__new__(NativeSemanticWatermarkHook)
    hook.prompt_len = 0
    hook.gen_length = 50
    hook.block_size = 25
    hook.cand_block_size = 8
    hook.num_candidates = 2
    hook.candidate_temperature = 0.5
    hook.candidate_position_mode = "same_positions"
    hook.resample_base_candidate = False
    hook.mask_id = mask_id
    hook.prev_x = torch.full((1, 50), mask_id, dtype=torch.long)
    hook.forced_candidate_mask = torch.zeros((1, 50), dtype=torch.bool)
    hook.forced_candidate_mask[0, [1, 2, 28, 29, 30]] = True
    hook.diag = []

    seen_canvases = []

    def fake_score(
        self,
        candidate_tokens,
        x,
        positions,
        block_id,
        logits,
        step,
        remaining_before_commit,
    ):
        seen_canvases.append(x.clone())
        return 1, {"block_id": block_id, "num_positions": int(positions.numel())}

    hook._score_candidates = types.MethodType(fake_score, hook)
    current = torch.full((1, 50), mask_id, dtype=torch.long)
    # Simulate native DREAM transfers; grouped mode must discard them first.
    current[0, [0, 25]] = 3
    logits = torch.zeros((1, 50, 8))
    logits[..., 4] = 5.0

    output = hook(0, current, logits)

    assert len(seen_canvases) == 2
    assert torch.equal(seen_canvases[0], seen_canvases[1])
    assert bool((seen_canvases[0][0, [1, 2, 28, 29, 30]] == mask_id).all())
    assert bool((output[0, [1, 2, 28, 29, 30]] != mask_id).all())
    assert len(hook.diag) == 2
    assert {row["semantic_group_count"] for row in hook.diag} == {2}
    assert {row["global_step_position_count"] for row in hook.diag} == {5}
