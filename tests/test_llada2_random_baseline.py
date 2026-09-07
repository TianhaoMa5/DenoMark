from types import SimpleNamespace

import torch

from denomark.baselines.common import llada_generate_unwatermarked


class DummyModel:
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


class DummyTokenizer:
    eos_token_id = None

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return [str(token_id) for token_id in ids]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in ids)


def generate(model, generator_family):
    return llada_generate_unwatermarked(
        torch.tensor([[0]]),
        model,
        DummyTokenizer(),
        mask_id=1,
        gen_length=2,
        block_size=1,
        steps=2,
        temperature=0.0,
        remasking="random",
        generator_family=generator_family,
    )


def test_llada2_random_baseline_forwards_only_through_active_block():
    model = DummyModel()

    generate(model, "llada2")

    assert model.input_lengths == [2, 3]


def test_legacy_llada_random_baseline_keeps_full_canvas():
    model = DummyModel()

    generate(model, "llada")

    assert model.input_lengths == [3, 3]
