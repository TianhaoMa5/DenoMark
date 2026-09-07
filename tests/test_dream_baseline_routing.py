from types import SimpleNamespace

import pytest
import torch

from denomark.baselines.dlm_kgw.model import HashDistribution, generate_hash_distribution
from denomark.baselines.dgmark.generate import resolve_decoding


class TinyDream(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids, *args, **kwargs):
        logits = torch.zeros((*ids.shape, 8))
        logits[..., 3] = 10
        return SimpleNamespace(logits=logits)


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def get_vocab(self):
        return {str(i): i for i in range(8)}

    def convert_ids_to_tokens(self, ids, **kwargs):
        return [str(i) for i in ids]

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


class Watermark:
    def __init__(self, target):
        self.target = target
        self.calls = 0

    def set_temperature(self, value):
        self.temperature = value

    def set_mask_token(self, value):
        self.mask = value

    def watermark_logits(self, ids, logits):
        self.calls += 1
        output = logits.clone()
        output[..., self.target] += 100
        return output, output

    def detect(self, ids):
        self.detected = ids.tolist()
        return {"z_score": 1.0}


@pytest.mark.parametrize("target,expected", [(4, [4] * 5), (2, [])])
def test_hash_dream_origin_hook_and_eos(target, expected):
    watermark = Watermark(target)
    text, ids, score = generate_hash_distribution(
        torch.tensor([[1]]), TinyDream(), Tokenizer(), 7, watermark,
        generator_family="dream", gen_length=5, block_size=25,
        steps=3, temperature=0, remasking="random",
    )
    assert ids == expected == watermark.detected
    assert text == " ".join(map(str, expected))
    assert watermark.calls > 0
    assert watermark.mask == 7
    assert score["z_score"] == 1


@pytest.mark.parametrize("family,expected,width", [
    ("llada", "aligned", 10), ("llada2", "aligned", 10),
    ("dream", "dream_dgmark", 32),
])
def test_dgmark_paper_defaults(family, expected, width):
    args = SimpleNamespace(generator_family=family, dgmark_decoding="auto",
                           dgmark_top_k=None, dgmark_beam_size=None)
    assert resolve_decoding(args) == expected
    assert args.dgmark_top_k == args.dgmark_beam_size == width


def test_dgmark_explicit_overrides():
    args = SimpleNamespace(generator_family="dream", dgmark_decoding="aligned",
                           dgmark_top_k=12, dgmark_beam_size=12)
    assert resolve_decoding(args) == "dream_dgmark"
    assert args.dgmark_top_k == args.dgmark_beam_size == 12


def test_origin_with_real_hash_distribution():
    tokenizer = Tokenizer()
    watermark = HashDistribution(tokenizer, delta=4, topk=4, device="cpu")
    _, ids, score = generate_hash_distribution(
        torch.tensor([[1]]), TinyDream(), tokenizer, 7, watermark,
        generator_family="dream", gen_length=5, steps=3,
        temperature=0.5, remasking="random",
    )
    assert len(ids) <= 5
    assert torch.isfinite(torch.tensor(score["z_score"]))
