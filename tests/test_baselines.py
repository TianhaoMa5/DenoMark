"""DLM-KGW, DGMark and UMR scoring and Dream decoding contracts."""

import math
from types import SimpleNamespace

import pytest
import torch

from denmark.baselines.dgmark.generate import resolve_decoding
from denmark.baselines.dlm_kgw.model import HashDistribution, generate_hash_distribution
from denmark.baselines.umr.generate import DreamNativeRegretController, DreamUMRApplyCompat
from denmark.baselines.umr.model import calculate_z_score, context_seed, score_umr_tokens


class FakeBitmap:
    def __init__(self, values):
        self.values = values

    def get_bit(self, x, y):
        return self.values.get((x, y), False)


def test_calculate_z_score_matches_umr_bernoulli_null():
    assert calculate_z_score(60, 100) == pytest.approx(2.0)
    assert calculate_z_score(50, 100) == pytest.approx(0.0)
    assert calculate_z_score(0, 0) == 0.0
    expected = (7 - 10 * 0.25) / math.sqrt(10 * 0.25 * 0.75)
    assert calculate_z_score(7, 10, gamma=0.25) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("hits", "total", "gamma"),
    [(-1, 10, 0.5), (11, 10, 0.5), (1, -1, 0.5), (1, 10, 0.0), (1, 10, 1.0)],
)
def test_calculate_z_score_rejects_invalid_inputs(hits, total, gamma):
    with pytest.raises(ValueError):
        calculate_z_score(hits, total, gamma)


def test_context_seed_matches_upstream_hash_construction():
    assert context_seed(123, 42) == context_seed(123, 42)
    assert context_seed(123, 42) != context_seed(124, 42)


def test_score_umr_tokens_counts_bitmap_hits_and_recovers_votes():
    previous = 7
    ids = [8, 9, 10]
    targets = []
    prev = previous
    for current in ids:
        targets.append((prev, current, context_seed(prev, 42) % 4))
        prev = current
    values = {}
    for prev, current, bit_index in targets:
        target = "1001"[bit_index]
        values[(prev, current)] = target == "1"

    result = score_umr_tokens(ids, FakeBitmap(values), previous_token=previous)

    assert result["total_gen_tokens"] == 3
    assert result["token_hit_count"] == sum(values.values())
    assert result["z_score"] == pytest.approx(
        calculate_z_score(result["token_hit_count"], result["total_gen_tokens"])
    )
    assert len(result["votes"]) == 4
    assert result["recovered_str"] == "".join(
        "?" if sum(votes) == 0 else ("1" if votes[1] > votes[0] else "0")
        for votes in result["votes"]
    )


def test_score_without_previous_token_uses_first_id_as_context():
    result = score_umr_tokens([1, 2], FakeBitmap({(1, 2): True}))
    assert result["total_gen_tokens"] == 1
    assert result["token_hit_count"] == 1


class UnusedUpstream:
    pass


def test_clean_dream_native_controller_never_changes_tokens_or_eos():
    controller = DreamNativeRegretController(
        UnusedUpstream(),
        watermark=None,
        apply_watermark=None,
        mask_id=9,
        prompt_len=2,
        gen_length=3,
        steps=3,
    )
    initial = torch.tensor([[1, 2, 9, 9, 9]])
    assert torch.equal(controller(None, initial.clone(), None), initial)

    # Treat token 4 as an EOS stand-in. The clean/native controller must leave
    # it untouched; minimum length is enforced only after generation.
    transferred = torch.tensor([[1, 2, 4, 9, 9]])
    assert torch.equal(
        controller(0, transferred.clone(), torch.zeros((1, 5, 10))),
        transferred,
    )


def test_dream_umr_preserves_native_eos_and_cannot_introduce_it():
    calls = []

    def fake_apply(*, x, x0, logits, batch_index, index, watermark, mask_id):
        calls.append(True)
        x0[batch_index, index] = logits[batch_index, index].argmax()

    compat = DreamUMRApplyCompat(
        fake_apply,
        convert_gumbel_scores=False,
        preserve_token_ids={4},
        gumbel_temperature=0.0,
    )
    x = torch.tensor([[1, 9]])
    logits = torch.tensor([[[0.0, 0.0, 0.0, 2.0, 10.0], [0.0] * 5]])

    native_eos = torch.tensor([[4, 9]])
    compat(x, native_eos, logits, 0, 0, object(), 9)
    assert native_eos[0, 0].item() == 4
    assert not calls

    native_regular = torch.tensor([[3, 9]])
    compat(x, native_regular, logits, 0, 0, object(), 9)
    assert native_regular[0, 0].item() == 3
    assert calls
    assert logits[0, 0, 4].item() == 10.0


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
        torch.tensor([[1]]),
        TinyDream(),
        Tokenizer(),
        7,
        watermark,
        generator_family="dream",
        gen_length=5,
        block_size=25,
        steps=3,
        temperature=0,
        remasking="random",
    )
    assert ids == expected == watermark.detected
    assert text == " ".join(map(str, expected))
    assert watermark.calls > 0
    assert watermark.mask == 7
    assert score["z_score"] == 1


@pytest.mark.parametrize(
    "family,expected,width",
    [
        ("llada", "aligned", 10),
        ("llada2", "aligned", 10),
        ("dream", "dream_dgmark", 32),
    ],
)
def test_dgmark_paper_defaults(family, expected, width):
    args = SimpleNamespace(
        generator_family=family, dgmark_decoding="auto", dgmark_top_k=None, dgmark_beam_size=None
    )
    assert resolve_decoding(args) == expected
    assert args.dgmark_top_k == args.dgmark_beam_size == width


def test_dgmark_explicit_overrides():
    args = SimpleNamespace(
        generator_family="dream", dgmark_decoding="aligned", dgmark_top_k=12, dgmark_beam_size=12
    )
    assert resolve_decoding(args) == "dream_dgmark"
    assert args.dgmark_top_k == args.dgmark_beam_size == 12


def test_origin_with_real_hash_distribution():
    tokenizer = Tokenizer()
    watermark = HashDistribution(tokenizer, delta=4, topk=4, device="cpu")
    _, ids, score = generate_hash_distribution(
        torch.tensor([[1]]),
        TinyDream(),
        tokenizer,
        7,
        watermark,
        generator_family="dream",
        gen_length=5,
        steps=3,
        temperature=0.5,
        remasking="random",
    )
    assert len(ids) <= 5
    assert torch.isfinite(torch.tensor(score["z_score"]))
