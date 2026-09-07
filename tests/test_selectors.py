import pytest
import torch

from denomark.core.selectors import select_candidate, select_candidate_max_watermark


def test_max_watermark_selects_highest_mean_signed_score():
    base_logprobs = torch.tensor([0.0, 3.0, 1.0])
    signed = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.8, 0.6]])
    selected, info = select_candidate_max_watermark(base_logprobs, signed)
    assert selected == 2
    assert info["selector"] == "max_watermark"
    assert info["selection_mode"] == "argmax_score"


def test_base_logprob_breaks_exact_watermark_ties():
    base_logprobs = torch.tensor([0.1, 0.9])
    signed = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    selected, _ = select_candidate_max_watermark(base_logprobs, signed)
    assert selected == 1


def test_top_fraction_restricts_candidate_pool():
    base_logprobs = torch.arange(8, dtype=torch.float32)
    signed = torch.zeros(8, 2)
    signed[0] = 100.0
    signed[4] = 10.0
    selected, info = select_candidate_max_watermark(
        base_logprobs,
        signed,
        argmax_logprob_top_frac=0.5,
    )
    assert selected == 4
    assert set(info["argmax_logprob_valid_indices"]) == {4, 5, 6, 7}


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_invalid_top_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        select_candidate_max_watermark(
            torch.zeros(2),
            torch.zeros(2, 2),
            argmax_logprob_top_frac=fraction,
        )


def test_public_dispatch_only_accepts_paper_selector():
    selected, _ = select_candidate(
        "max_watermark",
        torch.zeros(2),
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
    )
    assert selected == 1
    with pytest.raises(ValueError, match="unsupported paper selector"):
        select_candidate("valid_filter", torch.zeros(2), torch.zeros(2, 2))
