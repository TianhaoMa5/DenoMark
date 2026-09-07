"""Candidate selection used by DenoMark generation."""

from __future__ import annotations

import math

import torch


def _to_tensor(value, device=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device) if device is not None else value
    return torch.tensor(value, device=device)


def _top_logprob_indices(base_logprobs: torch.Tensor, top_fraction: float) -> torch.Tensor:
    """Return indices of the highest-probability candidate fraction."""
    fraction = float(top_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    num_candidates = int(base_logprobs.shape[0])
    num_keep = max(1, min(num_candidates, math.ceil(num_candidates * fraction)))
    if num_keep == num_candidates:
        return torch.arange(num_candidates, device=base_logprobs.device)
    return torch.topk(base_logprobs, k=num_keep, largest=True).indices


def select_candidate_max_watermark(
    base_logprobs,
    signed_scores,
    argmax_logprob_top_frac: float = 1.0,
):
    """Select the candidate with the largest mean signed watermark score.

    Base-model log-probability breaks exact watermark-score ties. The optional
    top-fraction restriction is retained for the paper's quality ablations;
    the main configuration uses the full candidate set.
    """
    signed = _to_tensor(signed_scores)
    base_lps = _to_tensor(base_logprobs, device=signed.device)
    if signed.ndim != 2 or signed.shape[0] == 0:
        raise ValueError("signed_scores must have shape [K, C] with K > 0")
    if base_lps.ndim != 1 or base_lps.shape[0] != signed.shape[0]:
        raise ValueError("base_logprobs must have one value per candidate")

    watermark_scores = signed.mean(dim=1)
    valid_indices = _top_logprob_indices(base_lps, argmax_logprob_top_frac)
    valid_scores = watermark_scores[valid_indices]
    best_score = valid_scores.max()
    ties = valid_indices[(valid_scores == best_score).nonzero(as_tuple=False).flatten()]
    selected = int(ties[base_lps[ties].argmax()].item()) if ties.numel() > 1 else int(ties[0])

    best_logprob_index = int(base_lps.argmax())
    selected_logprob = float(base_lps[selected])
    best_logprob = float(base_lps[best_logprob_index])
    return selected, {
        "selector": "max_watermark",
        "selection_mode": "argmax_score",
        "initial_K": int(signed.shape[0]),
        "fallback": False,
        "selected_candidate_index": selected,
        "argmax_logprob_top_frac": float(argmax_logprob_top_frac),
        "argmax_logprob_num_keep": int(valid_indices.numel()),
        "argmax_logprob_valid_indices": [
            int(index) for index in valid_indices.detach().cpu().tolist()
        ],
        "selected_watermark_score": float(watermark_scores[selected]),
        "best_watermark_score": float(best_score),
        "selected_logprob": selected_logprob,
        "best_logprob": best_logprob,
        "logprob_gap_to_best": best_logprob - selected_logprob,
        "mean_signed_score_selected": float(signed[selected].mean()),
        "mean_signed_score_best_logprob": float(signed[best_logprob_index].mean()),
    }


def select_candidate(
    selector: str,
    base_logprobs,
    signed_scores,
    *,
    argmax_logprob_top_frac: float = 1.0,
    **_unused,
):
    """Dispatch the paper's candidate selector.

    The public reproduction path intentionally exposes only DenoMark's
    max-watermark rule. Historical research selectors are not part of the
    release.
    """
    if selector != "max_watermark":
        raise ValueError(f"unsupported paper selector: {selector}")
    return select_candidate_max_watermark(
        base_logprobs,
        signed_scores,
        argmax_logprob_top_frac=argmax_logprob_top_frac,
    )


__all__ = [
    "select_candidate",
    "select_candidate_max_watermark",
]
