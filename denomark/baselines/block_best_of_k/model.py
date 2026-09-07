"""Block Best-of-K semi-autoregressive baseline for LLaDA.

At each semantic block, this baseline samples ``N`` complete LLaDA blocks from
the unwatermarked reference decoder, scores every block with the existing
signed semantic watermark key, commits the highest-scoring block, and then
continues from that prefix.

The paper baseline uses an argmax rule and is therefore distinct from
probabilistic rejection sampling, which samples candidates proportionally to a
non-negative acceptance weight.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from denomark.baselines.common import _add_gumbel_noise, _get_num_transfer_tokens
from denomark.core.scoring import block_decode
from denomark.core.model import encode_texts, get_model_logits, safe_decode


def semantic_candidate_scores(
    embeddings: torch.Tensor,
    directions_for_block: torch.Tensor,
    signs_for_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scalar and per-channel signed scores for candidate embeddings.

    This is exactly the per-block statistic used by the repository's semantic
    detector: project onto the keyed directions, apply the message signs, and
    average over watermark channels.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [num_candidates, embedding_dim]")
    if directions_for_block.ndim != 2:
        raise ValueError("directions_for_block must have shape [num_channels, embedding_dim]")
    if signs_for_block.ndim != 1:
        raise ValueError("signs_for_block must have shape [num_channels]")
    if embeddings.shape[1] != directions_for_block.shape[1]:
        raise ValueError(
            "encoder/direction dimension mismatch: "
            f"{embeddings.shape[1]} != {directions_for_block.shape[1]}"
        )
    if directions_for_block.shape[0] != signs_for_block.shape[0]:
        raise ValueError("direction/sign channel count mismatch")

    directions_for_block = directions_for_block.to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    signs_for_block = signs_for_block.to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    signed = (embeddings @ directions_for_block.T) * signs_for_block.unsqueeze(0)
    return signed.mean(dim=1), signed


def select_argmax_candidate(scores: torch.Tensor) -> int:
    """Choose the first maximum, matching ``torch.argmax`` tie semantics."""
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty 1-D tensor")
    return int(torch.argmax(scores).item())


@torch.no_grad()
def _sample_candidate_microbatch(
    base_canvas: torch.Tensor,
    llada,
    mask_id: int,
    block_start: int,
    block_end: int,
    steps_per_block: int,
    temperature: float,
    remasking: str,
    batch_size: int,
    generator_family: str,
) -> torch.Tensor:
    """Sample one microbatch of independent complete active-block candidates."""
    candidates = base_canvas.expand(batch_size, -1).clone()
    block_mask_index = candidates[:, block_start:block_end] == mask_id
    transfer_schedule = _get_num_transfer_tokens(block_mask_index, steps_per_block)

    for step_idx in range(steps_per_block):
        # LLaDA's adapter materializes full-canvas logits and returns a slice
        # view. Clone the small active slice immediately so the much larger
        # backing tensor can be released before float64 sampling temporaries.
        block_logits = get_model_logits(
            llada,
            candidates,
            generator_family,
            logit_start=block_start,
            logit_end=block_end,
        ).clone()
        sampled = torch.argmax(_add_gumbel_noise(block_logits, temperature), dim=-1)
        current_block = candidates[:, block_start:block_end]
        still_masked = current_block == mask_id

        if remasking == "low_confidence":
            probabilities = F.softmax(block_logits.to(torch.float64), dim=-1)
            confidence = torch.gather(
                probabilities,
                dim=-1,
                index=sampled.unsqueeze(-1),
            ).squeeze(-1)
        elif remasking == "random":
            confidence = torch.rand(
                sampled.shape,
                dtype=torch.float64,
                device=sampled.device,
            )
        elif remasking == "ar":
            confidence = (
                -torch.arange(sampled.shape[1], device=sampled.device, dtype=torch.float64)
                .unsqueeze(0)
                .expand(sampled.shape[0], -1)
                / max(1, sampled.shape[1])
            )
        else:
            raise ValueError(f"unsupported remasking policy: {remasking}")

        sampled = torch.where(still_masked, sampled, current_block)
        confidence = torch.where(
            still_masked,
            confidence,
            torch.full_like(confidence, -float("inf")),
        )
        transfer = torch.zeros_like(still_masked)
        for row_idx in range(batch_size):
            n_transfer = int(transfer_schedule[row_idx, step_idx].item())
            if n_transfer <= 0:
                continue
            selected_positions = torch.topk(
                confidence[row_idx],
                k=n_transfer,
            ).indices
            transfer[row_idx, selected_positions] = True
        current_block[transfer] = sampled[transfer]

    candidate_blocks = candidates[:, block_start:block_end]
    if (candidate_blocks == mask_id).any():
        remaining = int((candidate_blocks == mask_id).sum().item())
        raise RuntimeError(f"candidate block decoding left {remaining} MASK tokens")
    return candidate_blocks.detach().cpu()


@torch.no_grad()
def _score_candidate_texts(
    candidate_texts: list[str],
    encoder,
    encoder_tokenizer,
    directions_for_block: torch.Tensor,
    signs_for_block: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
    """Score non-empty candidate blocks; empty blocks are never preferred."""
    valid_indices = [idx for idx, text in enumerate(candidate_texts) if text.strip()]
    valid_index_set = set(valid_indices)
    valid_mask = [idx in valid_index_set for idx in range(len(candidate_texts))]
    scalar_scores = torch.full((len(candidate_texts),), -float("inf"), dtype=torch.float32)
    channel_scores = torch.full(
        (len(candidate_texts), int(signs_for_block.numel())),
        float("nan"),
        dtype=torch.float32,
    )
    if not valid_indices:
        return scalar_scores, channel_scores, valid_mask

    embeddings = encode_texts(
        [candidate_texts[idx] for idx in valid_indices],
        encoder,
        encoder_tokenizer,
        device,
        batch_sz=len(valid_indices),
        to_cpu=False,
    )
    valid_scalar, valid_channels = semantic_candidate_scores(
        embeddings,
        directions_for_block,
        signs_for_block,
    )
    if not torch.isfinite(valid_scalar).all() or not torch.isfinite(valid_channels).all():
        raise RuntimeError("semantic encoder produced a non-finite candidate score")
    valid_index_tensor = torch.tensor(valid_indices, dtype=torch.long)
    scalar_scores.index_copy_(0, valid_index_tensor, valid_scalar.detach().float().cpu())
    channel_scores.index_copy_(0, valid_index_tensor, valid_channels.detach().float().cpu())
    return scalar_scores, channel_scores, valid_mask


def _json_float(value: torch.Tensor | float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


@torch.no_grad()
def llada_generate_block_best_of_k(
    prompt: torch.Tensor,
    llada,
    encoder,
    encoder_tokenizer,
    llada_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    mask_id: int,
    *,
    gen_length: int = 300,
    block_size: int = 25,
    steps: int | None = None,
    temperature: float = 0.6,
    num_candidates: int = 16,
    candidate_batch_size: int | None = None,
    remasking: str = "low_confidence",
    generator_family: str = "llada",
    device: str | torch.device | None = None,
    record_candidate_texts: bool = False,
) -> tuple[str, list[int], list[dict[str, Any]]]:
    """Generate one response with semantic-unit Best-of-K selection.

    Each candidate is a complete block sampled with the same iterative LLaDA
    decoder used by the unwatermarked baseline.  Candidate blocks are generated
    from the same committed prefix and remain independent proposal samples.
    """
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("prompt must have shape [1, prompt_length]")
    if generator_family != "llada":
        raise ValueError("block-reject currently supports LLaDA-style generators only")
    if gen_length <= 0 or block_size <= 0 or gen_length % block_size != 0:
        raise ValueError("gen_length and block_size must be positive, with block_size dividing gen_length")
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    num_blocks = gen_length // block_size
    if directions.ndim != 3 or signs.ndim != 2:
        raise ValueError("directions/signs must have shapes [blocks, channels, dim] and [blocks, channels]")
    if directions.shape[:2] != signs.shape or directions.shape[0] < num_blocks:
        raise ValueError("watermark directions/signs do not cover every generation block")

    if steps is None:
        steps = gen_length
    if steps <= 0 or steps % num_blocks != 0:
        raise ValueError("steps must be positive and divisible by the number of blocks")
    steps_per_block = steps // num_blocks
    batch_cap = min(num_candidates, int(candidate_batch_size or num_candidates))
    if batch_cap <= 0:
        raise ValueError("candidate_batch_size must be positive")

    generation_device = prompt.device if device is None else torch.device(device)
    prompt = prompt.to(generation_device)
    prompt_length = int(prompt.shape[1])
    canvas = torch.full(
        (1, prompt_length + gen_length),
        int(mask_id),
        dtype=torch.long,
        device=generation_device,
    )
    canvas[:, :prompt_length] = prompt

    diagnostics: list[dict[str, Any]] = []
    for block_id in range(num_blocks):
        block_start = prompt_length + block_id * block_size
        block_end = block_start + block_size
        candidate_chunks: list[torch.Tensor] = []
        for candidate_start in range(0, num_candidates, batch_cap):
            microbatch_size = min(batch_cap, num_candidates - candidate_start)
            candidate_chunks.append(
                _sample_candidate_microbatch(
                    canvas,
                    llada,
                    mask_id,
                    block_start,
                    block_end,
                    steps_per_block,
                    temperature,
                    remasking,
                    microbatch_size,
                    generator_family,
                )
            )
        candidate_blocks = torch.cat(candidate_chunks, dim=0)
        candidate_texts = [
            block_decode(token_ids, 0, len(token_ids), llada_tokenizer)
            for token_ids in candidate_blocks.tolist()
        ]
        scalar_scores, channel_scores, valid_mask = _score_candidate_texts(
            candidate_texts,
            encoder,
            encoder_tokenizer,
            directions[block_id],
            signs[block_id],
            generation_device,
        )
        if any(valid_mask):
            selected_idx = select_argmax_candidate(scalar_scores)
            selection_reason = "argmax_semantic_watermark_score"
        else:
            # The detector treats blocks made entirely of special/EOT tokens as
            # inactive.  There is therefore no semantic statistic to maximize;
            # use a deterministic first-candidate fallback while still saving
            # all K fully decoded proposals in the diagnostics.
            selected_idx = 0
            selection_reason = "all_candidates_empty_fallback_first"
        selected_block = candidate_blocks[selected_idx].to(generation_device)
        canvas[0, block_start:block_end] = selected_block

        block_diag: dict[str, Any] = {
            "method": "block_best_of_k",
            "selection_rule": "argmax_semantic_watermark_score",
            "empty_block_policy": "first_candidate_when_all_semantically_empty",
            "selection_reason": selection_reason,
            "block_id": block_id,
            "block_start": block_id * block_size,
            "block_end": (block_id + 1) * block_size,
            "num_candidates": num_candidates,
            "candidate_batch_size": batch_cap,
            "temperature": float(temperature),
            "steps_per_block": steps_per_block,
            "remasking": remasking,
            "candidate_valid": valid_mask,
            "candidate_scores": [_json_float(score) for score in scalar_scores],
            "candidate_channel_scores": [
                [_json_float(value) for value in row] if valid else None
                for row, valid in zip(channel_scores, valid_mask)
            ],
            "selected_candidate_index": selected_idx,
            "selected_score": _json_float(scalar_scores[selected_idx]),
            "selected_text": candidate_texts[selected_idx],
            "selected_token_ids": candidate_blocks[selected_idx].tolist(),
        }
        if record_candidate_texts:
            block_diag["candidate_texts"] = candidate_texts
        diagnostics.append(block_diag)

    token_ids = canvas[0, prompt_length:].tolist()
    text = safe_decode(llada_tokenizer, token_ids, skip_special_tokens=True).strip()
    return text, token_ids, diagnostics


__all__ = [
    "llada_generate_block_best_of_k",
    "select_argmax_candidate",
    "semantic_candidate_scores",
]
