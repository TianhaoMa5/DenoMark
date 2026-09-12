"""SemStamp-style semantic rejection sampling for fixed LLaDA blocks.

This module keeps the repository's existing semantic encoder, but replaces the
continuous-score argmax used by Block Best-of-N with SemStamp's algorithmic
structure:

1. fixed random-projection LSH partitions the embedding space;
2. the previous semantic hash deterministically keys a valid set of next bins;
3. proposals are examined in sampling order and the first valid proposal is
   accepted (batched proposal decoding is only a vectorized acceleration);
4. detection counts valid hash transitions and reports a SemStamp z-score.

Unlike :mod:`denmark.baselines.block_best_of_k.model`, no candidate is selected because it
has the largest continuous watermark projection.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from denmark.baselines.block_best_of_k.model import _sample_candidate_microbatch
from denmark.core.scoring import block_decode
from denmark.core.model import encode_texts, safe_decode


SEMSTAMP_HASH_KEY = 15_485_863


def build_lsh_hyperplanes(
    lsh_dim: int,
    embedding_dim: int,
    seed: int = 1234,
) -> torch.Tensor:
    """Return NearPy-compatible unit LSH hyperplanes with shape ``[D, E]``.

    SemStamp constructs ``RandomBinaryProjections(..., rand_seed=1234)`` and
    NearPy initializes its normals with ``numpy.random.RandomState(seed).randn``.
    Normalizing each row preserves its hash sign and gives the cosine boundary
    distance used by SemStamp's margin filter.
    """
    if lsh_dim <= 0 or embedding_dim <= 0:
        raise ValueError("lsh_dim and embedding_dim must be positive")
    random_state = np.random.RandomState(int(seed))
    nearpy_normals = random_state.randn(lsh_dim, embedding_dim)
    return F.normalize(torch.from_numpy(nearpy_normals).float(), dim=-1)


def embedding_hashes_and_margins(
    embeddings: torch.Tensor,
    hyperplanes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute integer LSH bins, boundary margins, and raw projections.

    Bit zero is the most-significant bit, matching SemStamp's conversion of a
    projection bit string via ``int(bit_string, 2)``.  The margin is the
    minimum absolute cosine projection onto any LSH hyperplane.
    """
    if embeddings.ndim != 2 or hyperplanes.ndim != 2:
        raise ValueError("embeddings and hyperplanes must both be 2-D")
    if embeddings.shape[1] != hyperplanes.shape[1]:
        raise ValueError(
            "encoder/hyperplane dimension mismatch: "
            f"{embeddings.shape[1]} != {hyperplanes.shape[1]}"
        )
    normalized_embeddings = F.normalize(embeddings.float(), dim=-1)
    normalized_hyperplanes = F.normalize(hyperplanes.float(), dim=-1)
    projections = normalized_embeddings @ normalized_hyperplanes.T
    # NearPy emits "1" only for a strictly positive projection.
    bits = (projections > 0).to(torch.long)
    powers = 2 ** torch.arange(
        hyperplanes.shape[0] - 1,
        -1,
        -1,
        dtype=torch.long,
        device=bits.device,
    )
    hashes = (bits * powers.unsqueeze(0)).sum(dim=-1)
    margins = projections.abs().min(dim=-1).values
    return hashes, margins, projections


def valid_bins_from_previous_hash(
    previous_hash: int,
    lsh_dim: int,
    accept_rate: float = 0.25,
    hash_key: int = SEMSTAMP_HASH_KEY,
    rng_device: str | torch.device | None = None,
) -> tuple[int, ...]:
    """Return SemStamp's keyed valid-bin subset for one transition.

    Official SemStamp creates the generator and ``randperm`` on the active
    CUDA device.  PyTorch's CPU and CUDA permutations differ for the same seed,
    so experiment callers must pass their actual model device.  The automatic
    fallback is only for lightweight CPU tests.
    """
    if lsh_dim <= 0:
        raise ValueError("lsh_dim must be positive")
    if not 0.0 < accept_rate < 1.0:
        raise ValueError("accept_rate must be strictly between 0 and 1")
    num_bins = 2**lsh_dim
    if not 0 <= int(previous_hash) < num_bins:
        raise ValueError(f"previous_hash must be in [0, {num_bins})")
    num_accept = int(num_bins * float(accept_rate))
    if num_accept <= 0 or num_accept >= num_bins:
        raise ValueError(
            f"accept_rate={accept_rate} yields invalid valid-bin count {num_accept}/{num_bins}"
        )
    permutation_device = torch.device(
        rng_device
        if rng_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if permutation_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SemStamp CUDA mask requested but CUDA is unavailable")
    generator = torch.Generator(device=permutation_device)
    generator.manual_seed(int(hash_key) * int(previous_hash))
    bins = torch.randperm(
        num_bins,
        device=permutation_device,
        generator=generator,
    )[:num_accept].cpu().tolist()
    return tuple(int(value) for value in bins)


def first_accepted_candidate(
    candidate_hashes: Iterable[int],
    candidate_margins: Iterable[float],
    candidate_valid: Iterable[bool],
    valid_bins: Iterable[int],
    margin: float,
) -> int | None:
    """Return the first proposal accepted by SemStamp, never the best score."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    hashes = list(candidate_hashes)
    margins = list(candidate_margins)
    valid = list(candidate_valid)
    if not (len(hashes) == len(margins) == len(valid)):
        raise ValueError("candidate hash, margin, and valid arrays must align")
    allowed = set(int(value) for value in valid_bins)
    for candidate_idx, (hash_value, boundary_margin, is_valid) in enumerate(
        zip(hashes, margins, valid)
    ):
        if (
            is_valid
            and math.isfinite(float(boundary_margin))
            and float(boundary_margin) >= margin
            and int(hash_value) in allowed
        ):
            return candidate_idx
    return None


@torch.no_grad()
def _encode_nonempty_texts(
    texts: list[str],
    encoder,
    encoder_tokenizer,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[bool]]:
    valid_indices = [idx for idx, text in enumerate(texts) if text.strip()]
    valid_set = set(valid_indices)
    valid_mask = [idx in valid_set for idx in range(len(texts))]
    if not valid_indices:
        return torch.empty((0, 0), dtype=torch.float32), valid_mask
    embeddings = encode_texts(
        [texts[idx] for idx in valid_indices],
        encoder,
        encoder_tokenizer,
        device,
        batch_sz=len(valid_indices),
        to_cpu=True,
    ).float()
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("semantic encoder produced non-finite embeddings")
    return embeddings, valid_mask


def _expand_candidate_lsh(
    embeddings: torch.Tensor,
    valid_mask: list[bool],
    hyperplanes: torch.Tensor,
) -> tuple[list[int], list[float], list[list[float]]]:
    hashes = [-1] * len(valid_mask)
    margins = [float("nan")] * len(valid_mask)
    projections: list[list[float]] = [[] for _ in valid_mask]
    if embeddings.numel() == 0:
        return hashes, margins, projections
    valid_hashes, valid_margins, valid_projections = embedding_hashes_and_margins(
        embeddings,
        hyperplanes,
    )
    valid_idx = 0
    for candidate_idx, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue
        hashes[candidate_idx] = int(valid_hashes[valid_idx].item())
        margins[candidate_idx] = float(valid_margins[valid_idx].item())
        projections[candidate_idx] = [
            float(value) for value in valid_projections[valid_idx].tolist()
        ]
        valid_idx += 1
    return hashes, margins, projections


@torch.no_grad()
def llada_generate_semstamp_blocks(
    prompt: torch.Tensor,
    prompt_seed_text: str,
    llada,
    encoder,
    encoder_tokenizer,
    llada_tokenizer,
    hyperplanes: torch.Tensor,
    mask_id: int,
    *,
    gen_length: int = 300,
    block_size: int = 25,
    steps: int | None = None,
    temperature: float = 0.7,
    proposal_batch_size: int = 16,
    max_trials_per_block: int = 100,
    accept_rate: float = 0.25,
    margin: float = 0.02,
    hash_key: int = SEMSTAMP_HASH_KEY,
    remasking: str = "low_confidence",
    generator_family: str = "llada",
    device: str | torch.device | None = None,
    record_candidate_texts: bool = False,
) -> tuple[str, list[int], list[dict[str, Any]]]:
    """Generate fixed blocks using batched SemStamp rejection sampling."""
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("prompt must have shape [1, prompt_length]")
    if not prompt_seed_text.strip():
        raise ValueError("prompt_seed_text must be non-empty")
    if gen_length <= 0 or block_size <= 0 or gen_length % block_size != 0:
        raise ValueError("block_size must divide positive gen_length")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if proposal_batch_size <= 0 or max_trials_per_block <= 0:
        raise ValueError("proposal batch and max trials must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    num_blocks = gen_length // block_size
    if steps is None:
        steps = gen_length
    if steps <= 0 or steps % num_blocks != 0:
        raise ValueError("steps must be positive and divisible by the number of blocks")
    steps_per_block = steps // num_blocks

    generation_device = prompt.device if device is None else torch.device(device)
    prompt = prompt.to(generation_device)
    prompt_length = int(prompt.shape[1])
    hyperplanes_cpu = hyperplanes.detach().float().cpu()
    if hyperplanes_cpu.ndim != 2:
        raise ValueError("hyperplanes must have shape [lsh_dim, embedding_dim]")
    lsh_dim = int(hyperplanes_cpu.shape[0])

    prompt_embeddings, prompt_valid = _encode_nonempty_texts(
        [prompt_seed_text], encoder, encoder_tokenizer, generation_device
    )
    if prompt_valid != [True]:
        raise RuntimeError("prompt did not produce a semantic embedding")
    prompt_hashes, _, _ = embedding_hashes_and_margins(
        prompt_embeddings,
        hyperplanes_cpu,
    )
    previous_hash = int(prompt_hashes[0].item())

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
        allowed_bins = valid_bins_from_previous_hash(
            previous_hash,
            lsh_dim,
            accept_rate,
            hash_key,
            rng_device=generation_device,
        )
        round_diagnostics: list[dict[str, Any]] = []
        selected_block: torch.Tensor | None = None
        selected_text = ""
        selected_hash = -1
        selected_margin = float("nan")
        selected_global_idx = -1
        selection_reason = ""
        last_nonempty: tuple[torch.Tensor, str, int, float, int] | None = None
        trials = 0
        proposals_drawn = 0
        trials_examined = 0

        while trials < max_trials_per_block and selected_block is None:
            round_size = min(proposal_batch_size, max_trials_per_block - trials)
            candidate_blocks = _sample_candidate_microbatch(
                canvas,
                llada,
                mask_id,
                block_start,
                block_end,
                steps_per_block,
                temperature,
                remasking,
                round_size,
                generator_family,
            )
            candidate_texts = [
                block_decode(token_ids, 0, len(token_ids), llada_tokenizer)
                for token_ids in candidate_blocks.tolist()
            ]
            embeddings, valid_mask = _encode_nonempty_texts(
                candidate_texts,
                encoder,
                encoder_tokenizer,
                generation_device,
            )
            candidate_hashes, candidate_margins, candidate_projections = (
                _expand_candidate_lsh(embeddings, valid_mask, hyperplanes_cpu)
            )
            for idx, is_valid in enumerate(valid_mask):
                if is_valid:
                    last_nonempty = (
                        candidate_blocks[idx],
                        candidate_texts[idx],
                        candidate_hashes[idx],
                        candidate_margins[idx],
                        trials + idx,
                    )
            accepted_local_idx = first_accepted_candidate(
                candidate_hashes,
                candidate_margins,
                valid_mask,
                allowed_bins,
                margin,
            )
            round_diag: dict[str, Any] = {
                "round_start_trial": trials,
                "round_size": round_size,
                "candidate_valid": valid_mask,
                "candidate_hashes": candidate_hashes,
                "candidate_margins": [
                    value if math.isfinite(value) else None
                    for value in candidate_margins
                ],
                "candidate_projections": candidate_projections,
                "accepted_local_index": accepted_local_idx,
            }
            if record_candidate_texts:
                round_diag["candidate_texts"] = candidate_texts
            round_diagnostics.append(round_diag)
            proposals_drawn += round_size
            if accepted_local_idx is not None:
                selected_block = candidate_blocks[accepted_local_idx]
                selected_text = candidate_texts[accepted_local_idx]
                selected_hash = int(candidate_hashes[accepted_local_idx])
                selected_margin = float(candidate_margins[accepted_local_idx])
                selected_global_idx = trials + accepted_local_idx
                trials_examined = selected_global_idx + 1
                selection_reason = "first_valid_semstamp_transition"
            trials += round_size

        if selected_block is None:
            if last_nonempty is None:
                # SemStamp cannot hash an empty semantic segment.  Preserve the
                # generation trajectory with a deterministic final proposal.
                fallback = _sample_candidate_microbatch(
                    canvas,
                    llada,
                    mask_id,
                    block_start,
                    block_end,
                    steps_per_block,
                    temperature,
                    remasking,
                    1,
                    generator_family,
                )[0]
                proposals_drawn += 1
                selected_block = fallback
                selected_text = block_decode(
                    fallback.tolist(), 0, len(fallback), llada_tokenizer
                )
                selection_reason = "max_trials_all_candidates_empty_fallback"
            else:
                (
                    selected_block,
                    selected_text,
                    selected_hash,
                    selected_margin,
                    selected_global_idx,
                ) = last_nonempty
                selection_reason = "max_trials_last_nonempty_fallback"
            trials_examined = max_trials_per_block

        canvas[0, block_start:block_end] = selected_block.to(generation_device)
        transition_hit = selected_hash in set(allowed_bins)
        margin_pass = math.isfinite(selected_margin) and selected_margin >= margin
        block_diag: dict[str, Any] = {
            "method": "semstamp_candidate",
            "selection_rule": "first_valid_semstamp_transition",
            "fallback_policy": "last_nonempty_after_max_trials",
            "selection_reason": selection_reason,
            "block_id": block_id,
            "block_start": block_id * block_size,
            "block_end": (block_id + 1) * block_size,
            "temperature": float(temperature),
            "steps_per_block": steps_per_block,
            "remasking": remasking,
            "lsh_dim": lsh_dim,
            "accept_rate": float(accept_rate),
            "margin": float(margin),
            "hash_key": int(hash_key),
            "previous_hash": previous_hash,
            "valid_bins": list(allowed_bins),
            "proposal_batch_size": proposal_batch_size,
            "max_trials_per_block": max_trials_per_block,
            "num_trials": trials_examined,
            "num_proposals_drawn": proposals_drawn,
            "selected_candidate_global_index": selected_global_idx,
            "selected_hash": selected_hash if selected_hash >= 0 else None,
            "selected_margin": selected_margin if math.isfinite(selected_margin) else None,
            "transition_hit": transition_hit,
            "margin_pass": margin_pass,
            "selected_text": selected_text,
            "selected_token_ids": selected_block.tolist(),
            "rounds": round_diagnostics,
        }
        diagnostics.append(block_diag)
        if selected_hash >= 0:
            previous_hash = selected_hash

    token_ids = canvas[0, prompt_length:].tolist()
    text = safe_decode(llada_tokenizer, token_ids, skip_special_tokens=True).strip()
    return text, token_ids, diagnostics


@torch.no_grad()
def detect_semstamp_block_transitions(
    block_texts: list[str],
    prompt_seed_text: str,
    encoder,
    encoder_tokenizer,
    hyperplanes: torch.Tensor,
    *,
    accept_rate: float = 0.25,
    hash_key: int = SEMSTAMP_HASH_KEY,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Return transition hits and the natural SemStamp z-score."""
    active_texts = [text for text in block_texts if text.strip()]
    texts = [prompt_seed_text, *active_texts]
    embeddings, valid_mask = _encode_nonempty_texts(
        texts,
        encoder,
        encoder_tokenizer,
        device,
    )
    if not all(valid_mask) or len(embeddings) != len(texts):
        raise RuntimeError("prompt/active blocks did not all produce embeddings")
    hashes, margins, _ = embedding_hashes_and_margins(
        embeddings,
        hyperplanes.detach().float().cpu(),
    )
    hash_values = [int(value) for value in hashes.tolist()]
    hits: list[bool] = []
    valid_bin_sequence: list[list[int]] = []
    lsh_dim = int(hyperplanes.shape[0])
    for previous_hash, current_hash in zip(hash_values[:-1], hash_values[1:]):
        allowed = valid_bins_from_previous_hash(
            previous_hash,
            lsh_dim,
            accept_rate,
            hash_key,
            rng_device=device,
        )
        valid_bin_sequence.append(list(allowed))
        hits.append(current_hash in set(allowed))
    n_transitions = len(hits)
    n_hits = sum(hits)
    if n_transitions:
        denominator = math.sqrt(
            n_transitions * accept_rate * (1.0 - accept_rate)
        )
        z_score = (n_hits - accept_rate * n_transitions) / denominator
    else:
        z_score = 0.0
    return {
        "n_active_blocks": len(active_texts),
        "n_transitions": n_transitions,
        "n_hits": n_hits,
        "hit_rate": n_hits / n_transitions if n_transitions else 0.0,
        "z_score": float(z_score),
        "hashes": hash_values,
        "margins": [float(value) for value in margins.tolist()],
        "transition_hits": hits,
        "valid_bins": valid_bin_sequence,
        "mask_rng_device": str(torch.device(device)),
        "mask_scheme": "semstamp_torch_randperm_on_active_device_v2",
    }


__all__ = [
    "SEMSTAMP_HASH_KEY",
    "build_lsh_hyperplanes",
    "detect_semstamp_block_transitions",
    "embedding_hashes_and_margins",
    "first_accepted_candidate",
    "llada_generate_semstamp_blocks",
    "valid_bins_from_previous_hash",
]
