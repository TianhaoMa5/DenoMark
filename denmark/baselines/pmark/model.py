"""PMark multi-channel selection for complete fixed LLaDA blocks.

This ports PMark's supported ``prior``/offline path to diffusion blocks: sample
all candidates from the same committed prefix, project their semantic
embeddings onto fixed orthogonal pivots, and sequentially retain the half-space
specified by each secret bit.  The semantic encoder is supplied by the caller.
"""
from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from denmark.baselines.block_best_of_k.model import _sample_candidate_microbatch
from denmark.core.scoring import block_decode
from denmark.core.model import encode_texts, safe_decode


def build_pmark_pivots(
    embedding_dim: int,
    num_channels: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Return PMark-compatible orthogonal random pivots with shape ``[B, E]``."""
    if embedding_dim <= 0 or num_channels <= 0:
        raise ValueError("embedding_dim and num_channels must be positive")
    if num_channels > embedding_dim:
        raise ValueError("num_channels cannot exceed embedding_dim")
    rng = np.random.default_rng(int(seed))
    random_matrix = rng.standard_normal((embedding_dim, num_channels))
    q_matrix, _ = np.linalg.qr(random_matrix)
    return torch.from_numpy(q_matrix.T.copy()).float()


def build_pmark_secret_bits(
    num_blocks: int,
    num_channels: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Build a deterministic secret bit matrix for generated block indices.

    PMark reads a fixed secret bit stream and groups it into ``msig``-wide rows.
    A seeded private stream is equivalent while making the experiment key
    explicit and reproducible.
    """
    if num_blocks <= 0 or num_channels <= 0:
        raise ValueError("num_blocks and num_channels must be positive")
    rng = np.random.default_rng(int(seed))
    bits = rng.integers(0, 2, size=(num_blocks + 1, num_channels), dtype=np.int64)
    # Official generation starts at sentence id 1, leaving row zero unused.
    return torch.from_numpy(bits[1:].copy()).to(torch.bool)


def pmark_cosine_projections(
    embeddings: torch.Tensor,
    pivots: torch.Tensor,
) -> torch.Tensor:
    """Return cosine similarities between embeddings and PMark pivots."""
    if embeddings.ndim != 2 or pivots.ndim != 2:
        raise ValueError("embeddings and pivots must be two-dimensional")
    if embeddings.shape[1] != pivots.shape[1]:
        raise ValueError("encoder and pivot dimensions do not match")
    normalized_embeddings = F.normalize(embeddings.float(), dim=-1)
    normalized_pivots = F.normalize(pivots.float(), dim=-1)
    return normalized_embeddings @ normalized_pivots.T


def select_pmark_candidate(
    projections: torch.Tensor,
    target_bits: torch.Tensor,
    valid_mask: list[bool],
    *,
    median_method: str = "prior",
) -> tuple[int, list[float], list[int]]:
    """Apply PMark's online multi-channel filtering and select one survivor."""
    if projections.ndim != 2 or target_bits.ndim != 1:
        raise ValueError("projections/target_bits must have shapes [N,B] and [B]")
    if projections.shape[1] != target_bits.numel():
        raise ValueError("projection and target channel counts do not match")
    if projections.shape[0] != len(valid_mask) or not valid_mask:
        raise ValueError("valid_mask must align with a nonempty candidate batch")
    if median_method not in {"prior", "torch"}:
        raise ValueError("median_method must be 'prior' or 'torch'")

    alive = [index for index, valid in enumerate(valid_mask) if valid]
    if not alive:
        return 0, [], []
    medians: list[float] = []
    for channel, target in enumerate(target_bits.tolist()):
        if not alive:
            break
        values = projections[alive, channel]
        median = torch.tensor(0.0) if median_method == "prior" else torch.median(values)
        medians.append(float(median))
        keep = values >= median if bool(target) else values < median
        if int(keep.sum().item()) == 0:
            keep = values > median if bool(target) else values <= median
        if int(keep.sum().item()) == 0:
            equal_positions = torch.nonzero(values == median, as_tuple=False).flatten().tolist()
            if equal_positions:
                keep = torch.zeros(len(alive), dtype=torch.bool)
                keep[equal_positions[0]] = True
            else:
                keep = torch.zeros(len(alive), dtype=torch.bool)
                keep[0] = True
        alive = [alive[pos] for pos, is_kept in enumerate(keep.tolist()) if is_kept]

    if not alive:
        valid_indices = [index for index, valid in enumerate(valid_mask) if valid]
        return random.choice(valid_indices), medians, []
    return random.choice(alive), medians, alive


def pmark_soft_matches(
    projections: torch.Tensor,
    target_bits: torch.Tensor,
    *,
    tolerance: float = 0.001,
    decay: float = 250.0,
) -> torch.Tensor:
    """Return PMark detector's per-channel soft green indicators."""
    if tolerance < 0 or decay < 0:
        raise ValueError("tolerance and decay must be non-negative")
    target = target_bits.to(device=projections.device, dtype=torch.bool)
    hard = torch.where(target, projections >= -tolerance, projections <= tolerance)
    soft = torch.exp(-float(decay) * projections.abs())
    return torch.where(hard, torch.ones_like(projections), soft)


@torch.no_grad()
def _encode_candidate_texts(
    texts: list[str],
    encoder,
    encoder_tokenizer,
    pivots: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, list[bool]]:
    valid_indices = [index for index, text in enumerate(texts) if text.strip()]
    valid_set = set(valid_indices)
    valid_mask = [index in valid_set for index in range(len(texts))]
    projections = torch.full(
        (len(texts), int(pivots.shape[0])),
        float("nan"),
        dtype=torch.float32,
    )
    if not valid_indices:
        return projections, valid_mask
    embeddings = encode_texts(
        [texts[index] for index in valid_indices],
        encoder,
        encoder_tokenizer,
        device,
        batch_sz=len(valid_indices),
        to_cpu=True,
    ).float()
    valid_projections = pmark_cosine_projections(embeddings, pivots)
    if not torch.isfinite(valid_projections).all():
        raise RuntimeError("semantic encoder produced invalid PMark projections")
    projections[torch.tensor(valid_indices, dtype=torch.long)] = valid_projections
    return projections, valid_mask


@torch.no_grad()
def llada_generate_pmark_blocks(
    prompt: torch.Tensor,
    llada,
    encoder,
    encoder_tokenizer,
    llada_tokenizer,
    pivots: torch.Tensor,
    secret_bits: torch.Tensor,
    mask_id: int,
    *,
    gen_length: int = 300,
    block_size: int = 25,
    steps: int | None = None,
    temperature: float = 0.7,
    num_candidates: int = 16,
    candidate_batch_size: int | None = None,
    median_method: str = "prior",
    remasking: str = "low_confidence",
    generator_family: str = "llada",
    device: str | torch.device | None = None,
    record_candidate_texts: bool = False,
) -> tuple[str, list[int], list[dict[str, Any]]]:
    """Generate fixed LLaDA blocks with PMark multi-channel selection."""
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("prompt must have shape [1, prompt_length]")
    if gen_length <= 0 or block_size <= 0 or gen_length % block_size:
        raise ValueError("block_size must divide positive gen_length")
    if num_candidates <= 0 or temperature < 0:
        raise ValueError("candidate count must be positive and temperature non-negative")
    num_blocks = gen_length // block_size
    if pivots.ndim != 2 or secret_bits.ndim != 2:
        raise ValueError("pivots/secret_bits must be two-dimensional")
    if secret_bits.shape[0] < num_blocks or secret_bits.shape[1] != pivots.shape[0]:
        raise ValueError("secret bits do not cover all blocks and PMark channels")
    if steps is None:
        steps = gen_length
    if steps <= 0 or steps % num_blocks:
        raise ValueError("steps must be divisible by the number of blocks")
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
        chunks = []
        for candidate_start in range(0, num_candidates, batch_cap):
            chunks.append(
                _sample_candidate_microbatch(
                    canvas,
                    llada,
                    mask_id,
                    block_start,
                    block_end,
                    steps_per_block,
                    temperature,
                    remasking,
                    min(batch_cap, num_candidates - candidate_start),
                    generator_family,
                )
            )
        candidate_blocks = torch.cat(chunks, dim=0)
        candidate_texts = [
            block_decode(token_ids, 0, len(token_ids), llada_tokenizer)
            for token_ids in candidate_blocks.tolist()
        ]
        projections, valid_mask = _encode_candidate_texts(
            candidate_texts,
            encoder,
            encoder_tokenizer,
            pivots,
            generation_device,
        )
        target_bits = secret_bits[block_id]
        selected_index, medians, survivors = select_pmark_candidate(
            projections,
            target_bits,
            valid_mask,
            median_method=median_method,
        )
        selected_block = candidate_blocks[selected_index].to(generation_device)
        canvas[0, block_start:block_end] = selected_block
        selected_projection = projections[selected_index]
        selected_matches = (
            torch.where(target_bits, selected_projection >= 0, selected_projection < 0)
            if valid_mask[selected_index]
            else torch.zeros_like(target_bits)
        )
        diagnostic: dict[str, Any] = {
            "method": "pmark_block_multichannel",
            "selection_rule": "sequential_multichannel_halfspace_filter",
            "median_method": median_method,
            "block_id": block_id,
            "block_start": block_id * block_size,
            "block_end": (block_id + 1) * block_size,
            "num_candidates": num_candidates,
            "candidate_batch_size": batch_cap,
            "temperature": float(temperature),
            "steps_per_block": steps_per_block,
            "remasking": remasking,
            "target_bits": [bool(value) for value in target_bits.tolist()],
            "step_medians": medians,
            "candidate_valid": valid_mask,
            "candidate_projections": [
                [float(value) for value in row.tolist()] if valid else None
                for row, valid in zip(projections, valid_mask)
            ],
            "surviving_candidate_indices": survivors,
            "selected_candidate_index": selected_index,
            "selected_projections": (
                [float(value) for value in selected_projection.tolist()]
                if valid_mask[selected_index]
                else None
            ),
            "selected_matches": [bool(value) for value in selected_matches.tolist()],
            "selected_text": candidate_texts[selected_index],
            "selected_token_ids": candidate_blocks[selected_index].tolist(),
        }
        if record_candidate_texts:
            diagnostic["candidate_texts"] = candidate_texts
        diagnostics.append(diagnostic)

    token_ids = canvas[0, prompt_length:].tolist()
    text = safe_decode(llada_tokenizer, token_ids, skip_special_tokens=True).strip()
    return text, token_ids, diagnostics


@torch.no_grad()
def detect_pmark_blocks(
    block_texts: list[str],
    encoder,
    encoder_tokenizer,
    pivots: torch.Tensor,
    secret_bits: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    tolerance: float = 0.001,
    decay: float = 250.0,
) -> dict[str, Any]:
    """Detect PMark's prior/offline multi-channel signal in fixed blocks."""
    if len(block_texts) > secret_bits.shape[0]:
        raise ValueError("secret bit stream does not cover all detected blocks")
    active_indices = [index for index, text in enumerate(block_texts) if text.strip()]
    if not active_indices:
        return {
            "n_active_blocks": 0,
            "n_channels": int(pivots.shape[0]),
            "n_trials": 0,
            "soft_hits": 0.0,
            "hard_hits": 0,
            "hit_rate": 0.0,
            "z_score": 0.0,
            "per_block": [],
        }
    embeddings = encode_texts(
        [block_texts[index] for index in active_indices],
        encoder,
        encoder_tokenizer,
        device,
        batch_sz=len(active_indices),
        to_cpu=True,
    ).float()
    projections = pmark_cosine_projections(embeddings, pivots)
    rows = []
    all_soft = []
    all_hard = []
    for embedding_index, block_index in enumerate(active_indices):
        block_projection = projections[embedding_index]
        targets = secret_bits[block_index]
        soft = pmark_soft_matches(
            block_projection,
            targets,
            tolerance=tolerance,
            decay=decay,
        )
        hard = torch.where(targets, block_projection >= 0, block_projection < 0)
        all_soft.extend(float(value) for value in soft.tolist())
        all_hard.extend(bool(value) for value in hard.tolist())
        rows.append(
            {
                "block_id": block_index,
                "target_bits": [bool(value) for value in targets.tolist()],
                "projections": [float(value) for value in block_projection.tolist()],
                "soft_matches": [float(value) for value in soft.tolist()],
                "hard_matches": [bool(value) for value in hard.tolist()],
            }
        )
    n_trials = len(all_soft)
    soft_hits = float(sum(all_soft))
    denominator = math.sqrt(0.25 * n_trials) if n_trials else 0.0
    z_score = (soft_hits - 0.5 * n_trials) / denominator if denominator else 0.0
    return {
        "n_active_blocks": len(active_indices),
        "n_channels": int(pivots.shape[0]),
        "n_trials": n_trials,
        "soft_hits": soft_hits,
        "hard_hits": int(sum(all_hard)),
        "hit_rate": soft_hits / n_trials if n_trials else 0.0,
        "hard_hit_rate": sum(all_hard) / n_trials if n_trials else 0.0,
        "z_score": float(z_score),
        "per_block": rows,
    }


__all__ = [
    "build_pmark_pivots",
    "build_pmark_secret_bits",
    "detect_pmark_blocks",
    "llada_generate_pmark_blocks",
    "pmark_cosine_projections",
    "pmark_soft_matches",
    "select_pmark_candidate",
]
