"""Dream-native primitives for the unit-level reverse-hybrid diagnostic.

The production runner owns candidate scoring.  This module only makes Dream's
native denoising transition resumable and exposes the matched candidate process
so a saved canvas can be advanced under either pi or matched pi0.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from denmark.experiments.trajectory.dream_theory_diagnostic import _native_sample_tokens
from denmark.core.model import (dream_logits, sample_from_logits)


def stable_seed(*parts: int) -> int:
    value = 0x345678
    for part in parts:
        value = (value * 1_000_003 + int(part) + 97) & 0x7FFFFFFF
    return value


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@dataclass
class NativeProposal:
    before: torch.Tensor
    state: torch.Tensor
    logits: torch.Tensor
    native_step: int
    changed_by_block: dict[int, torch.Tensor]


@torch.no_grad()
def native_proposal_step(
    model,
    state: torch.Tensor,
    *,
    native_step: int,
    steps: int,
    eps: float,
    mask_id: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    alg: str,
    alg_temp: float | None,
    prompt_len: int,
    gen_length: int,
    block_size: int,
    seed: int,
) -> NativeProposal:
    """Run exactly one native Dream transfer without a watermark hook."""
    seed_all(seed)
    before = state.clone()
    x = state.clone()
    mask_index = x == int(mask_id)
    logits = dream_logits(model, x)
    if bool(mask_index.any().item()):
        native_sample = _native_sample_tokens(model)
        mask_logits = logits[mask_index]
        schedule = torch.linspace(1, float(eps), int(steps) + 1, device=x.device)
        t, s = schedule[int(native_step)], schedule[int(native_step) + 1]
        kwargs = dict(temperature=temperature, top_p=top_p, top_k=top_k)
        if alg == "origin":
            probability = 1 - s / t if native_step < steps - 1 else 1
            values = torch.full_like(x[mask_index], int(mask_id))
            transfer = torch.rand(*values.shape, device=x.device) < probability
            if bool(transfer.any().item()):
                _, sampled = native_sample(mask_logits[transfer], **kwargs)
                values[transfer] = sampled
            x[mask_index] = values
        else:
            if alg == "maskgit_plus":
                confidence, values = native_sample(mask_logits, **kwargs)
            elif alg == "topk_margin":
                confidence, values = native_sample(mask_logits, margin_confidence=True, **kwargs)
            elif alg == "entropy":
                confidence, values = native_sample(mask_logits, neg_entropy=True, **kwargs)
            else:
                raise ValueError(f"unknown Dream algorithm: {alg}")
            n_mask = mask_index.sum() / mask_index.shape[0]
            n_transfer = int(n_mask * (1 - s / t)) if native_step < steps - 1 else int(n_mask)
            if n_transfer > 0:
                full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype)
                full_confidence[mask_index] = confidence
                if alg_temp is None or float(alg_temp) == 0:
                    _, indices = torch.topk(full_confidence, n_transfer)
                else:
                    indices = torch.multinomial(
                        torch.softmax(full_confidence / float(alg_temp), dim=-1),
                        num_samples=n_transfer,
                    )
                sampled_state = torch.full_like(x, int(mask_id))
                sampled_state[mask_index] = values
                rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(indices)
                x[rows, indices] = sampled_state[rows, indices]

    gen_s, gen_e = int(prompt_len), int(prompt_len + gen_length)
    changed = (before[0, gen_s:gen_e] == int(mask_id)) & (
        x[0, gen_s:gen_e] != int(mask_id)
    )
    changed_positions = changed.nonzero(as_tuple=True)[0] + gen_s
    changed_by_block: dict[int, torch.Tensor] = {}
    if changed_positions.numel():
        block_ids = torch.div(changed_positions - gen_s, int(block_size), rounding_mode="floor")
        for block_id_tensor in block_ids.unique(sorted=True):
            block_id = int(block_id_tensor.item())
            changed_by_block[block_id] = changed_positions[block_ids == block_id_tensor]
    return NativeProposal(before, x, logits, int(native_step), changed_by_block)


@torch.no_grad()
def build_policy_candidates(
    state: torch.Tensor,
    logits: torch.Tensor,
    native_positions: torch.Tensor,
    *,
    block_id: int,
    prompt_len: int,
    gen_length: int,
    block_size: int,
    mask_id: int,
    num_candidates: int,
    update_size: int,
    candidate_temperature: float,
    collapse: bool,
    seed: int,
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, dict]:
    """Match production per-candidate-random-semantic-unit construction."""
    seed_all(seed)
    b_s = int(prompt_len + block_id * block_size)
    b_e = min(b_s + int(block_size), int(prompt_len + gen_length))
    masked = (state[0, b_s:b_e] == int(mask_id)).nonzero(as_tuple=True)[0] + b_s
    pool = torch.cat([native_positions, masked]).unique(sorted=False)
    if not pool.numel():
        raise RuntimeError("candidate pool is empty")
    n_pick = min(int(update_size), int(pool.numel()))
    positions: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    candidates: list[torch.Tensor] = []
    for candidate_idx in range(int(num_candidates)):
        if collapse and candidate_idx:
            pos = positions[0].clone()
            tok = tokens[0].clone()
        else:
            pos = pool[torch.randperm(pool.numel(), device=pool.device)[:n_pick]]
            tok = sample_from_logits(logits[0, pos].float(), float(candidate_temperature))
        candidate = state.clone()
        candidate[0, pos] = tok
        positions.append(pos)
        tokens.append(tok)
        candidates.append(candidate[0])
    stacked = torch.stack(candidates)
    keys = {
        tuple(stacked[index, b_s:b_e].detach().cpu().tolist())
        for index in range(stacked.shape[0])
    }
    meta = {
        "candidate_positions": [
            [int(value) for value in pos.detach().cpu().tolist()] for pos in positions
        ],
        "candidate_positions_in_unit": [
            [int(value - b_s) for value in pos.detach().cpu().tolist()] for pos in positions
        ],
        "candidate_token_ids": [
            [int(value) for value in tok.detach().cpu().tolist()] for tok in tokens
        ],
        "unique_candidate_count": len(keys),
        "candidate_collision_count": int(num_candidates) - len(keys),
        "candidate_collision_rate": (int(num_candidates) - len(keys)) / int(num_candidates),
        "candidate_seed": int(seed),
    }
    return stacked, positions, torch.stack(tokens), meta


def unit_masks(state: torch.Tensor, start: int, end: int, mask_id: int) -> int:
    return int((state[0, int(start):int(end)] == int(mask_id)).sum().item())


@torch.no_grad()
def score_and_commit(
    state: torch.Tensor,
    candidates: torch.Tensor,
    positions: list[torch.Tensor],
    candidate_tokens: torch.Tensor,
    *,
    block_id: int,
    logits: torch.Tensor,
    native_step: int,
    remaining_before: int,
    policy: str,
    hook,
    forced_candidate: int | None = None,
) -> tuple[torch.Tensor, int, dict | None]:
    """Commit a prepared set under pi, matched pi0, or a forced branch."""
    diag = None
    if forced_candidate is not None:
        winner = int(forced_candidate)
    elif policy == "pi":
        winner, diag = hook._score_candidates(
            candidate_tokens,
            state,
            positions,
            int(block_id),
            logits,
            int(native_step),
            int(remaining_before),
        )
    elif policy == "pi0":
        winner = 0
    else:
        raise ValueError(f"unknown policy: {policy}")
    out = state.clone()
    b_s = int(hook.prompt_len + block_id * hook.block_size)
    b_e = min(b_s + hook.block_size, hook.prompt_len + hook.gen_length)
    out[0, b_s:b_e] = candidates[int(winner), b_s:b_e]
    return out, int(winner), diag
