#!/usr/bin/env python3
"""Cumulative within-block mechanism diagnostic for LLaDA semantic watermarking.

This is an independent experiment runner. It reuses the production candidate
construction, semantic encoder, keyed directions, and LongForm prompt loader,
but never mutates the production generation runner or previous outputs.

The expensive continued-policy value is evaluated only for candidate 0 and the
de-duplicated R=1/3/5/10 rollout winners. Alignment outputs are therefore named
``selected_candidate_alignment_error`` rather than max-over-K eta.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denmark.core.generate import build_candidates_random_positions, deduplicate_candidates
from denmark.experiments.trajectory.llada_cumulative_metrics import (
    full_k_theorem_metrics,
    prefix_rollout_metrics,
)
from denmark.core.model import build_directions, encode_texts, resolve_mask_id
R_VALUES = (1, 3, 5, 10)


def build_chat_prompt(input_text, context, tokenizer, max_prompt_chars=0):
    """Build the same user-only chat prompt used by the paper generation runs."""
    user_text = (context + "\n\n" + input_text).strip() if context else input_text.strip()
    if max_prompt_chars and len(user_text) > max_prompt_chars:
        user_text = user_text[-max_prompt_chars:].lstrip()
        truncated = True
    else:
        truncated = False
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return tokenizer.decode(token_ids, skip_special_tokens=False), list(token_ids), truncated


def load_dataset(root, filename, limit, seed):
    """Load a deterministic subset from a WaterBench JSONL file."""
    with (root / filename).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit < len(rows):
        indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
        rows = [rows[index] for index in indices]
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--waterbench_file", type=Path, required=True)
    p.add_argument("--output_jsonl", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, default=None)
    p.add_argument("--model", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--n_prompts", type=int, default=10)
    p.add_argument("--prompt_shard", type=int, default=0)
    p.add_argument("--num_prompt_shards", type=int, default=1)
    p.add_argument("--prompt_indices", nargs="*", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gen_length", type=int, default=300)
    p.add_argument("--block_size", type=int, default=25)
    p.add_argument(
        "--pilot_blocks", type=int, default=12,
        help="Generate this many leading blocks before selecting a reproducible pre-EOS block.",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--candidate_update_size", type=int, default=1)
    p.add_argument("--channels_per_step", type=int, default=2)
    p.add_argument("--base_temperature", type=float, default=0.5)
    p.add_argument("--candidate_temperature", type=float, default=0.6)
    p.add_argument("--rollout_temperature", type=float, default=0.5)
    p.add_argument("--policy_rollouts", type=int, default=3)
    p.add_argument(
        "--policy_rollout_schedule", choices=("constant", "linear_decay"),
        default="linear_decay",
    )
    p.add_argument("--diagnostic_rollouts", type=int, default=10)
    p.add_argument("--continued_repeats", type=int, default=5)
    p.add_argument("--block_level_repeats", type=int, default=10)
    p.add_argument(
        "--block_endpoint_repeats",
        type=int,
        default=None,
        help="Alias for --block_level_repeats; preferred theorem-diagnostic name.",
    )
    p.add_argument(
        "--full_k_audit_prompt_indices",
        nargs="*",
        type=int,
        default=(),
        help="Prompt indices whose selected block evaluates continued Q for all K candidates.",
    )
    p.add_argument("--num_message_bits", type=int, default=2)
    p.add_argument("--direction_seed", type=int, default=42)
    p.add_argument("--message_seed", type=int, default=0)
    p.add_argument("--mask_id", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--continued_branch_batch_size", type=int, default=4,
        help="Batch independent continued-policy branches that share downstream seeds.",
    )
    p.add_argument(
        "--model_forward_batch_size", type=int, default=12,
        help="Maximum candidate rows per LLaDA forward during batched continuations.",
    )
    return p.parse_args()


def stable_seed(*parts: int) -> int:
    value = 0x345678
    for part in parts:
        value = (value * 1_000_003 + int(part) * 9_176 + 97) & 0x7FFFFFFF
    return value


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def completed_prompt_indices(path: Path) -> set[int]:
    done: set[int] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            if line.strip():
                done.add(int(json.loads(line)["prompt_idx"]))
    return done


def policy_rollout_count(
    remaining_before: int, block_size: int, target_rollouts: int, schedule: str
) -> int:
    if schedule == "constant" or block_size <= 1:
        return max(1, target_rollouts)
    raw = 1.0 + 2.0 * (max(1, target_rollouts) - 1) * (
        max(1, remaining_before) - 1
    ) / (block_size - 1)
    return max(1, int(math.floor(raw + 0.5)))


def special_ids(tokenizer, mask_id: int) -> set[int]:
    ids = {int(mask_id)}
    ids.update(int(v) for v in getattr(tokenizer, "all_special_ids", []) if v is not None)
    for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            ids.add(int(value))
    return ids


@torch.no_grad()
def score_completed_block(
    completed: torch.Tensor,
    *,
    prompt_len: int,
    block_id: int,
    block_size: int,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    device: str,
) -> torch.Tensor:
    start = prompt_len + block_id * block_size
    end = start + block_size
    texts = []
    for row in completed.detach().cpu().tolist():
        ids = [int(t) for t in row[start:end] if int(t) not in excluded_ids]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip() or "[empty]")
    embeddings = encode_texts(
        texts, encoder, encoder_tokenizer, device, batch_sz=32, to_cpu=False
    )
    signed = (
        embeddings @ directions[block_id].to(device).T
    ) * signs[block_id].to(device).unsqueeze(0)
    return signed.mean(dim=1).float().cpu()


@torch.no_grad()
def build_step_candidates_from_logits(
    x: torch.Tensor,
    logits: torch.Tensor,
    *,
    prompt_len: int,
    block_id: int,
    block_size: int,
    mask_id: int,
    num_candidates: int,
    update_size: int,
    candidate_temperature: float,
    seed: int,
) -> tuple[torch.Tensor, list[float], list[list[int]], list[list[int]]]:
    start = prompt_len + block_id * block_size
    end = start + block_size
    set_global_seed(seed)
    cands, base_lps = build_candidates_random_positions(
        x,
        logits,
        start,
        end,
        mask_id,
        num_candidates,
        update_size,
        candidate_temperature,
        cand_sampling="gumbel",
        top_p=0.9,
    )
    if cands is None:
        raise RuntimeError("candidate construction returned no candidates")
    positions: list[list[int]] = []
    tokens: list[list[int]] = []
    for k in range(cands.shape[0]):
        changed = (
            (cands[k] != x[0]) & (x[0] == mask_id)
        ).nonzero(as_tuple=True)[0]
        positions.append([int(v - start) for v in changed.tolist()])
        tokens.append([int(cands[k, v].item()) for v in changed.tolist()])
    return cands, [float(v) for v in base_lps], positions, tokens


@torch.no_grad()
def build_step_candidates(
    x: torch.Tensor,
    *,
    model,
    prompt_len: int,
    block_id: int,
    block_size: int,
    mask_id: int,
    num_candidates: int,
    update_size: int,
    candidate_temperature: float,
    seed: int,
) -> tuple[torch.Tensor, list[float], list[list[int]], list[list[int]]]:
    logits = model(x).logits
    return build_step_candidates_from_logits(
        x,
        logits,
        prompt_len=prompt_len,
        block_id=block_id,
        block_size=block_size,
        mask_id=mask_id,
        num_candidates=num_candidates,
        update_size=update_size,
        candidate_temperature=candidate_temperature,
        seed=seed,
    )


@torch.no_grad()
def one_shot_complete_block(
    cands: torch.Tensor,
    *,
    model,
    prompt_len: int,
    block_id: int,
    block_size: int,
    mask_id: int,
    temperature: float,
    seed: int,
    model_batch_size: int | None = None,
) -> torch.Tensor:
    """Complete remaining masks with paired/common Gumbel noise across candidates."""
    start = prompt_len + block_id * block_size
    end = start + block_size
    out = cands.clone()
    batch_size = max(1, int(model_batch_size or len(out)))
    logits = torch.cat([
        model(out[offset : offset + batch_size]).logits[:, start:end].float()
        for offset in range(0, len(out), batch_size)
    ], dim=0)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(seed)
    shared_noise = torch.rand(
        (1, logits.shape[1], logits.shape[2]),
        generator=generator,
        device=logits.device,
        dtype=logits.dtype,
    ).expand(logits.shape[0], -1, -1)
    sampled = (
        logits - torch.log(-torch.log(shared_noise + 1e-20) + 1e-20) * temperature
    ).argmax(dim=-1)
    block = out[:, start:end]
    out[:, start:end] = torch.where(block == mask_id, sampled, block)
    return out


@torch.no_grad()
def rollout_raw_scores(
    cands: torch.Tensor,
    n_rollouts: int,
    *,
    model,
    prompt_len: int,
    block_id: int,
    block_size: int,
    mask_id: int,
    rollout_temperature: float,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    device: str,
    seed_parts: Iterable[int],
    model_batch_size: int | None = None,
) -> torch.Tensor:
    columns = []
    seed_prefix = tuple(int(v) for v in seed_parts)
    for rho in range(n_rollouts):
        completed = one_shot_complete_block(
            cands,
            model=model,
            prompt_len=prompt_len,
            block_id=block_id,
            block_size=block_size,
            mask_id=mask_id,
            temperature=rollout_temperature,
            seed=stable_seed(*seed_prefix, 2, rho),
            model_batch_size=model_batch_size,
        )
        columns.append(
            score_completed_block(
                completed,
                prompt_len=prompt_len,
                block_id=block_id,
                block_size=block_size,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
                device=device,
            )
        )
        del completed
    return torch.stack(columns, dim=1)


def deduplicated_rollout_view(
    cands: torch.Tensor,
    base_lps: list[float],
    *,
    start: int,
    end: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    unique_cands, _unique_lps, inverse, duplicate_count = deduplicate_candidates(
        cands, base_lps, start, end
    )
    return unique_cands, inverse.cpu(), int(duplicate_count)


@torch.no_grad()
def watermark_policy_step(
    x: torch.Tensor,
    *,
    inner_step: int,
    stream_id: int,
    prompt_idx: int,
    args: argparse.Namespace,
    model,
    prompt_len: int,
    block_id: int,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    forced_rollouts: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
    start = prompt_len + block_id * args.block_size
    end = start + args.block_size
    remaining = int((x[0, start:end] == args.mask_id).sum().item())
    n_rollouts = forced_rollouts or policy_rollout_count(
        remaining, args.block_size, args.policy_rollouts, args.policy_rollout_schedule
    )
    prefix = (args.seed, prompt_idx, block_id, stream_id, inner_step)
    cands, base_lps, positions, tokens = build_step_candidates(
        x,
        model=model,
        prompt_len=prompt_len,
        block_id=block_id,
        block_size=args.block_size,
        mask_id=args.mask_id,
        num_candidates=args.num_candidates,
        update_size=args.candidate_update_size,
        candidate_temperature=args.candidate_temperature,
        seed=stable_seed(*prefix, 1),
    )
    unique_cands, inverse, duplicate_count = deduplicated_rollout_view(
        cands, base_lps, start=start, end=end
    )
    unique_raw = rollout_raw_scores(
        unique_cands,
        n_rollouts,
        model=model,
        prompt_len=prompt_len,
        block_id=block_id,
        block_size=args.block_size,
        mask_id=args.mask_id,
        rollout_temperature=args.rollout_temperature,
        tokenizer=tokenizer,
        encoder=encoder,
        encoder_tokenizer=encoder_tokenizer,
        directions=directions,
        signs=signs,
        excluded_ids=excluded_ids,
        device=args.device,
        seed_parts=prefix,
    )
    raw = unique_raw.index_select(0, inverse)
    winner = int(raw.mean(dim=1).argmax().item())
    return cands, raw, winner, {
        "candidate_base_logprobs": base_lps,
        "candidate_positions": positions,
        "candidate_tokens": tokens,
        "policy_rollouts": n_rollouts,
        "candidate_unique_count": int(unique_cands.shape[0]),
        "candidate_duplicate_count": duplicate_count,
    }


@torch.no_grad()
def continue_watermark_policy(
    x: torch.Tensor,
    *,
    start_inner_step: int,
    stream_id: int,
    prompt_idx: int,
    args: argparse.Namespace,
    model,
    prompt_len: int,
    block_id: int,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
) -> tuple[torch.Tensor, int]:
    start = prompt_len + block_id * args.block_size
    end = start + args.block_size
    out = x.clone()
    extra_steps = 0
    for inner in range(start_inner_step, args.block_size + 1):
        if not bool((out[0, start:end] == args.mask_id).any()):
            break
        cands, _raw, winner, _meta = watermark_policy_step(
            out,
            inner_step=inner,
            stream_id=stream_id,
            prompt_idx=prompt_idx,
            args=args,
            model=model,
            prompt_len=prompt_len,
            block_id=block_id,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
        )
        out = cands[winner : winner + 1].clone()
        extra_steps += 1
    if bool((out[0, start:end] == args.mask_id).any()):
        raise RuntimeError("continued watermark policy did not finish selected block")
    return out, extra_steps


@torch.no_grad()
def continue_watermark_policy_batch(
    states: torch.Tensor,
    *,
    start_inner_step: int,
    stream_id: int,
    prompt_idx: int,
    args: argparse.Namespace,
    model,
    prompt_len: int,
    block_id: int,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
) -> tuple[torch.Tensor, list[int]]:
    """Continue several candidate branches under the same downstream RNG stream.

    Branches remain logically independent.  Only deterministic LLaDA forwards,
    semantic encoding, and shared-noise rollouts are batched for throughput.
    """
    if states.ndim != 2 or len(states) < 1:
        raise ValueError("states must have shape [branches, sequence_length]")
    start = prompt_len + block_id * args.block_size
    end = start + args.block_size
    out = states.clone()
    extra_steps = [0 for _ in range(len(out))]
    model_batch = max(1, int(args.model_forward_batch_size))
    for inner in range(start_inner_step, args.block_size + 1):
        active = [
            index for index in range(len(out))
            if bool((out[index, start:end] == args.mask_id).any())
        ]
        if not active:
            break
        remaining = int((out[active[0], start:end] == args.mask_id).sum().item())
        if any(
            int((out[index, start:end] == args.mask_id).sum().item()) != remaining
            for index in active
        ):
            raise RuntimeError("batched continued branches lost synchronized block progress")
        n_rollouts = policy_rollout_count(
            remaining, args.block_size, args.policy_rollouts, args.policy_rollout_schedule
        )
        active_states = out[torch.tensor(active, device=out.device)]
        logits = torch.cat([
            model(active_states[offset : offset + model_batch]).logits
            for offset in range(0, len(active_states), model_batch)
        ], dim=0)
        prefix = (args.seed, prompt_idx, block_id, stream_id, inner)
        branch_candidates: list[torch.Tensor] = []
        branch_unique: list[torch.Tensor] = []
        branch_inverse: list[torch.Tensor] = []
        branch_unique_lengths: list[int] = []
        combined_base_lps: list[float] = []
        for local_index, state in enumerate(active_states):
            cands, base_lps, _positions, _tokens = build_step_candidates_from_logits(
                state.unsqueeze(0),
                logits[local_index : local_index + 1],
                prompt_len=prompt_len,
                block_id=block_id,
                block_size=args.block_size,
                mask_id=args.mask_id,
                num_candidates=args.num_candidates,
                update_size=args.candidate_update_size,
                candidate_temperature=args.candidate_temperature,
                seed=stable_seed(*prefix, 1),
            )
            unique_cands, unique_lps, inverse, _duplicates = deduplicate_candidates(
                cands, base_lps, start, end
            )
            branch_candidates.append(cands)
            branch_unique.append(unique_cands)
            branch_inverse.append(inverse)
            branch_unique_lengths.append(len(unique_cands))
            combined_base_lps.extend(float(value) for value in unique_lps)
        combined = torch.cat(branch_unique, dim=0)
        globally_unique, _global_lps, global_inverse, _global_duplicates = deduplicate_candidates(
            combined, combined_base_lps, start, end
        )
        global_raw = rollout_raw_scores(
            globally_unique,
            n_rollouts,
            model=model,
            prompt_len=prompt_len,
            block_id=block_id,
            block_size=args.block_size,
            mask_id=args.mask_id,
            rollout_temperature=args.rollout_temperature,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
            device=args.device,
            seed_parts=prefix,
        )
        combined_raw = global_raw.index_select(0, global_inverse.to(global_raw.device))
        offset = 0
        for local_index, branch_index in enumerate(active):
            width = branch_unique_lengths[local_index]
            raw_unique = combined_raw[offset : offset + width]
            raw = raw_unique.index_select(
                0, branch_inverse[local_index].to(raw_unique.device)
            )
            winner = int(raw.mean(dim=1).argmax().item())
            out[branch_index] = branch_candidates[local_index][winner]
            extra_steps[branch_index] += 1
            offset += width
        del logits, combined, globally_unique, global_raw, combined_raw
    if bool((out[:, start:end] == args.mask_id).any()):
        raise RuntimeError("batched continued watermark policy did not finish selected block")
    return out, extra_steps


@torch.no_grad()
def continue_reference_policy(
    x: torch.Tensor,
    *,
    start_inner_step: int,
    stream_id: int,
    prompt_idx: int,
    args: argparse.Namespace,
    model,
    prompt_len: int,
    block_id: int,
) -> tuple[torch.Tensor, int]:
    """Matched candidate construction, but always commit candidate index 0."""
    start = prompt_len + block_id * args.block_size
    end = start + args.block_size
    out = x.clone()
    extra_steps = 0
    for inner in range(start_inner_step, args.block_size + 1):
        if not bool((out[0, start:end] == args.mask_id).any()):
            break
        prefix = (args.seed, prompt_idx, block_id, stream_id, inner)
        cands, _base_lps, _positions, _tokens = build_step_candidates(
            out,
            model=model,
            prompt_len=prompt_len,
            block_id=block_id,
            block_size=args.block_size,
            mask_id=args.mask_id,
            num_candidates=args.num_candidates,
            update_size=args.candidate_update_size,
            candidate_temperature=args.candidate_temperature,
            seed=stable_seed(*prefix, 1),
        )
        out = cands[0:1].clone()
        extra_steps += 1
    if bool((out[0, start:end] == args.mask_id).any()):
        raise RuntimeError("matched reference policy did not finish selected block")
    return out, extra_steps


@torch.no_grad()
def pilot_full_generation(
    prompt: torch.Tensor,
    *,
    prompt_idx: int,
    args: argparse.Namespace,
    model,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    prompt_len = int(prompt.shape[1])
    x = torch.full(
        (1, prompt_len + args.gen_length),
        args.mask_id,
        dtype=torch.long,
        device=args.device,
    )
    x[:, :prompt_len] = prompt
    starts: list[torch.Tensor] = []
    n_blocks = min(args.pilot_blocks, args.gen_length // args.block_size)
    for block_id in range(n_blocks):
        starts.append(x.detach().cpu().clone())
        for inner in range(args.block_size):
            start = prompt_len + block_id * args.block_size
            end = start + args.block_size
            if not bool((x[0, start:end] == args.mask_id).any()):
                break
            cands, _raw, winner, _meta = watermark_policy_step(
                x,
                inner_step=inner,
                stream_id=0,
                prompt_idx=prompt_idx,
                args=args,
                model=model,
                prompt_len=prompt_len,
                block_id=block_id,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
            )
            x = cands[winner : winner + 1].clone()
        print(f"  pilot block {block_id + 1}/{n_blocks} complete", flush=True)
    return x, starts


def select_valid_block(
    generated_ids: list[int],
    *,
    prompt_idx: int,
    block_size: int,
    excluded_ids: set[int],
    eos_token_id: int | None,
    tokenizer,
    seed: int,
) -> tuple[int, int | None, list[int]]:
    first_eos = None
    if eos_token_id is not None and eos_token_id in generated_ids:
        first_eos = generated_ids.index(eos_token_id)
    valid = []
    for block_id in range(len(generated_ids) // block_size):
        start = block_id * block_size
        end = start + block_size
        block = generated_ids[start:end]
        if first_eos is not None and end > first_eos:
            continue
        if len(block) != block_size or any(int(v) in excluded_ids for v in block):
            continue
        decoded = tokenizer.decode(block, skip_special_tokens=True).strip()
        if not decoded or len(set(int(v) for v in block)) < 2:
            continue
        valid.append(block_id)
    if not valid:
        raise RuntimeError("pilot trajectory has no complete pre-EOS nondegenerate block")
    # Prefer early/middle body blocks, but choose randomly and reproducibly.
    cutoff = max(1, int(math.ceil(0.75 * len(valid))))
    pool = valid[:cutoff]
    if len(pool) >= 3:
        pool = pool[1:]
    selected = random.Random(seed + prompt_idx * 7_919).choice(pool)
    return int(selected), first_eos, valid


def candidate_diversity(cands: torch.Tensor, start: int, end: int) -> tuple[int, int]:
    unit_states = [tuple(int(v) for v in row[start:end].detach().cpu().tolist()) for row in cands]
    return len(set(unit_states)), len(set(unit_states))


def candidate_representatives(
    cands: torch.Tensor, candidate_indices: Iterable[int], start: int, end: int
) -> tuple[list[int], dict[int, int]]:
    """Deduplicate requested continued-policy branches by exact block state."""
    state_to_rep: dict[tuple[int, ...], int] = {}
    representative_for: dict[int, int] = {}
    representatives: list[int] = []
    for raw_index in candidate_indices:
        index = int(raw_index)
        state = tuple(int(v) for v in cands[index, start:end].detach().cpu().tolist())
        representative = state_to_rep.get(state)
        if representative is None:
            representative = index
            state_to_rep[state] = index
            representatives.append(index)
        representative_for[index] = representative
    return representatives, representative_for


def count_specials(x: torch.Tensor, start: int, end: int, excluded_ids: set[int]) -> int:
    return sum(int(v) in excluded_ids for v in x[0, start:end].detach().cpu().tolist())


def run_prompt(
    prompt_idx: int,
    record: dict,
    *,
    args: argparse.Namespace,
    model,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    cache_root: Path,
) -> dict:
    started = time.time()
    prompt_built = build_chat_prompt(
        record.get("input", ""), record.get("context", ""), tokenizer
    )
    # Main-runner versions in the two clusters return either
    # (prompt_text, prompt_ids) or (prompt_text, prompt_ids, was_truncated).
    # The diagnostic only needs the first two values.
    prompt_text, prompt_ids = prompt_built[:2]
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    prompt_len = int(prompt.shape[1])
    prompt_cache = cache_root / f"prompt_{prompt_idx:03d}"
    pilot_cache = prompt_cache / "pilot_full_trajectory.json"
    if pilot_cache.exists():
        cached_pilot = json.loads(pilot_cache.read_text())
        if cached_pilot.get("prompt_token_ids") != [int(v) for v in prompt_ids]:
            raise RuntimeError(f"pilot prompt mismatch for prompt {prompt_idx}")
        generated = [int(v) for v in cached_pilot["generated_token_ids"]]
        starts = [
            torch.tensor(values, dtype=torch.long).unsqueeze(0)
            for values in cached_pilot["block_start_full_state_token_ids"]
        ]
        print(f"prompt {prompt_idx}: restored complete pilot trajectory", flush=True)
    else:
        print(f"prompt {prompt_idx}: pilot full main-policy trajectory", flush=True)
        pilot, starts = pilot_full_generation(
            prompt,
            prompt_idx=prompt_idx,
            args=args,
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
        )
        generated = pilot[
            0, prompt_len : prompt_len + args.gen_length
        ].detach().cpu().tolist()
        atomic_write_json(
            pilot_cache,
            {
                "prompt_token_ids": [int(v) for v in prompt_ids],
                "generated_token_ids": [int(v) for v in generated],
                "block_start_full_state_token_ids": [
                    [int(v) for v in state[0].tolist()] for state in starts
                ],
            },
        )
    selected_block, first_eos, valid_blocks = select_valid_block(
        generated,
        prompt_idx=prompt_idx,
        block_size=args.block_size,
        excluded_ids=excluded_ids,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
        tokenizer=tokenizer,
        seed=args.seed,
    )
    block_start = starts[selected_block].to(args.device)
    start = prompt_len + selected_block * args.block_size
    end = start + args.block_size
    pilot_block_tokens = generated[
        selected_block * args.block_size : (selected_block + 1) * args.block_size
    ]
    print(
        f"prompt {prompt_idx}: selected block={selected_block}, pre-EOS valid={valid_blocks}",
        flush=True,
    )

    x = block_start.clone()
    steps: list[dict] = []
    for inner in range(args.block_size):
        remaining = int((x[0, start:end] == args.mask_id).sum().item())
        if remaining == 0:
            break
        step_cache = prompt_cache / f"step_{inner:02d}.json"
        partial_cache = prompt_cache / f"step_{inner:02d}.partial.json"
        prefix = (args.seed, prompt_idx, selected_block, 0, inner)
        cands, base_lps, positions, tokens = build_step_candidates(
            x,
            model=model,
            prompt_len=prompt_len,
            block_id=selected_block,
            block_size=args.block_size,
            mask_id=args.mask_id,
            num_candidates=args.num_candidates,
            update_size=args.candidate_update_size,
            candidate_temperature=args.candidate_temperature,
            seed=stable_seed(*prefix, 1),
        )
        if step_cache.exists():
            step_row = json.loads(step_cache.read_text())
            if step_row["candidate_positions"] != positions or step_row["candidate_tokens"] != tokens:
                raise RuntimeError(f"resume candidate mismatch at prompt {prompt_idx} step {inner}")
            expected_full_k = prompt_idx in set(args.full_k_audit_prompt_indices)
            if bool(step_row.get("full_k_theorem")) != expected_full_k:
                raise RuntimeError(
                    f"resume full-K mode mismatch at prompt {prompt_idx} step {inner}"
                )
            actual_winner = int(step_row["actual_policy_winner_zero_based"])
            steps.append(step_row)
            x = cands[actual_winner : actual_winner + 1].clone()
            print(f"  step {inner + 1}/25 restored from cache", flush=True)
            continue

        unique_cands, inverse, duplicate_count = deduplicated_rollout_view(
            cands, base_lps, start=start, end=end
        )
        raw = rollout_raw_scores(
            unique_cands,
            args.diagnostic_rollouts,
            model=model,
            prompt_len=prompt_len,
            block_id=selected_block,
            block_size=args.block_size,
            mask_id=args.mask_id,
            rollout_temperature=args.rollout_temperature,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
            device=args.device,
            seed_parts=prefix,
        )
        raw = raw.index_select(0, inverse)
        raw_np = raw.numpy()
        winners = {
            r: int(raw_np[:, :r].mean(axis=1).argmax()) for r in R_VALUES
        }
        is_full_k_audit = prompt_idx in set(args.full_k_audit_prompt_indices)
        selected_candidates = (
            list(range(args.num_candidates))
            if is_full_k_audit
            else sorted({0, *winners.values()})
        )
        branch_representatives, representative_for = candidate_representatives(
            cands, selected_candidates, start, end
        )
        q_raw: dict[str, list[float | None]] = {
            str(k): [None] * args.continued_repeats for k in selected_candidates
        }
        q_steps: dict[str, list[int | None]] = {
            str(k): [None] * args.continued_repeats for k in selected_candidates
        }
        if partial_cache.exists():
            cached = json.loads(partial_cache.read_text())
            if bool(cached.get("full_k_audit", False)) != is_full_k_audit:
                raise RuntimeError(
                    f"incompatible partial cache at prompt {prompt_idx} step {inner}"
                )
            q_raw.update(cached.get("q_raw", {}))
            q_steps.update(cached.get("q_steps", {}))
        for rep in range(args.continued_repeats):
            stream_id = 100_000 + inner * 100 + rep
            pending = [
                representative for representative in branch_representatives
                if not all(
                    q_raw[str(k)][rep] is not None
                    for k in selected_candidates
                    if representative_for[k] == representative
                )
            ]
            batch_size = max(1, int(args.continued_branch_batch_size))
            for offset in range(0, len(pending), batch_size):
                batch_representatives = pending[offset : offset + batch_size]
                completed, extras = continue_watermark_policy_batch(
                    cands[batch_representatives],
                    start_inner_step=inner + 1,
                    stream_id=stream_id,
                    prompt_idx=prompt_idx,
                    args=args,
                    model=model,
                    prompt_len=prompt_len,
                    block_id=selected_block,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    encoder_tokenizer=encoder_tokenizer,
                    directions=directions,
                    signs=signs,
                    excluded_ids=excluded_ids,
                )
                values = score_completed_block(
                    completed,
                    prompt_len=prompt_len,
                    block_id=selected_block,
                    block_size=args.block_size,
                    tokenizer=tokenizer,
                    encoder=encoder,
                    encoder_tokenizer=encoder_tokenizer,
                    directions=directions,
                    signs=signs,
                    excluded_ids=excluded_ids,
                    device=args.device,
                )
                for local_index, representative in enumerate(batch_representatives):
                    matching = [
                        k for k in selected_candidates
                        if representative_for[k] == representative
                    ]
                    for k in matching:
                        q_raw[str(k)][rep] = float(values[local_index])
                        q_steps[str(k)][rep] = int(extras[local_index])
                atomic_write_json(
                    partial_cache,
                    {
                        "full_k_audit": is_full_k_audit,
                        "q_raw": q_raw,
                        "q_steps": q_steps,
                    },
                )
            print(
                f"  step {inner + 1}/25 continued-pi repeat {rep + 1}/{args.continued_repeats}",
                flush=True,
            )
        q_means = {
            int(k): float(np.mean([float(v) for v in values]))
            for k, values in q_raw.items()
        }
        metrics = prefix_rollout_metrics(raw_np, q_means, R_VALUES)
        full_k_theorem = None
        if is_full_k_audit:
            q_vector = [q_means[k] for k in range(args.num_candidates)]
            full_k_theorem = full_k_theorem_metrics(raw_np, q_vector, R_VALUES)
            if full_k_theorem["max_identity_abs_error"] > 1e-10:
                raise RuntimeError(
                    "continued-policy theorem identity failed: "
                    f"{full_k_theorem['max_identity_abs_error']}"
                )
        actual_r = policy_rollout_count(
            remaining, args.block_size, args.policy_rollouts, args.policy_rollout_schedule
        )
        actual_winner = int(raw_np[:, :actual_r].mean(axis=1).argmax())
        unique_candidates, unique_states = candidate_diversity(cands, start, end)
        state_before = x[0, prompt_len : prompt_len + args.gen_length].detach().cpu().tolist()
        candidate_states = cands[
            :, prompt_len : prompt_len + args.gen_length
        ].detach().cpu().tolist()
        x = cands[actual_winner : actual_winner + 1].clone()
        remaining_after = int((x[0, start:end] == args.mask_id).sum().item())
        step_row = {
            "step": inner,
            "progress_before": float((args.block_size - remaining) / args.block_size),
            "progress": float((args.block_size - remaining_after) / args.block_size),
            "remaining_masks_before": remaining,
            "remaining_masks_after": remaining_after,
            "generation_state_before_token_ids": [int(v) for v in state_before],
            "candidate_state_token_ids": [
                [int(v) for v in candidate] for candidate in candidate_states
            ],
            "candidate_positions": positions,
            "candidate_tokens": tokens,
            "candidate_base_logprobs": base_lps,
            "rollout_raw_scores": raw_np.tolist(),
            "continued_pi_candidate_indices": selected_candidates,
            "continued_pi_unique_branch_representatives": branch_representatives,
            "continued_pi_unique_branch_count": len(branch_representatives),
            "Q_pi_block_raw_scores": q_raw,
            "Q_pi_extra_steps": q_steps,
            "metrics": metrics,
            "actual_policy_rollouts": actual_r,
            "actual_policy_winner_zero_based": actual_winner,
            "num_unique_candidates": unique_candidates,
            "num_unique_semantic_unit_candidate_states": unique_states,
            "candidate_duplicate_count": duplicate_count,
            "rollout_score_range_raw": float(raw_np.max() - raw_np.min()),
            "candidate_score_range_R10": float(raw_np.mean(axis=1).max() - raw_np.mean(axis=1).min()),
            "continued_Q_range": (
                None if full_k_theorem is None else full_k_theorem["Q_pi_range"]
            ),
            "full_k_theorem": full_k_theorem,
            "alignment_scope": (
                "all_K_candidates" if is_full_k_audit
                else "reference_and_deduplicated_R_winners_only"
            ),
        }
        atomic_write_json(step_cache, step_row)
        if partial_cache.exists():
            partial_cache.unlink()
        steps.append(step_row)
        print(
            f"  step {inner + 1}/25 saved; unique={unique_candidates}; actual_R={actual_r}",
            flush=True,
        )

    if len(steps) != args.block_size or bool((x[0, start:end] == args.mask_id).any()):
        raise RuntimeError(f"selected block recorded {len(steps)} decisions, expected {args.block_size}")
    actual_tokens = [int(v) for v in x[0, start:end].detach().cpu().tolist()]
    diagnostic_rng_preserved = actual_tokens == [int(v) for v in pilot_block_tokens]
    if not diagnostic_rng_preserved:
        raise RuntimeError("diagnostic RNG changed the reproduced main trajectory")

    block_cache = prompt_cache / "block_level.json"
    block_level = json.loads(block_cache.read_text()) if block_cache.exists() else {}
    pi_scores: list[float] = list(block_level.get("V_pi_raw_scores", []))
    pi0_scores: list[float] = list(block_level.get("V_pi0_raw_scores", []))
    pi_steps: list[int] = list(block_level.get("pi_completion_steps", []))
    pi0_steps: list[int] = list(block_level.get("pi0_completion_steps", []))
    pi_specials: list[int] = list(block_level.get("pi_special_token_counts", []))
    pi0_specials: list[int] = list(block_level.get("pi0_special_token_counts", []))
    if len(pi_scores) != len(pi0_scores):
        raise RuntimeError("mismatched paired block-level cache")
    for rep in range(len(pi_scores), args.block_level_repeats):
        stream_id = 200_000 + rep
        pi_end, n_pi = continue_watermark_policy(
            block_start,
            start_inner_step=0,
            stream_id=stream_id,
            prompt_idx=prompt_idx,
            args=args,
            model=model,
            prompt_len=prompt_len,
            block_id=selected_block,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
        )
        pi0_end, n_pi0 = continue_reference_policy(
                block_start,
                start_inner_step=0,
                stream_id=stream_id,
                prompt_idx=prompt_idx,
                args=args,
                model=model,
                prompt_len=prompt_len,
                block_id=selected_block,
            )
        pi_scores.append(float(score_completed_block(
                pi_end, prompt_len=prompt_len, block_id=selected_block,
                block_size=args.block_size, tokenizer=tokenizer, encoder=encoder,
                encoder_tokenizer=encoder_tokenizer, directions=directions, signs=signs,
                excluded_ids=excluded_ids, device=args.device,
            )[0]))
        pi0_scores.append(float(score_completed_block(
                pi0_end, prompt_len=prompt_len, block_id=selected_block,
                block_size=args.block_size, tokenizer=tokenizer, encoder=encoder,
                encoder_tokenizer=encoder_tokenizer, directions=directions, signs=signs,
                excluded_ids=excluded_ids, device=args.device,
            )[0]))
        pi_steps.append(n_pi)
        pi0_steps.append(n_pi0)
        pi_specials.append(count_specials(pi_end, start, end, excluded_ids))
        pi0_specials.append(count_specials(pi0_end, start, end, excluded_ids))
        block_level = {
            "V_pi_raw_scores": pi_scores,
            "V_pi0_raw_scores": pi0_scores,
            "pi_completion_steps": pi_steps,
            "pi0_completion_steps": pi0_steps,
            "pi_special_token_counts": pi_specials,
            "pi0_special_token_counts": pi0_specials,
        }
        atomic_write_json(block_cache, block_level)
        print(
            f"  block-level paired repeat {rep + 1}/{args.block_level_repeats}", flush=True
        )
    if len(block_level["V_pi_raw_scores"]) != args.block_level_repeats:
        raise RuntimeError("incomplete block-level cache; remove it and resume")

    cumulative = {f"R{r}": {"gamma": 0.0, "A": 0.0, "selected_error": 0.0, "selection_loss": 0.0} for r in R_VALUES}
    cumulative_rows = []
    running_a = {r: 0.0 for r in R_VALUES}
    running_gamma3 = 0.0
    for step_row in steps:
        for r in R_VALUES:
            metric = step_row["metrics"][f"R{r}"]
            cumulative[f"R{r}"]["gamma"] += float(metric["rollout_gain"])
            cumulative[f"R{r}"]["A"] += float(metric["continued_policy_advantage"])
            cumulative[f"R{r}"]["selected_error"] += float(metric["selected_candidate_alignment_error"])
            cumulative[f"R{r}"]["selection_loss"] += float(metric["empirical_selection_loss_vs_R10"])
            running_a[r] += float(metric["continued_policy_advantage"])
        running_gamma3 += float(step_row["metrics"]["R3"]["rollout_gain"])
        cumulative_rows.append({
            "decision_idx": int(step_row["step"]),
            "block_progress": float(step_row["progress"]),
            "progress": float(step_row["progress"]),
            "A_step_R3": float(step_row["metrics"]["R3"]["continued_policy_advantage"]),
            **{f"cumulative_A_R{r}": float(running_a[r]) for r in R_VALUES},
            "cumulative_A_R10": float(running_a[10]),
            "cumulative_rollout_gain_R3": float(running_gamma3),
            "cumulative_Gamma_R3": float(running_gamma3),
        })

    v_pi = float(np.mean(block_level["V_pi_raw_scores"]))
    v_pi0 = float(np.mean(block_level["V_pi0_raw_scores"]))
    final_uplift = v_pi - v_pi0
    diversity = [float(row["num_unique_candidates"]) for row in steps]
    full_k_audit = None
    if all(step.get("full_k_theorem") is not None for step in steps):
        gamma_sum = float(sum(step["full_k_theorem"]["Gamma_Q_pi"] for step in steps))
        by_r = {}
        for r in R_VALUES:
            epsilon_sum = float(sum(
                step["full_k_theorem"]["by_R"][f"R{r}"]["epsilon_Q_pi"]
                for step in steps
            ))
            advantage_sum = float(sum(
                step["full_k_theorem"]["by_R"][f"R{r}"]["continued_policy_advantage"]
                for step in steps
            ))
            difference_sum = gamma_sum - epsilon_sum
            by_r[f"R{r}"] = {
                "sum_Gamma_Q_pi": gamma_sum,
                "sum_epsilon_Q_pi": epsilon_sum,
                "sum_Gamma_minus_epsilon": difference_sum,
                "sum_A_from_full_K_Q": advantage_sum,
                "identity_abs_error": abs(difference_sum - advantage_sum),
                "mean_step_epsilon": epsilon_sum / len(steps),
                "positive_sum": bool(difference_sum > 0),
            }
        full_k_audit = {
            "num_steps": len(steps),
            "sum_Gamma_Q_pi": gamma_sum,
            "by_R": by_r,
            "max_step_identity_abs_error": float(max(
                step["full_k_theorem"]["max_identity_abs_error"] for step in steps
            )),
            "max_block_identity_abs_error": float(max(
                values["identity_abs_error"] for values in by_r.values()
            )),
        }
    block_summary = {
        **{f"sum_Gamma_R{r}": cumulative[f"R{r}"]["gamma"] for r in R_VALUES},
        **{f"sum_A_R{r}": cumulative[f"R{r}"]["A"] for r in R_VALUES},
        **{f"sum_selected_alignment_error_R{r}": cumulative[f"R{r}"]["selected_error"] for r in R_VALUES},
        **{f"empirical_selection_loss_R{r}": cumulative[f"R{r}"]["selection_loss"] for r in R_VALUES},
        "V_pi": v_pi,
        "V_pi0": v_pi0,
        "final_block_uplift": final_uplift,
        "mean_candidate_diversity": float(np.mean(diversity)),
        "fraction_candidate_collapse_steps": float(np.mean(np.asarray(diversity) <= 1.0)),
        "actual_final_block_score": float(score_completed_block(
            x, prompt_len=prompt_len, block_id=selected_block, block_size=args.block_size,
            tokenizer=tokenizer, encoder=encoder, encoder_tokenizer=encoder_tokenizer,
            directions=directions, signs=signs, excluded_ids=excluded_ids, device=args.device,
        )[0]),
        "full_k_theorem_audit": full_k_audit,
    }
    result = {
        "schema_version": 2,
        "dataset": "longform_qa",
        "prompt_idx": prompt_idx,
        "prompt_input": record.get("input", ""),
        "prompt_context": record.get("context", ""),
        "prompt_full": prompt_text,
        "prompt_token_ids": [int(v) for v in prompt_ids],
        "selected_block_id": selected_block,
        "block_start": selected_block * args.block_size,
        "block_end": (selected_block + 1) * args.block_size,
        "block_token_range_generation_zero_based": [
            selected_block * args.block_size,
            (selected_block + 1) * args.block_size,
        ],
        "selected_block_text_main_trajectory": tokenizer.decode(
            pilot_block_tokens, skip_special_tokens=True
        ).strip(),
        "first_eos_generation_index": first_eos,
        "pilot_generated_blocks": args.pilot_blocks,
        "valid_pre_eos_block_ids": valid_blocks,
        "block_start_state_token_ids": [
            int(v) for v in block_start[0, prompt_len : prompt_len + args.gen_length].cpu().tolist()
        ],
        "num_block_decisions": len(steps),
        "steps": steps,
        "cumulative_trajectory": cumulative_rows,
        "block_level": block_level,
        "block_summary": block_summary,
        "sanity": {
            "selected_block_before_first_eos": first_eos is None or (selected_block + 1) * args.block_size <= first_eos,
            "selected_block_no_special_or_mask": not any(v in excluded_ids for v in actual_tokens),
            "recorded_all_25_decisions": len(steps) == args.block_size,
            "selected_block_fully_resolved": not any(v == args.mask_id for v in actual_tokens),
            "continued_pi_hook_enabled": True,
            "pi0_keyed_selection_disabled": True,
            "diagnostic_rng_preserved_main_trajectory": diagnostic_rng_preserved,
            "all_candidates_same_parent_state": True,
            "rollout_prefixes_reused": True,
            "raw_values_complete": all(
                np.asarray(row["rollout_raw_scores"]).shape == (
                    args.num_candidates, args.diagnostic_rollouts
                )
                for row in steps
            ),
            "candidate_states_complete": all(
                np.asarray(row["candidate_state_token_ids"]).shape == (
                    args.num_candidates, args.gen_length
                )
                for row in steps
            ),
            "continued_branches_stop_exactly_at_block_end": all(
                int(extra) == int(row["remaining_masks_after"])
                for row in steps
                for values in row["Q_pi_extra_steps"].values()
                for extra in values
            ),
            "full_k_identity_checked": (
                full_k_audit is None
                or full_k_audit["max_block_identity_abs_error"] <= 1e-10
            ),
        },
        "config": {
            "execution_host": os.uname().nodename,
            "cuda_device_name": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
            "model": args.model,
            "encoder": args.encoder,
            "generation_length": args.gen_length,
            "semantic_block_size": args.block_size,
            "pilot_blocks": args.pilot_blocks,
            "K": args.num_candidates,
            "candidate_update_size": args.candidate_update_size,
            "channels_per_step": args.channels_per_step,
            "base_temperature": args.base_temperature,
            "candidate_temperature": args.candidate_temperature,
            "rollout_temperature": args.rollout_temperature,
            "diagnostic_rollouts": args.diagnostic_rollouts,
            "R_values": list(R_VALUES),
            "continued_repeats": args.continued_repeats,
            "block_level_repeats": args.block_level_repeats,
            "block_endpoint_repeats": args.block_level_repeats,
            "full_k_audit": full_k_audit is not None,
            "full_k_audit_prompt_indices": list(args.full_k_audit_prompt_indices),
            "policy_rollouts_target_average": args.policy_rollouts,
            "policy_rollout_schedule": args.policy_rollout_schedule,
            "selector": "max_watermark",
            "candidate_policy": "per_candidate_random_position_gumbel",
            "deduplicated_expensive_Q_candidates": True,
            "alignment_metric_scope": "selected_candidate_not_max_over_K",
            "shared_rollout_seeds_across_candidates": True,
            "paired_downstream_seeds": True,
            "direction_seed": args.direction_seed,
            "message_seed": args.message_seed,
            "num_message_bits": args.num_message_bits,
            "DLM_MIN_COMPLETION_TOKENS": os.environ.get("DLM_MIN_COMPLETION_TOKENS"),
            "DLM_MAX_COMPLETION_RETRIES": os.environ.get("DLM_MAX_COMPLETION_RETRIES"),
            "max_retries": 0,
        },
        "elapsed_minutes": (time.time() - started) / 60.0,
    }
    return result


def main() -> None:
    args = parse_args()
    if args.block_endpoint_repeats is not None:
        args.block_level_repeats = int(args.block_endpoint_repeats)
    if args.gen_length % args.block_size:
        raise ValueError("gen_length must be divisible by block_size")
    if args.pilot_blocks * args.block_size < args.gen_length:
        raise ValueError(
            "pilot_blocks must cover the complete generation length so first EOS "
            "and all valid正文 blocks are known"
        )
    if args.candidate_update_size != 1:
        raise ValueError("this diagnostic currently requires candidate_update_size=1")
    if args.diagnostic_rollouts < max(R_VALUES):
        raise ValueError("diagnostic_rollouts must be at least 10")
    if args.continued_repeats < 1 or args.block_level_repeats < 1:
        raise ValueError("continued and endpoint repeat counts must be positive")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir or args.output_jsonl.parent / (args.output_jsonl.stem + "_cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(
        args.waterbench_file.parent,
        args.waterbench_file.name,
        args.n_prompts,
        args.seed,
    )
    indexed = [
        (i, row) for i, row in enumerate(rows)
        if i % args.num_prompt_shards == args.prompt_shard
    ]
    if args.prompt_indices is not None:
        wanted = set(args.prompt_indices)
        indexed = [(i, row) for i, row in enumerate(rows) if i in wanted]
    done = completed_prompt_indices(args.output_jsonl)

    print(f"Loading LLaDA: {args.model}", flush=True)
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    args.mask_id = resolve_mask_id(tokenizer, "llada", args.mask_id)
    print(f"Loading semantic encoder: {args.encoder}", flush=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(
        args.encoder, torch_dtype=torch.float32
    ).to(args.device).eval()
    n_blocks = args.gen_length // args.block_size
    directions, signs = build_directions(
        n_blocks, args.num_message_bits, args.direction_seed, args.message_seed
    )
    excluded_ids = special_ids(tokenizer, args.mask_id)

    with args.output_jsonl.open("a") as out_f:
        for prompt_idx, record in indexed:
            if prompt_idx in done:
                print(f"prompt {prompt_idx}: already complete", flush=True)
                continue
            result = run_prompt(
                prompt_idx,
                record,
                args=args,
                model=model,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
                cache_root=cache_root,
            )
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            done.add(prompt_idx)
            print(
                f"prompt {prompt_idx}: complete in {result['elapsed_minutes']:.1f} min -> {args.output_jsonl}",
                flush=True,
            )


if __name__ == "__main__":
    main()
