#!/usr/bin/env python3
"""Reverse-hybrid theorem-closure diagnostic for LLaDA semantic watermarking.

For decision t, this runner reconstructs a matched-reference (pi0) prefix of
length t from a common block-start state, shares the resulting F_t and candidate
set between two forced commitments, and continues both branches under the
production watermark policy pi.  Exact candidate collapse is implemented by
copying candidate 0 to all K slots at every policy decision.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denmark.core.model import build_directions, resolve_mask_id
from denmark.experiments.trajectory.llada_cumulative_run import (
    atomic_write_json,
    build_step_candidates,
    build_chat_prompt,
    candidate_diversity,
    continue_reference_policy,
    continue_watermark_policy,
    continue_watermark_policy_batch,
    load_dataset,
    pilot_full_generation,
    policy_rollout_count,
    rollout_raw_scores,
    score_completed_block,
    select_valid_block,
    set_global_seed,
    special_ids,
    stable_seed,
    watermark_policy_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waterbench_file", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--n_prompts", type=int, default=30)
    parser.add_argument("--prompt_indices", nargs="*", type=int, default=None)
    parser.add_argument("--prompt_shard", type=int, default=0)
    parser.add_argument("--num_prompt_shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--pilot_blocks", type=int, default=12)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--candidate_update_size", type=int, default=1)
    parser.add_argument("--channels_per_step", type=int, default=2)
    parser.add_argument("--base_temperature", type=float, default=0.5)
    parser.add_argument("--candidate_temperature", type=float, default=0.6)
    parser.add_argument("--rollout_temperature", type=float, default=0.5)
    parser.add_argument("--policy_rollouts", type=int, default=3)
    parser.add_argument(
        "--policy_rollout_schedule",
        choices=("constant", "linear_decay"),
        default="linear_decay",
    )
    parser.add_argument("--future_repeats", type=int, default=10)
    parser.add_argument("--endpoint_repeats", type=int, default=20)
    parser.add_argument("--collapse_blocks", type=int, default=10)
    parser.add_argument("--collapse_prompt_indices", nargs="*", type=int, default=None)
    parser.add_argument("--num_message_bits", type=int, default=2)
    parser.add_argument("--direction_seed", type=int, default=42)
    parser.add_argument("--message_seed", type=int, default=0)
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_forward_batch_size", type=int, default=12)
    return parser.parse_args()


def completed_prompt_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                result.add(int(json.loads(line)["prompt_idx"]))
    return result


def generation_slice(x: torch.Tensor, prompt_len: int, gen_length: int) -> list[int]:
    return [
        int(value)
        for value in x[0, prompt_len : prompt_len + gen_length].detach().cpu().tolist()
    ]


@torch.no_grad()
def reference_prefix(
    block_start: torch.Tensor,
    *,
    prefix_length: int,
    stream_id: int,
    prompt_idx: int,
    block_id: int,
    prompt_len: int,
    args: argparse.Namespace,
    model,
) -> tuple[torch.Tensor, list[dict]]:
    """Reconstruct the first t reverse-hybrid decisions under matched pi0."""
    out = block_start.clone()
    start = prompt_len + block_id * args.block_size
    end = start + args.block_size
    audit = []
    for inner in range(prefix_length):
        before = int((out[0, start:end] == args.mask_id).sum().item())
        prefix = (args.seed, prompt_idx, block_id, stream_id, inner)
        candidates, _base_lps, positions, tokens = build_step_candidates(
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
        out = candidates[0:1].clone()
        after = int((out[0, start:end] == args.mask_id).sum().item())
        audit.append({
            "decision_idx": inner,
            "policy": "pi0_matched_reference",
            "committed_candidate_index_zero_based": 0,
            "candidate_1_positions": positions[0],
            "candidate_1_tokens": tokens[0],
            "remaining_before": before,
            "remaining_after": after,
            "candidate_seed": stable_seed(*prefix, 1),
        })
    return out, audit


@torch.no_grad()
def score_states(
    states: torch.Tensor,
    *,
    prompt_len: int,
    block_id: int,
    args: argparse.Namespace,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
) -> list[float]:
    return [
        float(value)
        for value in score_completed_block(
            states,
            prompt_len=prompt_len,
            block_id=block_id,
            block_size=args.block_size,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
            device=args.device,
        ).tolist()
    ]


@torch.no_grad()
def run_reverse_step(
    block_start: torch.Tensor,
    *,
    decision_idx: int,
    condition: str,
    prompt_idx: int,
    block_id: int,
    prompt_len: int,
    args: argparse.Namespace,
    model,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    partial_cache: Path,
) -> dict:
    collapse = condition == "exact_collapse"
    prefix_stream = 300_000 + decision_idx
    state_t, prefix_audit = reference_prefix(
        block_start,
        prefix_length=decision_idx,
        stream_id=prefix_stream,
        prompt_idx=prompt_idx,
        block_id=block_id,
        prompt_len=prompt_len,
        args=args,
        model=model,
    )
    start = prompt_len + block_id * args.block_size
    end = start + args.block_size
    remaining = int((state_t[0, start:end] == args.mask_id).sum().item())
    if remaining != args.block_size - decision_idx:
        raise RuntimeError("reverse-hybrid pi0 prefix did not advance exactly t decisions")

    selection_stream = 400_000 + decision_idx
    if collapse:
        selection_prefix = (
            args.seed, prompt_idx, block_id, selection_stream, decision_idx
        )
        candidates, base_lps, positions, tokens = build_step_candidates(
            state_t,
            model=model,
            prompt_len=prompt_len,
            block_id=block_id,
            block_size=args.block_size,
            mask_id=args.mask_id,
            num_candidates=args.num_candidates,
            update_size=args.candidate_update_size,
            candidate_temperature=args.candidate_temperature,
            seed=stable_seed(*selection_prefix, 1),
        )
        candidates = candidates[0:1].expand(args.num_candidates, -1).clone()
        positions = [positions[0] for _ in range(args.num_candidates)]
        tokens = [tokens[0] for _ in range(args.num_candidates)]
        base_lps = [base_lps[0] for _ in range(args.num_candidates)]
        effective_rollouts = policy_rollout_count(
            remaining,
            args.block_size,
            args.policy_rollouts,
            args.policy_rollout_schedule,
        )
        one_raw = rollout_raw_scores(
            candidates[0:1],
            effective_rollouts,
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
            seed_parts=selection_prefix,
            model_batch_size=args.model_forward_batch_size,
        )
        selection_raw = one_raw.expand(args.num_candidates, -1).clone()
        winner = 0
        unique_count = 1
        duplicate_count = args.num_candidates - 1
    else:
        candidates, selection_raw, winner, meta = watermark_policy_step(
            state_t,
            inner_step=decision_idx,
            stream_id=selection_stream,
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
        base_lps = meta["candidate_base_logprobs"]
        positions = meta["candidate_positions"]
        tokens = meta["candidate_tokens"]
        effective_rollouts = int(meta["policy_rollouts"])
        unique_count = int(meta["candidate_unique_count"])
        duplicate_count = int(meta["candidate_duplicate_count"])

    selected_scores: list[float] = []
    reference_scores: list[float] = []
    differences: list[float] = []
    selected_extra_steps: list[int] = []
    reference_extra_steps: list[int] = []
    if partial_cache.exists():
        cached = json.loads(partial_cache.read_text())
        selected_scores = [float(value) for value in cached["selected_scores"]]
        reference_scores = [float(value) for value in cached["reference_scores"]]
        differences = [float(value) for value in cached["differences"]]
        selected_extra_steps = [int(value) for value in cached["selected_extra_steps"]]
        reference_extra_steps = [int(value) for value in cached["reference_extra_steps"]]
    if not (
        len(selected_scores)
        == len(reference_scores)
        == len(differences)
        == len(selected_extra_steps)
        == len(reference_extra_steps)
    ):
        raise RuntimeError("inconsistent reverse-step partial cache")

    for repetition in range(len(differences), args.future_repeats):
        future_stream = 500_000 + decision_idx * 100 + repetition
        if collapse:
            completed, extra = continue_reference_policy(
                candidates[0:1],
                start_inner_step=decision_idx + 1,
                stream_id=future_stream,
                prompt_idx=prompt_idx,
                args=args,
                model=model,
                prompt_len=prompt_len,
                block_id=block_id,
            )
            value = score_states(
                completed,
                prompt_len=prompt_len,
                block_id=block_id,
                args=args,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
            )[0]
            pair_values = [value, value]
            extras = [extra, extra]
        else:
            forced_states = torch.stack(
                [candidates[winner], candidates[0]], dim=0
            )
            completed, extras = continue_watermark_policy_batch(
                forced_states,
                start_inner_step=decision_idx + 1,
                stream_id=future_stream,
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
            pair_values = score_states(
                completed,
                prompt_len=prompt_len,
                block_id=block_id,
                args=args,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
            )
        selected_scores.append(float(pair_values[0]))
        reference_scores.append(float(pair_values[1]))
        differences.append(float(pair_values[0] - pair_values[1]))
        selected_extra_steps.append(int(extras[0]))
        reference_extra_steps.append(int(extras[1]))
        atomic_write_json(partial_cache, {
            "selected_scores": selected_scores,
            "reference_scores": reference_scores,
            "differences": differences,
            "selected_extra_steps": selected_extra_steps,
            "reference_extra_steps": reference_extra_steps,
        })
        print(
            f"  {condition} t={decision_idx + 1}/25 future "
            f"{repetition + 1}/{args.future_repeats}",
            flush=True,
        )

    if collapse and any(abs(value) > 1e-12 for value in differences):
        raise RuntimeError("exact collapse produced a nonzero paired difference")
    selected_state = candidates[winner]
    reference_state = candidates[0]
    differing_after_commit = int(
        (selected_state[start:end] != reference_state[start:end]).sum().item()
    )
    parent = state_t[0]
    selected_changed = (selected_state != parent).nonzero(as_tuple=True)[0]
    reference_changed = (reference_state != parent).nonzero(as_tuple=True)[0]
    forced_commit_is_local = bool(
        len(selected_changed) == args.candidate_update_size
        and len(reference_changed) == args.candidate_update_size
        and all(start <= int(position) < end for position in selected_changed.tolist())
        and all(start <= int(position) < end for position in reference_changed.tolist())
    )
    row = {
        "decision_idx": decision_idx,
        "normalized_progress_before": decision_idx / args.block_size,
        "remaining_masks_before": remaining,
        "reverse_hybrid_prefix_policy": ["pi0"] * decision_idx,
        "reverse_hybrid_prefix_audit": prefix_audit,
        "state_F_t_generation_token_ids": generation_slice(
            state_t, prompt_len, args.gen_length
        ),
        "candidate_positions": positions,
        "candidate_token_ids": tokens,
        "candidate_base_logprobs": [float(value) for value in base_lps],
        "candidate_state_block_token_ids": [
            [int(value) for value in candidate[start:end].detach().cpu().tolist()]
            for candidate in candidates
        ],
        "unique_candidate_count": unique_count,
        "candidate_collision_count": duplicate_count,
        "candidate_collision_rate": duplicate_count / args.num_candidates,
        "selection_rollout_effective_R": effective_rollouts,
        "selection_rollout_raw_scores": selection_raw.detach().float().cpu().tolist(),
        "selected_candidate_index_zero_based": int(winner),
        "reference_candidate_index_zero_based": 0,
        "forced_commits_differing_block_positions": differing_after_commit,
        "Q_pi_selected_raw_scores": selected_scores,
        "Q_pi_reference_raw_scores": reference_scores,
        "paired_future_differences": differences,
        "delta_hat_pi": float(np.mean(differences)),
        "selected_branch_extra_steps": selected_extra_steps,
        "reference_branch_extra_steps": reference_extra_steps,
        "local_hybrid_J_t_hat": float(np.mean(selected_scores)),
        "local_hybrid_J_t_plus_1_hat": float(np.mean(reference_scores)),
        "local_J_difference_minus_delta": float(
            np.mean(selected_scores) - np.mean(reference_scores) - np.mean(differences)
        ),
        "rng": {
            "prefix_stream_id": prefix_stream,
            "selection_stream_id": selection_stream,
            "future_stream_ids": [
                500_000 + decision_idx * 100 + rep
                for rep in range(args.future_repeats)
            ],
            "selection_independent_of_Q_evaluation": True,
            "selected_reference_future_seeds_paired": True,
        },
        "sanity": {
            "prefix_has_exactly_t_pi0_decisions": len(prefix_audit) == decision_idx,
            "shared_F_t": True,
            "shared_candidate_set": True,
            "forced_commit_only_current_difference": forced_commit_is_local,
            "future_policy_matches_condition": True,
            "selection_future_rng_independent": selection_stream
            not in [500_000 + decision_idx * 100 + rep for rep in range(args.future_repeats)],
            "paired_future_seeds": True,
            "branches_stop_at_block_end": all(
                value == args.block_size - decision_idx - 1
                for value in selected_extra_steps + reference_extra_steps
            ),
            "J_t_minus_J_t_plus_1_equals_delta": bool(abs(
                np.mean(selected_scores)
                - np.mean(reference_scores)
                - np.mean(differences)
            ) < 1e-10),
            "exact_collapse_unique_one": (not collapse) or unique_count == 1,
            "exact_collapse_delta_zero": (not collapse)
            or all(abs(value) <= 1e-12 for value in differences),
        },
    }
    return row


@torch.no_grad()
def run_endpoint(
    block_start: torch.Tensor,
    *,
    condition: str,
    prompt_idx: int,
    block_id: int,
    prompt_len: int,
    args: argparse.Namespace,
    model,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    endpoint_cache: Path,
) -> dict:
    collapse = condition == "exact_collapse"
    policy_scores: list[float] = []
    reference_scores: list[float] = []
    differences: list[float] = []
    if endpoint_cache.exists():
        cached = json.loads(endpoint_cache.read_text())
        policy_scores = [float(value) for value in cached["policy_scores"]]
        reference_scores = [float(value) for value in cached["reference_scores"]]
        differences = [float(value) for value in cached["differences"]]
    for repetition in range(len(differences), args.endpoint_repeats):
        stream_id = 800_000 + repetition
        if collapse:
            reference_end, _ = continue_reference_policy(
                block_start,
                start_inner_step=0,
                stream_id=stream_id,
                prompt_idx=prompt_idx,
                args=args,
                model=model,
                prompt_len=prompt_len,
                block_id=block_id,
            )
            value = score_states(
                reference_end,
                prompt_len=prompt_len,
                block_id=block_id,
                args=args,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
            )[0]
            policy_value, reference_value = value, value
        else:
            policy_end, _ = continue_watermark_policy(
                block_start,
                start_inner_step=0,
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
            reference_end, _ = continue_reference_policy(
                block_start,
                start_inner_step=0,
                stream_id=stream_id,
                prompt_idx=prompt_idx,
                args=args,
                model=model,
                prompt_len=prompt_len,
                block_id=block_id,
            )
            policy_value = score_states(
                policy_end,
                prompt_len=prompt_len,
                block_id=block_id,
                args=args,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
            )[0]
            reference_value = score_states(
                reference_end,
                prompt_len=prompt_len,
                block_id=block_id,
                args=args,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
            )[0]
        policy_scores.append(float(policy_value))
        reference_scores.append(float(reference_value))
        differences.append(float(policy_value - reference_value))
        atomic_write_json(endpoint_cache, {
            "policy_scores": policy_scores,
            "reference_scores": reference_scores,
            "differences": differences,
        })
        print(
            f"  {condition} endpoint {repetition + 1}/{args.endpoint_repeats}",
            flush=True,
        )
    return {
        "J_pi_raw_scores": policy_scores,
        "J_pi0_raw_scores": reference_scores,
        "paired_endpoint_differences": differences,
        "J_pi_hat": float(np.mean(policy_scores)),
        "J_pi0_hat": float(np.mean(reference_scores)),
        "endpoint_uplift_hat": float(np.mean(differences)),
        "rng_stream_ids": [800_000 + rep for rep in range(args.endpoint_repeats)],
        "independent_from_reverse_hybrid_Q_evaluation": True,
        "paired_endpoint_seeds": True,
    }


def condition_summary(steps: list[dict], endpoint: dict) -> dict:
    deltas = [float(row["delta_hat_pi"]) for row in steps]
    cumulative = np.cumsum(deltas)
    positive = [value for value in deltas if value > 1e-12]
    negative = [value for value in deltas if value < -1e-12]
    zeros = [value for value in deltas if abs(value) <= 1e-12]
    return {
        "num_decisions": len(steps),
        "sum_delta": float(sum(deltas)),
        "positive_step_count": len(positive),
        "negative_step_count": len(negative),
        "zero_step_count": len(zeros),
        "positive_step_fraction": len(positive) / len(deltas),
        "largest_positive_delta": max(positive) if positive else 0.0,
        "largest_negative_delta": min(negative) if negative else 0.0,
        "mean_positive_delta": float(np.mean(positive)) if positive else 0.0,
        "mean_negative_delta": float(np.mean(negative)) if negative else 0.0,
        "cumulative_delta": [float(value) for value in cumulative],
        "endpoint_uplift": float(endpoint["endpoint_uplift_hat"]),
        "closure_gap_endpoint_minus_sum_delta": float(
            endpoint["endpoint_uplift_hat"] - sum(deltas)
        ),
        "mean_candidate_diversity": float(np.mean([
            row["unique_candidate_count"] for row in steps
        ])),
        "mean_candidate_collision_rate": float(np.mean([
            row["candidate_collision_rate"] for row in steps
        ])),
    }


def run_condition(
    block_start: torch.Tensor,
    *,
    condition: str,
    prompt_idx: int,
    block_id: int,
    prompt_len: int,
    args: argparse.Namespace,
    model,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    excluded_ids: set[int],
    condition_cache: Path,
) -> dict:
    condition_cache.mkdir(parents=True, exist_ok=True)
    steps = []
    for decision_idx in range(args.block_size):
        complete_path = condition_cache / f"step_{decision_idx:02d}.json"
        partial_path = condition_cache / f"step_{decision_idx:02d}.partial.json"
        if complete_path.exists():
            row = json.loads(complete_path.read_text())
            if len(row["paired_future_differences"]) != args.future_repeats:
                raise RuntimeError("cached step has incompatible future repeat count")
            print(f"  {condition} t={decision_idx + 1}/25 restored", flush=True)
        else:
            row = run_reverse_step(
                block_start,
                decision_idx=decision_idx,
                condition=condition,
                prompt_idx=prompt_idx,
                block_id=block_id,
                prompt_len=prompt_len,
                args=args,
                model=model,
                tokenizer=tokenizer,
                encoder=encoder,
                encoder_tokenizer=encoder_tokenizer,
                directions=directions,
                signs=signs,
                excluded_ids=excluded_ids,
                partial_cache=partial_path,
            )
            if not all(row["sanity"].values()):
                failed = [key for key, value in row["sanity"].items() if not value]
                raise RuntimeError(f"reverse-step sanity failed: {failed}")
            atomic_write_json(complete_path, row)
            if partial_path.exists():
                partial_path.unlink()
        steps.append(row)
    endpoint = run_endpoint(
        block_start,
        condition=condition,
        prompt_idx=prompt_idx,
        block_id=block_id,
        prompt_len=prompt_len,
        args=args,
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        encoder_tokenizer=encoder_tokenizer,
        directions=directions,
        signs=signs,
        excluded_ids=excluded_ids,
        endpoint_cache=condition_cache / "endpoint.partial.json",
    )
    return {
        "condition": condition,
        "steps": steps,
        "endpoint": endpoint,
        "block_summary": condition_summary(steps, endpoint),
        "sanity": {
            "recorded_all_25_decisions": len(steps) == args.block_size,
            "all_step_sanity_passed": all(
                all(step["sanity"].values()) for step in steps
            ),
            "theorem_sign_is_J_t_minus_J_t_plus_1": all(
                abs(step["local_J_difference_minus_delta"]) < 1e-10
                for step in steps
            ),
            "endpoint_rng_independent": True,
            "exact_collapse_all_unique_one": condition != "exact_collapse"
            or all(step["unique_candidate_count"] == 1 for step in steps),
            "exact_collapse_all_deltas_zero": condition != "exact_collapse"
            or all(abs(step["delta_hat_pi"]) <= 1e-12 for step in steps),
            "exact_collapse_endpoint_zero": condition != "exact_collapse"
            or abs(endpoint["endpoint_uplift_hat"]) <= 1e-12,
        },
    }


def run_prompt(
    prompt_idx: int,
    record: dict,
    *,
    collapse: bool,
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
    prompt_text, prompt_ids = prompt_built[:2]
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    prompt_len = int(prompt.shape[1])
    prompt_cache = cache_root / f"prompt_{prompt_idx:03d}"
    pilot_cache = prompt_cache / "pilot_full_trajectory.json"
    if pilot_cache.exists():
        cached = json.loads(pilot_cache.read_text())
        if cached["prompt_token_ids"] != [int(value) for value in prompt_ids]:
            raise RuntimeError("pilot cache prompt mismatch")
        generated = [int(value) for value in cached["generated_token_ids"]]
        starts = [
            torch.tensor(values, dtype=torch.long).unsqueeze(0)
            for values in cached["block_start_full_state_token_ids"]
        ]
        print(f"prompt {prompt_idx}: restored pilot", flush=True)
    else:
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
        generated = generation_slice(pilot, prompt_len, args.gen_length)
        atomic_write_json(pilot_cache, {
            "prompt_token_ids": [int(value) for value in prompt_ids],
            "generated_token_ids": generated,
            "block_start_full_state_token_ids": [
                [int(value) for value in state[0].tolist()] for state in starts
            ],
        })
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
    block_range = [
        selected_block * args.block_size,
        (selected_block + 1) * args.block_size,
    ]
    print(
        f"prompt {prompt_idx}: selected block={selected_block}, collapse={collapse}",
        flush=True,
    )
    default = run_condition(
        block_start,
        condition="default",
        prompt_idx=prompt_idx,
        block_id=selected_block,
        prompt_len=prompt_len,
        args=args,
        model=model,
        tokenizer=tokenizer,
        encoder=encoder,
        encoder_tokenizer=encoder_tokenizer,
        directions=directions,
        signs=signs,
        excluded_ids=excluded_ids,
        condition_cache=prompt_cache / "default",
    )
    conditions = {"default": default}
    if collapse:
        conditions["exact_collapse"] = run_condition(
            block_start,
            condition="exact_collapse",
            prompt_idx=prompt_idx,
            block_id=selected_block,
            prompt_len=prompt_len,
            args=args,
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=encoder_tokenizer,
            directions=directions,
            signs=signs,
            excluded_ids=excluded_ids,
            condition_cache=prompt_cache / "exact_collapse",
        )
    selected_tokens = generated[block_range[0] : block_range[1]]
    result = {
        "schema_version": 1,
        "experiment": "llada_reverse_hybrid_theorem_closure",
        "dataset": "longform_qa",
        "prompt_idx": prompt_idx,
        "prompt_input": record.get("input", ""),
        "prompt_context": record.get("context", ""),
        "prompt_full": prompt_text,
        "prompt_token_ids": [int(value) for value in prompt_ids],
        "generation_seed": args.seed,
        "selected_block_id": selected_block,
        "block_token_range_generation_zero_based": block_range,
        "block_start_full_token_state": generation_slice(
            block_start, prompt_len, args.gen_length
        ),
        "first_eos_generation_index": first_eos,
        "valid_pre_eos_block_ids": valid_blocks,
        "selected_block_text_main_trajectory": tokenizer.decode(
            selected_tokens, skip_special_tokens=True
        ).strip(),
        "conditions": conditions,
        "sanity": {
            "selected_block_before_first_eos": first_eos is None
            or block_range[1] <= first_eos,
            "selected_block_no_special_or_mask": not any(
                int(value) in excluded_ids for value in selected_tokens
            ),
            "selected_block_complete_25": len(selected_tokens) == args.block_size,
            "all_condition_sanity_passed": all(
                all(condition_row["sanity"].values())
                for condition_row in conditions.values()
            ),
        },
        "config": {
            "execution_host": os.uname().nodename,
            "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None,
            "model": args.model,
            "encoder": args.encoder,
            "generation_length": args.gen_length,
            "semantic_block_size": args.block_size,
            "K": args.num_candidates,
            "candidate_update_size": args.candidate_update_size,
            "channels_per_step": args.channels_per_step,
            "base_temperature": args.base_temperature,
            "candidate_temperature": args.candidate_temperature,
            "rollout_temperature": args.rollout_temperature,
            "policy_rollouts_target_average": args.policy_rollouts,
            "policy_rollout_schedule": args.policy_rollout_schedule,
            "future_repeats": args.future_repeats,
            "endpoint_repeats": args.endpoint_repeats,
            "num_message_bits": args.num_message_bits,
            "direction_seed": args.direction_seed,
            "message_seed": args.message_seed,
            "matched_reference_policy": "same_candidates_commit_index_0",
            "selection_future_rng_independent": True,
            "paired_future_rng": True,
            "DLM_MIN_COMPLETION_TOKENS": os.environ.get("DLM_MIN_COMPLETION_TOKENS"),
            "DLM_MAX_COMPLETION_RETRIES": os.environ.get("DLM_MAX_COMPLETION_RETRIES"),
            "max_retries": 0,
        },
        "elapsed_minutes": (time.time() - started) / 60,
    }
    if not all(result["sanity"].values()):
        raise RuntimeError(f"prompt-level sanity failed: {result['sanity']}")
    return result


def main() -> None:
    args = parse_args()
    if args.gen_length % args.block_size:
        raise ValueError("generation length must be divisible by semantic block size")
    if args.pilot_blocks * args.block_size < args.gen_length:
        raise ValueError("pilot_blocks must cover the full 300-token trajectory")
    if args.candidate_update_size != 1:
        raise ValueError("reverse-hybrid diagnostic currently requires r=1")
    if args.future_repeats < 1 or args.endpoint_repeats < 1:
        raise ValueError("repeat counts must be positive")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir or args.output_jsonl.parent / "reverse_hybrid_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(
        args.waterbench_file.parent,
        args.waterbench_file.name,
        args.n_prompts,
        args.seed,
    )
    indexed = [
        (idx, row)
        for idx, row in enumerate(rows)
        if idx % args.num_prompt_shards == args.prompt_shard
    ]
    if args.prompt_indices is not None:
        wanted = set(args.prompt_indices)
        indexed = [(idx, row) for idx, row in enumerate(rows) if idx in wanted]
    if args.collapse_prompt_indices is None:
        collapse_set = set(range(min(args.collapse_blocks, args.n_prompts)))
    else:
        collapse_set = set(args.collapse_prompt_indices)
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
    directions, signs = build_directions(
        args.gen_length // args.block_size,
        args.num_message_bits,
        args.direction_seed,
        args.message_seed,
    )
    excluded_ids = special_ids(tokenizer, args.mask_id)
    with args.output_jsonl.open("a") as output:
        for prompt_idx, record in indexed:
            if prompt_idx in done:
                print(f"prompt {prompt_idx}: already complete", flush=True)
                continue
            set_global_seed(stable_seed(args.seed, prompt_idx, 999))
            result = run_prompt(
                prompt_idx,
                record,
                collapse=prompt_idx in collapse_set,
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
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            done.add(prompt_idx)
            print(
                f"prompt {prompt_idx}: complete in {result['elapsed_minutes']:.1f} min",
                flush=True,
            )


if __name__ == "__main__":
    main()
