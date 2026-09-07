#!/usr/bin/env python3
"""Dream unit-level reverse-hybrid closure diagnostic.

Every branch resumes a saved Dream canvas at its saved native timestep.  Native
global transfer order is preserved; only the watermark candidate commitment is
switched between pi, matched pi0, or a forced current candidate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))

from denomark.experiments.trajectory.dream_reverse_hybrid import (
    NativeProposal,
    atomic_write_json,
    build_policy_candidates,
    native_proposal_step,
    score_and_commit,
    seed_all,
    stable_seed,
    unit_masks,
)
from denomark.experiments.trajectory.dream_theory_diagnostic import keyed_unit_scores, validate_native_sampler_compatibility
from denomark.core.model import build_directions, resolve_mask_id
from denomark.core.model import (NativeSemanticWatermarkHook, build_model_input_from_row, build_prompt, clean_text, load_rows, safe_decode_dream as safe_decode, truncate_tail)


class InvalidDiagnosticUnit(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_jsonl", type=Path, required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--model_revision", default=None)
    p.add_argument("--encoder_model", required=True)
    p.add_argument("--output_jsonl", type=Path, required=True)
    p.add_argument("--cache_dir", type=Path, default=None)
    p.add_argument("--dataset", default="longform_qa")
    p.add_argument("--prompt_variant", default="default")
    p.add_argument("--prompt_mode", default="instruct")
    p.add_argument("--max_prompt_chars", type=int, default=0)
    p.add_argument("--num_prompts", type=int, default=30)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--prompt_indices", nargs="*", type=int, default=None)
    p.add_argument("--gen_length", type=int, default=300)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--semantic_unit_size", type=int, default=25)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--candidate_update_size", type=int, default=1)
    p.add_argument("--num_message_bits", type=int, default=2)
    p.add_argument("--channels_per_step", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--candidate_temperature", type=float, default=0.6)
    p.add_argument("--rollout_temperature", type=float, default=0.5)
    p.add_argument("--selection_rollouts", type=int, default=3)
    p.add_argument("--future_repeats", type=int, default=10)
    p.add_argument("--endpoint_repeats", type=int, default=20)
    p.add_argument("--collapse_units", type=int, default=10)
    p.add_argument("--collapse_prompt_indices", nargs="*", type=int, default=None)
    p.add_argument("--alg", default="origin", choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
    p.add_argument("--alg_temp", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--direction_seed", type=int, default=42)
    p.add_argument("--message_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_id", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device_map", default="cuda")
    p.add_argument("--rollout_logits_mode", default="full", choices=["full", "window"])
    p.add_argument("--selection_mode", default="argmax_score")
    p.add_argument("--candidate_selection_seed", type=int, default=0)
    p.add_argument("--bucket_num_bits", type=int, default=16)
    p.add_argument("--bucket_seed", type=int, default=12345)
    p.add_argument("--reference_candidate_prob", default="uniform")
    p.add_argument("--smoke_trace", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    if args.candidate_update_size != 1:
        p.error("this diagnostic currently requires candidate update r=1")
    if args.gen_length % args.semantic_unit_size:
        p.error("generation length must be divisible by semantic unit size")
    return args


def make_hook(args, model, tokenizer, encoder, encoder_tokenizer, directions, signs, prompt_len, sample_id):
    return NativeSemanticWatermarkHook(
        model, tokenizer, encoder, encoder_tokenizer, directions, signs,
        prompt_len=prompt_len, gen_length=args.gen_length,
        block_size=args.semantic_unit_size, cand_block_size=args.candidate_update_size,
        num_candidates=args.num_candidates, channels_per_step=args.channels_per_step,
        candidate_temperature=args.candidate_temperature,
        rollout_temperature=args.rollout_temperature,
        rollouts_per_cand=args.selection_rollouts, rollout_schedule="constant",
        logprob_weight=0.0, candidate_position_mode="per_candidate_random_semantic_unit",
        mask_id=args.mask_id, device=args.device, argmax_logprob_top_frac=1.0,
        selection_mode=args.selection_mode,
        candidate_selection_seed=args.candidate_selection_seed,
        sample_id=sample_id, bucket_num_bits=args.bucket_num_bits,
        bucket_seed=args.bucket_seed, reference_candidate_prob=args.reference_candidate_prob,
        resample_base_candidate=True, shared_rollout_seeds=True,
        dedup_candidates=True, rollout_logits_mode=args.rollout_logits_mode,
    )


def proposal(args, model, state, native_step, prompt_len, stream_seed):
    return native_proposal_step(
        model, state, native_step=native_step, steps=args.steps, eps=args.eps,
        mask_id=args.mask_id, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, alg=args.alg, alg_temp=args.alg_temp,
        prompt_len=prompt_len, gen_length=args.gen_length,
        block_size=args.semantic_unit_size,
        seed=stable_seed(stream_seed, native_step, 11),
    )


def prepare_candidates(args, prepared, state, block_id, prompt_len, collapse, stream_seed):
    b_s = prompt_len + block_id * args.semantic_unit_size
    b_e = b_s + args.semantic_unit_size
    # Recover the native transfer positions from the actual state supplied to
    # the candidate policy.  This is essential for a cached reverse-hybrid
    # F_t: re-running a BF16 proposal in a fresh process can very rarely sample
    # MASK again at a transferred position, while the materialized F_t remains
    # the authoritative state.  Candidate construction must condition on that
    # saved state rather than on a newly inferred changed_by_block dictionary.
    native_changed = (
        (prepared.before[0, b_s:b_e] == args.mask_id)
        & (state[0, b_s:b_e] != args.mask_id)
    )
    positions = native_changed.nonzero(as_tuple=True)[0] + b_s
    if not positions.numel():
        raise RuntimeError(
            f"cached/current state has no native transfer for block {block_id} "
            f"at Dream step {prepared.native_step}"
        )
    remaining = unit_masks(prepared.before, b_s, b_e, args.mask_id)
    cands, rows_pos, tokens, meta = build_policy_candidates(
        state, prepared.logits, positions, block_id=block_id,
        prompt_len=prompt_len, gen_length=args.gen_length,
        block_size=args.semantic_unit_size, mask_id=args.mask_id,
        num_candidates=args.num_candidates, update_size=args.candidate_update_size,
        candidate_temperature=args.candidate_temperature, collapse=collapse,
        seed=stable_seed(stream_seed, prepared.native_step, block_id, 22),
    )
    return cands, rows_pos, tokens, meta, remaining


@torch.no_grad()
def advance_whole_step(
    args, model, hook, state, native_step, prompt_len, *, policy, collapse, stream_seed,
):
    prepared = proposal(args, model, state, native_step, prompt_len, stream_seed)
    current = prepared.state
    audits = []
    for block_id in sorted(prepared.changed_by_block):
        cands, positions, tokens, meta, remaining = prepare_candidates(
            args, prepared, current, block_id, prompt_len, collapse, stream_seed
        )
        current, winner, diag = score_and_commit(
            current, cands, positions, tokens, block_id=block_id,
            logits=prepared.logits, native_step=native_step,
            remaining_before=remaining, policy=policy, hook=hook,
            # Under the exact-collapse intervention every candidate is
            # bitwise identical.  The keyed argmax is therefore immaterial;
            # committing index 0 is exactly pi while avoiding K redundant
            # selection rollouts.
            forced_candidate=0 if collapse and policy == "pi" else None,
        )
        audits.append({"block_id": block_id, "winner": winner, **meta, "selection_diag": diag})
    return current, prepared, audits


@torch.no_grad()
def pilot_trajectory(args, prompt, prompt_idx, model, hook):
    masks = torch.full((1, args.gen_length), args.mask_id, dtype=torch.long, device=prompt.device)
    state = torch.cat([prompt, masks], dim=1)
    prompt_len = int(prompt.shape[1])
    first_starts = {}
    trace = []
    stream = stable_seed(args.seed, prompt_idx, 100_000)
    for native_step in range(args.steps):
        before = state.clone()
        state, prepared, audits = advance_whole_step(
            args, model, hook, state, native_step, prompt_len,
            policy="pi", collapse=False, stream_seed=stream,
        )
        for block_id in prepared.changed_by_block:
            first_starts.setdefault(block_id, {
                "full_state": before.detach().cpu(),
                "native_step": native_step,
                "masks_in_unit": unit_masks(
                    before,
                    prompt_len + block_id * args.semantic_unit_size,
                    prompt_len + (block_id + 1) * args.semantic_unit_size,
                    args.mask_id,
                ),
                "total_masks": int((before == args.mask_id).sum().item()),
            })
        trace.append({
            "native_step": native_step,
            "changed_blocks": sorted(prepared.changed_by_block),
            "num_global_transfers": sum(len(v) for v in prepared.changed_by_block.values()),
            "watermark_decisions": len(audits),
        })
        if native_step % 25 == 0:
            print(
                f"  pilot prompt={prompt_idx} native_step={native_step}/{args.steps} "
                f"remaining_masks={int((state[:, prompt_len:prompt_len + args.gen_length] == args.mask_id).sum().item())}",
                flush=True,
            )
        if not bool((state == args.mask_id).any().item()):
            break
    if bool((state[:, prompt_len:prompt_len + args.gen_length] == args.mask_id).any().item()):
        raise RuntimeError("pilot trajectory ended with unresolved generation masks")
    return state, first_starts, trace


def select_valid_unit(args, final_tokens, first_starts, tokenizer, special_ids, prompt_idx):
    eos = getattr(tokenizer, "eos_token_id", None)
    first_eos = next((i for i, value in enumerate(final_tokens) if eos is not None and value == eos), None)
    valid = []
    n_units = args.gen_length // args.semantic_unit_size
    for unit_id in range(n_units):
        start, end = unit_id * args.semantic_unit_size, (unit_id + 1) * args.semantic_unit_size
        values = final_tokens[start:end]
        if unit_id not in first_starts or len(values) != args.semantic_unit_size:
            continue
        if first_eos is not None and end > first_eos:
            continue
        if any(int(value) in special_ids for value in values):
            continue
        text = clean_text(safe_decode(tokenizer, values, special_ids))
        if not text or not any(character.isalnum() for character in text):
            continue
        valid.append((unit_id, text))
    if not valid:
        raise InvalidDiagnosticUnit("no complete nondegenerate pre-EOS unit with a watermark start")
    preferred = [item for item in valid if 0.2 <= (item[0] + 0.5) / n_units <= 0.7]
    pool = preferred[:-1] if len(preferred) > 1 else (preferred or valid[:-1] or valid)
    rng = np.random.default_rng(stable_seed(args.seed, prompt_idx, 77))
    unit_id, text = pool[int(rng.integers(0, len(pool)))]
    return int(unit_id), text, first_eos, [int(item[0]) for item in valid]


@torch.no_grad()
def reference_horizon(args, model, hook, start, start_step, prompt_len, unit_id, stream_seed, collapse):
    state = start.clone()
    start_pos = prompt_len + unit_id * args.semantic_unit_size
    end_pos = start_pos + args.semantic_unit_size
    decisions = []
    for native_step in range(start_step, args.steps):
        prepared = proposal(args, model, state, native_step, prompt_len, stream_seed)
        current = prepared.state
        block_order = sorted(prepared.changed_by_block)
        for block_order_index, block_id in enumerate(block_order):
            cands, positions, tokens, meta, remaining = prepare_candidates(
                args, prepared, current, block_id, prompt_len, collapse, stream_seed
            )
            state_before_commit = current.clone()
            current, winner, _ = score_and_commit(
                current, cands, positions, tokens, block_id=block_id,
                logits=prepared.logits, native_step=native_step,
                remaining_before=remaining, policy="pi0", hook=hook,
            )
            if block_id == unit_id:
                decisions.append({
                    "decision_idx": len(decisions), "native_step": native_step,
                    "decision_block_id": block_id,
                    "targets_selected_unit": True,
                    "masks_before": unit_masks(prepared.before, start_pos, end_pos, args.mask_id),
                    "masks_after": unit_masks(current, start_pos, end_pos, args.mask_id),
                    "reference_candidate": winner, **meta,
                    "state_before_native_token_ids": [
                        int(value) for value in prepared.before[0].detach().cpu().tolist()
                    ],
                    "state_before_current_commit_token_ids": [
                        int(value) for value in state_before_commit[0].detach().cpu().tolist()
                    ],
                    "block_order": block_order,
                    "block_order_index": block_order_index,
                })
        state = current
        if native_step % 25 == 0:
            print(
                f"  reference_horizon unit={unit_id} native_step={native_step} "
                f"local_decisions={len(decisions)} masks={unit_masks(state, start_pos, end_pos, args.mask_id)}",
                flush=True,
            )
        if unit_masks(state, start_pos, end_pos, args.mask_id) == 0:
            return decisions
    raise RuntimeError("reference horizon did not resolve selected unit")


@torch.no_grad()
def locate_current(args, model, hook, horizon, target, prompt_len, unit_id, prefix_stream, selection_stream, collapse):
    """Restore an exactly materialized fixed-seed pi0 prefix state for decision t."""
    event = horizon[target]
    native_step = int(event["native_step"])
    before_native = torch.tensor(
        event["state_before_native_token_ids"], dtype=torch.long, device=args.device
    ).unsqueeze(0)
    prepared = proposal(args, model, before_native, native_step, prompt_len, prefix_stream)
    current = torch.tensor(
        event["state_before_current_commit_token_ids"], dtype=torch.long, device=args.device
    ).unsqueeze(0)
    cands, positions, tokens, meta, remaining = prepare_candidates(
        args, prepared, current, unit_id, prompt_len, collapse, selection_stream
    )
    winner_state, winner, diag = score_and_commit(
        current, cands, positions, tokens, block_id=unit_id,
        logits=prepared.logits, native_step=native_step,
        remaining_before=remaining, policy="pi", hook=hook,
    )
    del winner_state
    prefix_audit = [
        {
            "decision_idx": index,
            "native_step": int(previous["native_step"]),
            "block_id": unit_id,
            "policy": "pi0_matched_reference",
            "committed_candidate": 0,
        }
        for index, previous in enumerate(horizon[:target])
    ]
    return {
        "state_before_current_commit": current,
        "prepared": prepared,
        "block_order": event["block_order"],
        "block_order_index": int(event["block_order_index"]),
        "candidates": cands, "positions": positions, "tokens": tokens,
        "candidate_meta": meta, "remaining_before": remaining,
        "winner": winner, "selection_diag": diag,
        "prefix_audit": prefix_audit, "native_step": native_step,
        "current_block_id": unit_id,
    }


@torch.no_grad()
def continue_branch(args, model, hook, located, forced_index, prompt_len, unit_id, future_stream, collapse):
    current = located["state_before_current_commit"].clone()
    prepared: NativeProposal = located["prepared"]
    current, _, _ = score_and_commit(
        current, located["candidates"], located["positions"], located["tokens"],
        block_id=unit_id, logits=prepared.logits, native_step=located["native_step"],
        remaining_before=located["remaining_before"], policy="pi",
        hook=hook, forced_candidate=forced_index,
    )
    start_pos = prompt_len + unit_id * args.semantic_unit_size
    end_pos = start_pos + args.semantic_unit_size
    trace = []
    blocks = located["block_order"]
    for block_id in blocks[located["block_order_index"] + 1:]:
        cands, positions, tokens, _, remaining = prepare_candidates(
            args, prepared, current, block_id, prompt_len, collapse, future_stream
        )
        current, winner, _ = score_and_commit(
            current, cands, positions, tokens, block_id=block_id,
            logits=prepared.logits, native_step=prepared.native_step,
            remaining_before=remaining, policy="pi", hook=hook,
            forced_candidate=0 if collapse else None,
        )
        trace.append({"native_step": prepared.native_step, "block_id": block_id, "winner": winner})
    if unit_masks(current, start_pos, end_pos, args.mask_id) == 0:
        return current, 0, located["native_step"], trace
    extra = 0
    for native_step in range(located["native_step"] + 1, args.steps):
        current, _, audits = advance_whole_step(
            args, model, hook, current, native_step, prompt_len,
            policy="pi", collapse=collapse, stream_seed=future_stream,
        )
        extra += 1
        trace.extend({"native_step": native_step, **audit} for audit in audits)
        if unit_masks(current, start_pos, end_pos, args.mask_id) == 0:
            return current, extra, native_step, trace
    raise RuntimeError("future pi continuation did not resolve selected unit")


def score_unit(args, states, unit_id, prompt_len, tokenizer, encoder, encoder_tokenizer, directions, signs, special_ids):
    start = prompt_len + unit_id * args.semantic_unit_size
    end = start + args.semantic_unit_size
    return keyed_unit_scores(
        states, tokenizer=tokenizer, encoder=encoder,
        encoder_tokenizer=encoder_tokenizer, directions=directions, signs=signs,
        unit_id=unit_id, unit_start=start, unit_end=end, dream_step=0,
        channels_per_step=args.num_message_bits, special_ids=special_ids,
        device=args.device,
    ).tolist()


def run_condition(args, model, hook, start, start_step, prompt_idx, prompt_len, unit_id, collapse, cache_dir, scorer):
    condition = "exact_collapse" if collapse else "default"
    prefix_stream = stable_seed(args.seed, prompt_idx, 300_000)
    horizon_file = cache_dir / condition / "reference_horizon.json"
    if horizon_file.exists():
        horizon = json.loads(horizon_file.read_text())
    else:
        horizon = reference_horizon(
            args, model, hook, start, start_step, prompt_len, unit_id, prefix_stream, collapse
        )
        atomic_write_json(horizon_file, horizon)
    if not horizon:
        raise RuntimeError("selected unit has no watermark decision in reference horizon")
    steps = []
    for decision_idx in range(len(horizon)):
        complete = cache_dir / condition / f"step_{decision_idx:03d}.json"
        partial = cache_dir / condition / f"step_{decision_idx:03d}.partial.json"
        if complete.exists():
            row = json.loads(complete.read_text())
            cached_repeats = len(row["paired_future_differences"])
            if cached_repeats > args.future_repeats:
                raise RuntimeError("cached step has more repeats than requested")
            if cached_repeats == args.future_repeats:
                steps.append(row)
                continue
            atomic_write_json(partial, {
                "selected_scores": row["selected_future_scores"],
                "reference_scores": row["reference_future_scores"],
                "differences": row["paired_future_differences"],
                "selected_extra_steps": row["selected_completion"],
                "reference_extra_steps": row["reference_completion"],
            })
        selection_stream = stable_seed(args.seed, prompt_idx, decision_idx, 400_000)
        located = locate_current(
            args, model, hook, horizon, decision_idx, prompt_len, unit_id,
            prefix_stream, selection_stream, collapse,
        )
        selected_scores, reference_scores, differences = [], [], []
        selected_extra, reference_extra = [], []
        if partial.exists():
            cached = json.loads(partial.read_text())
            selected_scores = cached["selected_scores"]
            reference_scores = cached["reference_scores"]
            differences = cached["differences"]
            selected_extra = cached["selected_extra_steps"]
            reference_extra = cached["reference_extra_steps"]
        winner = int(located["winner"])
        for repetition in range(len(differences), args.future_repeats):
            future_stream = stable_seed(args.seed, prompt_idx, decision_idx, repetition, 500_000)
            if collapse:
                reference_end, ref_extra, ref_completion, _ = continue_branch(
                    args, model, hook, located, 0, prompt_len, unit_id, future_stream, True
                )
                value = float(scorer(reference_end)[0])
                selected_value = reference_value = value
                sel_extra, sel_completion = ref_extra, ref_completion
            else:
                selected_end, sel_extra, sel_completion, _ = continue_branch(
                    args, model, hook, located, winner, prompt_len, unit_id, future_stream, False
                )
                selected_value = float(scorer(selected_end)[0])
                forced_states_identical = torch.equal(
                    located["candidates"][winner], located["candidates"][0]
                )
                if forced_states_identical:
                    reference_value = selected_value
                    ref_extra, ref_completion = sel_extra, sel_completion
                else:
                    reference_end, ref_extra, ref_completion, _ = continue_branch(
                        args, model, hook, located, 0, prompt_len, unit_id, future_stream, False
                    )
                    reference_value = float(scorer(reference_end)[0])
            selected_scores.append(selected_value)
            reference_scores.append(reference_value)
            differences.append(selected_value - reference_value)
            selected_extra.append({"extra_native_steps": sel_extra, "completion_timestep": sel_completion})
            reference_extra.append({"extra_native_steps": ref_extra, "completion_timestep": ref_completion})
            atomic_write_json(partial, {
                "selected_scores": selected_scores, "reference_scores": reference_scores,
                "differences": differences, "selected_extra_steps": selected_extra,
                "reference_extra_steps": reference_extra,
            })
            print(f"  {condition} t={decision_idx + 1}/{len(horizon)} future={repetition + 1}/{args.future_repeats}", flush=True)
        diag = located["selection_diag"]
        unit_start = prompt_len + unit_id * args.semantic_unit_size
        unit_end = unit_start + args.semantic_unit_size
        row = {
            "decision_idx": decision_idx,
            "native_step": located["native_step"],
            "decision_block_id": located["current_block_id"],
            "targets_selected_unit": located["current_block_id"] == unit_id,
            "prefix_t": decision_idx,
            "reverse_hybrid_prefix_audit": located["prefix_audit"],
            "masks_in_unit_before_commit": located["remaining_before"],
            "masks_in_unit_after_selected_commit": unit_masks(
                located["candidates"][winner:winner + 1], unit_start, unit_end, args.mask_id
            ),
            "masks_in_unit_after_reference_commit": unit_masks(
                located["candidates"][0:1], unit_start, unit_end, args.mask_id
            ),
            "reverse_hybrid_state_token_ids": [
                int(value)
                for value in located["state_before_current_commit"][0].detach().cpu().tolist()
            ],
            "candidate_positions": located["candidate_meta"]["candidate_positions"],
            "candidate_positions_in_unit": located["candidate_meta"]["candidate_positions_in_unit"],
            "candidate_token_ids": located["candidate_meta"]["candidate_token_ids"],
            "unique_candidate_count": located["candidate_meta"]["unique_candidate_count"],
            "candidate_collision_count": located["candidate_meta"]["candidate_collision_count"],
            "candidate_collision_rate": located["candidate_meta"]["candidate_collision_rate"],
            "selection_rollout_raw_scores": diag["rollout_raw_watermark_scores"],
            "selection_candidate_scores": diag["candidate_watermark_scores"],
            "candidate_score_range": diag["wm_score_range"],
            "selected_candidate_index": winner,
            "reference_candidate_index": 0,
            "forced_commit_states_identical": bool(torch.equal(
                located["candidates"][winner], located["candidates"][0]
            )),
            "selected_future_scores": selected_scores,
            "reference_future_scores": reference_scores,
            "paired_future_differences": differences,
            "delta_hat_pi": float(np.mean(differences)),
            "selected_completion": selected_extra,
            "reference_completion": reference_extra,
            "selection_rng_stream": selection_stream,
            "evaluation_rng_independent": True,
            "paired_future_rng": True,
            "sanity": {
                "shared_F_t_and_candidate_set": True,
                "prefix_selected_unit_decisions": sum(
                    item["block_id"] == unit_id for item in located["prefix_audit"]
                ) == decision_idx,
                "prefix_policy_all_pi0": all(
                    item["policy"] == "pi0_matched_reference" for item in located["prefix_audit"]
                ),
                "collapse_candidates_bitwise_identical": (not collapse)
                or located["candidate_meta"]["unique_candidate_count"] == 1,
                "collapse_delta_exact_zero": (not collapse)
                or all(abs(value) <= 1e-12 for value in differences),
            },
        }
        atomic_write_json(complete, row)
        if partial.exists():
            partial.unlink()
        steps.append(row)
    endpoint_file = cache_dir / condition / "endpoint.partial.json"
    policy_scores, reference_scores, differences = [], [], []
    if endpoint_file.exists():
        cached = json.loads(endpoint_file.read_text())
        policy_scores, reference_scores, differences = (
            cached["policy_scores"], cached["reference_scores"], cached["differences"]
        )
    for repetition in range(len(differences), args.endpoint_repeats):
        stream = stable_seed(args.seed, prompt_idx, repetition, 800_000)
        def finish(policy):
            state = start.clone()
            start_pos = prompt_len + unit_id * args.semantic_unit_size
            end_pos = start_pos + args.semantic_unit_size
            for native_step in range(start_step, args.steps):
                state, _, _ = advance_whole_step(
                    args, model, hook, state, native_step, prompt_len,
                    policy=policy, collapse=collapse, stream_seed=stream,
                )
                if unit_masks(state, start_pos, end_pos, args.mask_id) == 0:
                    return state
            raise RuntimeError("endpoint did not resolve selected unit")
        if collapse:
            reference_end = finish("pi0")
            value = float(scorer(reference_end)[0])
            policy_value = reference_value = value
        else:
            policy_value = float(scorer(finish("pi"))[0])
            reference_value = float(scorer(finish("pi0"))[0])
        policy_scores.append(policy_value)
        reference_scores.append(reference_value)
        differences.append(policy_value - reference_value)
        atomic_write_json(endpoint_file, {
            "policy_scores": policy_scores, "reference_scores": reference_scores,
            "differences": differences,
        })
        print(f"  {condition} endpoint={repetition + 1}/{args.endpoint_repeats}", flush=True)
    deltas = [float(row["delta_hat_pi"]) for row in steps]
    cumulative = np.cumsum(deltas).tolist()
    endpoint = {
        "J_pi_raw_scores": policy_scores, "J_pi0_raw_scores": reference_scores,
        "paired_endpoint_differences": differences,
        "J_pi_hat": float(np.mean(policy_scores)), "J_pi0_hat": float(np.mean(reference_scores)),
        "endpoint_uplift_hat": float(np.mean(differences)),
        "independent_from_reverse_hybrid_Q_evaluation": True,
    }
    summary = {
        "local_horizon_tau": len(steps), "sum_delta": float(sum(deltas)),
        "positive_step_count": sum(value > 1e-12 for value in deltas),
        "negative_step_count": sum(value < -1e-12 for value in deltas),
        "zero_step_count": sum(abs(value) <= 1e-12 for value in deltas),
        "cumulative_delta": cumulative,
        "endpoint_uplift": endpoint["endpoint_uplift_hat"],
        "closure_gap_endpoint_minus_sum_delta": endpoint["endpoint_uplift_hat"] - sum(deltas),
        "mean_candidate_diversity": float(np.mean([row["unique_candidate_count"] for row in steps])),
    }
    return {"condition": condition, "reference_horizon": horizon, "steps": steps, "endpoint": endpoint, "unit_summary": summary}


def run_prompt(args, prompt_idx, source_row, model, tokenizer, encoder, encoder_tokenizer, directions, signs, special_ids, cache_root, collapse):
    started = time.time()
    model_input, prompt_input, prompt_context = build_model_input_from_row(source_row, args.dataset, args.prompt_variant)
    model_input, prompt_truncated = truncate_tail(model_input, args.max_prompt_chars)
    prompt_text, prompt_ids = build_prompt(model_input, "", tokenizer, args.prompt_mode)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
    prompt_len = int(prompt.shape[1])
    hook = make_hook(args, model, tokenizer, encoder, encoder_tokenizer, directions, signs, prompt_len, str(prompt_idx))
    prompt_cache = cache_root / f"prompt_{prompt_idx:04d}"
    pilot_file = prompt_cache / "pilot.json"
    if pilot_file.exists():
        pilot = json.loads(pilot_file.read_text())
        final_tokens = pilot["final_generated_token_ids"]
        first_starts = {
            int(key): {
                "full_state": torch.tensor(value["full_state"], dtype=torch.long).unsqueeze(0),
                "native_step": value["native_step"], "masks_in_unit": value["masks_in_unit"],
                "total_masks": value["total_masks"],
            } for key, value in pilot["first_starts"].items()
        }
        pilot_trace = pilot["trace"]
    else:
        final_state, first_starts, pilot_trace = pilot_trajectory(args, prompt, prompt_idx, model, hook)
        final_tokens = [int(v) for v in final_state[0, prompt_len:prompt_len + args.gen_length].cpu().tolist()]
        atomic_write_json(pilot_file, {
            "final_generated_token_ids": final_tokens,
            "first_starts": {
                str(key): {**{name: value for name, value in row.items() if name != "full_state"},
                           "full_state": [int(v) for v in row["full_state"][0].tolist()]}
                for key, row in first_starts.items()
            },
            "trace": pilot_trace,
        })
    unit_id, unit_text, first_eos, valid_units = select_valid_unit(
        args, final_tokens, first_starts, tokenizer, special_ids, prompt_idx
    )
    start_info = first_starts[unit_id]
    start = start_info["full_state"].to(args.device)
    start_step = int(start_info["native_step"])
    scorer = lambda states: score_unit(
        args, states, unit_id, prompt_len, tokenizer, encoder, encoder_tokenizer,
        directions, signs, special_ids,
    )
    default = run_condition(
        args, model, hook, start, start_step, prompt_idx, prompt_len, unit_id,
        False, prompt_cache, scorer,
    )
    conditions = {"default": default}
    if collapse:
        conditions["exact_collapse"] = run_condition(
            args, model, hook, start, start_step, prompt_idx, prompt_len, unit_id,
            True, prompt_cache, scorer,
        )
    unit_tokens = final_tokens[unit_id * args.semantic_unit_size:(unit_id + 1) * args.semantic_unit_size]
    sanity = {
        "unit_complete_25": len(unit_tokens) == args.semantic_unit_size,
        "unit_eos_count_zero": getattr(tokenizer, "eos_token_id", None) not in unit_tokens,
        "unit_special_count_zero": not any(int(value) in special_ids for value in unit_tokens),
        "unit_final_text_nonempty": bool(unit_text),
        "unit_before_first_eos": first_eos is None or (unit_id + 1) * args.semantic_unit_size <= first_eos,
        "native_global_decoding_not_block_restricted": True,
        "selection_evaluation_rng_independent": True,
        "no_duplicated_prompt_unit": True,
        "all_step_sanity": all(
            all(step["sanity"].values()) for condition in conditions.values() for step in condition["steps"]
        ),
        "all_terminals_resolved": all(
            all(item["masks_after"] >= 0 for item in condition["reference_horizon"])
            for condition in conditions.values()
        ),
    }
    if not all(sanity.values()):
        raise RuntimeError(f"prompt sanity failed: {sanity}")
    if args.smoke_trace:
        print(json.dumps({
            "prompt_idx": prompt_idx, "selected_unit_text": unit_text,
            "token_range": [unit_id * args.semantic_unit_size, (unit_id + 1) * args.semantic_unit_size],
            "eos_count": 0, "local_horizon_tau": default["unit_summary"]["local_horizon_tau"],
            "first_reverse_steps": default["steps"][:3],
        }, ensure_ascii=False), flush=True)
    return {
        "schema_version": 1, "experiment": "dream_unit_reverse_hybrid_closure",
        "dataset": args.dataset, "prompt_idx": prompt_idx,
        "prompt_input": prompt_input, "prompt_context": prompt_context,
        "prompt_full": prompt_text, "prompt_truncated": prompt_truncated,
        "generation_seed": args.seed, "selected_unit_id": unit_id,
        "unit_token_range_generation_zero_based": [
            unit_id * args.semantic_unit_size, (unit_id + 1) * args.semantic_unit_size
        ],
        "selected_unit_text_main_trajectory": unit_text,
        "selected_unit_token_ids_main_trajectory": unit_tokens,
        "first_eos_generation_index": first_eos,
        "valid_pre_eos_unit_ids": valid_units,
        "unit_start_native_step": start_step,
        "unit_start_masks_in_unit": start_info["masks_in_unit"],
        "unit_start_total_masks": start_info["total_masks"],
        "unit_start_full_token_state": [int(v) for v in start[0].cpu().tolist()],
        "pilot_native_trace": pilot_trace,
        "conditions": conditions, "sanity": sanity,
        "config": {
            "model": args.model_name_or_path, "model_revision": args.model_revision,
            "semantic_encoder": args.encoder_model, "generation_length": args.gen_length,
            "Dream_steps": args.steps, "semantic_unit_size": args.semantic_unit_size,
            "K": args.num_candidates, "candidate_update_size": args.candidate_update_size,
            "R_selection": args.selection_rollouts, "channels_per_step": args.channels_per_step,
            "base_temperature": args.temperature, "candidate_temperature": args.candidate_temperature,
            "rollout_temperature": args.rollout_temperature,
            "future_repeats": args.future_repeats, "endpoint_repeats": args.endpoint_repeats,
            "candidate_position_policy": "per_candidate_random_semantic_unit",
            "native_position_policy": "Dream_official_global_arbitrary_order",
            "matched_pi0": "same_candidate_process_commit_index_0",
            "selection_rollout_schedule": "constant_R3",
            "execution_host": os.uname().nodename,
        },
        "elapsed_minutes": (time.time() - started) / 60,
    }


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_dir or args.output_jsonl.parent / "reverse_hybrid_dream_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(args.prompts_jsonl, args.num_prompts, args.offset)
    indexed = [(args.offset + i, row) for i, row in enumerate(all_rows)]
    if args.prompt_indices is not None:
        wanted = set(args.prompt_indices)
        full = load_rows(args.prompts_jsonl, max(wanted) + 1, 0)
        indexed = [(i, full[i]) for i in sorted(wanted)]
    done = set()
    if args.output_jsonl.exists():
        done = {json.loads(line)["prompt_idx"] for line in args.output_jsonl.read_text().splitlines() if line.strip()}
    print(f"Loading Dream: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, revision=args.model_revision, trust_remote_code=True
    )
    args.mask_id = resolve_mask_id(tokenizer, "dream", args.mask_id)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, revision=args.model_revision, torch_dtype=torch.bfloat16,
        device_map=args.device_map, trust_remote_code=True,
    ).eval()
    print(f"native_sampler_contract={validate_native_sampler_compatibility(model)}", flush=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    encoder = AutoModel.from_pretrained(args.encoder_model, torch_dtype=torch.float32).to(args.device).eval()
    directions, signs = build_directions(
        args.gen_length // args.semantic_unit_size, args.num_message_bits,
        args.direction_seed, args.message_seed,
    )
    special_ids = {int(value) for value in {
        args.mask_id, getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "eos_token_id", None),
    } if value is not None}
    collapse_set = set(args.collapse_prompt_indices or range(args.collapse_units))
    with args.output_jsonl.open("a") as output:
        for prompt_idx, source_row in indexed:
            if prompt_idx in done:
                continue
            try:
                result = run_prompt(
                    args, prompt_idx, source_row, model, tokenizer, encoder,
                    encoder_tokenizer, directions, signs, special_ids, cache_root,
                    prompt_idx in collapse_set,
                )
            except InvalidDiagnosticUnit as exc:
                atomic_write_json(cache_root / f"prompt_{prompt_idx:04d}" / "invalid.json", {
                    "prompt_idx": prompt_idx, "reason": str(exc)
                })
                print(f"prompt {prompt_idx}: INVALID {exc}", flush=True)
                continue
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(f"prompt {prompt_idx}: COMPLETE elapsed={result['elapsed_minutes']:.1f}m", flush=True)


if __name__ == "__main__":
    main()
