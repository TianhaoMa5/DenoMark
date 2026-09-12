#!/usr/bin/env python3
"""Run SemStamp-style rejection sampling over complete fixed LLaDA blocks."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from tqdm import tqdm

from denmark.core.scoring import block_decode
from denmark.baselines.semstamp.model import (
    SEMSTAMP_HASH_KEY,
    build_lsh_hyperplanes,
    detect_semstamp_block_transitions,
    llada_generate_semstamp_blocks,
)
from denmark.core.model import load_generator_model, resolve_mask_id
from denmark.baselines.block_best_of_k.generate import (
    DATASETS,
    build_chat_prompt,
    clean_text,
    load_selected_rows,
    rep_ngram,
    seed_generation_rng,
    stable_attempt_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waterbench_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), required=True)
    parser.add_argument("--n_per_dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--source_indices", nargs="*", type=int, default=None)
    parser.add_argument("--attempt_offset", type=int, default=0)

    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--encoder_model", required=True)
    parser.add_argument("--generator_family", choices=["llada", "llada2"], default="llada")
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)

    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--proposal_batch_size", type=int, default=16)
    parser.add_argument("--max_trials_per_block", type=int, default=100)
    parser.add_argument("--remasking", choices=["low_confidence", "random", "ar"], default="random")
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_rep4", type=float, default=0.2)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--record_candidate_texts", action="store_true")

    parser.add_argument("--lsh_dim", type=int, default=2)
    parser.add_argument("--lsh_seed", type=int, default=1234)
    parser.add_argument("--accept_rate", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--hash_key", type=int, default=SEMSTAMP_HASH_KEY)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_per_dataset <= 0:
        raise ValueError("--n_per_dataset must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("sharding must satisfy 0 <= shard_idx < num_shards")
    if args.max_new_tokens <= 0 or args.block_size <= 0:
        raise ValueError("generation lengths must be positive")
    if args.max_new_tokens % args.block_size:
        raise ValueError("--block_size must divide --max_new_tokens")
    num_blocks = args.max_new_tokens // args.block_size
    if args.steps <= 0 or args.steps % num_blocks:
        raise ValueError("--steps must be divisible by the number of blocks")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.proposal_batch_size <= 0 or args.max_trials_per_block <= 0:
        raise ValueError("proposal batch size and max trials must be positive")
    if args.lsh_dim <= 0:
        raise ValueError("--lsh_dim must be positive")
    if not 0 < args.accept_rate < 1:
        raise ValueError("--accept_rate must be strictly between 0 and 1")
    num_accept = int((2**args.lsh_dim) * args.accept_rate)
    if num_accept <= 0 or num_accept >= 2**args.lsh_dim:
        raise ValueError("accept rate must select at least one but not all LSH bins")
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")
    if args.max_retries < 0 or args.attempt_offset < 0:
        raise ValueError("retry counts and offsets must be non-negative")
    if args.source_indices is not None:
        if len(set(args.source_indices)) != len(args.source_indices):
            raise ValueError("--source_indices must be unique")
        if any(source_idx < 0 for source_idx in args.source_indices):
            raise ValueError("--source_indices must be non-negative")


def encoder_embedding_dim(encoder) -> int:
    config = encoder.config
    for key in ("hidden_size", "d_model", "dim"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("cannot infer encoder embedding dimension")


def main() -> None:
    args = parse_args()
    validate_args(args)
    from transformers import AutoModel, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_blocks = args.max_new_tokens // args.block_size

    print("=== LLaDA SemStamp-style fixed-block rejection sampling ===")
    print(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"datasets={args.datasets} n={args.n_per_dataset} seed={args.seed}")
    print(
        f"block={args.block_size} steps={args.steps} T={args.temperature} "
        f"proposal_batch={args.proposal_batch_size} max_trials={args.max_trials_per_block}"
    )
    print(
        f"LSH D={args.lsh_dim} lambda={args.accept_rate} margin={args.margin} "
        f"lsh_seed={args.lsh_seed} hash_key={args.hash_key}"
    )

    generator = load_generator_model(
        args.model_name_or_path,
        generator_family=args.generator_family,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map or args.device,
        trust_remote_code=True,
    )
    generator_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    mask_id = resolve_mask_id(generator_tokenizer, args.generator_family, args.mask_id)
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
    encoder = AutoModel.from_pretrained(
        args.encoder_model,
        torch_dtype=torch.float32,
    ).to(args.device).eval()
    hyperplanes = build_lsh_hyperplanes(
        args.lsh_dim,
        encoder_embedding_dim(encoder),
        args.lsh_seed,
    )
    print(f"model={args.model_name_or_path} encoder={args.encoder_model} mask_id={mask_id}")

    for dataset_key in args.datasets:
        dataset_name, filename = DATASETS[dataset_key]
        source_path = args.waterbench_dir / filename
        selected_rows, source_sha256 = load_selected_rows(
            source_path,
            args.n_per_dataset,
            args.seed,
        )
        requested_source_indices = None
        if args.source_indices is not None:
            requested_source_indices = set(args.source_indices)
            selected_source_indices = {source_idx for source_idx, _ in selected_rows}
            missing = sorted(requested_source_indices - selected_source_indices)
            if missing:
                raise ValueError(
                    f"requested indices are outside seeded selection: {missing}"
                )
        shard_rows = [
            (position, source_idx, record)
            for position, (source_idx, record) in enumerate(selected_rows)
            if position % args.num_shards == args.shard_idx
            and (
                requested_source_indices is None
                or source_idx in requested_source_indices
            )
        ]
        dataset_output_dir = args.output_dir / dataset_key
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = dataset_output_dir / f"generations_shard{args.shard_idx}.jsonl"
        config_path = dataset_output_dir / f"run_config_shard{args.shard_idx}.json"
        run_config = {
            "method": "semstamp_candidate",
            "selection_rule": "first_valid_semstamp_transition",
            "fallback_policy": "last_nonempty_after_max_trials",
            "dataset": dataset_key,
            "dataset_file": str(source_path),
            "dataset_sha256": source_sha256,
            "selected_source_indices": [idx for idx, _ in selected_rows],
            "requested_source_indices": (
                sorted(requested_source_indices)
                if requested_source_indices is not None
                else None
            ),
            "n_per_dataset": args.n_per_dataset,
            "seed": args.seed,
            "shard_idx": args.shard_idx,
            "num_shards": args.num_shards,
            "model": args.model_name_or_path,
            "encoder": args.encoder_model,
            "generator_family": args.generator_family,
            "mask_id": mask_id,
            "max_new_tokens": args.max_new_tokens,
            "block_size": args.block_size,
            "steps": args.steps,
            "temperature": args.temperature,
            "proposal_batch_size": args.proposal_batch_size,
            "max_trials_per_block": args.max_trials_per_block,
            "remasking": args.remasking,
            "lsh_dim": args.lsh_dim,
            "lsh_seed": args.lsh_seed,
            "accept_rate": args.accept_rate,
            "margin": args.margin,
            "hash_key": args.hash_key,
            "mask_rng_device": str(torch.device(args.device)),
            "mask_scheme": "semstamp_torch_randperm_on_active_device_v2",
            "min_response_tokens": args.min_response_tokens,
            "max_rep4": args.max_rep4,
            "max_retries": args.max_retries,
            "attempt_offset": args.attempt_offset,
        }
        config_path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n=== {dataset_name}: {len(shard_rows)} prompts -> {output_path} ===")

        with output_path.open("w", encoding="utf-8") as output_file:
            for selected_position, source_idx, record in tqdm(
                shard_rows,
                desc=dataset_key,
            ):
                input_text = record.get("raw_prompt") or record.get("input", "")
                context = "" if record.get("raw_prompt") else record.get("context", "")
                prompt_seed_text = (
                    (context + "\n\n" + input_text).strip()
                    if context
                    else input_text.strip()
                )
                prompt_text, prompt_ids = build_chat_prompt(
                    input_text,
                    context,
                    generator_tokenizer,
                )
                prompt_tensor = torch.tensor(
                    prompt_ids,
                    dtype=torch.long,
                    device=args.device,
                ).unsqueeze(0)

                for retry_index in range(args.max_retries + 1):
                    attempt = args.attempt_offset + retry_index
                    attempt_seed = stable_attempt_seed(
                        args.seed,
                        dataset_key,
                        source_idx,
                        "semstamp_candidate",
                        attempt,
                    )
                    seed_generation_rng(attempt_seed)
                    generated_text, generated_token_ids, block_diagnostics = (
                        llada_generate_semstamp_blocks(
                            prompt_tensor,
                            prompt_seed_text,
                            generator,
                            encoder,
                            encoder_tokenizer,
                            generator_tokenizer,
                            hyperplanes,
                            mask_id,
                            gen_length=args.max_new_tokens,
                            block_size=args.block_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            proposal_batch_size=args.proposal_batch_size,
                            max_trials_per_block=args.max_trials_per_block,
                            accept_rate=args.accept_rate,
                            margin=args.margin,
                            hash_key=args.hash_key,
                            remasking=args.remasking,
                            generator_family=args.generator_family,
                            device=args.device,
                            record_candidate_texts=args.record_candidate_texts,
                        )
                    )
                    generated_text = clean_text(generated_text)
                    retokenized_ids = generator_tokenizer(
                        generated_text,
                        add_special_tokens=False,
                    )["input_ids"][: args.max_new_tokens]
                    token_length = len(retokenized_ids)
                    repetition_4 = rep_ngram(generated_text, 4)
                    passed_quality = (
                        token_length >= args.min_response_tokens
                        and repetition_4 <= args.max_rep4
                    )
                    if passed_quality:
                        break

                block_texts = [
                    block_decode(
                        retokenized_ids,
                        block_id * args.block_size,
                        min((block_id + 1) * args.block_size, len(retokenized_ids)),
                        generator_tokenizer,
                    )
                    for block_id in range(num_blocks)
                    if block_id * args.block_size < len(retokenized_ids)
                ]
                detection = detect_semstamp_block_transitions(
                    block_texts,
                    prompt_seed_text,
                    encoder,
                    encoder_tokenizer,
                    hyperplanes,
                    accept_rate=args.accept_rate,
                    hash_key=args.hash_key,
                    device=args.device,
                )
                generation_hits = [
                    bool(diagnostic["transition_hit"])
                    for diagnostic in block_diagnostics
                    if diagnostic.get("selected_hash") is not None
                ]
                generation_fallbacks = [
                    diagnostic["block_id"]
                    for diagnostic in block_diagnostics
                    if "fallback" in diagnostic["selection_reason"]
                ]
                sample_id = f"{dataset_key}:{source_idx}:{attempt}"
                output_record = {
                    "method": "semstamp_candidate",
                    "selection_rule": "first_valid_semstamp_transition",
                    "fallback_policy": "last_nonempty_after_max_trials",
                    "dataset": dataset_name,
                    "ds_key": dataset_key,
                    "prompt_idx": selected_position,
                    "selected_position": selected_position,
                    "source_idx": source_idx,
                    "shard": args.shard_idx,
                    "num_shards": args.num_shards,
                    "sample_id": sample_id,
                    "watermark_sample_id": sample_id,
                    "prompt_input": input_text,
                    "prompt_context": context,
                    "prompt_full": prompt_text,
                    "prompt_seed_text": prompt_seed_text,
                    "text": generated_text,
                    "generated_token_ids": generated_token_ids,
                    "retokenized_token_ids": retokenized_ids,
                    "token_len": token_length,
                    "word_len": len(generated_text.split()),
                    "rep4": repetition_4,
                    "passed_quality": passed_quality,
                    "too_short": token_length < args.min_response_tokens,
                    "too_repetitive": repetition_4 > args.max_rep4,
                    "retry_attempts": attempt + 1,
                    "retry_attempt_offset": args.attempt_offset,
                    "retry_attempts_this_run": retry_index + 1,
                    "attempt_seed": attempt_seed,
                    "block_diagnostics": block_diagnostics,
                    "generation_transition_hits": sum(generation_hits),
                    "generation_transition_count": len(generation_hits),
                    "generation_hit_rate": (
                        sum(generation_hits) / len(generation_hits)
                        if generation_hits
                        else 0.0
                    ),
                    "generation_fallback_blocks": generation_fallbacks,
                    "semstamp_z_score": detection["z_score"],
                    "semstamp_hit_rate": detection["hit_rate"],
                    "semstamp_n_hits": detection["n_hits"],
                    "semstamp_n_transitions": detection["n_transitions"],
                    "semstamp_detection": detection,
                    "det_score": detection["z_score"],
                    "gen_config": run_config,
                    "watermark_config": {
                        "method": "semstamp_candidate",
                        "encoder": args.encoder_model,
                        "block_size": args.block_size,
                        "proposal_batch_size": args.proposal_batch_size,
                        "max_trials_per_block": args.max_trials_per_block,
                        "lsh_dim": args.lsh_dim,
                        "lsh_seed": args.lsh_seed,
                        "accept_rate": args.accept_rate,
                        "margin": args.margin,
                        "hash_key": args.hash_key,
                        "mask_rng_device": str(torch.device(args.device)),
                        "mask_scheme": "semstamp_torch_randperm_on_active_device_v2",
                    },
                }
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
        print(f"wrote {len(shard_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
