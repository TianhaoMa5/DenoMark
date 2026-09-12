#!/usr/bin/env python3
"""Run PMark prior/offline multi-channel selection over fixed LLaDA blocks."""
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
from denmark.baselines.pmark.model import (
    build_pmark_pivots,
    build_pmark_secret_bits,
    detect_pmark_blocks,
    llada_generate_pmark_blocks,
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
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--candidate_batch_size", type=int, default=16)
    parser.add_argument("--remasking", choices=["low_confidence", "random", "ar"], default="random")
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_rep4", type=float, default=0.2)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--record_candidate_texts", action="store_true")

    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--pivot_seed", type=int, default=42)
    parser.add_argument("--secret_seed", type=int, default=0)
    parser.add_argument("--median_method", choices=["prior", "torch"], default="prior")
    parser.add_argument("--detect_tolerance", type=float, default=0.001)
    parser.add_argument("--detect_decay", type=float, default=250.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_per_dataset <= 0:
        raise ValueError("n_per_dataset must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("invalid sharding")
    if args.max_new_tokens <= 0 or args.block_size <= 0 or args.max_new_tokens % args.block_size:
        raise ValueError("block_size must divide positive max_new_tokens")
    num_blocks = args.max_new_tokens // args.block_size
    if args.steps <= 0 or args.steps % num_blocks:
        raise ValueError("steps must be divisible by the number of blocks")
    if args.num_candidates <= 0 or args.candidate_batch_size <= 0:
        raise ValueError("candidate counts must be positive")
    if args.temperature < 0 or args.num_channels <= 0:
        raise ValueError("invalid temperature or PMark channel count")
    if args.max_retries < 0 or args.attempt_offset < 0:
        raise ValueError("retry counts must be non-negative")
    if args.detect_tolerance < 0 or args.detect_decay < 0:
        raise ValueError("detector tolerance and decay must be non-negative")
    if args.source_indices is not None:
        if len(set(args.source_indices)) != len(args.source_indices):
            raise ValueError("source_indices must be unique")
        if any(source_index < 0 for source_index in args.source_indices):
            raise ValueError("source_indices must be non-negative")


def encoder_embedding_dim(encoder) -> int:
    for key in ("hidden_size", "d_model", "dim"):
        value = getattr(encoder.config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("cannot infer encoder embedding dimension")


def main() -> None:
    args = parse_args()
    validate_args(args)
    from transformers import AutoModel, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_blocks = args.max_new_tokens // args.block_size
    print("=== LLaDA PMark fixed-block multi-channel selection ===")
    print(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"datasets={args.datasets} n={args.n_per_dataset} block={args.block_size} "
        f"K={args.num_candidates} T={args.temperature} median={args.median_method}"
    )
    print(
        f"channels={args.num_channels} pivot_seed={args.pivot_seed} "
        f"secret_seed={args.secret_seed}"
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
    pivots = build_pmark_pivots(
        encoder_embedding_dim(encoder),
        args.num_channels,
        args.pivot_seed,
    )
    secret_bits = build_pmark_secret_bits(
        num_blocks,
        args.num_channels,
        args.secret_seed,
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
        requested = set(args.source_indices) if args.source_indices is not None else None
        if requested is not None:
            missing = sorted(requested - {source_index for source_index, _ in selected_rows})
            if missing:
                raise ValueError(f"requested source indices are outside seeded selection: {missing}")
        shard_rows = [
            (position, source_index, row)
            for position, (source_index, row) in enumerate(selected_rows)
            if position % args.num_shards == args.shard_idx
            and (requested is None or source_index in requested)
        ]
        output_dir = args.output_dir / dataset_key
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"generations_shard{args.shard_idx}.jsonl"
        config_path = output_dir / f"run_config_shard{args.shard_idx}.json"
        run_config = {
            "method": "pmark_block_multichannel",
            "variant": "prior_offline",
            "selection_rule": "sequential_multichannel_halfspace_filter",
            "dataset": dataset_key,
            "dataset_file": str(source_path),
            "dataset_sha256": source_sha256,
            "selected_source_indices": [source_index for source_index, _ in selected_rows],
            "requested_source_indices": sorted(requested) if requested is not None else None,
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
            "num_candidates": args.num_candidates,
            "candidate_batch_size": args.candidate_batch_size,
            "remasking": args.remasking,
            "num_channels": args.num_channels,
            "pivot_seed": args.pivot_seed,
            "secret_seed": args.secret_seed,
            "median_method": args.median_method,
            "detect_tolerance": args.detect_tolerance,
            "detect_decay": args.detect_decay,
            "min_response_tokens": args.min_response_tokens,
            "max_rep4": args.max_rep4,
            "max_retries": args.max_retries,
            "attempt_offset": args.attempt_offset,
        }
        config_path.write_text(
            json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"=== {dataset_name}: {len(shard_rows)} prompts -> {output_path} ===")
        with output_path.open("w", encoding="utf-8") as output_file:
            for selected_position, source_index, row in tqdm(shard_rows, desc=dataset_key):
                input_text = row.get("raw_prompt") or row.get("input", "")
                context = "" if row.get("raw_prompt") else row.get("context", "")
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
                        source_index,
                        "pmark_block_multichannel",
                        attempt,
                    )
                    seed_generation_rng(attempt_seed)
                    generated_text, generated_token_ids, diagnostics = (
                        llada_generate_pmark_blocks(
                            prompt_tensor,
                            generator,
                            encoder,
                            encoder_tokenizer,
                            generator_tokenizer,
                            pivots,
                            secret_bits,
                            mask_id,
                            gen_length=args.max_new_tokens,
                            block_size=args.block_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            num_candidates=args.num_candidates,
                            candidate_batch_size=args.candidate_batch_size,
                            median_method=args.median_method,
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
                        start,
                        min(start + args.block_size, len(retokenized_ids)),
                        generator_tokenizer,
                    )
                    for start in range(0, len(retokenized_ids), args.block_size)
                ]
                detection = detect_pmark_blocks(
                    block_texts,
                    encoder,
                    encoder_tokenizer,
                    pivots,
                    secret_bits,
                    device=args.device,
                    tolerance=args.detect_tolerance,
                    decay=args.detect_decay,
                )
                generation_matches = [
                    bool(match)
                    for diagnostic in diagnostics
                    if diagnostic["selected_projections"] is not None
                    for match in diagnostic["selected_matches"]
                ]
                sample_id = f"{dataset_key}:{source_index}:{attempt}"
                output_record = {
                    "method": "pmark_block_multichannel",
                    "variant": "prior_offline",
                    "selection_rule": "sequential_multichannel_halfspace_filter",
                    "dataset": dataset_name,
                    "ds_key": dataset_key,
                    "prompt_idx": selected_position,
                    "selected_position": selected_position,
                    "source_idx": source_index,
                    "shard": args.shard_idx,
                    "num_shards": args.num_shards,
                    "sample_id": sample_id,
                    "watermark_sample_id": sample_id,
                    "prompt_input": input_text,
                    "prompt_context": context,
                    "prompt_full": prompt_text,
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
                    "block_diagnostics": diagnostics,
                    "generation_hard_hits": int(sum(generation_matches)),
                    "generation_trial_count": len(generation_matches),
                    "generation_hard_hit_rate": (
                        sum(generation_matches) / len(generation_matches)
                        if generation_matches
                        else 0.0
                    ),
                    "pmark_z_score": detection["z_score"],
                    "pmark_hit_rate": detection["hit_rate"],
                    "pmark_hard_hit_rate": detection["hard_hit_rate"],
                    "pmark_detection": detection,
                    "det_score": detection["z_score"],
                    "gen_config": run_config,
                    "watermark_config": {
                        "method": "pmark_block_multichannel",
                        "variant": "prior_offline",
                        "encoder": args.encoder_model,
                        "block_size": args.block_size,
                        "num_candidates": args.num_candidates,
                        "num_channels": args.num_channels,
                        "pivot_seed": args.pivot_seed,
                        "secret_seed": args.secret_seed,
                        "median_method": args.median_method,
                    },
                }
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
        print(f"wrote {len(shard_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
