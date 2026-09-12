#!/usr/bin/env python3
"""Run the LLaDA-8B block-reject / Block Best-of-N WaterBench baseline.

For every 25-token block, sample K complete unwatermarked LLaDA blocks at the
requested temperature, score them with the repository's signed semantic
watermark statistic, and commit the argmax block.  This is deliberately a
separate runner from ``run_waterbench_main_experiment.py`` because its K means
complete proposal blocks, not that runner's per-denoising-step perturbations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denmark.baselines.block_best_of_k.model import llada_generate_block_best_of_k
from denmark.core.scoring import detect_sample
from denmark.core.model import build_directions, load_generator_model, resolve_mask_id


DATASETS = {
    "longform_qa": ("ELI-5", "longform_qa.jsonl"),
    "finance_qa": ("FINANCE-QA", "finance_qa.jsonl"),
    "alpacafarm": ("ALPACA-FARM", "alpacafarm.jsonl"),
}
NEWLINE_CHAR = "NEWLINE_CHAR"


def clean_text(text: str) -> str:
    return " ".join((text or "").replace(NEWLINE_CHAR, " ").split())


def rep_ngram(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[idx : idx + n]) for idx in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))


def stable_attempt_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def seed_generation_rng(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_chat_prompt(input_text: str, context: str, tokenizer) -> tuple[str, list[int]]:
    user_message = (context + "\n\n" + input_text).strip() if context else input_text.strip()
    try:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if hasattr(encoded, "__contains__") and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        prompt_ids = [int(token_id) for token_id in encoded]
        return tokenizer.decode(prompt_ids, skip_special_tokens=False), prompt_ids
    except Exception:
        prompt_text = (
            "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt_text, tokenizer(prompt_text, add_special_tokens=False)["input_ids"]


def load_selected_rows(path: Path, n: int, seed: int) -> tuple[list[tuple[int, dict]], str]:
    raw_bytes = path.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if n > len(rows):
        raise ValueError(f"{path}: requested {n} rows but only {len(rows)} are available")
    rng = random.Random(seed)
    source_indices = sorted(rng.sample(range(len(rows)), n)) if n < len(rows) else list(range(len(rows)))
    return [(source_idx, rows[source_idx]) for source_idx in source_indices], hashlib.sha256(raw_bytes).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waterbench_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS))
    parser.add_argument("--n_per_dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--source_indices",
        nargs="*",
        type=int,
        default=None,
        help="Optionally generate only these members of the seeded N-sample selection.",
    )
    parser.add_argument(
        "--attempt_offset",
        type=int,
        default=0,
        help="Start deterministic quality-retry attempts at this offset (for targeted refills).",
    )

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

    parser.add_argument("--num_message_bits", type=int, default=2)
    parser.add_argument("--direction_seed", type=int, default=42)
    parser.add_argument("--message_seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_per_dataset <= 0:
        raise ValueError("--n_per_dataset must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_idx < args.num_shards:
        raise ValueError("sharding must satisfy 0 <= shard_idx < num_shards")
    if args.max_new_tokens <= 0 or args.block_size <= 0:
        raise ValueError("generation lengths must be positive")
    if args.max_new_tokens % args.block_size != 0:
        raise ValueError("--block_size must divide --max_new_tokens")
    num_blocks = args.max_new_tokens // args.block_size
    if args.steps <= 0 or args.steps % num_blocks != 0:
        raise ValueError("--steps must be positive and divisible by the number of blocks")
    if args.num_candidates <= 0 or args.candidate_batch_size <= 0:
        raise ValueError("candidate counts must be positive")
    if args.num_message_bits <= 0:
        raise ValueError("--num_message_bits must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    if args.max_retries < 0:
        raise ValueError("--max_retries must be non-negative")
    if args.attempt_offset < 0:
        raise ValueError("--attempt_offset must be non-negative")
    if args.source_indices is not None:
        if any(source_idx < 0 for source_idx in args.source_indices):
            raise ValueError("--source_indices must be non-negative")
        if len(set(args.source_indices)) != len(args.source_indices):
            raise ValueError("--source_indices must be unique")


def main() -> None:
    args = parse_args()
    validate_args(args)
    from transformers import AutoModel, AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    num_blocks = args.max_new_tokens // args.block_size
    directions, signs = build_directions(
        num_blocks,
        args.num_message_bits,
        args.direction_seed,
        args.message_seed,
    )

    print("=== LLaDA block-reject baseline (Block Best-of-N / argmax) ===")
    print(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"datasets={args.datasets} n_per_dataset={args.n_per_dataset} seed={args.seed}")
    print(f"shard={args.shard_idx}/{args.num_shards}")
    print(
        f"K={args.num_candidates} block_size={args.block_size} steps={args.steps} "
        f"T={args.temperature} remasking={args.remasking} candidate_batch={args.candidate_batch_size}"
    )
    print(
        f"encoder={args.encoder_model} B={args.num_message_bits} "
        f"direction_seed={args.direction_seed} message_seed={args.message_seed}"
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
    print(f"model={args.model_name_or_path} mask_id={mask_id}")

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
            missing_source_indices = sorted(requested_source_indices - selected_source_indices)
            if missing_source_indices:
                raise ValueError(
                    f"{dataset_key}: requested source indices are not in the seeded "
                    f"N={args.n_per_dataset} selection: {missing_source_indices}"
                )
        shard_rows = [
            (selected_position, source_idx, record)
            for selected_position, (source_idx, record) in enumerate(selected_rows)
            if selected_position % args.num_shards == args.shard_idx
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
            "method": "block_best_of_k",
            "selection_rule": "argmax_semantic_watermark_score",
            "empty_block_policy": "first_candidate_when_all_semantically_empty",
            "dataset": dataset_key,
            "dataset_file": str(source_path),
            "dataset_sha256": source_sha256,
            "selected_source_indices": [source_idx for source_idx, _ in selected_rows],
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
            "min_response_tokens": args.min_response_tokens,
            "max_rep4": args.max_rep4,
            "max_retries": args.max_retries,
            "attempt_offset": args.attempt_offset,
            "requested_source_indices": (
                sorted(requested_source_indices)
                if requested_source_indices is not None
                else None
            ),
            "num_message_bits": args.num_message_bits,
            "direction_seed": args.direction_seed,
            "message_seed": args.message_seed,
        }
        config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n=== {dataset_name} ({dataset_key}): {len(shard_rows)} prompts -> {output_path} ===")

        with output_path.open("w", encoding="utf-8") as output_file:
            for selected_position, source_idx, record in tqdm(shard_rows, desc=dataset_key):
                input_text = record.get("raw_prompt") or record.get("input", "")
                context = "" if record.get("raw_prompt") else record.get("context", "")
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
                        "block_best_of_k",
                        attempt,
                    )
                    seed_generation_rng(attempt_seed)
                    generated_text, generated_token_ids, block_diagnostics = (
                        llada_generate_block_best_of_k(
                            prompt_tensor,
                            generator,
                            encoder,
                            encoder_tokenizer,
                            generator_tokenizer,
                            directions,
                            signs,
                            mask_id,
                            gen_length=args.max_new_tokens,
                            block_size=args.block_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            num_candidates=args.num_candidates,
                            candidate_batch_size=args.candidate_batch_size,
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

                detection = detect_sample(
                    retokenized_ids,
                    num_blocks,
                    args.block_size,
                    args.max_new_tokens,
                    encoder,
                    encoder_tokenizer,
                    generator_tokenizer,
                    directions,
                    signs,
                    args.device,
                )
                selected_block_scores = [diag["selected_score"] for diag in block_diagnostics]
                finite_selected_scores = [score for score in selected_block_scores if score is not None]
                sample_id = f"{dataset_key}:{source_idx}:{attempt}"
                output_record = {
                    "method": "block_best_of_k",
                    "selection_rule": "argmax_semantic_watermark_score",
                    "empty_block_policy": "first_candidate_when_all_semantically_empty",
                    "dataset": dataset_name,
                    "ds_key": dataset_key,
                    "prompt_idx": selected_position,
                    "selected_position": selected_position,
                    "source_idx": source_idx,
                    "shard": args.shard_idx,
                    "num_shards": args.num_shards,
                    "sample_id": sample_id,
                    "watermark_sample_id": sample_id,
                    "K": args.num_candidates,
                    "B": args.num_message_bits,
                    "prompt_input": input_text,
                    "prompt_context": context,
                    "prompt_full": prompt_text,
                    "text": generated_text,
                    "generated_token_ids": generated_token_ids,
                    "watermarked_text": generated_text,
                    "watermarked_token_ids": generated_token_ids,
                    "retokenized_token_ids": retokenized_ids,
                    "token_len": token_length,
                    "word_len": len(generated_text.split()),
                    "rep4": repetition_4,
                    "passed_quality": passed_quality,
                    "too_short": token_length < args.min_response_tokens,
                    "too_repetitive": repetition_4 > args.max_rep4,
                    "watermarked_token_len": token_length,
                    "watermarked_word_len": len(generated_text.split()),
                    "watermarked_rep4": repetition_4,
                    "watermarked_passed_quality": passed_quality,
                    "watermarked_too_short": token_length < args.min_response_tokens,
                    "watermarked_too_repetitive": repetition_4 > args.max_rep4,
                    "retry_attempts": attempt + 1,
                    "watermarked_retry_attempts": attempt + 1,
                    "retry_attempt_offset": args.attempt_offset,
                    "retry_attempts_this_run": retry_index + 1,
                    "attempt_seed": attempt_seed,
                    "watermarked_attempt_seed": attempt_seed,
                    "block_diagnostics": block_diagnostics,
                    "selected_block_scores": selected_block_scores,
                    "generation_selected_score_mean": (
                        float(np.mean(finite_selected_scores)) if finite_selected_scores else None
                    ),
                    "det_score": detection["det_active"],
                    "block_best_of_k_det_score": detection["det_active"],
                    "watermarked_det_score": detection["det_active"],
                    "watermarked_det_full": detection["det_full"],
                    "watermarked_det_active_blocks": detection["n_active_blocks"],
                    "watermarked_per_block_det": detection["per_block_signed"],
                    "gen_config": run_config,
                    "watermark_config": {
                        "method": "block_best_of_k",
                        "selection_rule": "argmax_semantic_watermark_score",
                        "empty_block_policy": "first_candidate_when_all_semantically_empty",
                        "encoder": args.encoder_model,
                        "block_size": args.block_size,
                        "num_candidates": args.num_candidates,
                        "candidate_batch_size": args.candidate_batch_size,
                        "num_message_bits": args.num_message_bits,
                        "direction_seed": args.direction_seed,
                        "message_seed": args.message_seed,
                    },
                }
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()

        print(f"wrote {len(shard_rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
