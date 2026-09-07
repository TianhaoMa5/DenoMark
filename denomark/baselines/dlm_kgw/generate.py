#!/usr/bin/env python3
"""Run the paper's DLM-KGW baseline on selected WaterBench datasets."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

_real_find_spec = importlib.util.find_spec


def _find_spec_without_sklearn(name, package=None):
    # Transformers only uses sklearn for optional generation helpers. This
    # keeps a broken local sklearn install from blocking HashDistribution runs.
    if name == "sklearn" or name.startswith("sklearn."):
        return None
    return _real_find_spec(name, package)


importlib.util.find_spec = _find_spec_without_sklearn

from transformers import AutoTokenizer

from denomark.core.scoring import clean_text
from denomark.baselines.dlm_kgw.model import (
    HashDistribution,
    generate_hash_distribution,
)
from denomark.core.model import GENERATOR_FAMILIES, load_generator_model, resolve_mask_id


DATASETS = {
    "longform_qa": "longform_qa.jsonl",
    "finance_qa": "finance_qa.jsonl",
    "alpacafarm": "alpacafarm.jsonl",
}


def build_prompt(input_text: str, context: str, tokenizer, prompt_mode: str) -> tuple[str, list[int]]:
    user_msg = input_text.strip()
    if context:
        user_msg = (context + "\n\n" + input_text).strip()
    if prompt_mode == "base":
        ids = tokenizer(user_msg, add_special_tokens=True)["input_ids"]
        return user_msg, list(ids)
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        return tokenizer.decode(ids, skip_special_tokens=False), list(ids)
    except Exception:
        prompt = (
            "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        return prompt, tokenizer(prompt, add_special_tokens=False)["input_ids"]


def load_source_rows(waterbench_dir: Path, dataset: str):
    return [json.loads(line) for line in open(waterbench_dir / DATASETS[dataset]) if line.strip()]


def load_rows(waterbench_dir: Path, dataset: str, n: int, seed: int):
    rows = load_source_rows(waterbench_dir, dataset)
    rng = random.Random(seed)
    if n < len(rows):
        rows = [rows[i] for i in sorted(rng.sample(range(len(rows)), n))]
    return rows[:n]


def read_existing_prompts(path: Path, dataset: str) -> set[str]:
    if not path:
        return set()
    candidates = [
        path / dataset / f"{dataset}.jsonl",
        path / f"{dataset}.jsonl",
    ]
    ds_dir = path / dataset
    if ds_dir.exists():
        candidates.extend(sorted(ds_dir.glob("shard_*/*.jsonl")))
        candidates.extend(sorted(ds_dir.glob("*_shards/shard_*/*.jsonl")))
    prompts = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt_full")
                if prompt:
                    prompts.add(prompt)
    return prompts


def rep_ngram(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(v - 1 for v in counts.values() if v > 1) / max(1, len(grams))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waterbench_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(DATASETS))
    parser.add_argument("--n_per_dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_offset", type=int, default=0,
                        help="Slice start in the selected prompt list, for sharded continuation runs.")
    parser.add_argument("--end_offset", type=int, default=None,
                        help="Slice end in the selected prompt list, for sharded continuation runs.")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--exclude_output_dir", type=Path, default=None,
                        help="Directory containing existing dataset JSONL outputs; prompts are skipped.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--prompt_mode", choices=["base", "instruct"], default="instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator_family", default="llada", choices=GENERATOR_FAMILIES,
                        help="Generator logits adapter.")
    parser.add_argument("--mask_id", type=int, default=None,
                        help="MASK token id. Defaults to tokenizer.mask_token_id, then family fallback.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--remasking", choices=["low_confidence", "random", "none"], default="random")
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument(
        "--max_rep4",
        type=float,
        default=0.2,
        help="Retry if repeated 4-gram ratio is above this value. Default keeps old behavior.",
    )
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n_iter", type=int, default=1)
    parser.add_argument("--convolution_kernel", type=int, nargs="+", default=[-1])
    parser.add_argument("--seeding_scheme", default="sumhash", choices=["sumhash", "minhash"])
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name_or_path}", flush=True)
    model = load_generator_model(
        args.model_name_or_path,
        generator_family=args.generator_family,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    mask_id = resolve_mask_id(tokenizer, args.generator_family, args.mask_id)
    print(f"generator_family={args.generator_family} mask_id={mask_id}", flush=True)

    watermark = HashDistribution(
        tokenizer=tokenizer,
        delta=args.delta,
        enforce_kl=False,
        convolution_kernel=args.convolution_kernel,
        greenlist_type="bernoulli",
        greenlist_params={"gamma": args.gamma},
        topk=args.topk,
        n_iter=args.n_iter,
        seeding_scheme=args.seeding_scheme,
        device=args.device,
    )

    for dataset in args.datasets:
        if args.exclude_output_dir:
            excluded = read_existing_prompts(args.exclude_output_dir, dataset)
            source_rows = load_source_rows(args.waterbench_dir, dataset)
            candidates = []
            for waterbench_idx, rec in enumerate(source_rows):
                prompt_str, prompt_ids = build_prompt(
                    rec.get("raw_prompt") or rec.get("input", ""),
                    "" if rec.get("raw_prompt") else rec.get("context", ""),
                    tokenizer,
                    args.prompt_mode,
                )
                if prompt_str not in excluded:
                    candidates.append((waterbench_idx, rec, prompt_str, prompt_ids))
            rng = random.Random(args.seed)
            if args.n_per_dataset < len(candidates):
                selected_idx = sorted(rng.sample(range(len(candidates)), args.n_per_dataset))
                rows = [candidates[i] for i in selected_idx]
            else:
                rows = candidates[: args.n_per_dataset]
            if len(rows) < args.n_per_dataset:
                raise RuntimeError(
                    f"{dataset}: requested {args.n_per_dataset} non-duplicate prompts, "
                    f"but only found {len(rows)} after excluding {len(excluded)}."
                )
        else:
            rows = []
            for prompt_idx, rec in enumerate(load_rows(args.waterbench_dir, dataset, args.n_per_dataset, args.seed)):
                prompt_str, prompt_ids = build_prompt(
                    rec.get("raw_prompt") or rec.get("input", ""),
                    "" if rec.get("raw_prompt") else rec.get("context", ""),
                    tokenizer,
                    args.prompt_mode,
                )
                rows.append((prompt_idx, rec, prompt_str, prompt_ids))
        if args.start_offset or args.end_offset is not None:
            rows = rows[args.start_offset:args.end_offset]
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_idx]
        out_path = args.output_dir / f"{dataset}.jsonl"
        print(f"Dataset {dataset}: {len(rows)} prompts -> {out_path}", flush=True)

        with open(out_path, "w", encoding="utf-8") as out_f:
            for prompt_idx, rec, prompt_str, prompt_ids in tqdm(rows, desc=dataset):
                generation_started = time.perf_counter()
                prompt_ids_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)

                rep4 = 0.0
                for attempt in range(args.max_retries + 1):
                    text, tokens, detection = generate_hash_distribution(
                        prompt_ids_tensor,
                        model,
                        tokenizer,
                        mask_id,
                        watermark,
                        gen_length=args.max_new_tokens,
                        block_size=args.block_size,
                        steps=args.steps,
                        temperature=args.temperature,
                        cfg_scale=0.0,
                        remasking=args.remasking,
                        generator_family=args.generator_family,
                    )
                    text_clean = clean_text(text)
                    token_len = len(tokenizer(text_clean, add_special_tokens=False)["input_ids"])
                    rep4 = rep_ngram(text_clean, n=4)
                    long_enough = token_len >= args.min_response_tokens
                    not_repetitive = args.max_rep4 is None or rep4 <= args.max_rep4
                    if long_enough and not_repetitive:
                        break

                row = {
                    "dataset": dataset,
                    "prompt_idx": prompt_idx,
                    "waterbench_idx": prompt_idx,
                    "prompt_full": prompt_str,
                    "hash_distribution_text": text_clean,
                    "hash_distribution_token_ids": tokens,
                    "hash_distribution_token_len": token_len,
                    "hash_distribution_word_len": len(text_clean.split()),
                    "hash_distribution_rep4": rep4,
                    "hash_distribution_too_short": token_len < args.min_response_tokens,
                    "hash_distribution_too_repetitive": args.max_rep4 is not None and rep4 > args.max_rep4,
                    "hash_distribution_passed_quality": (
                        token_len >= args.min_response_tokens
                        and (args.max_rep4 is None or rep4 <= args.max_rep4)
                    ),
                    "hash_distribution_retry_attempts": attempt + 1,
                    "generation_seconds": time.perf_counter() - generation_started,
                    "hash_distribution_detector": {
                        "n_trials": detection["n_trials"],
                        "statistic": detection["statistic"],
                        "z_score": detection["z_score"],
                        "p_value": detection["p_value"],
                    },
                    "hash_distribution_config": watermark.get_key_params(),
                    "gen_config": {
                        "model": str(args.model_name_or_path),
                        "prompt_mode": args.prompt_mode,
                        "generator_family": args.generator_family,
                        "decoder": "dream_origin" if args.generator_family == "dream" else "blockwise",
                        "mask_id": mask_id,
                        "temperature": args.temperature,
                        "diffusion_steps": args.steps,
                        "max_new_tokens": args.max_new_tokens,
                        "response_length_filter": [args.min_response_tokens, args.max_new_tokens],
                        "remasking": args.remasking,
                        "max_rep4": args.max_rep4,
                    },
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
