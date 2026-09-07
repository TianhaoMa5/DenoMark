#!/usr/bin/env python3
"""Run unwatermarked LLaDA baseline on WaterBench datasets.
Computes DGMark z_score on the generated text so it can be used
to calibrate the detection threshold τ.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denomark.core.model import GENERATOR_FAMILIES


DATASETS = {
    "longform_qa": "longform_qa.jsonl",
    "finance_qa":  "finance_qa.jsonl",
    "alpacafarm":  "alpacafarm.jsonl",
}

NEWLINE_CHAR = "NEWLINE_CHAR"


def clean_text(text: str) -> str:
    if not text:
        return text
    return " ".join(text.replace(NEWLINE_CHAR, " ").split())


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


def load_source_rows(waterbench_dir: Path, dataset: str) -> list[dict]:
    path = waterbench_dir / DATASETS[dataset]
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_rows(waterbench_dir: Path, dataset: str, n: int, seed: int) -> list[tuple[int, dict]]:
    rows = load_source_rows(waterbench_dir, dataset)
    rng = random.Random(seed)
    if n < len(rows):
        indices = sorted(rng.sample(range(len(rows)), n))
    else:
        indices = list(range(min(n, len(rows))))
    return [(idx, rows[idx]) for idx in indices]


def read_existing_prompts(path: Path, dataset: str) -> set[str]:
    if not path:
        return set()
    candidates = [
        path / dataset / f"{dataset}.jsonl",
        path / f"{dataset}.jsonl",
    ]
    ds_dir = path / dataset
    if ds_dir.exists():
        candidates.extend(sorted(ds_dir.glob(f"{dataset}_llada_native_clean_valid*.jsonl")))
        candidates.extend(sorted(ds_dir.glob("shard_*/*.jsonl")))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waterbench_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--n_per_dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=None)
    parser.add_argument("--exclude_output_dir", type=Path, default=None,
                        help="Directory containing existing dataset JSONL outputs; prompts are skipped.")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--prompt_mode", choices=["base", "instruct"], default="instruct")
    parser.add_argument("--generator_family", default="llada", choices=GENERATOR_FAMILIES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None,
                        help="HF device_map for generator loading (default: same as --device; use 'auto' for 26B sharding).")
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="random")
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument(
        "--max_rep4",
        type=float,
        default=0.2,
        help="Retry if the word-level repeated 4-gram ratio exceeds this value.",
    )
    # DGMark detection params (for computing z_score on unwatermarked text)
    parser.add_argument("--dgmark_private_key", default=None)
    parser.add_argument("--dgmark_window_size", type=int, default=8)
    parser.add_argument(
        "--skip_dgmark_detection",
        action="store_true",
        help="Generate clean text without computing DGMark-specific diagnostic fields.",
    )
    return parser.parse_args()


def main() -> None:
    family_parser = argparse.ArgumentParser(add_help=False)
    family_parser.add_argument("--generator_family", default="llada")
    family, _ = family_parser.parse_known_args()
    if family.generator_family == "dream":
        from denomark.baselines.clean.model import generate_dream
        return generate_dream()
    args = parse_args()
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from denomark.baselines.common import llada_generate_unwatermarked
    from denomark.core.model import load_generator_model, resolve_mask_id
    if not args.skip_dgmark_detection:
        from denomark.baselines.dgmark.model import _score_dgmark_tokens, _dgmark_window_scores

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name_or_path}", flush=True)
    model = load_generator_model(
        args.model_name_or_path,
        generator_family=args.generator_family,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map or args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    mask_id = resolve_mask_id(tokenizer, args.generator_family, args.mask_id)
    print(f"generator_family={args.generator_family} mask_id={mask_id}", flush=True)

    private_key = args.dgmark_private_key
    if private_key is not None:
        try:
            private_key = int(private_key)
        except ValueError:
            pass

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
            for source_idx, rec in load_rows(args.waterbench_dir, dataset, args.n_per_dataset, args.seed):
                prompt_str, prompt_ids = build_prompt(
                    rec.get("raw_prompt") or rec.get("input", ""),
                    "" if rec.get("raw_prompt") else rec.get("context", ""),
                    tokenizer,
                    args.prompt_mode,
                )
                rows.append((source_idx, rec, prompt_str, prompt_ids))
        rows = rows[args.start_offset:args.end_offset]
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_idx]

        out_path = args.output_dir / f"{dataset}.jsonl"
        print(f"Dataset {dataset}: {len(rows)} prompts -> {out_path}", flush=True)

        with out_path.open("w", encoding="utf-8") as out_f:
            for local_idx, (source_idx, rec, prompt_str, prompt_ids) in enumerate(tqdm(rows, desc=dataset)):
                generation_started = time.perf_counter()
                canonical_idx = int(rec.get("clean_n1000_idx", source_idx))
                prompt_tensor = torch.tensor(
                    prompt_ids, dtype=torch.long, device=args.device,
                ).unsqueeze(0)

                rep4 = 0.0
                final_seed = int(args.seed)
                for attempt in range(args.max_retries + 1):
                    # Make retries reproducible and independent of shard layout.
                    final_seed = int(args.seed) + canonical_idx * 1000 + attempt
                    torch.manual_seed(final_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(final_seed)
                    text, tokens = llada_generate_unwatermarked(
                        prompt_tensor,
                        model,
                        tokenizer,
                        mask_id,
                        gen_length=args.max_new_tokens,
                        block_size=args.block_size,
                        steps=args.steps,
                        temperature=args.temperature,
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

                generation_seconds = time.perf_counter() - generation_started
                # Compute DGMark diagnostics separately from generation timing.
                detection = None
                if not args.skip_dgmark_detection:
                    detection = _score_dgmark_tokens(
                        tokens,
                        len(prompt_ids),
                        private_key,
                        eot_token_id=tokenizer.eos_token_id,
                    )
                    detection.update(
                        _dgmark_window_scores(
                            tokens,
                            len(prompt_ids),
                            args.dgmark_window_size,
                            private_key,
                        )
                    )

                row = {
                    "dataset": dataset,
                    "prompt_idx": canonical_idx,
                    "waterbench_idx": canonical_idx,
                    "source_row_idx": source_idx,
                    "shard": args.shard_idx,
                    "num_shards": args.num_shards,
                    "prompt_input": rec.get("input", ""),
                    "prompt_context": rec.get("context", ""),
                    "prompt_full": prompt_str,
                    "text": text_clean,
                    "token_ids": tokens,
                    "token_len": token_len,
                    "generation_seconds": generation_seconds,
                    "word_len": len(text_clean.split()),
                    "rep4": rep4,
                    "too_short": token_len < args.min_response_tokens,
                    "too_repetitive": args.max_rep4 is not None and rep4 > args.max_rep4,
                    "passed_quality": (
                        token_len >= args.min_response_tokens
                        and (args.max_rep4 is None or rep4 <= args.max_rep4)
                    ),
                    "retry_attempts": attempt + 1,
                    "seed": int(args.seed),
                    "final_seed": final_seed,
                    "dgmark_detector": detection,
                    "gen_config": {
                        "model": str(args.model_name_or_path),
                        "generator_family": args.generator_family,
                        "mask_id": mask_id,
                        "watermark": "none",
                        "temperature": args.temperature,
                        "diffusion_steps": args.steps,
                        "max_new_tokens": args.max_new_tokens,
                        "block_size": args.block_size,
                        "remasking": args.remasking,
                        "min_response_tokens": args.min_response_tokens,
                        "max_retries": args.max_retries,
                        "max_rep4": args.max_rep4,
                    },
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
