#!/usr/bin/env python3
"""Generate PatternMark/OrderAgnostic outputs on WaterBench.

This runner keeps the official PatternMark watermark implementation while
using this repository's generator adapters. In particular, LLaDA2 receives
the 4-D block attention mask required by its remote model code.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denomark.baselines.common import _add_gumbel_noise, _get_num_transfer_tokens
from denomark.core.model import get_model_logits, load_generator_model, resolve_mask_id, safe_decode
from denomark.baselines.clean.generate import (
    DATASETS,
    build_prompt,
    clean_text,
    load_rows,
    rep_ngram,
)


PATTERNS = ((0, 1, 0, 1), (1, 0, 1, 0))
TRANSITION_MATRIX = ((0.0, 1.0), (1.0, 0.0))
INITIAL_STATE = (0.5, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patternmark_repo", type=Path, required=True)
    parser.add_argument("--waterbench_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    parser.add_argument("--n_per_dataset", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--generator_family",
        default="llada",
        choices=("llada", "llada2", "dream"),
    )
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--prompt_mode", choices=("base", "instruct"), default="instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--remasking", choices=("low_confidence", "random", "ar"), default="random")
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_attempts", type=int, default=10)
    parser.add_argument("--max_rep4", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=4.0)
    return parser.parse_args()


def load_patternmark_class(repo: Path):
    source = repo / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"PatternMark source directory not found: {source}")
    sys.path.insert(0, str(source))
    from dlm_watermark.watermarks.order_agnostic import OrderAgnosticWatermark

    return OrderAgnosticWatermark


def watermark_llada2_block_logits(
    watermark,
    full_input_ids: torch.Tensor,
    block_logits: torch.Tensor,
    block_start: int,
    block_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply OrderAgnostic keys to LLaDA2 block logits at absolute positions."""
    if block_logits.shape[1] != block_end - block_start:
        raise ValueError(
            "LLaDA2 block logits are not aligned with the active block: "
            f"logits={block_logits.shape[1]} block={block_end - block_start}"
        )

    # Match OrderAgnosticWatermark.watermark_logits exactly for key sampling:
    # initialize one key for every absolute canvas position, then use the
    # active block's slice. This preserves key/RNG semantics without building
    # a full [batch, prompt+generation, vocab] logits tensor.
    if watermark.key_sequence is None:
        watermark.key_sequence = []
        for _ in range(full_input_ids.shape[1]):
            watermark.current_state = watermark.sample_key(watermark.current_state)
            watermark.key_sequence.append(watermark.current_state)
    if len(watermark.key_sequence) != full_input_ids.shape[1]:
        raise ValueError(
            "PatternMark key sequence length does not match the generation canvas: "
            f"keys={len(watermark.key_sequence)} canvas={full_input_ids.shape[1]}"
        )

    for local_position, key in enumerate(
        watermark.key_sequence[block_start:block_end]
    ):
        block_logits[:, local_position, watermark.greenlists[key]] += watermark.delta
    return block_logits, block_logits


@torch.no_grad()
def generate_patternmark(
    prompt: torch.Tensor,
    model,
    tokenizer,
    watermark,
    *,
    mask_id: int,
    gen_length: int,
    block_size: int,
    steps: int,
    temperature: float,
    remasking: str,
    generator_family: str,
) -> tuple[str, list[int]]:
    if gen_length % block_size:
        raise ValueError("max_new_tokens must be divisible by block_size")
    num_blocks = gen_length // block_size
    if steps % num_blocks:
        raise ValueError("steps must be divisible by the number of generation blocks")
    steps_per_block = steps // num_blocks

    prompt_len = prompt.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_len] = prompt
    eos_token_id = (
        getattr(tokenizer, "eos_token_id", None)
        if generator_family == "llada2"
        else None
    )

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = block_start + block_size
        block_mask = x[:, block_start:block_end] == mask_id
        transfers = _get_num_transfer_tokens(block_mask, steps_per_block)

        for step_idx in range(steps_per_block):
            if generator_family == "llada2":
                block_logits = get_model_logits(
                    model,
                    x,
                    generator_family,
                    logit_start=block_start,
                    logit_end=block_end,
                )
                sampling_logits, remasking_logits = watermark_llada2_block_logits(
                    watermark,
                    x,
                    block_logits,
                    block_start,
                    block_end,
                )
                block_x0 = torch.argmax(
                    _add_gumbel_noise(sampling_logits, temperature),
                    dim=-1,
                )
                active_mask = x[:, block_start:block_end] == mask_id

                if remasking == "low_confidence":
                    probabilities = F.softmax(
                        remasking_logits.to(torch.float64),
                        dim=-1,
                    )
                    block_confidence = torch.gather(
                        probabilities,
                        dim=-1,
                        index=block_x0.unsqueeze(-1),
                    ).squeeze(-1)
                elif remasking == "random":
                    block_confidence = torch.rand(
                        block_x0.shape,
                        device=block_x0.device,
                    )
                elif remasking == "ar":
                    block_confidence = (
                        -torch.arange(block_x0.shape[1], device=block_x0.device)
                        .unsqueeze(0)
                        .expand(block_x0.shape[0], -1)
                        / block_x0.shape[1]
                    )
                else:
                    raise ValueError(f"unsupported remasking mode: {remasking}")

                active_block = x[:, block_start:block_end]
                block_x0 = torch.where(active_mask, block_x0, active_block)
                block_confidence = torch.where(
                    active_mask,
                    block_confidence,
                    torch.full_like(block_confidence, -float("inf")),
                )
                transfer_index = torch.zeros_like(block_x0, dtype=torch.bool)
                for batch_idx in range(block_confidence.shape[0]):
                    count = int(transfers[batch_idx, step_idx].item())
                    if count:
                        selected = torch.topk(
                            block_confidence[batch_idx],
                            k=count,
                        ).indices
                        transfer_index[batch_idx, selected] = True
                active_block[transfer_index] = block_x0[transfer_index]
                continue

            mask_index = x == mask_id
            logits = get_model_logits(model, x, generator_family)
            sampling_logits, remasking_logits = watermark.watermark_logits(
                x,
                logits,
            )
            noisy_logits = _add_gumbel_noise(sampling_logits, temperature)
            x0 = torch.argmax(noisy_logits, dim=-1)

            if remasking == "low_confidence":
                probabilities = F.softmax(remasking_logits.to(torch.float64), dim=-1)
                confidence = torch.gather(
                    probabilities,
                    dim=-1,
                    index=x0.unsqueeze(-1),
                ).squeeze(-1)
            elif remasking == "random":
                confidence = torch.rand(x0.shape, device=x0.device)
            elif remasking == "ar":
                confidence = (
                    -torch.arange(x0.shape[1], device=x0.device)
                    .unsqueeze(0)
                    .expand(x0.shape[0], -1)
                    / x0.shape[1]
                )
            else:
                raise ValueError(f"unsupported remasking mode: {remasking}")

            confidence[:, block_end:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index,
                confidence,
                torch.full_like(confidence, -float("inf")),
            )
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for batch_idx in range(confidence.shape[0]):
                count = int(transfers[batch_idx, step_idx].item())
                if count:
                    selected = torch.topk(confidence[batch_idx], k=count).indices
                transfer_index[batch_idx, selected] = True
            x[transfer_index] = x0[transfer_index]

        if (
            eos_token_id is not None
            and (x[:, prompt_len:block_end] == int(eos_token_id)).any()
        ):
            break

    token_ids = x[0, prompt_len:].tolist()
    if eos_token_id is not None and int(eos_token_id) in token_ids:
        token_ids = token_ids[: token_ids.index(int(eos_token_id)) + 1]
    text = safe_decode(tokenizer, token_ids, skip_special_tokens=True).strip()
    return text, token_ids


@torch.no_grad()
def generate_patternmark_dream(
    prompt: torch.Tensor,
    model,
    tokenizer,
    watermark,
    *,
    mask_id: int,
    gen_length: int,
    steps: int,
    temperature: float,
) -> tuple[str, list[int]]:
    """Apply official PatternMark logits to Dream's native origin decoder."""
    from denomark.baselines.clean.model import dream_origin_diffusion_generate

    def watermark_hook(_step: int, current: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        sampling_logits, _ = watermark.watermark_logits(current, logits)
        return sampling_logits

    output = dream_origin_diffusion_generate(
        model=model,
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        gen_length=gen_length,
        steps=steps,
        temperature=temperature,
        mask_token_id=mask_id,
        generation_logits_hook_func=watermark_hook,
    )
    token_ids = output.sequences[0, prompt.shape[1] :].tolist()
    stop_ids = {
        int(value)
        for value in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        if value is not None
    }
    for index, token_id in enumerate(token_ids):
        if int(token_id) in stop_ids:
            token_ids = token_ids[:index]
            break
    text = safe_decode(tokenizer, token_ids, skip_special_tokens=True).strip()
    return text, token_ids


def color_lookup(watermark) -> torch.Tensor:
    lookup = torch.full((watermark.vocab_size,), -1, dtype=torch.long)
    for color, greenlist in enumerate(watermark.greenlists):
        lookup[greenlist.detach().cpu()] = color
    if (lookup < 0).any():
        raise RuntimeError("PatternMark vocabulary partition is incomplete")
    return lookup


@lru_cache(maxsize=None)
def null_distribution(sequence_length: int) -> tuple[float, ...]:
    pattern_length = len(PATTERNS[0])
    colors = 2
    if sequence_length < pattern_length:
        return (1.0,)

    state_count = colors ** (pattern_length - 1)
    max_hits = sequence_length - pattern_length
    dp = torch.zeros((max_hits + 1, state_count), dtype=torch.float64)
    dp[0, :] = (1.0 / colors) ** (pattern_length - 1)

    for _position in range(pattern_length, sequence_length):
        next_dp = torch.zeros_like(dp)
        for state in range(state_count):
            tail = []
            value = state
            for _ in range(pattern_length - 1):
                tail.append(value % colors)
                value //= colors
            tail.reverse()
            for color in range(colors):
                previous = (color,) + tuple(tail[:-1])
                previous_state = sum(
                    digit * (colors**index)
                    for index, digit in enumerate(reversed(previous))
                )
                hit = (color,) + tuple(tail) in PATTERNS
                if hit:
                    next_dp[1:, state] += dp[:-1, previous_state] / colors
                else:
                    next_dp[:, state] += dp[:, previous_state] / colors
        dp = next_dp
    return tuple(dp.sum(dim=1).tolist())


def detect_patternmark(token_ids: list[int], lookup: torch.Tensor) -> dict:
    valid = [token_id for token_id in token_ids if 0 <= token_id < lookup.numel()]
    colors = lookup[torch.tensor(valid, dtype=torch.long)].tolist() if valid else []
    pattern_length = len(PATTERNS[0])
    pattern_set = set(PATTERNS)
    hits = sum(
        tuple(colors[index - pattern_length : index]) in pattern_set
        for index in range(pattern_length, len(colors))
    )
    distribution = null_distribution(len(colors))
    p_value = math.fsum(distribution[hits:]) if hits < len(distribution) else 0.0
    return {
        "z_score": hits,
        "p_value": min(1.0, max(0.0, p_value)),
        "token_color": colors,
    }


def reset_watermark(watermark, seed: int, temperature: float) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    watermark.set_temperature(temperature)
    watermark.set_mask_token(watermark.mask_token_id)
    watermark.key_sequence = None
    watermark.current_state = watermark.sample_key(None)


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    patternmark_class = load_patternmark_class(args.patternmark_repo)
    model = load_generator_model(
        args.model_name_or_path,
        generator_family=args.generator_family,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map or args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    mask_id = resolve_mask_id(tokenizer, args.generator_family, args.mask_id)
    watermark = patternmark_class(
        l=2,
        transition_matrix=[list(row) for row in TRANSITION_MATRIX],
        initial_state=list(INITIAL_STATE),
        delta=args.delta,
        tokenizer=tokenizer,
        patterns=[list(pattern) for pattern in PATTERNS],
        pattern_length=4,
        device=args.device,
    )
    watermark.mask_token_id = mask_id
    lookup = color_lookup(watermark)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        selected = load_rows(args.waterbench_dir, dataset, args.n_per_dataset, args.seed)
        selected = [
            row for index, row in enumerate(selected)
            if index % args.num_shards == args.shard_idx
        ]
        output = args.output_dir / f"{dataset}.jsonl"
        print(
            f"dataset={dataset} shard={args.shard_idx}/{args.num_shards} "
            f"rows={len(selected)} output={output}",
            flush=True,
        )
        with output.open("w", encoding="utf-8") as handle:
            for source_idx, record in tqdm(selected, desc=dataset):
                generation_started = time.perf_counter()
                canonical_idx = int(record.get("clean_n1000_idx", source_idx))
                if args.generator_family == "dream":
                    from denomark.baselines.clean.model import build_prompt as build_dream_prompt
                    from denomark.core.model import (build_model_input_from_row)

                    model_input, _, _ = build_model_input_from_row(record, dataset, "default")
                    prompt_text, prompt_ids = build_dream_prompt(
                        tokenizer=tokenizer,
                        model_input=model_input,
                        chat=args.prompt_mode == "instruct",
                    )
                else:
                    prompt_text, prompt_ids = build_prompt(
                        record.get("raw_prompt") or record.get("input", ""),
                        "" if record.get("raw_prompt") else record.get("context", ""),
                        tokenizer,
                        args.prompt_mode,
                    )
                prompt = torch.tensor(prompt_ids, dtype=torch.long, device=model.device).unsqueeze(0)
                final_text = ""
                final_tokens: list[int] = []
                token_len = 0
                rep4 = 1.0
                attempt = 0
                final_seed = args.seed
                for attempt in range(1, args.max_attempts + 1):
                    final_seed = args.seed + canonical_idx * 1000 + attempt - 1
                    reset_watermark(watermark, final_seed, args.temperature)
                    if args.generator_family == "dream":
                        final_text, final_tokens = generate_patternmark_dream(
                            prompt,
                            model,
                            tokenizer,
                            watermark,
                            mask_id=mask_id,
                            gen_length=args.max_new_tokens,
                            steps=args.steps,
                            temperature=args.temperature,
                        )
                    else:
                        final_text, final_tokens = generate_patternmark(
                            prompt,
                            model,
                            tokenizer,
                            watermark,
                            mask_id=mask_id,
                            gen_length=args.max_new_tokens,
                            block_size=args.block_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            remasking=args.remasking,
                            generator_family=args.generator_family,
                        )
                    final_text = clean_text(final_text)
                    token_len = len(tokenizer(final_text, add_special_tokens=False)["input_ids"])
                    rep4 = rep_ngram(final_text)
                    if token_len >= args.min_response_tokens and rep4 <= args.max_rep4:
                        break

                generation_seconds = time.perf_counter() - generation_started
                detector = detect_patternmark(final_tokens, lookup)
                row = {
                    "dataset": dataset,
                    "prompt_idx": canonical_idx,
                    "waterbench_idx": canonical_idx,
                    "source_row_idx": int(source_idx),
                    "shard": args.shard_idx,
                    "num_shards": args.num_shards,
                    "prompt_input": record.get("input", ""),
                    "prompt_context": record.get("context", ""),
                    "prompt_full": prompt_text,
                    "text": final_text,
                    "completion": final_text,
                    "token_ids": final_tokens,
                    "token_len": token_len,
                    "length": token_len,
                    "rep4": rep4,
                    "word_rep4": rep4,
                    "passed_quality": token_len >= args.min_response_tokens and rep4 <= args.max_rep4,
                    "too_short": token_len < args.min_response_tokens,
                    "too_repetitive": rep4 > args.max_rep4,
                    "generation_attempts": attempt,
                    "generation_seconds": generation_seconds,
                    "seed": args.seed,
                    "final_seed": final_seed,
                    **detector,
                    "gen_config": {
                        "model": str(args.model_name_or_path),
                        "generator_family": args.generator_family,
                        "mask_id": mask_id,
                        "watermark": "PatternMark/OrderAgnostic",
                        "delta": args.delta,
                        "l": 2,
                        "pattern_length": 4,
                        "patterns": [list(pattern) for pattern in PATTERNS],
                        "temperature": args.temperature,
                        "steps": args.steps,
                        "max_new_tokens": args.max_new_tokens,
                        "block_size": args.block_size,
                        "remasking": args.remasking,
                        "min_response_tokens": args.min_response_tokens,
                        "max_attempts": args.max_attempts,
                        "max_rep4": args.max_rep4,
                        "llada2_attention_boundary": (
                            "active_block_logit_slice_with_absolute_pattern_keys"
                            if args.generator_family == "llada2"
                            else None
                        ),
                        "dream_decoder": (
                            "origin_full_sequence_random_transfer"
                            if args.generator_family == "dream"
                            else None
                        ),
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
