#!/usr/bin/env python3
"""Run one resumable model/method/task slice of the paper downstream evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denmark.baselines.common import llada_generate_unwatermarked
from denmark.baselines.dgmark.model import llada_generate_dgmark
from denmark.core.generate import build_directions, generate_one
from denmark.baselines.dlm_kgw.model import HashDistribution, llada_generate_hash_distribution
from denmark.core.model import load_generator_model, resolve_mask_id, safe_decode
from denmark.experiments.downstream.tasks import (
    BENCHMARKS,
    MODEL_DEFAULTS,
    TASK_SETTINGS,
    build_prompt,
    iter_records,
    score_non_code,
)
from denmark.baselines.patternmark.generate import (
    INITIAL_STATE,
    PATTERNS,
    TRANSITION_MATRIX,
    generate_patternmark,
    load_patternmark_class,
    reset_watermark,
)


METHODS = ("unwatermarked", "hash", "patternmark", "ours", "dgmark", "umr")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("llada8b", "llada15"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "configs/paper_experiments.json",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--encoder-model", default=os.environ.get("ENCODER_MODEL"))
    parser.add_argument("--patternmark-repo", type=Path, default=None)
    parser.add_argument("--umr-root", type=Path, default=None)
    parser.add_argument("--umr-bitmap-path", type=Path, default=None)
    parser.add_argument("--umr-ratio", type=float, default=0.5)
    parser.add_argument("--umr-delta", type=float, default=10.0)
    parser.add_argument("--umr-key", type=int, default=42)
    parser.add_argument("--umr-watermark-str", default="1001")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--remasking",
        choices=("random", "low_confidence"),
        default="random",
    )
    parser.add_argument(
        "--dgmark-sampling-strategy",
        choices=("greedy", "multinomial"),
        default="multinomial",
    )
    parser.add_argument("--gen-length", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--block-length", type=int)
    parser.add_argument("--ours-k", type=int, default=16)
    parser.add_argument("--ours-b", type=int, default=2)
    parser.add_argument("--ours-c", type=int, default=2)
    parser.add_argument("--ours-r", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trim_at_eos(tokens: list[int], tokenizer, mask_id: int) -> list[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    output = []
    for token in tokens:
        if int(token) == int(mask_id):
            continue
        output.append(int(token))
        if eos is not None and int(token) == int(eos):
            break
    return output


def make_chat_prompt(tokenizer, prompt: str, device: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        ids = tokenizer(prompt, return_tensors="pt").input_ids
    return ids.to(device)


def patternmark_instance(repo: Path, tokenizer, mask_id: int, delta: float, device: str):
    cls = load_patternmark_class(repo)
    watermark = cls(
        l=2,
        transition_matrix=[list(row) for row in TRANSITION_MATRIX],
        initial_state=list(INITIAL_STATE),
        delta=delta,
        tokenizer=tokenizer,
        patterns=[list(pattern) for pattern in PATTERNS],
        pattern_length=4,
        device=device,
    )
    watermark.mask_token_id = mask_id
    return watermark


class ForwardCounter:
    def __init__(self, model) -> None:
        self.count = 0
        self.handle = model.register_forward_pre_hook(self._hook)

    def _hook(self, _module, _inputs) -> None:
        self.count += 1

    def close(self) -> None:
        self.handle.remove()


def load_components(args: argparse.Namespace):
    model_path = args.model_path or MODEL_DEFAULTS[args.model]
    if not model_path:
        raise ValueError("--model-path is required; public scripts do not assume cluster paths")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = load_generator_model(
        model_path,
        generator_family="llada",
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    mask_id = resolve_mask_id(tokenizer, "llada")
    encoder = encoder_tokenizer = None
    if args.method == "ours":
        if not args.encoder_model:
            raise ValueError("--encoder-model is required for ours")
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model)
        encoder = AutoModel.from_pretrained(args.encoder_model, torch_dtype=torch.float32).to(args.device).eval()
    if args.method == "umr":
        if args.model not in {"llada8b", "llada15"}:
            raise ValueError("downstream UMR supports the two paper LLaDA-family backbones")
        if args.umr_root is None or not (args.umr_root / "watermark" / "watermark_config.py").is_file():
            raise ValueError("--umr-root must point to an upstream UMR checkout")
        if args.umr_bitmap_path is None or not args.umr_bitmap_path.is_file():
            raise ValueError("--umr-bitmap-path must point to the persistent UMR bitmap")
        sys.path.insert(0, str(args.umr_root.resolve()))
        umr_module = importlib.import_module("watermark.LLaDA.watermark_gen_UMR_L")
        from denmark.baselines.umr.generate import install_llada_gumbel_score_compat

        if args.temperature > 0:
            install_llada_gumbel_score_compat(umr_module, mask_id)
        watermark_config = importlib.import_module(
            "watermark.watermark_config"
        ).WatermarkConfig
        persistent_bitmap = importlib.import_module(
            "watermark.bitmap_persistent"
        ).PersistentBitmap
        watermark_class = importlib.import_module("watermark.watermarklogit").Watermark
        vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
        bitmap = persistent_bitmap(vocab_size, str(args.umr_bitmap_path), device=args.device)
        umr_config = watermark_config(
            vocab_size=vocab_size,
            ratio=args.umr_ratio,
            delta=args.umr_delta,
            key=args.umr_key,
            prebias=False,
            strategy="bidirectional",
            bitmap_path=str(args.umr_bitmap_path),
            watermark_str=args.umr_watermark_str,
        )
        watermark = watermark_class(umr_config, bitmap)

        def is_token_in_green_list(token: int, previous: int):
            return bool(bitmap.get_bit(int(previous), int(token))), None

        watermark.is_token_in_green_list = is_token_in_green_list
        args.umr_generate = umr_module.generate
        args.umr_watermark = watermark
    return model_path, model, tokenizer, mask_id, encoder, encoder_tokenizer


def _hash(tokenizer, temperature: float, mask_id: int, device: str) -> HashDistribution:
    watermark = HashDistribution(
        tokenizer,
        delta=4.0,
        convolution_kernel=[-1],
        greenlist_type="bernoulli",
        greenlist_params={"gamma": 0.25},
        seeding_scheme="sumhash",
        device=device,
    )
    watermark.set_temperature(temperature)
    watermark.set_mask_token(mask_id)
    return watermark


def generate_llada(args, model, tokenizer, mask_id, encoder, encoder_tokenizer, prompt_ids, sample_id):
    gen_length, steps, block_length = TASK_SETTINGS["llada"][args.benchmark]
    gen_length = args.gen_length or gen_length
    steps = args.steps or steps
    block_length = args.block_length or block_length
    if args.method == "unwatermarked":
        text, tokens = llada_generate_unwatermarked(
            prompt_ids, model, tokenizer, mask_id,
            gen_length=gen_length, block_size=block_length, steps=steps,
            temperature=args.temperature, remasking=args.remasking, generator_family="llada",
        )
    elif args.method == "hash":
        text, tokens, _ = llada_generate_hash_distribution(
            prompt_ids, model, tokenizer, mask_id, _hash(tokenizer, args.temperature, mask_id, args.device),
            gen_length=gen_length, block_size=block_length, steps=steps,
            temperature=args.temperature, remasking=args.remasking, generator_family="llada",
        )
    elif args.method == "patternmark":
        if args.patternmark_repo is None:
            raise ValueError("--patternmark-repo is required")
        watermark = patternmark_instance(args.patternmark_repo, tokenizer, mask_id, 4.0, args.device)
        reset_watermark(watermark, args.seed, args.temperature)
        text, tokens = generate_patternmark(
            prompt_ids, model, tokenizer, watermark,
            mask_id=mask_id, gen_length=gen_length, block_size=block_length,
            steps=steps, temperature=args.temperature, remasking=args.remasking, generator_family="llada",
        )
    elif args.method == "dgmark":
        text, tokens, _ = llada_generate_dgmark(
            prompt_ids, model, tokenizer, mask_id,
            gen_length=gen_length, block_size=block_length, steps=steps,
            temperature=args.temperature, remasking="low_confidence",
            sampling_strategy=args.dgmark_sampling_strategy, top_k=10, beam_size=10,
            private_key=None,
            eot_token_id=getattr(tokenizer, "eos_token_id", None), generator_family="llada",
        )
    elif args.method == "umr":
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        generated = args.umr_generate(
            model,
            prompt=prompt_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=args.temperature,
            remasking="low_confidence",
            mask_id=mask_id,
            watermark=args.umr_watermark,
            min_eos_tokens=0,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        tokens = generated[0, prompt_ids.shape[1]:prompt_ids.shape[1] + gen_length].detach().cpu().tolist()
        text = safe_decode(tokenizer, trim_at_eos(tokens, tokenizer, mask_id), skip_special_tokens=True).strip()
    else:
        semantic_block = min(25, gen_length)
        num_blocks = (gen_length + semantic_block - 1) // semantic_block
        directions, signs = build_directions(num_blocks, args.ours_b, args.seed, 0)
        text, tokens, _ = generate_one(
            prompt_ids, model, encoder, encoder_tokenizer, tokenizer,
            directions, signs, mask_id, gen_length, args.temperature,
            semantic_block, 1, args.ours_k, args.ours_c, args.device,
            perturb_temperature=0.6,
            rollouts_per_cand=args.ours_r,
            rollout_temperature=0.5,
            rollout_schedule="linear_decay",
            dedup_candidates=True,
            per_cand_positions=True,
            position_selection="random",
            shared_rollout_seeds=True,
            decode_schedule="sequential",
            generator_family="llada",
            decode_block_size=block_length,
            sample_id=sample_id,
        )
    tokens = trim_at_eos(tokens, tokenizer, mask_id)
    return safe_decode(tokenizer, tokens, skip_special_tokens=True).strip(), tokens


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["sample_id"])

    model_path, model, tokenizer, mask_id, encoder, encoder_tokenizer = load_components(args)
    counter = ForwardCounter(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        with args.output.open("a", encoding="utf-8") as handle:
            for record in iter_records(manifest, args.benchmark, args.limit):
                if record["sample_id"] in completed:
                    continue
                seed_all(args.seed)
                prompt = build_prompt(record)
                prompt_ids = make_chat_prompt(tokenizer, prompt, args.device)
                started = time.perf_counter()
                before_forwards = counter.count
                error = None
                raw_completion = ""
                tokens: list[int] = []
                parsed = None
                correct = False
                try:
                    raw_completion, tokens = generate_llada(
                        args, model, tokenizer, mask_id, encoder, encoder_tokenizer,
                        prompt_ids, record["sample_id"],
                    )
                    parsed, correct = score_non_code(record, raw_completion)
                    if parsed is None:
                        error = "unparsable_answer"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
                row = {
                    "model": args.model,
                    "model_checkpoint": model_path,
                    "method": args.method,
                    "benchmark": args.benchmark,
                    "sample_id": record["sample_id"],
                    "prompt": prompt,
                    "reference_answer": record["reference_answer"],
                    "raw_completion": raw_completion,
                    "parsed_answer": parsed,
                    "correct": bool(correct),
                    "generated_tokens": len(tokens),
                    "generated_token_ids": tokens,
                    "generation_seconds": elapsed,
                    "forward_passes": counter.count - before_forwards,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                    "seed": args.seed,
                    "config_path": str(args.config_path),
                    "error": error,
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    finally:
        counter.close()


if __name__ == "__main__":
    main()
