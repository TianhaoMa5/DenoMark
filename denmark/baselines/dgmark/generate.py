#!/usr/bin/env python3
"""Run DGMark on selected WaterBench-style datasets without a block encoder."""
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

from denmark.core.model import GENERATOR_FAMILIES


DATASETS = {
    "longform_qa": "longform_qa.jsonl",
    "finance_qa": "finance_qa.jsonl",
    "alpacafarm": "alpacafarm.jsonl",
}
DEFAULT_DATASETS = ["finance_qa", "alpacafarm", "longform_qa"]

NEWLINE_CHAR = "NEWLINE_CHAR"


def clean_text(text: str) -> str:
    if not text:
        return text
    return " ".join(text.replace(NEWLINE_CHAR, " ").split())


def build_chat_prompt(input_text: str, context: str, tokenizer) -> tuple[str, list[int]]:
    user_msg = (context + "\n\n" + input_text).strip() if context else input_text.strip()
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


def load_rows(
    waterbench_dir: Path,
    dataset: str,
    n: int,
    seed: int,
    allow_short_dataset: bool = False,
) -> list[tuple[int, dict]]:
    rows = load_source_rows(waterbench_dir, dataset)
    if n > len(rows) and not allow_short_dataset:
        raise ValueError(
            f"{dataset}: requested n_per_dataset={n}, but {waterbench_dir / DATASETS[dataset]} "
            f"has only {len(rows)} rows. Use the 500-prompt WaterBench root or pass "
            "--allow_short_dataset for an intentional partial run."
        )
    rng = random.Random(seed)
    if n < len(rows):
        indices = sorted(rng.sample(range(len(rows)), n))
    else:
        indices = list(range(min(n, len(rows))))
    return [(idx, rows[idx]) for idx in indices]


def load_rows_by_source_indices(
    waterbench_dir: Path,
    dataset: str,
    source_indices_file: Path,
) -> list[tuple[int, dict]]:
    rows = load_source_rows(waterbench_dir, dataset)
    indices = [
        int(line.strip())
        for line in source_indices_file.open(encoding="utf-8")
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(indices) != len(set(indices)):
        raise ValueError(f"duplicate source indices in {source_indices_file}")
    invalid = [idx for idx in indices if idx < 0 or idx >= len(rows)]
    if invalid:
        raise ValueError(
            f"{dataset}: source indices outside [0, {len(rows) - 1}]: {invalid[:20]}"
        )
    return [(idx, rows[idx]) for idx in indices]


def rep_ngram(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(v - 1 for v in counts.values() if v > 1) / max(1, len(grams))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--waterbench_dir",
        type=Path,
        required=True,
        help="Directory containing locally constructed WaterBench prompt JSONL files.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=sorted(DATASETS),
        help=f"Datasets to run. Defaults to the 500-prompt WMDLM set: {', '.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument("--n_per_dataset", type=int, default=100)
    parser.add_argument(
        "--allow_short_dataset",
        action="store_true",
        help="Allow n_per_dataset to exceed the source row count. By default this is an error.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=None)
    parser.add_argument(
        "--source_indices_file",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited source row indices. When provided, generate exactly "
            "these prompts instead of drawing a fresh n_per_dataset sample."
        ),
    )
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generator_family", default="llada", choices=GENERATOR_FAMILIES,
                        help="Generator logits adapter.")
    parser.add_argument("--mask_id", type=int, default=None,
                        help="MASK token id. Defaults to tokenizer.mask_token_id, then family fallback.")
    parser.add_argument("--eot_token_id", type=int, default=None,
                        help="End token used for DGMark trimming. Defaults to tokenizer.eos_token_id.")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--alg", "--remasking_alg", dest="alg", default="origin",
                        help="DREAM origin remasking algorithm.")
    parser.add_argument("--alg_temp", type=float, default=0.1,
                        help="DREAM alg_temp kept for eth-sri config compatibility.")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="DREAM diffusion timestep lower bound.")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Optional DREAM top-p filter. Defaults to eth-sri None.")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Optional DREAM top-k filter. Defaults to eth-sri None.")
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True,
                        help="Use tokenizer.apply_chat_template for DREAM prompts.")
    parser.add_argument("--prompt_variant", default="default", choices=["default", "long_output"],
                        help="WaterBench prompt variant for DREAM dataset templates.")
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument(
        "--max_rep4",
        type=float,
        default=0.2,
        help="Optional maximum repeated 4-gram ratio used by the retry filter.",
    )
    parser.add_argument(
        "--dgmark_decoding",
        choices=["auto", "legacy", "aligned", "dream_origin", "dream_dgmark"],
        default="auto",
        help=(
            "DGMark decoding adapter. Auto uses aligned for LLaDA and dream_dgmark for Dream; "
            "dream_dgmark uses DGMark's parity-guided decoder with ETH DREAM shifted logits."
        ),
    )
    parser.add_argument("--dgmark_sampling_strategy", default="multinomial",
                        choices=["greedy", "multinomial"])
    parser.add_argument("--dgmark_top_k", type=int, default=None,
                        help="Default: 32 for Dream, 10 for LLaDA.")
    parser.add_argument("--dgmark_beam_size", type=int, default=None,
                        help="Default: 32 for Dream, 10 for LLaDA.")
    parser.add_argument("--dgmark_private_key", default=None)
    parser.add_argument("--dgmark_window_size", type=int, default=8)
    parser.add_argument("--dgmark_position_bonus", type=float, default=1e6,
                        help="Large confidence bonus for DGMark-compliant LLaDA positions.")
    parser.add_argument("--dgmark_delta", type=float, default=2.0,
                        help="DREAM-origin parity logit bias for DGMark.")
    parser.add_argument("--run_dgmark", action="store_true",
                        help="Accepted for compatibility; this script always runs DGMark.")
    return parser.parse_args()


def resolve_decoding(args) -> str:
    """Select paper defaults while preserving explicit parameter overrides."""
    dream = args.generator_family == "dream"
    decoding = args.dgmark_decoding
    if decoding == "auto" or (dream and decoding == "aligned"):
        decoding = "dream_dgmark" if dream else "aligned"
    if decoding in {"dream_origin", "dream_dgmark"} and not dream:
        raise ValueError(f"--dgmark_decoding {decoding} requires --generator_family dream")
    if args.dgmark_top_k is None:
        args.dgmark_top_k = 32 if dream else 10
    if args.dgmark_beam_size is None:
        args.dgmark_beam_size = 32 if dream else 10
    return decoding


def main() -> None:
    args = parse_args()
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def find_spec_without_sklearn(name, package=None):
        # Transformers only uses sklearn for optional generation helpers. This
        # keeps a broken local sklearn install from blocking DGMark generation.
        if name == "sklearn" or name.startswith("sklearn."):
            return None
        return real_find_spec(name, package)

    importlib.util.find_spec = find_spec_without_sklearn
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from denmark.baselines.dgmark.model import (
        DGMarkLogitsBias,
        _dgmark_window_scores,
        _score_dgmark_tokens,
        llada_generate_dgmark,
        llada_generate_dgmark_aligned,
    )
    from denmark.core.model import load_generator_model, resolve_mask_id

    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_decoding = resolve_decoding(args)
    if args.generator_family == "dream":
        from denmark.baselines.clean.model import (
            build_prompt as build_dream_origin_prompt,
            dream_origin_diffusion_generate,
        )
        from denmark.core.model import (build_model_input_from_row, safe_decode_dream as dream_safe_decode)

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
    eot_token_id = args.eot_token_id
    if eot_token_id is None:
        eot_token_id = getattr(tokenizer, "eos_token_id", None)
    special_ids = {
        int(mask_id),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    }
    special_ids = {int(x) for x in special_ids if x is not None}
    print(
        f"generator_family={args.generator_family} dgmark_decoding={effective_decoding} "
        f"mask_id={mask_id} eot_token_id={eot_token_id}",
        flush=True,
    )
    print(f"waterbench_dir={args.waterbench_dir}", flush=True)

    for dataset in args.datasets:
        if args.source_indices_file is not None:
            rows = load_rows_by_source_indices(
                args.waterbench_dir,
                dataset,
                args.source_indices_file,
            )
        else:
            rows = load_rows(
                args.waterbench_dir,
                dataset,
                args.n_per_dataset,
                args.seed,
                allow_short_dataset=args.allow_short_dataset,
            )
        rows = rows[args.start_offset:args.end_offset]
        rows = [row for i, row in enumerate(rows) if i % args.num_shards == args.shard_idx]

        out_path = args.output_dir / f"{dataset}.jsonl"
        print(f"Dataset {dataset}: {len(rows)} prompts -> {out_path}", flush=True)

        with out_path.open("w", encoding="utf-8") as out_f:
            for local_idx, (source_idx, rec) in enumerate(tqdm(rows, desc=dataset)):
                generation_started = time.perf_counter()
                if args.generator_family == "dream" and effective_decoding in {
                    "dream_origin",
                    "dream_dgmark",
                }:
                    assert build_model_input_from_row is not None
                    model_input, prompt_input, prompt_context = build_model_input_from_row(
                        rec, dataset, args.prompt_variant
                    )
                    assert build_dream_origin_prompt is not None
                    prompt_str, prompt_ids = build_dream_origin_prompt(
                        tokenizer=tokenizer,
                        model_input=model_input,
                        chat=args.chat,
                    )
                else:
                    prompt_input = rec.get("raw_prompt") or rec.get("input", "")
                    prompt_context = "" if rec.get("raw_prompt") else rec.get("context", "")
                    prompt_str, prompt_ids = build_chat_prompt(
                        prompt_input,
                        prompt_context,
                        tokenizer,
                    )
                prompt_tensor = torch.tensor(
                    prompt_ids, dtype=torch.long, device=args.device,
                ).unsqueeze(0)

                final_seed = int(args.seed)
                for attempt in range(args.max_retries + 1):
                    final_seed = int(args.seed) + int(source_idx) * 1000 + attempt
                    torch.manual_seed(final_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(final_seed)

                    if args.generator_family == "dream" and effective_decoding == "dream_origin":
                        assert dream_origin_diffusion_generate is not None
                        attention_mask = torch.ones_like(
                            prompt_tensor,
                            dtype=next(model.parameters()).dtype,
                            device=args.device,
                        )
                        hook = DGMarkLogitsBias(
                            mask_id=mask_id,
                            delta=args.dgmark_delta,
                            private_key=args.dgmark_private_key,
                        )
                        output = dream_origin_diffusion_generate(
                            model=model,
                            input_ids=prompt_tensor,
                            attention_mask=attention_mask,
                            gen_length=args.max_new_tokens,
                            steps=args.steps,
                            temperature=args.temperature,
                            alg=args.alg,
                            alg_temp=args.alg_temp,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            eps=args.eps,
                            mask_token_id=mask_id,
                            output_history=False,
                            return_dict_in_generate=True,
                            generation_logits_hook_func=hook,
                        )
                        seq = output.sequences if hasattr(output, "sequences") else output
                        tokens = seq[
                            0,
                            prompt_tensor.shape[1] : prompt_tensor.shape[1] + args.max_new_tokens,
                        ].detach().cpu().tolist()
                        assert dream_safe_decode is not None
                        text = dream_safe_decode(tokenizer, tokens, special_ids)
                        detection = _score_dgmark_tokens(
                            tokens,
                            prompt_tensor.shape[1],
                            args.dgmark_private_key,
                            eot_token_id=eot_token_id,
                        )
                        detection.update(
                            _dgmark_window_scores(
                                tokens,
                                prompt_tensor.shape[1],
                                args.dgmark_window_size,
                                args.dgmark_private_key,
                            )
                        )
                    elif effective_decoding in {"legacy", "dream_dgmark"}:
                        text, tokens, detection = llada_generate_dgmark(
                            prompt_tensor,
                            model,
                            tokenizer,
                            mask_id,
                            gen_length=args.max_new_tokens,
                            block_size=args.block_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            cfg_scale=0.0,
                            remasking=args.remasking,
                            sampling_strategy=args.dgmark_sampling_strategy,
                            top_k=args.dgmark_top_k,
                            beam_size=args.dgmark_beam_size,
                            private_key=args.dgmark_private_key,
                            window_size=args.dgmark_window_size,
                            eot_token_id=eot_token_id,
                            generator_family=args.generator_family,
                        )
                    else:
                        text, tokens, detection = llada_generate_dgmark_aligned(
                            prompt_tensor,
                            model,
                            tokenizer,
                            mask_id,
                            gen_length=args.max_new_tokens,
                            block_size=args.block_size,
                            steps=args.steps,
                            temperature=args.temperature,
                            cfg_scale=0.0,
                            remasking=args.remasking,
                            sampling_strategy=args.dgmark_sampling_strategy,
                            top_k=args.dgmark_top_k,
                            beam_size=args.dgmark_beam_size,
                            private_key=args.dgmark_private_key,
                            window_size=args.dgmark_window_size,
                            eot_token_id=eot_token_id,
                            generator_family=args.generator_family,
                            dgmark_position_bonus=args.dgmark_position_bonus,
                        )
                    text_clean = clean_text(text)
                    token_len = len(tokenizer(text_clean, add_special_tokens=False)["input_ids"])
                    rep4 = rep_ngram(text_clean, n=4)
                    passed_quality = token_len >= args.min_response_tokens and (
                        args.max_rep4 is None or rep4 <= args.max_rep4
                    )
                    if passed_quality:
                        break

                row = {
                    "dataset": dataset,
                    "prompt_idx": args.start_offset + local_idx,
                    "waterbench_idx": source_idx,
                    "shard": args.shard_idx,
                    "num_shards": args.num_shards,
                    "prompt_input": prompt_input,
                    "prompt_context": prompt_context,
                    "prompt_full": prompt_str,
                    "dgmark_text": text_clean,
                    "dgmark_token_ids": tokens,
                    "dgmark_token_len": token_len,
                    "generation_seconds": time.perf_counter() - generation_started,
                    "dgmark_word_len": len(text_clean.split()),
                    "dgmark_rep4": rep4,
                    "dgmark_too_short": token_len < args.min_response_tokens,
                    "dgmark_too_repetitive": (
                        args.max_rep4 is not None and rep4 > args.max_rep4
                    ),
                    "dgmark_passed_quality": passed_quality,
                    "dgmark_retry_attempts": attempt + 1,
                    "seed": int(args.seed),
                    "final_seed": final_seed,
                    "dgmark_detector": detection,
                    "dgmark_config": {
                        "watermark": "DGMark",
                        "decoding": effective_decoding,
                        "sampling_strategy": (
                            "multinomial"
                            if effective_decoding == "dream_origin" and args.temperature > 0
                            else (
                                "greedy"
                                if effective_decoding == "dream_origin"
                                else args.dgmark_sampling_strategy
                            )
                        ),
                        "top_k": (
                            args.top_k
                            if effective_decoding == "dream_origin"
                            else args.dgmark_top_k
                        ),
                        "beam_size": (
                            None
                            if effective_decoding == "dream_origin"
                            else args.dgmark_beam_size
                        ),
                        "private_key": "***provided***" if args.dgmark_private_key else None,
                        "window_size": args.dgmark_window_size,
                        "position_bonus": (
                            args.dgmark_position_bonus if effective_decoding == "aligned" else None
                        ),
                        "delta": args.dgmark_delta if effective_decoding == "dream_origin" else None,
                    },
                    "gen_config": {
                        "model": str(args.model_name_or_path),
                        "generator_family": args.generator_family,
                        "dgmark_decoding": effective_decoding,
                        "mask_id": mask_id,
                        "eot_token_id": eot_token_id,
                        "chat": bool(args.chat),
                        "prompt_variant": args.prompt_variant,
                        "temperature": args.temperature,
                        "diffusion_steps": args.steps,
                        "max_new_tokens": args.max_new_tokens,
                        "block_size": args.block_size,
                        "response_length_filter": [
                            args.min_response_tokens,
                            args.max_new_tokens,
                        ],
                        "max_rep4": args.max_rep4,
                        "remasking": (
                            args.alg
                            if args.generator_family == "dream" and effective_decoding == "dream_origin"
                            else args.remasking
                        ),
                        "alg": args.alg,
                        "alg_temp": args.alg_temp,
                        "top_p": args.top_p,
                        "top_k": (
                            args.dgmark_top_k
                            if effective_decoding == "dream_dgmark"
                            else args.top_k
                        ),
                        "eps": args.eps,
                        "attention_mask_right_pad": (
                            1.0
                            if effective_decoding == "dream_origin"
                            else None
                        ),
                        "logits_shift": (
                            "torch.cat([logits[:, :1], logits[:, :-1]], dim=1)"
                            if args.generator_family == "dream"
                            else None
                        ),
                        "dream_decode_backend": (
                            "dgmark_parity_decoder_with_eth_shifted_logits"
                            if effective_decoding == "dream_dgmark"
                            else None
                        ),
                    },
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
