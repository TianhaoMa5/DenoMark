#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributions as dists
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))

from denomark.core.model import (build_model_input_from_row, clean_text, safe_decode_dream as safe_decode)
from denomark.core.model import resolve_mask_id


DEFAULT_MODEL = "Dream-org/Dream-v0-Instruct-7B"


@dataclass
class DreamOriginOutput:
    sequences: torch.Tensor
    history: list[torch.Tensor] | None = None


def rep_ngram(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeats = sum(v - 1 for v in counts.values() if v > 1)
    return repeats / max(1, len(grams))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(
    *,
    tokenizer,
    model_input: str,
    chat: bool,
) -> tuple[str, list[int]]:
    if chat:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": model_input.strip()}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return tokenizer.decode(ids, skip_special_tokens=False), list(ids)
    ids = tokenizer(model_input, add_special_tokens=True)["input_ids"]
    return model_input, list(ids)


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def top_k_top_p_filtering(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth = torch.topk(logits, k=k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits.float(), dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        sorted_remove = cumulative > float(top_p)
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(sorted_remove).scatter(-1, sorted_idx, sorted_remove)
        logits = logits.masked_fill(remove, -float("inf"))
    return logits


def sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    logits = top_k_top_p_filtering(logits.float(), top_k=top_k, top_p=top_p)
    if float(temperature) == 0.0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits / float(temperature), dim=-1)
    return dists.Categorical(probs=probs).sample()


def call_dream_model(model, x: torch.Tensor, attention_mask: torch.Tensor, tok_idx: torch.Tensor):
    try:
        return model(x, attention_mask, tok_idx)
    except TypeError:
        try:
            return model(x, attention_mask=attention_mask, tok_idx=tok_idx)
        except TypeError:
            return model(x, "full", None)


@torch.no_grad()
def dream_origin_diffusion_generate(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    gen_length: int = 300,
    steps: int = 300,
    temperature: float = 0.5,
    alg: str = "origin",
    alg_temp: float = 0.1,
    top_p: float | None = None,
    top_k: int | None = None,
    eps: float = 1e-3,
    mask_token_id: int | None = None,
    output_history: bool = False,
    return_dict_in_generate: bool = True,
    generation_logits_hook_func: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> DreamOriginOutput | torch.Tensor:
    if alg != "origin":
        raise ValueError(f"this wrapper implements alg='origin', got {alg!r}")
    if mask_token_id is None:
        raise ValueError("mask_token_id must be resolved from the active tokenizer")
    _ = alg_temp
    batch, input_len = input_ids.shape
    total_len = input_len + int(gen_length)
    device = input_ids.device
    attn_dtype = next(model.parameters()).dtype

    x = torch.full((batch, total_len), int(mask_token_id), dtype=input_ids.dtype, device=device)
    x[:, :input_len] = input_ids

    if attention_mask is None:
        attention_mask = torch.ones((batch, input_len), dtype=attn_dtype, device=device)
    else:
        attention_mask = attention_mask.to(device=device, dtype=attn_dtype)

    if torch.any(attention_mask == 0.0):
        pad_attn = torch.ones((batch, gen_length), dtype=attn_dtype, device=device)
        attention_mask = torch.cat([attention_mask, pad_attn], dim=1)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        model_attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1),
        )
    else:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch, gen_length), dtype=attn_dtype, device=device)],
            dim=1,
        )
        tok_idx = None
        model_attention_mask = "full"

    timesteps = torch.linspace(1.0, float(eps), int(steps) + 1, device=device)
    history: list[torch.Tensor] | None = [] if output_history else None

    hook = generation_logits_hook_func or (lambda step, cur_x, cur_logits: cur_logits)
    for step in range(int(steps)):
        t = timesteps[step]
        s = timesteps[step + 1]
        mask_index = x == int(mask_token_id)
        if not bool(mask_index.any().item()):
            break

        logits = call_dream_model(model, x, model_attention_mask, tok_idx).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        logits = hook(step, x, logits)

        sampled = x.clone()
        masked_logits = logits[mask_index]
        sampled[mask_index] = sample_tokens(
            masked_logits,
            temperature=float(temperature),
            top_k=top_k,
            top_p=top_p,
        ).to(sampled.dtype)

        if step == int(steps) - 1:
            p_transfer = torch.tensor(float("inf"), device=device)
        else:
            p_transfer = 1.0 - s / t

        confidence = 1.0 - torch.rand_like(sampled.float())
        transfer_index = mask_index & (confidence < p_transfer)
        x = torch.where(transfer_index, sampled, x)
        if history is not None:
            history.append(x.detach().clone())

    if return_dict_in_generate:
        return DreamOriginOutput(sequences=x, history=history)
    return x


def load_rows(path: Path, n: int, offset: int) -> list[dict]:
    rows = read_jsonl(path)
    return rows[int(offset) : int(offset) + int(n)]


def generate_dream() -> None:
    p = argparse.ArgumentParser(description="DREAM origin diffusion clean wrapper matching eth-sri style sampling.")
    p.add_argument("--generator_family", choices=["dream"], default="dream")
    p.add_argument("--prompts_jsonl", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--gen_length", "--max_new_tokens", dest="gen_length", type=int, default=300)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--alg", "--remasking", dest="alg", default="origin")
    p.add_argument("--alg_temp", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--mask_token_id", type=int, default=None)
    p.add_argument("--output_history", action="store_true")
    p.add_argument("--return_dict_in_generate", action="store_true", default=True)
    p.add_argument("--chat", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prompt_variant", default="default", choices=["default", "long_output"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_token_len", type=int, default=0)
    p.add_argument("--max_rep4", type=float, default=None)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device_map", default="cuda")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading DREAM model: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    args.mask_token_id = resolve_mask_id(tokenizer, "dream", args.mask_token_id)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
    ).eval()

    special_ids = {
        int(args.mask_token_id),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    }
    special_ids = {int(x) for x in special_ids if x is not None}
    rows = load_rows(args.prompts_jsonl, args.num_samples, args.offset)
    print(
        f"Generating DREAM eth-sri origin clean dataset={args.dataset} "
        f"offset={args.offset} n={len(rows)} steps={args.steps} gen_length={args.gen_length}",
        flush=True,
    )

    with args.output.open("w", encoding="utf-8") as out_f:
        for local_idx, row in enumerate(tqdm(rows, desc=f"dream-origin-{args.dataset}")):
            source_row_idx = int(args.offset) + local_idx
            prompt_idx = int(row.get("clean_n1000_idx", source_row_idx))
            model_input, prompt_input, prompt_context = build_model_input_from_row(
                row, args.dataset, args.prompt_variant
            )
            prompt_text, prompt_ids = build_prompt(tokenizer=tokenizer, model_input=model_input, chat=args.chat)
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
            attention_mask = torch.ones_like(
                input_ids,
                dtype=next(model.parameters()).dtype,
                device=args.device,
            )

            final_tokens: list[int] = []
            text = ""
            token_len = 0
            rep4 = 0.0
            attempts = 0
            final_seed = int(args.seed)
            for attempt in range(max(0, args.max_retries) + 1):
                attempts = attempt + 1
                final_seed = int(args.seed) + prompt_idx * 1000 + attempt
                set_seed(final_seed)
                output = dream_origin_diffusion_generate(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gen_length=args.gen_length,
                    steps=args.steps,
                    temperature=args.temperature,
                    alg=args.alg,
                    alg_temp=args.alg_temp,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    eps=args.eps,
                    mask_token_id=args.mask_token_id,
                    output_history=args.output_history,
                    return_dict_in_generate=args.return_dict_in_generate,
                    generation_logits_hook_func=None,
                )
                seq = output.sequences if hasattr(output, "sequences") else output
                final_tokens = seq[0, input_ids.shape[1] : input_ids.shape[1] + args.gen_length].detach().cpu().tolist()
                text = clean_text(safe_decode(tokenizer, final_tokens, special_ids))
                token_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                rep4 = rep_ngram(text)
                long_enough = token_len >= args.min_token_len
                not_repetitive = args.max_rep4 is None or rep4 <= args.max_rep4
                if long_enough and not_repetitive:
                    break

            rec = {
                "dataset": args.dataset,
                "prompt_idx": prompt_idx,
                "source_row_idx": source_row_idx,
                "prompt_full": prompt_text,
                "prompt_input": prompt_input,
                "prompt_context": prompt_context,
                "text": text,
                "generated_token_ids": final_tokens,
                "token_len": token_len,
                "word_len": len(text.split()),
                "rep4": rep4,
                "retry_attempts": attempts,
                "seed": int(args.seed),
                "final_seed": final_seed,
                "too_short": bool(args.min_token_len and token_len < args.min_token_len),
                "too_repetitive": bool(args.max_rep4 is not None and rep4 > args.max_rep4),
                "passed_quality": bool(
                    (not args.min_token_len or token_len >= args.min_token_len)
                    and (args.max_rep4 is None or rep4 <= args.max_rep4)
                ),
                "gen_config": {
                    "generator_family": "dream_ethsri_origin",
                    "model": args.model_name_or_path,
                    "chat": bool(args.chat),
                    "steps": int(args.steps),
                    "max_new_tokens": int(args.gen_length),
                    "gen_length": int(args.gen_length),
                    "temperature": float(args.temperature),
                    "alg": args.alg,
                    "remasking": args.alg,
                    "alg_temp": float(args.alg_temp),
                    "output_history": bool(args.output_history),
                    "return_dict_in_generate": bool(args.return_dict_in_generate),
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "eps": float(args.eps),
                    "mask_token_id": int(args.mask_token_id),
                    "attention_mask_right_pad": 1.0,
                    "attention_mask_forward_mode": "dream_official_full_when_unpadded",
                    "logits_shift": "torch.cat([logits[:, :1], logits[:, :-1]], dim=1)",
                    "origin_transfer": "p_transfer=1-s/t; final=inf; confidence=1-rand",
                    "hook_enabled": False,
                },
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()


