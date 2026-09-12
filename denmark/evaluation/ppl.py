"""Robust completion-only perplexity for causal language-model evaluators.

Unlike the legacy evaluator, this module locates the prompt/completion boundary
with tokenizer character offsets from the *jointly tokenized* prompt+completion
string. This avoids assuming that tokenization is prefix-stable at the text
boundary. It also reports both log(mean(PPL)) and mean(log(PPL)).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_literal_newlines(text: str) -> str:
    return str(text or "").replace("NEWLINE_CHAR", "\n")


def load_rows(
    path: Path,
    prompt_field: str,
    text_field: str,
    start_n: int,
    max_n: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid_index = 0
    with path.open(encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = normalize_literal_newlines(row.get(prompt_field, ""))
            completion = row.get(text_field, "") or row.get("completion", "")
            if isinstance(completion, list):
                completion = " ".join(map(str, completion))
            completion = normalize_literal_newlines(completion)
            if not completion.strip():
                continue
            if valid_index < start_n:
                valid_index += 1
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "source_line": source_line,
                    "prompt_idx": row.get("prompt_idx"),
                    "valid_index": valid_index,
                }
            )
            valid_index += 1
            if max_n and len(rows) >= max_n:
                break
    return rows


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("A fast tokenizer is required for offset-based boundaries")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    return tokenizer, model


@torch.no_grad()
def score_one(
    prompt: str,
    completion: str,
    tokenizer,
    model,
    max_length: int,
) -> dict[str, Any]:
    full_text = prompt + completion
    boundary = len(prompt)
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = list(map(int, encoded["input_ids"]))
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]

    completion_mask = [
        bool(end > boundary and end > start)
        for start, end in offsets
    ]
    original_tokens = len(input_ids)
    left_truncated_tokens = max(0, original_tokens - max_length)
    if left_truncated_tokens:
        input_ids = input_ids[left_truncated_tokens:]
        completion_mask = completion_mask[left_truncated_tokens:]

    ids = torch.tensor(input_ids, dtype=torch.long, device="cuda").unsqueeze(0)
    attention_mask = torch.ones_like(ids)
    logits = model(ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = ids[:, 1:]
    mask = torch.tensor(
        completion_mask[1:],
        dtype=torch.bool,
        device=ids.device,
    ).unsqueeze(0)
    n_scored = int(mask.sum().item())
    if n_scored == 0:
        raise ValueError("no completion tokens remained after tokenization/truncation")

    losses = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        labels,
        reduction="none",
    )
    mean_nll = float(losses.masked_select(mask).mean().float().item())
    return {
        "ppl": float(math.exp(mean_nll)),
        "log_ppl": mean_nll,
        "scored_completion_tokens": n_scored,
        "joint_tokens_before_truncation": original_tokens,
        "left_truncated_tokens": left_truncated_tokens,
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    ppls = np.asarray([sample["ppl"] for sample in samples], dtype=np.float64)
    log_ppls = np.asarray(
        [sample["log_ppl"] for sample in samples],
        dtype=np.float64,
    )
    return {
        "n": len(samples),
        "ppl_mean": float(np.mean(ppls)),
        "ppl_median": float(np.median(ppls)),
        "log_mean_ppl": float(np.log(np.mean(ppls))),
        "mean_log_ppl": float(np.mean(log_ppls)),
        "median_log_ppl": float(np.median(log_ppls)),
        "min_scored_completion_tokens": int(
            min(sample["scored_completion_tokens"] for sample in samples)
        ),
        "max_left_truncated_tokens": int(
            max(sample["left_truncated_tokens"] for sample in samples)
        ),
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--eval_model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--start_n", type=int, default=0)
    parser.add_argument("--max_n", type=int)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--text_field", default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model = load_model(args.eval_model)
    results: dict[str, Any] = {}
    for path in args.input_jsonl:
        rows = load_rows(
            path,
            args.prompt_field,
            args.text_field,
            args.start_n,
            args.max_n,
        )
        samples: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            scored = score_one(
                row["prompt"],
                row["completion"],
                tokenizer,
                model,
                args.max_length,
            )
            scored.update(
                {
                    "row_index": index,
                    "source_line": row["source_line"],
                    "prompt_idx": row["prompt_idx"],
                    "valid_index": row["valid_index"],
                }
            )
            samples.append(scored)
            print(
                f"{path.name} {index + 1}/{len(rows)} "
                f"ppl={scored['ppl']:.4f} logppl={scored['log_ppl']:.4f}",
                flush=True,
            )
        if samples:
            results[str(path)] = aggregate(samples)

    payload = {
        "evaluator": args.eval_model,
        "metric": "completion_only_joint_tokenization_offset_boundary_v2",
        "whitespace": "preserved_except_literal_NEWLINE_CHAR_to_newline",
        "max_length": args.max_length,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
