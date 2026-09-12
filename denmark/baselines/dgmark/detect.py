#!/usr/bin/env python3
"""Evaluate an explicit attacked text field with the DGMark detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from transformers import AutoTokenizer

import math
import numpy as np
from denmark.baselines.dgmark.model import _dgmark_window_scores, _score_dgmark_tokens
from denmark.evaluation.metrics import roc_interpolated_tpr


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_of(row: dict, attacked: bool) -> str:
    if attacked:
        value = row.get("attacked_text") or row.get("attack_text") or row.get("text")
    else:
        value = (
            row.get("eval_text")
            or row.get("unwatermarked_text")
            or row.get("dgmark_text")
            or row.get("text")
            or row.get("completion")
            or row.get("output")
        )
    return " ".join(str(value or "").replace("NEWLINE_CHAR", " ").split())


def prompt_length(row: dict, tokenizer) -> int:
    prompt = row.get("prompt_full") or row.get("prompt") or row.get("prompt_input") or ""
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def token_ids_of(row: dict, tokenizer, attacked: bool, force_retokenize: bool = False) -> list[int]:
    text = text_of(row, attacked)
    if attacked or force_retokenize:
        return [int(value) for value in tokenizer(text, add_special_tokens=False)["input_ids"]]
    ids = (
        row.get("dgmark_token_ids")
        or row.get("generated_token_ids")
        or row.get("unwatermarked_token_ids")
        or row.get("eval_token_ids")
        or row.get("token_ids")
    )
    if not isinstance(ids, list):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [int(value) for value in ids]


def score_rows(
    rows: list[dict],
    tokenizer,
    attacked: bool,
    window_size: int,
    force_retokenize: bool = False,
) -> list[dict]:
    scored = []
    for index, row in enumerate(rows):
        text = text_of(row, attacked)
        ids = token_ids_of(row, tokenizer, attacked, force_retokenize=force_retokenize)
        if not text or not ids:
            raise ValueError(f"empty row at index={index}")
        prompt_len = prompt_length(row, tokenizer)
        score = _score_dgmark_tokens(ids, prompt_len)
        score.update(_dgmark_window_scores(ids, prompt_len, window_size))
        scored.append(
            {
                "index": index,
                "id": row.get("id", row.get("prompt_idx", row.get("source_idx"))),
                "token_len": len(ids),
                "prompt_len": prompt_len,
                "z_score": float(score["z_score"]),
                "agg_z": float(score["agg_z"]),
            }
        )
    return scored


def threshold_at_fpr(values: list[float], fpr: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = int(math.ceil((1.0 - fpr) * len(ordered))) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def auc_rank(positive: list[float], negative: list[float]) -> float:
    scores = np.r_[negative, positive]
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[labels == 1].sum() - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def summarize(positive: list[dict], negative: list[dict], score_field: str) -> dict:
    pos = [row[score_field] for row in positive]
    neg = [row[score_field] for row in negative]
    threshold01 = threshold_at_fpr(neg, 0.001)
    threshold1 = threshold_at_fpr(neg, 0.01)
    threshold5 = threshold_at_fpr(neg, 0.05)
    threshold10 = threshold_at_fpr(neg, 0.10)
    pos_array = np.asarray(pos, dtype=float)
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_mean_z": float(np.mean(pos)),
        "neg_mean_z": float(np.mean(neg)),
        "threshold_0_1pct": threshold01,
        "threshold_1pct": threshold1,
        "threshold_5pct": threshold5,
        "threshold_10pct": threshold10,
        "tpr_at_0_1pct": float((pos_array > threshold01).mean()),
        "tpr_at_1pct": float((pos_array > threshold1).mean()),
        "tpr_at_5pct": float((pos_array > threshold5).mean()),
        "tpr_at_10pct": float((pos_array > threshold10).mean()),
        "roc_tpr_at_0_5pct": roc_interpolated_tpr(pos, neg, 0.005),
        "roc_tpr_at_1pct": roc_interpolated_tpr(pos, neg, 0.01),
        "roc_tpr_at_5pct": roc_interpolated_tpr(pos, neg, 0.05),
        "auc": auc_rank(pos, neg),
    }


def materialize(rows: list[dict], field: str, source: Path) -> list[dict]:
    output = []
    for index, row in enumerate(rows):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source}:{index + 1}: missing non-empty field {field!r}")
        output.append({**row, "attacked_text": value})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--positive-text-field", default="text")
    parser.add_argument("--negative-text-field", default="text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--score-field", choices=["z_score", "agg_z"], default="agg_z")
    parser.add_argument("--min-negative-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    positive_rows = materialize(read_jsonl(args.positive), args.positive_text_field, args.positive)
    negative_rows = materialize(read_jsonl(args.negative), args.negative_text_field, args.negative)
    positive_scores = score_rows(
        positive_rows,
        tokenizer,
        attacked=True,
        window_size=args.window_size,
    )
    negative_scores = score_rows(
        negative_rows,
        tokenizer,
        attacked=True,
        window_size=args.window_size,
    )
    negative_scores = [
        row for row in negative_scores if row["token_len"] >= args.min_negative_tokens
    ]
    metrics = summarize(positive_scores, negative_scores, args.score_field)
    result = {
        "detector": "dgmark",
        "dataset": args.dataset,
        "positive_source": str(args.positive),
        "negative_source": str(args.negative),
        "positive_text_field": args.positive_text_field,
        "negative_text_field": args.negative_text_field,
        "positive_filter": None,
        "negative_filter": f"token_len>={args.min_negative_tokens}",
        "window_size": args.window_size,
        "score_field": args.score_field,
        "metrics": metrics,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{args.dataset} npos={metrics['n_pos']} nneg={metrics['n_neg']} "
        f"tpr1={metrics['tpr_at_1pct']:.6f} "
        f"tpr5={metrics['tpr_at_5pct']:.6f} auc={metrics['auc']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
