#!/usr/bin/env python3
"""Evaluate clean or attacked DLM-KGW text against held-out clean negatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from denomark.baselines.dlm_kgw.model import HashDistribution
import math
import numpy as np
from denomark.evaluation.metrics import roc_interpolated_tpr


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("NEWLINE_CHAR", " ").split())


def threshold_at_fpr(neg: list[float], fpr: float = 0.01) -> float:
    arr = np.sort(np.asarray(neg, dtype=float))
    idx = int(math.ceil((1.0 - fpr) * len(arr))) - 1
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def auc_rank(pos: list[float], neg: list[float]) -> float:
    scores = np.r_[neg, pos]
    labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(pos: list[float], neg: list[float]) -> dict:
    pos_arr = np.asarray(pos, dtype=float)
    neg_arr = np.asarray(neg, dtype=float)
    threshold = threshold_at_fpr(neg, 0.01)
    threshold_5pct = threshold_at_fpr(neg, 0.05)
    return {
        "n_pos": int(len(pos_arr)),
        "n_neg": int(len(neg_arr)),
        "pos_mean": float(pos_arr.mean()) if len(pos_arr) else float("nan"),
        "neg_mean": float(neg_arr.mean()) if len(neg_arr) else float("nan"),
        "threshold_1pct": threshold,
        "threshold_5pct": threshold_5pct,
        "tpr_at_1pct": float((pos_arr > threshold).mean()) if len(pos_arr) else float("nan"),
        "tpr_at_5pct": float((pos_arr > threshold_5pct).mean()) if len(pos_arr) else float("nan"),
        "roc_tpr_at_0_5pct": roc_interpolated_tpr(pos, neg, 0.005),
        "roc_tpr_at_1pct": roc_interpolated_tpr(pos, neg, 0.01),
        "roc_tpr_at_5pct": roc_interpolated_tpr(pos, neg, 0.05),
        "auc": auc_rank(pos, neg) if len(pos_arr) and len(neg_arr) else float("nan"),
    }


POSITIVE_FIELDS = (
    "attacked_text",
    "attack_text",
    "hash_distribution_text",
    "text",
    "completion",
    "output",
)
NEGATIVE_FIELDS = (
    "text",
    "unwatermarked_text",
    "generation_text_skip_special",
    "completion",
    "output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-jsonl", type=Path, required=True)
    parser.add_argument("--negative-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--positive-field", default="auto")
    parser.add_argument("--negative-field", default="auto")
    parser.add_argument("--min-negative-tokens", type=int, default=150)
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--n-iter", type=int, default=1)
    parser.add_argument("--convolution-kernel", type=int, nargs="+", default=[-1])
    parser.add_argument("--seeding-scheme", choices=("sumhash", "minhash"), default="sumhash")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_text(row: dict, field: str, candidates: tuple[str, ...]) -> str:
    if field != "auto":
        return clean_text(str(row.get(field) or ""))
    for candidate in candidates:
        value = row.get(candidate)
        if isinstance(value, str) and value.strip():
            return clean_text(value)
    return ""


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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

    def score(text: str) -> tuple[float, int]:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError("cannot score empty token sequence")
        value = watermark.detect(
            torch.tensor(token_ids, dtype=torch.long, device=args.device)
        )["z_score"]
        return float(value), len(token_ids)

    positive_scores: list[float] = []
    positive_lengths: list[int] = []
    for index, row in enumerate(read_jsonl(args.positive_jsonl)):
        text = select_text(row, args.positive_field, POSITIVE_FIELDS)
        if not text:
            raise ValueError(f"{args.positive_jsonl}:{index + 1}: empty positive text")
        value, length = score(text)
        positive_scores.append(value)
        positive_lengths.append(length)

    negative_scores: list[float] = []
    negative_lengths: list[int] = []
    for index, row in enumerate(read_jsonl(args.negative_jsonl)):
        text = select_text(row, args.negative_field, NEGATIVE_FIELDS)
        if not text:
            raise ValueError(f"{args.negative_jsonl}:{index + 1}: empty negative text")
        value, length = score(text)
        if length >= args.min_negative_tokens:
            negative_scores.append(value)
            negative_lengths.append(length)

    if not positive_scores or not negative_scores:
        raise RuntimeError("positive and negative score sets must both be non-empty")
    metrics = summarize(positive_scores, negative_scores)
    payload = {
        "method": "DLM-KGW/HashDistribution",
        "score": "official green-token z-score",
        "positive_source": str(args.positive_jsonl),
        "negative_source": str(args.negative_jsonl),
        "positive_filter": None,
        "negative_filter": f"retokenized_token_len>={args.min_negative_tokens}",
        "retokenized_both_sides": True,
        "config": {
            "model": args.model,
            "delta": args.delta,
            "gamma": args.gamma,
            "topk": args.topk,
            "n_iter": args.n_iter,
            "convolution_kernel": args.convolution_kernel,
            "seeding_scheme": args.seeding_scheme,
        },
        "positive_token_lengths": positive_lengths,
        "negative_token_lengths": negative_lengths,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores,
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
