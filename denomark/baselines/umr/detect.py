#!/usr/bin/env python3
"""Evaluate UMR with dataset/base-matched empirical clean negatives."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))
sys.path.insert(0, str(THIS.parent))

from denomark.baselines.umr.model import score_umr_tokens
from denomark.evaluation.metrics import summarize_roc
from denomark.evaluation.roc import roc_interpolated_tpr_at_fpr


POS_FIELDS = ("attacked_text", "watermarked_text", "text", "output", "completion")
NEG_FIELDS = ("unwatermarked_text", "text", "output", "completion", "generated_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umr_root", type=Path, required=True)
    parser.add_argument("--bitmap_path", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Model/logit vocabulary size used to create the UMR bitmap.")
    parser.add_argument("--positive_jsonl", type=Path, required=True)
    parser.add_argument("--negative_jsonl", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_txt", type=Path, default=None)
    parser.add_argument("--positive_field", default="auto")
    parser.add_argument("--negative_field", default="auto")
    parser.add_argument("--min_token_len", type=int, default=150)
    parser.add_argument("--max_token_len", type=int, default=300)
    parser.add_argument("--no_filter_positive", action="store_true")
    parser.add_argument("--no_filter_negative", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--watermark_str", default="10")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_text(row: dict, field: str, candidates: tuple[str, ...]) -> str:
    if field != "auto":
        return str(row.get(field) or "")
    for candidate in candidates:
        value = row.get(candidate)
        if isinstance(value, str) and value.strip():
            return value
    source = row.get("source_row")
    if isinstance(source, dict):
        return select_text(source, "auto", candidates)
    return ""


def score_rows(source: list[dict], tokenizer, bitmap, args, *, positive: bool) -> tuple[list[float], dict]:
    field = args.positive_field if positive else args.negative_field
    candidates = POS_FIELDS if positive else NEG_FIELDS
    no_filter = args.no_filter_positive if positive else args.no_filter_negative
    scores = []
    stats = {"rows": len(source), "kept": 0, "empty": 0, "short": 0}
    for row in source:
        text = " ".join(select_text(row, field, candidates).replace("NEWLINE_CHAR", " ").split())
        if not text:
            stats["empty"] += 1
            continue
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not no_filter and len(token_ids) < args.min_token_len:
            stats["short"] += 1
            continue
        token_ids = token_ids[: args.max_token_len]
        result = score_umr_tokens(
            token_ids,
            bitmap,
            watermark_str=args.watermark_str,
            key=args.key,
            ratio=args.ratio,
        )
        scores.append(float(result["z_score"]))
    stats["kept"] = len(scores)
    return scores, stats


def auc_rank(pos: list[float], neg: list[float]) -> float:
    scores = np.asarray(neg + pos, dtype=float)
    labels = np.asarray([0] * len(neg) + [1] * len(pos))
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def strict_point(pos: list[float], neg: list[float], fpr: float) -> dict:
    allowed = int(math.floor(fpr * len(neg)))
    ordered = np.sort(np.asarray(neg, dtype=float))
    index = max(0, min(len(ordered) - 1, len(ordered) - allowed - 1))
    threshold = float(ordered[index])
    actual_fp = int((ordered > threshold).sum())
    return {
        "threshold": threshold,
        "allowed_fp": allowed,
        "actual_fp": actual_fp,
        "empirical_fpr": actual_fp / len(neg),
        "strict_tpr": float((np.asarray(pos) > threshold).mean()),
        "roc_interpolated_tpr": roc_interpolated_tpr_at_fpr(pos, neg, fpr),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.umr_root.resolve()))
    PersistentBitmap = importlib.import_module("watermark.bitmap_persistent").PersistentBitmap
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    bitmap = PersistentBitmap(args.vocab_size, str(args.bitmap_path), device=args.device)
    pos, pos_stats = score_rows(rows(args.positive_jsonl), tokenizer, bitmap, args, positive=True)
    neg, neg_stats = score_rows(rows(args.negative_jsonl), tokenizer, bitmap, args, positive=False)
    if not pos or not neg:
        raise RuntimeError(f"Need non-empty score sets; positive={len(pos)} negative={len(neg)}")
    result = {
        "protocol": "attack-before length filtering; empirical clean-negative calibration",
        "positive": pos_stats,
        "negative": neg_stats,
        "pos_mean_z": float(np.mean(pos)),
        "neg_mean_z": float(np.mean(neg)),
        "positive_scores": pos,
        "negative_scores": neg,
        "auc": auc_rank(pos, neg),
        "paper_metrics": summarize_roc(pos, neg, fprs=(0.005, 0.01, 0.05)),
        "tpr_at_0_1pct": strict_point(pos, neg, 0.001),
        "tpr_at_1pct": strict_point(pos, neg, 0.01),
        "tpr_at_5pct": strict_point(pos, neg, 0.05),
        "config": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_txt = args.output_txt or args.output_json.with_suffix(".txt")
    output_txt.write_text(
        "\n".join([
            f"n_pos={len(pos)} n_neg={len(neg)}",
            f"mean_z pos={result['pos_mean_z']:.6f} neg={result['neg_mean_z']:.6f}",
            f"auc={result['auc']:.6f}",
            f"TPR@1%FPR strict={result['tpr_at_1pct']['strict_tpr']:.6f} roc={result['tpr_at_1pct']['roc_interpolated_tpr']:.6f}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
