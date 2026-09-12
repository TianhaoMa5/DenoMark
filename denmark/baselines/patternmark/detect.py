#!/usr/bin/env python3
"""Evaluate official PatternMark p-values against held-out clean negatives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from transformers import AutoTokenizer

from denmark.evaluation.metrics import summarize_roc
from denmark.baselines.patternmark.generate import (
    INITIAL_STATE,
    PATTERNS,
    TRANSITION_MATRIX,
    color_lookup,
    detect_patternmark,
    load_patternmark_class,
)


POSITIVE_FIELDS = ("attacked_text", "attack_text", "text", "completion")
NEGATIVE_FIELDS = ("text", "unwatermarked_text", "completion", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patternmark-repo", type=Path, required=True)
    parser.add_argument("--positive-jsonl", type=Path, required=True)
    parser.add_argument("--negative-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--positive-field", default="auto")
    parser.add_argument("--negative-field", default="auto")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_text(row: dict, field: str, candidates: tuple[str, ...]) -> str:
    if field != "auto":
        value = row.get(field)
        return str(value or "").strip()
    for candidate in candidates:
        value = row.get(candidate)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> None:
    args = parse_args()
    patternmark_class = load_patternmark_class(args.patternmark_repo)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    watermark = patternmark_class(
        l=2,
        transition_matrix=[list(row) for row in TRANSITION_MATRIX],
        initial_state=list(INITIAL_STATE),
        delta=args.delta,
        tokenizer=tokenizer,
        patterns=[list(pattern) for pattern in PATTERNS],
        pattern_length=4,
        device="cpu",
    )
    lookup = color_lookup(watermark)

    def score(rows: list[dict], field: str, candidates: tuple[str, ...]) -> tuple[list[float], list[dict]]:
        values: list[float] = []
        details: list[dict] = []
        for index, row in enumerate(rows):
            text = " ".join(select_text(row, field, candidates).replace("NEWLINE_CHAR", " ").split())
            if not text:
                raise ValueError(f"empty text at row {index}")
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:300]
            result = detect_patternmark([int(value) for value in ids], lookup)
            p_value = float(result["p_value"])
            values.append(-p_value)
            details.append(
                {
                    "row": index,
                    "token_length": len(ids),
                    "pattern_count": int(result["z_score"]),
                    "p_value": p_value,
                }
            )
        return values, details

    positive, positive_details = score(
        read_jsonl(args.positive_jsonl), args.positive_field, POSITIVE_FIELDS
    )
    negative, negative_details = score(
        read_jsonl(args.negative_jsonl), args.negative_field, NEGATIVE_FIELDS
    )
    metrics = summarize_roc(positive, negative, fprs=(0.005, 0.01, 0.05))
    metrics["mean_positive_p"] = float(np.mean([-value for value in positive]))
    metrics["mean_negative_p"] = float(np.mean([-value for value in negative]))
    payload = {
        "method": "PatternMark/OrderAgnostic",
        "score": "negative official p-value; larger is more watermarked",
        "retokenized_both_sides": True,
        "completion_only": True,
        "positive_source": str(args.positive_jsonl),
        "negative_source": str(args.negative_jsonl),
        "model": args.model,
        "config": {
            "delta": args.delta,
            "l": 2,
            "pattern_length": 4,
            "patterns": [list(pattern) for pattern in PATTERNS],
        },
        "metrics": metrics,
        "positive_scores": positive_details,
        "negative_scores": negative_details,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
