#!/usr/bin/env python3
"""Apply the paper's pre-attack generation-quality filter to JSONL rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoTokenizer

from denmark.data.protocol import normalize_whitespace, repetition_ngram


DEFAULT_TEXT_FIELDS = (
    "text",
    "completion",
    "watermarked_text",
    "hash_distribution_text",
    "dgmark_text",
    "unwatermarked_text",
    "output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text-field", action="append", dest="text_fields")
    parser.add_argument("--min-tokens", type=int, default=150)
    parser.add_argument("--max-rep4", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_text(row: dict, fields: tuple[str, ...]) -> tuple[str, str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value), field
    return "", ""


def main() -> None:
    args = parse_args()
    fields = tuple(args.text_fields or DEFAULT_TEXT_FIELDS)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{args.input}:{line_number}: invalid JSON") from exc

    kept = []
    reasons: Counter[str] = Counter()
    seen_text: set[str] = set()
    for index, row in enumerate(rows):
        text, field = choose_text(row, fields)
        if not text:
            reasons["empty"] += 1
            continue
        token_length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        rep4 = repetition_ngram(text, 4)
        if token_length < args.min_tokens:
            reasons["short"] += 1
            continue
        if rep4 > args.max_rep4:
            reasons["repetitive"] += 1
            continue
        if text in seen_text:
            reasons["duplicate_text"] += 1
            continue
        seen_text.add(text)
        output_row = dict(row)
        output_row["text"] = text
        output_row["paper_filter"] = {
            "source_row": index,
            "source_text_field": field,
            "token_length": token_length,
            "word_rep4": rep4,
            "minimum_tokens": args.min_tokens,
            "maximum_word_rep4": args.max_rep4,
        }
        kept.append(output_row)
        if args.max_rows is not None and len(kept) >= args.max_rows:
            break

    for path in (args.output, args.audit):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "input": str(args.input),
        "output": str(args.output),
        "tokenizer": args.tokenizer,
        "text_fields": fields,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed": dict(sorted(reasons.items())),
        "minimum_tokens": args.min_tokens,
        "maximum_word_rep4": args.max_rep4,
        "deduplicate_by": "normalized_text",
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
