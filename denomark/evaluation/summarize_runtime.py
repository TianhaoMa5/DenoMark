#!/usr/bin/env python3
"""Summarize wall-clock generation cost recorded by paper experiment runners."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TOKEN_FIELDS = (
    "token_len",
    "generated_tokens",
    "dgmark_token_len",
    "hash_distribution_token_len",
    "patternmark_token_len",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def token_count(row: dict) -> int:
    for field in TOKEN_FIELDS:
        value = row.get(field)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    for field in ("token_ids", "generated_token_ids", "dgmark_token_ids"):
        value = row.get(field)
        if isinstance(value, list) and value:
            return len(value)
    raise ValueError("row has no positive generated-token count")


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                seconds = row.get("generation_seconds")
                if not isinstance(seconds, (int, float)) or seconds < 0:
                    raise ValueError(f"{path}:{line_number}: missing generation_seconds")
                rows.append(
                    {
                        "path": str(path),
                        "seconds": float(seconds),
                        "tokens": token_count(row),
                    }
                )
    if not rows:
        raise ValueError("no input rows")

    total_seconds = sum(row["seconds"] for row in rows)
    total_tokens = sum(row["tokens"] for row in rows)
    per_sample = [row["seconds"] for row in rows]
    result = {
        "samples": len(rows),
        "generated_tokens": total_tokens,
        "total_generation_seconds": total_seconds,
        "mean_seconds_per_sample": statistics.fmean(per_sample),
        "median_seconds_per_sample": statistics.median(per_sample),
        "seconds_per_generated_token": total_seconds / total_tokens,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else None,
        "inputs": [str(path) for path in args.inputs],
    }
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
