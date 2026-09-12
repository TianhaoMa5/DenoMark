#!/usr/bin/env python3
"""Merge Pegasus shards into a deduplicated DenMark encoder-training JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kept: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    counts = {"read": 0, "empty": 0, "unchanged": 0, "duplicate": 0}
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                counts["read"] += 1
                row = json.loads(line)
                text = str(row.get("text") or "").strip()
                positive = str(row.get("positive") or "").strip()
                if not text or not positive:
                    counts["empty"] += 1
                    continue
                key = (normalize(text), normalize(positive))
                if key[0] == key[1]:
                    counts["unchanged"] += 1
                    continue
                if key in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(key)
                kept.append({"text": text, "positive": positive})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts["kept"] = len(kept)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
