#!/usr/bin/env python3
"""Deterministically select source segments for paper encoder training."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=25)
    args = parser.parse_args()

    rows: list[str] = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            text = str(json.loads(line).get("text") or "").strip()
            if text and len(text.split()) <= args.max_words:
                rows.append(text)
    if args.n > len(rows):
        raise ValueError(f"requested {args.n} rows, only {len(rows)} are available")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")

    selected = random.Random(args.seed).sample(rows, args.n)
    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for pair_index, text in enumerate(selected):
        shards[pair_index % args.num_shards].append(
            {"text": text}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard_index, shard in enumerate(shards):
        path = args.output_dir / "source_shards" / f"shard_{shard_index:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in shard:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "input_name": args.input.name,
        "available_rows": len(rows),
        "selected_rows": args.n,
        "num_shards": args.num_shards,
        "seed": args.seed,
        "maximum_source_words": args.max_words,
        "shard_sizes": [len(shard) for shard in shards],
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
