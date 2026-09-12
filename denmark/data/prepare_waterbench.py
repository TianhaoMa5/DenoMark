#!/usr/bin/env python3
"""Build prompt-only DenMark inputs from user-supplied WaterBench JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DATASET_SPECS = {
    "finance_qa": {
        "filename": "finance_qa.jsonl",
        "template": (
            "You are a helpful assistant, please answer the following question "
            "with financial knowledge within 300 words:\n\n{content}"
        ),
        "fields": ("input", "question", "context", "instruction", "prompt"),
    },
    "alpacafarm": {
        "filename": "alpacafarm.jsonl",
        "template": (
            "You are a helpful assistant, please answer the following instruction: "
            "\n{content}\n"
        ),
        "fields": ("context", "instruction", "input", "question", "prompt"),
    },
    "longform_qa": {
        "filename": "longform_qa.jsonl",
        "template": (
            "You are a helpful assistant, please answer the following question "
            "within 300 words:\n\n{content}"
        ),
        "fields": ("input", "question", "context", "instruction", "prompt"),
    },
}


def _first_text(row: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    raise ValueError(f"row has none of the supported text fields: {', '.join(fields)}")


def normalize_prompt(row: dict[str, Any], dataset: str) -> dict[str, str]:
    """Remove references and metadata while preserving the paper prompt template."""
    if dataset not in DATASET_SPECS:
        raise ValueError(f"unsupported dataset: {dataset}")
    specification = DATASET_SPECS[dataset]
    content = _first_text(row, specification["fields"])
    raw_prompt = str(row.get("raw_prompt") or "").strip()
    if not raw_prompt:
        raw_prompt = specification["template"].format(content=content)
    return {
        "prompt": raw_prompt,
        "raw_prompt": raw_prompt,
        "input": content if dataset != "alpacafarm" else "",
        "context": content if dataset == "alpacafarm" else "",
    }


def prepare_file(
    source: Path,
    destination: Path,
    dataset: str,
    limit: int,
    selection: str,
    seed: int,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    if len(rows) < limit:
        raise ValueError(f"{source} contains {len(rows)} rows, fewer than --limit={limit}")
    if selection == "random":
        indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    else:
        indices = list(range(limit))
    normalized = [normalize_prompt(rows[index], dataset) for index in indices]
    if len({row["prompt"] for row in normalized}) != len(normalized):
        raise ValueError(f"{dataset}: selected prompts are not unique")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in normalized:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "dataset": dataset,
        "source": source.name,
        "available_rows": len(rows),
        "written_rows": len(normalized),
        "selection": selection,
        "seed": seed if selection == "random" else None,
        "output": destination.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finance", type=Path)
    parser.add_argument("--alpaca", type=Path)
    parser.add_argument("--longform", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--selection", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    sources = {
        "finance_qa": args.finance,
        "alpacafarm": args.alpaca,
        "longform_qa": args.longform,
    }
    if not any(sources.values()):
        parser.error("provide at least one of --finance, --alpaca, or --longform")

    manifest = []
    for dataset, source in sources.items():
        if source is None:
            continue
        destination = args.output_dir / DATASET_SPECS[dataset]["filename"]
        manifest.append(
            prepare_file(
                source,
                destination,
                dataset,
                args.limit,
                args.selection,
                args.seed,
            )
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "construction_manifest.json").write_text(
        json.dumps({"datasets": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"datasets": manifest}, indent=2))


if __name__ == "__main__":
    main()
