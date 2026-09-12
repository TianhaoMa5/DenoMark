#!/usr/bin/env python3
"""Build the paper mean-log-PPL comparison from evaluator JSON outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "base",
    "dataset",
    "method",
    "n_method",
    "n_clean",
    "method_mean_log_ppl",
    "clean_mean_log_ppl",
    "delta_mean_log_ppl",
    "method_result",
    "clean_result",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "JSONL rows with base, dataset, method, method_result, method_key, "
            "clean_result, and clean_key. Keys identify entries under results."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_aggregate(path: Path, key: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict) or key not in results:
        raise KeyError(f"{path}: results has no key {key!r}")
    aggregate = results[key]
    if not isinstance(aggregate, dict) or "mean_log_ppl" not in aggregate:
        raise ValueError(f"{path}: invalid PPL aggregate for {key!r}")
    return aggregate


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    with manifest.open(encoding="utf-8") as handle:
        specifications = [json.loads(line) for line in handle if line.strip()]
    if not specifications:
        raise ValueError("manifest is empty")
    rows = []
    for index, spec in enumerate(specifications, 1):
        required = (
            "base",
            "dataset",
            "method",
            "method_result",
            "method_key",
            "clean_result",
            "clean_key",
        )
        missing = [field for field in required if field not in spec]
        if missing:
            raise ValueError(f"manifest row {index} is missing {missing}")
        method_path = resolve(manifest.parent, str(spec["method_result"]))
        clean_path = resolve(manifest.parent, str(spec["clean_result"]))
        method = load_aggregate(method_path, str(spec["method_key"]))
        clean = load_aggregate(clean_path, str(spec["clean_key"]))
        method_mean = float(method["mean_log_ppl"])
        clean_mean = float(clean["mean_log_ppl"])
        rows.append(
            {
                "base": spec["base"],
                "dataset": spec["dataset"],
                "method": spec["method"],
                "n_method": int(method["n"]),
                "n_clean": int(clean["n"]),
                "method_mean_log_ppl": method_mean,
                "clean_mean_log_ppl": clean_mean,
                "delta_mean_log_ppl": method_mean - clean_mean,
                "method_result": str(method_path),
                "clean_result": str(clean_path),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
