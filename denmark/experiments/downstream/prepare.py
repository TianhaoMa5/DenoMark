#!/usr/bin/env python3
"""Prepare the paper's full downstream evaluation manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from denmark.experiments.downstream.tasks import BENCHMARKS, prepare_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-per-benchmark",
        type=int,
        default=None,
        help="Deterministic smoke-test cap. Omit for complete official splits.",
    )
    args = parser.parse_args()
    manifest = prepare_manifest(args.output, args.limit_per_benchmark, args.seed)
    counts = {name: len(manifest["datasets"][name]) for name in BENCHMARKS}
    print(json.dumps({"output": str(args.output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
