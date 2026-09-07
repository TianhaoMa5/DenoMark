#!/usr/bin/env python3
"""Evaluate cached DenoMark raw scans with disjoint calibration and ROC pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from denomark.evaluation.metrics import (
    calibrated_scan_scores,
    score_matrix,
    summarize_roc,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_ids(rows: list[dict], field: str, label: str) -> set[str]:
    values = [str(row[field]) for row in rows if row.get(field) is not None]
    if len(values) != len(rows):
        raise ValueError(f"{label}: every row must contain {field!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label}: duplicate {field!r} values")
    return set(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--negative-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scan-min", type=int, default=12)
    parser.add_argument("--scan-max", type=int, default=37)
    parser.add_argument("--score-field", default="raw_scores_by_unit_size")
    parser.add_argument("--source-id-field", default="source_id")
    parser.add_argument(
        "--fpr",
        type=float,
        nargs="+",
        default=(0.005, 0.01, 0.05),
        help="exact FPR targets for linearly interpolated empirical ROC",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_min <= 0 or args.scan_max < args.scan_min:
        raise ValueError("invalid scan range")
    unit_sizes = list(range(args.scan_min, args.scan_max + 1))
    positives = read_jsonl(args.positive_jsonl)
    calibration = read_jsonl(args.calibration_jsonl)
    negatives = read_jsonl(args.negative_jsonl)
    if not positives or not calibration or not negatives:
        raise ValueError("positive, calibration, and negative files must be non-empty")

    calibration_ids = unique_ids(
        calibration, args.source_id_field, "calibration"
    )
    negative_ids = unique_ids(negatives, args.source_id_field, "held-out negatives")
    overlap = calibration_ids & negative_ids
    if overlap:
        raise ValueError(
            f"calibration and held-out negatives overlap on {len(overlap)} source IDs"
        )

    calibration_matrix = score_matrix(calibration, unit_sizes, args.score_field)
    positive_scores = calibrated_scan_scores(
        score_matrix(positives, unit_sizes, args.score_field),
        calibration_matrix,
    )
    negative_scores = calibrated_scan_scores(
        score_matrix(negatives, unit_sizes, args.score_field),
        calibration_matrix,
    )
    result = {
        "method": "DenoMark",
        "statistic": "per-size empirical p-value + Bonferroni; score=-log(p_scan)",
        "scan_unit_sizes": unit_sizes,
        "calibration_count": len(calibration),
        "heldout_negative_count": len(negatives),
        "calibration_heldout_id_overlap": 0,
        "metrics": summarize_roc(positive_scores, negative_scores, args.fpr),
        "inputs": {
            "positive": {"path": str(args.positive_jsonl), "sha256": sha256(args.positive_jsonl)},
            "calibration": {
                "path": str(args.calibration_jsonl),
                "sha256": sha256(args.calibration_jsonl),
            },
            "heldout_negative": {
                "path": str(args.negative_jsonl),
                "sha256": sha256(args.negative_jsonl),
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
