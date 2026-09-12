#!/usr/bin/env python3
"""Collect heterogeneous detector JSON files into one paper-metric CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = (
    "base",
    "dataset",
    "method",
    "condition",
    "setting",
    "n_pos",
    "n_neg",
    "tpr_at_0_5pct",
    "tpr_at_1pct",
    "tpr_at_5pct",
    "auc",
    "result_path",
    "metric_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "JSONL rows with base, dataset, method, condition, result_path, and "
            "optional metric_path/setting. metric_path is dot-separated."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(payload: Any, path: str) -> Any:
    value = payload
    for component in path.split(".") if path else ():
        if not isinstance(value, dict) or component not in value:
            raise KeyError(f"metric path {path!r} has no component {component!r}")
        value = value[component]
    return value


def default_metric_path(method: str, payload: dict[str, Any]) -> str:
    if method == "denmark":
        return "subsets.pos_all_neg>=150.calibrated.calibrated_scan"
    if "metrics" in payload:
        return "metrics"
    if "paper_metrics" in payload:
        return "paper_metrics"
    raise ValueError(
        f"cannot infer metric path for method={method!r}; set metric_path in the manifest"
    )


def first_number(metrics: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    raise KeyError(f"none of the metric keys are present: {keys}")


def extract_tpr(metrics: dict[str, Any], fpr: float, flat_keys: tuple[str, ...]) -> float:
    tpr = metrics.get("tpr")
    if isinstance(tpr, dict):
        for key in (str(fpr), f"{fpr:g}"):
            if isinstance(tpr.get(key), (int, float)):
                return float(tpr[key])
    return first_number(metrics, *flat_keys)


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    if not isinstance(metrics, dict):
        raise TypeError("resolved metric payload must be an object")
    return {
        "n_pos": int(first_number(metrics, "n_pos", "n_positive")),
        "n_neg": int(first_number(metrics, "n_neg", "n_negative")),
        "tpr_at_0_5pct": extract_tpr(
            metrics,
            0.005,
            ("roc_tpr_at_0_5pct", "roc_tpr0_5", "tpr_at_0_5pct"),
        ),
        "tpr_at_1pct": extract_tpr(
            metrics,
            0.01,
            ("roc_tpr_at_1pct", "roc_tpr1", "tpr_at_1pct"),
        ),
        "tpr_at_5pct": extract_tpr(
            metrics,
            0.05,
            ("roc_tpr_at_5pct", "roc_tpr5", "tpr_at_5pct"),
        ),
        "auc": first_number(metrics, "auc"),
    }


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("manifest is empty")
    output_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows, 1):
        for field in ("base", "dataset", "method", "condition", "result_path"):
            if not str(row.get(field, "")).strip():
                raise ValueError(f"manifest row {index} is missing {field}")
        result_path = Path(str(row["result_path"]))
        if not result_path.is_absolute():
            result_path = manifest.parent / result_path
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        metric_path = str(
            row.get("metric_path")
            or default_metric_path(str(row["method"]), payload)
        )
        normalized = normalize_metrics(resolve_path(payload, metric_path))
        identity = tuple(
            str(row.get(field, ""))
            for field in ("base", "dataset", "method", "condition", "setting")
        )
        if identity in seen:
            raise ValueError(f"duplicate paper result identity at manifest row {index}: {identity}")
        seen.add(identity)
        output_rows.append(
            {
                "base": row["base"],
                "dataset": row["dataset"],
                "method": row["method"],
                "condition": row["condition"],
                "setting": row.get("setting", ""),
                **normalized,
                "result_path": str(result_path),
                "metric_path": metric_path,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({"rows": len(output_rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
