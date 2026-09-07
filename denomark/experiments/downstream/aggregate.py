#!/usr/bin/env python3
"""Aggregate paper downstream JSONL files into CSVs and a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from denomark.experiments.downstream.tasks import BENCHMARKS

METHODS = ("unwatermarked", "hash", "patternmark", "ours", "dgmark", "umr")
MODELS = ("llada8b", "llada15")


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/downstream"))
    args = parser.parse_args()
    per_benchmark = []
    runtime = []
    errors = []
    by_key = {}
    for model in MODELS:
        for method in METHODS:
            all_rows = []
            for benchmark in BENCHMARKS:
                rows = read_rows(args.root / "raw" / model / method / f"{benchmark}.jsonl")
                all_rows.extend(rows)
                correct = sum(bool(row.get("correct")) for row in rows)
                total = len(rows)
                score = correct / total if total else None
                key = (model, method, benchmark)
                by_key[key] = score
                per_benchmark.append({
                    "model": model, "method": method, "benchmark": benchmark,
                    "correct": correct, "total": total,
                    "score": "" if score is None else f"{score:.6f}",
                })
                for row in rows:
                    if row.get("error"):
                        errors.append({
                            "model": model, "method": method, "benchmark": benchmark,
                            "sample_id": row["sample_id"], "error": row["error"],
                        })
            seconds = sum(float(row.get("generation_seconds", 0.0)) for row in all_rows)
            forward = sum(int(row.get("forward_passes", 0)) for row in all_rows)
            runtime.append({
                "model": model, "method": method, "examples": len(all_rows),
                "successful": sum(not row.get("error") for row in all_rows),
                "failed": sum(bool(row.get("error")) for row in all_rows),
                "total_seconds": f"{seconds:.3f}",
                "mean_seconds": f"{seconds / len(all_rows):.3f}" if all_rows else "",
                "mean_generated_tokens": (
                    f"{sum(int(row.get('generated_tokens', 0)) for row in all_rows) / len(all_rows):.3f}"
                    if all_rows else ""
                ),
                "forward_passes": forward,
                "peak_gpu_memory_bytes": max((int(row.get("peak_gpu_memory_bytes", 0)) for row in all_rows), default=0),
            })

    macro = []
    for model in MODELS:
        baseline = [by_key.get((model, "unwatermarked", benchmark)) for benchmark in BENCHMARKS]
        baseline_avg = sum(x for x in baseline if x is not None) / len([x for x in baseline if x is not None]) if any(x is not None for x in baseline) else None
        for method in METHODS:
            vals = [by_key.get((model, method, benchmark)) for benchmark in BENCHMARKS]
            valid = [x for x in vals if x is not None]
            avg = sum(valid) / len(valid) if valid else None
            macro.append({
                "model": model, "method": method,
                "macro_average": "" if avg is None else f"{avg:.6f}",
                "delta_vs_unwatermarked": "" if avg is None or baseline_avg is None else f"{avg - baseline_avg:.6f}",
                "benchmarks_complete": len(valid),
            })

    summary_dir = args.root / "summaries"
    write_csv(summary_dir / "per_benchmark.csv", per_benchmark, ["model", "method", "benchmark", "correct", "total", "score"])
    write_csv(summary_dir / "macro_average.csv", macro, ["model", "method", "macro_average", "delta_vs_unwatermarked", "benchmarks_complete"])
    write_csv(summary_dir / "errors.csv", errors, ["model", "method", "benchmark", "sample_id", "error"])
    write_csv(summary_dir / "runtime.csv", runtime, list(runtime[0]) if runtime else [])

    lines = [
        "# Downstream benchmark evaluation",
        "",
        "Scores are percentages with raw counts. Missing or partial shards remain visible in the denominators.",
        "",
    ]
    for model in MODELS:
        lines.extend([
            f"## {model}",
            "",
            "| Method | MMLU | HellaSwag | ARC-C | GSM8K | Average |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            cells = []
            for benchmark in BENCHMARKS:
                row = next(item for item in per_benchmark if item["model"] == model and item["method"] == method and item["benchmark"] == benchmark)
                cells.append("-" if not row["total"] else f"{100 * float(row['score']):.1f}% ({row['correct']}/{row['total']})")
            avg_row = next(item for item in macro if item["model"] == model and item["method"] == method)
            avg = "-" if not avg_row["macro_average"] else f"{100 * float(avg_row['macro_average']):.1f}%"
            lines.append(f"| {method} | " + " | ".join(cells) + f" | {avg} |")
        lines.append("")
    error_counts = Counter(item["error"].split(":", 1)[0] for item in errors)
    lines.extend([
        "## Errors",
        "",
        json.dumps(error_counts, indent=2),
        "",
        "## Declared deviations",
        "",
        "- Dream multiple-choice tasks use generation scoring because likelihood-only scoring would bypass generation-time watermarks.",
    ])
    (args.root / "benchmark_report.md").write_text("\n".join(lines) + "\n")
    print(args.root / "benchmark_report.md")


if __name__ == "__main__":
    main()
