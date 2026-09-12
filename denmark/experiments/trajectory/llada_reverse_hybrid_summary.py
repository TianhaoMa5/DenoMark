#!/usr/bin/env python3
"""Validate and summarize completed LLaDA reverse-hybrid raw shards."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denmark.experiments.trajectory.llada_reverse_hybrid_metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_mean_ci,
    summarize_block_values,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_prompts", type=int, default=30)
    parser.add_argument("--expected_collapse_blocks", type=int, default=10)
    parser.add_argument("--bootstrap_replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def input_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        elif path.exists():
            files.append(path)
    return files


def load_rows(paths: list[Path]) -> list[dict]:
    by_prompt: dict[int, dict] = {}
    source_by_prompt: dict[int, Path] = {}
    duplicates = []
    for path in input_files(paths):
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_idx = int(row["prompt_idx"])
                if prompt_idx in by_prompt:
                    duplicates.append((prompt_idx, source_by_prompt[prompt_idx], path))
                    if row != by_prompt[prompt_idx]:
                        raise RuntimeError(
                            f"conflicting duplicate prompt {prompt_idx}: "
                            f"{source_by_prompt[prompt_idx]} vs {path}"
                        )
                    continue
                by_prompt[prompt_idx] = row
                source_by_prompt[prompt_idx] = path
    if duplicates:
        raise RuntimeError(f"duplicate prompt rows are not allowed: {duplicates}")
    return [by_prompt[index] for index in sorted(by_prompt)]


def validate_condition(condition: dict, *, collapse: bool) -> None:
    steps = condition["steps"]
    if len(steps) != 25 or [step["decision_idx"] for step in steps] != list(range(25)):
        raise RuntimeError("missing or reordered reverse-hybrid decision")
    recomputed = []
    for step in steps:
        selected = np.asarray(step["Q_pi_selected_raw_scores"], dtype=float)
        reference = np.asarray(step["Q_pi_reference_raw_scores"], dtype=float)
        differences = np.asarray(step["paired_future_differences"], dtype=float)
        if not np.allclose(selected - reference, differences, atol=1e-10, rtol=0):
            raise RuntimeError("paired future raw scores do not reproduce differences")
        if not np.isclose(differences.mean(), step["delta_hat_pi"], atol=1e-10):
            raise RuntimeError("raw paired differences do not reproduce delta")
        if len(step["candidate_positions"]) != 16 or len(step["candidate_token_ids"]) != 16:
            raise RuntimeError("candidate metadata does not have K=16 entries")
        if len(step["reverse_hybrid_prefix_audit"]) != step["decision_idx"]:
            raise RuntimeError("reverse-hybrid prefix audit length mismatch")
        if any(item["policy"] != "pi0_matched_reference" for item in step["reverse_hybrid_prefix_audit"]):
            raise RuntimeError("reverse-hybrid prefix contains a non-pi0 decision")
        if not all(step["sanity"].values()):
            raise RuntimeError(f"failed step sanity: {step['sanity']}")
        if collapse and (
            step["unique_candidate_count"] != 1
            or not np.allclose(differences, 0, atol=1e-12, rtol=0)
        ):
            raise RuntimeError("exact collapse is not exact")
        recomputed.append(float(differences.mean()))
    if not np.isclose(sum(recomputed), condition["block_summary"]["sum_delta"], atol=1e-10):
        raise RuntimeError("block cumulative delta does not equal all 25 steps")
    endpoint = condition["endpoint"]
    endpoint_diff = np.asarray(endpoint["paired_endpoint_differences"], dtype=float)
    pi = np.asarray(endpoint["J_pi_raw_scores"], dtype=float)
    pi0 = np.asarray(endpoint["J_pi0_raw_scores"], dtype=float)
    if not np.allclose(pi - pi0, endpoint_diff, atol=1e-10, rtol=0):
        raise RuntimeError("endpoint paired scores do not reproduce endpoint differences")
    if not np.isclose(endpoint_diff.mean(), endpoint["endpoint_uplift_hat"], atol=1e-10):
        raise RuntimeError("endpoint raw differences do not reproduce uplift")
    if collapse and not np.allclose(endpoint_diff, 0, atol=1e-12, rtol=0):
        raise RuntimeError("exact collapse endpoint is nonzero")
    if not all(condition["sanity"].values()):
        raise RuntimeError(f"failed condition sanity: {condition['sanity']}")


def safe_correlations(x: list[float], y: list[float]) -> dict:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def step_sign_summary(rows: list[dict], condition_name: str) -> dict:
    values = np.asarray([
        step["delta_hat_pi"]
        for row in rows
        for step in row["conditions"][condition_name]["steps"]
    ], dtype=float)
    positive = values[values > 1e-12]
    negative = values[values < -1e-12]
    zero = values[np.abs(values) <= 1e-12]
    return {
        "condition": condition_name,
        "total_steps": int(len(values)),
        "positive_steps": int(len(positive)),
        "negative_steps": int(len(negative)),
        "zero_steps": int(len(zero)),
        "positive_step_fraction": float(len(positive) / len(values)),
        "negative_step_fraction": float(len(negative) / len(values)),
        "zero_step_fraction": float(len(zero) / len(values)),
        "mean_positive_delta": float(positive.mean()) if len(positive) else 0.0,
        "mean_negative_delta": float(negative.mean()) if len(negative) else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    global pearsonr, plt, spearmanr
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    rows = load_rows(args.inputs)
    ids = [int(row["prompt_idx"]) for row in rows]
    if ids != list(range(args.expected_prompts)):
        raise RuntimeError(
            f"expected prompt ids 0..{args.expected_prompts - 1}, got {ids}"
        )
    collapse_rows = [row for row in rows if "exact_collapse" in row["conditions"]]
    if len(collapse_rows) != args.expected_collapse_blocks:
        raise RuntimeError(
            f"expected {args.expected_collapse_blocks} collapse blocks, "
            f"got {len(collapse_rows)}"
        )
    block_keys = [
        (row["prompt_idx"], row["selected_block_id"]) for row in rows
    ]
    if len(set(block_keys)) != len(block_keys):
        raise RuntimeError("duplicate prompt/block keys")
    for row in rows:
        if not all(row["sanity"].values()):
            raise RuntimeError(f"failed prompt sanity for {row['prompt_idx']}")
        validate_condition(row["conditions"]["default"], collapse=False)
        if "exact_collapse" in row["conditions"]:
            validate_condition(row["conditions"]["exact_collapse"], collapse=True)

    default_cumulative = [
        row["conditions"]["default"]["block_summary"]["sum_delta"] for row in rows
    ]
    default_endpoint = [
        row["conditions"]["default"]["endpoint"]["endpoint_uplift_hat"] for row in rows
    ]
    collapse_cumulative = [
        row["conditions"]["exact_collapse"]["block_summary"]["sum_delta"]
        for row in collapse_rows
    ]
    collapse_endpoint = [
        row["conditions"]["exact_collapse"]["endpoint"]["endpoint_uplift_hat"]
        for row in collapse_rows
    ]
    default_summary = summarize_block_values(
        default_cumulative,
        default_endpoint,
        n_bootstrap=args.bootstrap_replicates,
        seed=args.seed,
    )
    collapse_summary = (
        summarize_block_values(
            collapse_cumulative,
            collapse_endpoint,
            n_bootstrap=args.bootstrap_replicates,
            seed=args.seed + 100,
        )
        if collapse_rows
        else None
    )
    default_diversity = [
        row["conditions"]["default"]["block_summary"]["mean_candidate_diversity"]
        for row in rows
    ]
    diversity_correlation = safe_correlations(default_diversity, default_cumulative)
    endpoint_default_paired = [
        row["conditions"]["default"]["endpoint"]["endpoint_uplift_hat"]
        for row in collapse_rows
    ]
    endpoint_reduction = np.asarray(endpoint_default_paired) - np.asarray(collapse_endpoint)
    reduction_ci = (
        paired_bootstrap_mean_ci(
            endpoint_default_paired,
            collapse_endpoint,
            n_bootstrap=args.bootstrap_replicates,
            seed=args.seed + 200,
        )
        if collapse_rows
        else (None, None)
    )
    default_sign = step_sign_summary(rows, "default")
    collapse_sign = (
        step_sign_summary(collapse_rows, "exact_collapse")
        if collapse_rows
        else {
            "condition": "exact_collapse",
            "total_steps": 0,
            "positive_steps": 0,
            "negative_steps": 0,
            "zero_steps": 0,
            "positive_step_fraction": None,
            "negative_step_fraction": None,
            "zero_step_fraction": None,
            "mean_positive_delta": 0.0,
            "mean_negative_delta": 0.0,
        }
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_reverse_hybrid.jsonl"
    with raw_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    main_table = []
    condition_summaries = [("Default", default_summary)]
    if collapse_summary is not None:
        condition_summaries.append(("Exact collapse", collapse_summary))
    for label, summary in condition_summaries:
        main_table.append({
            "Condition": label,
            "Blocks": summary["blocks"],
            "Mean sum_delta_pi": summary["cumulative_delta_mean"],
            "Cumulative 95% CI lower": summary["cumulative_delta_ci95"][0],
            "Cumulative 95% CI upper": summary["cumulative_delta_ci95"][1],
            "Endpoint uplift": summary["endpoint_uplift_mean"],
            "Endpoint 95% CI lower": summary["endpoint_uplift_ci95"][0],
            "Endpoint 95% CI upper": summary["endpoint_uplift_ci95"][1],
            "Closure gap": summary["closure_gap_mean"],
            "Closure 95% CI lower": summary["closure_gap_ci95"][0],
            "Closure 95% CI upper": summary["closure_gap_ci95"][1],
            "Positive blocks": summary["cumulative_delta_positive_block_ratio"],
        })
    write_csv(output / "summary_reverse_hybrid.csv", main_table)

    closure_rows = []
    for row in rows:
        for condition_name, condition in row["conditions"].items():
            summary = condition["block_summary"]
            closure_rows.append({
                "prompt_idx": row["prompt_idx"],
                "block_id": row["selected_block_id"],
                "condition": condition_name,
                "sum_delta": summary["sum_delta"],
                "endpoint_uplift": summary["endpoint_uplift"],
                "closure_gap_endpoint_minus_sum_delta": summary[
                    "closure_gap_endpoint_minus_sum_delta"
                ],
                "positive_steps": summary["positive_step_count"],
                "negative_steps": summary["negative_step_count"],
                "zero_steps": summary["zero_step_count"],
                "mean_candidate_diversity": summary["mean_candidate_diversity"],
                "first_eos_generation_index": row["first_eos_generation_index"],
            })
    write_csv(output / "closure_by_block.csv", closure_rows)

    collapse_table = []
    for row in collapse_rows:
        default = row["conditions"]["default"]["block_summary"]
        collapse = row["conditions"]["exact_collapse"]["block_summary"]
        collapse_table.append({
            "prompt_idx": row["prompt_idx"],
            "block_id": row["selected_block_id"],
            "default_sum_delta": default["sum_delta"],
            "collapse_sum_delta": collapse["sum_delta"],
            "default_endpoint_uplift": default["endpoint_uplift"],
            "collapse_endpoint_uplift": collapse["endpoint_uplift"],
            "endpoint_uplift_reduction": default["endpoint_uplift"]
            - collapse["endpoint_uplift"],
            "collapse_mean_unique_candidates": collapse["mean_candidate_diversity"],
        })
    write_csv(output / "collapse_summary.csv", collapse_table)

    curves = np.asarray([
        row["conditions"]["default"]["block_summary"]["cumulative_delta"]
        for row in rows
    ], dtype=float)
    rng = np.random.default_rng(args.seed + 300)
    boot_curves = np.empty((args.bootstrap_replicates, 25), dtype=float)
    for offset in range(0, args.bootstrap_replicates, 500):
        width = min(500, args.bootstrap_replicates - offset)
        indices = rng.integers(0, len(rows), size=(width, len(rows)))
        boot_curves[offset : offset + width] = curves[indices].mean(axis=1)
    curve_lo, curve_hi = np.quantile(boot_curves, [0.025, 0.975], axis=0)
    curve_mean = curves.mean(axis=0)
    curve_rows = []
    for row_index, row in enumerate(rows):
        for decision_idx in range(25):
            step = row["conditions"]["default"]["steps"][decision_idx]
            curve_rows.append({
                "row_type": "block",
                "prompt_idx": row["prompt_idx"],
                "block_id": row["selected_block_id"],
                "decision_idx": decision_idx,
                "normalized_progress": (decision_idx + 1) / 25,
                "delta_t": step["delta_hat_pi"],
                "cumulative_delta": curves[row_index, decision_idx],
                "mean_cumulative_delta": "",
                "ci95_lower": "",
                "ci95_upper": "",
                "endpoint_mean_uplift": "",
            })
    for decision_idx in range(25):
        curve_rows.append({
            "row_type": "aggregate",
            "prompt_idx": "",
            "block_id": "",
            "decision_idx": decision_idx,
            "normalized_progress": (decision_idx + 1) / 25,
            "delta_t": "",
            "cumulative_delta": "",
            "mean_cumulative_delta": curve_mean[decision_idx],
            "ci95_lower": curve_lo[decision_idx],
            "ci95_upper": curve_hi[decision_idx],
            "endpoint_mean_uplift": default_summary["endpoint_uplift_mean"],
        })
    write_csv(output / "cumulative_curve.csv", curve_rows)

    progress = np.arange(1, 26) / 25
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    for curve in curves:
        axis.plot(progress, curve, color="#4C78A8", alpha=0.18, linewidth=0.9)
    axis.fill_between(progress, curve_lo, curve_hi, color="#F2CF5B", alpha=0.35, label="95% block bootstrap CI")
    axis.plot(progress, curve_mean, color="#D1495B", linewidth=2.6, label="Mean cumulative reverse-hybrid advantage")
    axis.scatter([1.0], [default_summary["endpoint_uplift_mean"]], color="#111111", s=55, zorder=5, label="Independent mean endpoint uplift")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Normalized semantic-block decoding progress")
    axis.set_ylabel(r"$\sum_{s\leq t}\widehat{\Delta}^{\pi}_s$")
    axis.set_title("LLaDA-8B LongForm reverse-hybrid theorem closure")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(output / "cumulative_curve.png", dpi=180)
    plt.close(figure)

    summary = {
        "complete": True,
        "prompts": len(rows),
        "blocks": len(rows),
        "default_decisions": len(rows) * 25,
        "collapse_blocks": len(collapse_rows),
        "collapse_decisions": len(collapse_rows) * 25,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_unit": "prompt_block",
        "default": default_summary,
        "exact_collapse": collapse_summary,
        "default_step_signs": default_sign,
        "collapse_step_signs": collapse_sign,
        "candidate_diversity_vs_cumulative_delta": diversity_correlation,
        "collapse_endpoint_uplift_reduction": {
            "mean_default_minus_collapse": (
                float(endpoint_reduction.mean()) if collapse_rows else None
            ),
            "paired_bootstrap_ci95": list(reduction_ci),
        },
        "endpoint_vs_cumulative_correlations": safe_correlations(
            default_cumulative, default_endpoint
        ),
        "main_table": main_table,
        "sanity": {
            "all_prompt_ids_present": ids == list(range(args.expected_prompts)),
            "no_duplicate_prompt_block": len(set(block_keys)) == len(block_keys),
            "all_25_steps_present": all(
                len(row["conditions"]["default"]["steps"]) == 25 for row in rows
            ),
            "all_eos_checks_pass": all(
                row["sanity"]["selected_block_before_first_eos"] for row in rows
            ),
            "all_rng_checks_pass": all(
                row["sanity"]["all_condition_sanity_passed"] for row in rows
            ),
            "collapse_exact_zero": all(
                abs(value) <= 1e-12
                for value in collapse_cumulative + collapse_endpoint
            ),
        },
    }
    (output / "summary_reverse_hybrid.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    required = [
        "raw_reverse_hybrid.jsonl",
        "summary_reverse_hybrid.json",
        "summary_reverse_hybrid.csv",
        "cumulative_curve.csv",
        "closure_by_block.csv",
    ]
    if collapse_rows:
        required.append("collapse_summary.csv")
    for name in required:
        if not (output / name).exists() or (output / name).stat().st_size == 0:
            raise RuntimeError(f"required output missing or empty: {name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
