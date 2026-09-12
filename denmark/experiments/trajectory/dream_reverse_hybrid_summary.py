#!/usr/bin/env python3
"""Merge and summarize Dream unit-level reverse-hybrid diagnostic shards."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))

from denmark.experiments.trajectory.llada_reverse_hybrid_metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_mean_ci,
    summarize_block_values,
)


GRID = np.linspace(0.0, 1.0, 11)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", nargs="+", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--expected_units", type=int, default=30)
    p.add_argument("--expected_collapse_units", type=int, default=10)
    p.add_argument("--bootstrap_replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_rows(paths):
    by_key = {}
    for path in paths:
        if path.is_dir():
            candidates = sorted(path.rglob("*.jsonl"))
        else:
            candidates = [path]
        for candidate in candidates:
            for line in candidate.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("experiment") != "dream_unit_reverse_hybrid_closure":
                    continue
                key = (int(row["prompt_idx"]), int(row["selected_unit_id"]))
                if key in by_key:
                    if by_key[key] != row:
                        raise RuntimeError(f"conflicting duplicate prompt/unit {key}")
                    continue
                by_key[key] = row
    return [by_key[key] for key in sorted(by_key)]


def write_csv(path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2
        i = j
    return result


def correlations(left, right):
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
    }


def validate_condition(condition, collapse):
    steps = condition["steps"]
    tau = int(condition["unit_summary"]["local_horizon_tau"])
    if len(steps) != tau or [row["decision_idx"] for row in steps] != list(range(tau)):
        raise RuntimeError("missing/nonconsecutive Dream local decision")
    if not all(all(row["sanity"].values()) for row in steps):
        raise RuntimeError("failed step sanity")
    if any(len(row["paired_future_differences"]) == 0 for row in steps):
        raise RuntimeError("missing raw future scores")
    if collapse:
        if any(row["unique_candidate_count"] != 1 for row in steps):
            raise RuntimeError("collapse candidate set is not exact")
        if abs(condition["unit_summary"]["sum_delta"]) > 1e-12:
            raise RuntimeError("collapse cumulative delta is not exact zero")
        if abs(condition["endpoint"]["endpoint_uplift_hat"]) > 1e-12:
            raise RuntimeError("collapse endpoint is not exact zero")


def sign_summary(rows, condition_name):
    deltas = [
        float(step["delta_hat_pi"])
        for row in rows for step in row["conditions"][condition_name]["steps"]
    ]
    pos = [value for value in deltas if value > 1e-12]
    neg = [value for value in deltas if value < -1e-12]
    zero = [value for value in deltas if abs(value) <= 1e-12]
    total = len(deltas)
    return {
        "total_steps": total,
        "positive_steps": len(pos), "negative_steps": len(neg), "zero_steps": len(zero),
        "positive_step_fraction": len(pos) / total if total else None,
        "negative_step_fraction": len(neg) / total if total else None,
        "zero_step_fraction": len(zero) / total if total else None,
        "mean_positive_delta": float(np.mean(pos)) if pos else 0.0,
        "mean_negative_delta": float(np.mean(neg)) if neg else 0.0,
    }


def interpolated_curve(condition):
    cumulative = np.asarray(condition["unit_summary"]["cumulative_delta"], dtype=float)
    tau = len(cumulative)
    progress = np.arange(1, tau + 1, dtype=float) / tau
    return np.interp(GRID, np.concatenate([[0.0], progress]), np.concatenate([[0.0], cumulative]))


def main():
    args = parse_args()
    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows(args.input_jsonl)
    if len(rows) != args.expected_units:
        raise RuntimeError(f"expected {args.expected_units} unique valid units, got {len(rows)}")
    collapse_rows = [row for row in rows if "exact_collapse" in row["conditions"]]
    if len(collapse_rows) != args.expected_collapse_units:
        raise RuntimeError(
            f"expected {args.expected_collapse_units} collapse units, got {len(collapse_rows)}"
        )
    if len({row["prompt_idx"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate prompt")
    for row in rows:
        if not all(row["sanity"].values()):
            raise RuntimeError(f"failed prompt sanity: {row['prompt_idx']}")
        validate_condition(row["conditions"]["default"], False)
        if "exact_collapse" in row["conditions"]:
            validate_condition(row["conditions"]["exact_collapse"], True)

    default_c = [row["conditions"]["default"]["unit_summary"]["sum_delta"] for row in rows]
    default_u = [row["conditions"]["default"]["endpoint"]["endpoint_uplift_hat"] for row in rows]
    collapse_c = [row["conditions"]["exact_collapse"]["unit_summary"]["sum_delta"] for row in collapse_rows]
    collapse_u = [row["conditions"]["exact_collapse"]["endpoint"]["endpoint_uplift_hat"] for row in collapse_rows]
    default_summary = summarize_block_values(
        default_c, default_u, n_bootstrap=args.bootstrap_replicates, seed=args.seed
    )
    collapse_summary = None
    if collapse_rows:
        collapse_summary = summarize_block_values(
            collapse_c, collapse_u, n_bootstrap=args.bootstrap_replicates, seed=args.seed + 100
        )
    diversity = [row["conditions"]["default"]["unit_summary"]["mean_candidate_diversity"] for row in rows]
    diversity_corr = correlations(diversity, default_c)
    horizons = [row["conditions"]["default"]["unit_summary"]["local_horizon_tau"] for row in rows]
    default_signs = sign_summary(rows, "default")
    collapse_signs = sign_summary(collapse_rows, "exact_collapse")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "raw_reverse_hybrid_dream.jsonl"
    with raw_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    table = []
    conditions = [("Default", default_summary)]
    if collapse_summary is not None:
        conditions.append(("Exact collapse", collapse_summary))
    for label, summary in conditions:
        table.append({
            "Condition": label, "Units": summary["blocks"],
            "Mean sum Delta": summary["cumulative_delta_mean"],
            "Cumulative CI lower": summary["cumulative_delta_ci95"][0],
            "Cumulative CI upper": summary["cumulative_delta_ci95"][1],
            "Endpoint uplift": summary["endpoint_uplift_mean"],
            "Endpoint CI lower": summary["endpoint_uplift_ci95"][0],
            "Endpoint CI upper": summary["endpoint_uplift_ci95"][1],
            "Closure gap": summary["closure_gap_mean"],
            "Closure CI lower": summary["closure_gap_ci95"][0],
            "Closure CI upper": summary["closure_gap_ci95"][1],
            "Positive units": summary["cumulative_delta_positive_block_ratio"],
        })
    write_csv(output / "summary_reverse_hybrid_dream.csv", table)

    closure_rows = []
    for row in rows:
        for condition_name, condition in row["conditions"].items():
            unit = condition["unit_summary"]
            closure_rows.append({
                "prompt_idx": row["prompt_idx"], "unit_id": row["selected_unit_id"],
                "condition": condition_name, "local_horizon_tau": unit["local_horizon_tau"],
                "sum_delta": unit["sum_delta"], "endpoint_uplift": unit["endpoint_uplift"],
                "closure_gap": unit["closure_gap_endpoint_minus_sum_delta"],
                "positive_steps": unit["positive_step_count"],
                "negative_steps": unit["negative_step_count"], "zero_steps": unit["zero_step_count"],
                "mean_candidate_diversity": unit["mean_candidate_diversity"],
                "unit_text": row["selected_unit_text_main_trajectory"],
            })
    write_csv(output / "closure_by_unit_dream.csv", closure_rows)

    collapse_table = []
    for row in collapse_rows:
        default = row["conditions"]["default"]["unit_summary"]
        collapse = row["conditions"]["exact_collapse"]["unit_summary"]
        collapse_table.append({
            "prompt_idx": row["prompt_idx"], "unit_id": row["selected_unit_id"],
            "default_sum_delta": default["sum_delta"], "collapse_sum_delta": collapse["sum_delta"],
            "default_endpoint_uplift": default["endpoint_uplift"],
            "collapse_endpoint_uplift": collapse["endpoint_uplift"],
            "endpoint_uplift_reduction": default["endpoint_uplift"] - collapse["endpoint_uplift"],
        })
    collapse_path = output / "collapse_summary_dream.csv"
    if collapse_table:
        write_csv(collapse_path, collapse_table)
    else:
        collapse_path.write_text(
            "prompt_idx,unit_id,default_sum_delta,collapse_sum_delta,"
            "default_endpoint_uplift,collapse_endpoint_uplift,endpoint_uplift_reduction\n"
        )

    curves = np.asarray([interpolated_curve(row["conditions"]["default"]) for row in rows])
    rng = np.random.default_rng(args.seed + 300)
    boot = np.empty((args.bootstrap_replicates, len(GRID)))
    for offset in range(0, args.bootstrap_replicates, 500):
        width = min(500, args.bootstrap_replicates - offset)
        indices = rng.integers(0, len(rows), size=(width, len(rows)))
        boot[offset:offset + width] = curves[indices].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    mean_curve = curves.mean(axis=0)
    curve_rows = []
    for row_index, row in enumerate(rows):
        condition = row["conditions"]["default"]
        cumulative = condition["unit_summary"]["cumulative_delta"]
        tau = len(cumulative)
        for decision_idx, step in enumerate(condition["steps"]):
            curve_rows.append({
                "row_type": "unit", "prompt_idx": row["prompt_idx"],
                "unit_id": row["selected_unit_id"], "native_decision_idx": decision_idx,
                "normalized_progress": (decision_idx + 1) / tau,
                "delta_t": step["delta_hat_pi"], "cumulative_delta": cumulative[decision_idx],
                "mean_cumulative_delta": "", "ci95_lower": "", "ci95_upper": "",
            })
        for grid_idx, progress in enumerate(GRID):
            curve_rows.append({
                "row_type": "interpolated_unit", "prompt_idx": row["prompt_idx"],
                "unit_id": row["selected_unit_id"], "native_decision_idx": "",
                "normalized_progress": progress, "delta_t": "",
                "cumulative_delta": curves[row_index, grid_idx],
                "mean_cumulative_delta": "", "ci95_lower": "", "ci95_upper": "",
            })
    for grid_idx, progress in enumerate(GRID):
        curve_rows.append({
            "row_type": "aggregate", "prompt_idx": "", "unit_id": "",
            "native_decision_idx": "", "normalized_progress": progress, "delta_t": "",
            "cumulative_delta": "", "mean_cumulative_delta": mean_curve[grid_idx],
            "ci95_lower": lo[grid_idx], "ci95_upper": hi[grid_idx],
        })
    write_csv(output / "cumulative_curve_dream.csv", curve_rows)

    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    for curve in curves:
        axis.plot(GRID, curve, color="#4C78A8", alpha=0.17, linewidth=0.9)
    axis.fill_between(GRID, lo, hi, color="#F2CF5B", alpha=0.35, label="95% unit bootstrap CI")
    axis.plot(GRID, mean_curve, color="#D1495B", linewidth=2.6, label="Mean cumulative advantage")
    axis.scatter([1.0], [default_summary["endpoint_uplift_mean"]], color="#111111", s=55,
                 label="Independent endpoint uplift")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set_xlabel("Normalized Dream unit-resolution progress")
    axis.set_ylabel(r"$\sum_{s\leq t}\widehat{\Delta}^{\pi}_s$")
    axis.set_title("Dream-7B LongForm unit-level reverse-hybrid closure")
    axis.grid(alpha=0.18); axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output / "cumulative_curve_dream.png", dpi=180)
    plt.close(figure)

    paired_default_endpoint = [
        row["conditions"]["default"]["endpoint"]["endpoint_uplift_hat"] for row in collapse_rows
    ]
    reduction = np.asarray(paired_default_endpoint) - np.asarray(collapse_u)
    collapse_reduction = None
    if collapse_rows:
        collapse_reduction = {
            "mean_default_minus_collapse": float(reduction.mean()),
            "paired_bootstrap_ci95": list(paired_bootstrap_mean_ci(
                paired_default_endpoint, collapse_u,
                n_bootstrap=args.bootstrap_replicates, seed=args.seed + 400,
            )),
        }
    summary = {
        "complete": True, "prompts": len(rows), "valid_units": len(rows),
        "default_native_decisions": int(sum(horizons)),
        "collapse_units": len(collapse_rows),
        "collapse_native_decisions": int(collapse_signs["total_steps"]),
        "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_unit": "prompt_semantic_unit",
        "default": default_summary, "exact_collapse": collapse_summary,
        "local_horizon_tau": {
            "mean": float(np.mean(horizons)), "median": float(np.median(horizons)),
            "min": int(min(horizons)), "max": int(max(horizons)), "values": horizons,
        },
        "default_step_signs": default_signs, "collapse_step_signs": collapse_signs,
        "mean_unique_candidates": float(np.mean(diversity)),
        "candidate_diversity_vs_cumulative_delta": diversity_corr,
        "endpoint_vs_cumulative_correlations": correlations(default_c, default_u),
        "collapse_endpoint_uplift_reduction": collapse_reduction,
        "main_table": table,
        "sanity": {
            "unique_prompt_unit": len({(row["prompt_idx"], row["selected_unit_id"]) for row in rows}) == len(rows),
            "all_EOS_special_checks_pass": all(
                row["sanity"]["unit_eos_count_zero"] and row["sanity"]["unit_special_count_zero"]
                for row in rows
            ),
            "all_variable_horizons_complete": all(len(row["conditions"]["default"]["steps"]) > 0 for row in rows),
            "all_rng_checks_pass": all(row["sanity"]["selection_evaluation_rng_independent"] for row in rows),
            "collapse_exact_zero": all(abs(value) <= 1e-12 for value in collapse_c + collapse_u),
        },
    }
    (output / "summary_reverse_hybrid_dream.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    required = [
        "raw_reverse_hybrid_dream.jsonl", "summary_reverse_hybrid_dream.json",
        "summary_reverse_hybrid_dream.csv", "closure_by_unit_dream.csv",
        "cumulative_curve_dream.csv", "collapse_summary_dream.csv",
        "cumulative_curve_dream.png",
    ]
    for name in required:
        if not (output / name).exists() or (output / name).stat().st_size == 0:
            raise RuntimeError(f"missing output: {name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
