#!/usr/bin/env python3
"""Validate and summarize the LLaDA continued-policy block diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[3]))

from denomark.experiments.trajectory.llada_cumulative_metrics import (  # noqa: E402
    bootstrap_mean_ci,
    full_k_theorem_metrics,
    pearson_safe,
    sign_agreement,
    spearman_safe,
    summarize_vector,
)

R_VALUES = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_prompts", type=int, default=10)
    parser.add_argument("--expected_full_k_blocks", type=int, default=3)
    parser.add_argument("--bootstrap_seed", type=int, default=20260814)
    parser.add_argument("--bootstrap_repeats", type=int, default=10_000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    by_prompt: dict[int, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_idx = int(row["prompt_idx"])
                if prompt_idx in by_prompt:
                    raise ValueError(
                        f"duplicate prompt_idx={prompt_idx}: latest at {path}:{line_no}"
                    )
                by_prompt[prompt_idx] = row
    return [by_prompt[index] for index in sorted(by_prompt)]


def q_means(step: dict[str, Any]) -> dict[int, float]:
    output = {}
    for key, values in step["Q_pi_block_raw_scores"].items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"invalid continued-Q values for candidate {key}")
        output[int(key)] = float(array.mean())
    return output


def validate_and_recompute(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    prompt_idx = int(row["prompt_idx"])
    config = row["config"]
    block_size = int(config["semantic_block_size"])
    k = int(config["K"])
    n_rollouts = int(config["diagnostic_rollouts"])
    continued_repeats = int(config["continued_repeats"])
    endpoint_repeats = int(config["block_endpoint_repeats"])
    steps = row.get("steps", [])
    if len(steps) != block_size:
        errors.append(f"prompt {prompt_idx}: {len(steps)} decisions != {block_size}")

    sums_a = {r: 0.0 for r in R_VALUES}
    sums_rollout = {r: 0.0 for r in R_VALUES}
    full_k_step_recomputed: list[dict[str, Any]] = []
    for expected_step, step in enumerate(steps):
        if int(step.get("step", -1)) != expected_step:
            errors.append(f"prompt {prompt_idx}: missing/reordered decision {expected_step}")
        raw = np.asarray(step.get("rollout_raw_scores"), dtype=float)
        if raw.shape != (k, n_rollouts) or not np.isfinite(raw).all():
            errors.append(
                f"prompt {prompt_idx} step {expected_step}: rollout shape/finite failure {raw.shape}"
            )
            continue
        candidate_states = np.asarray(step.get("candidate_state_token_ids"))
        if candidate_states.shape != (k, int(config["generation_length"])):
            errors.append(
                f"prompt {prompt_idx} step {expected_step}: candidate-state shape {candidate_states.shape}"
            )
        means = q_means(step)
        for candidate, values in step["Q_pi_block_raw_scores"].items():
            if len(values) != continued_repeats:
                errors.append(
                    f"prompt {prompt_idx} step {expected_step}: candidate {candidate} Q repeats incomplete"
                )
        if 0 not in means:
            errors.append(f"prompt {prompt_idx} step {expected_step}: missing reference Q")
            continue
        for r in R_VALUES:
            mu = raw[:, :r].mean(axis=1)
            winner = int(np.argmax(mu))
            if winner not in means:
                errors.append(
                    f"prompt {prompt_idx} step {expected_step}: missing R={r} winner Q"
                )
                continue
            advantage = means[winner] - means[0]
            rollout_gain = float(mu[winner] - mu[0])
            stored = step["metrics"][f"R{r}"]
            if abs(advantage - float(stored["continued_policy_advantage"])) > 1e-10:
                errors.append(
                    f"prompt {prompt_idx} step {expected_step}: R={r} A mismatch"
                )
            if abs(rollout_gain - float(stored["rollout_gain"])) > 1e-10:
                errors.append(
                    f"prompt {prompt_idx} step {expected_step}: R={r} rollout gain mismatch"
                )
            sums_a[r] += advantage
            sums_rollout[r] += rollout_gain

        stored_full_k = step.get("full_k_theorem")
        if stored_full_k is not None:
            if set(means) != set(range(k)):
                errors.append(
                    f"prompt {prompt_idx} step {expected_step}: full-K Q indices incomplete"
                )
            else:
                recomputed = full_k_theorem_metrics(
                    raw, [means[index] for index in range(k)], R_VALUES
                )
                if recomputed["max_identity_abs_error"] > 1e-10:
                    errors.append(
                        f"prompt {prompt_idx} step {expected_step}: theorem identity failed"
                    )
                if abs(
                    recomputed["Gamma_Q_pi"] - float(stored_full_k["Gamma_Q_pi"])
                ) > 1e-10:
                    errors.append(
                        f"prompt {prompt_idx} step {expected_step}: Gamma_Q_pi mismatch"
                    )
                full_k_step_recomputed.append(recomputed)

    summary = row["block_summary"]
    for r in R_VALUES:
        if abs(sums_a[r] - float(summary[f"sum_A_R{r}"])) > 1e-10:
            errors.append(f"prompt {prompt_idx}: cumulative A R={r} mismatch")
        if abs(sums_rollout[r] - float(summary[f"sum_Gamma_R{r}"])) > 1e-10:
            errors.append(f"prompt {prompt_idx}: cumulative rollout gain R={r} mismatch")

    block_level = row.get("block_level", {})
    for name in ("V_pi_raw_scores", "V_pi0_raw_scores"):
        values = np.asarray(block_level.get(name, []), dtype=float)
        if values.shape != (endpoint_repeats,) or not np.isfinite(values).all():
            errors.append(f"prompt {prompt_idx}: {name} incomplete")
    if steps and not math.isclose(float(steps[-1]["progress"]), 1.0):
        errors.append(f"prompt {prompt_idx}: final block progress is not one")
    for name, value in row.get("sanity", {}).items():
        if value is not True:
            errors.append(f"prompt {prompt_idx}: sanity {name}={value!r}")

    full_k = None
    if full_k_step_recomputed:
        if len(full_k_step_recomputed) != len(steps):
            errors.append(f"prompt {prompt_idx}: partial full-K block")
        gamma_sum = float(sum(step["Gamma_Q_pi"] for step in full_k_step_recomputed))
        by_r = {}
        for r in R_VALUES:
            eps_sum = float(sum(step["by_R"][f"R{r}"]["epsilon_Q_pi"] for step in full_k_step_recomputed))
            advantage_sum = float(sum(step["by_R"][f"R{r}"]["continued_policy_advantage"] for step in full_k_step_recomputed))
            difference = gamma_sum - eps_sum
            identity_error = abs(difference - advantage_sum)
            if identity_error > 1e-10:
                errors.append(f"prompt {prompt_idx}: cumulative theorem identity R={r} failed")
            by_r[f"R{r}"] = {
                "sum_Gamma_Q_pi": gamma_sum,
                "sum_epsilon_Q_pi": eps_sum,
                "sum_Gamma_minus_epsilon": difference,
                "sum_A": advantage_sum,
                "identity_abs_error": identity_error,
                "mean_step_epsilon": eps_sum / max(1, len(full_k_step_recomputed)),
                "positive_sum": bool(difference > 0),
            }
        full_k = {
            "prompt_idx": prompt_idx,
            "block_id": int(row["selected_block_id"]),
            "num_decisions": len(steps),
            "sum_Gamma_Q_pi": gamma_sum,
            "by_R": by_r,
        }
    return errors, {
        "sum_A": sums_a,
        "sum_rollout": sums_rollout,
        "full_k": full_k,
    }


def bootstrap_curve(matrix: np.ndarray, seed: int, repeats: int) -> tuple[np.ndarray, np.ndarray]:
    if len(matrix) == 1:
        return matrix[0], matrix[0]
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(matrix), size=(repeats, len(matrix)))
    sampled = matrix[draw_indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975], axis=0)
    return low, high


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.inputs)
    recomputed: dict[int, dict[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        row_errors, values = validate_and_recompute(row)
        errors.extend(row_errors)
        recomputed[int(row["prompt_idx"])] = values
    if errors:
        raise SystemExit("validation failed:\n- " + "\n- ".join(errors))

    raw_path = args.output_dir / "raw.jsonl"
    with raw_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    endpoint = np.asarray(
        [float(row["block_summary"]["final_block_uplift"]) for row in rows], dtype=float
    )
    endpoint_summary = summarize_vector(endpoint, seed=args.bootstrap_seed + 500)
    main_table: list[dict[str, Any]] = []
    for r in R_VALUES:
        cumulative_a = np.asarray(
            [recomputed[int(row["prompt_idx"])]["sum_A"][r] for row in rows], dtype=float
        )
        cumulative_rollout = np.asarray(
            [recomputed[int(row["prompt_idx"])]["sum_rollout"][r] for row in rows], dtype=float
        )
        a_summary = summarize_vector(cumulative_a, seed=args.bootstrap_seed + r)
        mean_r10 = float(np.mean([
            recomputed[int(row["prompt_idx"])]["sum_A"][10] for row in rows
        ])) if rows else float("nan")
        relative = float(a_summary["mean"] / mean_r10) if mean_r10 != 0 else float("nan")
        main_table.append({
            "R": r,
            "N blocks": len(rows),
            "Cumulative Continued Gain Mean": a_summary["mean"],
            "Cumulative Continued Gain Median": a_summary["median"],
            "Cumulative Continued Gain CI95 Low": a_summary["bootstrap_ci95_low"],
            "Cumulative Continued Gain CI95 High": a_summary["bootstrap_ci95_high"],
            "Positive Block Ratio": a_summary["positive_ratio"],
            "Final Block Uplift Mean": endpoint_summary["mean"],
            "Final Block Uplift CI95 Low": endpoint_summary["bootstrap_ci95_low"],
            "Final Block Uplift CI95 High": endpoint_summary["bootstrap_ci95_high"],
            "Pearson(sum_A, Delta_end)": pearson_safe(cumulative_a, endpoint),
            "Spearman(sum_A, Delta_end)": spearman_safe(cumulative_a, endpoint),
            "Sign Agreement": sign_agreement(cumulative_a, endpoint),
            "MAE(sum_A, Delta_end)": float(np.mean(np.abs(cumulative_a - endpoint))),
            "Cumulative Rollout Gain Mean": float(np.mean(cumulative_rollout)),
            "Relative to R10": relative,
            "Continued Value Loss vs R10": float(mean_r10 - a_summary["mean"]),
        })
    summary_csv = args.output_dir / "summary.csv"
    write_csv(summary_csv, main_table)

    curve_rows: list[dict[str, Any]] = []
    r3_matrix = []
    for row in rows:
        prompt_idx = int(row["prompt_idx"])
        endpoint_uplift = float(row["block_summary"]["final_block_uplift"])
        trajectory = row["cumulative_trajectory"]
        r3_matrix.append([float(point["cumulative_A_R3"]) for point in trajectory])
        for point, step in zip(trajectory, row["steps"], strict=True):
            curve_rows.append({
                "curve_type": "individual",
                "prompt_idx": prompt_idx,
                "block_id": int(row["selected_block_id"]),
                "decision_idx": int(point["decision_idx"]),
                "block_progress": float(point["block_progress"]),
                "A_step_R3": float(step["metrics"]["R3"]["continued_policy_advantage"]),
                "cumulative_A_R3": float(point["cumulative_A_R3"]),
                "cumulative_A_R10": float(point["cumulative_A_R10"]),
                "cumulative_rollout_gain_R3": float(point["cumulative_rollout_gain_R3"]),
                "endpoint_uplift": endpoint_uplift,
                "cumulative_A_R3_ci95_low": "",
                "cumulative_A_R3_ci95_high": "",
            })
    if r3_matrix:
        matrix = np.asarray(r3_matrix, dtype=float)
        low, high = bootstrap_curve(matrix, args.bootstrap_seed, args.bootstrap_repeats)
        mean = matrix.mean(axis=0)
        for decision_idx in range(matrix.shape[1]):
            curve_rows.append({
                "curve_type": "mean",
                "prompt_idx": "",
                "block_id": "",
                "decision_idx": decision_idx,
                "block_progress": (decision_idx + 1) / matrix.shape[1],
                "A_step_R3": mean[decision_idx] - (mean[decision_idx - 1] if decision_idx else 0.0),
                "cumulative_A_R3": mean[decision_idx],
                "cumulative_A_R10": float(np.mean([
                    row["cumulative_trajectory"][decision_idx]["cumulative_A_R10"]
                    for row in rows
                ])),
                "cumulative_rollout_gain_R3": float(np.mean([
                    row["cumulative_trajectory"][decision_idx]["cumulative_rollout_gain_R3"]
                    for row in rows
                ])),
                "endpoint_uplift": float(endpoint.mean()),
                "cumulative_A_R3_ci95_low": float(low[decision_idx]),
                "cumulative_A_R3_ci95_high": float(high[decision_idx]),
            })
    curve_csv = args.output_dir / "cumulative_curve.csv"
    write_csv(curve_csv, curve_rows)
    plot_path: Path | None = None
    if r3_matrix:
        try:
            import matplotlib.pyplot as plt

            matrix = np.asarray(r3_matrix, dtype=float)
            x = np.arange(1, matrix.shape[1] + 1) / matrix.shape[1]
            low, high = bootstrap_curve(matrix, args.bootstrap_seed, args.bootstrap_repeats)
            mean = matrix.mean(axis=0)
            fig, ax = plt.subplots(figsize=(7.4, 4.8))
            for curve in matrix:
                ax.plot(x, curve, color="#4C78A8", alpha=0.20, linewidth=1.0)
            ax.fill_between(x, low, high, color="#4C78A8", alpha=0.16, linewidth=0)
            ax.plot(x, mean, color="#1F4E79", linewidth=2.5, label="Mean cumulative A (R=3)")
            ax.scatter(
                np.ones(len(endpoint)), endpoint, color="#C58A00", alpha=0.42,
                s=22, marker="D", label="Block endpoint uplift",
            )
            ax.scatter([1.0], [endpoint.mean()], color="#8A5A00", s=58, marker="D", zorder=4)
            ax.axhline(0.0, color="#555555", linewidth=0.9)
            ax.set_xlim(0.0, 1.03)
            ax.set_xlabel("Semantic block decoding progress")
            ax.set_ylabel("Empirical cumulative continued-policy drift")
            fig.suptitle(
                "LLaDA-8B LongForm cumulative block drift",
                x=0.125, y=0.985, ha="left", fontsize=15,
            )
            ax.set_title(
                f"R=3; {len(rows)} blocks; shaded band is block-bootstrap 95% CI",
                loc="left", fontsize=9, color="#555555", pad=10,
            )
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            plot_path = args.output_dir / "cumulative_curve_R3.png"
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"plot warning: {exc!r}", file=sys.stderr)

    full_k_blocks = [
        values["full_k"] for values in recomputed.values() if values["full_k"] is not None
    ]
    full_k_table: list[dict[str, Any]] = []
    for r in R_VALUES:
        gamma = np.asarray([block["sum_Gamma_Q_pi"] for block in full_k_blocks], dtype=float)
        epsilon = np.asarray([block["by_R"][f"R{r}"]["sum_epsilon_Q_pi"] for block in full_k_blocks], dtype=float)
        difference = gamma - epsilon
        full_k_table.append({
            "R": r,
            "N full-K blocks": len(full_k_blocks),
            "Mean sum_t Gamma_Q_pi": float(gamma.mean()) if len(gamma) else float("nan"),
            "Total sum_t Gamma_Q_pi": float(gamma.sum()),
            "Mean sum_t epsilon_Q_pi": float(epsilon.mean()) if len(epsilon) else float("nan"),
            "Total sum_t epsilon_Q_pi": float(epsilon.sum()),
            "Mean sum_t(Gamma-epsilon)": float(difference.mean()) if len(difference) else float("nan"),
            "Total sum_t(Gamma-epsilon)": float(difference.sum()),
            "Positive Sum Blocks": int(np.sum(difference > 0)),
            "Positive Sum Block Ratio": float(np.mean(difference > 0)) if len(difference) else float("nan"),
            "Mean step epsilon_t": float(epsilon.sum() / sum(block["num_decisions"] for block in full_k_blocks)) if full_k_blocks else float("nan"),
        })
    full_k_csv = args.output_dir / "full_k_theorem_audit.csv"
    write_csv(full_k_csv, full_k_table)

    all_steps = [(row, step) for row in rows for step in row["steps"]]
    unique_counts = np.asarray([step["num_unique_candidates"] for _, step in all_steps], dtype=float)
    a_step_r3 = np.asarray([
        step["metrics"]["R3"]["continued_policy_advantage"] for _, step in all_steps
    ], dtype=float)
    rollout_gain_r3 = np.asarray([
        step["metrics"]["R3"]["rollout_gain"] for _, step in all_steps
    ], dtype=float)
    audit_steps = [step for _, step in all_steps if step.get("full_k_theorem") is not None]
    audit_unique = np.asarray([step["num_unique_candidates"] for step in audit_steps], dtype=float)
    audit_gamma = np.asarray([
        step["full_k_theorem"]["Gamma_Q_pi"] for step in audit_steps
    ], dtype=float)
    collapse_audit_gamma = [
        float(step["full_k_theorem"]["Gamma_Q_pi"])
        for step in audit_steps if int(step["num_unique_candidates"]) <= 1
    ]
    mean_block_diversity = np.asarray([
        row["block_summary"]["mean_candidate_diversity"] for row in rows
    ], dtype=float)
    sum_a_r3 = np.asarray([
        recomputed[int(row["prompt_idx"])]["sum_A"][3] for row in rows
    ], dtype=float)
    diversity = {
        "all_steps": len(all_steps),
        "candidate_collapse_steps": int(np.sum(unique_counts <= 1)),
        "step_unique_count_vs_A_R3_pearson": pearson_safe(unique_counts, a_step_r3),
        "step_unique_count_vs_A_R3_spearman": spearman_safe(unique_counts, a_step_r3),
        "step_unique_count_vs_rollout_gain_R3_pearson": pearson_safe(unique_counts, rollout_gain_r3),
        "full_k_unique_count_vs_Gamma_Q_pi_pearson": pearson_safe(audit_unique, audit_gamma),
        "full_k_unique_count_vs_Gamma_Q_pi_spearman": spearman_safe(audit_unique, audit_gamma),
        "block_mean_diversity_vs_sum_A_R3_pearson": pearson_safe(mean_block_diversity, sum_a_r3),
        "block_mean_diversity_vs_endpoint_uplift_pearson": pearson_safe(mean_block_diversity, endpoint),
        "collapse_full_k_Gamma_values": collapse_audit_gamma,
        "all_full_k_collapse_steps_have_zero_Gamma": (
            bool(collapse_audit_gamma)
            and all(abs(value) <= 1e-12 for value in collapse_audit_gamma)
        ),
    }

    full_k_json = args.output_dir / "full_k_theorem_audit.json"
    full_k_payload = {
        "schema_version": 1,
        "definition": "continued-policy Q_pi block value; not rollout-score gain",
        "n_blocks": len(full_k_blocks),
        "expected_blocks": args.expected_full_k_blocks,
        "complete": len(full_k_blocks) == args.expected_full_k_blocks,
        "blocks": full_k_blocks,
        "table": full_k_table,
        "identity_tolerance": 1e-10,
        "all_identities_passed": all(
            block["by_R"][f"R{r}"]["identity_abs_error"] <= 1e-10
            for block in full_k_blocks for r in R_VALUES
        ),
        "candidate_diversity": diversity,
    }
    full_k_json.write_text(json.dumps(full_k_payload, ensure_ascii=False, indent=2) + "\n")

    r3_row = next(item for item in main_table if item["R"] == 3)
    summary = {
        "schema_version": 2,
        "experiment": "llada8b_longform_cumulative_continued_policy_diagnostic",
        "complete": len(rows) == args.expected_prompts and len(full_k_blocks) == args.expected_full_k_blocks,
        "n_prompts": len(rows),
        "n_blocks": len(rows),
        "prompt_indices": [int(row["prompt_idx"]) for row in rows],
        "total_block_decisions": sum(int(row["num_block_decisions"]) for row in rows),
        "mean_block_decisions": float(np.mean([row["num_block_decisions"] for row in rows])) if rows else float("nan"),
        "endpoint_uplift": endpoint_summary,
        "R3_cumulative_gain_significantly_positive": bool(
            r3_row["Cumulative Continued Gain CI95 Low"] > 0
        ),
        "main_table": main_table,
        "candidate_diversity": diversity,
        "degeneracy": {
            "eos_or_special_selected_blocks": sum(
                not row["sanity"]["selected_block_no_special_or_mask"] for row in rows
            ),
            "missing_step_blocks": sum(len(row["steps"]) != row["config"]["semantic_block_size"] for row in rows),
            "candidate_collapse_steps": int(np.sum(unique_counts <= 1)),
            "all_rng_trajectory_checks_passed": all(
                row["sanity"]["diagnostic_rng_preserved_main_trajectory"] for row in rows
            ),
        },
        "full_k_audit": {
            "n_blocks": len(full_k_blocks),
            "expected_blocks": args.expected_full_k_blocks,
            "complete": len(full_k_blocks) == args.expected_full_k_blocks,
            "table": full_k_table,
        },
        "paths": {
            "raw_jsonl": str(raw_path.resolve()),
            "summary_json": str((args.output_dir / "summary.json").resolve()),
            "summary_csv": str(summary_csv.resolve()),
            "cumulative_curve_csv": str(curve_csv.resolve()),
            "full_k_theorem_audit_json": str(full_k_json.resolve()),
            "full_k_theorem_audit_csv": str(full_k_csv.resolve()),
            **({"cumulative_curve_png": str(plot_path.resolve())} if plot_path else {}),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({
        "n_prompts": len(rows),
        "n_full_k_blocks": len(full_k_blocks),
        "complete": summary["complete"],
        "output_dir": str(args.output_dir.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

