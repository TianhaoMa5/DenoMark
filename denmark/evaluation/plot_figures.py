#!/usr/bin/env python3
"""Render all result-driven paper figures from documented tidy CSV schemas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable


METHOD_ORDER = ("denmark", "dlm_kgw", "dgmark", "patternmark", "umr")
METHOD_NAMES = {
    "denmark": "DenMark",
    "dlm_kgw": "DLM-KGW",
    "dgmark": "DGMark",
    "patternmark": "PatternMark",
    "umr": "UMR",
}
COLORS = {
    "denmark": "#286F9B",
    "dlm_kgw": "#D98E2F",
    "dgmark": "#55A868",
    "patternmark": "#C44E52",
    "umr": "#8172B3",
}
METRICS = (
    ("tpr_at_0_5pct", "TPR@0.5% FPR"),
    ("tpr_at_1pct", "TPR@1% FPR"),
    ("tpr_at_5pct", "TPR@5% FPR"),
    ("auc", "AUC"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="figure", required=True)
    for name in (
        "sensitivity",
        "scan-range",
        "max-length",
        "temperature-position",
        "open-source-attacks",
        "token-level-attacks",
        "trajectory",
    ):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--input", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"empty CSV: {path}")
    return result


def require(source: list[dict[str, str]], fields: Iterable[str]) -> None:
    missing = set(fields) - set(source[0])
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def setup_plotting():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelweight": "bold",
            "axes.edgecolor": "#42484D",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def finish(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")


def style_axis(axis, *, percentage: bool = False) -> None:
    if percentage:
        axis.set_ylim(0.0, 1.02)
    axis.grid(True, color="#C9CED3", linewidth=0.55, alpha=0.72)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def group_mean(
    source: list[dict[str, str]],
    group_fields: tuple[str, ...],
    value_field: str,
) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in source:
        grouped[tuple(row[field] for field in group_fields)].append(float(row[value_field]))
    return {key: fmean(values) for key, values in grouped.items()}


def plot_sensitivity(source: list[dict[str, str]], output: Path) -> None:
    require(
        source,
        ("sweep", "x", "dataset", "tpr_at_1pct", "auc", "delta_mean_logppl"),
    )
    plt = setup_plotting()
    sweep_specs = (
        ("candidate_temperature", "Candidate temperature"),
        ("num_candidates", r"Candidates $K_t$"),
        ("rollout_count", r"Rollouts $R$"),
        ("semantic_unit_size", r"Unit size $m$"),
    )
    metrics = (
        ("tpr_at_1pct", "TPR@1% FPR"),
        ("auc", "AUC"),
        ("delta_mean_logppl", r"$\Delta\,\mathrm{mean}(\log\mathrm{PPL})$"),
    )
    dataset_names = {"finance_qa": "Finance-QA", "alpacafarm": "AlpacaFarm"}
    dataset_colors = {"finance_qa": "#286F9B", "alpacafarm": "#D98E2F"}
    fig, axes = plt.subplots(3, 4, figsize=(13.8, 7.7))
    for column, (sweep, title) in enumerate(sweep_specs):
        sweep_rows = [row for row in source if row["sweep"] == sweep]
        if not sweep_rows:
            raise ValueError(f"no rows for sensitivity sweep {sweep!r}")
        for metric_index, (metric, label) in enumerate(metrics):
            axis = axes[metric_index, column]
            for dataset in dataset_names:
                points = sorted(
                    (row for row in sweep_rows if row["dataset"] == dataset),
                    key=lambda row: float(row["x"]),
                )
                if not points:
                    raise ValueError(f"no {dataset} rows for sweep {sweep}")
                axis.plot(
                    [float(row["x"]) for row in points],
                    [float(row[metric]) for row in points],
                    marker="o",
                    linewidth=2.0,
                    color=dataset_colors[dataset],
                    label=dataset_names[dataset],
                )
            style_axis(axis, percentage=metric.startswith("tpr"))
            if metric_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.91, bottom=0.07, wspace=0.25, hspace=0.27)
    finish(fig, output)
    plt.close(fig)


def scan_label(row: dict[str, str]) -> str:
    minimum, maximum = int(row["scan_min"]), int(row["scan_max"])
    return str(minimum) if minimum == maximum else f"{minimum}-{maximum}"


def plot_scan_range(source: list[dict[str, str]], output: Path) -> None:
    require(source, ("scan_min", "scan_max", "condition", *(item[0] for item in METRICS)))
    plt = setup_plotting()
    ranges = sorted(
        {(int(row["scan_min"]), int(row["scan_max"])) for row in source},
        key=lambda pair: (pair[1] - pair[0], pair[0]),
    )
    conditions = sorted({row["condition"] for row in source})
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.25), sharex=True)
    for axis, (metric, title) in zip(axes, METRICS):
        means = group_mean(source, ("scan_min", "scan_max", "condition"), metric)
        for index, condition in enumerate(conditions):
            values = [means[(str(lo), str(hi), condition)] for lo, hi in ranges]
            axis.plot(range(len(ranges)), values, marker="o", linewidth=1.8, label=condition)
        axis.set_xticks(range(len(ranges)), [str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in ranges], rotation=30)
        axis.set_title(title)
        style_axis(axis, percentage=metric.startswith("tpr"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.80, bottom=0.22, wspace=0.24)
    finish(fig, output)
    plt.close(fig)


def plot_max_length(source: list[dict[str, str]], output: Path) -> None:
    require(source, ("base", "dataset", "max_length", *(item[0] for item in METRICS)))
    plt = setup_plotting()
    panels = sorted({(row["base"], row["dataset"]) for row in source})
    if len(panels) != 4:
        raise ValueError(f"maximum-length figure expects four base/dataset panels, got {panels}")
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.3), sharey=True)
    for axis, panel in zip(axes, panels):
        selected = [row for row in source if (row["base"], row["dataset"]) == panel]
        for metric, label in METRICS:
            points = sorted(selected, key=lambda row: int(row["max_length"]))
            axis.plot(
                [int(row["max_length"]) for row in points],
                [float(row[metric]) for row in points],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        axis.set_title(f"{panel[0]} / {panel[1]}")
        axis.set_xlabel("Maximum tokens")
        style_axis(axis, percentage=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.80, bottom=0.17, wspace=0.18)
    finish(fig, output)
    plt.close(fig)


def plot_temperature_position(source: list[dict[str, str]], output: Path) -> None:
    require(source, ("base", "dataset", "position_selection", "temperature", "tpr_at_1pct", "auc"))
    plt = setup_plotting()
    bases = sorted({row["base"] for row in source})
    if len(bases) != 2:
        raise ValueError("temperature-position figure expects two backbones")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1), sharex=True)
    for column, base in enumerate(bases):
        for row_index, (metric, title) in enumerate((("tpr_at_1pct", "TPR@1% FPR"), ("auc", "AUC"))):
            axis = axes[row_index, column]
            means = group_mean(
                [row for row in source if row["base"] == base],
                ("position_selection", "temperature"),
                metric,
            )
            for position in ("random", "low_confidence"):
                points = sorted(
                    ((float(temp), value) for (mode, temp), value in means.items() if mode == position),
                )
                if not points:
                    raise ValueError(f"missing {base} {position} temperature series")
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    linewidth=2.0,
                    label=position.replace("_", " ").title(),
                )
            axis.set_title(f"{base}: {title}")
            style_axis(axis, percentage=metric.startswith("tpr"))
            if row_index == 1:
                axis.set_xlabel("Candidate temperature")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.89, bottom=0.09, wspace=0.19, hspace=0.28)
    finish(fig, output)
    plt.close(fig)


def plot_open_source(source: list[dict[str, str]], output: Path) -> None:
    require(source, ("family", "setting", "base", "dataset", "method", "tpr_at_0_5pct", "tpr_at_1pct"))
    plt = setup_plotting()
    panels = (("parrot", "tpr_at_0_5pct"), ("parrot", "tpr_at_1pct"), ("dipper", "tpr_at_0_5pct"), ("dipper", "tpr_at_1pct"))
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.25), sharey=True)
    for axis, (family, metric) in zip(axes, panels):
        selected = [row for row in source if row["family"] == family]
        grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
        for row in selected:
            grouped[(row["method"], float(row["setting"]))].append(float(row[metric]))
        means = {key: fmean(values) for key, values in grouped.items()}
        settings = sorted({float(row["setting"]) for row in selected})
        for method in METHOD_ORDER:
            if not any(key[0] == method for key in means):
                continue
            axis.plot(
                settings,
                [means[(method, setting)] for setting in settings],
                color=COLORS[method],
                marker="o",
                linewidth=1.9,
                label=METHOD_NAMES[method],
            )
        axis.set_title(f"{family.title()}: {'TPR@0.5%' if metric.endswith('0_5pct') else 'TPR@1%'}")
        axis.set_xlabel("Candidate prefix size" if family == "parrot" else "Lexical diversity")
        style_axis(axis, percentage=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.80, bottom=0.17, wspace=0.17)
    finish(fig, output)
    plt.close(fig)


def plot_token_level(source: list[dict[str, str]], output: Path) -> None:
    require(source, ("attack", "ratio", "base", "dataset", "method", *(item[0] for item in METRICS)))
    plt = setup_plotting()
    import numpy as np

    attacks = ("deletion", "context_aware_substitution", "adjacent_swap")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), sharey=True)
    width = 0.18
    x = np.arange(len(METHOD_ORDER))
    metric_colors = ("#4C72B0", "#DD8452", "#55A868", "#C44E52")
    for axis, attack in zip(axes, attacks):
        selected = [row for row in source if row["attack"] == attack]
        for metric_index, ((metric, label), color) in enumerate(zip(METRICS, metric_colors)):
            means = group_mean(selected, ("method",), metric)
            values = [100.0 * means[(method,)] for method in METHOD_ORDER]
            axis.bar(x + (metric_index - 1.5) * width, values, width, label=label, color=color)
        axis.set_title(attack.replace("_", " ").title())
        axis.set_xticks(x, [METHOD_NAMES[method] for method in METHOD_ORDER], rotation=18)
        axis.set_ylim(0, 103)
        axis.grid(axis="y", color="#C9CED3", linewidth=0.55, alpha=0.72)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.80, bottom=0.20, wspace=0.16)
    finish(fig, output)
    plt.close(fig)


def plot_trajectory(source: list[dict[str, str]], output: Path) -> None:
    require(source, ("base", "prompt_id", "step", "cumulative_advantage"))
    plt = setup_plotting()
    import numpy as np

    bases = sorted({row["base"] for row in source})
    if len(bases) != 2:
        raise ValueError("trajectory figure expects exactly two backbones")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.65))
    rng = np.random.default_rng(42)
    for axis, base in zip(axes, bases):
        selected = [row for row in source if row["base"] == base]
        prompt_ids = sorted({row["prompt_id"] for row in selected})
        steps = sorted({int(row["step"]) for row in selected})
        lookup = {(row["prompt_id"], int(row["step"])): float(row["cumulative_advantage"]) for row in selected}
        matrix = np.asarray([[lookup[(prompt_id, step)] for step in steps] for prompt_id in prompt_ids])
        for trajectory in matrix:
            axis.plot(steps, trajectory, color="#7FA8C2", alpha=0.18, linewidth=0.6)
        means = matrix.mean(axis=0)
        bootstrap = matrix[rng.integers(0, len(matrix), size=(2000, len(matrix)))].mean(axis=1)
        lower, upper = np.quantile(bootstrap, (0.025, 0.975), axis=0)
        axis.fill_between(steps, lower, upper, color="#286F9B", alpha=0.22)
        axis.plot(steps, means, color="#286F9B", linewidth=2.4)
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_title(base)
        axis.set_xlabel("Decoding step")
        style_axis(axis)
    axes[0].set_ylabel("Cumulative continued-policy advantage")
    fig.subplots_adjust(left=0.10, right=0.995, top=0.88, bottom=0.16, wspace=0.19)
    finish(fig, output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = rows(args.input)
    functions = {
        "sensitivity": plot_sensitivity,
        "scan-range": plot_scan_range,
        "max-length": plot_max_length,
        "temperature-position": plot_temperature_position,
        "open-source-attacks": plot_open_source,
        "token-level-attacks": plot_token_level,
        "trajectory": plot_trajectory,
    }
    functions[args.figure](source, args.output)
    print(json.dumps({"figure": args.figure, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
