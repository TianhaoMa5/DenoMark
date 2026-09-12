#!/usr/bin/env python3
"""Audit that every paper experiment has consistent public reproduction assets."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from denmark.evaluation.build_release import audit_file, selected_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def config_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            paths.extend(config_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(config_paths(item))
    elif isinstance(value, str) and value.startswith(("denmark/", "configs/", "data/", "docs/")):
        paths.append(value)
    return paths


def literal_default(source: Path, flag: str) -> Any:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    matches: list[Any] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or first.value != flag:
            continue
        default = next((keyword.value for keyword in node.keywords if keyword.arg == "default"), None)
        if default is None:
            matches.append(None)
        else:
            matches.append(ast.literal_eval(default))
    if len(matches) != 1:
        raise ValueError(f"expected one {flag} definition in {source}, found {len(matches)}")
    return matches[0]


def local_import_issues(root: Path, sources: list[Path]) -> list[str]:
    issues: list[str] = []
    for source in sources:
        if source.suffix != ".py":
            continue
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("denmark."):
                        candidate = root / Path(*alias.name.split("."))
                        if not candidate.with_suffix(".py").is_file() and not (candidate / "__init__.py").is_file():
                            issues.append(f"{source.relative_to(root)} imports missing {alias.name}")
            if module and module.startswith("denmark."):
                candidate = root / Path(*module.split("."))
                if not candidate.with_suffix(".py").is_file() and not (candidate / "__init__.py").is_file():
                    issues.append(f"{source.relative_to(root)} imports missing {module}")
    return issues


def check(condition: bool, message: str, issues: list[str]) -> None:
    if not condition:
        issues.append(message)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = root / "configs/paper_experiments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    check(
        set(config["datasets"]) == {"finance_qa", "alpacafarm", "longform_qa"},
        "paper dataset matrix is incomplete",
        issues,
    )
    check(
        set(config["backbones"]) == {"llada8b", "llada15", "llada20mini", "dream"},
        "paper backbone matrix is incomplete",
        issues,
    )
    check(
        set(config["methods"]) == {"denmark", "dlm_kgw", "dgmark", "patternmark", "umr"},
        "paper method matrix is incomplete",
        issues,
    )
    check(config["metrics"]["tpr_fprs"] == [0.005, 0.01, 0.05], "wrong paper FPR targets", issues)

    ablations = config["ablations"]
    check(ablations["candidate_temperature"] == [0.3, 0.5, 0.6, 0.75, 0.9], "candidate-temperature axis is incomplete", issues)
    check(ablations["rollout_count"] == [1, 2, 3, 5, 8], "rollout axis is incomplete", issues)
    check(ablations["scan_ranges"] == [[25, 25], [23, 27], [21, 29], [17, 33], [12, 37], [10, 40]], "scan-range axis is incomplete", issues)
    check(ablations["maximum_generation_length"] == [100, 150, 200, 250, 300], "maximum-length axis is incomplete", issues)
    check(ablations["candidate_positions_per_step"] == [1, 2, 4, 8], "multi-position axis is incomplete", issues)

    expected_coverage = {
        "main_semantic_robustness",
        "open_source_semantic_attacks",
        "token_level_robustness",
        "quality",
        "sensitivity",
        "scan_range",
        "maximum_generation_length",
        "candidate_temperature_position",
        "multi_position",
        "encoder_ablation",
        "backtranslation",
        "document_length_attacks",
        "nonuniform_local_length",
        "semi_ar",
        "tied_candidate_channel",
        "downstream",
        "runtime",
        "trajectory",
        "qualitative",
    }
    check(set(config["paper_experiment_coverage"]) == expected_coverage, "paper experiment coverage index is incomplete", issues)

    for relative in sorted(set(config_paths(config))):
        check((root / relative).exists(), f"configured reproduction asset is missing: {relative}", issues)

    data_root = root / "data"
    check(not data_root.exists(), "release must not contain bundled datasets", issues)
    check(
        not (root / "RELEASE_MANIFEST.json").exists(),
        "generated release manifest must not be committed",
        issues,
    )
    construction = config["data_construction"]
    check(construction["rows_per_dataset"] == 500, "wrong prompt construction target", issues)
    check(
        construction["script"] == "denmark/data/prepare_waterbench.py",
        "prompt construction entry point changed",
        issues,
    )

    encoder = config["semantic_encoder_training"]
    check(encoder["checkpoint_name"] == "DenMark-Encoder", "wrong encoder name", issues)
    check(encoder["source_rows"] == 8000, "wrong encoder source-row target", issues)
    check(encoder["reported_valid_pairs"] == 7993, "wrong reported pair count", issues)

    defaults = (
        ("denmark/core/generate.py", "--num_message_bits", 2),
        ("denmark/core/generate.py", "--channels_per_step", 2),
        ("denmark/core/generate.py", "--rollouts_per_cand", 3),
        ("denmark/core/generate.py", "--rollout_schedule", "linear_decay"),
        ("denmark/core/generate.py", "--position_selection", "random"),
        ("denmark/evaluation/detect.py", "--num_message_bits", 2),
        ("denmark/baselines/block_best_of_k/generate.py", "--num_message_bits", 2),
        ("denmark/baselines/pmark/generate.py", "--num_channels", 2),
        ("denmark/baselines/semstamp/generate.py", "--lsh_dim", 2),
        ("denmark/baselines/semantic_detect.py", "--num_channels", 2),
        ("denmark/baselines/semantic_detect.py", "--lsh_dim", 2),
    )
    for relative, flag, expected in defaults:
        try:
            actual = literal_default(root / relative, flag)
            check(actual == expected, f"{relative} {flag} default is {actual!r}, expected {expected!r}", issues)
        except (OSError, SyntaxError, ValueError) as error:
            issues.append(str(error))

    selected = selected_files(root)
    for source in selected:
        issues.extend(audit_file(source, source.relative_to(root)))
    issues.extend(local_import_issues(root, selected))

    report = {
        "status": "pass" if not issues else "fail",
        "paper_experiment_groups": len(config["paper_experiment_coverage"]),
        "release_files": len(selected),
        "bundled_datasets": 0,
        "encoder_checkpoint": encoder["checkpoint_name"],
        "issues": issues,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
