#!/usr/bin/env python3
"""Validate and merge sharded LLaDA block-reject generations."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--expected_per_dataset", type=int, default=50)
    parser.add_argument("--expected_num_shards", type=int, default=1)
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument(
        "--mask_id",
        type=int,
        default=None,
        help="Expected MASK id; defaults to each shard's recorded run configuration.",
    )
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_rep4", type=float, default=0.2)
    return parser.parse_args()


def load_rows(paths: list[Path]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                    continue
                row["_audit_source_file"] = str(path)
                row["_audit_source_line"] = line_number
                rows.append(row)
    return rows, errors


def rep_ngram(text: str, n: int = 4) -> float:
    words = (text or "").split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[idx : idx + n]) for idx in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))


def audit_run_configs(
    paths: list[Path],
    dataset: str,
    expected_num_shards: int,
) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    configs: list[dict] = []
    for path in paths:
        try:
            configs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid run config: {exc}")
    if len(configs) != expected_num_shards:
        errors.append(f"run_config_count={len(configs)} expected={expected_num_shards}")
    if not configs:
        return None, errors

    shard_ids = [config.get("shard_idx") for config in configs]
    if set(shard_ids) != set(range(expected_num_shards)):
        errors.append(f"config_shards={sorted(shard_ids)} expected=0..{expected_num_shards - 1}")

    invariant_keys = [
        "method",
        "alias",
        "selection_rule",
        "empty_block_policy",
        "dataset",
        "dataset_file",
        "dataset_sha256",
        "selected_source_indices",
        "n_per_dataset",
        "seed",
        "num_shards",
        "model",
        "encoder",
        "generator_family",
        "mask_id",
        "max_new_tokens",
        "block_size",
        "steps",
        "temperature",
        "num_candidates",
        "candidate_batch_size",
        "remasking",
        "min_response_tokens",
        "max_rep4",
        "max_retries",
        "num_message_bits",
        "direction_seed",
        "message_seed",
    ]
    canonical = configs[0]
    for config_index, config in enumerate(configs):
        for key in invariant_keys:
            if config.get(key) != canonical.get(key):
                errors.append(
                    f"config[{config_index}] key={key}: {config.get(key)!r} "
                    f"!= canonical {canonical.get(key)!r}"
                )
        if config.get("dataset") != dataset:
            errors.append(f"config[{config_index}] dataset={config.get('dataset')} expected={dataset}")
        if config.get("num_shards") != expected_num_shards:
            errors.append(
                f"config[{config_index}] num_shards={config.get('num_shards')} "
                f"expected={expected_num_shards}"
            )
    return canonical, errors


def audit_dataset(
    dataset: str,
    rows: list[dict],
    *,
    expected: int,
    gen_length: int,
    block_size: int,
    num_candidates: int,
    mask_id: int,
    min_response_tokens: int,
    max_rep4: float,
    expected_source_indices: list[int] | None = None,
    expected_num_shards: int | None = None,
    canonical_config: dict | None = None,
) -> dict:
    structural_errors: list[str] = []
    quality_bad: list[dict] = []
    expected_blocks = gen_length // block_size

    if len(rows) != expected:
        structural_errors.append(f"row_count={len(rows)} expected={expected}")
    source_indices = [row.get("source_idx") for row in rows]
    source_counts = Counter(source_indices)
    duplicates = sorted(
        (idx for idx, count in source_counts.items() if count > 1),
        key=lambda value: (value is None, str(value)),
    )
    if duplicates:
        structural_errors.append(f"duplicate_source_indices={duplicates}")
    if len(set(source_indices)) != len(source_indices):
        structural_errors.append("source indices are not unique")
    if expected_source_indices is not None and source_indices != expected_source_indices:
        structural_errors.append(
            f"source_indices={source_indices} expected={expected_source_indices}"
        )
    selected_positions = [row.get("selected_position") for row in rows]
    if set(selected_positions) != set(range(expected)):
        structural_errors.append(
            f"selected_positions={sorted(selected_positions, key=lambda value: (value is None, str(value)))} "
            f"expected=0..{expected - 1}"
        )
    if expected_num_shards is not None:
        shard_file_counts = Counter(
            Path(str(row.get("_audit_source_file"))).name
            for row in rows
            if row.get("_audit_source_file")
        )
        for shard_id in range(expected_num_shards):
            filename = f"generations_shard{shard_id}.jsonl"
            expected_in_shard = sum(
                1 for selected_position in range(expected)
                if selected_position % expected_num_shards == shard_id
            )
            actual_in_shard = shard_file_counts.get(filename, 0)
            if actual_in_shard != expected_in_shard:
                structural_errors.append(
                    f"{filename}: rows={actual_in_shard} expected={expected_in_shard}"
                )

    for row_index, row in enumerate(rows):
        label = f"row={row_index} source_idx={row.get('source_idx')}"
        selected_position = row.get("selected_position")
        shard = row.get("shard")
        num_shards = row.get("num_shards")
        if not isinstance(selected_position, int) or not isinstance(shard, int) or not isinstance(num_shards, int):
            structural_errors.append(f"{label}: invalid shard metadata")
        elif num_shards <= 0 or selected_position % num_shards != shard:
            structural_errors.append(
                f"{label}: shard={shard}/{num_shards} disagrees with selected_position={selected_position}"
            )
        if expected_num_shards is not None and num_shards != expected_num_shards:
            structural_errors.append(
                f"{label}: num_shards={num_shards} expected={expected_num_shards}"
            )
        source_file = row.get("_audit_source_file")
        if source_file:
            source_stem = Path(str(source_file)).stem
            try:
                source_file_shard = int(source_stem.rsplit("shard", 1)[1])
            except (IndexError, ValueError):
                structural_errors.append(
                    f"{label}: unrecognized shard filename={Path(str(source_file)).name}"
                )
            else:
                if shard != source_file_shard:
                    structural_errors.append(
                        f"{label}: row shard={shard} but file shard={source_file_shard}"
                    )
        if row.get("ds_key") != dataset:
            structural_errors.append(f"{label}: ds_key={row.get('ds_key')} expected={dataset}")
        if canonical_config is not None:
            row_config = row.get("gen_config") or {}
            for key in (
                "method",
                "selection_rule",
                "empty_block_policy",
                "dataset_sha256",
                "seed",
                "model",
                "encoder",
                "max_new_tokens",
                "block_size",
                "steps",
                "temperature",
                "num_candidates",
                "candidate_batch_size",
                "remasking",
                "num_message_bits",
                "direction_seed",
                "message_seed",
            ):
                if row_config.get(key) != canonical_config.get(key):
                    structural_errors.append(
                        f"{label}: gen_config[{key}]={row_config.get(key)!r} "
                        f"expected={canonical_config.get(key)!r}"
                    )
        token_ids = row.get("generated_token_ids") or []
        if len(token_ids) != gen_length:
            structural_errors.append(f"{label}: raw_token_len={len(token_ids)} expected={gen_length}")
        if mask_id in token_ids:
            structural_errors.append(f"{label}: output contains MASK id {mask_id}")

        diagnostics = row.get("block_diagnostics") or []
        if len(diagnostics) != expected_blocks:
            structural_errors.append(
                f"{label}: block_diagnostics={len(diagnostics)} expected={expected_blocks}"
            )
        selected_token_ids: list[int] = []
        for block_id, diagnostic in enumerate(diagnostics):
            if diagnostic.get("selection_rule") != "argmax_semantic_watermark_score":
                structural_errors.append(
                    f"{label} block={block_id}: selection_rule={diagnostic.get('selection_rule')!r}"
                )
            if diagnostic.get("empty_block_policy") != "first_candidate_when_all_semantically_empty":
                structural_errors.append(
                    f"{label} block={block_id}: empty_block_policy="
                    f"{diagnostic.get('empty_block_policy')!r}"
                )
            if diagnostic.get("num_candidates") != num_candidates:
                structural_errors.append(
                    f"{label} block={block_id}: num_candidates={diagnostic.get('num_candidates')} "
                    f"expected={num_candidates}"
                )
            if diagnostic.get("block_id") != block_id:
                structural_errors.append(
                    f"{label} block={block_id}: recorded block_id={diagnostic.get('block_id')}"
                )
            if diagnostic.get("block_start") != block_id * block_size:
                structural_errors.append(
                    f"{label} block={block_id}: block_start={diagnostic.get('block_start')} "
                    f"expected={block_id * block_size}"
                )
            if diagnostic.get("block_end") != (block_id + 1) * block_size:
                structural_errors.append(
                    f"{label} block={block_id}: block_end={diagnostic.get('block_end')} "
                    f"expected={(block_id + 1) * block_size}"
                )
            selected_block = diagnostic.get("selected_token_ids") or []
            if len(selected_block) != block_size:
                structural_errors.append(
                    f"{label} block={block_id}: selected_token_ids={len(selected_block)} "
                    f"expected={block_size}"
                )
            selected_token_ids.extend(selected_block)
            scores = diagnostic.get("candidate_scores") or []
            if len(scores) != num_candidates:
                structural_errors.append(
                    f"{label} block={block_id}: candidate_scores={len(scores)} "
                    f"expected={num_candidates}"
                )
                continue
            valid_mask = diagnostic.get("candidate_valid")
            if (
                not isinstance(valid_mask, list)
                or len(valid_mask) != num_candidates
                or not all(isinstance(value, bool) for value in valid_mask)
            ):
                structural_errors.append(
                    f"{label} block={block_id}: invalid candidate_valid mask"
                )
                valid_mask = [score is not None for score in scores]

            finite: list[tuple[int, float]] = []
            finite_mask: list[bool] = []
            for candidate_idx, score in enumerate(scores):
                is_finite = False
                if score is not None:
                    try:
                        numeric_score = float(score)
                    except (TypeError, ValueError):
                        structural_errors.append(
                            f"{label} block={block_id} candidate={candidate_idx}: "
                            f"invalid score={score!r}"
                        )
                    else:
                        is_finite = math.isfinite(numeric_score)
                        if is_finite:
                            finite.append((candidate_idx, numeric_score))
                        else:
                            structural_errors.append(
                                f"{label} block={block_id} candidate={candidate_idx}: "
                                f"non-finite score={score!r}"
                            )
                finite_mask.append(is_finite)
            if valid_mask != finite_mask:
                structural_errors.append(
                    f"{label} block={block_id}: candidate_valid disagrees with finite scores"
                )

            actual_idx = diagnostic.get("selected_candidate_index")
            actual_score = diagnostic.get("selected_score")
            selection_reason = diagnostic.get("selection_reason")
            if not finite:
                if any(valid_mask):
                    structural_errors.append(
                        f"{label} block={block_id}: valid candidates but no finite scores"
                    )
                if actual_idx != 0:
                    structural_errors.append(
                        f"{label} block={block_id}: all candidates empty but selected={actual_idx}"
                    )
                if actual_score is not None:
                    structural_errors.append(
                        f"{label} block={block_id}: all candidates empty but "
                        f"selected_score={actual_score}"
                    )
                if selection_reason != "all_candidates_empty_fallback_first":
                    structural_errors.append(
                        f"{label} block={block_id}: empty selection_reason={selection_reason!r}"
                    )
                continue
            expected_idx, expected_score = max(finite, key=lambda item: item[1])
            if selection_reason != "argmax_semantic_watermark_score":
                structural_errors.append(
                    f"{label} block={block_id}: selection_reason={selection_reason!r}"
                )
            if actual_idx != expected_idx:
                structural_errors.append(
                    f"{label} block={block_id}: selected={actual_idx} argmax={expected_idx}"
                )
            if actual_score is None or not math.isclose(
                float(actual_score), expected_score, rel_tol=1e-6, abs_tol=1e-7
            ):
                structural_errors.append(
                    f"{label} block={block_id}: selected_score={actual_score} max={expected_score}"
                )

        if selected_token_ids != token_ids:
            structural_errors.append(
                f"{label}: concatenated selected blocks do not equal generated_token_ids"
            )

        retokenized_ids = row.get("retokenized_token_ids") or []
        token_len = len(retokenized_ids)
        recorded_token_len = row.get("token_len")
        if recorded_token_len != token_len:
            structural_errors.append(
                f"{label}: recorded token_len={recorded_token_len} retokenized_ids={token_len}"
            )
        rep4 = rep_ngram(row.get("text", ""), 4)
        recorded_rep4 = row.get("rep4")
        if recorded_rep4 is None or not math.isclose(
            float(recorded_rep4), rep4, rel_tol=1e-9, abs_tol=1e-12
        ):
            structural_errors.append(
                f"{label}: recorded rep4={recorded_rep4} recomputed={rep4}"
            )
        passed_quality = token_len >= min_response_tokens and rep4 <= max_rep4
        if bool(row.get("passed_quality")) != passed_quality:
            structural_errors.append(
                f"{label}: recorded passed_quality={row.get('passed_quality')} "
                f"recomputed={passed_quality}"
            )
        if not passed_quality:
            quality_bad.append(
                {
                    "source_idx": row.get("source_idx"),
                    "token_len": token_len,
                    "rep4": rep4,
                    "retry_attempts": row.get("retry_attempts"),
                }
            )

    return {
        "dataset": dataset,
        "n_rows": len(rows),
        "n_unique_source_indices": len(set(source_indices)),
        "source_indices": sorted(source_indices, key=lambda value: (value is None, str(value))),
        "n_structural_errors": len(structural_errors),
        "structural_errors": structural_errors,
        "n_quality_bad": len(quality_bad),
        "quality_bad": quality_bad,
        "valid": not structural_errors and not quality_bad and len(rows) == expected,
    }


def main() -> None:
    args = parse_args()
    if args.gen_length % args.block_size != 0:
        raise ValueError("block_size must divide gen_length")
    final_dir = args.run_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}
    all_valid = True

    for dataset in args.datasets:
        shard_dir = args.run_root / "raw" / dataset
        shard_paths = sorted(shard_dir.glob("generations_shard*.jsonl"))
        config_paths = sorted(shard_dir.glob("run_config_shard*.json"))
        expected_generation_names = {
            f"generations_shard{shard_id}.jsonl"
            for shard_id in range(args.expected_num_shards)
        }
        expected_config_names = {
            f"run_config_shard{shard_id}.json"
            for shard_id in range(args.expected_num_shards)
        }
        shard_file_errors: list[str] = []
        actual_generation_names = {path.name for path in shard_paths}
        actual_config_names = {path.name for path in config_paths}
        if actual_generation_names != expected_generation_names:
            shard_file_errors.append(
                f"generation_shard_files={sorted(actual_generation_names)} "
                f"expected={sorted(expected_generation_names)}"
            )
        if actual_config_names != expected_config_names:
            shard_file_errors.append(
                f"run_config_shard_files={sorted(actual_config_names)} "
                f"expected={sorted(expected_config_names)}"
            )
        canonical_config, config_errors = audit_run_configs(
            config_paths,
            dataset,
            args.expected_num_shards,
        )
        rows, load_errors = load_rows(shard_paths)
        rows.sort(key=lambda row: (int(row.get("selected_position", -1)), int(row.get("source_idx", -1))))
        summary = audit_dataset(
            dataset,
            rows,
            expected=args.expected_per_dataset,
            gen_length=args.gen_length,
            block_size=args.block_size,
            num_candidates=args.num_candidates,
            mask_id=(
                args.mask_id
                if args.mask_id is not None
                else int(canonical_config["mask_id"])
                if canonical_config and canonical_config.get("mask_id") is not None
                else -1
            ),
            min_response_tokens=args.min_response_tokens,
            max_rep4=args.max_rep4,
            expected_source_indices=(
                canonical_config.get("selected_source_indices") if canonical_config else None
            ),
            expected_num_shards=args.expected_num_shards,
            canonical_config=canonical_config,
        )
        summary["shard_files"] = [str(path) for path in shard_paths]
        summary["run_config_files"] = [str(path) for path in config_paths]
        if shard_file_errors or config_errors or load_errors:
            summary["structural_errors"] = (
                shard_file_errors + config_errors + load_errors + summary["structural_errors"]
            )
            summary["n_structural_errors"] = len(summary["structural_errors"])
            summary["valid"] = False
        summaries[dataset] = summary
        all_valid = all_valid and summary["valid"]

        if summary["valid"]:
            merged_path = final_dir / f"{dataset}.jsonl"
            with merged_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    row.pop("_audit_source_file", None)
                    row.pop("_audit_source_line", None)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary["merged_jsonl"] = str(merged_path)

        print(
            f"{dataset}: rows={summary['n_rows']} unique={summary['n_unique_source_indices']} "
            f"structural_errors={summary['n_structural_errors']} "
            f"quality_bad={summary['n_quality_bad']} valid={summary['valid']}"
        )
        for error in summary["structural_errors"]:
            print(f"  ERROR {error}")
        for bad in summary["quality_bad"]:
            print(
                f"  BAD source_idx={bad['source_idx']} token_len={bad['token_len']} "
                f"rep4={bad['rep4']:.6f} retries={bad['retry_attempts']}"
            )

    summary_path = final_dir / "audit_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "valid": all_valid,
                "expected_per_dataset": args.expected_per_dataset,
                "expected_num_shards": args.expected_num_shards,
                "gen_length": args.gen_length,
                "block_size": args.block_size,
                "num_candidates": args.num_candidates,
                "datasets": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"audit_summary={summary_path}")
    if not all_valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
