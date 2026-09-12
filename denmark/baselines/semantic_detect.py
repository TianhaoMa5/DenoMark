#!/usr/bin/env python3
"""Evaluate PMark and SemStamp with an empirically calibrated block-size scan."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from denmark.core.calibration import EPS, empirical_right_tail_p
from denmark.core.scoring import block_decode
from denmark.data.protocol import (
    canonical_prompt_key,
    grouped_crossfit_folds,
    quality_and_dedup_indices,
    repetition_ngram,
    resolve_prompt_seed,
)
from denmark.baselines.pmark.model import (
    build_pmark_pivots,
    build_pmark_secret_bits,
    pmark_cosine_projections,
    pmark_soft_matches,
)
from denmark.baselines.semstamp.model import (
    SEMSTAMP_HASH_KEY,
    build_lsh_hyperplanes,
    embedding_hashes_and_margins,
    valid_bins_from_previous_hash,
)
from denmark.core.model import encode_texts


import math
from denmark.evaluation.metrics import roc_interpolated_tpr

TEXT_FIELDS = (
    "attacked_text",
    "attack_text",
    "text",
    "generation_text_skip_special",
    "generation",
    "completion",
    "output",
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def row_text(row: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.replace("NEWLINE_CHAR", " ").split())
    return ""


def encoder_embedding_dim(encoder) -> int:
    for key in ("hidden_size", "d_model", "dim"):
        value = getattr(encoder.config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("cannot infer encoder embedding dimension")


def auc_rank(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return float("nan")
    scores = np.r_[negative, positive]
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and scores[order[end]] == scores[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    n_positive = len(positive)
    n_negative = len(negative)
    numerator = ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2
    return float(numerator / (n_positive * n_negative))


def empirical_threshold(
    positive: list[float],
    negative: list[float],
    target_fpr: float,
) -> dict[str, float]:
    ordered = np.sort(np.asarray(negative, dtype=float))
    threshold_index = int(math.ceil((1.0 - target_fpr) * len(ordered))) - 1
    threshold_index = max(0, min(threshold_index, len(ordered) - 1))
    threshold = float(ordered[threshold_index])
    positive_array = np.asarray(positive, dtype=float)
    negative_array = np.asarray(negative, dtype=float)
    return {
        "threshold": threshold,
        "tpr_strict_gt": float((positive_array > threshold).mean()),
        "observed_fpr_strict_gt": float((negative_array > threshold).mean()),
    }


def summarize(positive: list[float], negative: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n_pos": len(positive),
        "n_neg": len(negative),
        "pos_mean": float(np.mean(positive)),
        "pos_median": float(np.median(positive)),
        "neg_mean": float(np.mean(negative)),
        "neg_median": float(np.median(negative)),
        "auc": auc_rank(positive, negative),
    }
    for label, target in (("0_1pct", 0.001), ("1pct", 0.01), ("5pct", 0.05)):
        result[label] = empirical_threshold(positive, negative, target)
    result.update(
        {
            "roc_tpr_at_0_5pct": roc_interpolated_tpr(positive, negative, 0.005),
            "roc_tpr_at_1pct": roc_interpolated_tpr(positive, negative, 0.01),
            "roc_tpr_at_5pct": roc_interpolated_tpr(positive, negative, 0.05),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pmark_positive_jsonl", type=Path, required=True)
    parser.add_argument("--semstamp_positive_jsonl", type=Path)
    parser.add_argument("--negative_jsonl", type=Path, required=True)
    parser.add_argument("--waterbench_jsonl", type=Path, required=True)
    parser.add_argument(
        "--additional_waterbench_jsonl",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--scan_min", type=int, default=12)
    parser.add_argument("--scan_max", type=int, default=37)
    parser.add_argument("--encoder_batch_size", type=int, default=128)
    parser.add_argument("--min_token_len", type=int, default=150)
    parser.add_argument("--max_rep4", type=float, default=0.2)
    parser.add_argument(
        "--deduplicate_negative_prompts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--calibration_folds", type=int, default=5)
    parser.add_argument("--calibration_seed", type=int, default=42)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--pivot_seed", type=int, default=42)
    parser.add_argument("--secret_seed", type=int, default=0)
    parser.add_argument("--detect_tolerance", type=float, default=0.001)
    parser.add_argument("--detect_decay", type=float, default=250.0)
    parser.add_argument("--lsh_dim", type=int, default=2)
    parser.add_argument("--lsh_seed", type=int, default=1234)
    parser.add_argument("--accept_rate", type=float, default=0.25)
    parser.add_argument("--hash_key", type=int, default=SEMSTAMP_HASH_KEY)
    parser.add_argument(
        "--mask_rng_device",
        choices=("cpu", "cuda"),
        default=None,
        help="Defaults to --device; use cpu only to audit legacy CPU-mask runs.",
    )
    return parser.parse_args()


def prepare_items(
    rows: list[dict[str, Any]],
    tokenizer,
    source_rows: list[dict[str, Any]] | None,
    require_prompt: bool,
) -> list[dict[str, Any]]:
    items = []
    for row_index, row in enumerate(rows):
        text = row_text(row)
        if not text:
            raise ValueError(f"empty text at row {row_index}")
        source_idx = int(row.get("source_idx", row_index))
        prompt_seed = None
        prompt_seed_source = None
        prompt_key = None
        if require_prompt:
            prompt_seed, prompt_seed_source = resolve_prompt_seed(
                row,
                source_rows=source_rows,
                row_index=row_index,
            )
            if not str(prompt_seed).strip():
                raise ValueError(f"empty prompt seed at row {row_index}")
            prompt_key = canonical_prompt_key(str(prompt_seed), source_rows)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        items.append(
            {
                "source_idx": source_idx,
                "sample_id": row.get("sample_id"),
                "token_ids": [int(value) for value in token_ids],
                "token_len": len(token_ids),
                "rep4": repetition_ngram(text, 4),
                "prompt_seed": str(prompt_seed).strip() if prompt_seed is not None else None,
                "prompt_seed_source": prompt_seed_source,
                "prompt_key": prompt_key,
            }
        )
    return items


def split_blocks(items, tokenizer, block_size: int, gen_length: int):
    flat_texts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for item in items:
        token_ids = item["token_ids"][:gen_length]
        block_ids = []
        start_offset = len(flat_texts)
        for block_id, start in enumerate(range(0, len(token_ids), block_size)):
            block_text = block_decode(
                token_ids,
                start,
                min(start + block_size, len(token_ids)),
                tokenizer,
            )
            if block_text.strip():
                block_ids.append(block_id)
                flat_texts.append(block_text)
        metadata.append(
            {
                "block_ids": block_ids,
                "start": start_offset,
                "end": len(flat_texts),
            }
        )
    return flat_texts, metadata


def pmark_raw_score(
    projections: torch.Tensor,
    block_ids: list[int],
    secret_bits: torch.Tensor,
    tolerance: float,
    decay: float,
) -> float:
    soft_hits = 0.0
    n_trials = 0
    for projection, block_id in zip(projections, block_ids):
        soft = pmark_soft_matches(
            projection,
            secret_bits[block_id],
            tolerance=tolerance,
            decay=decay,
        )
        soft_hits += float(soft.sum().item())
        n_trials += int(soft.numel())
    denominator = math.sqrt(0.25 * n_trials) if n_trials else 0.0
    return (soft_hits - 0.5 * n_trials) / denominator if denominator else 0.0


def semstamp_raw_score(
    block_hashes: list[int],
    prompt_hash: int,
    lsh_dim: int,
    accept_rate: float,
    hash_key: int,
    mask_rng_device: str | torch.device,
) -> float:
    hashes = [int(prompt_hash), *[int(value) for value in block_hashes]]
    hits = 0
    transitions = 0
    for previous_hash, current_hash in zip(hashes[:-1], hashes[1:]):
        valid = valid_bins_from_previous_hash(
            previous_hash,
            lsh_dim,
            accept_rate,
            hash_key,
            rng_device=mask_rng_device,
        )
        hits += int(current_hash in set(valid))
        transitions += 1
    denominator = math.sqrt(transitions * accept_rate * (1.0 - accept_rate)) if transitions else 0.0
    return (hits - accept_rate * transitions) / denominator if denominator else 0.0


def calibrated_scan(
    block_sizes: list[int],
    negative_raw_by_size: dict[int, list[float]],
    positive_raw_by_size: dict[int, list[float]],
    negative_fold_ids: list[int],
) -> dict[str, Any]:
    n_negative = len(next(iter(negative_raw_by_size.values())))
    n_positive = len(next(iter(positive_raw_by_size.values())))

    if len(negative_fold_ids) != n_negative:
        raise ValueError("negative fold assignments do not align with scores")
    all_negative_indices = list(range(n_negative))

    def score_one(
        raw_by_size: dict[int, list[float]],
        index: int,
        calibration_indices: list[int],
    ) -> dict[str, Any]:
        if not calibration_indices:
            raise ValueError("empty empirical calibration fold")
        best_block_size = -1
        best_raw = 0.0
        best_p = 1.0
        per_size_raw = {}
        per_size_p = {}
        for block_size in block_sizes:
            raw = float(raw_by_size[block_size][index])
            calibration = [
                negative_raw_by_size[block_size][negative_index]
                for negative_index in calibration_indices
            ]
            p_value = empirical_right_tail_p(raw, calibration)
            per_size_raw[str(block_size)] = raw
            per_size_p[str(block_size)] = p_value
            if best_block_size < 0 or p_value < best_p:
                best_block_size = block_size
                best_raw = raw
                best_p = p_value
        p_scan = min(1.0, len(block_sizes) * best_p)
        return {
            "score": float(-math.log(p_scan + EPS)),
            "p_value": float(p_scan),
            "best_single_size_p": float(best_p),
            "best_block_size": int(best_block_size),
            "best_raw_score": float(best_raw),
            "raw_scores_by_block_size": per_size_raw,
            "p_values_by_block_size": per_size_p,
            "n_calibration": len(calibration_indices),
        }

    negative_records = []
    for index, fold_id in enumerate(negative_fold_ids):
        calibration_indices = [
            negative_index
            for negative_index in all_negative_indices
            if negative_fold_ids[negative_index] != fold_id
        ]
        record = score_one(negative_raw_by_size, index, calibration_indices)
        record["calibration_fold"] = int(fold_id)
        negative_records.append(record)
    positive_records = [
        score_one(positive_raw_by_size, index, all_negative_indices)
        for index in range(n_positive)
    ]
    negative_scores = [record["score"] for record in negative_records]
    positive_scores = [record["score"] for record in positive_records]
    return {
        "protocol": (
            "grouped cross-fit negative calibration per block size; positives calibrated "
            "on all filtered unique clean negatives; minimum p-value over scan; "
            "Bonferroni correction; score=-log(p); strict score > threshold"
        ),
        "summary": summarize(positive_scores, negative_scores),
        "positive_records": positive_records,
        "negative_records": negative_records,
        "negative_scores": negative_scores,
        "negative_fold_ids": negative_fold_ids,
        "negative_raw_scores_by_block_size": {
            str(block_size): values
            for block_size, values in negative_raw_by_size.items()
        },
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.scan_min <= 0 or args.scan_min > args.scan_max:
        raise ValueError("invalid scan range")
    if args.gen_length <= 0 or args.encoder_batch_size <= 0:
        raise ValueError("length and batch size must be positive")
    if args.min_token_len < 0 or args.max_rep4 < 0:
        raise ValueError("quality thresholds must be non-negative")
    if args.calibration_folds < 2:
        raise ValueError("calibration folds must be at least two")
    block_sizes = list(range(args.scan_min, args.scan_max + 1))
    if 25 not in block_sizes:
        raise ValueError("scan range must include fixed block size 25")

    source_rows = read_jsonl(args.waterbench_jsonl)
    for source_path in args.additional_waterbench_jsonl:
        source_rows.extend(read_jsonl(source_path))
    negative_rows = read_jsonl(args.negative_jsonl)
    pmark_rows = read_jsonl(args.pmark_positive_jsonl)
    use_semstamp = args.semstamp_positive_jsonl is not None
    semstamp_rows = read_jsonl(args.semstamp_positive_jsonl) if use_semstamp else []
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(
        args.encoder,
        torch_dtype=torch.float32,
    ).to(args.device).eval()

    negative_items = prepare_items(negative_rows, tokenizer, source_rows, True)
    pmark_items = prepare_items(pmark_rows, tokenizer, None, False)
    semstamp_items = prepare_items(semstamp_rows, tokenizer, source_rows, True) if use_semstamp else []

    negative_indices, negative_filter = quality_and_dedup_indices(
        negative_items,
        min_token_len=args.min_token_len,
        max_rep4=args.max_rep4,
        deduplicate_prompt=args.deduplicate_negative_prompts,
    )
    pmark_indices, pmark_filter = quality_and_dedup_indices(
        pmark_items,
        min_token_len=args.min_token_len,
        max_rep4=args.max_rep4,
        deduplicate_prompt=False,
    )
    if use_semstamp:
        semstamp_indices, semstamp_filter = quality_and_dedup_indices(
            semstamp_items,
            min_token_len=args.min_token_len,
            max_rep4=args.max_rep4,
            deduplicate_prompt=False,
        )
    else:
        semstamp_indices, semstamp_filter = [], {"disabled": True}
    negative_items = [negative_items[index] for index in negative_indices]
    pmark_items = [pmark_items[index] for index in pmark_indices]
    semstamp_items = [semstamp_items[index] for index in semstamp_indices]
    if not negative_items or not pmark_items or (use_semstamp and not semstamp_items):
        raise RuntimeError("quality filtering produced an empty scan input")
    negative_fold_ids = grouped_crossfit_folds(
        [str(item["prompt_key"]) for item in negative_items],
        n_folds=args.calibration_folds,
        seed=args.calibration_seed,
    )
    all_items = [*negative_items, *pmark_items, *semstamp_items]
    n_negative = len(negative_items)
    pmark_start = n_negative
    semstamp_start = pmark_start + len(pmark_items)

    embedding_dim = encoder_embedding_dim(encoder)
    pivots = build_pmark_pivots(embedding_dim, args.num_channels, args.pivot_seed)
    secret_bits = build_pmark_secret_bits(
        math.ceil(args.gen_length / args.scan_min),
        args.num_channels,
        args.secret_seed,
    )
    if use_semstamp:
        prompt_item_indices = [*range(n_negative), *range(semstamp_start, len(all_items))]
        prompt_embeddings = encode_texts(
            [all_items[index]["prompt_seed"] for index in prompt_item_indices],
            encoder,
            encoder_tokenizer,
            args.device,
            batch_sz=args.encoder_batch_size,
            to_cpu=True,
        ).float()
        hyperplanes = build_lsh_hyperplanes(args.lsh_dim, embedding_dim, args.lsh_seed)
        prompt_hashes, _, _ = embedding_hashes_and_margins(prompt_embeddings, hyperplanes)
        prompt_hash_by_item = {
            item_index: int(prompt_hashes[position].item())
            for position, item_index in enumerate(prompt_item_indices)
        }
    else:
        hyperplanes = None
        prompt_hash_by_item = {}

    pmark_negative_raw: dict[int, list[float]] = {}
    pmark_positive_raw: dict[int, list[float]] = {}
    semstamp_negative_raw: dict[int, list[float]] = {}
    semstamp_positive_raw: dict[int, list[float]] = {}

    for block_size in block_sizes:
        flat_texts, metadata = split_blocks(
            all_items,
            tokenizer,
            block_size,
            args.gen_length,
        )
        embeddings = encode_texts(
            flat_texts,
            encoder,
            encoder_tokenizer,
            args.device,
            batch_sz=args.encoder_batch_size,
            to_cpu=True,
        ).float()
        projections = pmark_cosine_projections(embeddings, pivots)
        block_hashes = None
        if use_semstamp:
            block_hashes, _, _ = embedding_hashes_and_margins(embeddings, hyperplanes)

        def pmark_score(item_index: int) -> float:
            meta = metadata[item_index]
            return pmark_raw_score(
                projections[meta["start"]:meta["end"]],
                meta["block_ids"],
                secret_bits,
                args.detect_tolerance,
                args.detect_decay,
            )

        def semstamp_score(item_index: int) -> float:
            meta = metadata[item_index]
            return semstamp_raw_score(
                block_hashes[meta["start"]:meta["end"]].tolist(),
                prompt_hash_by_item[item_index],
                args.lsh_dim,
                args.accept_rate,
                args.hash_key,
                args.mask_rng_device or args.device,
            )

        pmark_negative_raw[block_size] = [pmark_score(index) for index in range(n_negative)]
        pmark_positive_raw[block_size] = [
            pmark_score(index)
            for index in range(pmark_start, semstamp_start)
        ]
        if use_semstamp:
            semstamp_negative_raw[block_size] = [semstamp_score(index) for index in range(n_negative)]
            semstamp_positive_raw[block_size] = [
                semstamp_score(index)
                for index in range(semstamp_start, len(all_items))
            ]
        print(f"scored block_size={block_size}", flush=True)

    pmark_fixed = summarize(pmark_positive_raw[25], pmark_negative_raw[25])
    result = {
        "protocol": "method-native PMark/SemStamp block-size scan with matched-clean empirical calibration",
        "negative_source": str(args.negative_jsonl),
        "pmark_positive_source": str(args.pmark_positive_jsonl),
        "semstamp_positive_source": str(args.semstamp_positive_jsonl) if use_semstamp else None,
        "encoder": args.encoder,
        "model_tokenizer": args.model,
        "n_negative": n_negative,
        "n_pmark_positive": len(pmark_items),
        "n_semstamp_positive": len(semstamp_items) if use_semstamp else 0,
        "filters": {
            "negative": negative_filter,
            "pmark_positive": pmark_filter,
            "semstamp_positive": semstamp_filter,
        },
        "min_token_len": args.min_token_len,
        "max_rep4": args.max_rep4,
        "deduplicate_negative_prompts": args.deduplicate_negative_prompts,
        "calibration_folds": args.calibration_folds,
        "calibration_seed": args.calibration_seed,
        "mask_rng_device": args.mask_rng_device or args.device,
        "gen_length": args.gen_length,
        "scan_block_sizes": block_sizes,
        "methods": {
            "pmark": {
                "fixed25": pmark_fixed,
                "calibrated_scan": calibrated_scan(
                    block_sizes,
                    pmark_negative_raw,
                    pmark_positive_raw,
                    negative_fold_ids,
                ),
            },
        },
    }
    if use_semstamp:
        result["methods"]["semstamp"] = {
            "fixed25": summarize(semstamp_positive_raw[25], semstamp_negative_raw[25]),
            "calibrated_scan": calibrated_scan(
                block_sizes,
                semstamp_negative_raw,
                semstamp_positive_raw,
                negative_fold_ids,
            ),
        }
    method_items = [("pmark", pmark_items)]
    if use_semstamp:
        method_items.append(("semstamp", semstamp_items))
    for method, items in method_items:
        negative_records = result["methods"][method]["calibrated_scan"]["negative_records"]
        for item, record in zip(negative_items, negative_records):
            record["source_idx"] = item["source_idx"]
            record["sample_id"] = item["sample_id"]
            record["token_len"] = len(item["token_ids"])
        records = result["methods"][method]["calibrated_scan"]["positive_records"]
        for item, record in zip(items, records):
            record["source_idx"] = item["source_idx"]
            record["sample_id"] = item["sample_id"]
            record["token_len"] = len(item["token_ids"])

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        method: {
            "fixed25": result["methods"][method]["fixed25"],
            "calibrated_scan": result["methods"][method]["calibrated_scan"]["summary"],
        }
        for method in result["methods"]
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
