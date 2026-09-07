#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[2]))

from denomark.core.calibration import CalibratedDetectorSuite, RawSemanticDetector
from denomark.core.scoring import clean_text
from denomark.evaluation.metrics import roc_interpolated_tpr
from denomark.core.model import build_directions


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def auc_rank(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    scores = np.r_[neg, pos]
    labels = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def threshold_tpr(pos: list[float], neg: list[float], fpr: float = 0.01) -> tuple[float, float]:
    if not neg:
        return float("inf"), float("nan")
    neg_sorted = np.sort(np.asarray(neg, dtype=float))
    idx = int(math.ceil((1.0 - fpr) * len(neg_sorted))) - 1
    idx = max(0, min(idx, len(neg_sorted) - 1))
    threshold = float(neg_sorted[idx])
    return threshold, float((np.asarray(pos, dtype=float) > threshold).mean()) if pos else float("nan")


def summarize(pos: list[float], neg: list[float]) -> dict:
    threshold01, tpr01 = threshold_tpr(pos, neg, 0.001)
    threshold05, tpr05 = threshold_tpr(pos, neg, 0.005)
    threshold, tpr = threshold_tpr(pos, neg)
    threshold5, tpr5 = threshold_tpr(pos, neg, 0.05)
    threshold10, tpr10 = threshold_tpr(pos, neg, 0.10)
    return {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "pos_mean": float(np.mean(pos)) if pos else float("nan"),
        "pos_median": float(np.median(pos)) if pos else float("nan"),
        "neg_mean": float(np.mean(neg)) if neg else float("nan"),
        "neg_median": float(np.median(neg)) if neg else float("nan"),
        "threshold_0_1pct": threshold01,
        "threshold_0_5pct": threshold05,
        "threshold_1pct": threshold,
        "threshold_5pct": threshold5,
        "threshold_10pct": threshold10,
        "tpr_at_0_1pct": tpr01,
        "tpr_at_0_5pct": tpr05,
        "tpr_at_1pct": tpr,
        "tpr_at_5pct": tpr5,
        "tpr_at_10pct": tpr10,
        "roc_tpr_at_0_1pct": roc_interpolated_tpr(pos, neg, 0.001),
        "roc_tpr_at_0_5pct": roc_interpolated_tpr(pos, neg, 0.005),
        "roc_tpr_at_1pct": roc_interpolated_tpr(pos, neg, 0.01),
        "roc_tpr_at_5pct": roc_interpolated_tpr(pos, neg, 0.05),
        "roc_tpr_at_10pct": roc_interpolated_tpr(pos, neg, 0.10),
        "auc": auc_rank(pos, neg),
    }


def len_stats(vals: list[int]) -> dict:
    return {
        "n": len(vals),
        "min": min(vals) if vals else None,
        "median": statistics.median(vals) if vals else None,
        "mean": float(np.mean(vals)) if vals else None,
        "max": max(vals) if vals else None,
        "ge100": sum(v >= 100 for v in vals),
        "ge150": sum(v >= 150 for v in vals),
    }


def make_item(
    row: dict,
    idx: int,
    tokenizer,
    gen_length: int,
    *,
    force_retokenize: bool = False,
) -> dict:
    attacked_text = row.get("attacked_text") or row.get("attack_text")
    text = clean_text(
        attacked_text
        or row.get("text")
        or row.get("watermarked_text")
        or row.get("unwatermarked_text")
        or row.get("generation_text_skip_special")
        or row.get("generation")
        or row.get("completion")
        or row.get("output")
        or ""
    )
    if attacked_text or force_retokenize:
        full_token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        token_len = len(full_token_ids)
    else:
        full_token_ids = (
            row.get("generated_token_ids")
            or row.get("generation_token_ids")
            or row.get("watermarked_token_ids")
            or tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        token_len = int(row.get("token_len") or row.get("generation_token_len") or len(full_token_ids))
    token_ids = full_token_ids[:gen_length]
    item = {"idx": idx, "text": text, "token_ids": token_ids, "token_len": token_len}
    if row.get("source_id") is not None:
        item["source_id"] = str(row["source_id"])
    for key in ("sample_nonce", "sample_id", "watermark_sample_id"):
        if row.get(key) is not None:
            item[key] = row[key]
            break
    diag = row.get("wm_gen_diag")
    if "sample_id" not in item and "sample_nonce" not in item and isinstance(diag, list) and diag:
        if isinstance(diag[0], dict) and diag[0].get("sample_id") is not None:
            item["sample_id"] = diag[0]["sample_id"]
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_jsonl", type=Path, required=True)
    parser.add_argument("--negative_jsonl", type=Path, required=True)
    parser.add_argument(
        "--calibration_jsonl",
        type=Path,
        help=(
            "Independent calibration pool. If omitted, the negative pool is "
            "reused with leave-one-out scoring for backward compatibility."
        ),
    )
    parser.add_argument("--negative_label", default="clean")
    parser.add_argument("--max_positive_rows", type=int, default=None)
    parser.add_argument("--max_negative_rows", type=int, default=None)
    parser.add_argument("--max_calibration_rows", type=int, default=None)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model identifier or local generator checkpoint",
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="Hugging Face model identifier or local DenoMark encoder checkpoint",
    )
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=25)
    parser.add_argument("--num_message_bits", type=int, default=2)
    parser.add_argument("--direction_seed", type=int, default=42)
    parser.add_argument("--message_seed", type=int, default=0)
    parser.add_argument("--orthogonal_directions", action="store_true")
    parser.add_argument("--scan_min", type=int, default=12)
    parser.add_argument("--scan_max", type=int, default=37)
    parser.add_argument("--detectors", default="fixed25_mean,calibrated_scan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_scores", action="store_true")
    parser.add_argument(
        "--subsets",
        default="all,pos_all_neg>=150,len>=150,pos>=150_neg_all",
        help="Comma-separated evaluation subsets. Use pos_all_neg>=150 for attacked positives without post-filtering.",
    )
    parser.add_argument(
        "--retokenize_positive",
        action="store_true",
        help="Ignore stored positive token IDs and tokenize the selected text field.",
    )
    parser.add_argument(
        "--retokenize_negative",
        action="store_true",
        help="Ignore stored negative token IDs and tokenize the selected text field.",
    )
    parser.add_argument(
        "--retokenize_calibration",
        action="store_true",
        help="Ignore stored calibration token IDs and tokenize the selected text field.",
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_txt", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    enc_tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.float32).to(args.device).eval()

    dirs, signs = build_directions(
        math.ceil(args.gen_length / args.block_size),
        args.num_message_bits,
        args.direction_seed,
        args.message_seed,
        orthogonal=args.orthogonal_directions,
    )
    raw_detector = RawSemanticDetector(
        enc,
        enc_tok,
        tokenizer,
        dirs,
        signs,
        args.device,
        gen_length=args.gen_length,
        fixed_block_size=args.block_size,
        scan_block_sizes=range(args.scan_min, args.scan_max + 1),
    )

    pos_rows = read_jsonl(args.positive_jsonl)
    neg_rows = read_jsonl(args.negative_jsonl)
    cal_rows = read_jsonl(args.calibration_jsonl) if args.calibration_jsonl else None
    if args.max_positive_rows is not None:
        pos_rows = pos_rows[: args.max_positive_rows]
    if args.max_negative_rows is not None:
        neg_rows = neg_rows[: args.max_negative_rows]
    if cal_rows is None:
        cal_rows = neg_rows
    elif args.max_calibration_rows is not None:
        cal_rows = cal_rows[: args.max_calibration_rows]
    pos_all = [
        make_item(
            row,
            idx,
            tokenizer,
            args.gen_length,
            force_retokenize=args.retokenize_positive,
        )
        for idx, row in enumerate(pos_rows)
    ]
    neg_all = [
        make_item(
            row,
            idx,
            tokenizer,
            args.gen_length,
            force_retokenize=args.retokenize_negative,
        )
        for idx, row in enumerate(neg_rows)
    ]
    cal_all = [
        make_item(
            row,
            idx,
            tokenizer,
            args.gen_length,
            force_retokenize=args.retokenize_calibration,
        )
        for idx, row in enumerate(cal_rows)
    ]
    calibration_is_disjoint = args.calibration_jsonl is not None
    calibration_negative_overlap = None
    if calibration_is_disjoint:
        cal_id_values = [item.get("source_id") for item in cal_all]
        neg_id_values = [item.get("source_id") for item in neg_all]
        if any(value is None for value in (*cal_id_values, *neg_id_values)):
            raise ValueError(
                "disjoint calibration requires source_id on every calibration "
                "and held-out negative row"
            )
        cal_ids = {str(value) for value in cal_id_values}
        neg_ids = {str(value) for value in neg_id_values}
        if len(cal_ids) != len(cal_all) or len(neg_ids) != len(neg_all):
            raise ValueError("calibration and held-out source_id values must be unique")
        calibration_negative_overlap = len(cal_ids & neg_ids)
        if calibration_negative_overlap:
            raise ValueError(
                "calibration and held-out negative pools overlap on "
                f"{calibration_negative_overlap} source IDs"
            )

    detectors = [x.strip() for x in args.detectors.split(",") if x.strip()]
    result = {
        "mode": "active_blocks_only",
        "note": "Scores average valid units; calibrated_scan uses per-size empirical p-values and Bonferroni correction.",
        "negative_source": str(args.negative_jsonl),
        "calibration_source": str(args.calibration_jsonl or args.negative_jsonl),
        "calibration_mode": (
            "disjoint_fixed_pool" if calibration_is_disjoint else "shared_pool_leave_one_out"
        ),
        "calibration_negative_source_id_overlap": calibration_negative_overlap,
        "negative_label": args.negative_label,
        "positive_source": str(args.positive_jsonl),
        "positive_retokenized": args.retokenize_positive,
        "negative_retokenized": args.retokenize_negative,
        "calibration_retokenized": args.retokenize_calibration,
        "calibration_len_stats": len_stats([x["token_len"] for x in cal_all]),
        "negative_len_stats": len_stats([x["token_len"] for x in neg_all]),
        "positive_len_stats": len_stats([x["token_len"] for x in pos_all]),
        "detectors": detectors,
        "subsets": {},
    }
    lines = [
        "mode=active_blocks_only",
        f"baseline={args.negative_label}, n_neg_all={len(neg_all)}",
        "subset detector npos nneg pos_mean neg_mean ROC_TPR0.1% ROC_TPR1% ROC_TPR5% ROC_TPR10% AUC",
    ]

    subset_specs = [
        ("all", lambda x: True, lambda x: True),
        ("pos_all_neg>=150", lambda x: True, lambda x: x["token_len"] >= 150),
        ("len>=150", lambda x: x["token_len"] >= 150, lambda x: x["token_len"] >= 150),
        ("pos>=150_neg_all", lambda x: x["token_len"] >= 150, lambda x: True),
    ]
    requested_subsets = {value.strip() for value in args.subsets.split(",") if value.strip()}
    known_subsets = {name for name, _, _ in subset_specs}
    unknown_subsets = requested_subsets - known_subsets
    if unknown_subsets:
        raise ValueError(f"unknown subsets: {sorted(unknown_subsets)}")
    subsets = [spec for spec in subset_specs if spec[0] in requested_subsets]
    if not subsets:
        raise ValueError("at least one evaluation subset is required")
    for subset_name, pos_filter, neg_filter in subsets:
        pos = [x for x in pos_all if pos_filter(x)]
        neg = [x for x in neg_all if neg_filter(x)]
        calibration = [x for x in cal_all if neg_filter(x)]
        suite = CalibratedDetectorSuite(
            raw_detector,
            neg,
            use_length_buckets=False,
            detectors=detectors,
            calibration_items=(calibration if calibration_is_disjoint else None),
        )
        subset_result = {
            "n_pos": len(pos),
            "n_neg": len(neg),
            "n_calibration": len(calibration),
            "calibrated": {},
        }
        for det in detectors:
            neg_scores = [suite.score_item(item, det)["score"] for item in neg]
            pos_scores = [suite.score_item(item, det)["score"] for item in pos]
            summary = summarize(pos_scores, neg_scores)
            if args.save_scores:
                summary["positive_scores"] = pos_scores
                summary["negative_scores"] = neg_scores
            subset_result["calibrated"][det] = summary
            lines.append(
                f"{subset_name} {det} {summary['n_pos']} {summary['n_neg']} "
                f"{summary['pos_mean']:.4f} {summary['neg_mean']:.4f} "
                f"{summary['roc_tpr_at_0_1pct']:.4f} "
                f"{summary['roc_tpr_at_0_5pct']:.4f} "
                f"{summary['roc_tpr_at_1pct']:.4f} "
                f"{summary['roc_tpr_at_5pct']:.4f} "
                f"{summary['roc_tpr_at_10pct']:.4f} {summary['auc']:.4f}"
            )
        result["subsets"][subset_name] = subset_result

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
