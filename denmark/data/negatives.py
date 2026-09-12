#!/usr/bin/env python3
"""Build the paper's tokenizer-matched C4 calibration and ROC pools.

For every requested backbone tokenizer this script creates 40,000 unique C4
RealNewsLike crops by default. Token lengths are sampled uniformly from
150--300, then the rows are split deterministically into 30,000 calibration
examples and 10,000 disjoint held-out ROC negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "realnewslike"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
DEFAULT_TOKENIZERS = (
    ("llada8b", "GSAI-ML/LLaDA-8B-Instruct", "08b83a6feb34df1a6011b80c3c00c7563e963b07"),
    ("llada15", "GSAI-ML/LLaDA-1.5", "84346fd91ba60252d260022201ad6fc5a3468fb2"),
    ("llada20mini", "inclusionAI/LLaDA2.0-mini", "dad945cac317da394b390f82c7b40691d8a881ed"),
    ("dream", "Dream-org/Dream-v0-Instruct-7B", "05334cb9faaf763692dcf9d8737c642be2b2a6ae"),
)


@dataclass(frozen=True)
class TokenizerSpec:
    label: str
    model: str
    revision: str | None


def parse_tokenizer_spec(value: str) -> TokenizerSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=MODEL or LABEL=MODEL@REVISION")
    label, model_revision = value.split("=", 1)
    model, separator, revision = model_revision.rpartition("@")
    if not separator:
        model, revision = model_revision, None
    if not label or not model:
        raise argparse.ArgumentTypeError("tokenizer label and model must be non-empty")
    return TokenizerSpec(label, model, revision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-tokenizer", type=int, default=40_000)
    parser.add_argument("--calibration-size", type=int, default=30_000)
    parser.add_argument("--heldout-size", type=int, default=10_000)
    parser.add_argument("--min-tokens", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument(
        "--tokenizer",
        action="append",
        type=parse_tokenizer_spec,
        dest="tokenizers",
        help="repeatable LABEL=MODEL[@REVISION]; defaults to all paper backbones",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-examples", type=int, default=20)
    return parser.parse_args()


def default_specs() -> list[TokenizerSpec]:
    return [TokenizerSpec(*values) for values in DEFAULT_TOKENIZERS]


def stable_seed(master_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def exact_crop(
    tokenizer: Any,
    ids: list[int],
    target_length: int,
    rng: random.Random,
) -> tuple[str, int] | None:
    if len(ids) < target_length:
        return None
    start = rng.randint(0, len(ids) - target_length)
    text = tokenizer.decode(
        ids[start : start + target_length],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if len(token_ids(tokenizer, text)) != target_length:
        return None
    return text, start


def write_jsonl(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_args(args: argparse.Namespace) -> None:
    if args.samples_per_tokenizer != args.calibration_size + args.heldout_size:
        raise ValueError(
            "samples-per-tokenizer must equal calibration-size + heldout-size"
        )
    if not 0 < args.min_tokens <= args.max_tokens:
        raise ValueError("require 0 < min-tokens <= max-tokens")
    labels = [spec.label for spec in args.tokenizers]
    if len(labels) != len(set(labels)):
        raise ValueError("tokenizer labels must be unique")


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    args.tokenizers = args.tokenizers or default_specs()
    validate_args(args)

    tokenizers = {
        spec.label: AutoTokenizer.from_pretrained(
            spec.model,
            revision=spec.revision,
            trust_remote_code=True,
        )
        for spec in args.tokenizers
    }
    if args.smoke_test:
        stream = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split="train",
            streaming=True,
            revision=args.dataset_revision,
        )
        for source_index, row in enumerate(stream):
            lengths = {
                label: len(token_ids(tokenizer, str(row.get("text") or "")))
                for label, tokenizer in tokenizers.items()
            }
            print(json.dumps({"source_index": source_index, "token_lengths": lengths}))
            if source_index + 1 >= args.smoke_examples:
                return
        raise RuntimeError("C4 stream ended during smoke test")

    rngs = {
        spec.label: random.Random(stable_seed(args.seed, spec.label))
        for spec in args.tokenizers
    }
    accepted: dict[str, list[dict[str, Any]]] = {
        spec.label: [] for spec in args.tokenizers
    }
    skipped_short: Counter[str] = Counter()
    skipped_roundtrip: Counter[str] = Counter()
    specs = {spec.label: spec for spec in args.tokenizers}

    stream = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=args.dataset_revision,
    )
    source_documents_seen = 0
    last_reported = -1
    for source_index, row in enumerate(stream):
        source_documents_seen += 1
        source_text = str(row.get("text") or "")
        for label, tokenizer in tokenizers.items():
            if len(accepted[label]) >= args.samples_per_tokenizer:
                continue
            ids = token_ids(tokenizer, source_text)
            rng = rngs[label]
            target_length = rng.randint(args.min_tokens, args.max_tokens)
            if len(ids) < target_length:
                skipped_short[label] += 1
                continue
            cropped = exact_crop(tokenizer, ids, target_length, rng)
            if cropped is None:
                skipped_roundtrip[label] += 1
                continue
            text, crop_start = cropped
            accepted[label].append(
                {
                    "source_id": f"c4-realnewslike:{source_index}",
                    "source_index": source_index,
                    "text": text,
                    "token_length": target_length,
                    "crop_start": crop_start,
                    "tokenizer": label,
                    "tokenizer_name": specs[label].model,
                    "tokenizer_revision": specs[label].revision,
                }
            )
        minimum_count = min(map(len, accepted.values()))
        report_bucket = minimum_count // args.progress_every if args.progress_every else -1
        if args.progress_every and report_bucket > last_reported:
            last_reported = report_bucket
            print(
                " ".join(f"{label}={len(rows)}" for label, rows in accepted.items()),
                flush=True,
            )
        if all(len(rows) == args.samples_per_tokenizer for rows in accepted.values()):
            break

    incomplete = {
        label: len(rows)
        for label, rows in accepted.items()
        if len(rows) != args.samples_per_tokenizer
    }
    if incomplete:
        raise RuntimeError(f"C4 stream ended before quotas were met: {incomplete}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_summary: dict[str, Any] = {
        "protocol": "paper_c4_tokenizer_matched_disjoint_calibration_and_roc",
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": args.dataset_revision,
        "source_documents_seen": source_documents_seen,
        "tokenizers": {},
    }
    for label, rows in accepted.items():
        calibration = rows[: args.calibration_size]
        heldout = rows[args.calibration_size :]
        cal_ids = {row["source_id"] for row in calibration}
        heldout_ids = {row["source_id"] for row in heldout}
        overlap = cal_ids & heldout_ids
        if overlap:
            raise RuntimeError(f"{label}: calibration/heldout overlap={len(overlap)}")
        base_dir = args.output_dir / label
        write_jsonl(base_dir / "calibration_30000.jsonl", calibration, args.overwrite)
        write_jsonl(base_dir / "heldout_10000.jsonl", heldout, args.overwrite)
        lengths = [int(row["token_length"]) for row in rows]
        global_summary["tokenizers"][label] = {
            "model": specs[label].model,
            "revision": specs[label].revision,
            "total": len(rows),
            "calibration": len(calibration),
            "heldout": len(heldout),
            "source_id_overlap": len(overlap),
            "token_length_min": min(lengths),
            "token_length_max": max(lengths),
            "token_length_histogram": dict(sorted(Counter(lengths).items())),
            "skipped_short": skipped_short[label],
            "skipped_roundtrip": skipped_roundtrip[label],
        }
    summary_path = args.output_dir / "audit.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary_path.write_text(
        json.dumps(global_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
