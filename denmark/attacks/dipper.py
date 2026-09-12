#!/usr/bin/env python3
"""Attack JSONL ``text`` fields with the contextual DIPPER paraphraser.

The runner is deliberately attack-only: it does not load a watermark detector
or any of the source generation models.  A JSONL manifest describes the input
files and output layout, while ``--start``/``--end`` make jobs restartable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


ALLOWED_DIVERSITIES = set(range(0, 101, 10))
STANDARD_DIVERSITIES = {0, 20, 40, 60, 80, 100}
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class EmptyDipperOutput(RuntimeError):
    """Raised when sampled DIPPER output is empty."""


def clean_text(text: str) -> str:
    return " ".join(text.replace("NEWLINE_CHAR", " ").split())


def load_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            required = {"source_id", "method", "base_model", "dataset", "input_path", "output_rel"}
            missing = required.difference(entry)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing keys {sorted(missing)}")
            entries.append(entry)
    if not entries:
        raise ValueError(f"empty manifest: {path}")
    return entries


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_no}: non-empty string field 'text' is required")
            rows.append(row)
    return rows


class Dipper:
    def __init__(self, model_name: str, tokenizer_name: str, dtype: str, device: str) -> None:
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=DTYPES[dtype],
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.device = device

    def paraphrase(
        self,
        text: str,
        *,
        lex_diversity: int,
        order_diversity: int,
        sent_interval: int,
        top_p: float,
        max_length: int,
    ) -> str:
        from nltk.tokenize import sent_tokenize

        lex_code = 100 - lex_diversity
        order_code = 100 - order_diversity
        sentences = sent_tokenize(clean_text(text))
        if not sentences:
            raise ValueError("DIPPER received text with no sentences")

        prefix = ""
        outputs: list[str] = []
        for sent_idx in range(0, len(sentences), sent_interval):
            window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            model_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                model_text += f" {prefix}"
            model_text += f" <sent> {window} </sent>"
            model_input = self.tokenizer(
                [model_text],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            model_input = {key: value.to(self.device) for key, value in model_input.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **model_input,
                    do_sample=True,
                    top_p=top_p,
                    top_k=None,
                    max_length=max_length,
                )
            generated = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            generated = clean_text(generated)
            if not generated:
                raise EmptyDipperOutput(
                    f"DIPPER returned empty text for sentence window {sent_idx}"
                )
            outputs.append(generated)
            prefix = clean_text(f"{prefix} {generated}")
        return clean_text(" ".join(outputs))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def output_path(output_root: Path, output_rel: str, lex: int, order: int, start: int, end: int) -> Path:
    profile = f"l{lex}_o{order}"
    return output_root / profile / output_rel / f"rows_{start:04d}_{end:04d}.jsonl"


def write_atomic(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    lex_group = parser.add_mutually_exclusive_group(required=True)
    lex_group.add_argument(
        "--lex-diversity",
        type=int,
        choices=sorted(ALLOWED_DIVERSITIES),
        help="one lexical-diversity setting",
    )
    lex_group.add_argument(
        "--lex-diversities",
        type=int,
        nargs="+",
        choices=sorted(ALLOWED_DIVERSITIES),
        help="multiple settings to run while reusing one loaded DIPPER model",
    )
    parser.add_argument("--order-diversity", type=int, required=True, choices=sorted(ALLOWED_DIVERSITIES))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1, help="exclusive row index; -1 means all rows")
    parser.add_argument("--sent-interval", type=int, default=3)
    parser.add_argument("--top-p", type=float, default=0.75)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--model", default="kalpeshk2011/dipper-paraphraser-xxl")
    parser.add_argument("--tokenizer", default="google/t5-v1_1-xxl")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a non-empty output chunk instead of failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lex_diversities = (
        [args.lex_diversity]
        if args.lex_diversity is not None
        else list(dict.fromkeys(args.lex_diversities))
    )
    if args.start < 0 or args.end == 0 or args.end < -1:
        raise ValueError("invalid --start/--end")
    if args.sent_interval <= 0:
        raise ValueError("--sent-interval must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest = load_manifest(args.manifest)
    print(
        json.dumps(
            {
                "event": "config",
                "manifest_entries": len(manifest),
                "lex_diversities": lex_diversities,
                "nonstandard_interpolation_values": [
                    value for value in lex_diversities if value not in STANDARD_DIVERSITIES
                ],
                "order_diversity": args.order_diversity,
                "start": args.start,
                "end": args.end,
                "sent_interval": args.sent_interval,
                "top_p": args.top_p,
                "max_length": args.max_length,
                "seed": args.seed,
                "max_attempts": args.max_attempts,
                "model": args.model,
                "tokenizer": args.tokenizer,
                "dtype": args.dtype,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dipper = Dipper(args.model, args.tokenizer, args.dtype, args.device)

    total_written = 0
    total_files = 0
    started = time.time()
    for lex_diversity in lex_diversities:
        for entry in manifest:
            input_path = Path(entry["input_path"])
            source_rows = load_rows(input_path)
            stop = len(source_rows) if args.end == -1 else min(args.end, len(source_rows))
            if args.start >= stop:
                raise ValueError(
                    f"{input_path}: requested [{args.start}, {args.end}) but file has {len(source_rows)} rows"
                )
            path = output_path(
                args.output_root,
                entry["output_rel"],
                lex_diversity,
                args.order_diversity,
                args.start,
                stop,
            )
            if path.exists() and path.stat().st_size > 0:
                if args.skip_existing:
                    print(
                        json.dumps(
                            {
                                "event": "file_skipped",
                                "source_id": entry["source_id"],
                                "output": str(path),
                                "reason": "non_empty_output_exists",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                if not args.overwrite:
                    raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
            attacked_rows: list[dict[str, Any]] = []
            source_seed_offset = int.from_bytes(
                hashlib.sha256(entry["source_id"].encode("utf-8")).digest()[:4],
                "big",
            )
            for row_idx in range(args.start, stop):
                source_row = source_rows[row_idx]
                source_text = source_row["text"]
                base_row_seed = args.seed + source_seed_offset + row_idx
                attacked_text = ""
                row_seed = base_row_seed
                attempts_used = 0
                for attempt_idx in range(args.max_attempts):
                    attempts_used = attempt_idx + 1
                    row_seed = base_row_seed + attempt_idx * 10_000_019
                    seed_everything(row_seed)
                    try:
                        attacked_text = dipper.paraphrase(
                            source_text,
                            lex_diversity=lex_diversity,
                            order_diversity=args.order_diversity,
                            sent_interval=args.sent_interval,
                            top_p=args.top_p,
                            max_length=args.max_length,
                        )
                        break
                    except EmptyDipperOutput as exc:
                        print(
                            json.dumps(
                                {
                                    "event": "row_retry",
                                    "source_id": entry["source_id"],
                                    "source_row_index": row_idx,
                                    "attempt": attempts_used,
                                    "max_attempts": args.max_attempts,
                                    "seed": row_seed,
                                    "error": str(exc),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if attempts_used == args.max_attempts:
                            raise RuntimeError(
                                f"{entry['source_id']} row {row_idx}: DIPPER returned empty "
                                f"output in all {args.max_attempts} attempts"
                            ) from exc
                attacked = dict(source_row)
                attacked["source_text"] = source_text
                attacked["text"] = attacked_text
                attacked["attack"] = "dipper"
                attacked["dipper_profile"] = f"l{lex_diversity}_o{args.order_diversity}"
                attacked["dipper_lex_diversity"] = lex_diversity
                attacked["dipper_order_diversity"] = args.order_diversity
                attacked["dipper_sent_interval"] = args.sent_interval
                attacked["dipper_top_p"] = args.top_p
                attacked["dipper_max_length"] = args.max_length
                attacked["dipper_seed"] = row_seed
                attacked["dipper_base_seed"] = base_row_seed
                attacked["dipper_attempts"] = attempts_used
                attacked["source_path"] = str(input_path)
                attacked["source_row_index"] = row_idx
                attacked["source_id"] = entry["source_id"]
                attacked["watermark_method"] = entry["method"]
                attacked["base_model"] = entry["base_model"]
                attacked["dataset"] = entry["dataset"]
                attacked_rows.append(attacked)

            write_atomic(path, attacked_rows, args.overwrite)
            total_written += len(attacked_rows)
            total_files += 1
            print(
                json.dumps(
                    {
                        "event": "file_done",
                        "source_id": entry["source_id"],
                        "lex_diversity": lex_diversity,
                        "input": str(input_path),
                        "output": str(path),
                        "source_rows": len(source_rows),
                        "written": len(attacked_rows),
                        "elapsed_seconds": round(time.time() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "complete",
                "files": total_files,
                "rows": total_written,
                "elapsed_seconds": round(time.time() - started, 3),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
