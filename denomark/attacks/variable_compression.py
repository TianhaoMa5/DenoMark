#!/usr/bin/env python3
"""Apply heterogeneous document-local GPT compression in three contiguous regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from denomark.attacks.common import join_sentences, split_sentences


MODEL_NAME = "openai/gpt-4o-mini"
RATIO_BANDS = ((0.50, 0.60), (0.60, 0.70), (0.70, 0.80))
SYSTEM_PROMPT = "You are a helpful assistant that rewrites text while preserving its meaning."
USER_PROMPT = """Rewrite the following passage in a shorter and more concise way.
Preserve the original meaning, factual content, named entities, numbers, technical terms, and essential qualifiers.
Preserve the sentence order and do not add new information.
The original passage contains approximately {original_word_count} words.
Target approximately {ratio_min_pct}% to {ratio_max_pct}% of the original length: about {target_min_words} to {target_max_words} words.
Remove only unnecessary modifiers, redundancy, and verbose phrasing.
Do not add explanations.
Return only the rewritten passage.

Passage:
{text}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--text-fields", default="text,watermarked_text,generated_text,hash_distribution_text,eval_text")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--api-retries", type=int, default=5)
    parser.add_argument("--length-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def choose_text(row: dict[str, Any], fields: list[str]) -> tuple[str, str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    raise ValueError("no supported non-empty text field")


def balanced_regions(sentences: list[str]) -> list[list[str]]:
    n_regions = min(3, len(sentences))
    if n_regions == 0:
        return []
    base, extra = divmod(len(sentences), n_regions)
    regions: list[list[str]] = []
    cursor = 0
    for region_idx in range(n_regions):
        width = base + (1 if region_idx < extra else 0)
        regions.append(sentences[cursor : cursor + width])
        cursor += width
    return regions


def row_key(row: dict[str, Any], line_no: int) -> str:
    for field in ("sample_id", "prompt_idx", "source_row_idx", "id"):
        if row.get(field) is not None:
            return str(row[field])
    return str(line_no)


def ratio_plan(seed: int, source: Path, row: dict[str, Any], line_no: int, n_regions: int) -> list[tuple[float, float]]:
    digest = hashlib.sha256(f"{seed}:{source.resolve()}:{row_key(row, line_no)}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    bands = list(RATIO_BANDS)
    rng.shuffle(bands)
    return bands[:n_regions]


def cache_key(model: str, ratio_min: float, ratio_max: float, text: str) -> str:
    payload = {
        "attack": "gpt_variable_local_compression_three_region",
        "model": model,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "text": text,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.key_locks: dict[str, Lock] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row.get("key"), str):
                        self.rows[row["key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            return self.rows.get(key)

    def key_lock(self, key: str) -> Lock:
        with self.lock:
            return self.key_locks.setdefault(key, Lock())

    def put(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.rows[row["key"]] = row
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def retry_sleep(attempt: int) -> None:
    time.sleep(min(20.0, 1.5 * (2**attempt)) + random.random())


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, max_retries=1)
    fields = [field.strip() for field in args.text_fields.split(",") if field.strip()]
    cache = JsonlCache(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    source_rows: list[tuple[int, dict[str, Any]]] = []
    with args.input.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line.strip():
                source_rows.append((line_no, json.loads(line)))

    error_lock = Lock()

    def rewrite_region(text: str, ratio_min: float, ratio_max: float) -> tuple[str, dict[str, Any]]:
        key = cache_key(args.model, ratio_min, ratio_max, text)
        cached = cache.get(key)
        if cached and cached.get("ok") and isinstance(cached.get("rewrite"), str):
            return cached["rewrite"], {"cache": "hit", "key": key, "attempts": cached.get("attempts", 1)}
        with cache.key_lock(key):
            cached = cache.get(key)
            if cached and cached.get("ok") and isinstance(cached.get("rewrite"), str):
                return cached["rewrite"], {"cache": "hit", "key": key, "attempts": cached.get("attempts", 1)}

            original_words = max(1, len(text.split()))
            target_min = max(1, math.ceil(original_words * ratio_min))
            target_max = max(target_min, math.floor(original_words * ratio_max))
            best: tuple[float, str, float] | None = None
            last_error = ""
            total_attempts = 0
            for length_attempt in range(args.length_retries + 1):
                prompt = USER_PROMPT.format(
                    original_word_count=original_words,
                    ratio_min_pct=int(ratio_min * 100),
                    ratio_max_pct=int(ratio_max * 100),
                    target_min_words=target_min,
                    target_max_words=target_max,
                    text=text,
                )
                if length_attempt:
                    prompt = (
                        f"The previous rewrite missed the requested length. Follow the {int(ratio_min*100)}% to "
                        f"{int(ratio_max*100)}% target strictly.\n\n" + prompt
                    )
                response_text = ""
                for api_attempt in range(args.api_retries):
                    total_attempts += 1
                    try:
                        response = client.chat.completions.create(
                            model=args.model,
                            temperature=args.temperature,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            timeout=args.request_timeout,
                        )
                        response_text = str(response.choices[0].message.content or "").strip()
                        if not response_text:
                            raise RuntimeError("empty rewrite")
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        if api_attempt + 1 < args.api_retries:
                            retry_sleep(api_attempt)
                if not response_text:
                    continue
                actual_ratio = len(response_text.split()) / original_words
                distance = max(ratio_min - actual_ratio, 0.0, actual_ratio - ratio_max)
                if best is None or distance < best[0]:
                    best = (distance, response_text, actual_ratio)
                if ratio_min <= actual_ratio <= ratio_max:
                    break

            if best is None:
                with error_lock, args.errors.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"key": key, "error": last_error, "text": text}, ensure_ascii=False) + "\n")
                return text, {"cache": "miss_failed", "key": key, "attempts": total_attempts, "error": last_error}

            _, rewrite, actual_ratio = best
            cache.put({
                "key": key,
                "ok": True,
                "attack_name": "gpt_variable_local_compression_three_region",
                "model_name": args.model,
                "ratio_min": ratio_min,
                "ratio_max": ratio_max,
                "text": text,
                "rewrite": rewrite,
                "actual_ratio": actual_ratio,
                "attempts": total_attempts,
                "in_band": ratio_min <= actual_ratio <= ratio_max,
            })
            return rewrite, {"cache": "miss", "key": key, "attempts": total_attempts}

    def process(line_no: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        source_field, text = choose_text(row, fields)
        sentences = split_sentences(text)
        regions = balanced_regions(sentences)
        plan = ratio_plan(args.seed, args.input, row, line_no, len(regions))
        rewritten_regions: list[str] = []
        region_meta: list[dict[str, Any]] = []
        for region_idx, (region_sentences, (ratio_min, ratio_max)) in enumerate(zip(regions, plan)):
            region_text = join_sentences(region_sentences)
            rewrite, call_meta = rewrite_region(region_text, ratio_min, ratio_max)
            original_words = max(1, len(region_text.split()))
            rewritten_words = len(rewrite.split())
            rewritten_regions.append(rewrite)
            region_meta.append({
                "region_idx": region_idx,
                "sentence_count": len(region_sentences),
                "ratio_min": ratio_min,
                "ratio_max": ratio_max,
                "original_words": original_words,
                "attacked_words": rewritten_words,
                "actual_ratio": rewritten_words / original_words,
                **call_meta,
            })
        attacked = join_sentences(rewritten_regions)
        previous_token_len = row.get("token_len")
        for stale in ("token_ids", "generated_token_ids", "completion_token_ids", "watermarked_token_ids", "input_ids", "token_len"):
            row.pop(stale, None)
        row["text"] = attacked
        row["attack_text"] = attacked
        row["text_gpt_variable_local_compression"] = attacked
        row["_variable_local_compression_meta"] = {
            "attack_name": "gpt_variable_local_compression_three_region",
            "model": args.model,
            "temperature": args.temperature,
            "source_field": source_field,
            "sentence_count": len(sentences),
            "region_count": len(regions),
            "pre_attack_token_len": previous_token_len,
            "seed": args.seed,
            "regions": region_meta,
        }
        return line_no, row

    processed: list[tuple[int, dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {executor.submit(process, line_no, row): line_no for line_no, row in source_rows}
        for completed, future in enumerate(as_completed(futures), 1):
            processed.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"{args.input.name}: {completed}/{len(futures)} rows", flush=True)

    processed.sort(key=lambda item: item[0])
    with args.output.open("w", encoding="utf-8") as handle:
        for _, row in processed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    region_rows = [
        region
        for _, row in processed
        for region in row["_variable_local_compression_meta"]["regions"]
    ]
    per_band: dict[str, dict[str, Any]] = {}
    for ratio_min, ratio_max in RATIO_BANDS:
        selected = [r for r in region_rows if r["ratio_min"] == ratio_min and r["ratio_max"] == ratio_max]
        per_band[f"{int(ratio_min*100)}-{int(ratio_max*100)}"] = {
            "regions": len(selected),
            "in_band": sum(ratio_min <= r["actual_ratio"] <= ratio_max for r in selected),
            "mean_actual_ratio": sum(r["actual_ratio"] for r in selected) / max(1, len(selected)),
            "min_actual_ratio": min((r["actual_ratio"] for r in selected), default=None),
            "max_actual_ratio": max((r["actual_ratio"] for r in selected), default=None),
        }
    audit = {
        "attack": "gpt_variable_local_compression_three_region",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "rows": len(processed),
        "rows_with_three_regions": sum(row["_variable_local_compression_meta"]["region_count"] == 3 for _, row in processed),
        "rows_with_fewer_than_three_regions": sum(row["_variable_local_compression_meta"]["region_count"] < 3 for _, row in processed),
        "regions": len(region_rows),
        "api_failures": sum(bool(r.get("error")) for r in region_rows),
        "per_band": per_band,
    }
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
