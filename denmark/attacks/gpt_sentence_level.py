#!/usr/bin/env python3
"""Sentence-level GPT-4o-mini rewriting, compression, and expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from denmark.attacks.common import (
    GPT_SENTENCE_ATTACKS,
    SYSTEM_PROMPT,
    join_sentences,
    sentence_attack_spec,
    sleep_with_backoff,
    split_sentences,
)


TEXT_FIELDS = (
    "text",
    "completion",
    "watermarked_text",
    "hash_distribution_text",
    "dgmark_text",
    "output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path)
    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=tuple(GPT_SENTENCE_ATTACKS),
        default=tuple(GPT_SENTENCE_ATTACKS),
    )
    parser.add_argument("--provider", choices=("openrouter", "openai"), default="openrouter")
    parser.add_argument("--model")
    parser.add_argument("--input-field", action="append", dest="input_fields")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def make_client(args: argparse.Namespace) -> tuple[Any, str]:
    from openai import OpenAI

    if args.provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        return (
            OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, max_retries=1),
            args.model or "openai/gpt-4o-mini",
        )
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=key, max_retries=1), args.model or "gpt-4o-mini"


def cache_key(attack: str, model: str, sentence: str) -> str:
    payload = json.dumps(
        {"attack_name": attack, "model_name": model, "sentence": sentence},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class JsonlCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row.get("key"), str):
                        self.rows[row["key"]] = row

    def get(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            return self.rows.get(key)

    def put(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.rows[row["key"]] = row
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_text(row: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, str]:
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
    raise ValueError(f"none of the text fields is present: {fields}")


def rewrite_unit(
    client: Any,
    cache: JsonlCache,
    attack: str,
    model: str,
    sentence: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    key = cache_key(attack, model, sentence)
    cached = cache.get(key)
    if cached and cached.get("ok") and cached.get("output"):
        return cached

    prompt, default_temperature, bounds = sentence_attack_spec(attack, sentence)
    temperature = args.temperature if args.temperature is not None else default_temperature
    best_output = sentence
    best_distance = float("inf")
    error = None
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=args.max_output_tokens,
                timeout=args.timeout,
            )
            output = str(response.choices[0].message.content or "").strip()
            if not output:
                raise ValueError("empty rewrite")
            if bounds is not None:
                words = len(output.split())
                distance = min(abs(words - bounds[0]), abs(words - bounds[1]))
                if distance < best_distance:
                    best_output, best_distance = output, distance
                if not bounds[0] <= words <= bounds[1]:
                    raise ValueError(
                        f"compression word count {words} outside [{bounds[0]}, {bounds[1]}]"
                    )
            row = {
                "key": key,
                "attack_name": attack,
                "model_name": model,
                "ok": True,
                "output": output,
                "error": None,
            }
            cache.put(row)
            return row
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                sleep_with_backoff(attempt)
    row = {
        "key": key,
        "attack_name": attack,
        "model_name": model,
        "ok": False,
        "output": best_output,
        "error": error,
    }
    cache.put(row)
    return row


def output_path(input_path: Path, output_dir: Path, root: Path) -> Path:
    relative = input_path.resolve().relative_to(root.resolve())
    return output_dir / relative.parent / f"{relative.stem}.attacked.jsonl"


def main() -> None:
    args = parse_args()
    client, model = make_client(args)
    root = args.path_root or Path(
        os.path.commonpath([str(path.resolve().parent) for path in args.inputs])
    )
    fields = tuple(args.input_fields or TEXT_FIELDS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = JsonlCache(args.cache or args.output_dir / "sentence_cache.jsonl")

    source_rows: dict[Path, list[dict[str, Any]]] = {}
    units: dict[tuple[Path, int], list[str]] = {}
    pending: dict[str, tuple[str, str]] = {}
    results: dict[str, dict[str, Any]] = {}
    source_fields: dict[tuple[Path, int], str] = {}
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        source_rows[path] = rows
        for index, row in enumerate(rows):
            field, text = select_text(row, fields)
            source_fields[(path, index)] = field
            row_units = split_sentences(text)
            units[(path, index)] = row_units
            for attack in args.attacks:
                for sentence in row_units:
                    key = cache_key(attack, model, sentence)
                    cached = cache.get(key)
                    if cached and cached.get("ok") and cached.get("output"):
                        results[key] = cached
                    else:
                        pending.setdefault(key, (attack, sentence))

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                rewrite_unit, client, cache, attack, model, sentence, args
            ): key
            for key, (attack, sentence) in pending.items()
        }
        for count, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if count % 50 == 0 or count == len(futures):
                print(f"completed={count}/{len(futures)}", flush=True)

    summary: dict[str, Any] = {
        "model": model,
        "attacks": args.attacks,
        "cache": str(cache.path),
        "files": {},
    }
    for path, rows in source_rows.items():
        output_rows = []
        failed_units = 0
        for index, row in enumerate(rows):
            output_row = dict(row)
            row_units = units[(path, index)]
            row_failed_units = 0
            for attack in args.attacks:
                unit_results = [
                    results[cache_key(attack, model, sentence)]
                    for sentence in row_units
                ]
                attack_failures = sum(not result.get("ok") for result in unit_results)
                row_failed_units += attack_failures
                failed_units += attack_failures
                output_row[f"text_gpt_{attack}_sentence"] = join_sentences(
                    [str(result["output"]) for result in unit_results]
                )
            output_row["_sentence_attack_meta"] = {
                "model_name": model,
                "temperature": args.temperature if args.temperature is not None else 0.7,
                "source_field": source_fields[(path, index)],
                "sentence_count": len(row_units),
                "failed_units": row_failed_units,
                "attacks": args.attacks,
            }
            output_rows.append(output_row)
        destination = output_path(path, args.output_dir, root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["files"][str(destination)] = {
            "rows": len(output_rows),
            "failed_units": failed_units,
        }
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
