#!/usr/bin/env python3
"""Document-level GPT rewriting, compression, and expansion on JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

SYSTEM_PROMPT = "You are a helpful assistant that rewrites text while preserving its meaning."
ATTACKS = {
    "rewrite": """Rewrite the following text in a natural and fluent way.
Preserve the original meaning, factual content, named entities, numbers, and technical terms.
Change the overall organization, sentence structure, wording, and phrasing as much as possible.
You may reorganize sentences and paragraphs, but do not summarize, omit details, or add new information.
Keep the total length roughly similar to the original.
Do not add explanations, labels, or commentary.
Return only the rewritten text.

Text:
{text}""",
    "compress_60_70": """Rewrite the following text into a shorter and more concise version.
The original contains approximately {original_word_count} whitespace-separated words. Your rewrite must contain between {target_min_words} and {target_max_words} words (60-70% of the original length); aim for exactly {target_words} words.
Preserve the original meaning, factual content, named entities, numbers, and technical terms.
Remove redundancy, repetition, unnecessary modifiers, and verbose phrasing.
You may reorganize sentences and paragraphs, but do not change the core claims, omit essential information, or add new information.
Do not add explanations, labels, or commentary.
Return only the rewritten text.

Text:
{text}""",
    "expand_130_150": """Rewrite the following text into a more detailed and explicit version.
The original contains approximately {original_word_count} whitespace-separated words. Your rewrite must contain between {target_min_words} and {target_max_words} words (130-150% of the original length); aim for exactly {target_words} words.
Preserve the original meaning, factual content, named entities, numbers, and technical terms.
You may clarify implicit relationships, reorganize sentences and paragraphs, and use richer phrasing, but do not add new facts, examples, numbers, named entities, or claims.
Do not add explanations, labels, or commentary.
Return only the rewritten text.

Text:
{text}""",
}


def target_word_bounds(attack_name: str, word_count: int) -> tuple[int, int] | None:
    if attack_name == "compress_60_70":
        return max(1, math.ceil(word_count * 0.60)), max(1, math.floor(word_count * 0.70))
    if attack_name == "expand_130_150":
        return max(1, math.ceil(word_count * 1.30)), max(1, math.floor(word_count * 1.50))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--attack-name",
        choices=tuple(ATTACKS),
        default="rewrite",
    )
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--input-field", default="text")
    parser.add_argument("--output-field", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cache_key(attack_name: str, model: str, text: str) -> str:
    payload = json.dumps(
        {"attack_name": attack_name, "model_name": model, "text": text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = Lock()
        self.rows: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
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


def rewrite(
    client: Any,
    cache: Cache,
    attack_name: str,
    model: str,
    text: str,
    temperature: float,
    max_retries: int,
    timeout: float,
    max_output_tokens: int,
) -> tuple[str, dict[str, Any]]:
    key = cache_key(attack_name, model, text)
    input_words = max(1, len(text.split()))
    word_bounds = target_word_bounds(attack_name, input_words)
    best_output = ""
    best_distance = float("inf")
    previous_word_count: int | None = None
    cached = cache.get(key)
    if cached and cached.get("ok") and str(cached.get("output") or "").strip():
        cached_output = str(cached["output"]).strip()
        cached_words = len(cached_output.split())
        if word_bounds is None or word_bounds[0] <= cached_words <= word_bounds[1]:
            return cached_output, {
                "cache_hit": True,
                "error": None,
                "length_target_met": True,
            }
    elif cached and str(cached.get("output") or "").strip() and word_bounds is not None:
        best_output = str(cached["output"]).strip()
        previous_word_count = len(best_output.split())
        best_distance = min(
            abs(previous_word_count - word_bounds[0]),
            abs(previous_word_count - word_bounds[1]),
        )

    error = None
    for attempt in range(max_retries + 1):
        try:
            target_min_words, target_max_words = word_bounds or (input_words, input_words)
            target_words = (target_min_words + target_max_words) // 2
            user_content = ATTACKS[attack_name].format(
                text=text,
                original_word_count=input_words,
                target_min_words=target_min_words,
                target_max_words=target_max_words,
                target_words=target_words,
            )
            if previous_word_count is not None:
                direction = "too short" if previous_word_count < target_min_words else "too long"
                user_content += (
                    f"\n\nYour previous attempt contained {previous_word_count} words and was {direction}. "
                    f"Try again and keep the rewrite between {target_min_words} and "
                    f"{target_max_words} words, aiming for exactly {target_words} words."
                )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
                timeout=timeout,
            )
            output = str(response.choices[0].message.content or "").strip()
            if not output:
                raise ValueError("empty rewrite")
            output_words = len(output.split())
            previous_word_count = output_words
            if word_bounds is not None and not (
                word_bounds[0] <= output_words <= word_bounds[1]
            ):
                distance = min(
                    abs(output_words - word_bounds[0]),
                    abs(output_words - word_bounds[1]),
                )
                if distance < best_distance:
                    best_output = output
                    best_distance = distance
                raise ValueError(
                    f"word_count={output_words} outside [{word_bounds[0]}, {word_bounds[1]}]"
                )
            cache.put(
                {
                    "key": key,
                    "attack_name": attack_name,
                    "model_name": model,
                    "ok": True,
                    "output": output,
                }
            )
            return output, {
                "cache_hit": False,
                "error": None,
                "length_target_met": True,
            }
        except Exception as exc:  # API failures must preserve the original text.
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                if not (
                    isinstance(exc, ValueError)
                    and str(exc).startswith("word_count=")
                ):
                    time.sleep(min(30.0, 1.5 * (2**attempt)))

    fallback = best_output or text
    cache.put(
        {
            "key": key,
            "attack_name": attack_name,
            "model_name": model,
            "ok": False,
            "output": fallback,
            "error": error,
        }
    )
    return fallback, {
        "cache_hit": False,
        "error": error,
        "length_target_met": False,
    }


def output_path(input_path: Path, output_dir: Path, path_root: Path) -> Path:
    relative = input_path.resolve().relative_to(path_root.resolve())
    return output_dir / relative.parent / f"{relative.stem}.attacked.jsonl"


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    if args.output_field is None:
        args.output_field = f"text_gpt_{args.attack_name}_document"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=1,
    )
    cache = Cache(args.cache)

    tasks = []
    for input_path in args.inputs:
        rows = read_jsonl(input_path)
        for index, row in enumerate(rows):
            text = row.get(args.input_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing {args.input_field}: {input_path}:{index + 1}")
            tasks.append((input_path, index, row, text.strip()))

    completed: dict[Path, dict[int, dict[str, Any]]] = {path: {} for path in args.inputs}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                rewrite,
                client,
                cache,
                args.attack_name,
                args.model,
                text,
                args.temperature,
                args.max_retries,
                args.timeout,
                args.max_output_tokens,
            ): (input_path, index, row, text)
            for input_path, index, row, text in tasks
        }
        done = 0
        for future in as_completed(futures):
            input_path, index, row, text = futures[future]
            output, metadata = future.result()
            result = dict(row)
            result[args.output_field] = output
            result["_whole_text_attack_meta"] = {
                "attack_name": args.attack_name,
                "model_name": args.model,
                "temperature": args.temperature,
                "input_field": args.input_field,
                "output_field": args.output_field,
                "input_words": len(text.split()),
                "output_words": len(output.split()),
                **metadata,
            }
            completed[input_path][index] = result
            done += 1
            if done % 25 == 0 or done == len(tasks):
                print(f"completed={done}/{len(tasks)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"attack_name": args.attack_name, "model": args.model, "files": {}}
    for input_path in args.inputs:
        destination = output_path(input_path, args.output_dir, args.path_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        ordered = [completed[input_path][index] for index in sorted(completed[input_path])]
        with destination.open("w", encoding="utf-8") as handle:
            for row in ordered:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        failures = sum(bool(row["_whole_text_attack_meta"]["error"]) for row in ordered)
        ratios = [
            row["_whole_text_attack_meta"]["output_words"]
            / max(1, row["_whole_text_attack_meta"]["input_words"])
            for row in ordered
        ]
        summary["files"][str(destination)] = {
            "rows": len(ordered),
            "target_misses_best_candidate_kept": failures,
            "mean_word_ratio": sum(ratios) / len(ratios),
            "min_word_ratio": min(ratios),
            "max_word_ratio": max(ratios),
        }
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
