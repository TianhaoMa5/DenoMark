#!/usr/bin/env python3
"""Run cached English-Chinese-English back-translation attacks on JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from denomark.attacks.common import join_sentences, split_sentences
except ImportError:
    from denomark.attacks.common import join_sentences, split_sentences


SYSTEM_PROMPT = (
    "You are a professional translator. Follow the user's instructions exactly and "
    "return valid JSON only."
)

SENTENCE_PROMPT = """You are performing a sentence-level back-translation attack for a text watermark robustness evaluation.

Given ONE English sentence, perform the following two steps:

1. Translate the English sentence into natural and fluent Chinese while preserving its original meaning.
2. Translate ONLY your Chinese translation from Step 1 back into natural and fluent English.

Important requirements:

* Treat the input as a single independent sentence.
* Preserve the original semantic meaning as closely as possible.
* Do not intentionally copy the wording or syntactic structure of the original English sentence when translating back.
* Use natural phrasing that a professional translator would produce.
* Do not add, remove, summarize, explain, or expand any information.
* The second English translation must be based on the Chinese intermediate translation, not directly on the original English sentence.
* Do not mention the watermark or the attack.

Return exactly the following JSON format:

{{
  "intermediate_translation": "",
  "back_translation": ""
}}

Input sentence:
{text}"""

DOCUMENT_PROMPT = """You are performing a document-level back-translation attack for a text watermark robustness evaluation.

Given the complete English text below, perform the following two steps on the ENTIRE TEXT AS A SINGLE UNIT:

1. Translate the full English text into natural and fluent Chinese.
2. Translate ONLY your complete Chinese translation from Step 1 back into natural and fluent English.

Important requirements:

* Process the entire input jointly rather than translating individual sentences independently.
* Preserve the overall meaning, facts, logical relationships, discourse flow, and tone of the original text.
* Preserve paragraph boundaries when possible.
* Do not intentionally copy the original English wording or sentence structures when translating back.
* Allow natural changes in lexical choice, syntax, sentence boundaries, and discourse expressions that normally arise through translation.
* Do not summarize, shorten, expand, explain, or introduce new information.
* The final English text must be translated from the Chinese intermediate text, not directly rewritten from the original English.
* Do not mention the watermark or the attack.

Return exactly the following JSON format:

{{
  "intermediate_translation": "",
  "back_translation": ""
}}

Input text:
{text}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mode", choices=("sentence", "document"), required=True)
    parser.add_argument("--input-field", default="text")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-output-tokens", type=int, default=2400)
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
    return hashlib.sha256(payload.encode()).hexdigest()


class JsonlCache:
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


def parse_translation(content: str) -> tuple[str, str]:
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    intermediate = payload.get("intermediate_translation")
    back = payload.get("back_translation")
    if not isinstance(intermediate, str) or not intermediate.strip():
        raise ValueError("missing intermediate_translation")
    if not isinstance(back, str) or not back.strip():
        raise ValueError("missing back_translation")
    return intermediate.strip(), back.strip()


def translate_one(
    client: Any,
    cache: JsonlCache,
    *,
    attack_name: str,
    model: str,
    prompt_template: str,
    text: str,
    temperature: float,
    max_retries: int,
    timeout: float,
    max_output_tokens: int,
) -> dict[str, Any]:
    key = cache_key(attack_name, model, text)
    cached = cache.get(key)
    if cached and cached.get("ok") and cached.get("back_translation"):
        return cached

    error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_template.format(text=text)},
                ],
                temperature=temperature,
                max_tokens=max_output_tokens,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
            content = str(response.choices[0].message.content or "")
            intermediate, back = parse_translation(content)
            row = {
                "key": key,
                "attack_name": attack_name,
                "model_name": model,
                "ok": True,
                "intermediate_translation": intermediate,
                "back_translation": back,
                "error": None,
            }
            cache.put(row)
            return row
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                delay = min(45.0, 1.5 * (2**attempt)) * random.uniform(0.8, 1.2)
                time.sleep(delay)

    row = {
        "key": key,
        "attack_name": attack_name,
        "model_name": model,
        "ok": False,
        "intermediate_translation": "",
        "back_translation": text,
        "error": error,
    }
    cache.put(row)
    return row


def output_path(input_path: Path, output_dir: Path, path_root: Path) -> Path:
    relative = input_path.resolve().relative_to(path_root.resolve())
    return output_dir / relative.parent / f"{relative.stem}.attacked.jsonl"


def main() -> None:
    args = parse_args()
    from openai import OpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        max_retries=1,
    )
    cache = JsonlCache(args.cache)
    attack_name = f"gpt_backtranslation_{args.mode}_en_zh_en"
    prompt_template = SENTENCE_PROMPT if args.mode == "sentence" else DOCUMENT_PROMPT

    source_rows: dict[Path, list[dict[str, Any]]] = {}
    units_by_row: dict[tuple[Path, int], list[str]] = {}
    unique_units: dict[str, str] = {}
    for path in args.inputs:
        rows = read_jsonl(path)
        source_rows[path] = rows
        for index, row in enumerate(rows):
            text = row.get(args.input_field)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"missing {args.input_field}: {path}:{index + 1}")
            units = split_sentences(text) if args.mode == "sentence" else [text.strip()]
            units_by_row[(path, index)] = units
            for unit in units:
                key = cache_key(attack_name, args.model, unit)
                unique_units.setdefault(key, unit)

    results: dict[str, dict[str, Any]] = {}
    pending: dict[str, str] = {}
    for key, unit in unique_units.items():
        cached = cache.get(key)
        if cached and cached.get("ok") and cached.get("back_translation"):
            results[key] = cached
        else:
            pending[key] = unit

    print(
        f"mode={args.mode} rows={sum(map(len, source_rows.values()))} "
        f"units={sum(map(len, units_by_row.values()))} unique={len(unique_units)} "
        f"cached={len(results)} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                translate_one,
                client,
                cache,
                attack_name=attack_name,
                model=args.model,
                prompt_template=prompt_template,
                text=unit,
                temperature=args.temperature,
                max_retries=args.max_retries,
                timeout=args.timeout,
                max_output_tokens=args.max_output_tokens,
            ): key
            for key, unit in pending.items()
        }
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            results[key] = future.result()
            if done % 50 == 0 or done == len(futures):
                failures = sum(not row.get("ok") for row in results.values())
                print(f"completed={done}/{len(futures)} failures={failures}", flush=True)

    summary: dict[str, Any] = {
        "mode": args.mode,
        "attack_name": attack_name,
        "model": args.model,
        "files": {},
    }
    for input_path, rows in source_rows.items():
        output_rows = []
        for index, source_row in enumerate(rows):
            units = units_by_row[(input_path, index)]
            unit_results = [
                results[cache_key(attack_name, args.model, unit)] for unit in units
            ]
            back_translations = [row["back_translation"] for row in unit_results]
            output_row = dict(source_row)
            if args.mode == "sentence":
                output_row["text_bt_sentence"] = join_sentences(back_translations)
                output_row["bt_sentence_intermediate_translations"] = [
                    row["intermediate_translation"] for row in unit_results
                ]
                output_row["bt_sentence_back_translations"] = back_translations
            else:
                output_row["text_bt_document"] = back_translations[0]
                output_row["bt_document_intermediate_translation"] = unit_results[0][
                    "intermediate_translation"
                ]
                output_row["bt_document_back_translation"] = back_translations[0]
            output_row["_backtranslation_meta"] = {
                "attack_name": attack_name,
                "model_name": args.model,
                "temperature": args.temperature,
                "unit_count": len(units),
                "failed_units": sum(not row.get("ok") for row in unit_results),
            }
            output_rows.append(output_row)

        destination = output_path(input_path, args.output_dir, args.path_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for row in output_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["files"][str(destination)] = {
            "rows": len(output_rows),
            "failed_rows": sum(
                bool(row["_backtranslation_meta"]["failed_units"])
                for row in output_rows
            ),
            "failed_units": sum(
                row["_backtranslation_meta"]["failed_units"] for row in output_rows
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
