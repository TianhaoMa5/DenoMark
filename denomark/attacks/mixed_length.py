#!/usr/bin/env python3
"""Build a sentence-wise mixed compression/expansion attack from GPT caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from denomark.attacks.common import join_sentences, split_sentences


MODEL_NAME = "openai/gpt-4o-mini"
COMPRESS_ATTACK = "gpt_compress_60_70_sentence"
EXPAND_ATTACK = "gpt_expand_sentence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compress-cache", nargs="+", type=Path, required=True)
    parser.add_argument("--expand-cache", nargs="+", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--output-field")
    parser.add_argument(
        "--assignment",
        choices=("alternating", "random_runs", "half_split"),
        default="alternating",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--switch-probability", type=float, default=0.35)
    return parser.parse_args()


def load_cache(path: Path, attack_name: str) -> dict[str, str]:
    rewrites: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("ok")
                and row.get("attack_name") == attack_name
                and row.get("model_name") == MODEL_NAME
            ):
                sentence = row.get("sentence")
                rewrite = row.get("rewrite")
                if isinstance(sentence, str) and isinstance(rewrite, str) and rewrite.strip():
                    rewrites[sentence] = rewrite.strip()
    return rewrites


def load_caches(paths: list[Path], attack_name: str) -> dict[str, str]:
    rewrites: dict[str, str] = {}
    for path in paths:
        rewrites.update(load_cache(path, attack_name))
    return rewrites


def word_count(text: str) -> int:
    return len(text.split())


def assignment_plan(
    sentence_count: int,
    *,
    assignment: str,
    seed: int,
    switch_probability: float,
    row_key: str,
) -> list[str]:
    if assignment == "alternating":
        return [
            COMPRESS_ATTACK if sentence_idx % 2 == 0 else EXPAND_ATTACK
            for sentence_idx in range(sentence_count)
        ]

    if assignment == "half_split":
        split_index = max(1, sentence_count // 2)
        return [
            COMPRESS_ATTACK if sentence_idx < split_index else EXPAND_ATTACK
            for sentence_idx in range(sentence_count)
        ]

    digest = hashlib.sha256(f"{seed}:{row_key}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    current = rng.choice((COMPRESS_ATTACK, EXPAND_ATTACK))
    plan: list[str] = []
    for sentence_idx in range(sentence_count):
        if sentence_idx and rng.random() < switch_probability:
            current = EXPAND_ATTACK if current == COMPRESS_ATTACK else COMPRESS_ATTACK
        plan.append(current)

    # Ensure that a multi-sentence document actually contains both local
    # transformations while preserving random contiguous runs elsewhere.
    if sentence_count > 1 and len(set(plan)) == 1:
        flip_idx = rng.randrange(1, sentence_count)
        plan[flip_idx] = (
            EXPAND_ATTACK if plan[flip_idx] == COMPRESS_ATTACK else COMPRESS_ATTACK
        )
    return plan


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.switch_probability <= 1.0:
        raise ValueError("--switch-probability must be between 0 and 1")
    attack_names = {
        "alternating": "gpt_mixed_length_sentence",
        "random_runs": "gpt_mixed_length_sentence_random_runs",
        "half_split": "gpt_mixed_length_sentence_half_split",
    }
    output_fields = {
        "alternating": "text_gpt_mixed_length_sentence",
        "random_runs": "text_gpt_mixed_length_sentence_random_runs",
        "half_split": "text_gpt_mixed_length_sentence_half_split",
    }
    attack_name = attack_names[args.assignment]
    output_field = args.output_field or output_fields[args.assignment]
    compress = load_caches(args.compress_cache, COMPRESS_ATTACK)
    expand = load_caches(args.expand_cache, EXPAND_ATTACK)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "attack": attack_name,
        "model": MODEL_NAME,
        "assignment": args.assignment,
        "seed": args.seed,
        "switch_probability": (
            args.switch_probability if args.assignment == "random_runs" else None
        ),
        "compress_cache": [str(path.resolve()) for path in args.compress_cache],
        "expand_cache": [str(path.resolve()) for path in args.expand_cache],
        "files": [],
    }

    for input_path in args.inputs:
        output_path = args.output_dir / input_path.name
        mode_counts: Counter[str] = Counter()
        original_words = attacked_words = 0
        sentence_ratios: list[float] = []
        rows = 0
        with input_path.open(encoding="utf-8") as source, output_path.open(
            "w", encoding="utf-8"
        ) as output:
            for line_no, line in enumerate(source):
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get(args.text_field)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{input_path}:{line_no}: missing {args.text_field}")

                sentences = split_sentences(text)
                row_key = str(
                    row.get("sample_id")
                    or row.get("prompt_idx")
                    or row.get("source_row_idx")
                    or line_no
                )
                plan = assignment_plan(
                    len(sentences),
                    assignment=args.assignment,
                    seed=args.seed,
                    switch_probability=args.switch_probability,
                    row_key=f"{input_path.resolve()}:{row_key}",
                )
                rewritten: list[str] = []
                sentence_plan: list[dict[str, Any]] = []
                for sentence_idx, sentence in enumerate(sentences):
                    requested = plan[sentence_idx]
                    selected = requested
                    cache = compress if requested == COMPRESS_ATTACK else expand
                    replacement = cache.get(sentence)
                    fallback = None
                    if replacement is None:
                        alternate = EXPAND_ATTACK if requested == COMPRESS_ATTACK else COMPRESS_ATTACK
                        alternate_cache = expand if alternate == EXPAND_ATTACK else compress
                        replacement = alternate_cache.get(sentence)
                        selected = alternate
                        fallback = f"missing_{requested}_used_{alternate}"
                    if replacement is None:
                        raise KeyError(
                            f"No cached GPT rewrite for {input_path}:{line_no}:{sentence_idx}"
                        )

                    source_words = max(1, word_count(sentence))
                    rewritten_words = word_count(replacement)
                    ratio = rewritten_words / source_words
                    sentence_ratios.append(ratio)
                    original_words += source_words
                    attacked_words += rewritten_words
                    mode_counts[selected] += 1
                    if fallback:
                        mode_counts["fallback"] += 1
                    rewritten.append(replacement)
                    sentence_plan.append(
                        {
                            "sentence_idx": sentence_idx,
                            "requested": requested,
                            "selected": selected,
                            "original_words": source_words,
                            "attacked_words": rewritten_words,
                            "length_ratio": ratio,
                            "fallback": fallback,
                        }
                    )

                attacked_text = join_sentences(rewritten)
                previous_token_len = row.get("token_len")
                row[output_field] = attacked_text
                row["attack_text"] = attacked_text
                row["text"] = attacked_text
                for stale_field in (
                    "token_ids",
                    "generated_token_ids",
                    "completion_token_ids",
                    "watermarked_token_ids",
                    "input_ids",
                    "token_len",
                ):
                    row.pop(stale_field, None)
                row["_mixed_length_attack_meta"] = {
                    "attack_name": attack_name,
                    "model": MODEL_NAME,
                    "source_field": args.text_field,
                    "output_field": output_field,
                    "assignment": args.assignment,
                    "seed": args.seed,
                    "switch_probability": (
                        args.switch_probability
                        if args.assignment == "random_runs"
                        else None
                    ),
                    "sentence_count": len(sentences),
                    "pre_attack_token_len": previous_token_len,
                    "sentence_plan": sentence_plan,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1

        summary["files"].append(
            {
                "input": str(input_path.resolve()),
                "output": str(output_path.resolve()),
                "rows": rows,
                "sentences": sum(mode_counts[a] for a in (COMPRESS_ATTACK, EXPAND_ATTACK)),
                "compress_sentences": mode_counts[COMPRESS_ATTACK],
                "expand_sentences": mode_counts[EXPAND_ATTACK],
                "fallbacks": mode_counts["fallback"],
                "document_word_ratio": attacked_words / max(1, original_words),
                "sentence_ratio_min": min(sentence_ratios),
                "sentence_ratio_max": max(sentence_ratios),
                "sentence_ratio_mean": sum(sentence_ratios) / len(sentence_ratios),
            }
        )

    (args.output_dir / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

