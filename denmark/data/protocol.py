"""Shared, lightweight evaluation protocol helpers.

These helpers deliberately avoid model or tokenizer imports so prompt identity,
quality filtering, and calibration splits can be unit-tested on a login/local
CPU without loading an experiment model.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable, Sequence


USER_SPANS = (
    ("<|start_header_id|>user<|end_header_id|>", "<|eot_id|>"),
    ("<|im_start|>user", "<|im_end|>"),
    ("<start_of_turn>user", "<end_of_turn>"),
)


def normalize_whitespace(value: object) -> str:
    return " ".join(str(value or "").replace("NEWLINE_CHAR", " ").split())


def repetition_ngram(text: str, n: int = 4) -> float:
    words = normalize_whitespace(text).split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1) / len(grams)


def extract_user_prompt(prompt_full: object) -> str:
    """Extract the user span from common serialized chat templates.

    Returning an empty string is intentional: callers must not silently hash a
    serialized prompt they failed to parse.
    """
    value = str(prompt_full or "")
    if not value.strip():
        return ""
    for start_marker, end_marker in USER_SPANS:
        if start_marker not in value:
            continue
        user_span = value.split(start_marker, 1)[1]
        if end_marker not in user_span:
            assistant_markers = (
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|im_start|>assistant",
                "<start_of_turn>model",
            )
            if any(marker in user_span for marker in assistant_markers):
                return ""
            # Some retained clean rows store a prompt truncated inside the
            # user turn.  The user span is still unambiguous when no assistant
            # marker follows it.
            return user_span.strip()
        return user_span.split(end_marker, 1)[0].strip()
    # A plain prompt is safe only when it does not contain a known assistant
    # delimiter.  This prevents hashing a full serialized conversation.
    assistant_markers = (
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|im_start|>assistant",
        "<start_of_turn>model",
    )
    if any(marker in value for marker in assistant_markers):
        return ""
    return value.strip()


def source_prompt_seed(row: dict[str, Any]) -> str:
    raw_prompt = row.get("raw_prompt")
    if isinstance(raw_prompt, str) and raw_prompt.strip():
        return raw_prompt.strip()
    input_text = str(row.get("input") or "").strip()
    context = str(row.get("context") or "").strip()
    return f"{context}\n\n{input_text}".strip() if context else input_text


def resolve_prompt_seed(
    row: dict[str, Any],
    *,
    source_rows: Sequence[dict[str, Any]] | None = None,
    row_index: int = 0,
) -> tuple[str, str]:
    """Resolve a row's actual prompt without trusting a reindexed source_idx.

    The row itself is authoritative.  Dataset lookup is only a last resort for
    legacy rows that contain no prompt fields at all.
    """
    explicit = row.get("prompt_seed_text")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip(), "prompt_seed_text"

    prompt_full = row.get("prompt_full")
    if isinstance(prompt_full, str) and prompt_full.strip():
        extracted = extract_user_prompt(prompt_full)
        if not extracted:
            raise ValueError(f"could not parse prompt_full at row {row_index}")
        return extracted, "prompt_full_user_span"

    for field in ("raw_prompt", "prompt_input", "input", "prompt"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            context = str(row.get("prompt_context") or row.get("context") or "").strip()
            prompt = f"{context}\n\n{value}".strip() if context and field != "raw_prompt" else value.strip()
            return prompt, field

    if source_rows is None:
        raise ValueError(f"row {row_index} has no recoverable prompt")
    source_index = int(row.get("source_idx", row_index))
    if not 0 <= source_index < len(source_rows):
        raise ValueError(f"source_idx={source_index} is outside source rows")
    fallback = source_prompt_seed(source_rows[source_index])
    if not fallback:
        raise ValueError(f"empty source prompt at row {row_index}")
    return fallback, "source_idx_fallback"


def canonical_prompt_key(
    prompt_seed: str,
    source_rows: Sequence[dict[str, Any]] | None = None,
) -> str:
    """Map template variants of one WaterBench question to the same key."""
    prompt = normalize_whitespace(prompt_seed)
    if not prompt:
        raise ValueError("prompt seed cannot be empty")
    if source_rows:
        candidates = []
        for source_row in source_rows:
            question = normalize_whitespace(source_row.get("input"))
            if question and question in prompt:
                candidates.append(question)
        if candidates:
            longest = max(map(len, candidates))
            best = sorted({question for question in candidates if len(question) == longest})
            if len(best) != 1:
                raise ValueError(f"ambiguous source prompt match: {best}")
            return f"waterbench-input:{best[0]}"
    return f"prompt:{prompt}"


def quality_and_dedup_indices(
    items: Sequence[dict[str, Any]],
    *,
    min_token_len: int,
    max_rep4: float,
    deduplicate_prompt: bool,
) -> tuple[list[int], dict[str, Any]]:
    if min_token_len < 0 or max_rep4 < 0:
        raise ValueError("quality thresholds must be non-negative")
    quality_indices = [
        index
        for index, item in enumerate(items)
        if int(item["token_len"]) >= min_token_len
        and float(item["rep4"]) <= max_rep4
    ]
    selected = []
    seen = set()
    duplicates = 0
    for index in quality_indices:
        key = str(items[index].get("prompt_key") or f"row:{index}")
        if deduplicate_prompt and key in seen:
            duplicates += 1
            continue
        seen.add(key)
        selected.append(index)
    return selected, {
        "n_input": len(items),
        "n_quality": len(quality_indices),
        "n_selected": len(selected),
        "n_removed_token_or_rep4": len(items) - len(quality_indices),
        "n_removed_duplicate_prompt": duplicates,
        "min_token_len": min_token_len,
        "max_rep4": max_rep4,
        "deduplicate_prompt": deduplicate_prompt,
    }


def grouped_crossfit_folds(
    keys: Iterable[str],
    *,
    n_folds: int,
    seed: int,
) -> list[int]:
    """Create deterministic balanced folds while keeping equal keys together."""
    if n_folds < 2:
        raise ValueError("cross-fit evaluation requires at least two folds")
    key_list = [str(key) for key in keys]
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(key_list):
        groups.setdefault(key, []).append(index)
    if len(groups) < n_folds:
        raise ValueError("number of distinct prompt groups is smaller than n_folds")

    def digest(key: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{key}".encode("utf-8")).digest()

    fold_sizes = [0] * n_folds
    assignments = [-1] * len(key_list)
    # Large groups go first; the seeded digest gives deterministic tie-breaking.
    ordered = sorted(groups.items(), key=lambda pair: (-len(pair[1]), digest(pair[0])))
    for _, indices in ordered:
        fold = min(range(n_folds), key=lambda value: (fold_sizes[value], value))
        for index in indices:
            assignments[index] = fold
        fold_sizes[fold] += len(indices)
    if any(value < 0 for value in assignments):
        raise RuntimeError("incomplete cross-fit assignment")
    return assignments
