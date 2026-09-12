"""Shared sentence splitting and paper GPT-attack prompt definitions."""

from __future__ import annotations

import math
import random
import re
import time
from typing import Any


SYSTEM_PROMPT = "You are a helpful assistant that rewrites text while preserving its meaning."
GPT_SENTENCE_ATTACKS: dict[str, dict[str, Any]] = {
    "rewrite": {
        "prompt": (
            "Rewrite the following sentence in a natural and fluent way.\n"
            "Preserve the original meaning, factual content, named entities, numbers, and technical terms.\n"
            "Change the sentence structure, wording, and phrasing as much as possible.\n"
            "Keep the length roughly similar.\n"
            "Do not add explanations.\n"
            "Return only the rewritten sentence.\n\nSentence:\n{sentence}"
        ),
        "temperature": 0.7,
    },
    "compress_60_70": {
        "prompt": (
            "Rewrite the following sentence in a shorter and more concise way.\n"
            "Preserve the original meaning, factual content, named entities, numbers, technical terms, and essential qualifiers.\n"
            "Keep the result as exactly one sentence; do not split or merge sentences.\n"
            "The original sentence contains approximately {original_word_count} words.\n"
            "Target approximately 60% to 70% of the original length: about {target_min_words} to {target_max_words} words.\n"
            "Remove only unnecessary modifiers, redundancy, and verbose phrasing.\n"
            "Do not change the core claim or add new information.\n"
            "Do not add explanations.\n"
            "Return only the rewritten sentence.\n\nSentence:\n{sentence}"
        ),
        "temperature": 0.7,
    },
    "expand": {
        "prompt": (
            "Rewrite the following sentence in a more detailed and explicit way.\n"
            "Preserve the original meaning, factual content, named entities, numbers, and technical terms.\n"
            "You may clarify implicit relationships and use richer phrasing, but do not add new facts, examples, numbers, named entities, or claims.\n"
            "Change the sentence structure, wording, and phrasing as much as possible.\n"
            "Do not add explanations.\n"
            "Return only the rewritten sentence.\n\nSentence:\n{sentence}"
        ),
        "temperature": 0.7,
    },
}

ABBREVIATIONS = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "No.",
    "vs.", "etc.", "e.g.", "i.e.", "U.S.", "U.K.", "a.m.", "p.m.",
)


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    normalized = re.sub(r"\s+", " ", text.strip())
    placeholders: dict[str, str] = {}
    protected = normalized
    for index, abbreviation in enumerate(ABBREVIATIONS):
        placeholder = f"__ABBR_{index}__"
        protected = protected.replace(abbreviation, placeholder)
        placeholders[placeholder] = abbreviation
    parts = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", protected)
    sentences = []
    for part in parts:
        restored = part.strip()
        for placeholder, abbreviation in placeholders.items():
            restored = restored.replace(placeholder, abbreviation)
        if restored:
            sentences.append(restored)
    return sentences or [normalized]


def join_sentences(sentences: list[str]) -> str:
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip())


def sentence_attack_spec(name: str, sentence: str) -> tuple[str, float, tuple[int, int] | None]:
    spec = GPT_SENTENCE_ATTACKS[name]
    word_count = max(1, len(sentence.split()))
    lower = max(1, math.ceil(word_count * 0.60))
    upper = max(lower, math.floor(word_count * 0.70))
    prompt = spec["prompt"].format(
        sentence=sentence,
        original_word_count=word_count,
        target_min_words=lower,
        target_max_words=upper,
    )
    bounds = (lower, upper) if name == "compress_60_70" else None
    return prompt, float(spec["temperature"]), bounds


def sleep_with_backoff(attempt: int) -> None:
    delay = min(30.0, 1.5 * (2**attempt)) * random.uniform(0.8, 1.2)
    time.sleep(delay)


__all__ = [
    "GPT_SENTENCE_ATTACKS",
    "SYSTEM_PROMPT",
    "join_sentences",
    "sentence_attack_spec",
    "sleep_with_backoff",
    "split_sentences",
]
