"""Small, dependency-light helpers for the UMR baseline.

The generation algorithm remains in the upstream UMR checkout.  This module
only implements result scoring and validation so that UMR can share this
repository's JSONL and empirical-ROC evaluation pipeline.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, Protocol


class BitmapLike(Protocol):
    def get_bit(self, x: int, y: int) -> bool: ...


def calculate_z_score(hit_count: int, total_tokens: int, gamma: float = 0.5) -> float:
    """Return UMR's token-hit z-score under a Bernoulli(gamma) null."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be strictly between 0 and 1")
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if not 0 <= hit_count <= total_tokens:
        raise ValueError("hit_count must be between 0 and total_tokens")
    if total_tokens == 0:
        return 0.0
    expected = total_tokens * gamma
    variance = total_tokens * gamma * (1.0 - gamma)
    return (hit_count - expected) / math.sqrt(variance)


def context_seed(previous_token: int, key: int) -> int:
    digest = hashlib.sha256(f"{int(previous_token)}{int(key)}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


def score_umr_tokens(
    token_ids: Iterable[int],
    bitmap: BitmapLike,
    *,
    watermark_str: str = "1001",
    key: int = 42,
    ratio: float = 0.5,
    previous_token: int | None = None,
    stop_token_ids: set[int] | None = None,
) -> dict:
    """Score completion tokens using the exact upstream UMR bitmap semantics.

    The persisted bitmap already encodes the target bit selected by each
    previous-token context.  Consequently a set bitmap bit is a token-level
    watermark hit.  We invert that hit for target-bit 0 only when recovering
    the underlying base partition and message bits.
    """
    ids = [int(token_id) for token_id in token_ids]
    if not watermark_str or any(bit not in "01" for bit in watermark_str):
        raise ValueError("watermark_str must be a non-empty binary string")
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be strictly between 0 and 1")
    if previous_token is None:
        if len(ids) < 2:
            return _empty_score(watermark_str)
        previous_token = ids.pop(0)

    stops = stop_token_ids or set()
    votes = [[0, 0] for _ in watermark_str]
    hits = 0
    total = 0
    prev = int(previous_token)
    for current in ids:
        if current in stops:
            break
        seed = context_seed(prev, key)
        bit_index = seed % len(watermark_str)
        target_bit = watermark_str[bit_index]
        satisfied = bool(bitmap.get_bit(prev, current))
        base_green = satisfied if target_bit == "1" else not satisfied
        votes[bit_index][1 if base_green else 0] += 1
        hits += int(satisfied)
        total += 1
        prev = current

    recovered = []
    confidences = []
    matches = 0
    for index, (vote0, vote1) in enumerate(votes):
        count = vote0 + vote1
        if count == 0:
            bit = "?"
            confidence = 0.5
        elif vote1 > vote0:
            bit = "1"
            confidence = vote1 / count
        elif vote0 > vote1:
            bit = "0"
            confidence = vote0 / count
        else:
            bit = "-"
            confidence = 0.5
        recovered.append(bit)
        confidences.append(confidence)
        matches += int(bit == watermark_str[index])

    z_score = calculate_z_score(hits, total, gamma=ratio)
    return {
        "watermark_acc": matches / len(watermark_str),
        "watermark_conf": sum(confidences) / len(confidences),
        "recovered_str": "".join(recovered),
        "token_hit_count": hits,
        "total_gen_tokens": total,
        "token_hit_rate": hits / total if total else 0.0,
        "z_score": z_score,
        "votes": votes,
    }


def _empty_score(watermark_str: str) -> dict:
    return {
        "watermark_acc": 0.0,
        "watermark_conf": 0.5,
        "recovered_str": "?" * len(watermark_str),
        "token_hit_count": 0,
        "total_gen_tokens": 0,
        "token_hit_rate": 0.0,
        "z_score": 0.0,
        "votes": [[0, 0] for _ in watermark_str],
    }
