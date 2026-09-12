"""Paper-facing calibrated unit-size scan detector.

For each candidate unit size, DenMark computes a raw semantic score and an
empirical right-tail p-value against an independent calibration pool. The
minimum per-size p-value is Bonferroni-corrected over the scanned sizes. This
is the detector reported in the paper; experimental scan variants are omitted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from denmark.core.scoring import block_decode, clean_text
from denmark.core.model import encode_texts


EPS = 1e-12


def empirical_right_tail_p(score: float, null_scores: Iterable[float]) -> float:
    """Finite-sample empirical right-tail p-value with add-one correction."""
    null = np.asarray(list(null_scores), dtype=float)
    if null.size == 0:
        return 1.0
    return float((1.0 + np.sum(null >= float(score))) / (null.size + 1.0))


def empirical_right_tail_p_loo(score: float, null_scores: Iterable[float]) -> float:
    """Right-tail p-value for a sample contained in its calibration pool.

    This compatibility path removes exactly the evaluated occurrence and keeps
    every other tied score. Paper experiments should pass a disjoint pool.
    """
    null = np.asarray(list(null_scores), dtype=float)
    if null.size <= 1:
        return 1.0
    count_including_self = int(np.sum(null >= float(score)))
    if count_including_self < 1:
        raise ValueError("LOO calibration item is not represented in its pool")
    return float(count_including_self / null.size)


@dataclass(frozen=True)
class RawDetectionResult:
    raw: float
    best_block_size: int | None = None
    n_valid_blocks: int = 0


class RawSemanticDetector:
    """Compute fixed-size and unit-size-grid semantic watermark scores."""

    def __init__(
        self,
        enc,
        enc_tok,
        llada_tok,
        dirs: torch.Tensor,
        signs: torch.Tensor,
        device: str,
        gen_length: int = 300,
        fixed_block_size: int = 25,
        scan_block_sizes: Iterable[int] = range(12, 38),
    ):
        if dirs.ndim != 3:
            raise ValueError("dirs must have shape [units, channels, embedding_dim]")
        if tuple(signs.shape) != tuple(dirs.shape[:2]):
            raise ValueError("signs must match the first two dimensions of dirs")

        self.enc = enc
        self.enc_tok = enc_tok
        self.llada_tok = llada_tok
        self.dirs = dirs.detach().cpu()
        self.signs = signs.detach().cpu()
        self.device = device
        self.gen_length = int(gen_length)
        self.fixed_block_size = int(fixed_block_size)
        self.scan_block_sizes = tuple(int(size) for size in scan_block_sizes)
        if not self.scan_block_sizes or any(size <= 0 for size in self.scan_block_sizes):
            raise ValueError("scan_block_sizes must contain positive integers")

        self.num_blocks = int(self.dirs.shape[0])
        self._raw_cache: dict[tuple[str, tuple[int, ...], int | None], RawDetectionResult] = {}
        self._scan_grid_cache: dict[tuple[int, ...], dict[int, RawDetectionResult]] = {}

    def token_ids(
        self,
        text: str | None = None,
        token_ids: list[int] | None = None,
    ) -> list[int]:
        if token_ids is not None:
            return [int(token_id) for token_id in token_ids[: self.gen_length]]
        return self.llada_tok(
            clean_text(text or ""),
            add_special_tokens=False,
        )["input_ids"][: self.gen_length]

    def _score_segments(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> RawDetectionResult:
        texts: list[str] = []
        owners: list[int] = []
        for block_idx in range(self.num_blocks):
            start = block_idx * block_size
            end = min(start + block_size, self.gen_length)
            if start >= len(token_ids):
                break
            text = block_decode(token_ids, start, end, self.llada_tok)
            if text:
                texts.append(text)
                owners.append(block_idx)

        if not texts:
            return RawDetectionResult(0.0, block_size, 0)

        embeddings = encode_texts(
            texts,
            self.enc,
            self.enc_tok,
            self.device,
            batch_sz=32,
        )
        block_scores: list[float] = []
        for embedding, block_idx in zip(embeddings, owners):
            projections = (embedding @ self.dirs[block_idx].T).numpy()
            signed = projections * self.signs[block_idx].numpy()
            block_scores.append(float(np.mean(signed)))

        return RawDetectionResult(
            raw=float(np.mean(block_scores)),
            best_block_size=block_size,
            n_valid_blocks=len(block_scores),
        )

    def fixed_mean_raw(self, token_ids: list[int]) -> RawDetectionResult:
        return self._score_segments(token_ids, self.fixed_block_size)

    def scan_raw_for_block_size(
        self,
        token_ids: list[int],
        block_size: int,
    ) -> RawDetectionResult:
        return self._score_segments(token_ids, int(block_size))

    def scan_raw_grid(self, token_ids: list[int]) -> dict[int, RawDetectionResult]:
        key = tuple(token_ids)
        cached = self._scan_grid_cache.get(key)
        if cached is not None:
            return cached
        result = {
            size: self.scan_raw_for_block_size(token_ids, size)
            for size in self.scan_block_sizes
        }
        self._scan_grid_cache[key] = result
        return result

    def raw(
        self,
        detector: str,
        text: str | None = None,
        token_ids: list[int] | None = None,
        block_size: int | None = None,
    ) -> RawDetectionResult:
        ids = self.token_ids(text, token_ids)
        key = (detector, tuple(ids), block_size)
        cached = self._raw_cache.get(key)
        if cached is not None:
            return cached

        if detector in {"fixed25_mean", "fixed_unit"}:
            result = self.fixed_mean_raw(ids)
        elif detector == "scan_raw":
            if block_size is None:
                raise ValueError("block_size is required for scan_raw")
            size = int(block_size)
            result = self.scan_raw_grid(ids).get(size)
            if result is None:
                result = self.scan_raw_for_block_size(ids, size)
        else:
            raise ValueError(f"unknown paper detector: {detector}")

        self._raw_cache[key] = result
        return result


class CalibratedDetectorSuite:
    """Fit and apply the paper's empirical per-size scan calibration."""

    _SUPPORTED = {"fixed25_mean", "fixed_unit", "calibrated_scan"}

    def __init__(
        self,
        raw_detector: RawSemanticDetector,
        negative_items: list[dict],
        use_length_buckets: bool = False,
        min_bucket_negatives: int = 30,
        detectors: Iterable[str] | None = None,
        calibration_items: list[dict] | None = None,
    ):
        del min_bucket_negatives
        if use_length_buckets:
            raise ValueError("length-bucket calibration is not part of the paper detector")
        self.raw_detector = raw_detector
        self.negative_items = list(negative_items)
        self.calibration_items = (
            self.negative_items if calibration_items is None else list(calibration_items)
        )
        if not self.calibration_items:
            raise ValueError("calibration_items must be non-empty")

        self._calibration_item_ids = {id(item) for item in self.calibration_items}
        self.detectors = set(detectors or ("calibrated_scan",))
        unsupported = self.detectors - self._SUPPORTED
        if unsupported:
            raise ValueError(f"unsupported paper detector(s): {sorted(unsupported)}")

        self.neg_raw: dict[str, object] = {}
        self.neg_scores: dict[str, list[float]] = {}
        self._fit()

    @staticmethod
    def _item_tokens(item: dict) -> list[int] | None:
        return item.get("token_ids")

    @staticmethod
    def _item_text(item: dict) -> str:
        return str(item.get("text") or "")

    def _raw_score(
        self,
        item: dict,
        detector: str,
        block_size: int | None = None,
    ) -> float:
        return self.raw_detector.raw(
            detector,
            self._item_text(item),
            self._item_tokens(item),
            block_size=block_size,
        ).raw

    def _fit(self) -> None:
        if self.detectors & {"fixed25_mean", "fixed_unit"}:
            self.neg_raw["fixed_unit"] = [
                self._raw_score(item, "fixed_unit")
                for item in self.calibration_items
            ]

        if "calibrated_scan" in self.detectors:
            self.neg_raw["scan_by_block_size"] = {
                size: [
                    self._raw_score(item, "scan_raw", block_size=size)
                    for item in self.calibration_items
                ]
                for size in self.raw_detector.scan_block_sizes
            }

        self.neg_scores = {
            detector: [self.score_item(item, detector)["score"] for item in self.negative_items]
            for detector in self.detectors
        }

    def _empirical_p(
        self,
        item: dict,
        raw: float,
        pool: Iterable[float],
    ) -> float:
        if id(item) in self._calibration_item_ids:
            return empirical_right_tail_p_loo(raw, pool)
        return empirical_right_tail_p(raw, pool)

    def _scan_p(self, item: dict) -> tuple[float, float, int]:
        pools = self.neg_raw["scan_by_block_size"]
        if not isinstance(pools, dict):
            raise RuntimeError("scan calibration was not fitted")

        best_raw = -float("inf")
        best_p = 1.0
        best_size = -1
        for size in self.raw_detector.scan_block_sizes:
            raw = self._raw_score(item, "scan_raw", block_size=size)
            p_value = self._empirical_p(item, raw, pools[size])
            if p_value < best_p or best_size < 0:
                best_raw = raw
                best_p = p_value
                best_size = size

        corrected = min(1.0, len(self.raw_detector.scan_block_sizes) * best_p)
        return best_raw, corrected, best_size

    def score_item(
        self,
        item: dict,
        detector: str,
        use_buckets: bool | None = None,
        force_bucket: str | None = None,
    ) -> dict:
        if use_buckets:
            raise ValueError("length-bucket calibration is not part of the paper detector")
        if force_bucket is not None:
            raise ValueError("force_bucket is not part of the paper detector")

        if detector in {"fixed25_mean", "fixed_unit"}:
            raw = self._raw_score(item, "fixed_unit")
            pool = self.neg_raw["fixed_unit"]
            if not isinstance(pool, list):
                raise RuntimeError("fixed-unit calibration was not fitted")
            p_value = self._empirical_p(item, raw, pool)
            return {
                "detector": detector,
                "raw": raw,
                "p_value": p_value,
                "score": -math.log(max(p_value, EPS)),
            }

        if detector == "calibrated_scan":
            raw, p_value, best_size = self._scan_p(item)
            return {
                "detector": detector,
                "raw": raw,
                "p_value": p_value,
                "score": -math.log(max(p_value, EPS)),
                "best_block_size": best_size,
            }

        raise ValueError(f"unknown paper detector: {detector}")


__all__ = [
    "EPS",
    "CalibratedDetectorSuite",
    "RawDetectionResult",
    "RawSemanticDetector",
    "empirical_right_tail_p",
    "empirical_right_tail_p_loo",
]
