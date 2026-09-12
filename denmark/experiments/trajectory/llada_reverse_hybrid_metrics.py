"""Pure statistical helpers for the LLaDA reverse-hybrid diagnostic."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def bootstrap_mean_ci(
    values: Iterable[float], *, n_bootstrap: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling independent prompt/block units."""
    x = np.asarray(list(values), dtype=float)
    if x.ndim != 1 or len(x) < 1:
        raise ValueError("bootstrap_mean_ci requires at least one scalar")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for offset in range(0, n_bootstrap, 1_000):
        width = min(1_000, n_bootstrap - offset)
        indices = rng.integers(0, len(x), size=(width, len(x)))
        means[offset : offset + width] = x[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_block_values(
    cumulative: Iterable[float],
    endpoint: Iterable[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict:
    cumulative_array = np.asarray(list(cumulative), dtype=float)
    endpoint_array = np.asarray(list(endpoint), dtype=float)
    if cumulative_array.shape != endpoint_array.shape or cumulative_array.ndim != 1:
        raise ValueError("cumulative and endpoint values must be paired vectors")
    if len(cumulative_array) < 1:
        raise ValueError("no blocks to summarize")
    gap = endpoint_array - cumulative_array
    return {
        "blocks": int(len(cumulative_array)),
        "cumulative_delta_mean": float(cumulative_array.mean()),
        "cumulative_delta_median": float(np.median(cumulative_array)),
        "cumulative_delta_ci95": list(bootstrap_mean_ci(
            cumulative_array, n_bootstrap=n_bootstrap, seed=seed + 1
        )),
        "cumulative_delta_positive_block_ratio": float(np.mean(cumulative_array > 0)),
        "endpoint_uplift_mean": float(endpoint_array.mean()),
        "endpoint_uplift_median": float(np.median(endpoint_array)),
        "endpoint_uplift_ci95": list(bootstrap_mean_ci(
            endpoint_array, n_bootstrap=n_bootstrap, seed=seed + 2
        )),
        "endpoint_uplift_positive_block_ratio": float(np.mean(endpoint_array > 0)),
        "closure_gap_mean": float(gap.mean()),
        "closure_gap_median": float(np.median(gap)),
        "closure_gap_ci95": list(bootstrap_mean_ci(
            gap, n_bootstrap=n_bootstrap, seed=seed + 3
        )),
        "closure_ci_contains_zero": bool(
            bootstrap_mean_ci(gap, n_bootstrap=n_bootstrap, seed=seed + 3)[0]
            <= 0
            <= bootstrap_mean_ci(gap, n_bootstrap=n_bootstrap, seed=seed + 3)[1]
        ),
    }


def paired_bootstrap_mean_ci(
    left: Iterable[float],
    right: Iterable[float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("paired samples must have identical shape")
    return bootstrap_mean_ci(
        left_array - right_array, n_bootstrap=n_bootstrap, seed=seed
    )

