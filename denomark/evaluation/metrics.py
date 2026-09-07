"""Pure NumPy calibration and ROC utilities for DenoMark scan scores."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def score_matrix(
    records: Sequence[Mapping[str, object]],
    unit_sizes: Sequence[int],
    field: str = "raw_scores_by_unit_size",
) -> np.ndarray:
    """Extract a dense ``[num_records, num_unit_sizes]`` raw-score matrix."""
    rows: list[list[float]] = []
    for index, record in enumerate(records):
        values = record.get(field)
        if not isinstance(values, Mapping):
            raise ValueError(f"record {index} has no mapping field {field!r}")
        try:
            rows.append([float(values[str(size)]) for size in unit_sizes])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"record {index} is missing a finite raw score for one or more unit sizes"
            ) from exc
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(records), len(unit_sizes)):
        raise ValueError(f"unexpected score matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("raw scan scores must all be finite")
    return matrix


def calibrated_scan_scores(
    evaluation: np.ndarray,
    calibration: np.ndarray,
) -> np.ndarray:
    """Calibrate per-size scores, Bonferroni-correct, and return ``-log(p)``.

    ``evaluation`` and ``calibration`` must have one column per candidate unit
    size. Evaluation rows are assumed to be disjoint from the calibration pool,
    so no leave-one-out correction is applied.
    """
    evaluation = np.asarray(evaluation, dtype=np.float64)
    calibration = np.asarray(calibration, dtype=np.float64)
    if evaluation.ndim != 2 or calibration.ndim != 2:
        raise ValueError("evaluation and calibration scores must be two-dimensional")
    if evaluation.shape[1] != calibration.shape[1]:
        raise ValueError("evaluation and calibration must use the same unit-size grid")
    if calibration.shape[0] == 0 or calibration.shape[1] == 0:
        raise ValueError("calibration matrix must be non-empty")
    if not np.isfinite(evaluation).all() or not np.isfinite(calibration).all():
        raise ValueError("scan score matrices must contain only finite values")

    n_calibration = calibration.shape[0]
    p_values = np.empty_like(evaluation)
    for column in range(calibration.shape[1]):
        sorted_null = np.sort(calibration[:, column])
        count_ge = n_calibration - np.searchsorted(
            sorted_null,
            evaluation[:, column],
            side="left",
        )
        p_values[:, column] = (1.0 + count_ge) / (n_calibration + 1.0)
    corrected = np.minimum(1.0, calibration.shape[1] * np.min(p_values, axis=1))
    return -np.log(corrected)


def rank_auc(positive: Iterable[float], negative: Iterable[float]) -> float:
    """Mann-Whitney AUC with half credit for tied scores."""
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.sort(np.asarray(list(negative), dtype=np.float64))
    if pos.size == 0 or neg.size == 0:
        raise ValueError("positive and negative score arrays must be non-empty")
    left = np.searchsorted(neg, pos, side="left")
    right = np.searchsorted(neg, pos, side="right")
    return float(np.sum(left + 0.5 * (right - left)) / (pos.size * neg.size))


def roc_interpolated_tpr(
    positive: Iterable[float],
    negative: Iterable[float],
    target_fpr: float,
) -> float:
    """Return linearly interpolated empirical-ROC TPR at an exact FPR."""
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("positive and negative score arrays must be non-empty")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must lie in [0, 1]")

    thresholds = np.unique(np.concatenate([pos, neg]))[::-1]
    fpr = np.concatenate(
        ([0.0], np.asarray([(neg >= threshold).mean() for threshold in thresholds]), [1.0])
    )
    tpr = np.concatenate(
        ([0.0], np.asarray([(pos >= threshold).mean() for threshold in thresholds]), [1.0])
    )
    # Duplicate FPR values form vertical ROC segments. Keep the highest TPR at
    # each FPR before interpolating horizontally.
    unique_fpr = np.unique(fpr)
    upper_tpr = np.asarray([tpr[fpr == value].max() for value in unique_fpr])
    return float(np.interp(target_fpr, unique_fpr, upper_tpr))


def summarize_roc(
    positive: Iterable[float],
    negative: Iterable[float],
    fprs: Sequence[float] = (0.005, 0.01, 0.05),
) -> dict[str, object]:
    """Summarize rank AUC and interpolated TPRs."""
    pos = list(positive)
    neg = list(negative)
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "tpr": {str(fpr): roc_interpolated_tpr(pos, neg, fpr) for fpr in fprs},
        "auc": rank_auc(pos, neg),
    }


__all__ = [
    "calibrated_scan_scores",
    "rank_auc",
    "roc_interpolated_tpr",
    "score_matrix",
    "summarize_roc",
]

