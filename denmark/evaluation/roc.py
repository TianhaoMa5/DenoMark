"""Small dependency-free ROC helpers shared by experiment evaluators."""

from __future__ import annotations


def roc_interpolated_tpr_at_fpr(
    pos: list[float],
    neg: list[float],
    fpr: float,
    *,
    smaller_is_positive: bool = False,
) -> float:
    """Linearly interpolate the ROC TPR at an exact target FPR.

    Empirical p-value detectors can assign one score to many examples. A
    deterministic threshold then has to include or exclude the whole tied
    group, which can make the achieved FPR much smaller than the requested
    value. ROC interpolation reports the standard randomized-tie operating
    point while leaving the detector scores and strict metrics unchanged.
    """
    if not pos or not neg:
        return float("nan")
    if not 0.0 <= fpr <= 1.0:
        raise ValueError(f"fpr must be in [0, 1], got {fpr}")

    direction = -1.0 if smaller_is_positive else 1.0
    pos_counts: dict[float, int] = {}
    neg_counts: dict[float, int] = {}
    for value in pos:
        score = direction * float(value)
        pos_counts[score] = pos_counts.get(score, 0) + 1
    for value in neg:
        score = direction * float(value)
        neg_counts[score] = neg_counts.get(score, 0) + 1

    true_positives = 0
    false_positives = 0
    previous_fpr = 0.0
    previous_tpr = 0.0
    for score in sorted(set(pos_counts) | set(neg_counts), reverse=True):
        next_true_positives = true_positives + pos_counts.get(score, 0)
        next_false_positives = false_positives + neg_counts.get(score, 0)
        next_fpr = next_false_positives / len(neg)
        next_tpr = next_true_positives / len(pos)
        if fpr <= next_fpr:
            if next_fpr == previous_fpr:
                return float(next_tpr)
            weight = (fpr - previous_fpr) / (next_fpr - previous_fpr)
            return float(previous_tpr + weight * (next_tpr - previous_tpr))
        true_positives = next_true_positives
        false_positives = next_false_positives
        previous_fpr = next_fpr
        previous_tpr = next_tpr
    return 1.0

