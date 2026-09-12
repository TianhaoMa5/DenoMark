"""Pure metric helpers for the LLaDA cumulative block diagnostic."""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties, matching scipy.stats.rankdata(method='average')."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearson_safe(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) < 2 or len(a) != len(b) or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_safe(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson_safe(rankdata(x), rankdata(y))


def sign_agreement(x: Sequence[float], y: Sequence[float], atol: float = 1e-12) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if len(a) == 0 or len(a) != len(b):
        return float("nan")
    sa = np.where(np.abs(a) <= atol, 0, np.sign(a))
    sb = np.where(np.abs(b) <= atol, 0, np.sign(b))
    return float(np.mean(sa == sb))


def bootstrap_mean_ci(
    values: Sequence[float], *, seed: int = 12345, n_bootstrap: int = 10_000
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(x, size=(n_bootstrap, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def summarize_vector(values: Iterable[float], *, seed: int = 12345) -> dict:
    x = np.asarray(list(values), dtype=float)
    lo, hi = bootstrap_mean_ci(x, seed=seed)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "bootstrap_ci95_low": lo,
        "bootstrap_ci95_high": hi,
        "positive_ratio": float(np.mean(x > 0)) if len(x) else float("nan"),
    }


def prefix_rollout_metrics(
    rollout_scores: Sequence[Sequence[float]],
    q_means: Mapping[int, float],
    r_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, dict]:
    """Compute step metrics using candidate 0 as the reference.

    ``q_means`` may contain only candidate 0 and the de-duplicated R winners.
    Consequently the returned alignment error is explicitly winner-selected,
    not a max over all K candidates.
    """
    raw = np.asarray(rollout_scores, dtype=float)
    if raw.ndim != 2:
        raise ValueError("rollout_scores must have shape [K, diagnostic_rollouts]")
    if 0 not in q_means:
        raise ValueError("q_means must contain reference candidate 0")
    mu10 = raw.mean(axis=1)
    winner10 = int(np.argmax(mu10))
    observed_d = float(raw.max() - raw.min())
    out: dict[str, dict] = {}
    for r in r_values:
        if r < 1 or r > raw.shape[1]:
            raise ValueError(f"invalid rollout prefix R={r} for width {raw.shape[1]}")
        mu = raw[:, :r].mean(axis=1)
        order = np.argsort(-mu, kind="mergesort")
        winner = int(order[0])
        gamma = float(mu[winner] - mu[0])
        margin = float(mu[order[0]] - mu[order[1]]) if len(order) > 1 else 0.0
        selection_loss = float(mu10[winner10] - mu10[winner])
        if winner not in q_means:
            raise ValueError(f"q_means missing R={r} winner candidate {winner}")
        advantage = float(q_means[winner] - q_means[0])
        out[f"R{r}"] = {
            "winner_zero_based": winner,
            "rollout_gain": gamma,
            "continued_policy_advantage": advantage,
            "selected_candidate_alignment_error": abs(advantage - gamma),
            "top1_top2_margin": margin,
            "winner_agreement_with_R10": bool(winner == winner10),
            "empirical_selection_loss_vs_R10": max(0.0, selection_loss),
            "observed_D": observed_d,
            "observed_worst_case_finite_R_diagnostic": float(
                2.0 * observed_d * math.sqrt(math.log(raw.shape[0]) / (2.0 * r))
            ),
        }
    return out


def full_k_theorem_metrics(
    rollout_scores: Sequence[Sequence[float]],
    q_means: Sequence[float],
    r_values: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """Compute the continued-policy theorem terms for one complete-K step.

    Candidate zero is the matched reference.  ``Gamma`` is defined in the
    continued-policy value space, while each ``epsilon_R`` is the regret of the
    rollout-prefix winner relative to the best continued-policy candidate.
    """
    raw = np.asarray(rollout_scores, dtype=float)
    q = np.asarray(q_means, dtype=float)
    if raw.ndim != 2:
        raise ValueError("rollout_scores must have shape [K, diagnostic_rollouts]")
    if q.shape != (raw.shape[0],):
        raise ValueError(f"q_means shape {q.shape} does not match K={raw.shape[0]}")
    if not np.isfinite(raw).all() or not np.isfinite(q).all():
        raise ValueError("full-K theorem inputs must be finite")
    best_q = float(q.max())
    best_q_idx = int(np.argmax(q))
    gamma = float(best_q - q[0])
    by_r: dict[str, dict] = {}
    max_identity_error = 0.0
    for r in r_values:
        if r < 1 or r > raw.shape[1]:
            raise ValueError(f"invalid rollout prefix R={r} for width {raw.shape[1]}")
        winner = int(np.argmax(raw[:, :r].mean(axis=1)))
        epsilon = float(best_q - q[winner])
        advantage = float(q[winner] - q[0])
        identity_error = abs((gamma - epsilon) - advantage)
        max_identity_error = max(max_identity_error, identity_error)
        by_r[f"R{r}"] = {
            "rollout_winner_zero_based": winner,
            "epsilon_Q_pi": epsilon,
            "continued_policy_advantage": advantage,
            "gamma_minus_epsilon": float(gamma - epsilon),
            "identity_abs_error": float(identity_error),
        }
    return {
        "best_Q_pi_candidate_zero_based": best_q_idx,
        "Q_pi_range": float(q.max() - q.min()),
        "Gamma_Q_pi": gamma,
        "by_R": by_r,
        "max_identity_abs_error": float(max_identity_error),
    }
