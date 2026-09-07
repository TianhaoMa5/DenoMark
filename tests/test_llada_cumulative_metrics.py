import math

import numpy as np

from denomark.experiments.trajectory.llada_cumulative_metrics import (
    full_k_theorem_metrics,
    prefix_rollout_metrics,
    rankdata,
    sign_agreement,
    spearman_safe,
)


def test_rankdata_averages_ties():
    assert rankdata([3, 1, 1, 2]).tolist() == [4.0, 1.5, 1.5, 3.0]


def test_prefix_metrics_reuses_rollout_prefixes_and_selected_q():
    raw = np.asarray(
        [
            [0.0] * 10,
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5] * 10,
        ]
    )
    metrics = prefix_rollout_metrics(raw, {0: 0.1, 1: 0.7, 2: 0.4})
    assert metrics["R1"]["winner_zero_based"] == 1
    assert metrics["R3"]["rollout_gain"] == 1.0
    assert metrics["R10"]["winner_zero_based"] == 1
    assert math.isclose(metrics["R3"]["continued_policy_advantage"], 0.6)
    assert math.isclose(metrics["R3"]["selected_candidate_alignment_error"], 0.4)
    assert metrics["R10"]["empirical_selection_loss_vs_R10"] == 0.0


def test_safe_correlations_and_signs():
    assert math.isclose(spearman_safe([1, 2, 3], [10, 20, 30]), 1.0)
    assert sign_agreement([1, -1, 0], [2, -3, 0]) == 1.0


def test_full_k_theorem_identity_uses_continued_policy_q():
    raw = np.asarray(
        [
            [0.0] * 10,
            [1.0] * 10,
            [0.5] * 10,
        ]
    )
    result = full_k_theorem_metrics(raw, [0.2, 0.5, 0.9])
    assert math.isclose(result["Gamma_Q_pi"], 0.7)
    assert result["best_Q_pi_candidate_zero_based"] == 2
    # Rollout chooses candidate 1, whose Q advantage is 0.3 and Q regret is 0.4.
    r3 = result["by_R"]["R3"]
    assert r3["rollout_winner_zero_based"] == 1
    assert math.isclose(r3["epsilon_Q_pi"], 0.4)
    assert math.isclose(r3["gamma_minus_epsilon"], 0.3)
    assert math.isclose(r3["continued_policy_advantage"], 0.3)
    assert result["max_identity_abs_error"] < 1e-12
