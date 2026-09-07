import numpy as np

from denomark.experiments.trajectory.llada_reverse_hybrid_metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_mean_ci,
    summarize_block_values,
)


def test_constant_bootstrap_interval_is_exact():
    assert bootstrap_mean_ci([2.0, 2.0, 2.0], n_bootstrap=100) == (2.0, 2.0)


def test_closure_sign_is_endpoint_minus_cumulative():
    summary = summarize_block_values(
        [1.0, 2.0], [1.5, 2.5], n_bootstrap=100, seed=7
    )
    assert summary["closure_gap_mean"] == 0.5
    assert summary["cumulative_delta_mean"] == 1.5
    assert summary["endpoint_uplift_mean"] == 2.0


def test_paired_bootstrap_operates_on_differences():
    assert paired_bootstrap_mean_ci(
        [3.0, 4.0], [1.0, 2.0], n_bootstrap=100
    ) == (2.0, 2.0)


def test_zero_closure_interval_contains_zero():
    summary = summarize_block_values(
        np.arange(4.0), np.arange(4.0), n_bootstrap=100
    )
    assert summary["closure_ci_contains_zero"] is True
