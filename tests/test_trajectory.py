"""Paper rollout diagnostics, continuation checkpoints, and paired statistics."""

import math
import types

import numpy as np
import torch

from denmark.experiments.trajectory.dream_theory_diagnostic import (
    FirstCrossingCheckpointHook,
    checkpoint_metrics,
    continue_pi0_to_unit,
    deterministic_full_unit,
    validate_native_sampler_compatibility,
)
from denmark.experiments.trajectory.llada_cumulative_metrics import (
    full_k_theorem_metrics,
    prefix_rollout_metrics,
    rankdata,
    sign_agreement,
    spearman_safe,
)
from denmark.experiments.trajectory.llada_reverse_hybrid_metrics import (
    bootstrap_mean_ci,
    paired_bootstrap_mean_ci,
    summarize_block_values,
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


def test_constant_bootstrap_interval_is_exact():
    assert bootstrap_mean_ci([2.0, 2.0, 2.0], n_bootstrap=100) == (2.0, 2.0)


def test_closure_sign_is_endpoint_minus_cumulative():
    summary = summarize_block_values([1.0, 2.0], [1.5, 2.5], n_bootstrap=100, seed=7)
    assert summary["closure_gap_mean"] == 0.5
    assert summary["cumulative_delta_mean"] == 1.5
    assert summary["endpoint_uplift_mean"] == 2.0


def test_paired_bootstrap_operates_on_differences():
    assert paired_bootstrap_mean_ci([3.0, 4.0], [1.0, 2.0], n_bootstrap=100) == (2.0, 2.0)


def test_zero_closure_interval_contains_zero():
    summary = summarize_block_values(np.arange(4.0), np.arange(4.0), n_bootstrap=100)
    assert summary["closure_ci_contains_zero"] is True


def test_deterministic_unit_uses_only_full_units():
    values = [deterministic_full_unit(i, 312, 25, 42) for i in range(100)]
    assert all(0 <= v < 12 for v in values)
    assert values == [deterministic_full_unit(i, 312, 25, 42) for i in range(100)]


def test_checkpoint_hook_uses_first_crossing_and_records_actual_progress():
    seen = []

    def callback(step, state, crossed, actual, remaining):
        seen.append((step, tuple(crossed), actual, remaining))
        return {"dream_step": step, "actual_progress": actual, "remaining_mask_count": remaining}

    hook = FirstCrossingCheckpointHook(0, 8, 99, (0.25, 0.50, 0.75), callback)
    hook(0, torch.tensor([[1, 1, 99, 99, 99, 99, 99, 99]]))  # 25%
    hook(1, torch.tensor([[1, 1, 1, 1, 1, 99, 99, 99]]))  # 62.5%, crosses 50
    hook(2, torch.tensor([[1, 1, 1, 1, 1, 1, 1, 99]]))  # 87.5%, crosses 75
    assert [r["target_progress"] for r in hook.records] == [0.25, 0.50, 0.75]
    assert [r["actual_progress"] for r in hook.records] == [0.25, 0.625, 0.875]
    assert not hook.pending


def test_checkpoint_metrics_reuses_rollout_prefixes_and_r10_reference():
    rollout = [[k + rho / 10 for rho in range(10)] for k in range(4)]
    pi0 = [[k + j / 20 for j in range(10)] for k in range(4)]
    metrics = checkpoint_metrics(rollout, pi0, prefixes=(1, 3, 5, 10))
    assert set(metrics) == {"metrics_R1", "metrics_R3", "metrics_R5", "metrics_R10"}
    assert metrics["metrics_R10"]["winner_agreement_with_R10"] is True
    assert metrics["metrics_R10"]["observed_finite_R_bound_diagnostic"] > 0


class _FakeDream:
    device = torch.device("cpu")

    def __init__(self):
        # Expose a bound method whose globals include sample_tokens, matching
        # the remote-code introspection contract.
        self._sample = types.MethodType(_fake_sample_owner, self)

    def __call__(self, x, *_args, **_kwargs):
        vocab = 8
        logits = torch.zeros(x.shape[0], x.shape[1], vocab)
        logits[..., 1] = 10
        return types.SimpleNamespace(logits=logits)


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, **kwargs):
    probs = torch.softmax(logits, -1)
    values, ids = probs.max(-1)
    return values, ids


def _fake_sample_owner(self):
    # Function body is irrelevant; sample_tokens exists in its globals.
    return sample_tokens


def _compatible_native_sample(self, generation_tokens_hook_func):
    eps, steps, i, x, logits = 0.001, 300, 0, torch.ones(1), torch.ones(1)
    s, t, num_mask_token = x, x, 1
    torch.linspace(1, eps, steps + 1)
    p_transfer = 1 - s / t
    generation_tokens_hook_func(i, x, logits)
    number_transfer_tokens = int(num_mask_token * (1 - s / t))
    return p_transfer, number_transfer_tokens, sample_tokens


def test_native_sampler_contract_accepts_pinned_dream_variable_names():
    model = types.SimpleNamespace()
    model._sample = types.MethodType(_compatible_native_sample, model)
    assert all(validate_native_sampler_compatibility(model).values())


def test_pi0_resume_starts_at_checkpoint_plus_one_without_reinitializing_canvas():
    model = _FakeDream()
    state = torch.tensor([[7, 7, 99, 99]])
    out, extra, completion, trace = continue_pi0_to_unit(
        model,
        state,
        checkpoint_step=1,
        unit_start=2,
        unit_end=4,
        steps=4,
        eps=1e-3,
        mask_id=99,
        temperature=0,
        top_p=None,
        top_k=None,
        alg="origin",
        alg_temp=0.1,
        downstream_seed=123,
    )
    assert trace[0] == 2
    assert completion >= 2
    assert extra == completion - 1
    assert out[0, :2].tolist() == [7, 7]  # saved prefix/canvas preserved
    assert not bool((out[0, 2:4] == 99).any())
