from __future__ import annotations

import types

import torch

from denomark.core.model import (DreamSemanticUnitDecodeController, NativeSemanticWatermarkHook)


def test_random_global_grouped_selects_exact_global_budget_and_records_groups():
    torch.manual_seed(7)
    prompt_len = 3
    controller = DreamSemanticUnitDecodeController(
        prompt_len=prompt_len,
        gen_length=50,
        block_size=25,
        cand_block_size=8,
        mask_id=99,
        steps=7,
        eps=1e-3,
        mode="random_global_grouped",
    )
    x = torch.full((1, prompt_len + 50), 99, dtype=torch.long)
    x[:, :prompt_len] = 1
    logits = torch.zeros((1, prompt_len + 50, 4))

    assert controller.logits_hook(0, x, logits) is logits
    picked = controller.allowed_mask[0].nonzero(as_tuple=False).flatten()
    assert picked.numel() == 8
    assert bool(((picked >= prompt_len) & (picked < prompt_len + 50)).all())
    diag = controller.diag[-1]
    assert diag["selected_position_count"] == 8
    assert sum(len(group) for group in diag["selected_positions_by_block"].values()) == 8
    assert diag["selected_group_count"] == len(diag["selected_positions_by_block"])


def test_random_global_grouped_uses_remaining_positions_on_tail_step():
    controller = DreamSemanticUnitDecodeController(
        prompt_len=1,
        gen_length=25,
        block_size=25,
        cand_block_size=8,
        mask_id=99,
        steps=4,
        eps=1e-3,
        mode="random_global_grouped",
    )
    x = torch.arange(26).unsqueeze(0)
    x[0, [4, 8, 21]] = 99
    logits = torch.zeros((1, 26, 4))

    controller.logits_hook(3, x, logits)
    picked = controller.allowed_mask[0].nonzero(as_tuple=False).flatten().tolist()
    assert sorted(picked) == [4, 8, 21]


def test_hook_scores_all_semantic_groups_from_same_precommit_canvas():
    mask_id = 99
    hook = NativeSemanticWatermarkHook.__new__(NativeSemanticWatermarkHook)
    hook.prompt_len = 0
    hook.gen_length = 50
    hook.block_size = 25
    hook.cand_block_size = 8
    hook.num_candidates = 2
    hook.candidate_temperature = 0.5
    hook.candidate_position_mode = "same_positions"
    hook.resample_base_candidate = False
    hook.mask_id = mask_id
    hook.prev_x = torch.full((1, 50), mask_id, dtype=torch.long)
    hook.forced_candidate_mask = torch.zeros((1, 50), dtype=torch.bool)
    hook.forced_candidate_mask[0, [1, 2, 28, 29, 30]] = True
    hook.diag = []

    seen_canvases = []

    def fake_score(
        self,
        candidate_tokens,
        x,
        positions,
        block_id,
        logits,
        step,
        remaining_before_commit,
    ):
        seen_canvases.append(x.clone())
        return 1, {"block_id": block_id, "num_positions": int(positions.numel())}

    hook._score_candidates = types.MethodType(fake_score, hook)
    current = torch.full((1, 50), mask_id, dtype=torch.long)
    # Simulate native DREAM transfers; grouped mode must discard them first.
    current[0, [0, 25]] = 3
    logits = torch.zeros((1, 50, 8))
    logits[..., 4] = 5.0

    output = hook(0, current, logits)

    assert len(seen_canvases) == 2
    assert torch.equal(seen_canvases[0], seen_canvases[1])
    assert bool((seen_canvases[0][0, [1, 2, 28, 29, 30]] == mask_id).all())
    assert bool((output[0, [1, 2, 28, 29, 30]] != mask_id).all())
    assert len(hook.diag) == 2
    assert {row["semantic_group_count"] for row in hook.diag} == {2}
    assert {row["global_step_position_count"] for row in hook.diag} == {5}
