#!/usr/bin/env python3
"""Run upstream UMR on this repository's WaterBench protocol."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parents[3]))

from denomark.baselines.umr.model import score_umr_tokens
from denomark.core.model import (
    get_model_logits,
    load_generator_model,
    resolve_mask_id,
    safe_decode,
)


DATASETS = {
    "finance_qa": "finance_qa.jsonl",
    "alpacafarm": "alpacafarm.jsonl",
    "longform_qa": "longform_qa.jsonl",
}
BASE_FAMILY = {
    "llada8b": "llada",
    "llada15": "llada",
    "llada20mini": "llada2",
    "dream": "dream",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umr_root", type=Path, required=True)
    parser.add_argument("--waterbench_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", choices=sorted(BASE_FAMILY), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bitmap_path", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--n_per_dataset", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional post-selection cap, used only for compute-node smoke tests.")
    parser.add_argument("--gen_length", type=int, default=300)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--block_length", type=int, default=25)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--remasking", choices=("low_confidence", "random"), default="low_confidence")
    parser.add_argument("--selection_mode", choices=("sample_seeded", "first_n"), default="sample_seeded")
    parser.add_argument(
        "--source_indices_file",
        type=Path,
        default=None,
        help="Optional newline-delimited source row indices; overrides selection_mode/n_per_dataset.",
    )
    parser.add_argument("--min_response_tokens", type=int, default=150)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--max_rep4", type=float, default=0.2)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=4.0)
    parser.add_argument("--key", type=int, default=42)
    parser.add_argument("--watermark_str", default="10")
    parser.add_argument(
        "--raw_upstream_gumbel_scores",
        action="store_true",
        help=(
            "Reproduce upstream's direct positive-score-to-softmax path. By default, "
            "convert its positive Gumbel scores back to log scores before UMR modulation."
        ),
    )
    parser.add_argument(
        "--llada_min_length_eos_suppression",
        action="store_true",
        help=(
            "For LLaDA only, suppress EOS logits in the first min_response_tokens "
            "completion positions. This is a prefix-scoped form of upstream "
            "logits_eos_inf and is intended for prompts that repeatedly terminate early."
        ),
    )
    parser.add_argument(
        "--disable_watermark",
        action="store_true",
        help="Generate a clean completion with the same base-specific decode loop.",
    )
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join((text or "").replace("NEWLINE_CHAR", " ").split())


def build_prompt(row: dict, tokenizer) -> tuple[str, list[int]]:
    raw = row.get("raw_prompt")
    user = raw or row.get("input", "")
    if not raw and row.get("context"):
        user = f"{row['context']}\n\n{user}"
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user.strip()}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return tokenizer.decode(ids, skip_special_tokens=False), list(ids)
    except Exception:
        return user, list(tokenizer(user, add_special_tokens=True)["input_ids"])


def rep4(text: str) -> float:
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    counts = Counter(grams)
    return sum(value - 1 for value in counts.values() if value > 1) / max(1, len(grams))


def trim(ids: list[int], stops: set[int]) -> list[int]:
    for index, token_id in enumerate(ids):
        if token_id in stops:
            return ids[:index]
    return ids


def gumbel_scores_to_logits(scores: torch.Tensor) -> torch.Tensor:
    """Restore log-space scores expected by UMR without changing their argmax.

    Upstream LLaDA's ``add_gumbel_noise`` returns positive scores of the form
    ``exp(logit) / noise``.  ``Watermark.modulate_logits_probs`` subsequently
    applies softmax and therefore expects log-space input.  Passing the positive
    scores directly effectively exponentiates twice and makes ``delta`` nearly
    inert.  Taking log restores the declared interface while preserving the
    exact unwatermarked token ranking.
    """
    if not scores.is_floating_point():
        raise TypeError("Gumbel scores must be floating point")
    tiny = torch.finfo(scores.dtype).tiny
    return torch.log(scores.clamp_min(tiny))


def install_llada_gumbel_score_compat(module, resolved_mask_id: int) -> None:
    """Patch upstream's transfer helper at its module-global call site."""
    upstream_apply = module.apply_single_watermark

    @torch.no_grad()
    def apply_single_watermark_compat(
        x: torch.Tensor,
        x0: torch.Tensor,
        logits: torch.Tensor,
        batch_index: int,
        index: int,
        watermark,
        mask_id: int | None = None,
    ):
        effective_mask_id = resolved_mask_id if mask_id is None else int(mask_id)
        return upstream_apply(
            x=x,
            x0=x0,
            logits=gumbel_scores_to_logits(logits),
            batch_index=batch_index,
            index=index,
            watermark=watermark,
            mask_id=effective_mask_id,
        )

    module.apply_single_watermark = apply_single_watermark_compat


class DreamUMRApplyCompat:
    """Count upstream modulations and preserve native special-token decisions.

    UMR's bitmap spans the padded model vocabulary, including EOS and other
    special rows.  At high delta the watermark argmax can therefore replace a
    normal Dream sample with EOS even when Dream itself did not choose to stop.
    Preserve a native special-token sample, but exclude special tokens only
    from the watermark's replacement choice.  This does not suppress natural
    EOS and does not impose a minimum generation length.
    """

    def __init__(
        self,
        upstream_apply,
        *,
        convert_gumbel_scores: bool,
        preserve_token_ids: set[int] | None = None,
        gumbel_temperature: float = 0.0,
    ) -> None:
        self.upstream_apply = upstream_apply
        self.convert_gumbel_scores = bool(convert_gumbel_scores)
        self.gumbel_temperature = float(gumbel_temperature)
        self.preserve_token_ids = {
            int(token_id) for token_id in (preserve_token_ids or set())
        }
        self.num_modulated = 0
        self.num_native_specials = 0

    def reset(self) -> None:
        self.num_modulated = 0
        self.num_native_specials = 0

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        logits: torch.Tensor,
        batch_index: int,
        index: int,
        watermark,
        mask_id: int,
    ):
        if watermark is None:
            return self.upstream_apply(
                x=x,
                x0=x0,
                logits=logits,
                batch_index=batch_index,
                index=index,
                watermark=watermark,
                mask_id=mask_id,
            )

        native_token = int(x0[batch_index, index].item())
        if native_token in self.preserve_token_ids:
            self.num_native_specials += 1
            return

        self.num_modulated += 1
        if self.convert_gumbel_scores:
            logits = gumbel_scores_to_logits(logits)

        # UMR's standalone Dream loop applies watermarking to Gumbel-perturbed
        # scores, whereas its native Dream mixin accidentally watermarks raw
        # logits with a deterministic argmax even when temperature > 0. Restore
        # the declared temperature semantics only for the current position and
        # masked neighbors that upstream may use as provisional context.
        affected_positions = {int(index)}
        if index > 0 and int(x[batch_index, index - 1].item()) == int(mask_id):
            affected_positions.add(int(index - 1))
        if (
            index + 1 < x.shape[1]
            and int(x[batch_index, index + 1].item()) == int(mask_id)
        ):
            affected_positions.add(int(index + 1))
        position_ids = torch.tensor(
            sorted(affected_positions),
            dtype=torch.long,
            device=logits.device,
        )
        saved_rows = logits[batch_index, position_ids].clone()
        if self.gumbel_temperature > 0:
            selected = saved_rows.float()
            uniform = torch.rand_like(selected)
            gumbel = -torch.log(-torch.log(uniform.clamp_min(1e-20)))
            selected = selected + self.gumbel_temperature * gumbel
            logits[batch_index, position_ids] = selected.to(logits.dtype)

        token_ids = None
        saved_specials = None
        if self.preserve_token_ids:
            token_ids = torch.tensor(
                sorted(self.preserve_token_ids),
                dtype=torch.long,
                device=logits.device,
            )
            saved_specials = logits[batch_index, index, token_ids].clone()
            logits[batch_index, index, token_ids] = -torch.inf
        try:
            return self.upstream_apply(
                x=x,
                x0=x0,
                logits=logits,
                batch_index=batch_index,
                index=index,
                watermark=watermark,
                mask_id=mask_id,
            )
        finally:
            if token_ids is not None and saved_specials is not None:
                logits[batch_index, index, token_ids] = saved_specials
            logits[batch_index, position_ids] = saved_rows


class DreamNativeRegretController:
    """Run UMR regret remasking around Dream's released native sampler.

    UMR's standalone Dream loop calls the current model in a way that collapses
    instruction prompts to EOS.  The repository also ships a patched copy of
    Dream's native generation mixin that applies UMR at exactly the positions
    selected for transfer.  This controller supplies only the regret-remasking
    state machine after each native transfer.  EOS is never modified here.
    """

    def __init__(
        self,
        upstream_module,
        *,
        watermark,
        apply_watermark,
        mask_id: int,
        prompt_len: int,
        gen_length: int,
        steps: int,
    ) -> None:
        self.upstream = upstream_module
        self.watermark = watermark
        self.apply_watermark = apply_watermark
        self.mask_id = int(mask_id)
        self.prompt_len = int(prompt_len)
        self.gen_length = int(gen_length)
        self.steps = int(steps)
        self.previous_x: torch.Tensor | None = None
        self.cw_map: list[dict] | None = None
        self.cw_history: list[set] | None = None
        self.roll_back_times = 0
        self.re_green_count = 0

    def __call__(
        self,
        step: int | None,
        x: torch.Tensor,
        logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if step is None:
            self.previous_x = x.clone()
            self.cw_map = [{} for _ in range(x.shape[0])]
            self.cw_history = [set() for _ in range(x.shape[0])]
            return x

        if self.previous_x is None or self.cw_map is None or self.cw_history is None:
            raise RuntimeError("Dream native UMR controller was not initialized")

        previous_x = self.previous_x
        newly_filled = (previous_x == self.mask_id) & (x != self.mask_id)
        updated_indices_list = [
            torch.nonzero(newly_filled[batch_index], as_tuple=False).flatten()
            for batch_index in range(x.shape[0])
        ]

        if self.watermark is None:
            self.previous_x = x.clone()
            return x
        if logits is None:
            raise RuntimeError("Dream native UMR controller requires aligned logits")

        # Dream origin decoding has already selected and transferred these
        # positions. Apply UMR afterward so transfer scheduling/confidence and
        # natural EOS behavior remain native. The compatibility wrapper adds
        # the requested Gumbel temperature and preserves native specials.
        for batch_index, positions in enumerate(updated_indices_list):
            for position in positions:
                self.apply_watermark(
                    x=previous_x,
                    x0=x,
                    logits=logits,
                    batch_index=batch_index,
                    index=int(position.item()),
                    watermark=self.watermark,
                    mask_id=self.mask_id,
                )

        # Preserve upstream ordering: only candidates already queued before
        # this step can consume the current regret-remask opportunity.
        extra_budget_flags = [
            any(
                info.get("flag") is False and int(info.get("pn", 0)) > 0
                for info in self.cw_map[batch_index].values()
            )
            for batch_index in range(x.shape[0])
        ]

        confidence = torch.full(
            x.shape,
            -torch.inf,
            dtype=logits.dtype,
            device=x.device,
        )
        for batch_index, positions in enumerate(updated_indices_list):
            if positions.numel() == 0:
                continue
            selected_logits = logits[batch_index, positions].float()
            selected_tokens = x[batch_index, positions].unsqueeze(-1)
            selected_confidence = torch.softmax(selected_logits, dim=-1).gather(
                -1, selected_tokens
            ).squeeze(-1)
            confidence[batch_index, positions] = selected_confidence.to(confidence.dtype)

        self.roll_back_times, self.re_green_count = (
            self.upstream.check_rollback_and_cleanup(
                x,
                self.cw_map,
                self.watermark,
                self.mask_id,
                self.roll_back_times,
                self.re_green_count,
            )
        )
        self.upstream.check_watermark_violations(
            x=x,
            confidence=confidence,
            updated_indices_list=updated_indices_list,
            watermark=self.watermark,
            cw_map=self.cw_map,
            cw_history=self.cw_history,
            mask_id=self.mask_id,
        )
        self.upstream.execute_remasking(
            step_idx=int(step),
            total_steps=max(0, self.steps - 1),
            x=x,
            cw_map=self.cw_map,
            extra_budget_flags=extra_budget_flags,
            watermark=self.watermark,
            prompt_len=self.prompt_len,
            num_block=0,
            block_length=self.gen_length,
            mask_id=self.mask_id,
            confidence=confidence,
            updated_indices_list=updated_indices_list,
        )
        self.previous_x = x.clone()
        return x


def _apply_llada2_denomark(
    *,
    x: torch.Tensor,
    block_x0: torch.Tensor,
    watermark_scores: torch.Tensor,
    batch_index: int,
    local_index: int,
    block_start: int,
    block_end: int,
    watermark,
    mask_id: int,
) -> None:
    """Apply upstream UMR to one LLaDA2 active-block token.

    LLaDA2 must only materialize logits for the active semi-autoregressive
    block.  Translate its block-local logits back to the global neighbor
    context expected by UMR.  A masked neighbor outside the active block has
    no valid current-step logits and is therefore left unresolved; UMR then
    falls back to the available (normally previous-token) direction.
    """
    if watermark is None:
        return

    global_index = block_start + int(local_index)
    if int(x[batch_index, global_index - 1].item()) == int(mask_id):
        prev_token = None
        prev_logits = (
            watermark_scores[batch_index, local_index - 1]
            if local_index > 0
            else None
        )
    else:
        prev_token = x[batch_index, global_index - 1]
        prev_logits = None

    if global_index + 1 >= x.shape[1]:
        next_token = None
        next_logits = None
    elif int(x[batch_index, global_index + 1].item()) == int(mask_id):
        next_token = None
        next_logits = (
            watermark_scores[batch_index, local_index + 1]
            if global_index + 1 < block_end
            else None
        )
    else:
        next_token = x[batch_index, global_index + 1]
        next_logits = None

    # A non-empty instruction prompt guarantees a resolved previous token at
    # the first generated position. Keep a defensive fallback for unit tests
    # and malformed empty prompts.
    if prev_token is None and prev_logits is None:
        block_x0[batch_index, local_index] = torch.argmax(
            watermark_scores[batch_index, local_index]
        )
        return

    block_x0[batch_index, local_index] = watermark.modulate_logits_probs(
        current_logits=watermark_scores[batch_index, local_index],
        prev_token=prev_token,
        prev_logits=prev_logits,
        next_token=next_token,
        next_logits=next_logits,
    )


@torch.no_grad()
def generate_llada2_umr(
    *,
    model,
    prompt: torch.Tensor,
    upstream_module,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    remasking: str,
    mask_id: int,
    watermark,
    convert_gumbel_scores: bool,
) -> torch.Tensor:
    """Run UMR with LLaDA2's required active-block attention topology.

    The transfer budget and regret-remasking state machine are taken directly
    from upstream UMR.  Only model forwarding and global/block-local index
    translation are adapted for LLaDA2.
    """
    if remasking not in {"low_confidence", "random"}:
        raise ValueError(f"unsupported LLaDA2 remasking policy: {remasking}")
    if gen_length % block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks:
        raise ValueError("steps must be divisible by gen_length/block_length")
    steps_per_block = steps // num_blocks

    prompt_length = int(prompt.shape[1])
    x = torch.full(
        (prompt.shape[0], prompt_length + gen_length),
        int(mask_id),
        dtype=torch.long,
        device=prompt.device,
    )
    x[:, :prompt_length] = prompt
    cw_history = [set() for _ in range(x.shape[0])]

    for block_index in range(num_blocks):
        block_start = prompt_length + block_index * block_length
        block_end = block_start + block_length
        cw_map = [{} for _ in range(x.shape[0])]
        block_mask_index = x[:, block_start:block_end] == int(mask_id)
        transfer_budget = upstream_module.get_num_transfer_tokens(
            block_mask_index,
            steps_per_block,
        )

        # The extra iteration is part of upstream UMR's regret-remasking
        # budget and is intentionally preserved.
        for step_index in range(steps_per_block + 1):
            block_logits = get_model_logits(
                model,
                x,
                "llada2",
                logit_start=block_start,
                logit_end=block_end,
            )
            sampled_scores = upstream_module.add_gumbel_noise(
                block_logits,
                temperature=temperature,
            )
            watermark_scores = (
                gumbel_scores_to_logits(sampled_scores)
                if convert_gumbel_scores and temperature > 0
                else sampled_scores
            )
            sampled_tokens = torch.argmax(sampled_scores, dim=-1)
            current_mask = x[:, block_start:block_end] == int(mask_id)

            if remasking == "low_confidence":
                probabilities = torch.softmax(block_logits, dim=-1)
                block_confidence = torch.gather(
                    probabilities,
                    dim=-1,
                    index=sampled_tokens.unsqueeze(-1),
                ).squeeze(-1)
            else:
                block_confidence = torch.rand(
                    sampled_tokens.shape,
                    device=sampled_tokens.device,
                )

            sampled_tokens = torch.where(
                current_mask,
                sampled_tokens,
                x[:, block_start:block_end],
            )
            block_confidence = torch.where(
                current_mask,
                block_confidence,
                torch.full_like(block_confidence, -torch.inf),
            )
            confidence = torch.full(
                x.shape,
                -torch.inf,
                dtype=block_confidence.dtype,
                device=x.device,
            )
            confidence[:, block_start:block_end] = block_confidence

            updated_indices_list: list[torch.Tensor] = []
            extra_budget_flags = [False] * x.shape[0]
            for batch_index in range(x.shape[0]):
                has_valid_candidate = any(
                    info.get("flag") is False and int(info.get("pn", 0)) > 0
                    for info in cw_map[batch_index].values()
                )
                if step_index < steps_per_block - 1 and has_valid_candidate:
                    count = int(transfer_budget[batch_index, step_index].item()) + 1
                    extra_budget_flags[batch_index] = True
                elif step_index == steps_per_block - 1 and has_valid_candidate:
                    count = int(transfer_budget[batch_index, step_index].item())
                    extra_budget_flags[batch_index] = True
                elif step_index == steps_per_block:
                    count = 1
                else:
                    count = int(transfer_budget[batch_index, step_index].item())

                if count <= 0:
                    updated_indices_list.append(
                        torch.empty(0, dtype=torch.long, device=x.device)
                    )
                    continue

                count = min(count, block_length)
                local_indices = torch.topk(
                    block_confidence[batch_index],
                    k=count,
                ).indices
                global_indices = local_indices + block_start
                updated_indices_list.append(global_indices)
                for local_index in local_indices.tolist():
                    _apply_llada2_denomark(
                        x=x,
                        block_x0=sampled_tokens,
                        watermark_scores=watermark_scores,
                        batch_index=batch_index,
                        local_index=int(local_index),
                        block_start=block_start,
                        block_end=block_end,
                        watermark=watermark,
                        mask_id=mask_id,
                    )
                x[batch_index, global_indices] = sampled_tokens[
                    batch_index,
                    local_indices,
                ]

            upstream_module.check_rollback_and_cleanup(
                x,
                cw_map,
                watermark,
                mask_id,
            )
            upstream_module.check_watermark_violations(
                x=x,
                confidence=confidence,
                updated_indices_list=updated_indices_list,
                watermark=watermark,
                cw_map=cw_map,
                cw_history=cw_history,
                mask_id=mask_id,
            )
            upstream_module.execute_remasking(
                step_idx=step_index,
                total_steps=steps_per_block,
                x=x,
                cw_map=cw_map,
                extra_budget_flags=extra_budget_flags,
                watermark=watermark,
                prompt_len=prompt_length,
                num_block=block_index,
                block_length=block_length,
                mask_id=mask_id,
                confidence=confidence,
                updated_indices_list=updated_indices_list,
            )
    return x


def main() -> None:
    args = parse_args()
    if args.gen_length % args.block_length:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = args.gen_length // args.block_length
    if args.steps % num_blocks:
        raise ValueError("steps must be divisible by gen_length/block_length")
    if not (args.umr_root / "watermark" / "watermark_config.py").is_file():
        raise FileNotFoundError(f"Not an upstream UMR checkout: {args.umr_root}")
    sys.path.insert(0, str(args.umr_root.resolve()))

    WatermarkConfig = importlib.import_module("watermark.watermark_config").WatermarkConfig
    PersistentBitmap = importlib.import_module("watermark.bitmap_persistent").PersistentBitmap
    Watermark = importlib.import_module("watermark.watermarklogit").Watermark
    if args.base == "dream":
        # Keep Dream's released origin sampler intact, including its required
        # logit alignment and natural EOS behavior. Apply UMR through the
        # post-transfer token hook below. UMR's standalone low-confidence loop
        # and patched maskgit sampler collapse current instruction prompts to
        # first-token EOS even with watermarking disabled.
        dream_umr = importlib.import_module("watermark.Dream.watermark_generation_UMR")
        dream_native = importlib.import_module("watermark.Dream.watermark_generation_utils")
        dream_apply_compat = DreamUMRApplyCompat(
            dream_native.apply_single_watermark,
            convert_gumbel_scores=False,
            gumbel_temperature=(
                0.0 if args.raw_upstream_gumbel_scores else args.temperature
            ),
        )
    else:
        llada_umr = importlib.import_module("watermark.LLaDA.watermark_gen_UMR_L")
        # At temperature 0 upstream returns the original model logits, which
        # are already in the log-score space expected by UMR.  The
        # compatibility conversion is needed only for temperature > 0, where
        # upstream returns positive Gumbel scores of the form
        # exp(logit) / noise.
        generate = llada_umr.generate

    family = BASE_FAMILY[args.base]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_generator_model(args.model, generator_family=family, device_map=args.device)
    mask_id = resolve_mask_id(tokenizer, family, args.mask_id)
    if (
        args.base != "dream"
        and not args.raw_upstream_gumbel_scores
        and args.temperature > 0
    ):
        install_llada_gumbel_score_compat(llada_umr, mask_id)
    if args.base == "dream":
        dream_special_ids = {
            int(token_id)
            for token_id in (
                mask_id,
                tokenizer.pad_token_id,
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
            )
            if token_id is not None
        }
        dream_apply_compat.preserve_token_ids = dream_special_ids
    # UMR operates on the model-logit vocabulary, which can include padded
    # output rows not exposed by ``len(tokenizer)`` (126464 vs 126349 for
    # LLaDA).  The bitmap must therefore follow the model config/logits size.
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))
    if args.disable_watermark:
        bitmap = None
        watermark = None
    else:
        config = WatermarkConfig(
            vocab_size=vocab_size,
            ratio=args.ratio,
            delta=args.delta,
            key=args.key,
            prebias=False,
            strategy="bidirectional",
            bitmap_path=str(args.bitmap_path),
            watermark_str=args.watermark_str,
        )
        bitmap = PersistentBitmap(vocab_size, str(args.bitmap_path), device=args.device)
        watermark = Watermark(config, bitmap)

    # Upstream's assertion helper copies the complete 126k/152k bitmap row to
    # CPU and converts it to a Python list on every neighbor check.  Generation
    # only consumes the boolean result, so avoid that diagnostic-only transfer
    # while preserving the exact bitmap decision.
    def is_token_in_green_list(token: int, previous: int):
        return bool(bitmap.get_bit(int(previous), int(token))), None

    if watermark is not None:
        watermark.is_token_in_green_list = is_token_in_green_list

    rows = [json.loads(line) for line in (args.waterbench_dir / DATASETS[args.dataset]).open() if line.strip()]
    rng = random.Random(args.seed)
    if args.source_indices_file is not None:
        indices = [
            int(line.strip())
            for line in args.source_indices_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(indices) != len(set(indices)):
            raise ValueError("source_indices_file contains duplicate indices")
        bad = [index for index in indices if not 0 <= index < len(rows)]
        if bad:
            raise ValueError(f"source_indices_file contains out-of-range indices: {bad}")
    elif args.selection_mode == "sample_seeded" and args.n_per_dataset < len(rows):
        indices = sorted(rng.sample(range(len(rows)), args.n_per_dataset))
    else:
        indices = list(range(min(args.n_per_dataset, len(rows))))
    indexed = [(index, rows[index]) for index in indices]
    indexed = [row for position, row in enumerate(indexed) if position % args.num_shards == args.shard_idx]
    if args.max_samples is not None:
        indexed = indexed[: args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Match upstream post-processing. Dream stops only at EOS; treating its
    # pad-like vocabulary row as EOS can incorrectly collapse a full sample to
    # zero tokens. LLaDA additionally treats a residual MASK as termination.
    stops = set() if args.base == "dream" else {mask_id}
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))

    with args.output.open("w", encoding="utf-8") as handle:
        for source_index, row in tqdm(indexed, desc=f"{args.base}:{args.dataset}"):
            generation_started = time.perf_counter()
            prompt_text, prompt_ids = build_prompt(row, tokenizer)
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
            # Recent PyTorch SDPA rejects Dream's legacy integer attention
            # mask; bool is semantically identical and accepted by Dream.
            attention = torch.ones_like(prompt, dtype=torch.bool if args.base == "dream" else torch.long)
            final = None
            for attempt in range(args.max_retries + 1):
                final_seed = args.seed + source_index * 1000 + attempt
                torch.manual_seed(final_seed)
                torch.cuda.manual_seed_all(final_seed)
                if args.base == "dream":
                    dream_apply_compat.reset()
                    dream_controller = DreamNativeRegretController(
                        dream_umr,
                        watermark=watermark,
                        apply_watermark=dream_apply_compat,
                        mask_id=mask_id,
                        prompt_len=prompt.shape[1],
                        gen_length=args.gen_length,
                        steps=args.steps,
                    )
                    generated = model.diffusion_generate(
                        prompt,
                        max_new_tokens=args.gen_length,
                        steps=args.steps,
                        eps=1e-3,
                        temperature=args.temperature,
                        top_p=None,
                        top_k=None,
                        alg="origin",
                        alg_temp=0.0,
                        mask_token_id=mask_id,
                        generation_tokens_hook_func=dream_controller,
                    )
                    num_modulated = dream_apply_compat.num_modulated
                elif args.base == "llada20mini":
                    generated = generate_llada2_umr(
                        model=model,
                        prompt=prompt,
                        upstream_module=llada_umr,
                        steps=args.steps,
                        gen_length=args.gen_length,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        remasking=args.remasking,
                        mask_id=mask_id,
                        watermark=watermark,
                        convert_gumbel_scores=(
                            not args.raw_upstream_gumbel_scores
                        ),
                    )
                    num_modulated = None
                else:
                    generated = generate(
                        model,
                        prompt=prompt,
                        attention_mask=attention,
                        steps=args.steps,
                        gen_length=args.gen_length,
                        block_length=args.block_length,
                        temperature=args.temperature,
                        remasking=args.remasking,
                        mask_id=mask_id,
                        watermark=watermark,
                        min_eos_tokens=(
                            args.min_response_tokens
                            if args.llada_min_length_eos_suppression
                            else 0
                        ),
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    num_modulated = None
                raw_completion_ids = generated[0, len(prompt_ids):].tolist()
                first_stop_index = next(
                    (index for index, token_id in enumerate(raw_completion_ids) if token_id in stops),
                    None,
                )
                if args.base == "dream":
                    print(
                        f"Dream raw attempt={attempt + 1} head={raw_completion_ids[:16]} "
                        f"eos_id={tokenizer.eos_token_id} first_stop_index={first_stop_index}",
                        flush=True,
                    )
                completion_ids = trim(raw_completion_ids, stops)
                text = clean_text(safe_decode(tokenizer, completion_ids, skip_special_tokens=True))
                repetition = rep4(text)
                final = (
                    attempt,
                    final_seed,
                    completion_ids,
                    text,
                    repetition,
                    raw_completion_ids[:16],
                    first_stop_index,
                    num_modulated,
                )
                if len(completion_ids) >= args.min_response_tokens and repetition <= args.max_rep4:
                    break
            (
                attempt,
                final_seed,
                completion_ids,
                text,
                repetition,
                raw_completion_head,
                first_stop_index,
                num_modulated,
            ) = final
            generation_seconds = time.perf_counter() - generation_started
            detection = None
            if bitmap is not None:
                detection = score_umr_tokens(
                    completion_ids,
                    bitmap,
                    watermark_str=args.watermark_str,
                    key=args.key,
                    ratio=args.ratio,
                    previous_token=prompt_ids[-1],
                )
            record = {
                "base": args.base,
                "method": "clean" if args.disable_watermark else "umr",
                "dataset": args.dataset,
                "prompt_idx": int(row.get("clean_n1000_idx", source_index)),
                "source_row_idx": source_index,
                "prompt_input": row.get("input", ""),
                "prompt_context": row.get("context", ""),
                "prompt_full": prompt_text,
                "prompt_last_token_id": prompt_ids[-1],
                "watermarked_text": text,
                "text": text,
                "token_ids": completion_ids,
                "raw_completion_head": raw_completion_head,
                "first_stop_index": first_stop_index,
                "umr_modulated_transfers": num_modulated,
                "umr_native_special_transfers": (
                    dream_apply_compat.num_native_specials
                    if args.base == "dream"
                    else None
                ),
                "token_len": len(completion_ids),
                "generation_seconds": generation_seconds,
                "rep4": repetition,
                "passed_quality": len(completion_ids) >= args.min_response_tokens and repetition <= args.max_rep4,
                "retry_attempts": attempt + 1,
                "seed": args.seed,
                "final_seed": final_seed,
                "umr_detector": detection,
                "gen_config": vars(args) | {
                    "umr_root": str(args.umr_root),
                    "output": str(args.output),
                    "bitmap_path": str(args.bitmap_path),
                    "mask_id": mask_id,
                    "vocab_size": vocab_size,
                    "watermark_disabled": args.disable_watermark,
                    "llada_gumbel_score_compat": (
                        args.base != "dream"
                        and not args.raw_upstream_gumbel_scores
                        and args.temperature > 0
                    ),
                    "dream_backend": (
                        "official_dream_origin_umr_regret_hook"
                        if args.base == "dream"
                        else None
                    ),
                    "dream_native_alg": "origin" if args.base == "dream" else None,
                    "dream_native_watermark_gumbel_temperature": (
                        dream_apply_compat.gumbel_temperature
                        if args.base == "dream"
                        else None
                    ),
                    "dream_preserve_native_special_tokens": (
                        sorted(dream_apply_compat.preserve_token_ids)
                        if args.base == "dream"
                        else None
                    ),
                    "dream_native_regret_rollbacks": (
                        dream_controller.roll_back_times if args.base == "dream" else None
                    ),
                    "dream_native_regreens": (
                        dream_controller.re_green_count if args.base == "dream" else None
                    ),
                    "dream_eos_suppression": False if args.base == "dream" else None,
                    "dream_min_length_eos_suppression": None,
                    "llada_min_length_eos_suppression": (
                        args.min_response_tokens
                        if args.base != "dream" and args.llada_min_length_eos_suppression
                        else None
                    ),
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
