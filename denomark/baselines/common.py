"""
Unwatermarked baseline: 1:1 mirror of LLaDA's official `generate()` function
(without the optional `watermarker` hook).

Reference: GSAI-ML/LLaDA, `generate.py` in the model repo.

Use this as the false-positive control for detector calibration.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np

from denomark.core.model import get_model_logits, safe_decode


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits - torch.log(-torch.log(noise + 1e-20) + 1e-20) * temperature


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Distribute `mask_num` MASK positions roughly evenly across `steps` steps.
    Earlier steps get one extra if not divisible (LLaDA linear noise schedule).
    Returns [B, steps] long tensor.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def llada_generate_unwatermarked(
    prompt: torch.Tensor,
    llada,
    llada_tok,
    mask_id: int,
    gen_length: int = 300,
    block_size: int = 25,
    steps: int = None,
    temperature: float = 0.5,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    generator_family: str = "llada",
):
    """
    Unwatermarked diffusion generation.

    LLaDA2 follows the same active-block forward topology as our semantic
    generator while retaining the requested remasking policy.

    Args:
        prompt: [1, L_p] token ids
        gen_length: total new tokens to generate
        block_size: semi-AR block size (must divide gen_length)
        steps: total denoising steps (must divide num_blocks). Default = gen_length
        temperature: sampling temperature (0 = greedy)
        cfg_scale: classifier-free guidance scale (0 = off)
        remasking: 'low_confidence' (default) | 'random' | 'ar'

    Returns:
        (text: str, token_ids: list[int])  — same shape as denomark.core.generate.generate_one
    """
    assert gen_length % block_size == 0, "gen_length must be divisible by block_size"
    num_blocks = gen_length // block_size
    if steps is None:
        steps = gen_length  # default: 1 token transferred per step
    assert steps % num_blocks == 0, f"steps={steps} must be divisible by num_blocks={num_blocks}"
    steps_per_block = steps // num_blocks

    L_p = prompt.shape[1]
    x = torch.full((1, L_p + gen_length), mask_id, dtype=torch.long, device=llada.device)
    x[:, :L_p] = prompt.clone()
    prompt_index = x != mask_id
    llada2_eos_token_id = (
        getattr(llada_tok, "eos_token_id", None)
        if generator_family == "llada2" and llada_tok is not None
        else None
    )

    for num_block in range(num_blocks):
        b_s = L_p + num_block * block_size
        b_e = L_p + (num_block + 1) * block_size
        block_mask_index = (x[:, b_s:b_e] == mask_id)
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            if generator_family == "llada2":
                # Match the LLaDA2 path used by our candidate rollouts: only
                # forward through the active block and only materialize logits
                # for that block. Passing the full future MASK canvas changes
                # LLaDA2's block-attention topology and severely hurts quality.
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    block_logits = get_model_logits(
                        llada,
                        x_,
                        generator_family,
                        logit_start=b_s,
                        logit_end=b_e,
                    )
                    block_logits, un_logits = torch.chunk(block_logits, 2, dim=0)
                    block_logits = un_logits + (cfg_scale + 1) * (block_logits - un_logits)
                else:
                    block_logits = get_model_logits(
                        llada,
                        x,
                        generator_family,
                        logit_start=b_s,
                        logit_end=b_e,
                    )

                block_x0 = torch.argmax(
                    _add_gumbel_noise(block_logits, temperature),
                    dim=-1,
                )
                block_mask = x[:, b_s:b_e] == mask_id
                if remasking == "low_confidence":
                    block_probs = F.softmax(block_logits.to(torch.float64), dim=-1)
                    block_confidence = torch.gather(
                        block_probs,
                        dim=-1,
                        index=block_x0.unsqueeze(-1),
                    ).squeeze(-1)
                elif remasking == "random":
                    block_confidence = torch.rand(
                        block_x0.shape,
                        device=block_x0.device,
                    )
                elif remasking == "ar":
                    block_confidence = (
                        -torch.arange(block_x0.shape[1], device=block_x0.device)
                        .unsqueeze(0)
                        .repeat(block_x0.shape[0], 1)
                        / block_x0.shape[1]
                    )
                else:
                    raise NotImplementedError(remasking)

                block_x0 = torch.where(block_mask, block_x0, x[:, b_s:b_e])
                block_confidence = torch.where(
                    block_mask,
                    block_confidence,
                    torch.full_like(block_confidence, -float("inf")),
                )
                transfer_index = torch.zeros_like(block_x0, dtype=torch.bool)
                for j in range(block_confidence.shape[0]):
                    k = int(num_transfer_tokens[j, i].item())
                    if k > 0:
                        _, selected = torch.topk(block_confidence[j], k=k)
                        transfer_index[j, selected] = True
                active_block = x[:, b_s:b_e]
                active_block[transfer_index] = block_x0[transfer_index]
                continue

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = get_model_logits(llada, x_, generator_family)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = get_model_logits(llada, x, generator_family)

            logits_with_noise = _add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == "ar":
                x0_p = (
                    -torch.arange(x0.shape[1], device=x0.device).unsqueeze(0)
                    .repeat(x0.shape[0], 1) / x0.shape[1]
                )
            else:
                raise NotImplementedError(remasking)

            # only allow transferring within current block
            x0_p[:, b_e:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, sel = torch.topk(confidence[j], k=int(num_transfer_tokens[j, i].item()))
                transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

        if (
            llada2_eos_token_id is not None
            and (x[:, L_p:b_e] == int(llada2_eos_token_id)).any()
        ):
            break

    tokens = x[0, L_p:].tolist()
    if llada2_eos_token_id is not None:
        if int(llada2_eos_token_id) in tokens:
            eos_index = tokens.index(int(llada2_eos_token_id))
            tokens = tokens[:eos_index + 1]
    text = safe_decode(llada_tok, tokens, skip_special_tokens=True).strip()
    return text, tokens
