"""DenMark generation for diffusion language models such as LLaDA.

Algorithm per inner step:
  1. Pick r MASK positions in the current block.
  2. LLaDA forward -> logits at those positions.
  3. Build K candidates by Gumbel/top-p sampling at those r positions.
  4. For each candidate: one-shot complete the block's remaining MASKs.
  5. T_eta(completed block text) -> 768-d embedding.
  6. Project on θ_block_idx -> per-channel signed scores.
  7. Select a candidate with the requested selector.
     Current experiments use max_watermark: largest mean signed score.
  8. Commit candidate's r perturbed positions (rollout fills are discarded).

CLI:
  python -m denmark.core.generate \
      --prompts_jsonl /path/to/prompts.jsonl \
      --llada_model /path/to/LLaDA-8B-Instruct \
      --encoder_model /path/to/DenMark-Encoder \
      --output runs/gen.jsonl \
      --num_samples 5 --gen_length 300 --block_size 25 \
      --num_candidates 16 --rollouts_per_cand 3 --rollout_schedule linear_decay \
      --cand_block_size 1 --num_message_bits 2 --channels_per_step 2 \
      --temperature 0.5 --perturb_temperature 0.6 --rollout_temperature 0.5 \
      --selector max_watermark --selection_mode argmax_score \
      --position_selection random --per_cand_positions \
      --dedup_candidates --shared_rollout_seeds
"""
import argparse
import sys
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from denmark.core.model import (
    _gumbel, build_directions, complete_block_one_shot, encode_texts,
    GENERATOR_FAMILIES, get_model_logits, load_generator_model, resolve_mask_id,
    safe_batch_decode, safe_decode,
)
from denmark.core.selectors import select_candidate


# ---------------------------------------------------------------------------
# Per-block-step helpers
# ---------------------------------------------------------------------------

def select_positions_in_block(x, L_p, block_id, block_size, gen_length, r, mask_id):
    s = L_p + block_id * block_size
    e = min(s + block_size, L_p + gen_length)
    in_block = (x[0, s:e] == mask_id).nonzero(as_tuple=True)[0] + s
    if len(in_block) == 0:
        return None
    perm = torch.randperm(len(in_block), device=in_block.device)[:r]
    return in_block[perm]


def candidate_region_bounds(L_p, block_id, block_size, gen_length, inner_step,
                            candidate_region_size, r):
    block_s = L_p + block_id * block_size
    block_e = min(block_s + block_size, L_p + gen_length)
    if not candidate_region_size or candidate_region_size <= 0:
        return block_s, block_e
    region_size = min(int(candidate_region_size), max(1, block_e - block_s))
    steps_per_region = max(1, math.ceil(region_size / max(1, int(r))))
    n_regions = math.ceil((block_e - block_s) / region_size)
    region_id = min(max(0, inner_step // steps_per_region), max(0, n_regions - 1))
    region_s = block_s + region_id * region_size
    region_e = min(region_s + region_size, block_e)
    return region_s, region_e


def choose_random_active_block(x, L_p, num_blocks, block_size, gen_length, r, mask_id):
    """Pick a random non-finished block, preferring blocks with at least r MASKs."""
    eligible = []
    fallback = []
    for block_id in range(num_blocks):
        s = L_p + block_id * block_size
        e = min(s + block_size, L_p + gen_length)
        n_mask = int((x[0, s:e] == mask_id).sum().item())
        if n_mask >= r:
            eligible.append(block_id)
        elif n_mask > 0:
            fallback.append(block_id)

    candidates = eligible if eligible else fallback
    if not candidates:
        return None
    pick = torch.randint(len(candidates), (1,), device=x.device).item()
    return candidates[pick]


def choose_random_active_semantic_block_in_decode_block(
    x,
    L_p,
    decode_block_id,
    decode_block_size,
    semantic_block_size,
    gen_length,
    r,
    mask_id,
    seed=None,
):
    """Pick an active semantic sub-block inside the current decode block."""
    decode_s = decode_block_id * decode_block_size
    decode_e = min(decode_s + decode_block_size, gen_length)
    first_sem = decode_s // semantic_block_size
    last_sem = math.ceil(decode_e / semantic_block_size)
    eligible = []
    fallback = []
    for semantic_block_id in range(first_sem, last_sem):
        s = L_p + semantic_block_id * semantic_block_size
        e = min(s + semantic_block_size, L_p + gen_length)
        s = max(s, L_p + decode_s)
        e = min(e, L_p + decode_e)
        if e <= s:
            continue
        n_mask = int((x[0, s:e] == mask_id).sum().item())
        if n_mask >= r:
            eligible.append(semantic_block_id)
        elif n_mask > 0:
            fallback.append(semantic_block_id)

    candidates = eligible if eligible else fallback
    if not candidates:
        return None
    if seed is None:
        pick = torch.randint(len(candidates), (1,), device=x.device).item()
    else:
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        pick = torch.randint(len(candidates), (1,), generator=gen, device=x.device).item()
    return candidates[pick]


def select_positions_low_confidence(x, block_logits, b_s, b_e, mask_id, r, temperature,
                                    logit_start=None):
    """Select positions using LLaDA-style remasking confidence.

    In diffusion decoding, low-confidence remasking means we commit the most
    confident currently masked positions and leave the uncertain ones masked for
    later steps.
    """
    in_block = (x[0, b_s:b_e] == mask_id).nonzero(as_tuple=True)[0] + b_s
    if len(in_block) == 0:
        return None
    n_pick = min(r, len(in_block))
    logit_start = b_s if logit_start is None else logit_start
    pl = block_logits[0, in_block - logit_start].float()
    sampled = torch.argmax(_gumbel(pl, temperature), dim=-1)
    probs = F.softmax(pl, dim=-1)
    confidence = probs[torch.arange(len(in_block), device=in_block.device), sampled]
    _, order = torch.topk(confidence, k=n_pick)
    return in_block[order]


def build_candidates_with_lp(x, block_logits, positions, K, temperature,
                              cand_sampling="gumbel", top_p=0.9,
                              logit_start=0):
    """Returns ([K cand tensors], [K base_logprobs])."""
    pl = block_logits[0, positions - logit_start].float()       # [r, V]
    log_probs = F.log_softmax(pl, dim=-1)
    n_pos = len(positions)

    if cand_sampling == "top_p":
        pl_t = pl / max(temperature, 1e-6) if temperature != 1.0 else pl
        probs = F.softmax(pl_t, dim=-1)
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_p.cumsum(dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        sorted_p_kept = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        sorted_p_kept = sorted_p_kept / sorted_p_kept.sum(dim=-1, keepdim=True)
        probs_filt = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_p_kept)
        probs_flat = probs_filt.unsqueeze(0).expand(K, -1, -1).reshape(K * n_pos, -1)
        tok = torch.multinomial(probs_flat, num_samples=1).view(K, n_pos)
    else:
        tok = torch.argmax(
            _gumbel(pl.unsqueeze(0).expand(K, -1, -1), temperature), dim=-1
        )

    cands = x.expand(K, -1).clone()
    cands[:, positions] = tok
    token_lps = log_probs.unsqueeze(0).expand(K, -1, -1).gather(
        2, tok.unsqueeze(-1)
    ).squeeze(-1)
    lps = token_lps.sum(dim=1).detach().cpu().tolist()
    return cands, lps


def build_candidates_random_positions(x, block_logits, b_s, b_e, mask_id, K, r,
                                      temperature, cand_sampling="gumbel",
                                      top_p=0.9, logit_start=0):
    """Each of K candidates picks its OWN r random MASK positions in the block
    and samples tokens there. Returns ([K cand tensors], [K base_logprobs]).

    Unlike build_candidates_with_lp (shared positions), here every candidate
    can modify a different subset of MASK positions.
    """
    in_block = (x[0, b_s:b_e] == mask_id).nonzero(as_tuple=True)[0] + b_s
    if len(in_block) == 0:
        return None, None
    masked_logits = block_logits[0, in_block - logit_start].float()      # [M, V]
    block_logprobs = F.log_softmax(masked_logits, dim=-1)        # [M, V]
    n_pick = min(r, len(in_block))

    # Vectorized equivalent of K independent randperm(... )[:n_pick].
    # Top-k over iid uniform scores gives a uniform subset without replacement.
    perm_scores = torch.rand(K, len(in_block), device=in_block.device)
    picked = perm_scores.topk(k=n_pick, dim=1).indices
    positions = in_block[picked]                         # [K, n_pick]
    pl = masked_logits[picked]                           # [K, n_pick, V]

    if cand_sampling == "top_p":
        pl_t = pl / max(temperature, 1e-6) if temperature != 1.0 else pl
        probs = F.softmax(pl_t, dim=-1)
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cumsum = sorted_p.cumsum(dim=-1)
        keep = cumsum <= top_p
        keep[..., 0] = True
        sorted_p_kept = torch.where(keep, sorted_p, torch.zeros_like(sorted_p))
        sorted_p_kept = sorted_p_kept / sorted_p_kept.sum(dim=-1, keepdim=True)
        probs_filt = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_p_kept)
        tok = torch.multinomial(
            probs_filt.reshape(K * n_pick, -1), num_samples=1
        ).view(K, n_pick)
    else:
        tok = torch.argmax(_gumbel(pl, temperature), dim=-1)

    cands = x.expand(K, -1).clone()
    cands.scatter_(1, positions, tok)
    lps = block_logprobs[picked, tok].sum(dim=1).detach().cpu().tolist()
    return cands, lps


def deduplicate_candidates(cands, base_lps, b_s, b_e):
    """Deduplicate identical candidate block states before expensive rollouts.

    Returns unique candidates plus an inverse map back to the original K rows.
    Selection still happens over the original K candidates; duplicates share the
    same rollout-derived embedding instead of spending GPU on duplicate rollouts.
    """
    seen = {}
    unique_indices = []
    inverse = []
    for i in range(cands.shape[0]):
        key = tuple(cands[i, b_s:b_e].detach().cpu().tolist())
        if key not in seen:
            seen[key] = len(unique_indices)
            unique_indices.append(i)
        inverse.append(seen[key])

    if len(unique_indices) == cands.shape[0]:
        return cands, base_lps, torch.arange(cands.shape[0], dtype=torch.long), 0

    unique_t = torch.tensor(unique_indices, dtype=torch.long, device=cands.device)
    inverse_t = torch.tensor(inverse, dtype=torch.long, device=cands.device)
    unique_cands = cands.index_select(0, unique_t)
    unique_base_lps = [base_lps[i] for i in unique_indices]
    return unique_cands, unique_base_lps, inverse_t, cands.shape[0] - len(unique_indices)


# ---------------------------------------------------------------------------
# One inner DenMark step
# ---------------------------------------------------------------------------

@torch.no_grad()
def _do_inner_step(x, block_id, _inner, llada, t_eta, t_eta_tok, llada_tok,
                   directions, signs, mask_id, temperature, L_p, block_size,
                   gen_length, r, K, C, B, device, commit_rollout,
                   perturb_temperature=None, rollouts_per_cand=3,
                   rollout_temperature=None, rollout_n_iter=1,
                   cand_sampling="gumbel", top_p=0.9,
                   candidate_selection_seed=0,
                   argmax_logprob_top_frac=1.0,
                   per_cand_positions=False,
                   dedup_candidates=False,
                   position_selection="random",
                   shared_rollout_seeds=False,
                   candidate_region_size=None,
                   score_without_rollout=False,
                   generator_family="llada"):
    b_s = L_p + block_id * block_size
    b_e = min(b_s + block_size, L_p + gen_length)
    cand_s, cand_e = candidate_region_bounds(
        L_p, block_id, block_size, gen_length, _inner, candidate_region_size, r
    )
    dirs_b = directions[block_id]
    signs_b = signs[block_id]

    block_logits = get_model_logits(
        llada, x, generator_family, logit_start=b_s, logit_end=b_e
    )
    sample_temp = perturb_temperature if perturb_temperature is not None else temperature

    if position_selection == "low_confidence":
        # Shared high-confidence positions, matching LLaDA remasking semantics.
        positions = select_positions_low_confidence(
            x, block_logits, cand_s, cand_e, mask_id, r, sample_temp,
            logit_start=b_s,
        )
        if positions is None:
            return x, None
        cands, base_lps = build_candidates_with_lp(
            x, block_logits, positions, K, sample_temp,
            cand_sampling=cand_sampling, top_p=top_p, logit_start=b_s,
        )
    elif per_cand_positions:
        # Each candidate picks its own r random MASK positions.
        cands, base_lps = build_candidates_random_positions(
            x, block_logits, cand_s, cand_e, mask_id, K, r, sample_temp,
            cand_sampling=cand_sampling, top_p=top_p, logit_start=b_s,
        )
        if cands is None:
            return x, None
    else:
        # Shared positions: pick r positions once, all K candidates use them.
        positions = select_positions_in_block(
            x, cand_s, 0, cand_e - cand_s, cand_e - cand_s, r, mask_id
        )
        if positions is None:
            return x, None
        cands, base_lps = build_candidates_with_lp(
            x, block_logits, positions, K, sample_temp,
            cand_sampling=cand_sampling, top_p=top_p, logit_start=b_s,
        )

    K_orig = cands.shape[0]
    if dedup_candidates:
        rollout_cands, _rollout_lps, inverse_idx, duplicate_count = deduplicate_candidates(
            cands, base_lps, b_s, b_e
        )
    else:
        rollout_cands = cands
        inverse_idx = torch.arange(K_orig, dtype=torch.long, device=cands.device)
        duplicate_count = 0

    R = max(1, rollouts_per_cand)
    score_R = R
    n_rollout_cands = rollout_cands.shape[0]
    cands_to_score = (
        rollout_cands.repeat_interleave(score_R, dim=0)
        if score_R > 1
        else rollout_cands
    )

    rollout_temp = rollout_temperature if rollout_temperature is not None else temperature
    rollout_noise_seed = (
        int(candidate_selection_seed) * 1_000_003
        + int(block_id) * 1009
        + int(_inner)
    ) & 0x7FFFFFFF
    if score_without_rollout:
        completed_rows = cands_to_score.to(device).clone()
        block_view = completed_rows[:, b_s:b_e]
        is_mask = block_view == mask_id
        current_logits = block_logits[0].float()
        if rollout_temp == 0:
            sampled = current_logits.argmax(dim=-1).unsqueeze(0).expand(completed_rows.shape[0], -1)
        else:
            logits_for_rows = current_logits.unsqueeze(0).expand(completed_rows.shape[0], -1, -1)
            if shared_rollout_seeds and R > 1:
                noise = torch.empty_like(logits_for_rows)
                rollout_ids = torch.arange(completed_rows.shape[0], device=device) % R
                for rollout_i in rollout_ids.unique(sorted=True).tolist():
                    gen = torch.Generator(device=device)
                    seed = (
                        int(rollout_noise_seed) * 1_000_003
                        + int(rollout_i)
                    ) & 0x7FFFFFFF
                    gen.manual_seed(seed)
                    shared_noise = torch.rand(
                        current_logits.shape,
                        generator=gen,
                        device=device,
                        dtype=logits_for_rows.dtype,
                    )
                    noise[rollout_ids == int(rollout_i)] = shared_noise
            else:
                noise = torch.rand_like(logits_for_rows)
            sampled = (
                logits_for_rows
                - torch.log(-torch.log(noise + 1e-20) + 1e-20) * rollout_temp
            ).argmax(dim=-1)
        completed_rows[:, b_s:b_e] = torch.where(is_mask, sampled, block_view)
        completed = completed_rows.detach().cpu()
    else:
        completed = complete_block_one_shot(
            cands_to_score.to(device), llada, mask_id, rollout_temp,
            L_p, block_id, block_size, gen_length, batch_sz=16,
            n_iter=rollout_n_iter,
            shared_noise_group_size=R if shared_rollout_seeds else 0,
            shared_noise_seed=rollout_noise_seed,
            generator_family=generator_family,
        )

    block_token_rows = completed[:, b_s:b_e].tolist()
    special_token_ids = {
        int(token_id)
        for token_id in (getattr(llada_tok, "all_special_ids", None) or [])
    }
    special_token_ids.add(int(mask_id))
    block_token_rows = [
        [t for t in toks if int(t) not in special_token_ids]
        for toks in block_token_rows
    ]
    block_texts = [
        txt.strip() or "[empty]"
        for txt in safe_batch_decode(llada_tok, block_token_rows, skip_special_tokens=True)
    ]

    embs_all = encode_texts(block_texts, t_eta, t_eta_tok, device, to_cpu=False)
    if score_R > 1:
        embs_unique = embs_all.view(n_rollout_cands, score_R, -1).mean(dim=1)
    else:
        embs_unique = embs_all
    embs = embs_unique.index_select(0, inverse_idx.to(embs_unique.device))

    step_key = block_id * 1000 + _inner
    start = (step_key * C) % B
    chan_idx = (start + torch.arange(C, device=embs.device)) % B

    dirs_sel = dirs_b.to(embs.device, non_blocking=True).index_select(0, chan_idx)
    signs_sel = signs_b.to(embs.device, non_blocking=True).index_select(0, chan_idx)
    proj = embs @ dirs_sel.T
    signed = proj * signs_sel.unsqueeze(0)
    base_lps_t = torch.tensor(base_lps, device=signed.device)
    sel, sel_info = select_candidate(
        selector="max_watermark",
        base_logprobs=base_lps_t,
        signed_scores=signed,
        argmax_logprob_top_frac=argmax_logprob_top_frac,
    )

    if commit_rollout:
        new_x = completed[sel:sel + 1].to(device).clone()
    else:
        new_x = cands[sel:sel + 1].clone()

    # Selected candidate's rollout signal — what the algorithm THOUGHT this commit
    # would contribute to the watermark (mean over C channels of signed projection).
    gen_signed_mean = float(signed[sel].mean())
    gen_signed_max  = float(signed[sel].max())
    gen_signed_min  = float(signed[sel].min())
    best_signed_mean = float(signed.mean(dim=1).max())

    diag = {
        "block_id": block_id, "inner_step": _inner,
        "gen_signed_mean": gen_signed_mean,
        "gen_signed_max":  gen_signed_max,
        "gen_signed_min":  gen_signed_min,
        "best_signed_mean": best_signed_mean,
        "dedup_candidates": bool(dedup_candidates),
        "candidate_unique_count": int(n_rollout_cands),
        "candidate_duplicate_count": int(duplicate_count),
        "rollout_batch_size": int(n_rollout_cands * score_R),
        "score_without_rollout": bool(score_without_rollout),
        "rollout_source": "current_logits_no_extra_forward" if score_without_rollout else "candidate_conditioned_forward",
        "position_selection": position_selection,
        "shared_rollout_seeds": bool(shared_rollout_seeds),
        "candidate_region_size": None if not candidate_region_size else int(candidate_region_size),
        "candidate_region_start": int(cand_s - b_s),
        "candidate_region_end": int(cand_e - b_s),
        "generator_family": generator_family,
    }
    diag.update(sel_info)
    # Always provide 'fallback' even for selectors that never fall back.
    diag.setdefault("fallback", False)
    return new_x, diag


# ---------------------------------------------------------------------------
# Full-sample generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_one(
    prompt_ids, llada, t_eta, t_eta_tok, llada_tok,
    directions, signs, mask_id, gen_length, temperature,
    block_size, r, K, C, device,
    commit_rollout=False,
    perturb_temperature=None, rollouts_per_cand=3,
    rollout_temperature=None, rollout_n_iter=1,
    cand_sampling="gumbel", top_p=0.9,
    candidate_selection_seed=0,
    per_cand_positions=False,
    rollout_schedule="linear_decay",
    dedup_candidates=False,
    position_selection="random",
    shared_rollout_seeds=False,
    decode_schedule="sequential",
    generator_family="llada",
    decode_block_size=None,
    candidate_region_size=None,
    score_without_rollout=False,
    argmax_logprob_top_frac=1.0,
):
    if argmax_logprob_top_frac <= 0.0 or argmax_logprob_top_frac > 1.0:
        raise ValueError("argmax_logprob_top_frac must be in (0, 1]")
    decode_block_size = int(decode_block_size or block_size)
    if decode_block_size < block_size:
        raise ValueError("decode_block_size must be >= block_size")
    if decode_block_size % block_size != 0 and decode_block_size < gen_length:
        raise ValueError(
            "decode_block_size must be a multiple of block_size unless it spans "
            "the complete generation canvas"
        )
    L_p = prompt_ids.shape[1]
    num_blocks = math.ceil(gen_length / block_size)
    num_decode_blocks = math.ceil(gen_length / decode_block_size)
    B = directions.shape[1]

    x = torch.full((1, L_p + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :L_p] = prompt_ids
    diag = []

    def run_step(block_id, inner_step):
        b_s = L_p + block_id * block_size
        b_e = min(b_s + block_size, L_p + gen_length)
        block_len = max(1, b_e - b_s)
        effective_rollouts = max(1, rollouts_per_cand)
        if rollout_schedule in (None, "none", "constant"):
            pass
        elif rollout_schedule == "linear_decay":
            remaining = int((x[0, b_s:b_e] == mask_id).sum().item())
            target_avg = max(1, rollouts_per_cand)
            if block_len <= 1:
                raw_rollouts = float(target_avg)
            else:
                raw_rollouts = 1.0 + 2.0 * (target_avg - 1) * (remaining - 1) / (block_len - 1)
            effective_rollouts = max(1, int(math.floor(raw_rollouts + 0.5)))
        else:
            raise ValueError(f"unknown rollout_schedule: {rollout_schedule}")

        return _do_inner_step(
            x, block_id, inner_step, llada, t_eta, t_eta_tok, llada_tok,
            directions, signs, mask_id, temperature, L_p, block_size,
            gen_length, r, K, C, B, device, commit_rollout,
            perturb_temperature, effective_rollouts,
            rollout_temperature, rollout_n_iter,
            cand_sampling, top_p,
            candidate_selection_seed=candidate_selection_seed,
            argmax_logprob_top_frac=argmax_logprob_top_frac,
            per_cand_positions=per_cand_positions,
            dedup_candidates=dedup_candidates,
            position_selection=position_selection,
            shared_rollout_seeds=shared_rollout_seeds,
            candidate_region_size=candidate_region_size,
            score_without_rollout=score_without_rollout,
            generator_family=generator_family,
        ), effective_rollouts

    if decode_schedule == "sequential":
        inner_by_block = [0 for _ in range(num_blocks)]
        for decode_block_id in range(num_decode_blocks):
            decode_s = decode_block_id * decode_block_size
            decode_e = min(decode_s + decode_block_size, gen_length)
            decode_len = max(1, decode_e - decode_s)
            for local_step in range(math.ceil(decode_len / r) + math.ceil(decode_len / block_size) + 2):
                if decode_block_size == block_size:
                    block_id = decode_block_id
                else:
                    region_seed = (
                        int(candidate_selection_seed) * 1_000_003
                        + int(decode_block_id) * 1009
                        + int(local_step)
                    ) & 0x7FFFFFFF
                    block_id = choose_random_active_semantic_block_in_decode_block(
                        x, L_p, decode_block_id, decode_block_size, block_size,
                        gen_length, r, mask_id, seed=region_seed,
                    )
                    if block_id is None:
                        break
                inner_step = inner_by_block[block_id]
                (x, d), effective_rollouts = run_step(block_id, inner_step)
                inner_by_block[block_id] += 1
                if d is None:
                    if decode_block_size == block_size:
                        break
                    continue
                d["decode_block_id"] = decode_block_id
                d["decode_block_size"] = decode_block_size
                d["semantic_block_id"] = block_id
                d["semantic_block_size"] = block_size
                d["rollouts_per_cand_effective"] = effective_rollouts
                d["rollouts_per_cand_target_avg"] = rollouts_per_cand
                d["rollout_schedule"] = rollout_schedule or "none"
                d["decode_schedule"] = decode_schedule
                diag.append(d)
    elif decode_schedule == "random_block":
        inner_by_block = [0 for _ in range(num_blocks)]
        max_steps = gen_length + num_blocks
        for _global_step in range(max_steps):
            if decode_block_size == block_size:
                block_id = choose_random_active_block(
                    x, L_p, num_blocks, block_size, gen_length, r, mask_id
                )
                decode_block_id = block_id
            else:
                decode_block_id = choose_random_active_block(
                    x, L_p, num_decode_blocks, decode_block_size, gen_length, r, mask_id
                )
                if decode_block_id is None:
                    break
                block_id = choose_random_active_semantic_block_in_decode_block(
                    x, L_p, decode_block_id, decode_block_size, block_size,
                    gen_length, r, mask_id,
                )
            if block_id is None:
                break
            inner_step = inner_by_block[block_id]
            (x, d), effective_rollouts = run_step(block_id, inner_step)
            inner_by_block[block_id] += 1
            if d is None:
                continue
            d["global_step"] = _global_step
            d["decode_block_id"] = decode_block_id
            d["decode_block_size"] = decode_block_size
            d["semantic_block_id"] = block_id
            d["semantic_block_size"] = block_size
            d["rollouts_per_cand_effective"] = effective_rollouts
            d["rollouts_per_cand_target_avg"] = rollouts_per_cand
            d["rollout_schedule"] = rollout_schedule or "none"
            d["decode_schedule"] = decode_schedule
            diag.append(d)
    else:
        raise ValueError(f"unknown decode_schedule: {decode_schedule}")

    tokens = x[0, L_p:].tolist()
    text = safe_decode(llada_tok, tokens, skip_special_tokens=True).strip()
    return text, tokens, diag


@torch.no_grad()
def detect_full_doc(text, enc, enc_tok, directions, signs, device):
    """Run a quick full-document score; use the paper detector for calibrated results."""
    inp = enc_tok([text], return_tensors="pt", padding=True,
                  truncation=True, max_length=512).to(device)
    out = enc(**inp)
    attn = inp["attention_mask"].unsqueeze(-1).float()
    emb = F.normalize(
        (out.last_hidden_state * attn).sum(1) / attn.sum(1).clamp(min=1e-9), dim=-1
    )
    num_blocks = directions.shape[0]
    block_scores = []
    for b in range(num_blocks):
        proj = (emb @ directions[b].T.to(device)).squeeze(0)
        block_scores.append((proj * signs[b].to(device)).cpu())
    mean_signed = torch.stack(block_scores, dim=0).mean(0)
    return float(mean_signed.mean()), float(mean_signed.mean() / (mean_signed.std() + 1e-8))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_jsonl", required=True,
                   help="JSONL with 'input_ids' (tokenized prompt) and optional 'prompt'.")
    p.add_argument("--llada_model", required=True, help="Path/name of generator model")
    p.add_argument("--encoder_model", required=True, help="Path to T_eta block encoder")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--gen_length", type=int, default=300)
    p.add_argument("--block_size", type=int, default=25)
    p.add_argument("--decode_block_size", type=int, default=None,
                   help="Outer diffusion decode block size. Defaults to --block_size.")
    p.add_argument("--cand_block_size", type=int, default=1,
                   help="r: number of MASK positions to perturb per inner step")
    p.add_argument("--candidate_region_size", type=int, default=None,
                   help="Restrict candidate positions to a local window inside "
                        "the semantic block; rollout/scoring still use block_size.")
    p.add_argument("--num_candidates", type=int, default=16, help="K")
    p.add_argument("--rollouts_per_cand", type=int, default=3, help="R")
    p.add_argument("--rollout_schedule", default="linear_decay",
                   choices=["none", "constant", "linear_decay"],
                   help="Per-step rollout schedule; linear_decay treats R as the per-block average.")
    p.add_argument("--rollout_n_iter", type=int, default=1)
    p.add_argument("--num_message_bits", type=int, default=2, help="B")
    p.add_argument("--channels_per_step", type=int, default=2, help="C")
    p.add_argument("--temperature", type=float, default=0.5, help="LLaDA base temperature")
    p.add_argument("--perturb_temperature", type=float, default=0.6, help="pT")
    p.add_argument("--rollout_temperature", type=float, default=0.5, help="rT")
    p.add_argument("--cand_sampling", default="gumbel", choices=["gumbel", "top_p"])
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--commit_rollout", action="store_true",
                   help="Commit full rollout (not just r positions)")
    p.add_argument("--direction_seed", type=int, default=42)
    p.add_argument("--message_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=None,
                   help="Optional global torch RNG seed for generation sampling.")
    p.add_argument("--mask_id", type=int, default=None,
                   help="MASK token id. Defaults to tokenizer.mask_token_id, then a family-specific fallback.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device_map", default=None,
                   help="HF device_map for generator loading (default: same as --device; use 'auto' for sharded 26B loads).")
    p.add_argument("--candidate_selection_seed", type=int, default=0,
                   help="Seed used for deterministic rollout coupling.")
    p.add_argument("--argmax_logprob_top_frac", type=float, default=1.0,
                   help="Optional quality ablation: restrict selection to the top base-logprob fraction.")
    p.add_argument("--dedup_candidates", action=argparse.BooleanOptionalAction, default=True,
                   help="Roll out identical candidate block states only once.")
    p.add_argument("--per_cand_positions", action=argparse.BooleanOptionalAction, default=True,
                   help="Each candidate picks its own perturbed positions inside the chosen block.")
    p.add_argument("--position_selection", default="random",
                   choices=["random", "low_confidence"],
                   help="How to choose MASK positions before candidate sampling.")
    p.add_argument("--shared_rollout_seeds", action=argparse.BooleanOptionalAction, default=True,
                   help="Use common random Gumbel noise across candidates for each rollout index.")
    p.add_argument("--score_without_rollout_forward", action="store_true",
                   help="Ablation: complete candidate blocks from current logits instead of doing the extra candidate-conditioned rollout forward before scoring.")
    p.add_argument("--decode_schedule", default="sequential",
                   choices=["sequential", "random_block"],
                   help="Order for committing MASK positions; random_block samples one active block per step.")
    p.add_argument("--generator_family", default="llada", choices=GENERATOR_FAMILIES,
                   help="Generator logits/forward adapter.")
    p.add_argument("--retokenize_prompts", action="store_true",
                   help="Tokenize the JSONL 'prompt' text with the generator tokenizer instead of using input_ids.")
    return p.parse_args()


def main():
    family_parser = argparse.ArgumentParser(add_help=False)
    family_parser.add_argument("--generator_family", default="llada")
    family, _ = family_parser.parse_known_args()
    if family.generator_family == "dream":
        from denmark.core.model import generate_dream
        return generate_dream()
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer
    if args.argmax_logprob_top_frac <= 0.0 or args.argmax_logprob_top_frac > 1.0:
        raise ValueError("--argmax_logprob_top_frac must be in (0, 1]")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    num_blocks = math.ceil(args.gen_length / args.block_size)
    dirs, signs = build_directions(
        num_blocks, args.num_message_bits, args.direction_seed, args.message_seed
    )

    print(f"Loading generator model from {args.llada_model}")
    llada = load_generator_model(
        args.llada_model,
        generator_family=args.generator_family,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map or args.device,
        trust_remote_code=True,
    )
    llada_tok = AutoTokenizer.from_pretrained(args.llada_model, trust_remote_code=True)
    mask_id = resolve_mask_id(llada_tok, args.generator_family, args.mask_id)

    print(f"Loading T_eta encoder from {args.encoder_model}")
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_model)
    enc = AutoModel.from_pretrained(
        args.encoder_model, torch_dtype=torch.float32,
    ).to(args.device).eval()

    with open(args.prompts_jsonl) as f:
        all_prompts = [json.loads(l) for l in f if l.strip()]
    prompts = all_prompts[args.offset:args.offset + args.num_samples]
    print(f"Generating {len(prompts)} samples")
    print(f"  bs={args.block_size}  decode_bs={args.decode_block_size or args.block_size}  "
          f"r={args.cand_block_size}  K={args.num_candidates}  "
          f"R={args.rollouts_per_cand}  B={args.num_message_bits}  C={args.channels_per_step}")
    print(f"  T={args.temperature}  pT={args.perturb_temperature}  "
          f"rT={args.rollout_temperature}  rollout_schedule={args.rollout_schedule}")
    print(f"  decode_schedule={args.decode_schedule}  generator_family={args.generator_family}  mask_id={mask_id}")
    print(f"  argmax_logprob_top_frac={args.argmax_logprob_top_frac}")
    print(f"  candidate_region_size={args.candidate_region_size}")
    print("  selector=max_watermark  selection_mode=argmax_score")

    with open(args.output, "w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(tqdm(prompts, desc="gen")):
            generation_started = time.perf_counter()
            if args.retokenize_prompts or "input_ids" not in rec:
                prompt_text = rec.get("prompt") or rec.get("input") or rec.get("text") or ""
                prompt_ids = llada_tok(prompt_text, return_tensors="pt").input_ids.to(args.device)
            else:
                prompt_ids = torch.tensor(
                    rec["input_ids"], dtype=torch.long, device=args.device,
                ).unsqueeze(0)

            sample_id = args.offset + idx
            text, tokens, diagnostics = generate_one(
                prompt_ids, llada, enc, enc_tok, llada_tok,
                dirs, signs, mask_id, args.gen_length, args.temperature,
                args.block_size, args.cand_block_size,
                args.num_candidates, args.channels_per_step, args.device,
                commit_rollout=args.commit_rollout,
                perturb_temperature=args.perturb_temperature,
                rollouts_per_cand=args.rollouts_per_cand,
                rollout_temperature=args.rollout_temperature,
                rollout_n_iter=args.rollout_n_iter,
                cand_sampling=args.cand_sampling, top_p=args.top_p,
                candidate_selection_seed=args.candidate_selection_seed,
                rollout_schedule=args.rollout_schedule,
                dedup_candidates=args.dedup_candidates,
                per_cand_positions=args.per_cand_positions,
                position_selection=args.position_selection,
                shared_rollout_seeds=args.shared_rollout_seeds,
                score_without_rollout=args.score_without_rollout_forward,
                decode_schedule=args.decode_schedule,
                generator_family=args.generator_family,
                decode_block_size=args.decode_block_size,
                candidate_region_size=args.candidate_region_size,
                argmax_logprob_top_frac=args.argmax_logprob_top_frac,
            )

            generation_seconds = time.perf_counter() - generation_started
            det, z = detect_full_doc(text, enc, enc_tok, dirs, signs, args.device)

            out_f.write(json.dumps({
                "prompt_idx": args.offset + idx,
                "sample_id": sample_id,
                "watermark_sample_id": sample_id,
                "prompt": rec.get("prompt", ""),
                "text": text,
                "generated_token_ids": tokens,
                "wm_gen_diag": diagnostics,
                "det_score": det,
                "z_score": z,
                "generation_seconds": generation_seconds,
                "seconds_per_visible_token": generation_seconds / max(1, len(tokens)),
                "watermark_config": {
                    "schema_version": 1,
                    "generator_model": args.llada_model,
                    "generator_family": args.generator_family,
                    "encoder_model": args.encoder_model,
                    "mask_id": mask_id,
                    "gen_length": args.gen_length,
                    "block_size": args.block_size,
                    "decode_block_size": args.decode_block_size or args.block_size,
                    "decode_schedule": args.decode_schedule,
                    "cand_block_size": args.cand_block_size,
                    "candidate_region_size": args.candidate_region_size,
                    "num_candidates": args.num_candidates,
                    "num_message_bits": args.num_message_bits,
                    "channels_per_step": args.channels_per_step,
                    "rollouts_per_cand": args.rollouts_per_cand,
                    "rollout_schedule": args.rollout_schedule,
                    "rollout_n_iter": args.rollout_n_iter,
                    "temperature": args.temperature,
                    "perturb_temperature": args.perturb_temperature,
                    "rollout_temperature": args.rollout_temperature,
                    "cand_sampling": args.cand_sampling,
                    "top_p": args.top_p,
                    "position_selection": args.position_selection,
                    "per_cand_positions": args.per_cand_positions,
                    "dedup_candidates": args.dedup_candidates,
                    "shared_rollout_seeds": args.shared_rollout_seeds,
                    "commit_rollout": args.commit_rollout,
                    "score_without_rollout_forward": args.score_without_rollout_forward,
                    "selector": "max_watermark",
                    "selection_mode": "argmax_score",
                    "argmax_logprob_top_frac": args.argmax_logprob_top_frac,
                    "direction_seed": args.direction_seed,
                    "message_seed": args.message_seed,
                    "candidate_selection_seed": args.candidate_selection_seed,
                    "global_seed": args.seed,
                },
            }, ensure_ascii=False) + "\n")
            out_f.flush()
    print(f"Done. {len(prompts)} samples -> {args.output}")


if __name__ == "__main__":
    main()
