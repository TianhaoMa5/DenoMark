"""
DGMark: decoding-guided watermarking for diffusion language models.

This adapts the core idea from pyomin/dgmark-watermarking for the local LLaDA
generation interface: keep model token probabilities intact, but prefer
unmasking positions whose predicted token satisfies a parity condition.
"""
from __future__ import annotations

import math
import random
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F

from denomark.baselines.common import _add_gumbel_noise, _get_num_transfer_tokens
from denomark.core.model import get_model_logits, safe_decode


_KEY_SEQUENCE_CACHE = {}


def generate_key_sequence(private_key: Union[int, str], max_length: int = 10000) -> list[int]:
    seed = sum(ord(c) for c in private_key) if isinstance(private_key, str) else int(private_key)
    rng = random.Random(seed)
    return [rng.randint(0, 1) for _ in range(max_length)]


def get_key_based_parity(position: int, private_key: Optional[Union[int, str]] = None) -> int:
    if private_key is None:
        return position % 2

    cache_key = str(private_key)
    if cache_key not in _KEY_SEQUENCE_CACHE:
        _KEY_SEQUENCE_CACHE[cache_key] = generate_key_sequence(private_key)
    key_sequence = _KEY_SEQUENCE_CACHE[cache_key]
    return key_sequence[(position - 1) % len(key_sequence)]


def check_dgmark_compliance(
    position: int,
    token_id: int,
    private_key: Optional[Union[int, str]] = None,
) -> bool:
    expected_parity = get_key_based_parity(position, private_key)
    token_parity = int(token_id) % 2

    if private_key is not None:
        cache_key = str(private_key)
        if cache_key not in _KEY_SEQUENCE_CACHE:
            _KEY_SEQUENCE_CACHE[cache_key] = generate_key_sequence(private_key)
        key_bit = _KEY_SEQUENCE_CACHE[cache_key][(position - 1) % len(_KEY_SEQUENCE_CACHE[cache_key])]
        token_parity ^= key_bit

    return expected_parity == token_parity


def _compliant_token_parity(
    position: int,
    private_key: Optional[Union[int, str]] = None,
) -> int:
    """Return the token-id parity that satisfies DGMark at a 1-based position."""
    return 0 if check_dgmark_compliance(position, 0, private_key) else 1


def _compliant_parity_by_position(
    positions: torch.Tensor,
    private_key: Optional[Union[int, str]] = None,
) -> torch.Tensor:
    vals = [
        _compliant_token_parity(int(pos), private_key)
        for pos in positions.detach().cpu().tolist()
    ]
    return torch.tensor(vals, dtype=torch.long, device=positions.device)


def dgmark_compliance_for_tokens(
    token_ids: torch.Tensor,
    private_key: Optional[Union[int, str]] = None,
) -> torch.Tensor:
    """Vectorized DGMark compliance for token predictions at sequence positions.

    token_ids is [B, L]. Sequence positions are treated as 1-based absolute
    positions, matching _score_dgmark_tokens(prompt_len + offset + 1, token).
    """
    seq_len = token_ids.shape[1]
    positions = torch.arange(1, seq_len + 1, device=token_ids.device)
    wanted = _compliant_parity_by_position(positions, private_key)
    return (token_ids.long() % 2) == wanted.unsqueeze(0)


def apply_dgmark_logit_bias(
    logits: torch.Tensor,
    *,
    target_mask: torch.Tensor,
    delta: float,
    private_key: Optional[Union[int, str]] = None,
) -> torch.Tensor:
    """Add a parity bias at target positions, used by DREAM origin decoding.

    This is the DREAM-compatible DGMark adapter: ETH-style DREAM origin chooses
    transfer positions randomly, so the position-order signal cannot be imposed
    directly. Biasing logits lets randomly accepted tokens still carry DGMark's
    parity signal while preserving DREAM's origin remasking/acceptance loop.
    """
    if float(delta) == 0.0 or not bool(target_mask.any().item()):
        return logits
    out = logits.clone()
    vocab_parity = torch.arange(out.shape[-1], device=out.device) % 2
    rows, pos = target_mask.nonzero(as_tuple=True)
    for cur_pos in pos.unique(sorted=True):
        row_sel = rows[pos == cur_pos]
        wanted = _compliant_token_parity(int(cur_pos.item()) + 1, private_key)
        tok_sel = vocab_parity == int(wanted)
        tok_idx = tok_sel.nonzero(as_tuple=True)[0]
        for row in row_sel.detach().cpu().tolist():
            out[int(row), int(cur_pos.item()), tok_idx] += float(delta)
    return out


class DGMarkLogitsBias:
    """Callable logits hook for DREAM/ETH-origin diffusion_generate loops."""

    def __init__(
        self,
        *,
        mask_id: int,
        delta: float = 2.0,
        private_key: Optional[Union[int, str]] = None,
        target_mask_getter=None,
    ) -> None:
        self.mask_id = int(mask_id)
        self.delta = float(delta)
        self.private_key = private_key
        self.target_mask_getter = target_mask_getter

    def __call__(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        target_mask = x == self.mask_id
        if self.target_mask_getter is not None:
            extra = self.target_mask_getter()
            if extra is not None:
                target_mask = target_mask & extra.to(device=x.device, dtype=torch.bool)
        return apply_dgmark_logit_bias(
            logits,
            target_mask=target_mask,
            delta=self.delta,
            private_key=self.private_key,
        )


def _top_k_sample(logits: torch.Tensor, k: int) -> torch.Tensor:
    _, _, vocab_size = logits.shape
    k = min(int(k), vocab_size)
    values, indices = torch.topk(logits, k=k, dim=-1)
    probs = F.softmax(values, dim=-1)
    sampled_idx = torch.multinomial(probs.reshape(-1, k), num_samples=1).squeeze(-1)
    chosen = indices.reshape(-1, k)[torch.arange(indices.numel() // k, device=logits.device), sampled_idx]
    return chosen.view(logits.shape[:2])


def _select_tokens(
    logits_with_noise: torch.Tensor,
    sampling_strategy: Literal["greedy", "multinomial"],
    top_k: int,
) -> torch.Tensor:
    if sampling_strategy == "greedy":
        return torch.argmax(logits_with_noise, dim=-1)
    if sampling_strategy == "multinomial":
        return _top_k_sample(logits_with_noise, top_k)
    raise ValueError(f"Unknown DGMark sampling strategy: {sampling_strategy}")


def _score_dgmark_tokens(
    token_ids: list[int],
    prompt_len: int,
    private_key: Optional[Union[int, str]] = None,
    eot_token_id: Optional[int] = None,
) -> dict:
    trimmed_ids = token_ids
    if eot_token_id is not None:
        try:
            trimmed_ids = token_ids[: token_ids.index(eot_token_id) + 1]
        except ValueError:
            pass

    matches = 0
    for offset, token_id in enumerate(trimmed_ids):
        real_pos = prompt_len + offset + 1
        if check_dgmark_compliance(real_pos, token_id, private_key):
            matches += 1

    n = len(trimmed_ids)
    ratio = matches / n if n else 0.0
    z_score = (matches - 0.5 * n) / math.sqrt(max(0.25 * n, 1e-12))
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
    return {
        "matched_count": int(matches),
        "trimmed_length": int(n),
        "match_ratio": float(ratio),
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def _dgmark_window_scores(
    token_ids: list[int],
    prompt_len: int,
    window_size: int = 8,
    private_key: Optional[Union[int, str]] = None,
) -> dict:
    if len(token_ids) < window_size:
        return {"window_size": window_size, "n_windows": 0, "agg_z": 0.0}

    ratios = []
    for start in range(len(token_ids) - window_size + 1):
        window = token_ids[start:start + window_size]
        matches = 0
        for offset, token_id in enumerate(window):
            real_pos = prompt_len + start + offset + 1
            if check_dgmark_compliance(real_pos, token_id, private_key):
                matches += 1
        ratios.append(matches / window_size)

    ratio_t = torch.tensor(ratios, dtype=torch.float32)
    var = 0.5 * 0.5 / window_size
    z_scores = (ratio_t - 0.5) / math.sqrt(var)
    return {
        "window_size": int(window_size),
        "n_windows": int(len(ratios)),
        "agg_z": float((z_scores.square()).mean().item()),
    }


@torch.no_grad()
def llada_generate_dgmark_aligned(
    prompt: torch.Tensor,
    llada,
    llada_tok,
    mask_id: int,
    gen_length: int = 300,
    block_size: int = 25,
    steps: Optional[int] = None,
    temperature: float = 0.5,
    cfg_scale: float = 0.0,
    remasking: Literal["low_confidence", "random", "ar", "none"] = "low_confidence",
    sampling_strategy: Literal["greedy", "multinomial"] = "greedy",
    top_k: int = 3,
    beam_size: int = 1,
    private_key: Optional[Union[int, str]] = None,
    window_size: int = 8,
    eot_token_id: Optional[int] = None,
    generator_family: str = "llada",
    dgmark_position_bonus: float = 1e6,
):
    """Generate DGMark with the same block transfer schedule as local LLaDA.

    The model token sample is drawn first. DGMark only changes which masked
    positions are committed by boosting positions whose sampled token satisfies
    the parity rule, matching the decoding-order interpretation of DGMark while
    keeping the local LLaDA/hash-style schedule intact.
    """
    assert gen_length % block_size == 0, "gen_length must be divisible by block_size"
    num_blocks = gen_length // block_size
    if steps is None:
        steps = gen_length
    assert steps % num_blocks == 0, f"steps={steps} must be divisible by num_blocks={num_blocks}"
    steps_per_block = steps // num_blocks

    prompt_len = prompt.shape[1]
    x = torch.full(
        (prompt.shape[0], prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=llada.device,
    )
    x[:, :prompt_len] = prompt.clone()
    prompt_index = x != mask_id
    if eot_token_id is None:
        eot_token_id = getattr(llada_tok, "eos_token_id", None)

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = prompt_len + (block_idx + 1) * block_size
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
            block_mask_index = x[:, block_start:block_end] == mask_id
            if not bool(block_mask_index.any().item()):
                break

            mask_index = x == mask_id
            logits_kwargs = (
                {"attention_boundary": block_start}
                if generator_family == "llada2"
                else {}
            )
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = get_model_logits(llada, x_, generator_family, **logits_kwargs)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = get_model_logits(llada, x, generator_family, **logits_kwargs)

            logits_with_noise = _add_gumbel_noise(logits, temperature)
            x0 = _select_tokens(logits_with_noise, sampling_strategy, top_k)

            if remasking == "low_confidence":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking in {"ar", "none"}:
                x0_p = (
                    -torch.arange(x0.shape[1], device=x0.device).unsqueeze(0)
                    .repeat(x0.shape[0], 1)
                    / x0.shape[1]
                )
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            block_candidates = torch.zeros_like(x0, dtype=torch.bool)
            block_candidates[:, block_start:block_end] = mask_index[:, block_start:block_end]

            for row_idx in range(x0.shape[0]):
                k = int(num_transfer_tokens[row_idx, step_idx].item())
                if k <= 0:
                    continue

                active_pos = block_candidates[row_idx].nonzero(as_tuple=False).squeeze(-1)
                if active_pos.numel() == 0:
                    continue

                compliant = dgmark_compliance_for_tokens(x0[row_idx:row_idx + 1], private_key)[0]
                matched_pos = active_pos[compliant[active_pos]]
                unmatched_pos = active_pos[~compliant[active_pos]]

                if k == 1 and remasking == "low_confidence" and matched_pos.numel() and beam_size > 1:
                    beam_pos = matched_pos[
                        torch.argsort(x0_p[row_idx, matched_pos], descending=True)[: int(beam_size)]
                    ]
                    best_pos = None
                    best_match_count = -1
                    for pos_t in beam_pos:
                        pos = int(pos_t.item())
                        x_sim = x.clone()
                        x_sim[row_idx, pos] = x0[row_idx, pos]
                        logits_sim = get_model_logits(
                            llada,
                            x_sim,
                            generator_family,
                            **logits_kwargs,
                        )
                        pred_ids_sim = _select_tokens(
                            _add_gumbel_noise(logits_sim, temperature),
                            sampling_strategy,
                            top_k,
                        )
                        future_pos = active_pos[active_pos != pos_t]
                        future_tokens = pred_ids_sim[row_idx, future_pos]
                        future_ok = dgmark_compliance_for_tokens(
                            pred_ids_sim[row_idx:row_idx + 1], private_key
                        )[0, future_pos]
                        valid_future = (future_tokens != eot_token_id) & (future_tokens != mask_id)
                        next_match_count = int((future_ok & valid_future).sum().item())
                        if next_match_count > best_match_count:
                            best_match_count = next_match_count
                            best_pos = pos
                    if best_pos is not None:
                        transfer_index[row_idx, best_pos] = True
                    continue

                score = x0_p[row_idx].clone()
                score[:block_start] = -float("inf")
                score[block_end:] = -float("inf")
                score[~mask_index[row_idx]] = -float("inf")
                if float(dgmark_position_bonus) != 0.0:
                    score = score + compliant.to(score.dtype) * float(dgmark_position_bonus)
                _, selected = torch.topk(score, k=min(k, int(active_pos.numel())))
                transfer_index[row_idx, selected] = True
            x[transfer_index] = x0[transfer_index]

    tokens = x[0, prompt_len:].tolist()
    text = safe_decode(llada_tok, tokens, skip_special_tokens=True).strip()
    detection = _score_dgmark_tokens(tokens, prompt_len, private_key, eot_token_id=eot_token_id)
    detection.update(_dgmark_window_scores(tokens, prompt_len, window_size, private_key))
    return text, tokens, detection


@torch.no_grad()
def llada_generate_dgmark(
    prompt: torch.Tensor,
    llada,
    llada_tok,
    mask_id: int,
    gen_length: int = 300,
    block_size: int = 25,
    steps: Optional[int] = None,
    temperature: float = 0.5,
    cfg_scale: float = 0.0,
    remasking: Literal["low_confidence", "random"] = "low_confidence",
    sampling_strategy: Literal["greedy", "multinomial"] = "greedy",
    top_k: int = 3,
    beam_size: int = 1,
    private_key: Optional[Union[int, str]] = None,
    window_size: int = 8,
    eot_token_id: Optional[int] = None,
    generator_family: str = "llada",
):
    """Generate with DGMark's parity-guided unmasking order."""
    assert gen_length % block_size == 0, "gen_length must be divisible by block_size"
    num_blocks = gen_length // block_size
    if steps is None:
        steps = gen_length
    assert steps % num_blocks == 0, f"steps={steps} must be divisible by num_blocks={num_blocks}"
    steps_per_block = steps // num_blocks

    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=llada.device)
    x[:, :prompt_len] = prompt.clone()
    prompt_index = x != mask_id
    if eot_token_id is None:
        eot_token_id = getattr(llada_tok, "eos_token_id", None)

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = prompt_len + (block_idx + 1) * block_size

        for _ in range(steps_per_block):
            mask_index = x == mask_id
            block_mask = mask_index[0, block_start:block_end]
            gen_positions = block_mask.nonzero(as_tuple=False).squeeze(-1) + block_start
            if gen_positions.numel() == 0:
                break

            logits_kwargs = (
                {"attention_boundary": block_start}
                if generator_family == "llada2"
                else {}
            )
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = get_model_logits(llada, x_, generator_family, **logits_kwargs)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = get_model_logits(llada, x, generator_family, **logits_kwargs)

            logits_with_noise = _add_gumbel_noise(logits, temperature)
            x0 = _select_tokens(logits_with_noise, sampling_strategy, top_k)

            if remasking == "low_confidence":
                probs = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(probs, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                matched = []
                unmatched = []

                for pos_t in gen_positions:
                    pos = int(pos_t.item())
                    token_id = int(x0[0, pos].item())
                    score = float(x0_p[0, pos].item())
                    if check_dgmark_compliance(pos + 1, token_id, private_key):
                        matched.append((pos, score))
                    else:
                        unmatched.append((pos, score))

                matched.sort(key=lambda item: item[1], reverse=True)
                unmatched.sort(key=lambda item: item[1], reverse=True)

                if matched and beam_size > 1:
                    best_pos = None
                    best_match_count = -1
                    for pos, _score in matched[:beam_size]:
                        x_sim = x.clone()
                        x_sim[0, pos] = x0[0, pos]
                        logits_sim = get_model_logits(
                            llada,
                            x_sim,
                            generator_family,
                            **logits_kwargs,
                        )
                        pred_ids_sim = _select_tokens(
                            _add_gumbel_noise(logits_sim, temperature),
                            sampling_strategy,
                            top_k,
                        )

                        next_match_count = 0
                        for future_pos_t in gen_positions:
                            future_pos = int(future_pos_t.item())
                            if future_pos == pos:
                                continue
                            future_token = int(pred_ids_sim[0, future_pos].item())
                            if future_token in {eot_token_id, mask_id}:
                                continue
                            if check_dgmark_compliance(future_pos + 1, future_token, private_key):
                                next_match_count += 1
                        if next_match_count > best_match_count:
                            best_match_count = next_match_count
                            best_pos = pos
                    selected_positions = [best_pos]
                elif matched:
                    selected_positions = [matched[0][0]]
                elif unmatched:
                    selected_positions = [unmatched[0][0]]
                else:
                    selected_positions = []
            elif remasking == "random":
                selected_positions = gen_positions.tolist()[:1]
            else:
                raise NotImplementedError(remasking)

            x0 = torch.where(mask_index, x0, x)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for pos in selected_positions:
                if pos is not None:
                    transfer_index[0, int(pos)] = True
            x[transfer_index] = x0[transfer_index]

    tokens = x[0, prompt_len:].tolist()
    text = safe_decode(llada_tok, tokens, skip_special_tokens=True).strip()
    detection = _score_dgmark_tokens(tokens, prompt_len, private_key, eot_token_id=eot_token_id)
    detection.update(_dgmark_window_scores(tokens, prompt_len, window_size, private_key))
    return text, tokens, detection
