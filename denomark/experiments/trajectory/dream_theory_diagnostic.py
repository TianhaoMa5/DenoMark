"""Dream-native theory diagnostics without changing the production sampler.

This module deliberately reuses the Dream runner's aligned-logit, token
sampling, decoding, encoder, and key helpers.  The only native sampler logic
implemented here is a *resumable single step*: Dream's public
``diffusion_generate`` cannot accept an intermediate canvas/timestep because it
always pads a prompt with a fresh mask canvas.  ``continue_pi0_to_unit`` mirrors
that sampler from ``checkpoint_step + 1`` while preserving the saved canvas.
"""
from __future__ import annotations

import copy
import inspect
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from denomark.core.model import encode_texts
from denomark.core.model import (dream_logits, safe_decode_dream as safe_decode, sample_from_logits)


DEFAULT_PREFIXES = (1, 3, 5, 10)


def deterministic_full_unit(prompt_idx: int, gen_length: int, unit_size: int, seed: int) -> int:
    """Choose among complete continuation units only."""
    n_full = int(gen_length) // int(unit_size)
    if n_full <= 0:
        raise ValueError("generation length contains no complete semantic unit")
    return random.Random(int(seed) + int(prompt_idx) * 1_000_003).randrange(n_full)


def unit_bounds(prompt_len: int, unit_id: int, unit_size: int) -> tuple[int, int]:
    start = int(prompt_len) + int(unit_id) * int(unit_size)
    return start, start + int(unit_size)


def unit_progress(x: torch.Tensor, start: int, end: int, mask_id: int) -> tuple[float, int]:
    remaining = int((x[0, start:end] == int(mask_id)).sum().item())
    length = max(1, int(end - start))
    return 1.0 - remaining / length, remaining


@contextmanager
def preserve_all_rng_state():
    """Make a diagnostic branch observational with respect to generation RNG."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def branch_seed(base: int, prompt_idx: int, step: int, branch: int, repeat: int = 0) -> int:
    value = (
        int(base) * 1_000_003
        + int(prompt_idx) * 100_003
        + int(step) * 1_009
        + int(branch) * 10_007
        + int(repeat) * 1_000_000_007
    )
    return value & 0x7FFFFFFF


def _native_sample_tokens(model):
    """Get the exact ``sample_tokens`` function used by loaded Dream remote code."""
    method = getattr(model, "_sample", None)
    func = getattr(method, "__func__", method)
    native = getattr(func, "__globals__", {}).get("sample_tokens") if func else None
    if native is None:
        raise RuntimeError("loaded Dream model does not expose native sample_tokens")
    return native


def validate_native_sampler_compatibility(model) -> dict[str, bool]:
    """Fail closed unless Dream's loaded ``_sample`` matches this adapter.

    ``diffusion_generate`` is remote code, so a future model revision could
    change the native schedule or transfer rule.  The diagnostic must not
    silently call an adapter written for a different sampler.
    """
    method = getattr(model, "_sample", None)
    func = getattr(method, "__func__", method)
    if func is None:
        raise RuntimeError("loaded Dream model does not expose native _sample")
    try:
        compact = "".join(inspect.getsource(func).split())
    except (OSError, TypeError) as exc:
        raise RuntimeError("cannot inspect loaded Dream native _sample source") from exc
    markers = {
        "native_schedule": "torch.linspace(1,eps,steps+1" in compact,
        "origin_transfer": "p_transfer=1-s/t" in compact,
        "post_step_tokens_hook": "generation_tokens_hook_func(i,x,logits)" in compact,
        "maskgit_transfer_count": any(
            marker in compact for marker in (
                "number_transfer_tokens=int(num_mask*(1-s/t))",
                "number_transfer_tokens=int(num_mask_token*(1-s/t))",
            )
        ),
        "native_sample_tokens": "sample_tokens" in getattr(func, "__globals__", {}),
    }
    missing = [name for name, present in markers.items() if not present]
    if missing:
        raise RuntimeError(
            "loaded Dream native sampler is incompatible with the resumable pi0 adapter; "
            f"missing expected semantics: {', '.join(missing)}"
        )
    return markers


@torch.no_grad()
def dream_native_pi0_step(
    model,
    x: torch.Tensor,
    *,
    step: int,
    steps: int,
    eps: float,
    mask_id: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    alg: str,
    alg_temp: float | None,
) -> torch.Tensor:
    """One hook-free Dream native denoising step on an existing canvas.

    The ordering matches Dream's remote ``_sample``: forward/alignment, native
    sampling, and transfer.  There is intentionally no watermark/logits/tokens
    hook and no canvas initialization.
    """
    sample_tokens = _native_sample_tokens(model)
    mask_index = x == int(mask_id)
    if not bool(mask_index.any().item()):
        return x
    logits = dream_logits(model, x)
    mask_logits = logits[mask_index]
    timesteps = torch.linspace(1, float(eps), int(steps) + 1, device=x.device)
    t, s = timesteps[int(step)], timesteps[int(step) + 1]

    if alg == "origin":
        p_transfer = 1 - s / t if int(step) < int(steps) - 1 else 1
        x0 = torch.full_like(x[mask_index], int(mask_id))
        transfer = torch.rand(*x0.shape, device=x.device) < p_transfer
        if bool(transfer.any().item()):
            _, sampled = sample_tokens(
                mask_logits[transfer], temperature=temperature, top_p=top_p, top_k=top_k
            )
            x0[transfer] = sampled
        x[mask_index] = x0
        return x

    kwargs = dict(temperature=temperature, top_p=top_p, top_k=top_k)
    if alg == "maskgit_plus":
        confidence, x0 = sample_tokens(mask_logits, **kwargs)
    elif alg == "topk_margin":
        confidence, x0 = sample_tokens(mask_logits, margin_confidence=True, **kwargs)
    elif alg == "entropy":
        confidence, x0 = sample_tokens(mask_logits, neg_entropy=True, **kwargs)
    else:
        raise ValueError(f"unknown Dream algorithm: {alg}")

    num_mask = mask_index.sum() / mask_index.shape[0]
    n_transfer = int(num_mask * (1 - s / t)) if int(step) < int(steps) - 1 else int(num_mask)
    if n_transfer <= 0:
        return x
    full_conf = torch.full_like(x, -torch.inf, dtype=logits.dtype)
    full_conf[mask_index] = confidence
    if alg_temp is None or float(alg_temp) == 0.0:
        _, transfer_index = torch.topk(full_conf, n_transfer)
    else:
        transfer_index = torch.multinomial(
            F.softmax(full_conf / float(alg_temp), dim=-1), num_samples=n_transfer
        )
    x_ = torch.full_like(x, int(mask_id))
    x_[mask_index] = x0
    rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(transfer_index)
    x[rows, transfer_index] = x_[rows, transfer_index]
    return x


@torch.no_grad()
def continue_pi0_to_unit(
    model,
    candidate_state: torch.Tensor,
    *,
    checkpoint_step: int,
    unit_start: int,
    unit_end: int,
    steps: int,
    eps: float,
    mask_id: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    alg: str,
    alg_temp: float | None,
    downstream_seed: int,
) -> tuple[torch.Tensor, int, int, list[int]]:
    """Resume pi0 from the checkpoint's next native timestep until unit completion."""
    x = candidate_state.clone()
    trace: list[int] = []
    if not bool((x[0, unit_start:unit_end] == int(mask_id)).any().item()):
        return x, 0, int(checkpoint_step), trace
    for native_step in range(int(checkpoint_step) + 1, int(steps)):
        # Re-seeding per native step makes repeat j as paired as possible across
        # candidates while keeping the exact native random-call ordering.
        seed_all((int(downstream_seed) + native_step * 1_000_003) & 0x7FFFFFFF)
        dream_native_pi0_step(
            model, x, step=native_step, steps=steps, eps=eps, mask_id=mask_id,
            temperature=temperature, top_p=top_p, top_k=top_k,
            alg=alg, alg_temp=alg_temp,
        )
        trace.append(native_step)
        if not bool((x[0, unit_start:unit_end] == int(mask_id)).any().item()):
            return x, native_step - int(checkpoint_step), native_step, trace
    remaining = int((x[0, unit_start:unit_end] == int(mask_id)).sum().item())
    raise RuntimeError(
        f"selected semantic unit still has {remaining} masks after Dream schedule ended"
    )


@torch.no_grad()
def keyed_unit_scores(
    states: torch.Tensor,
    *,
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    unit_id: int,
    unit_start: int,
    unit_end: int,
    dream_step: int,
    channels_per_step: int,
    special_ids: set[int],
    device: str,
) -> torch.Tensor:
    texts = [
        safe_decode(tokenizer, row[unit_start:unit_end].detach().cpu().tolist(), special_ids)
        or "[empty]"
        for row in states
    ]
    embs = encode_texts(
        texts, encoder, encoder_tokenizer, device, batch_sz=min(32, len(texts))
    )
    b = min(int(unit_id), int(directions.shape[0]) - 1)
    bits = int(directions.shape[1])
    start = (int(dream_step) * int(channels_per_step)) % bits
    channels = torch.tensor(
        [(start + j) % bits for j in range(int(channels_per_step))], dtype=torch.long
    )
    signed = (embs @ directions[b, channels].T) * signs[b, channels].unsqueeze(0)
    return signed.mean(dim=1).float().cpu()


@torch.no_grad()
def construct_candidates(
    model,
    state: torch.Tensor,
    *,
    unit_start: int,
    unit_end: int,
    mask_id: int,
    num_candidates: int,
    update_size: int,
    candidate_temperature: float,
) -> tuple[torch.Tensor, list[dict]]:
    """Current per-candidate-random-semantic-unit state-conditional policy."""
    masked = (state[0, unit_start:unit_end] == int(mask_id)).nonzero(as_tuple=True)[0]
    masked = masked + int(unit_start)
    if masked.numel() == 0:
        raise ValueError("checkpoint semantic unit has no masked candidate position")
    n_pick = min(int(update_size), int(masked.numel()))
    logits = dream_logits(model, state)
    states, meta = [], []
    for k in range(int(num_candidates)):
        pos = masked[torch.randperm(masked.numel(), device=masked.device)[:n_pick]]
        tok = sample_from_logits(logits[0, pos].float(), float(candidate_temperature))
        cand = state.clone()
        cand[0, pos] = tok
        states.append(cand[0])
        meta.append({
            "candidate_index": k,
            "selected_positions": [int(v) for v in pos.detach().cpu().tolist()],
            "selected_positions_in_unit": [int(v - unit_start) for v in pos.detach().cpu().tolist()],
            "sampled_tokens": [int(v) for v in tok.detach().cpu().tolist()],
            "candidate_state_token_ids": [int(v) for v in cand[0].detach().cpu().tolist()],
        })
    return torch.stack(states, dim=0), meta


@torch.no_grad()
def rollout_raw_scores(
    model,
    candidate_states: torch.Tensor,
    *,
    repeats: int,
    rollout_temperature: float,
    unit_start: int,
    unit_end: int,
    mask_id: int,
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    seed: int,
    batch_size: int,
) -> list[list[float]]:
    k = int(candidate_states.shape[0])
    result = np.empty((k, int(repeats)), dtype=float)
    for rho in range(int(repeats)):
        for begin in range(0, k, max(1, int(batch_size))):
            end = min(k, begin + max(1, int(batch_size)))
            seed_all((int(seed) + rho * 1_000_003 + begin * 1009) & 0x7FFFFFFF)
            batch = candidate_states[begin:end].clone()
            logits = dream_logits(model, batch, logit_start=unit_start, logit_end=unit_end)
            block = batch[:, unit_start:unit_end]
            unresolved = block == int(mask_id)
            sampled = sample_from_logits(logits.float(), float(rollout_temperature))
            batch[:, unit_start:unit_end] = torch.where(unresolved, sampled, block)
            result[begin:end, rho] = score_fn(batch).numpy()
    return result.tolist()


def _rankdata(values: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    rx, ry = _rankdata(x), _rankdata(y)
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def checkpoint_metrics(
    rollout_scores: list[list[float]],
    pi0_scores: list[list[float]],
    *,
    prefixes: Iterable[int] = DEFAULT_PREFIXES,
    close_zero: float = 1e-4,
) -> dict[str, dict]:
    rollout = np.asarray(rollout_scores, dtype=float)
    pi0 = np.asarray(pi0_scores, dtype=float)
    q = pi0.mean(axis=1)
    dq = q - q[0]
    q_order = np.argsort(-q, kind="stable")
    q_margin = float(q[q_order[0]] - q[q_order[1]])
    observed_d = float(rollout.max() - rollout.min())
    k = rollout.shape[0]
    out = {}
    mu10 = rollout[:, :10].mean(axis=1)
    winner10 = int(np.argmax(mu10))
    top2_10 = set(np.argsort(-mu10, kind="stable")[:2].tolist())
    for r in prefixes:
        r = int(r)
        mu = rollout[:, :r].mean(axis=1)
        dm = mu - mu[0]
        err = np.abs(dq[1:] - dm[1:])
        order = np.argsort(-mu, kind="stable")
        margin = float(mu[order[0]] - mu[order[1]])
        out[f"metrics_R{r}"] = {
            "R": r,
            "alignment_error_mean": float(err.mean()),
            "alignment_error_median": float(np.median(err)),
            "alignment_error_p90": float(np.quantile(err, 0.9)),
            "spearman": spearman(dm[1:], dq[1:]),
            "sign_agreement": float(np.mean(np.sign(dm[1:]) == np.sign(dq[1:]))),
            "best_of_k_gain": float(mu[order[0]] - mu[0]),
            "top1_top2_margin": margin,
            "margin_close_to_zero": bool(margin <= float(close_zero)),
            "winner_index": int(order[0]),
            "winner_agreement_with_R10": bool(int(order[0]) == winner10),
            "top2_agreement_with_R10": bool(int(order[0]) in top2_10),
            "pi0_top1_top2_margin": q_margin,
            "pi0_winner_index": int(q_order[0]),
            "observed_D": observed_d,
            "observed_finite_R_bound_diagnostic": float(
                2 * observed_d * math.sqrt(math.log(k) / (2 * r))
            ),
        }
    return out


@dataclass
class FirstCrossingCheckpointHook:
    """Observe post-watermark Dream states using first-crossing semantics."""

    unit_start: int
    unit_end: int
    mask_id: int
    targets: tuple[float, ...]
    callback: Callable[[int, torch.Tensor, list[float], float, int], dict]

    def __post_init__(self):
        self.pending = list(sorted(float(v) for v in self.targets))
        self.records: list[dict] = []

    def __call__(self, step: int | None, x: torch.Tensor) -> None:
        if step is None or not self.pending:
            return
        actual, remaining = unit_progress(x, self.unit_start, self.unit_end, self.mask_id)
        crossed = [target for target in self.pending if actual >= target]
        if not crossed:
            return
        self.pending = [target for target in self.pending if target not in crossed]
        base = self.callback(int(step), x.detach().clone(), crossed, actual, remaining)
        for target in crossed:
            record = copy.deepcopy(base)
            record["target_progress"] = float(target)
            self.records.append(record)
