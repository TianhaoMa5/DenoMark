"""DLM-KGW baseline for diffusion language models.

This adapts the core logits-hook method from
https://github.com/eth-sri/diffusion-lm-watermark into this repository's LLaDA
generation interface. It is intentionally independent from DenMark.

Default settings mirror the paper repository's quick-use config:
  delta=2.0, convolution_kernel=[-1], topk=50, greenlist=bernoulli(gamma=0.25).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from denmark.baselines.common import _add_gumbel_noise, _get_num_transfer_tokens
from denmark.core.model import get_model_logits, safe_decode


class OnTheFlyGreenlist:
    """Stateless deterministic map from (context hash, token id) to green score."""

    def __init__(
        self,
        hash_size: int,
        vocab_size: int,
        mode: str = "bernoulli",
        distrib_params: Optional[dict] = None,
        seed: int = 42,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.idx_offset = max(vocab_size, hash_size)
        self.seed = seed
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.distrib_params = distrib_params or {"gamma": 0.25}

        self._C1 = torch.tensor(0x9E3779B97F4A7C15 - (1 << 64), dtype=torch.int64, device=device)
        self._C2 = torch.tensor(0xBF58476D1CE4E5B9 - (1 << 64), dtype=torch.int64, device=device)
        self._C3 = torch.tensor(0x94D049BB133111EB - (1 << 64), dtype=torch.int64, device=device)
        self._MASK53 = torch.tensor((1 << 53) - 1, dtype=torch.int64, device=device)

    def _splitmix64(self, z: torch.LongTensor) -> torch.LongTensor:
        z = z + self._C1
        z = (z ^ (z >> 30)) * self._C2
        z = (z ^ (z >> 27)) * self._C3
        return z ^ (z >> 31)

    def _uniform_from_keys(self, keys: torch.LongTensor) -> torch.Tensor:
        keys = keys.to(dtype=torch.int64, device=self.device)
        z = self._splitmix64(keys)
        return (z & self._MASK53).to(torch.float64) / float(1 << 53)

    def _normal_from_uniform(self, u: torch.Tensor) -> torch.Tensor:
        normal = torch.distributions.Normal(0, 1)
        return normal.icdf(u.clamp(1e-12, 1 - 1e-12)).to(self.dtype)

    def _get_final_distribution(self, u: torch.Tensor) -> torch.Tensor:
        if self.mode == "bernoulli":
            p = float(self.distrib_params["gamma"])
            return (u < p).to(self.dtype)
        if self.mode == "gaussian":
            return self._normal_from_uniform(u)
        if self.mode == "lognormal":
            return torch.exp(self._normal_from_uniform(u))
        if self.mode == "uniform":
            return u.to(self.dtype)
        raise ValueError(f"Unknown greenlist mode: {self.mode}")

    def lookup(self, h_inds: torch.LongTensor, v_inds: torch.LongTensor) -> torch.Tensor:
        h_inds = h_inds.to(dtype=torch.int64, device=self.device)
        v_inds = v_inds.to(dtype=torch.int64, device=self.device)
        h_b, v_b = torch.broadcast_tensors(h_inds, v_inds)
        keys = h_b.flatten() + v_b.flatten() * self.idx_offset + int(self.seed)
        values = self._get_final_distribution(self._uniform_from_keys(keys))
        return values.view(h_b.shape).to(self.dtype)


def _offset_contexts(input_ids: torch.LongTensor, offsets: torch.LongTensor) -> torch.LongTensor:
    """Gather context token ids for valid target positions."""
    _, seq_len = input_ids.shape
    start_at = max(-int(offsets.min()), 0)
    end_at = seq_len - max(int(offsets.max()), 0)
    target_pos = torch.arange(start_at, end_at, device=input_ids.device)
    gather_pos = target_pos[:, None] + offsets[None, :]
    return input_ids[:, gather_pos]


def _binom_sf(k: int, n: int, p: float) -> float:
    """P[X >= k] for X~Binom(n,p), computed without scipy."""
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0:
        return 0.0 if k > 0 else 1.0
    if p >= 1:
        return 1.0 if k <= n else 0.0

    prob = (1.0 - p) ** n
    tail = 0.0
    for i in range(0, n + 1):
        if i >= k:
            tail += prob
        if i < n:
            prob *= (n - i) / (i + 1) * p / (1.0 - p)
    return float(min(1.0, max(0.0, tail)))


def _mask_repetitions(contexts: torch.LongTensor) -> torch.BoolTensor:
    """Keep only the first occurrence of each context row along sequence length."""
    _, length, _ = contexts.shape
    match = (contexts.unsqueeze(2) == contexts.unsqueeze(1)).all(-1)
    previous = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=contexts.device),
        diagonal=-1,
    )
    return ~(match & previous.unsqueeze(0)).any(dim=-1)


class HashDistribution:
    """Paper-style diffusion watermark that biases model logits during denoising."""

    def __init__(
        self,
        tokenizer,
        delta: float = 2.0,
        enforce_kl: bool = False,
        convolution_kernel: Optional[list[int]] = None,
        greenlist_type: str = "bernoulli",
        greenlist_params: Optional[dict] = None,
        topk: int = 50,
        n_iter: int = 1,
        seeding_scheme: str = "sumhash",
        booster_only: bool = False,
        greenify_only: bool = False,
        device: str = "cuda",
    ):
        if booster_only and greenify_only:
            raise ValueError("booster_only and greenify_only cannot both be enabled.")
        if seeding_scheme not in {"sumhash", "minhash"}:
            raise ValueError("seeding_scheme must be 'sumhash' or 'minhash'.")

        self.device = device
        self.delta = float(delta)
        self.enforce_kl = enforce_kl
        self.convolution_kernel = torch.tensor(
            convolution_kernel or [-1], device=device, dtype=torch.long
        )
        self.context_size = len(self.convolution_kernel)
        self.greenlist_type = greenlist_type
        self.greenlist_params = greenlist_params or {"gamma": 0.25}
        self.topk = int(topk)
        self.n_iter = int(n_iter)
        self.seeding_scheme = seeding_scheme
        self.booster_only = booster_only
        self.greenify_only = greenify_only
        self.vocab_size = len(tokenizer.get_vocab())
        self.temperature = None
        self.mask_token_id = None

        self.greenlist = OnTheFlyGreenlist(
            hash_size=self.context_size * self.vocab_size,
            vocab_size=self.vocab_size,
            mode=self.greenlist_type,
            distrib_params=self.greenlist_params,
            seed=42,
            device=device,
            dtype=torch.float32,
        )

        if self.seeding_scheme == "minhash":
            with torch.random.fork_rng():
                torch.manual_seed(0)
                self.permutation = torch.randperm(self.vocab_size, device=device, dtype=torch.long)
                self.inv_permutation = torch.argsort(self.permutation)

    def get_key_params(self) -> dict:
        return {
            "watermark": "HashDistribution",
            "delta": self.delta,
            "enforce_kl": self.enforce_kl,
            "convolution_kernel": self.convolution_kernel.tolist(),
            "greenlist_type": self.greenlist_type,
            "greenlist_params": self.greenlist_params,
            "topk": self.topk,
            "n_iter": self.n_iter,
            "seeding_scheme": self.seeding_scheme,
        }

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)

    def set_mask_token(self, mask_token_id: int):
        self.mask_token_id = int(mask_token_id)

    def get_boundaries(self, seq_len: int) -> tuple[int, int]:
        start_at = max(-int(self.convolution_kernel.min()), 0)
        end_at = seq_len - max(int(self.convolution_kernel.max()), 0)
        return start_at, end_at

    def get_hashes_sequences(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        contexts = _offset_contexts(input_ids, self.convolution_kernel)
        if self.seeding_scheme == "minhash":
            return self.permutation[contexts].min(dim=-1).values
        return contexts.sum(dim=-1)

    def get_hashes_prob(self, probs: torch.FloatTensor) -> torch.FloatTensor:
        """Return hash probability for each valid target position.

        This adapter currently supports the paper config's single-offset kernel
        exactly. Multi-offset kernels need the original repo's FFT convolution.
        """
        if self.context_size != 1:
            raise NotImplementedError(
                "HashDistribution currently supports single-offset kernels "
                "such as [-1]."
            )
        _, seq_len, _ = probs.shape
        start_at, end_at = self.get_boundaries(seq_len)
        src = torch.arange(start_at, end_at, device=probs.device) + int(self.convolution_kernel[0])
        if self.seeding_scheme == "minhash":
            return probs[:, src, :][:, :, self.inv_permutation]
        return probs[:, src, :]

    def watermark_logits(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if self.temperature is None or self.mask_token_id is None:
            raise ValueError("temperature and mask_token_id must be set before watermarking.")

        mask_index = input_ids == self.mask_token_id
        inv_mask = ~mask_index
        masked_logits = logits.clone()
        masked_logits[inv_mask] = -1e9
        masked_logits[inv_mask, input_ids[inv_mask]] = 0

        with torch.enable_grad():
            logit_booster = self.compute_logit_booster(masked_logits, mask_index)
        watermarked_logits = logits + logit_booster.to(logits.dtype)
        return watermarked_logits, watermarked_logits

    def _compute_energy_topk(
        self,
        probs: torch.Tensor,
        hashes_prob: torch.Tensor,
        top_k_v: int,
        top_k_h: int,
    ) -> torch.Tensor:
        batch, length, vocab = probs.shape
        hash_count = hashes_prob.shape[-1]
        top_k_v = min(top_k_v, vocab)
        top_k_h = min(top_k_h, hash_count)

        if self.booster_only:
            hashes_prob = hashes_prob.detach()
        if self.greenify_only:
            probs = probs.detach()

        vals_v, idx_v = torch.topk(probs, k=top_k_v, dim=-1)
        vals_h, idx_h = torch.topk(hashes_prob, k=top_k_h, dim=-1)

        flat = batch * length
        h_inds = idx_h.reshape(flat, top_k_h).unsqueeze(-1)
        v_inds = idx_v.reshape(flat, top_k_v).unsqueeze(1)
        green = self.greenlist.lookup(h_inds, v_inds)
        green = green.reshape(batch, length, top_k_h, top_k_v)

        return (
            vals_h.unsqueeze(-1)
            * vals_v.unsqueeze(-2)
            * green.to(probs.dtype)
        ).sum(dim=(1, 2, 3))

    def kl_from_delta(
        self,
        d: torch.Tensor,
        logits: torch.Tensor,
        probs: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        boosted = torch.zeros_like(logits)
        boosted[:, :, : self.vocab_size] = d * alpha
        q = F.softmax((logits + boosted) / self.temperature, dim=-1)
        q_vocab = q[:, :, : self.vocab_size]
        return (q_vocab * torch.log(q_vocab.clamp_min(1e-30) / probs.clamp_min(1e-30))).sum(
            dim=-1, keepdim=True
        ).clamp_min(0.0)

    def find_delta(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        alpha: torch.Tensor,
        var_alpha: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        lo = torch.zeros_like(var_alpha)
        hi = torch.sqrt(2 * self.delta / (var_alpha + 1e-8)).mul(2).clamp_max(1e4)
        hi = torch.where(mask, hi, torch.zeros_like(hi))
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            too_high = (self.kl_from_delta(mid, logits, probs, alpha) > self.delta) & mask
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_high, lo, mid)
        return torch.where(mask, lo, torch.zeros_like(lo))

    def compute_logit_booster(
        self,
        logits: torch.FloatTensor,
        masked_inputs: Optional[torch.BoolTensor] = None,
    ) -> torch.FloatTensor:
        batch, seq_len, _ = logits.shape
        if masked_inputs is None:
            masked_inputs = torch.ones((batch, seq_len), dtype=torch.bool, device=logits.device)

        first_mask = masked_inputs.nonzero(as_tuple=False)
        min_mask_idx = int(first_mask[:, 1].min().item()) if first_mask.numel() else 0
        start_idx = max(min_mask_idx + int(self.convolution_kernel.min().item()), 0)

        slice_logits = logits[:, start_idx:, : self.vocab_size]
        probs = F.softmax(slice_logits / self.temperature, dim=-1).detach().requires_grad_(True)
        original_probs = probs.clone()
        active_mask = masked_inputs[:, start_idx:]

        alpha = torch.zeros_like(probs)
        delta = torch.zeros((*probs.shape[:2], 1), device=probs.device, dtype=probs.dtype)
        for _ in range(self.n_iter):
            start_at, end_at = self.get_boundaries(probs.shape[1])
            hashes_prob = self.get_hashes_prob(probs)
            sliced_probs = probs[:, start_at:end_at, :]
            energy = self._compute_energy_topk(
                sliced_probs, hashes_prob, top_k_v=self.topk, top_k_h=self.topk
            )
            energy.sum().backward()
            alpha = probs.grad

            if self.enforce_kl:
                mu_alpha = (original_probs * alpha).sum(dim=-1, keepdim=True)
                var_alpha = (original_probs * (alpha - mu_alpha).square()).sum(
                    dim=-1, keepdim=True
                )
                delta = self.find_delta(slice_logits, original_probs, alpha, var_alpha, active_mask)
            else:
                delta = torch.full(
                    (*probs.shape[:2], 1),
                    self.delta / (self.context_size + 1),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                delta = torch.where(active_mask.unsqueeze(-1), delta, torch.zeros_like(delta))

            probs = F.softmax((slice_logits + delta * alpha) / self.temperature, dim=-1)
            probs = probs.detach().requires_grad_(True)
            probs.grad = None

        logit_booster = torch.zeros_like(logits)
        logit_booster[:, start_idx:, : self.vocab_size] = alpha * delta
        return logit_booster

    def detect(self, input_ids: torch.LongTensor) -> dict:
        input_ids = input_ids.view(1, -1).to(self.device)
        if input_ids.shape[1] <= self.context_size:
            return {"n_trials": 0, "statistic": 0.0, "z_score": 0.0, "p_value": 1.0}

        start_at, end_at = self.get_boundaries(input_ids.shape[1])
        hashes = self.get_hashes_sequences(input_ids)
        token_scores = self.greenlist.lookup(hashes, input_ids[:, start_at:end_at])[0]

        repetition_offsets = self.convolution_kernel
        if not torch.any(repetition_offsets == 0):
            repetition_offsets = torch.cat([
                repetition_offsets,
                torch.tensor([0], device=self.device, dtype=torch.long),
            ])
        repetition_mask = _mask_repetitions(_offset_contexts(input_ids, repetition_offsets))[0]
        scores = token_scores[repetition_mask].detach().cpu().float()
        n_trials = int(scores.numel())
        mean = float(scores.mean().item()) if n_trials else 0.0

        if self.greenlist_type == "bernoulli":
            gamma = float(self.greenlist_params["gamma"])
            successes = int(scores.sum().item())
            denom = math.sqrt(max(n_trials * gamma * (1.0 - gamma), 1e-12))
            z_score = (successes - n_trials * gamma) / denom
            p_value = _binom_sf(successes, n_trials, gamma)
            statistic = successes / max(1, n_trials)
        elif self.greenlist_type == "gaussian":
            z_score = mean / math.sqrt(1.0 / max(1, n_trials))
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))
            statistic = mean
        else:
            statistic = mean
            z_score = mean / math.sqrt(1.0 / max(1, n_trials))
            p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))

        return {
            "n_trials": n_trials,
            "statistic": float(statistic),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "token_color": token_scores.detach().cpu().float().tolist(),
            "mask": repetition_mask.detach().cpu().tolist(),
        }


@torch.no_grad()
def generate_hash_distribution(
    prompt, model, tokenizer, mask_id, watermark, *, generator_family="llada", **kwargs
):
    """Dispatch to blockwise LLaDA or Dream origin, keeping the same watermark."""
    if generator_family != "dream":
        return llada_generate_hash_distribution(
            prompt, model, tokenizer, mask_id, watermark,
            generator_family=generator_family, **kwargs,
        )
    from denmark.baselines.clean.model import dream_origin_diffusion_generate

    if kwargs.get("cfg_scale", 0.0) != 0.0:
        raise ValueError("Dream origin does not support cfg_scale")
    if kwargs.get("remasking", "random") != "random":
        raise ValueError("Dream origin uses random position transfers; use --remasking random")
    temperature = kwargs.get("temperature", 0.5)
    watermark.set_temperature(temperature)
    watermark.set_mask_token(mask_id)

    def watermark_hook(step, current, logits):
        sampling_logits, _ = watermark.watermark_logits(current, logits)
        return sampling_logits

    gen_length = kwargs.get("gen_length", 300)
    output = dream_origin_diffusion_generate(
        model=model, input_ids=prompt, attention_mask=torch.ones_like(prompt),
        gen_length=gen_length, steps=kwargs.get("steps") or gen_length,
        temperature=temperature, alg="origin", mask_token_id=mask_id,
        generation_logits_hook_func=watermark_hook,
    )
    tokens = output.sequences[0, prompt.shape[1]:].tolist()
    stops = {value for value in (
        tokenizer.eos_token_id, tokenizer.pad_token_id,
    ) if value is not None}
    stop = next((i for i, value in enumerate(tokens) if value in stops), len(tokens))
    tokens = tokens[:stop]
    text = safe_decode(tokenizer, tokens, skip_special_tokens=True).strip()
    detection = watermark.detect(torch.tensor(tokens, dtype=torch.long, device=prompt.device))
    return text, tokens, detection


@torch.no_grad()
def llada_generate_hash_distribution(
    prompt: torch.Tensor,
    llada,
    llada_tok,
    mask_id: int,
    watermark: HashDistribution,
    gen_length: int = 300,
    block_size: int = 25,
    steps: Optional[int] = None,
    temperature: float = 0.5,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    generator_family: str = "llada",
):
    """LLaDA generation with the HashDistribution logits hook enabled."""
    assert gen_length % block_size == 0, "gen_length must be divisible by block_size"
    num_blocks = gen_length // block_size
    if steps is None:
        steps = gen_length
    assert steps % num_blocks == 0, f"steps={steps} must be divisible by num_blocks={num_blocks}"
    steps_per_block = steps // num_blocks

    watermark.set_temperature(temperature)
    watermark.set_mask_token(mask_id)

    prompt_len = prompt.shape[1]
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=llada.device)
    x[:, :prompt_len] = prompt.clone()
    prompt_index = x != mask_id

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = prompt_len + (block_idx + 1) * block_size
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step_idx in range(steps_per_block):
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

            sampling_logits, remask_logits = watermark.watermark_logits(x, logits)
            logits_with_noise = _add_gumbel_noise(sampling_logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                probs = F.softmax(remask_logits.to(torch.float64), dim=-1)
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

            x0_p[:, block_end:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -float("inf")))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row_idx in range(confidence.shape[0]):
                k = int(num_transfer_tokens[row_idx, step_idx].item())
                if k > 0:
                    _, selected = torch.topk(confidence[row_idx], k=k)
                    transfer_index[row_idx, selected] = True
            x[transfer_index] = x0[transfer_index]

    tokens = x[0, prompt_len:].tolist()
    text = safe_decode(llada_tok, tokens, skip_special_tokens=True).strip()
    detection = watermark.detect(torch.tensor(tokens, dtype=torch.long, device=llada.device))
    return text, tokens, detection
