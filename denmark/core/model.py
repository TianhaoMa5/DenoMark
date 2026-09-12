"""Shared model, tokenizer, and encoding helpers for DenMark.

- `_gumbel`              : Gumbel-max noise for temperature sampling.
- `encode_texts`         : Batched text → 768-d normalized embeddings via T_eta encoder.
- `build_directions`     : Generate random θ directions + signs (the "watermark key").
- `complete_block_one_shot`: Roll out (one-shot LLaDA fill) MASK positions in a block.
"""
from __future__ import annotations
import os
from typing import Optional

import torch
import torch.nn.functional as F

LLADA_STYLE_GENERATOR_FAMILIES = ("llada",)
GENERATOR_FAMILIES = ("llada", "llada2", "dream")
DEFAULT_MASK_TOKEN_IDS = {
    "llada": 126336,
    "llada2": 156895,
    "dream": 151666,
}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _gumbel(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Official LLaDA-style float64 Gumbel-max sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits - torch.log(-torch.log(noise + 1e-20) + 1e-20) * temperature


def load_generator_model(
    model_name_or_path: str,
    generator_family: str = "llada",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda",
    trust_remote_code: bool = True,
):
    """Load a generator model with the right HF class for its family."""
    if generator_family not in GENERATOR_FAMILIES:
        raise ValueError(f"unknown generator_family: {generator_family}")

    if generator_family == "llada2":
        _ensure_transformers_kwargs()
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()

    from transformers import AutoModel

    return AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    ).eval()


def _ensure_transformers_kwargs() -> None:
    """Provide the newer TransformersKwargs typing symbol for older 4.x builds."""
    import transformers.utils as transformers_utils

    if hasattr(transformers_utils, "TransformersKwargs"):
        return

    from typing import TypedDict

    class TransformersKwargs(TypedDict, total=False):
        pass

    transformers_utils.TransformersKwargs = TransformersKwargs


def resolve_mask_id(tokenizer, generator_family: str = "llada", mask_id: Optional[int] = None) -> int:
    """Resolve MASK token id, preferring an explicit CLI value."""
    if mask_id is not None:
        return int(mask_id)
    tok_mask = getattr(tokenizer, "mask_token_id", None)
    if tok_mask is not None:
        return int(tok_mask)
    if generator_family not in DEFAULT_MASK_TOKEN_IDS:
        raise ValueError(f"unknown generator_family: {generator_family}")
    return DEFAULT_MASK_TOKEN_IDS[generator_family]


def filter_decodable_token_ids(tokenizer, token_ids) -> list[int]:
    """Drop token ids that a tokenizer cannot map back to string tokens."""
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    ids: list[int] = []
    for token_id in token_ids:
        if token_id is None:
            continue
        try:
            ids.append(int(token_id))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []

    try:
        tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=False)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(ids)
    if tokens is None:
        tokens = [None] * len(ids)
    elif isinstance(tokens, str):
        tokens = [tokens]
    return [token_id for token_id, token in zip(ids, tokens) if token is not None]


def safe_decode(tokenizer, token_ids, skip_special_tokens: bool = True, **kwargs) -> str:
    return tokenizer.decode(
        filter_decodable_token_ids(tokenizer, token_ids),
        skip_special_tokens=skip_special_tokens,
        **kwargs,
    )


def safe_batch_decode(tokenizer, batch_token_ids, skip_special_tokens: bool = True, **kwargs) -> list[str]:
    return [
        safe_decode(tokenizer, token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        for token_ids in batch_token_ids
    ]


def _llada2_logits(
    model,
    input_ids: torch.Tensor,
    logit_start: Optional[int],
    logit_end: Optional[int],
    attention_boundary: Optional[int] = None,
) -> torch.Tensor:
    """Run LLaDA2 with the 4D attention mask required by its remote code."""
    forward_ids = input_ids[:, :logit_end] if logit_end is not None else input_ids
    batch_size, seq_len = forward_ids.shape
    dtype = torch.bfloat16 if forward_ids.is_cuda else torch.float32
    base_mask = torch.zeros((seq_len, seq_len), device=forward_ids.device, dtype=dtype)

    boundary = attention_boundary if attention_boundary is not None else logit_start
    if boundary is not None:
        boundary = int(boundary)
        if not 0 <= boundary <= seq_len:
            raise ValueError(f"LLaDA2 attention boundary {boundary} is outside [0, {seq_len}]")
        if boundary > 0:
            base_mask[:boundary, boundary:] = float("-inf")

    attention_mask = base_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    position_ids = torch.arange(seq_len, device=forward_ids.device).unsqueeze(0)
    logits = model(
        forward_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits

    if logit_start is not None or logit_end is not None:
        return logits[:, logit_start:logit_end]
    return logits


def get_model_logits(
    model,
    input_ids: torch.Tensor,
    generator_family: str = "llada",
    logit_start: Optional[int] = None,
    logit_end: Optional[int] = None,
    attention_boundary: Optional[int] = None,
):
    """Return logits aligned to the requested generator family.

    With ``logit_start`` and ``logit_end``, return the aligned sequence slice.
    """
    if generator_family == "llada2":
        return _llada2_logits(
            model,
            input_ids,
            logit_start,
            logit_end,
            attention_boundary=attention_boundary,
        )
    if generator_family in LLADA_STYLE_GENERATOR_FAMILIES and logit_start is not None and logit_end is not None:
        start = int(logit_start)
        end = int(logit_end)
        # LLaDA-style diffusion logits depend on the full masked canvas. Do not trim
        # the sequence to `end`; old successful WP runs forwarded the full
        # prompt + generation canvas and only sliced logits afterwards.
        logits = model(input_ids).logits
        return logits[:, start:end]

    logits = model(input_ids).logits
    if generator_family == "dream":
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    elif generator_family not in LLADA_STYLE_GENERATOR_FAMILIES:
        raise ValueError(f"unknown generator_family: {generator_family}")

    if logit_start is not None or logit_end is not None:
        return logits[:, logit_start:logit_end]
    return logits


# ---------------------------------------------------------------------------
# T_eta encoder helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(
    texts, enc, enc_tok, device, batch_sz: int = 32, to_cpu: bool = True
) -> torch.Tensor:
    """Mean-pool over attention mask, then L2 normalize.

    Detection callers keep the historical CPU return. Generation can pass
    to_cpu=False to avoid a CPU round trip before projection/selection.
    """
    batch_cap = os.getenv("BLOCK_WM_ENCODE_BATCH_SIZE")
    if batch_cap:
        batch_sz = max(1, min(batch_sz, int(batch_cap)))
    all_embs = []
    for i in range(0, len(texts), batch_sz):
        batch = texts[i:i + batch_sz]
        inp = enc_tok(batch, padding=True, truncation=True,
                      max_length=512, return_tensors="pt").to(device)
        out = enc(**inp)
        attn = inp["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
        embs = F.normalize(pooled, dim=-1)
        all_embs.append(embs.cpu() if to_cpu else embs)
    return torch.cat(all_embs, dim=0)


# ---------------------------------------------------------------------------
# Watermark direction key
# ---------------------------------------------------------------------------

def build_directions(num_blocks: int, num_bits: int,
                     direction_seed: int = 42, message_seed: int = 0,
                     emb_dim: int = 768, orthogonal: bool = False):
    """
    θ is num_blocks × num_bits random unit vectors in R^{emb_dim}.
    sign is num_blocks × num_bits ∈ {+1, -1}.
    Both depend only on the seeds rather than learned parameters. Deployments
    must treat the seed material used to reconstruct them as the secret key.
    """
    rng_d = torch.Generator(); rng_d.manual_seed(direction_seed)
    n_dirs = num_blocks * num_bits
    if orthogonal:
        if n_dirs > emb_dim:
            raise ValueError(f"Cannot build {n_dirs} orthogonal directions in R^{emb_dim}")
        raw = torch.randn(emb_dim, n_dirs, generator=rng_d)
        dirs = torch.linalg.qr(raw, mode="reduced").Q.T.contiguous()
    else:
        dirs = F.normalize(torch.randn(n_dirs, emb_dim, generator=rng_d), dim=-1)
    dirs = dirs.view(num_blocks, num_bits, emb_dim)
    rng_m = torch.Generator(); rng_m.manual_seed(message_seed)
    signs = (
        torch.randint(0, 2, (num_blocks * num_bits,), generator=rng_m).float() * 2 - 1
    ).view(num_blocks, num_bits)
    return dirs, signs


# ---------------------------------------------------------------------------
# LLaDA rollout (one-shot fill of MASKs in a block)
# ---------------------------------------------------------------------------

@torch.no_grad()
def complete_block_one_shot(
    cand_batch: torch.Tensor, model, mask_id: int, temperature: float,
    L_p: int, target_block: int, block_size: int, gen_length: int,
    batch_sz: int = 8, n_iter: int = 1,
    shared_noise_group_size: int = 0, shared_noise_seed: int = 0,
    generator_family: str = "llada",
) -> torch.Tensor:
    """
    For each candidate, fill remaining MASK positions IN target_block ONLY.
    Other MASKs (later blocks) are left as MASK.

    n_iter=1 (default): one-shot fill (all MASKs sampled in one forward).
    n_iter>1: iterative denoising. At each step, fill ceil(remaining/steps_left) MASK
              positions by highest confidence (low_confidence remasking, LLaDA-style).
              Last step fills all remaining. ~n_iter× cost.

    Returns [K, L] on CPU.
    """
    b_s = L_p + target_block * block_size
    b_e = min(b_s + block_size, L_p + gen_length)

    K = cand_batch.shape[0]
    out_chunks = []
    for s in range(0, K, batch_sz):
        chunk = cand_batch[s: s + batch_sz].clone()
        for step in range(n_iter):
            block_logits = get_model_logits(
                model,
                chunk,
                generator_family,
                logit_start=b_s,
                logit_end=b_e,
            )
            if temperature == 0:
                sampled = block_logits.argmax(dim=-1)
            else:
                sampling_logits = block_logits.to(torch.float64)
                if shared_noise_group_size and shared_noise_group_size > 0:
                    noise = torch.empty_like(sampling_logits)
                    rollout_ids = (
                        torch.arange(s, s + chunk.shape[0], device=sampling_logits.device)
                        % int(shared_noise_group_size)
                    )
                    for rollout_i in rollout_ids.unique(sorted=True).tolist():
                        gen = torch.Generator(device=sampling_logits.device)
                        seed = (
                            int(shared_noise_seed) * 1_000_003
                            + int(step) * 10_007
                            + int(rollout_i)
                        ) & 0x7FFFFFFF
                        gen.manual_seed(seed)
                        shared_noise = torch.rand(
                            sampling_logits.shape[1:],
                            generator=gen,
                            device=sampling_logits.device,
                            dtype=torch.float64,
                        )
                        noise[rollout_ids == int(rollout_i)] = shared_noise
                else:
                    noise = torch.rand_like(sampling_logits, dtype=torch.float64)
                sampled = (sampling_logits
                           - torch.log(-torch.log(noise + 1e-20) + 1e-20) * temperature
                           ).argmax(dim=-1)
            chunk_block = chunk[:, b_s:b_e]
            is_mask = (chunk_block == mask_id)
            if step == n_iter - 1:
                new_block = torch.where(is_mask, sampled, chunk_block)
            else:
                conf = block_logits.softmax(dim=-1).max(dim=-1).values
                conf_masked = conf.masked_fill(~is_mask, float('-inf'))
                n_mask = is_mask.sum(dim=-1)
                steps_left = n_iter - step
                n_commit = (n_mask.float() / steps_left).ceil().long()
                new_block = chunk_block.clone()
                for b in range(chunk.shape[0]):
                    nk = min(int(n_commit[b].item()), int(is_mask[b].sum().item()))
                    if nk > 0:
                        top_idx = conf_masked[b].topk(nk).indices
                        new_block[b, top_idx] = sampled[b, top_idx]
            chunk[:, b_s:b_e] = new_block
        out_chunks.append(chunk.cpu())
    return torch.cat(out_chunks, dim=0)


# Dream decoding adapter.
#!/usr/bin/env python3

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


from denmark.core.selectors import _top_logprob_indices


NEWLINE_CHAR = "NEWLINE_CHAR"
FINANCE_QA_PROMPT_TEMPLATE = (
    "You are a helpful assistant, please answer the following question with financial knowledge within 300 words:\n"
    "{context}\n"
    "{input}"
)


def clean_text(text: str) -> str:
    if not text:
        return text
    return " ".join(text.replace(NEWLINE_CHAR, " ").split())


def rep_ngram(text: str, n: int = 4) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    repeats = sum(v - 1 for v in counts.values() if v > 1)
    return repeats / max(1, len(grams))


def build_chat_prompt(input_text: str, context: str, tokenizer) -> tuple[str, list[int]]:
    user_msg = (context + "\n\n" + input_text).strip() if context else input_text
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg.strip()}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
        )
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return tokenizer.decode(ids, skip_special_tokens=False), list(ids)
    except Exception:
        ids = tokenizer(user_msg, add_special_tokens=True)["input_ids"]
        return user_msg, list(ids)


def build_prompt(input_text: str, context: str, tokenizer, prompt_mode: str) -> tuple[str, list[int]]:
    if prompt_mode == "base":
        user_msg = (context + "\n\n" + input_text).strip() if context else input_text
        ids = tokenizer(user_msg, add_special_tokens=True)["input_ids"]
        return user_msg, list(ids)
    return build_chat_prompt(input_text, context, tokenizer)


def build_model_input_from_row(row: dict, dataset: str | None = None, prompt_variant: str = "default") -> tuple[str, str, str]:
    if row.get("raw_prompt"):
        return str(row.get("raw_prompt") or "").strip(), str(row.get("input") or ""), str(row.get("context") or "")
    ds = (dataset or row.get("dataset") or "").lower()
    variant = (prompt_variant or "default").lower()
    if ds in {"finance_qa", "finance"}:
        context = str(row.get("context") or "")
        input_text = str(row.get("input") or "")
        return FINANCE_QA_PROMPT_TEMPLATE.format(context=context, input=input_text).strip(), input_text, context
    if ds in {"wp", "writingprompts", "writing_prompts"}:
        prompt = str(row.get("prompt") or row.get("input") or "")
        if variant == "long_output":
            model_input = (
                "Write a detailed, coherent story of about 300 words based on the following prompt. "
                "Do not stop after a short opening.\n"
                f"{prompt}"
            )
            return model_input.strip(), prompt, ""
        return prompt.strip(), prompt, ""
    if ds in {"c4", "realnews", "realnewslike", "realnews_like"}:
        text = str(row.get("text") or row.get("input") or "")
        if variant == "long_output":
            model_input = (
                "Continue the passage below with a coherent continuation of about 300 words. "
                "Do not stop after one sentence.\n"
                f"{text}"
            )
            return model_input.strip(), text, ""
        return text.strip(), text, ""
    input_text = str(row.get("input") or row.get("prompt") or row.get("text") or "")
    context = str(row.get("context") or "")
    user_msg = (context + "\n\n" + input_text).strip() if context else input_text.strip()
    return user_msg, input_text, context


def truncate_tail(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[-max_chars:].lstrip(), True
    return text, False


def safe_decode_dream(tokenizer, ids: list[int], special_ids: set[int]) -> str:
    toks = [int(t) for t in ids if int(t) not in special_ids]
    if not toks:
        return ""
    raw = tokenizer.convert_ids_to_tokens(toks, skip_special_tokens=True)
    raw = [tok for tok in raw if tok is not None]
    if not raw:
        return ""
    try:
        return tokenizer.convert_tokens_to_string(raw).strip()
    except Exception:
        return tokenizer.decode(toks, skip_special_tokens=True).strip()


def dream_logits(
    model,
    x: torch.Tensor,
    *,
    logit_start: int | None = None,
    logit_end: int | None = None,
) -> torch.Tensor:
    """Return logits aligned to token positions, optionally for a small window.

    Dream's remote model can keep only the last logits. For long RealNews
    prompts, materializing full prompt+generation logits inside every semantic
    rollout can OOM even on 95GB GPUs. Request the minimal suffix needed for the
    target block, then slice the aligned window.
    """
    if logit_start is not None and logit_end is not None:
        start = max(0, int(logit_start))
        end = max(start, int(logit_end))
        if end <= start:
            return x.new_empty((x.shape[0], 0, 0), dtype=torch.float32)
        keep_start = max(0, start - 1)
        keep = int(x.shape[1] - keep_start)
        try:
            logits = model(x, "full", None, num_logits_to_keep=keep).logits
            if start == 0:
                aligned = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                # First kept hidden state is at start-1, so its logits predict
                # token position start after the usual Dream one-token shift.
                aligned = logits
            return aligned[:, : end - start]
        except TypeError:
            pass
    logits = model(x, "full", None).logits
    aligned = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    if logit_start is not None and logit_end is not None:
        return aligned[:, int(logit_start) : int(logit_end)]
    return aligned


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    *,
    shared_noise_group_size: int = 0,
    shared_noise_seed: int = 0,
) -> torch.Tensor:
    if temperature == 0:
        return logits.float().argmax(dim=-1)
    logits = logits.float()
    if shared_noise_group_size and shared_noise_group_size > 0:
        noise = torch.empty_like(logits)
        rollout_ids = torch.arange(logits.shape[0], device=logits.device) % int(shared_noise_group_size)
        for rollout_i in rollout_ids.unique(sorted=True).tolist():
            gen = torch.Generator(device=logits.device)
            seed = (int(shared_noise_seed) * 1_000_003 + int(rollout_i)) & 0x7FFFFFFF
            gen.manual_seed(seed)
            shared_noise = torch.rand(
                logits.shape[1:],
                generator=gen,
                device=logits.device,
                dtype=logits.dtype,
            )
            noise[rollout_ids == int(rollout_i)] = shared_noise
        return (logits - torch.log(-torch.log(noise + 1e-20) + 1e-20) * temperature).argmax(dim=-1)
    noise = torch.rand_like(logits)
    return (
        logits
        - torch.log(-torch.log(noise + 1e-20) + 1e-20) * temperature
    ).argmax(dim=-1)


class DreamSemanticUnitDecodeController:
    """Choose positions for watermark candidate sampling.

    DREAM's native sampler remains responsible for logits, token sampling, and
    origin transfer/remasking. This controller only records the positions that
    the watermark hook may edit. ``random_semantic_unit`` keeps the legacy
    one-unit-at-a-time behavior. ``random_global_grouped`` chooses one global
    random position set, which the hook later splits by semantic unit.
    """

    def __init__(
        self,
        *,
        prompt_len: int,
        gen_length: int,
        block_size: int,
        cand_block_size: int,
        mask_id: int,
        steps: int,
        eps: float,
        mode: str,
    ) -> None:
        self.prompt_len = int(prompt_len)
        self.gen_length = int(gen_length)
        self.block_size = int(block_size)
        self.cand_block_size = max(1, int(cand_block_size))
        self.mask_id = int(mask_id)
        self.steps = int(steps)
        self.eps = float(eps)
        self.mode = str(mode)
        self.pre_x: torch.Tensor | None = None
        self.allowed_mask: torch.Tensor | None = None
        self.diag: list[dict] = []

    def _num_transfer_tokens(self, x: torch.Tensor, step: int) -> int:
        mask_index = x == self.mask_id
        num_mask_token = int(mask_index.sum().item() // max(1, x.shape[0]))
        if step >= self.steps - 1:
            return num_mask_token
        t = 1.0 + (self.eps - 1.0) * (float(step) / float(self.steps))
        s = 1.0 + (self.eps - 1.0) * (float(step + 1) / float(self.steps))
        return int(num_mask_token * (1.0 - s / max(t, 1e-12)))

    def logits_hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.mode == "dream_official" or step is None:
            return logits
        if self.mode not in {"random_semantic_unit", "random_global_grouped"}:
            raise ValueError(f"unknown decode_position_mode: {self.mode}")

        self.pre_x = x.detach().clone()
        selected_unit_mask = torch.zeros_like(x, dtype=torch.bool)
        gen_s = self.prompt_len
        gen_e = self.prompt_len + self.gen_length
        n_transfer = self._num_transfer_tokens(x, int(step))
        candidate_positions_per_candidate = max(1, self.cand_block_size)

        for row in range(x.shape[0]):
            gen_mask = x[row, gen_s:gen_e] == self.mask_id
            if not bool(gen_mask.any().item()):
                continue
            if self.mode == "random_global_grouped":
                unresolved = gen_mask.nonzero(as_tuple=False).flatten() + gen_s
                n_pick = min(candidate_positions_per_candidate, int(unresolved.numel()))
                picked = unresolved[
                    torch.randperm(unresolved.numel(), device=x.device)[:n_pick]
                ]
                selected_unit_mask[row, picked] = True
                by_block: dict[int, list[int]] = {}
                for position in picked.detach().cpu().tolist():
                    block_id = (int(position) - gen_s) // self.block_size
                    by_block.setdefault(int(block_id), []).append(int(position))
                self.diag.append({
                    "step": int(step),
                    "row": int(row),
                    "mode": self.mode,
                    "selected_positions": [int(p) for p in picked.detach().cpu().tolist()],
                    "selected_positions_by_block": {
                        str(block_id): positions
                        for block_id, positions in sorted(by_block.items())
                    },
                    "selected_group_count": int(len(by_block)),
                    "selected_position_count": int(n_pick),
                    "remaining_before": int(unresolved.numel()),
                    "native_num_transfer": int(n_transfer),
                    "global_position_budget": int(candidate_positions_per_candidate),
                })
                continue

            active = []
            for block_id in range(math.ceil(self.gen_length / self.block_size)):
                b_s = gen_s + block_id * self.block_size
                b_e = min(b_s + self.block_size, gen_e)
                count = int((x[row, b_s:b_e] == self.mask_id).sum().item())
                if count > 0:
                    active.append((block_id, count))
            enough = [item for item in active if item[1] >= candidate_positions_per_candidate]
            pool = enough if enough else active
            pick_i = int(torch.randint(len(pool), (1,), device=x.device).item())
            selected_block = int(pool[pick_i][0])
            b_s = gen_s + selected_block * self.block_size
            b_e = min(b_s + self.block_size, gen_e)
            unit_positions = (x[row, b_s:b_e] == self.mask_id).nonzero(as_tuple=False).flatten() + b_s
            selected_unit_mask[row, unit_positions] = True
            active_blocks = [int(b) for b, _ in active]
            self.diag.append({
                "step": int(step),
                "row": int(row),
                "mode": self.mode,
                "selected_block": int(selected_block),
                "selected_unit_mask_positions": [int(p) for p in unit_positions.detach().cpu().tolist()],
                "active_blocks": active_blocks,
                "native_num_transfer": int(n_transfer),
                "candidate_positions_per_candidate": int(candidate_positions_per_candidate),
                "selected_unit_mask_count": int(selected_unit_mask[row].sum().item()),
            })

        self.allowed_mask = selected_unit_mask
        return logits

    def filter_transfers(self, step: int, x: torch.Tensor) -> torch.Tensor:
        # Kept for compatibility with older callers. Do not remask native DREAM
        # transfers; eth-sri origin transfer behavior should pass through.
        return x


class NativeSemanticWatermarkHook:
    """Rerank only tokens that DREAM official decoding just transferred.

    DREAM decides the transfer positions and schedule. The hook observes newly
    filled positions after each denoising step, samples K alternatives only at
    those exact positions, rolls out the rest of each semantic block for scoring,
    and writes back the best candidate tokens at the same positions.
    """

    def __init__(
        self,
        model,
        tokenizer,
        encoder,
        encoder_tokenizer,
        directions: torch.Tensor,
        signs: torch.Tensor,
        *,
        prompt_len: int,
        gen_length: int,
        block_size: int,
        cand_block_size: int,
        num_candidates: int,
        channels_per_step: int,
        candidate_temperature: float,
        rollout_temperature: float,
        rollouts_per_cand: int,
        rollout_schedule: str,
        logprob_weight: float,
        candidate_position_mode: str,
        mask_id: int,
        device: str,
        argmax_logprob_top_frac: float = 1.0,
        sample_id=None,
        resample_base_candidate: bool = False,
        shared_rollout_seeds: bool = False,
        dedup_candidates: bool = False,
        rollout_logits_mode: str = "full",
        rollout_batch_size: int = 0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.encoder_tokenizer = encoder_tokenizer
        self.directions = directions.cpu()
        self.signs = signs.cpu()
        self.prompt_len = int(prompt_len)
        self.gen_length = int(gen_length)
        self.block_size = int(block_size)
        self.cand_block_size = max(1, int(cand_block_size))
        self.num_candidates = int(num_candidates)
        self.channels_per_step = int(channels_per_step)
        self.candidate_temperature = float(candidate_temperature)
        self.rollout_temperature = float(rollout_temperature)
        self.rollouts_per_cand = max(1, int(rollouts_per_cand))
        self.rollout_schedule = rollout_schedule or "none"
        self.logprob_weight = float(logprob_weight)
        self.candidate_position_mode = candidate_position_mode
        self.mask_id = int(mask_id)
        self.device = device
        self.argmax_logprob_top_frac = float(argmax_logprob_top_frac)
        self.sample_id = sample_id
        self.resample_base_candidate = bool(resample_base_candidate)
        self.shared_rollout_seeds = bool(shared_rollout_seeds)
        self.dedup_candidates = bool(dedup_candidates)
        self.rollout_logits_mode = str(rollout_logits_mode)
        self.rollout_batch_size = max(0, int(rollout_batch_size))
        if self.argmax_logprob_top_frac <= 0.0 or self.argmax_logprob_top_frac > 1.0:
            raise ValueError("argmax_logprob_top_frac must be in (0, 1]")
        if self.rollout_schedule not in {"none", "constant", "linear_decay"}:
            raise ValueError(f"unknown rollout_schedule: {self.rollout_schedule}")
        if self.rollout_logits_mode not in {"full", "window"}:
            raise ValueError(f"unknown rollout_logits_mode: {self.rollout_logits_mode}")
        self.prev_x: torch.Tensor | None = None
        self.forced_candidate_mask: torch.Tensor | None = None
        self.diag: list[dict] = []

        specials = {
            self.mask_id,
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
        }
        self.special_ids = {int(x) for x in specials if x is not None}

    def _effective_rollouts(self, block_len: int, remaining_before_commit: int) -> int:
        target_avg = max(1, self.rollouts_per_cand)
        if self.rollout_schedule in {"none", "constant"}:
            return target_avg
        if block_len <= 1:
            raw_rollouts = float(target_avg)
        else:
            remaining = max(1, int(remaining_before_commit))
            raw_rollouts = 1.0 + 2.0 * (target_avg - 1) * (remaining - 1) / (block_len - 1)
        return max(1, int(math.floor(raw_rollouts + 0.5)))

    def _score_candidates(
        self,
        candidate_tokens: torch.Tensor,
        x: torch.Tensor,
        positions,
        block_id: int,
        logits: torch.Tensor,
        step: int,
        remaining_before_commit: int,
    ) -> tuple[int, dict]:
        k = candidate_tokens.shape[0]
        b_s = self.prompt_len + block_id * self.block_size
        b_e = min(b_s + self.block_size, self.prompt_len + self.gen_length)
        block_len = max(1, b_e - b_s)
        effective_rollouts = self._effective_rollouts(block_len, remaining_before_commit)
        logical_rollouts = effective_rollouts

        cands = x.expand(k, -1).clone()
        if isinstance(positions, torch.Tensor):
            cands[:, positions] = candidate_tokens
            position_rows = positions.unsqueeze(0).expand(k, -1)
        else:
            position_rows = torch.stack(positions, dim=0)
            for row_idx, pos in enumerate(positions):
                cands[row_idx, pos] = candidate_tokens[row_idx, : pos.numel()]

        if self.dedup_candidates:
            seen = {}
            unique_indices = []
            inverse = []
            for i in range(k):
                key = tuple(cands[i, b_s:b_e].detach().cpu().tolist())
                if key not in seen:
                    seen[key] = len(unique_indices)
                    unique_indices.append(i)
                inverse.append(seen[key])
            unique_idx = torch.tensor(unique_indices, dtype=torch.long, device=cands.device)
            inverse_idx = torch.tensor(inverse, dtype=torch.long, device=cands.device)
            rollout_cands_base = cands.index_select(0, unique_idx)
            duplicate_count = k - len(unique_indices)
        else:
            inverse_idx = torch.arange(k, dtype=torch.long, device=cands.device)
            rollout_cands_base = cands
            duplicate_count = 0
        rollout_candidate_count = int(rollout_cands_base.shape[0])

        block = rollout_cands_base[:, b_s:b_e]
        unresolved = block == self.mask_id
        if unresolved.any():
            rollout_cands = rollout_cands_base.repeat_interleave(effective_rollouts, dim=0)
            rollout_block = rollout_cands[:, b_s:b_e]
            rollout_unresolved = rollout_block == self.mask_id
            rollout_noise_seed = (int(step) * 1_000_003 + int(block_id) * 1009) & 0x7FFFFFFF
            rollout_batch_size = self.rollout_batch_size or int(rollout_cands.shape[0])
            if effective_rollouts > 1:
                rollout_batch_size = max(
                    effective_rollouts,
                    (rollout_batch_size // effective_rollouts) * effective_rollouts,
                )
            sampled_parts = []
            for rollout_start in range(0, int(rollout_cands.shape[0]), rollout_batch_size):
                rollout_end = min(rollout_start + rollout_batch_size, int(rollout_cands.shape[0]))
                rollout_cands_part = rollout_cands[rollout_start:rollout_end]
                if self.rollout_logits_mode == "window":
                    rollout_logits = dream_logits(
                        self.model,
                        rollout_cands_part,
                        logit_start=b_s,
                        logit_end=b_e,
                    )
                else:
                    rollout_logits = dream_logits(self.model, rollout_cands_part)[:, b_s:b_e]
                sampled_parts.append(
                    sample_from_logits(
                        rollout_logits,
                        self.rollout_temperature,
                        shared_noise_group_size=effective_rollouts if self.shared_rollout_seeds else 0,
                        shared_noise_seed=rollout_noise_seed,
                    )
                )
                del rollout_logits
            sampled = torch.cat(sampled_parts, dim=0)
            block = torch.where(rollout_unresolved, sampled, rollout_block)
        else:
            effective_rollouts = 1

        texts = [
            safe_decode_dream(self.tokenizer, block[i].detach().cpu().tolist(), self.special_ids)
            or "[empty]"
            for i in range(block.shape[0])
        ]
        embs_all = encode_texts(
            texts,
            self.encoder,
            self.encoder_tokenizer,
            self.device,
            batch_sz=min(32, block.shape[0]),
        )
        raw_rollout_scores = None
        if effective_rollouts > 1:
            b = min(block_id, self.directions.shape[0] - 1)
            num_bits = self.directions.shape[1]
            raw_start = (int(step) * self.channels_per_step) % num_bits
            raw_channels = torch.tensor(
                [(raw_start + j) % num_bits for j in range(self.channels_per_step)],
                dtype=torch.long,
            )
            raw_signed = (
                embs_all @ self.directions[b, raw_channels].T
            ) * self.signs[b, raw_channels].unsqueeze(0)
            raw_unique = raw_signed.mean(dim=1).view(
                rollout_candidate_count, effective_rollouts
            )
            raw_rollout_scores = raw_unique.index_select(
                0, inverse_idx.to(raw_unique.device)
            )
        if effective_rollouts > 1:
            embs_unique = embs_all.view(rollout_candidate_count, effective_rollouts, -1).mean(dim=1)
        else:
            embs_unique = embs_all
        embs = embs_unique.index_select(0, inverse_idx.to(embs_unique.device))

        b = min(block_id, self.directions.shape[0] - 1)
        num_bits = self.directions.shape[1]
        start = (int(step) * self.channels_per_step) % num_bits
        chan_idx = torch.tensor(
            [(start + j) % num_bits for j in range(self.channels_per_step)],
            dtype=torch.long,
        )
        dirs = self.directions[b, chan_idx]
        signs = self.signs[b, chan_idx]
        signed = (embs @ dirs.T) * signs.unsqueeze(0)
        wm_scores = signed.mean(dim=1)

        mean_lps_vals = []
        for row_idx in range(k):
            row_pos = position_rows[row_idx]
            row_tok = candidate_tokens[row_idx, : row_pos.numel()]
            row_log_probs = F.log_softmax(logits[0, row_pos].float(), dim=-1)
            row_lps = row_log_probs.gather(1, row_tok.unsqueeze(-1)).squeeze(-1)
            mean_lps_vals.append(row_lps.sum())
        mean_lps = torch.stack(mean_lps_vals).cpu()
        total_scores = wm_scores + self.logprob_weight * mean_lps
        valid_indices = _top_logprob_indices(mean_lps.to(total_scores.device), self.argmax_logprob_top_frac)
        argmax_best = int(valid_indices[torch.argmax(total_scores[valid_indices])].item())
        best = argmax_best
        selection_info = {
            "selector": "max_watermark",
            "selection_mode": "argmax_score",
            "selected_candidate_index": int(best),
        }

        unique = len(
            {
                tuple(
                    zip(
                        position_rows[i].detach().cpu().tolist(),
                        candidate_tokens[i, : position_rows[i].numel()].detach().cpu().tolist(),
                    )
                )
                for i in range(k)
            }
        )
        diag = {
            "step": int(step),
            "block_id": int(block_id),
            "num_positions": int(position_rows[0].numel()),
            "unique_candidate_position_count": int(
                len({tuple(row.detach().cpu().tolist()) for row in position_rows})
            ),
            "best_candidate_index": best,
            "argmax_candidate_index": int(argmax_best),
            "argmax_logprob_top_frac": float(self.argmax_logprob_top_frac),
            "argmax_logprob_num_keep": int(valid_indices.numel()),
            "argmax_logprob_valid_indices": [int(i) for i in valid_indices.detach().cpu().tolist()],
            "base_watermark_score": float(wm_scores[0].item()),
            "best_watermark_score": float(wm_scores[best].item()),
            "old_argmax_watermark_score": float(wm_scores[argmax_best].item()),
            "old_argmax_total_score": float(total_scores[argmax_best].item()),
            "wm_score_range": float((wm_scores.max() - wm_scores.min()).item()),
            "unique_candidate_count": int(unique),
            "candidate_position_mode": self.candidate_position_mode,
            "candidate_temperature": self.candidate_temperature,
            "rollout_temperature": self.rollout_temperature,
            # A fully resolved candidate has a deterministic continuation.  We
            # compute it once, but it represents `logical_rollouts` identical
            # R samples so the diagnostic raw shape still matches constant R.
            "rollouts_per_cand_effective": int(logical_rollouts),
            "rollouts_per_cand_computed": int(effective_rollouts),
            "rollouts_per_cand_target_avg": int(self.rollouts_per_cand),
            "rollout_schedule": self.rollout_schedule,
            "rollout_batch_size": int(block.shape[0]),
            "shared_rollout_seeds": self.shared_rollout_seeds,
            "dedup_candidates": self.dedup_candidates,
            "rollout_logits_mode": self.rollout_logits_mode,
            "candidate_duplicate_count": int(duplicate_count),
            "rollout_candidate_count": int(rollout_candidate_count),
            "remaining_before_commit": int(remaining_before_commit),
            "rollout_raw_watermark_scores": (
                raw_rollout_scores.tolist()
                if raw_rollout_scores is not None
                else [
                    [float(value)] * int(logical_rollouts)
                    for value in wm_scores.tolist()
                ]
            ),
            "candidate_watermark_scores": [float(value) for value in wm_scores.tolist()],
            "resample_base_candidate": self.resample_base_candidate,
            "selection_mode": "argmax_score",
        }
        diag.update(selection_info)
        return best, diag

    @torch.no_grad()
    def __call__(self, step, x: torch.Tensor, logits: torch.Tensor | None) -> torch.Tensor:
        if step is None or logits is None or self.prev_x is None:
            self.prev_x = x.detach().clone()
            return x

        gen_s = self.prompt_len
        gen_e = self.prompt_len + self.gen_length
        prev_gen = self.prev_x[:, gen_s:gen_e]
        cur_gen = x[:, gen_s:gen_e]
        changed = (prev_gen == self.mask_id) & (cur_gen != self.mask_id)
        forced = None
        if self.forced_candidate_mask is not None:
            forced = self.forced_candidate_mask[:, gen_s:gen_e].to(device=x.device)
            forced = forced & (prev_gen == self.mask_id)

        forced_active = forced is not None and forced.any()
        native_changed = changed
        if forced_active:
            gen_view = x[:, gen_s:gen_e]
            gen_view = torch.where(native_changed, torch.full_like(gen_view, self.mask_id), gen_view)
            x[:, gen_s:gen_e] = gen_view
            cur_gen = x[:, gen_s:gen_e]
            changed = (prev_gen == self.mask_id) & (cur_gen != self.mask_id)

        candidate_mask = forced if forced_active else changed
        if not candidate_mask.any():
            self.prev_x = x.detach().clone()
            self.forced_candidate_mask = None
            return x

        changed_pos = candidate_mask[0].nonzero(as_tuple=True)[0] + gen_s
        block_ids = torch.div(changed_pos - gen_s, self.block_size, rounding_mode="floor")
        unique_block_ids = block_ids.unique(sorted=True)
        # Every semantic group in a DREAM step must be scored from the same
        # pre-commit canvas. Otherwise later groups would condition their
        # rollouts on tokens chosen by earlier groups in the same step.
        score_x = x.detach().clone()
        pending_assignments = []
        for group_index, block_id_t in enumerate(unique_block_ids):
            block_id = int(block_id_t.item())
            positions = changed_pos[block_ids == block_id_t]
            if positions.numel() == 0:
                continue

            base_tokens = score_x[0, positions].clone()
            if self.candidate_position_mode == "same_positions":
                b_s = self.prompt_len + block_id * self.block_size
                b_e = min(b_s + self.block_size, self.prompt_len + self.gen_length)
                block_mask = (score_x[0, b_s:b_e] == self.mask_id).nonzero(as_tuple=True)[0] + b_s
                remaining_before_commit = int(
                    torch.cat([block_mask, positions]).unique().numel()
                )
                pos_logits = logits[0, positions].float()
                num_samples = self.num_candidates if self.resample_base_candidate else max(0, self.num_candidates - 1)
                samples = sample_from_logits(
                    pos_logits.unsqueeze(0).expand(num_samples, -1, -1),
                    self.candidate_temperature,
                )
                if self.resample_base_candidate:
                    candidate_tokens = samples
                else:
                    needs_base_sample = base_tokens == self.mask_id
                    if bool(needs_base_sample.any().item()):
                        sampled_base = sample_from_logits(pos_logits, self.candidate_temperature)
                        base_tokens = torch.where(needs_base_sample, sampled_base, base_tokens)
                    candidate_tokens = torch.cat([base_tokens.unsqueeze(0), samples], dim=0)
                candidate_positions = positions
            elif self.candidate_position_mode in {
                "per_candidate_random_block",
                "per_candidate_random_semantic_unit",
            }:
                b_s = self.prompt_len + block_id * self.block_size
                b_e = min(b_s + self.block_size, self.prompt_len + self.gen_length)
                block_mask = (score_x[0, b_s:b_e] == self.mask_id).nonzero(as_tuple=True)[0] + b_s
                pool = torch.cat([positions, block_mask]).unique(sorted=False)
                remaining_before_commit = int(pool.numel())
                n_pick = min(self.cand_block_size, int(pool.numel()))
                rows_pos = []
                rows_tok = []
                if not self.resample_base_candidate:
                    needs_base_sample = base_tokens == self.mask_id
                    if bool(needs_base_sample.any().item()):
                        sampled_base = sample_from_logits(logits[0, positions].float(), self.candidate_temperature)
                        base_tokens = torch.where(needs_base_sample, sampled_base, base_tokens)
                    rows_pos.append(positions)
                    rows_tok.append(base_tokens)
                for _ in range(self.num_candidates - len(rows_pos)):
                    pick = pool[torch.randperm(pool.numel(), device=pool.device)[:n_pick]]
                    tok = sample_from_logits(logits[0, pick].float(), self.candidate_temperature)
                    rows_pos.append(pick)
                    rows_tok.append(tok)
                candidate_positions = rows_pos
                candidate_tokens = torch.stack(rows_tok, dim=0)
            else:
                raise ValueError(f"unknown candidate_position_mode: {self.candidate_position_mode}")
            best, diag = self._score_candidates(
                candidate_tokens,
                score_x,
                candidate_positions,
                block_id,
                logits,
                int(step),
                remaining_before_commit,
            )
            diag["forced_semantic_unit"] = bool(forced_active)
            diag["native_transfers_discarded"] = bool(forced_active)
            diag["semantic_group_index"] = int(group_index)
            diag["semantic_group_count"] = int(unique_block_ids.numel())
            diag["global_step_position_count"] = int(changed_pos.numel())
            diag["global_step_positions"] = [
                int(p) for p in changed_pos.detach().cpu().tolist()
            ]
            if isinstance(candidate_positions, torch.Tensor):
                pending_assignments.append(
                    (candidate_positions, candidate_tokens[best].clone())
                )
            else:
                best_pos = candidate_positions[best]
                pending_assignments.append(
                    (best_pos, candidate_tokens[best, : best_pos.numel()].clone())
                )
            self.diag.append(diag)

        for positions, tokens in pending_assignments:
            x[0, positions] = tokens

        self.prev_x = x.detach().clone()
        self.forced_candidate_mask = None
        return x


def score_generated_tokens(
    tokens: list[int],
    tokenizer,
    encoder,
    encoder_tokenizer,
    directions: torch.Tensor,
    signs: torch.Tensor,
    *,
    gen_length: int,
    block_size: int,
    special_ids: set[int],
    device: str,
) -> dict:
    num_blocks = math.ceil(gen_length / block_size)
    texts = []
    owners = []
    for b in range(num_blocks):
        s = b * block_size
        e = min(s + block_size, gen_length)
        text = safe_decode_dream(tokenizer, tokens[s:e], special_ids)
        if text:
            texts.append(text)
            owners.append(b)
    if not texts:
        return {"det_active": 0.0, "n_active_blocks": 0, "per_block_signed": []}

    embs = encode_texts(texts, encoder, encoder_tokenizer, device, batch_sz=32)
    per_block = [None] * num_blocks
    vals = []
    for emb, b in zip(embs, owners):
        proj = (emb @ directions[b].T).numpy()
        signed = proj * signs[b].numpy()
        val = float(signed.mean())
        per_block[b] = val
        vals.append(val)
    return {
        "det_active": float(np.mean(vals)),
        "n_active_blocks": len(vals),
        "per_block_signed": per_block,
    }


def load_rows(path: Path, n: int, offset: int) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[offset : offset + n]


def generate_dream() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generator_family", choices=["dream"], default="dream")
    p.add_argument("--prompts_jsonl", type=Path, required=True)
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--encoder_model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--prompt_variant", default="default", choices=["default", "long_output"])
    p.add_argument("--max_prompt_chars", type=int, default=0,
                   help="If >0, keep only the tail of the model input before chat templating.")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--gen_length", type=int, default=300)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--block_size", type=int, default=25)
    p.add_argument("--cand_block_size", type=int, default=1,
                   help=(
                       "r: number of MASK positions each candidate perturbs inside the selected semantic unit; "
                       "with random_global_grouped, the global positions decoded per DREAM step."
                   ))
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--num_message_bits", type=int, default=2)
    p.add_argument("--channels_per_step", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--candidate_temperature", type=float, default=0.6)
    p.add_argument("--rollout_temperature", type=float, default=0.5)
    p.add_argument("--rollouts_per_cand", type=int, default=3, help="Target average rollout count per candidate.")
    p.add_argument(
        "--rollout_schedule",
        default="linear_decay",
        choices=["none", "constant", "linear_decay"],
        help="linear_decay matches the LLaDA block-watermark schedule: high R early, low R late.",
    )
    p.add_argument(
        "--shared_rollout_seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use common rollout Gumbel noise across candidates for each rollout index.",
    )
    p.add_argument(
        "--dedup_candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Roll out identical candidate semantic-block states only once, then map scores back to original candidates.",
    )
    p.add_argument(
        "--rollout_logits_mode",
        default="full",
        choices=["full", "window"],
        help=(
            "full matches the original Dream semantic-watermark runs; window uses "
            "num_logits_to_keep to save memory for very long prompts."
        ),
    )
    p.add_argument(
        "--rollout_batch_size",
        type=int,
        default=0,
        help="Maximum rollout rows per generator/scoring microbatch; 0 keeps the full batch.",
    )
    p.add_argument("--logprob_weight", type=float, default=0.0)
    p.add_argument("--argmax_logprob_top_frac", type=float, default=1.0,
                   help="Restrict argmax watermark selection to the top fraction of candidates by base logprob.")
    p.add_argument(
        "--resample_base_candidate",
        action="store_true",
        help="Sample all K candidates from model logits instead of keeping candidate 0 as the DREAM base token.",
    )
    p.add_argument(
        "--candidate_position_mode",
        default="same_positions",
        choices=["same_positions", "per_candidate_random_block", "per_candidate_random_semantic_unit"],
        help=(
            "same_positions: K candidates resample the official DREAM transfer positions; "
            "per_candidate_random_semantic_unit: all K candidates pick their own random positions "
            "inside the selected semantic unit. per_candidate_random_block is kept as an alias."
        ),
    )
    p.add_argument(
        "--decode_position_mode",
        default="dream_official",
        choices=["dream_official", "random_semantic_unit", "random_global_grouped"],
        help=(
            "dream_official: keep DREAM's native global confidence transfer order; "
            "random_semantic_unit: choose one random active semantic block per step, "
            "then choose transfer positions randomly inside that block; "
            "random_global_grouped: choose r unresolved positions uniformly over the "
            "whole generation canvas, group them by semantic block, and independently "
            "rerank K candidates/rollouts for each group."
        ),
    )
    p.add_argument("--alg", default="origin")
    p.add_argument("--alg_temp", type=float, default=0.1)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--min_token_len", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument(
        "--max_rep4",
        type=float,
        default=None,
        help="Retry if repeated 4-gram ratio is above this value. Default keeps old behavior.",
    )
    p.add_argument("--direction_seed", type=int, default=42)
    p.add_argument("--message_seed", type=int, default=0)
    p.add_argument(
        "--orthogonal_directions",
        action="store_true",
        help="Use a QR-orthogonal semantic direction key instead of random unit vectors.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_id", type=int, default=None)
    p.add_argument("--prompt_mode", choices=["instruct", "base"], default="instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--device_map", default="cuda")
    args = p.parse_args()
    if args.argmax_logprob_top_frac <= 0.0 or args.argmax_logprob_top_frac > 1.0:
        raise ValueError("--argmax_logprob_top_frac must be in (0, 1]")
    if (
        args.decode_position_mode == "random_global_grouped"
        and args.candidate_position_mode != "same_positions"
    ):
        raise ValueError(
            "random_global_grouped requires --candidate_position_mode same_positions: "
            "the global random draw fixes each semantic group's positions before K candidates are sampled"
        )
    if args.decode_position_mode == "random_global_grouped":
        minimum_steps = math.ceil(args.gen_length / max(1, args.cand_block_size))
        if args.steps < minimum_steps:
            raise ValueError(
                f"random_global_grouped needs at least {minimum_steps} steps for "
                f"gen_length={args.gen_length}, r={args.cand_block_size}; got {args.steps}"
            )

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    num_blocks = math.ceil(args.gen_length / args.block_size)
    dirs, signs = build_directions(
        num_blocks,
        args.num_message_bits,
        args.direction_seed,
        args.message_seed,
        orthogonal=args.orthogonal_directions,
    )

    print(f"Loading DREAM native model: {args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    args.mask_id = resolve_mask_id(tokenizer, "dream", args.mask_id)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    ).eval()

    print(f"Loading encoder: {args.encoder_model}", flush=True)
    enc_tok = AutoTokenizer.from_pretrained(args.encoder_model)
    enc = AutoModel.from_pretrained(args.encoder_model, torch_dtype=torch.float32).to(args.device).eval()

    rows = load_rows(args.prompts_jsonl, args.num_samples, args.offset)
    print(
        f"Generating {len(rows)} DREAM-native semantic-watermark samples "
        f"K={args.num_candidates} bs={args.block_size} steps={args.steps} alg={args.alg}",
        flush=True,
    )
    print(
        f"  candT={args.candidate_temperature} rT={args.rollout_temperature} "
        f"R={args.rollouts_per_cand} rollout_schedule={args.rollout_schedule} "
        f"orthogonal_directions={args.orthogonal_directions} "
        f"shared_rollout_seeds={args.shared_rollout_seeds} "
        f"dedup_candidates={args.dedup_candidates} "
        f"rollout_logits_mode={args.rollout_logits_mode} "
        f"rollout_batch_size={args.rollout_batch_size} "
        f"r={args.cand_block_size} "
        f"candidate_position_mode={args.candidate_position_mode} "
        f"decode_position_mode={args.decode_position_mode} "
        "selection_mode=argmax_score",
        flush=True,
    )

    special_ids = {
        args.mask_id,
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    }
    special_ids = {int(x) for x in special_ids if x is not None}

    with args.output.open("w") as out_f:
        for local_idx, row in enumerate(tqdm(rows, desc="dream-native-wm")):
            generation_started = time.perf_counter()
            model_input, prompt_input, prompt_context = build_model_input_from_row(row, args.dataset, args.prompt_variant)
            model_input, prompt_truncated = truncate_tail(model_input, args.max_prompt_chars)
            prompt_text, prompt_ids = build_prompt(model_input, "", tokenizer, args.prompt_mode)
            input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=args.device).unsqueeze(0)
            tokens: list[int] = []
            text = ""
            token_len = 0
            rep4 = 0.0
            attempts = 0
            generation_attempt_seed = None
            hook = None
            decode_controller = None
            for attempt in range(max(0, args.max_retries) + 1):
                attempts = attempt + 1
                if args.seed is not None:
                    prompt_seed = int(args.seed) + (args.offset + local_idx) * 1000 + attempt
                    generation_attempt_seed = prompt_seed
                    torch.manual_seed(prompt_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(prompt_seed)
                hook = NativeSemanticWatermarkHook(
                    model,
                    tokenizer,
                    enc,
                    enc_tok,
                    dirs,
                    signs,
                    prompt_len=input_ids.shape[1],
                    gen_length=args.gen_length,
                    block_size=args.block_size,
                    cand_block_size=args.cand_block_size,
                    num_candidates=args.num_candidates,
                    channels_per_step=args.channels_per_step,
                    candidate_temperature=args.candidate_temperature,
                    rollout_temperature=args.rollout_temperature,
                    rollouts_per_cand=args.rollouts_per_cand,
                    rollout_schedule=args.rollout_schedule,
                    logprob_weight=args.logprob_weight,
                    candidate_position_mode=args.candidate_position_mode,
                    mask_id=args.mask_id,
                    device=args.device,
                    argmax_logprob_top_frac=args.argmax_logprob_top_frac,
                    sample_id=f"{args.offset + local_idx}:{attempt}",
                    resample_base_candidate=args.resample_base_candidate,
                    shared_rollout_seeds=args.shared_rollout_seeds,
                    dedup_candidates=args.dedup_candidates,
                    rollout_logits_mode=args.rollout_logits_mode,
                    rollout_batch_size=args.rollout_batch_size,
                )
                decode_controller = DreamSemanticUnitDecodeController(
                    prompt_len=input_ids.shape[1],
                    gen_length=args.gen_length,
                    block_size=args.block_size,
                    cand_block_size=args.cand_block_size,
                    mask_id=args.mask_id,
                    steps=args.steps,
                    eps=args.eps,
                    mode=args.decode_position_mode,
                )

                def tokens_hook(step, x, logits):
                    hook.forced_candidate_mask = decode_controller.allowed_mask
                    return hook(step, x, logits)

                output = model.diffusion_generate(
                    input_ids,
                    max_new_tokens=args.gen_length,
                    steps=args.steps,
                    eps=args.eps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    alg=args.alg,
                    alg_temp=args.alg_temp,
                    mask_token_id=args.mask_id,
                    generation_logits_hook_func=decode_controller.logits_hook,
                    generation_tokens_hook_func=tokens_hook,
                )
                seq = output.sequences if hasattr(output, "sequences") else output
                tokens = seq[0, input_ids.shape[1] : input_ids.shape[1] + args.gen_length].tolist()
                text = clean_text(safe_decode_dream(tokenizer, tokens, special_ids))
                token_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                rep4 = rep_ngram(text, 4)
                long_enough = token_len >= args.min_token_len
                not_repetitive = args.max_rep4 is None or rep4 <= args.max_rep4
                if long_enough and not_repetitive:
                    break
            generation_seconds = time.perf_counter() - generation_started
            det = score_generated_tokens(
                tokens,
                tokenizer,
                enc,
                enc_tok,
                dirs,
                signs,
                gen_length=args.gen_length,
                block_size=args.block_size,
                special_ids=special_ids,
                device=args.device,
            )
            out_f.write(
                json.dumps(
                    {
                        "prompt_idx": args.offset + local_idx,
                        "source_index": row.get("_ablation_source_index"),
                        "sample_id": f"{args.offset + local_idx}:{max(0, attempts - 1)}",
                        "watermark_sample_id": f"{args.offset + local_idx}:{max(0, attempts - 1)}",
                        "generation_attempt_seed": generation_attempt_seed,
                        "prompt_full": prompt_text,
                        "prompt_input": prompt_input,
                        "prompt_context": prompt_context,
                        "prompt_truncated": bool(prompt_truncated),
                        "max_prompt_chars": int(args.max_prompt_chars),
                        "text": text,
                        "generated_token_ids": tokens,
                        "token_len": token_len,
                        "generation_seconds": generation_seconds,
                        "seconds_per_visible_token": generation_seconds / max(1, token_len),
                        "word_len": len(text.split()),
                        "rep4": rep4,
                        "retry_attempts": attempts,
                        "too_short": bool(args.min_token_len and token_len < args.min_token_len),
                        "too_repetitive": bool(args.max_rep4 is not None and rep4 > args.max_rep4),
                        "passed_quality": bool(
                            (not args.min_token_len or token_len >= args.min_token_len)
                            and (args.max_rep4 is None or rep4 <= args.max_rep4)
                        ),
                        "det_score": det["det_active"],
                        "det_active_blocks": det["n_active_blocks"],
                        "per_block_det": det["per_block_signed"],
                        "wm_gen_diag": hook.diag if hook is not None else [],
                        "decode_order_diag": decode_controller.diag if decode_controller is not None else [],
                        "gen_config": {
                            "generator_family": "dream_native",
                            "steps": args.steps,
                            "eps": args.eps,
                            "alg": args.alg,
                            "temperature": args.temperature,
                            "candidate_temperature": args.candidate_temperature,
                            "rollout_temperature": args.rollout_temperature,
                            "rollouts_per_cand": args.rollouts_per_cand,
                            "rollout_schedule": args.rollout_schedule,
                            "shared_rollout_seeds": args.shared_rollout_seeds,
                            "dedup_candidates": args.dedup_candidates,
                            "rollout_logits_mode": args.rollout_logits_mode,
                            "rollout_batch_size": args.rollout_batch_size,
                            "candidate_position_mode": args.candidate_position_mode,
                            "decode_position_mode": args.decode_position_mode,
                            "block_size": args.block_size,
                            "cand_block_size": args.cand_block_size,
                            "num_candidates": args.num_candidates,
                            "channels_per_step": args.channels_per_step,
                            "num_message_bits": args.num_message_bits,
                            "orthogonal_directions": args.orthogonal_directions,
                            "logprob_weight": args.logprob_weight,
                            "argmax_logprob_top_frac": args.argmax_logprob_top_frac,
                            "selection_mode": "argmax_score",
                            "watermark_sample_id": f"{args.offset + local_idx}:{max(0, attempts - 1)}",
                            "resample_base_candidate": args.resample_base_candidate,
                            "prompt_variant": args.prompt_variant,
                            "max_prompt_chars": args.max_prompt_chars,
                            "min_token_len": args.min_token_len,
                            "max_retries": args.max_retries,
                            "max_rep4": args.max_rep4,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

    print(f"Saved {args.output}", flush=True)


