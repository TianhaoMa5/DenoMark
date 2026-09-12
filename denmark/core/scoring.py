"""DenMark semantic-unit detection.

For each block_i in the (re-tokenized) text:
  1. Decode tokens at positions [i*block_size, (i+1)*block_size] -> block text.
  2. T_eta(block text) -> 768-d embedding.
  3. Project onto θ_i, multiply by sign_i, mean over message bits.
  4. Aggregate per-block scores via:
     - det_full   : mean over ALL blocks (empty blocks contribute 0)
     - det_active : mean over blocks with non-empty text

Threshold for binary decision is computed externally (calibrated against
clean baseline scores for a target FPR, e.g. 1%).

"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from denmark.core.model import build_directions, safe_decode

# Dataset quirk: WaterBench multinews encodes '\n' as literal 'NEWLINE_CHAR' (13 chars)
NEWLINE_CHAR_TOKEN = "NEWLINE_CHAR"


def clean_text(text: str) -> str:
    """Strip dataset-quirk literal tokens and collapse whitespace."""
    if not text:
        return text
    text = text.replace(NEWLINE_CHAR_TOKEN, " ")
    return " ".join(text.split())


@torch.no_grad()
def encode_one(text, enc, enc_tok, device):
    if not text.strip():
        return None
    inp = enc_tok([text], padding=True, truncation=True,
                  max_length=512, return_tensors="pt").to(device)
    out = enc(**inp)
    attn = inp["attention_mask"].unsqueeze(-1).float()
    emb = F.normalize(
        (out.last_hidden_state * attn).sum(1) / attn.sum(1).clamp(min=1e-9), dim=-1
    )
    return emb[0].cpu()


def special_ids_for_tokenizer(tokenizer):
    """Return only IDs declared special by the active tokenizer.

    Token IDs are tokenizer-specific. In particular, ordinary vocabulary IDs
    such as 0, 1, or 2 must not be removed unless this tokenizer explicitly
    marks them as special.
    """
    special = set()
    for tok_id in getattr(tokenizer, "all_special_ids", []) or []:
        if tok_id is not None:
            special.add(int(tok_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"):
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is not None:
            special.add(int(tok_id))
    return special


def block_decode(token_ids, b_s, b_e, tokenizer):
    block_toks = token_ids[b_s:b_e]
    special = special_ids_for_tokenizer(tokenizer)
    filtered = [int(t) for t in block_toks if int(t) not in special]
    if not filtered:
        return ""
    return safe_decode(tokenizer, filtered, skip_special_tokens=True).strip()


def detect_sample(token_ids, num_blocks, block_size, gen_length,
                  enc, enc_tok, llada_tok, dirs, signs, device):
    """Returns dict with det_full / det_active / per-block scores."""
    block_signed = np.zeros((num_blocks, dirs.shape[1]))
    block_valid = [False] * num_blocks
    for b in range(num_blocks):
        b_s = b * block_size
        b_e = min(b_s + block_size, gen_length)
        if b_s >= len(token_ids):
            continue
        block_text = block_decode(token_ids, b_s, b_e, llada_tok)
        if not block_text:
            continue
        emb = encode_one(block_text, enc, enc_tok, device)
        if emb is None:
            continue
        block_valid[b] = True
        proj = (emb @ dirs[b].T).numpy()
        block_signed[b] = proj * signs[b].numpy()

    active_idx = [b for b in range(num_blocks) if block_valid[b]]
    det_full = float(block_signed.mean(axis=0).mean())
    det_active = (float(block_signed[active_idx].mean(axis=0).mean())
                  if active_idx else 0.0)
    per_block_signed = [float(block_signed[b].mean()) if block_valid[b] else None
                        for b in range(num_blocks)]
    return {
        "n_active_blocks": len(active_idx),
        "det_full": det_full,
        "det_active": det_active,
        "per_block_signed": per_block_signed,
    }

