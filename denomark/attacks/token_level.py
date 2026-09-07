"""Paper token-level attacks: deletion, contextual substitution, and swapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


PUNCTUATION = re.compile(r"^[\W_]+$")
ATTACK_NAMES = ("deletion", "context_aware_substitution", "adjacent_swap")


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("NEWLINE_CHAR", " ").split())


def stable_rng(seed: int, row_id: str, attack: str, ratio: float) -> random.Random:
    payload = f"{seed}\0{row_id}\0{attack}\0{ratio:.8f}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


def row_identifier(row: dict[str, Any], index: int) -> str:
    for field in ("source_id", "id", "sample_id", "prompt_id", "prompt_idx"):
        if row.get(field) is not None:
            return str(row[field])
    return str(index)


def delete_words(words: list[str], ratio: float, rng: random.Random) -> tuple[list[str], int]:
    if not words or ratio == 0:
        return list(words), 0
    remove_count = min(len(words) - 1, int(round(len(words) * ratio)))
    removed = set(rng.sample(range(len(words)), remove_count)) if remove_count else set()
    return [word for index, word in enumerate(words) if index not in removed], len(removed)


def adjacent_swap(
    words: list[str], ratio: float, rng: random.Random
) -> tuple[list[str], int]:
    target_words = int(round(len(words) * ratio))
    starts = list(range(max(0, len(words) - 1)))
    rng.shuffle(starts)
    selected: list[int] = []
    occupied: set[int] = set()
    for start in starts:
        if start in occupied or start + 1 in occupied:
            continue
        selected.append(start)
        occupied.update((start, start + 1))
        if len(occupied) >= target_words:
            break
    output = list(words)
    for start in selected:
        output[start], output[start + 1] = output[start + 1], output[start]
    return output, len(occupied)


class ContextSubstituter:
    def __init__(self, model_name: str, device: str, top_k: int) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.model_name = model_name
        self.device = device
        self.top_k = top_k

    @staticmethod
    def eligible(word: str) -> bool:
        return bool(word) and not PUNCTUATION.match(word)

    def substitute(
        self, words: list[str], ratio: float, rng: random.Random
    ) -> tuple[list[str], int, int]:
        eligible = [index for index, word in enumerate(words) if self.eligible(word)]
        target = min(len(eligible), int(round(len(words) * ratio)))
        selected = rng.sample(eligible, target) if target else []
        if not selected:
            return list(words), 0, len(eligible)

        masked_texts = []
        for index in selected:
            masked = list(words)
            masked[index] = self.tokenizer.mask_token
            masked_texts.append(" ".join(masked))
        encoded = self.tokenizer(
            masked_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits

        output = list(words)
        changed = 0
        for batch_index, word_index in enumerate(selected):
            mask_positions = encoded.input_ids[batch_index].eq(
                self.tokenizer.mask_token_id
            ).nonzero(as_tuple=True)[0]
            if len(mask_positions) != 1:
                continue
            candidate_ids = torch.topk(
                logits[batch_index, mask_positions[0]], self.top_k
            ).indices.tolist()
            original = words[word_index]
            replacement = None
            for candidate in self.tokenizer.convert_ids_to_tokens(candidate_ids):
                if candidate.startswith("##") or PUNCTUATION.match(candidate):
                    continue
                if candidate.lower() == original.lower() or not candidate.isalpha():
                    continue
                replacement = candidate.capitalize() if original[:1].isupper() else candidate
                break
            if replacement is not None:
                output[word_index] = replacement
                changed += 1
        return output, changed, len(eligible)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--attacks", nargs="+", choices=ATTACK_NAMES, required=True)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=(0.1, 0.2, 0.3, 0.4, 0.5),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlm-model", default="bert-base-uncased")
    parser.add_argument("--mlm-top-k", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(not 0 <= ratio <= 1 for ratio in args.ratios):
        raise ValueError("ratios must lie in [0, 1]")
    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]

    context_substituter = None
    if "context_aware_substitution" in args.attacks:
        context_substituter = ContextSubstituter(
            args.mlm_model,
            args.device,
            args.mlm_top_k,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, row in enumerate(tqdm(rows, desc="token attacks")):
            text = normalize_text(row.get(args.text_field))
            if not text:
                raise ValueError(f"empty {args.text_field!r} at input row {index}")
            words = text.split()
            identifier = row_identifier(row, index)
            for attack in args.attacks:
                for ratio in args.ratios:
                    rng = stable_rng(args.seed, identifier, attack, ratio)
                    eligible = len(words)
                    if attack == "deletion":
                        attacked_words, changed = delete_words(words, ratio, rng)
                    elif attack == "adjacent_swap":
                        attacked_words, changed = adjacent_swap(words, ratio, rng)
                    else:
                        assert context_substituter is not None
                        attacked_words, changed, eligible = context_substituter.substitute(
                            words, ratio, rng
                        )
                    attacked_text = " ".join(attacked_words)
                    result = dict(row)
                    result.update(
                        {
                            "original_text": text,
                            "attacked_text": attacked_text,
                            "attack_type": attack,
                            "attack_ratio": ratio,
                            "requested_modified_words": int(round(len(words) * ratio)),
                            "actual_modified_words": changed,
                            "eligible_word_count": eligible,
                            "original_word_count": len(words),
                            "attacked_word_count": len(attacked_words),
                            "actual_modification_ratio": changed / max(1, len(words)),
                            "attack_seed": args.seed,
                            "attack_model": (
                                args.mlm_model
                                if attack == "context_aware_substitution"
                                else None
                            ),
                        }
                    )
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
