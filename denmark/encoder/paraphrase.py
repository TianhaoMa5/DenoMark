#!/usr/bin/env python3
"""Generate sentence-level Pegasus positives for DenMark encoder training."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def split_sentences(text: str) -> list[str]:
    try:
        import nltk

        return [sentence.strip() for sentence in nltk.sent_tokenize(text) if sentence.strip()]
    except Exception:
        return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text) if piece.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="tuner007/pegasus_paraphrase")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    source_sentences: list[list[str]] = []
    sentence_rows: list[tuple[int, int, str]] = []
    for row_index, row in enumerate(rows):
        text = str(row.get("text") or "").strip()
        sentences = split_sentences(text) or ([text] if text else [])
        source_sentences.append(sentences)
        sentence_rows.extend(
            (row_index, sentence_index, sentence)
            for sentence_index, sentence in enumerate(sentences)
        )

    tokenizer = PegasusTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = PegasusForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device).eval()
    paraphrases = [["" for _ in sentences] for sentences in source_sentences]

    with torch.inference_mode():
        for start in range(0, len(sentence_rows), args.batch_size):
            batch = sentence_rows[start : start + args.batch_size]
            encoded = tokenizer(
                [item[2] for item in batch],
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(args.device)
            generated = model.generate(
                **encoded,
                max_length=args.max_length,
                num_beams=args.num_beams,
                num_return_sequences=1,
                repetition_penalty=1.03,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for (row_index, sentence_index, _), value in zip(batch, decoded):
                paraphrases[row_index][sentence_index] = value.strip()
            print(
                f"sentences={min(start + len(batch), len(sentence_rows))}/{len(sentence_rows)}",
                flush=True,
            )

    kept = 0
    dropped = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row, sentence_outputs in zip(rows, paraphrases):
            original = str(row.get("text") or "").strip()
            positive = " ".join(value for value in sentence_outputs if value).strip()
            if not positive or positive.casefold() == original.casefold():
                dropped += 1
                continue
            output_row = {"text": original, "positive": positive}
            handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"input_rows": len(rows), "kept_pairs": kept, "dropped": dropped}))


if __name__ == "__main__":
    main()
