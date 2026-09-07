#!/usr/bin/env python3
"""Train the paper's E5 semantic encoder on original-Pegasus pairs."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


class PairDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                anchor = str(row.get("text") or "").strip()
                positive = str(row.get("positive") or "").strip()
                if not anchor or not positive:
                    raise ValueError(f"empty training pair at line {line_number}")
                self.rows.append((anchor, positive))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.rows[index]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)


def embed(model, tokenized: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(**tokenized)
    return F.normalize(mean_pool(output.last_hidden_state, tokenized["attention_mask"]), dim=-1)


def make_collate(tokenizer, max_length: int):
    def collate(batch: list[tuple[str, str]]):
        anchors, positives = zip(*batch)
        options = {
            "padding": True,
            "truncation": True,
            "max_length": max_length,
            "return_tensors": "pt",
        }
        return tokenizer(list(anchors), **options), tokenizer(list(positives), **options)

    return collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default="intfloat/e5-base-v2")
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(args.model_name_or_path).to(args.device).train()
    dataset = PairDataset(args.train_file)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=make_collate(tokenizer, args.max_seq_length),
        generator=generator,
    )
    total_steps = len(loader) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_bf16 = bool(args.bf16 and args.device.startswith("cuda"))

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.num_train_epochs):
        for anchors, positives in loader:
            anchors = {key: value.to(args.device) for key, value in anchors.items()}
            positives = {key: value.to(args.device) for key, value in positives.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with context:
                anchor_embeddings = embed(model, anchors)
                positive_embeddings = embed(model, positives)
                logits = anchor_embeddings @ positive_embeddings.T / args.temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.log_every == 0:
                accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
                print(
                    f"epoch={epoch + 1} step={global_step}/{total_steps} "
                    f"loss={loss.item():.6f} in_batch_accuracy={accuracy:.4f}",
                    flush=True,
                )

    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    config = {
        **vars(args),
        "train_file": str(args.train_file),
        "output_dir": str(args.output_dir),
        "training_pairs": len(dataset),
        "optimizer_steps": global_step,
        "warmup_steps": warmup_steps,
    }
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"saved={final_dir}")


if __name__ == "__main__":
    main()
