"""Parrot sentence-wise paraphrase attack with bigram candidate selection.

This module extracts the SemStamp ``bigram=True`` candidate selection idea into
standalone utilities: generate or receive paraphrase candidates, keep candidates
whose semantic score does not drop too much, then pick the one with the smallest
token bigram overlap with the original text.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

def clean_text(text: str) -> str:
    return " ".join((text or "").replace("NEWLINE_CHAR", " ").split())


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [part.strip() for part in parts if part.strip()]


def _to_list_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "detach"):
        encoded = encoded.detach().cpu()
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(x) for x in encoded]


def tokenize(tokenizer: Any, text: str, *, add_special_tokens: bool = False) -> list[int]:
    """Tokenize text to ids while accepting HF and small test tokenizers."""
    try:
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    except TypeError:
        encoded = tokenizer(text)
    return _to_list_ids(encoded)


def build_bigrams(input_ids: Sequence[int]) -> list[tuple[int, int]]:
    bigrams = []
    for i in range(len(input_ids) - 1):
        bigrams.append((int(input_ids[i]), int(input_ids[i + 1])))
    return bigrams


def compare_ngram_overlap(
    input_ngram: Sequence[tuple[int, ...]],
    para_ngram: Sequence[tuple[int, ...]],
) -> int:
    """Count candidate ngrams whose type appears in the original.

    This intentionally mirrors SemStamp's counting rule: multiplicity is taken
    from the paraphrase side, not the minimum count across both sides.
    """
    input_c = Counter(input_ngram)
    para_c = Counter(para_ngram)
    overlap = 0
    for item in input_c.keys() & para_c.keys():
        overlap += para_c[item]
    return int(overlap)


class SemanticScorer(Protocol):
    def score(self, original: str, candidates: Sequence[str]) -> list[float]:
        ...


class ConstantSemanticScorer:
    """Semantic scorer for tests or pre-filtered candidates."""

    def __init__(self, score: float = 1.0):
        self.value = float(score)

    def score(self, original: str, candidates: Sequence[str]) -> list[float]:
        return [self.value for _ in candidates]


class BertScoreSemanticScorer:
    """BERTScore F1 scorer matching SemStamp's default model."""

    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        *,
        device: str | None = None,
        lang: str = "en",
        rescale_with_baseline: bool = True,
    ):
        try:
            import torch
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError("BertScoreSemanticScorer requires `bert-score` and `torch`.") from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scorer = BERTScorer(
            model_type=model_type,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
            lang=lang,
        )

    def score(self, original: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        _, _, f1 = self.scorer.score([original] * len(candidates), list(candidates))
        return [float(x) for x in f1.detach().cpu().tolist()]


class SentenceTransformerSemanticScorer:
    """Cosine-similarity fallback using the repo's sentence-transformers dep."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", *, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("SentenceTransformerSemanticScorer requires `sentence-transformers`.") from exc
        self.model = SentenceTransformer(model_name, device=device)

    def score(self, original: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        import numpy as np

        embeddings = self.model.encode([original, *candidates], normalize_embeddings=True)
        original_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        return [float(np.dot(original_emb, emb)) for emb in candidate_embs]


@dataclass(frozen=True)
class CandidateScore:
    index: int
    text: str
    token_length: int
    bigram_overlap: int
    semantic_score: float
    semantic_drop: float
    accepted: bool


@dataclass(frozen=True)
class BigramAttackResult:
    original: str
    selected: str
    selected_index: int
    baseline_semantic_score: float
    bert_threshold: float
    max_length_ratio: float
    candidates: list[CandidateScore]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidates"] = [asdict(c) for c in self.candidates]
        return data


def select_by_bigram_overlap(
    sent: str,
    para_sents: Sequence[str],
    tokenizer: Any,
    semantic_scorer: SemanticScorer | None = None,
    *,
    bert_threshold: float = 0.03,
    max_length_ratio: float = 1.5,
    min_similarity: float | None = None,
    baseline: str = "first",
    add_special_tokens: bool = False,
) -> BigramAttackResult:
    """Pick the accepted candidate with the smallest token-bigram overlap.

    ``bert_threshold`` follows SemStamp: a candidate is accepted when its
    semantic-score drop from the baseline is at most
    ``bert_threshold * abs(baseline_score)``. The SemStamp baseline is the first
    paraphrase candidate; pass ``baseline="best"`` to compare against the best
    candidate score instead.
    """
    original = clean_text(sent)
    candidates = [clean_text(p) for p in para_sents if clean_text(p)]
    if not candidates:
        return BigramAttackResult(
            original=original,
            selected=original,
            selected_index=-1,
            baseline_semantic_score=1.0,
            bert_threshold=bert_threshold,
            max_length_ratio=max_length_ratio,
            candidates=[],
        )
    if baseline not in {"first", "best"}:
        raise ValueError("baseline must be 'first' or 'best'")

    semantic_scorer = semantic_scorer or ConstantSemanticScorer()
    input_ids = tokenize(tokenizer, original, add_special_tokens=add_special_tokens)
    input_bigrams = build_bigrams(input_ids)
    candidate_ids = [tokenize(tokenizer, candidate, add_special_tokens=add_special_tokens) for candidate in candidates]
    candidate_bigrams = [build_bigrams(ids) for ids in candidate_ids]
    semantic_scores = semantic_scorer.score(original, candidates)
    if len(semantic_scores) != len(candidates):
        raise RuntimeError(
            f"semantic scorer returned {len(semantic_scores)} scores for {len(candidates)} candidates"
        )

    baseline_score = semantic_scores[0] if baseline == "first" else max(semantic_scores)
    allowed_drop = bert_threshold * max(abs(baseline_score), 1e-12)

    scored: list[CandidateScore] = []
    for i, candidate in enumerate(candidates):
        overlap = compare_ngram_overlap(input_bigrams, candidate_bigrams[i])
        semantic_drop = baseline_score - semantic_scores[i]
        accepted = (
            len(candidate_ids[i]) <= max_length_ratio * max(1, len(input_ids))
            and semantic_drop <= allowed_drop
            and (min_similarity is None or semantic_scores[i] >= min_similarity)
        )
        scored.append(
            CandidateScore(
                index=i,
                text=candidate,
                token_length=len(candidate_ids[i]),
                bigram_overlap=overlap,
                semantic_score=float(semantic_scores[i]),
                semantic_drop=float(semantic_drop),
                accepted=bool(accepted),
            )
        )

    accepted_scores = [candidate for candidate in scored if candidate.accepted]
    selected = min(accepted_scores, key=lambda c: (c.bigram_overlap, c.index)) if accepted_scores else scored[0]
    return BigramAttackResult(
        original=original,
        selected=selected.text,
        selected_index=selected.index,
        baseline_semantic_score=float(baseline_score),
        bert_threshold=bert_threshold,
        max_length_ratio=max_length_ratio,
        candidates=scored,
    )


def accept_by_bigram_overlap(
    sent: str,
    para_sents: Sequence[str],
    tokenizer: Any,
    semantic_scorer: SemanticScorer | None = None,
    *,
    bert_threshold: float = 0.03,
    max_length_ratio: float = 1.5,
    min_similarity: float | None = None,
    baseline: str = "first",
    add_special_tokens: bool = False,
) -> str:
    """SemStamp-compatible helper that returns only the selected text."""
    return select_by_bigram_overlap(
        sent,
        para_sents,
        tokenizer,
        semantic_scorer,
        bert_threshold=bert_threshold,
        max_length_ratio=max_length_ratio,
        min_similarity=min_similarity,
        baseline=baseline,
        add_special_tokens=add_special_tokens,
    ).selected


def install_markllm(markllm_dir: Path):
    sys.path.insert(0, str(markllm_dir))
    import attack_utils as au  # type: ignore

    return au


def build_safe_parrot_class(au):
    class SafeCandidateParrot(au.SParrot):
        """Keep the original Parrot path, with a valid deterministic one-beam case."""

        def __init__(self):
            super().__init__()
            self.candidate_counts: list[int] = []

        def reset_trace(self) -> None:
            self.candidate_counts.clear()

        def augment(self, *args, **kwargs):
            max_return_phrases = int(kwargs.get("max_return_phrases", 10))
            if max_return_phrases != 1:
                result = super().augment(*args, **kwargs)
                self.candidate_counts.append(len(result))
                return result

            input_phrase = kwargs.get("input_phrase", args[0] if args else "")
            use_gpu = bool(kwargs.get("use_gpu", False))
            max_length = int(kwargs.get("max_length", 32))
            adequacy_threshold = float(kwargs.get("adequacy_threshold", 0.90))
            fluency_threshold = float(kwargs.get("fluency_threshold", 0.90))
            diversity_ranker = str(kwargs.get("diversity_ranker", "levenshtein"))
            device = "cuda" if use_gpu else "cpu"
            self.model = self.model.to(device)

            if len(input_phrase) >= max_length:
                max_length += 32
            normalized = re.sub(r"[^a-zA-Z0-9 \?'\-/\:\.]", "", input_phrase)
            model_input = "paraphrase: " + normalized
            input_ids = self.tokenizer.encode(model_input, return_tensors="pt").to(device)
            preds = self.model.generate(
                input_ids,
                do_sample=False,
                max_length=max_length,
                num_beams=1,
                early_stopping=True,
                num_return_sequences=1,
            )
            paraphrases = set()
            for pred in preds:
                text = self.tokenizer.decode(pred, skip_special_tokens=True).lower()
                paraphrases.add(re.sub(r"[^a-zA-Z0-9 \?'\-]", "", text))

            if getattr(self, "_transformers_fallback", False):
                result = sorted(paraphrases)
                self.candidate_counts.append(len(result))
                return result

            adequacy = self.adequacy_score.filter(
                model_input, paraphrases, adequacy_threshold, device
            )
            if not adequacy:
                adequacy = paraphrases
            fluency = self.fluency_score.filter(adequacy, fluency_threshold, device)
            if not fluency:
                fluency = adequacy
            ranked = self.diversity_score.rank(model_input, fluency, diversity_ranker)
            result = [
                text
                for text, _ in sorted(
                    ranked.items(), key=lambda item: item[1], reverse=True
                )
            ]
            self.candidate_counts.append(len(result))
            return result

    return SafeCandidateParrot


def build_attacker(au, num_candidates: int):
    cfg = au.ParrotParaphraseConfig()
    cfg.num_beams = num_candidates
    cfg.use_bigram_filter = num_candidates > 1
    safe_parrot = build_safe_parrot_class(au)()
    attacker = au.ParrotParaphrase(cfg=cfg, parrot=safe_parrot)
    attacker.reset_trace = safe_parrot.reset_trace
    attacker.candidate_counts = safe_parrot.candidate_counts
    return attacker



def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markllm-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, choices=(1, 4, 7, 10), required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.output.is_file() and args.output.stat().st_size > 0:
        if args.skip_existing:
            print(f"skip existing {args.output}", flush=True)
            return
        raise FileExistsError(args.output)

    rows = read_jsonl(args.input)
    attack_utils = install_markllm(args.markllm_dir)
    attacker = build_attacker(attack_utils, args.num_candidates)
    attacked_rows = []
    for index, row in enumerate(rows):
        original = str(row.get(args.text_field) or "")
        if not original.strip():
            raise ValueError(f"{args.input}:{index + 1}: empty text")
        attacker.reset_trace()
        attacked = attacker.edit(original)
        attacked = attacked.strip() or original
        out = dict(row)
        out.update(
            {
                "original_text": original,
                "text": attacked,
                "completion": attacked,
                "attack_text": attacked,
                "attack": "parrot",
                "parrot_num_candidates": args.num_candidates,
                "parrot_bigram_selection": args.num_candidates > 1,
                "parrot_candidate_counts": list(attacker.candidate_counts),
            }
        )
        attacked_rows.append(out)
        if (index + 1) % 10 == 0:
            print(f"progress={index + 1}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in attacked_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"wrote={len(attacked_rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
