#!/usr/bin/env python3
"""Dataset preparation, prompts, and deterministic scoring for paper tasks."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("mmlu", "hellaswag", "arc_challenge", "gsm8k")
MC_BENCHMARKS = {"mmlu", "hellaswag", "arc_challenge"}
MODEL_DEFAULTS = {"llada8b": None, "llada15": None}
TASK_SETTINGS = {
    "llada": {
        "mmlu": (3, 3, 3),
        "hellaswag": (3, 3, 3),
        "arc_challenge": (512, 512, 512),
        "gsm8k": (512, 512, 512),
    },
}
MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
)


def _rows(dataset: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def _load_dataset(name: str, config: str | None, split: str):
    from datasets import load_dataset

    if config is None:
        return load_dataset(name, split=split)
    return load_dataset(name, config, split=split)


def _choice_letter(value: Any, labels: list[str] | None = None) -> str:
    alphabet = "ABCDE"
    if isinstance(value, int):
        return alphabet[value]
    text = str(value).strip().upper()
    if labels and text in labels:
        return alphabet[labels.index(text)]
    match = re.search(r"[A-E]", text)
    if not match:
        raise ValueError(f"cannot normalize choice answer: {value!r}")
    return match.group(0)


def _sample(rows: list[dict[str, Any]], n: int, seed: int) -> list[tuple[int, dict[str, Any]]]:
    if len(rows) < n:
        raise ValueError(f"dataset only contains {len(rows)} rows; need {n}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), n))
    return [(idx, rows[idx]) for idx in indices]


def _mmlu_records(limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Load the complete official MMLU test split with subject-matched demos."""
    test = _rows(_load_dataset("cais/mmlu", "all", "test"))
    dev = _rows(_load_dataset("cais/mmlu", "all", "dev"))
    demos_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in dev:
        demos_by_subject.setdefault(str(row["subject"]), []).append(row)
    unknown_subjects = {str(row["subject"]) for row in test} - set(MMLU_SUBJECTS)
    if unknown_subjects:
        raise ValueError(f"unexpected MMLU subjects: {sorted(unknown_subjects)}")
    indexed = list(enumerate(test))
    if limit is not None and limit < len(indexed):
        indexed = _sample(test, limit, seed)
    return [
        normalize_record(
            "mmlu",
            index,
            row,
            demonstrations=demos_by_subject[str(row["subject"])][:5],
        )
        for index, row in indexed
    ]


def normalize_record(
    benchmark: str,
    index: int,
    row: dict[str, Any],
    *,
    demonstrations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base = {"benchmark": benchmark, "source_index": int(index)}
    if benchmark == "mmlu":
        choices = list(row["choices"])
        return {
            **base,
            "sample_id": f"mmlu:{row['subject']}:{index}",
            "subject": row["subject"],
            "question": row["question"],
            "choices": choices,
            "reference_answer": _choice_letter(row["answer"]),
            "demonstrations": demonstrations or [],
        }
    if benchmark == "hellaswag":
        endings = list(row.get("endings") or [row[k] for k in ("A", "B", "C", "D")])
        return {
            **base,
            "sample_id": f"hellaswag:{row.get('ind', index)}",
            "context": row.get("ctx") or f"{row.get('ctx_a', '')} {row.get('ctx_b', '')}".strip(),
            "choices": endings,
            "reference_answer": _choice_letter(int(row["label"])),
        }
    if benchmark == "arc_challenge":
        choices_obj = row["choices"]
        labels = [str(x).upper() for x in choices_obj["label"]]
        texts = list(choices_obj["text"])
        if not 2 <= len(texts) <= 5:
            raise ValueError(f"unexpected ARC choice count {len(texts)} for {index}")
        return {
            **base,
            "sample_id": f"arc_challenge:{row.get('id', index)}",
            "question": row["question"],
            "choices": texts,
            "reference_answer": _choice_letter(row["answerKey"], labels),
        }
    if benchmark == "gsm8k":
        answer = str(row["answer"])
        return {
            **base,
            "sample_id": f"gsm8k:{index}",
            "question": row["question"],
            "reference_answer": extract_number(answer),
            "reference_rationale": answer,
            "demonstrations": demonstrations or [],
        }
    raise ValueError(benchmark)


def prepare_manifest(
    output: Path,
    n: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = {"mmlu": _mmlu_records(n, seed)}
    gsm8k_demonstrations = _rows(_load_dataset("openai/gsm8k", "main", "train"))[:4]
    specs = {
        "hellaswag": ("Rowan/hellaswag", None, "validation"),
        "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation"),
        "gsm8k": ("openai/gsm8k", "main", "test"),
    }
    for offset, (benchmark, (name, config, split)) in enumerate(specs.items(), start=1):
        rows = _rows(_load_dataset(name, config, split))
        indexed = list(enumerate(rows)) if n is None else _sample(rows, n, seed + offset)
        selected[benchmark] = [
            normalize_record(
                benchmark,
                idx,
                row,
                demonstrations=gsm8k_demonstrations if benchmark == "gsm8k" else None,
            )
            for idx, row in indexed
        ]

    payload = {
        "seed": seed,
        "limit_per_benchmark": n,
        "selection_policy": (
            "complete official evaluation splits"
            if n is None
            else "deterministic random smoke-test subset"
        ),
        "datasets": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return payload


def build_prompt(record: dict[str, Any]) -> str:
    benchmark = record["benchmark"]
    if benchmark == "mmlu":
        subject = str(record["subject"]).replace("_", " ")
        hint = f"There is a single choice question about {subject}. Answer the question by replying A, B, C or D."
        chunks = []
        for demo in record.get("demonstrations", []):
            choices = demo["choices"]
            chunks.append(
                f"{hint}\nQuestion: {demo['question']}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer: {_choice_letter(demo['answer'])}"
            )
        c = record["choices"]
        chunks.append(f"{hint}\nQ: {record['question']}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nA:")
        return "\n\n".join(chunks)
    if benchmark == "hellaswag":
        c = record["choices"]
        return (
            f"{record['context']}\nQuestion: Which ending makes the most sense?\n"
            f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\n"
            "You may choose from 'A', 'B', 'C', 'D'.\nAnswer:"
        )
    if benchmark == "arc_challenge":
        c = record["choices"]
        options = "\n".join(f"{'ABCDE'[idx]}. {value}" for idx, value in enumerate(c))
        return f"Question: {record['question']}\n{options}\nAnswer:"
    if benchmark == "gsm8k":
        turns = []
        for demonstration in record.get("demonstrations", []):
            answer = str(demonstration["answer"]).replace("####", "The answer is")
            turns.append(
                f"Question: {demonstration['question']}\n"
                f"Let's think step by step\nAnswer:\n{answer}"
            )
        turns.append(
            f"Question: {record['question']}\nLet's think step by step\nAnswer:"
        )
        return "\n\n".join(turns)
    raise ValueError(benchmark)


def extract_choice(text: str) -> str | None:
    patterns = (
        r"(?i)the correct answer is\s*\(?\s*([A-E])",
        r"(?i)answer\s*[:=]\s*\(?\s*([A-E])",
        r"(?<![A-Za-z])([A-E])(?![A-Za-z])",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def extract_number(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text.replace("$", ""))
    return matches[-1].replace(",", "") if matches else None


def score_non_code(record: dict[str, Any], completion: str) -> tuple[str | None, bool | None]:
    benchmark = record["benchmark"]
    if benchmark in MC_BENCHMARKS:
        parsed = extract_choice(completion)
    elif benchmark == "gsm8k":
        parsed = extract_number(completion)
    else:
        raise ValueError(f"unsupported paper benchmark: {benchmark}")
    return parsed, parsed is not None and parsed == str(record["reference_answer"])


def iter_records(manifest: dict[str, Any], benchmark: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    rows = manifest["datasets"][benchmark]
    yield from rows if limit is None else rows[:limit]
