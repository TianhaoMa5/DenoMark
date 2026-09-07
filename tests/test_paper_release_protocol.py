import json
from pathlib import Path

from denomark.evaluation.build_release import selected_files
from denomark.evaluation.collect_metrics import normalize_metrics
from denomark.attacks.common import sentence_attack_spec, split_sentences
from denomark.evaluation.summarize_runtime import token_count
from denomark.data.prepare_waterbench import normalize_prompt


ROOT = Path(__file__).resolve().parent.parent


def test_machine_readable_matrix_covers_every_paper_axis():
    config = json.loads((ROOT / "configs/paper_experiments.json").read_text())
    assert set(config["datasets"]) == {"finance_qa", "alpacafarm", "longform_qa"}
    assert set(config["backbones"]) == {"llada8b", "llada15", "llada20mini", "dream"}
    assert set(config["methods"]) == {
        "denomark",
        "dlm_kgw",
        "dgmark",
        "patternmark",
        "umr",
    }
    assert set(config["downstream"]["benchmarks"]) == {
        "mmlu",
        "hellaswag",
        "arc_challenge",
        "gsm8k",
    }
    assert config["downstream"]["backbones"] == ["llada8b", "llada15"]
    assert config["downstream"]["temperature"] == 0.1
    assert config["downstream"]["evaluation_splits"] == "complete official evaluation splits"
    assert config["semi_ar_baselines"]["pmark"]["channels"] == 2
    assert config["semi_ar_baselines"]["semstamp"]["lsh_bits"] == 2
    assert config["ablations"]["candidate_temperature"] == [0.3, 0.5, 0.6, 0.75, 0.9]
    assert config["ablations"]["rollout_count"] == [1, 2, 3, 5, 8]
    assert config["ablations"]["scan_ranges"] == [
        [25, 25],
        [23, 27],
        [21, 29],
        [17, 33],
        [12, 37],
        [10, 40],
    ]
    assert len(config["paper_experiment_coverage"]) == 19


def test_data_construction_protocol_has_no_bundled_rows():
    config = json.loads((ROOT / "configs/paper_experiments.json").read_text())
    construction = config["data_construction"]
    assert construction["rows_per_dataset"] == 500
    assert construction["bundled_rows"] == 0
    assert not (ROOT / "data").exists()
    assert not (ROOT / "RELEASE_MANIFEST.json").exists()


def test_prompt_builder_removes_answers_and_applies_templates():
    source = {
        "question": "Why does diversification reduce risk?",
        "answer": "This must not be copied.",
        "source_id": "private-row-label",
    }
    row = normalize_prompt(source, "finance_qa")
    assert set(row) == {"prompt", "raw_prompt", "input", "context"}
    assert row["input"] == source["question"]
    assert row["context"] == ""
    assert source["answer"] not in row.values()
    assert source["source_id"] not in row.values()
    assert "financial knowledge within 300 words" in row["prompt"]


def test_release_builder_keeps_all_paper_methods_and_entrypoints():
    paths = {path.relative_to(ROOT).as_posix() for path in selected_files(ROOT)}
    assert "denomark/baselines/block_best_of_k/model.py" in paths
    assert "denomark/baselines/block_best_of_k/generate.py" in paths
    assert "denomark/baselines/pmark/generate.py" in paths
    assert "denomark/baselines/semstamp/generate.py" in paths
    assert "denomark/attacks/gpt_sentence_level.py" in paths
    assert "denomark/experiments/downstream/run.py" in paths
    assert "denomark/baselines/dlm_kgw/detect.py" in paths
    assert "denomark/evaluation/collect_metrics.py" in paths
    assert "denomark/evaluation/compare_ppl.py" in paths
    assert "denomark/evaluation/plot_figures.py" in paths
    assert "denomark/evaluation/audit_reproducibility.py" in paths
    assert "denomark/encoder/prepare.py" in paths
    assert "denomark/encoder/paraphrase.py" in paths
    assert "denomark/encoder/merge.py" in paths
    assert "denomark/data/prepare_waterbench.py" in paths
    assert "docs/DENOMARK_ENCODER.md" in paths
    assert not any(path.startswith("data/") for path in paths)


def test_encoder_release_name_and_training_recipe_are_stable():
    config = json.loads((ROOT / "configs/paper_experiments.json").read_text())
    encoder = config["semantic_encoder_training"]
    assert encoder["checkpoint_name"] == "DenoMark-Encoder"
    assert encoder["release_asset"] == "DenoMark-Encoder.tar.gz"
    assert encoder["source_rows"] == 8000
    assert encoder["reported_valid_pairs"] == 7993
    assert encoder["source_preparer"] == "denomark/encoder/prepare.py"
    assert encoder["paraphrase_generator"] == "denomark/encoder/paraphrase.py"
    assert encoder["pair_finalizer"] == "denomark/encoder/merge.py"


def test_sentence_attack_split_and_compression_target_are_stable():
    text = "Dr. Smith measured 10 samples. The result was stable!"
    sentences = split_sentences(text)
    assert sentences == ["Dr. Smith measured 10 samples.", "The result was stable!"]
    prompt, temperature, bounds = sentence_attack_spec("compress_60_70", sentences[0])
    assert temperature == 0.7
    assert bounds == (3, 3)
    assert "about 3 to 3 words" in prompt


def test_runtime_token_count_accepts_all_runner_schemas():
    assert token_count({"token_len": 151}) == 151
    assert token_count({"dgmark_token_len": 152}) == 152
    assert token_count({"hash_distribution_token_len": 153}) == 153
    assert token_count({"generated_token_ids": [1, 2, 3]}) == 3


def test_metric_collector_normalizes_flat_and_nested_tpr_schemas():
    flat = normalize_metrics(
        {
            "n_pos": 300,
            "n_neg": 10000,
            "roc_tpr_at_0_5pct": 0.5,
            "roc_tpr_at_1pct": 0.6,
            "roc_tpr_at_5pct": 0.8,
            "auc": 0.9,
        }
    )
    nested = normalize_metrics(
        {
            "n_positive": 300,
            "n_negative": 10000,
            "tpr": {"0.005": 0.5, "0.01": 0.6, "0.05": 0.8},
            "auc": 0.9,
        }
    )
    assert flat == nested
