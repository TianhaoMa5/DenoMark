# Reproducing every paper experiment

This index maps every experiment reported in the DenoMark paper to a public
configuration and executable entry point. The machine-readable source of
truth is `configs/paper_experiments.json`; DenoMark's method-only defaults are
in `configs/denomark_paper.json`.

## 1. Environment and external implementations

Install the repository and all optional experiment dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[eval,attacks,data,plots]"
```

The paper compares against official methods whose repositories and generator
weights are not redistributed here. Clone them separately and pass their roots
through command-line arguments. The only released model artifact is
`DenoMark-Encoder`.

- PatternMark: required by `--patternmark-repo`.
- UMR: required by `--umr_root`, together with its generated bitmap.
- MarkLLM/Parrot: required by `--markllm_root`.
- DIPPER: downloaded from its Hugging Face checkpoint by the attack runner.

No script contains a user-specific filesystem path. On a cluster, run model
generation, attacks, detection, and PPL inside scheduler jobs; the commands
below are the payloads to place in those jobs.

## 2. Paper matrix

Main robustness experiments cover the Cartesian product below.

| Axis | Values |
| --- | --- |
| Backbones | LLaDA-8B, LLaDA1.5-8B, LLaDA2.0-mini, Dream-v0-Instruct-7B |
| Datasets | Finance-QA, AlpacaFarm, LongForm-QA |
| Methods | DenoMark, DLM-KGW, DGMark, PatternMark, UMR |
| Main semantic attacks | sentence rewrite, sentence compression, sentence expansion, document rewrite |
| Metrics | ROC-interpolated TPR@0.5/1/5% FPR and rank AUC |

Every method/backbone/dataset starts from up to 300 generated positives. Apply
`denomark/data/filter.py` before any attack. Never filter attacked text
again. Every method for one backbone is evaluated against that backbone's same
10,000 held-out C4 negatives.

The configuration covers 19 paper experiment groups. Run
`denomark/evaluation/audit_reproducibility.py` before launching a sweep; it
verifies the complete paper axes, public entry points, method defaults, local
imports, and release safety rules, including the requirement that no dataset
payload is committed.

The repository does not redistribute benchmark rows. Construct prompt-only
inputs from locally obtained WaterBench-style JSONL sources:

```bash
python -m denomark.data.prepare_waterbench \
  --finance /path/to/finance_source.jsonl \
  --alpaca /path/to/alpaca_source.jsonl \
  --longform /path/to/longform_source.jsonl \
  --output-dir runs/prompts --limit 500
```

Use `WATERBENCH=runs/prompts` in the commands below. The construction code
preserves upstream `raw_prompt` values when available, otherwise applies the
paper templates, and strips answers and source metadata from its outputs.

### Semantic encoder

The exact paper checkpoint is distributed as the GitHub Release asset
`DenoMark-Encoder.tar.gz`; see `docs/DENOMARK_ENCODER.md`. No training examples
or paraphrase pairs are redistributed. To reconstruct the data from a local
JSONL source corpus containing a `text` field, select 8,000 rows with:

```bash
python -m denomark.encoder.prepare \
  --input /path/to/source.jsonl --output-dir runs/encoder_pairs \
  --n 8000 --num-shards 8 --max-words 25 --seed 42
```

Run `denomark.encoder.paraphrase` on every emitted shard with Pegasus
beam size 10 and maximum length 60, then merge them with
`denomark.encoder.merge`. The reported run retained 7,993 valid
pairs after dropping empty, unchanged, and duplicate pairs. Train with
`denomark.encoder.train` using the settings in
`docs/DENOMARK_ENCODER.md`.

## 3. Calibration and held-out negatives

Create 40,000 tokenizer-matched C4 RealNewsLike crops for each backbone. The
script samples target lengths approximately uniformly over 150--300 and writes
disjoint 30,000/10,000 splits with an overlap audit:

```bash
python denomark/data/negatives.py \
  --output-dir runs/c4_negative_pools
```

Outputs are:

```text
runs/c4_negative_pools/<backbone>/calibration_30000.jsonl
runs/c4_negative_pools/<backbone>/heldout_10000.jsonl
runs/c4_negative_pools/audit.json
```

Only DenoMark uses the 30,000 calibration rows. All five methods use the same
backbone-specific 10,000 rows for final ROC metrics. The two sets must have
zero `source_id` overlap.

## 4. Generation and filtering

### DenoMark

Use the command in the README for each backbone and dataset. Change only
`--llada_model`, `--generator_family`, input, and output. The paper settings are
`m=25`, `K=16`, two channels, random per-candidate positions, candidate
temperature 0.6, and a linear 5-to-1 rollout schedule averaging three
rollouts. Dream uses its dedicated runner:

```bash
python denomark/core/generate.py --generator_family dream --help
```

### DLM-KGW

```bash
python denomark/baselines/dlm_kgw/generate.py \
  --waterbench_dir "$WATERBENCH" \
  --output_dir runs/kgw/<backbone> \
  --datasets finance_qa alpacafarm longform_qa \
  --n_per_dataset 300 \
  --model_name_or_path "$MODEL" \
  --generator_family "$FAMILY" \
  --delta "$DELTA" --gamma 0.25 \
  --convolution_kernel -1 --topk 50 --n_iter 1 \
  --temperature 0.5 --remasking random \
  --max_rep4 0.2 --max_retries 10
```

Use delta 4 for LLaDA-8B, LLaDA1.5-8B, and Dream, and 4.25 for
LLaDA2.0-mini. Dream dispatches to the origin sampler with HashDistribution
applied to aligned logits before sampling; it uses random position transfers,
not the LLaDA block loop. Its `block_size` argument does not control decoding.

### DGMark

```bash
python denomark/baselines/dgmark/generate.py --help
```

For LLaDA-family checkpoints use multinomial top-10 decoding, beam size 10,
and low-confidence remasking. For Dream use top-32, beam size 32, 25-token
blocks and low-confidence remasking (`--dgmark_decoding dream_dgmark`). These
settings are selected automatically for Dream. This branch uses DGMark parity
candidate selection, not the separate origin parity-logit-bias adapter.
The Dream blockwise
adapter is intentional: the non-blockwise path repeatedly produced unusably
short continuations in the paper setup.

### PatternMark

```bash
python denomark/baselines/patternmark/generate.py \
  --patternmark_repo "$PATTERNMARK_REPO" \
  --waterbench_dir "$WATERBENCH" \
  --output_dir runs/patternmark/<backbone> \
  --datasets finance_qa alpacafarm longform_qa \
  --n_per_dataset 300 \
  --model_name_or_path "$MODEL" \
  --generator_family "$FAMILY" \
  --delta 4 --temperature 0.5 \
  --max_attempts 10 --max_rep4 0.2
```

The runner supports `llada`, `llada2`, and Dream's native `origin` decoder.
It uses two colors, length-four patterns `0101` and `1010`, and delta 4.

### UMR

```bash
python denomark/baselines/umr/generate.py --help
```

Use message `1001`, ratio 0.5, key 42, temperature 0.5, and low-confidence
remasking. LLaDA-family delta is 10 for Finance/LongForm and 13 for Alpaca;
Dream uses delta 10 for all three datasets.

For attacked positives pass `--no_filter_positive` to the UMR evaluator. The
positive cohort was already filtered before attack and must not be shortened
again after rewrite, compression, or expansion. The held-out negative pool is
still filtered/validated according to the fixed C4 construction.

### Shared pre-attack filter

Run this on every raw positive file and retain the resulting JSONL as the
immutable attack input:

```bash
python denomark/data/filter.py \
  --input runs/raw.jsonl \
  --output runs/filtered.jsonl \
  --audit runs/filtered.audit.json \
  --tokenizer "$MODEL" \
  --min-tokens 150 --max-rep4 0.2 --max-rows 300
```

The filter re-tokenizes text, removes empty, short, repetitive, and duplicate
outputs, and records every removal count.

## 5. Main GPT attacks

Set `OPENROUTER_API_KEY` in the environment. Credentials are never accepted as
CLI arguments or written to output files.

Sentence-level rewrite, 60--70% compression, and expansion:

```bash
python denomark/attacks/gpt_sentence_level.py --help
```

Use `openai/gpt-4o-mini`, temperature 0.7, and the three attack names recorded
in `configs/paper_experiments.json`. Each sentence is called independently and
cached by attack name, model, and sentence hash.

Document-level rewrite, compression, and expansion:

```bash
python denomark/attacks/gpt_document_level.py --help
```

The main table uses document rewrite in addition to the three sentence-level
columns. Document compression and expansion are reported in the extended
document-level table.

## 6. Additional semantic attacks

| Experiment | Entry point | Sweep |
| --- | --- | --- |
| Parrot | `denomark/attacks/parrot.py` | candidate prefix 1, 4, 7, 10 |
| DIPPER | `denomark/attacks/dipper.py` | lexical diversity 20, 40, 60, 80; order 0 |
| Back-translation | `denomark/attacks/backtranslation.py` | sentence and document; English-Chinese-English |
| Alternating/local runs | `denomark/attacks/mixed_length.py` | alternating and random compress/expand runs |
| Variable local compression | `denomark/attacks/variable_compression.py` | contiguous 50--60, 60--70, 70--80% bands |

Parrot and DIPPER figures use Finance and Alpaca on LLaDA-8B and Dream.
Back-translation uses Alpaca and LongForm on the same two backbones. The
nonuniform local-length diagnostic uses Dream Finance and LongForm.

## 7. Token-level attacks

The paper uses only deletion, context-aware substitution, and adjacent word
swapping at ratios 0.1, 0.2, 0.3, 0.4, and 0.5:

```bash
python -m denomark.attacks.token_level \
  --input runs/filtered.jsonl \
  --output runs/token_attacks.jsonl \
  --attacks deletion context_aware_substitution adjacent_swap \
  --ratios 0.1 0.2 0.3 0.4 0.5 \
  --mlm-model bert-base-uncased
```

The token-level figure averages those five strengths on Finance and LongForm
for LLaDA-8B and Dream.

## 8. Detection

### DenoMark Method A

```bash
python denomark/evaluation/detect.py \
  --positive_jsonl runs/positive_or_attack.jsonl \
  --calibration_jsonl runs/c4_negative_pools/<backbone>/calibration_30000.jsonl \
  --negative_jsonl runs/c4_negative_pools/<backbone>/heldout_10000.jsonl \
  --model "$MODEL" --encoder "$ENCODER" \
  --num_message_bits 2 \
  --detectors calibrated_scan --subsets pos_all_neg\>=150 \
  --retokenize_positive --retokenize_calibration --retokenize_negative \
  --scan_min 12 --scan_max 37 \
  --output_json runs/detection.json --output_txt runs/detection.txt
```

The detector computes an empirical right-tail p-value independently at each
unit size, takes the minimum, applies Bonferroni over sizes 12--37, and ranks by
negative log corrected p-value. Calibration and ROC negatives are disjoint.

### Baseline detectors

| Method | Entry point | Paper ranking score |
| --- | --- | --- |
| DLM-KGW | `denomark/baselines/dlm_kgw/detect.py` | green-token z-score |
| DGMark | `denomark/baselines/dgmark/detect.py` | mean squared z over all windows of size 8 |
| PatternMark | `denomark/baselines/patternmark/detect.py` | negative official pattern-count p-value |
| UMR | `denomark/baselines/umr/detect.py` | official z-score on first 300 retokenized tokens |

All attacked files must be re-tokenized. Use each backbone's held-out 10k pool
and the `roc_*` or `paper_metrics` fields emitted by the evaluators.
`denomark.baselines.dlm_kgw.detect` applies no positive-side length filter.
Use this same entry point for original and attacked text.

Collect heterogeneous detector outputs into one auditable CSV using a JSONL
manifest:

```json
{"base":"llada8b","dataset":"finance_qa","method":"denomark","condition":"rewrite","result_path":"detection/ours_rewrite.json"}
{"base":"llada8b","dataset":"finance_qa","method":"dlm_kgw","condition":"rewrite","result_path":"detection/kgw_rewrite.json"}
```

```bash
python denomark/evaluation/collect_metrics.py \
  --manifest runs/result_manifest.jsonl \
  --output runs/paper_metrics.csv
```

For nonstandard/nested outputs, add a dot-separated `metric_path` to the
manifest row. Semi-AR examples use
`methods.pmark.calibrated_scan.summary` or
`methods.semstamp.calibrated_scan.summary`.

## 9. Quality evaluation

Completion-only Qwen2.5-32B PPL:

```bash
python -m denomark.evaluation.ppl --help
```

Join outputs to same-backbone, same-dataset clean references and report
`mean(log PPL_watermarked) - mean(log PPL_clean)`. Do not use the log of mean
PPL. Run the four-axis GPT judge with:

```bash
python denomark/evaluation/judge.py --help
```

The judge reports style, consistency, accuracy, and ethics using
GPT-4o-mini. Keep raw JSON responses and cache records.

After scoring method and clean files, compute the exact quality-table delta
with a manifest and `denomark/evaluation/compare_ppl.py`. Each manifest row names
the method and clean evaluator JSON files plus their keys under `results`:

```bash
python denomark/evaluation/compare_ppl.py \
  --manifest runs/ppl_manifest.jsonl \
  --output runs/ppl_comparison.csv
```

## 10. Ablations and diagnostics

All generation ablations use the DenoMark generator and vary only the named
flag. Values are in `configs/paper_experiments.json`.

| Paper experiment | Vary |
| --- | --- |
| Candidate temperature | `--perturb_temperature` = 0.3, 0.5, 0.6, 0.75, 0.9 |
| Candidate count | `--num_candidates` = 8, 16, 24, 32 |
| Channel count | fixed two, or tied 1, 2, 3, 4 for K = 8, 16, 24, 32 |
| Rollout count | `--rollouts_per_cand` = 1, 2, 3, 5, 8 |
| Semantic-unit size | `--block_size` = 15, 25, 40, 50, 100 |
| Scan range | 23--27, 21--29, 17--33, 12--37, 10--40 |
| Maximum length | 100, 150, 200, 250, 300 |
| Positions per step | `--cand_block_size` = 1, 2, 4, 8 |
| Position rule | random versus low-confidence |
| Encoder | contrastive checkpoint versus untouched E5-base-v2 |

The scan-range axis is exactly fixed 25, 23--27, 21--29, 17--33, 12--37,
and 10--40. The maximum-length axis is 100, 150, 200, 250, and 300 tokens.
The multi-position table compares `r=2,4,8` to the main `r=1` implementation;
its reported rows use 50 retained Finance-QA positives per backbone.

For every generation ablation, use its matching unit size as the center of the
scan-range experiment; do not silently reuse a fixed 25-token detector when the
generation unit size changed.

Trajectory/continued-policy analysis:

```text
denomark/experiments/trajectory/llada_cumulative_run.py
denomark/experiments/trajectory/llada_reverse_hybrid_run.py
denomark/experiments/trajectory/dream_reverse_hybrid_run.py
denomark/experiments/trajectory/llada_cumulative_summary.py
denomark/experiments/trajectory/llada_reverse_hybrid_summary.py
denomark/experiments/trajectory/dream_reverse_hybrid_summary.py
```

The paper uses 30 prompts per backbone for the reverse-hybrid trajectory plot.

## 11. Semi-autoregressive baselines

The appendix compares DenoMark with Block Best-of-K, PMark, and SemStamp
semi-autoregressive adaptations. Their public implementations are grouped in
`denomark/baselines/`.

```text
denomark/baselines/block_best_of_k/generate.py
denomark/baselines/pmark/generate.py
denomark/baselines/semstamp/generate.py
denomark/baselines/semantic_detect.py
```

All three use 25-token units, random position selection, two semantic
bits/channels, and temperature 0.9 in the reported comparison. PMark therefore
uses `--num_channels 2`; SemStamp uses `--lsh_dim 2`, `--accept_rate 0.25`, and
the default released hash key. Their scanned semantic scores are calibrated
with grouped five-fold cross-fitting on the held-out negative pool, so each
negative is scored using calibration rows from the other folds.

Block Best-of-K is detected with the DenoMark semantic detector using
`--num_message_bits 2`; PMark and SemStamp use their method-native scores via
`evaluate_pmark_semstamp_calibrated_scan.py`. All reported Semi-AR TPR columns
use the evaluator's `roc_tpr_at_0_5pct`, `roc_tpr_at_1pct`, and
`roc_tpr_at_5pct` fields, while strict discrete thresholds are retained in the
JSON for auditability.

## 12. Downstream tasks and runtime

The downstream table evaluates MMLU, HellaSwag, ARC-Challenge, and GSM8K on
LLaDA-8B and LLaDA1.5-8B:

```bash
python denomark/experiments/downstream/prepare.py \
  --output runs/downstream/official_manifest.json
python denomark/experiments/downstream/run.py --help
python denomark/experiments/downstream/aggregate.py --help
```

The preparation command uses the complete official evaluation splits by
default. `--limit-per-benchmark` is only for an explicit smoke test. Accuracy
is higher-is-better. Runtime is measured from generator timing metadata as wall
seconds divided by visible completion tokens; retain per-sample timing records
before averaging by method.

## 13. Paper figures

`denomark/evaluation/plot_figures.py` renders every result-driven figure without
repository-specific result paths. Each subcommand accepts tidy CSV and writes
both vector PDF and a 300-dpi PNG preview:

```bash
python denomark/evaluation/plot_figures.py sensitivity --input sensitivity.csv --output sensitivity.pdf
python denomark/evaluation/plot_figures.py scan-range --input scan.csv --output scan.pdf
python denomark/evaluation/plot_figures.py max-length --input max_length.csv --output max_length.pdf
python denomark/evaluation/plot_figures.py temperature-position --input temperature_position.csv --output temperature_position.pdf
python denomark/evaluation/plot_figures.py open-source-attacks --input open_source.csv --output open_source.pdf
python denomark/evaluation/plot_figures.py token-level-attacks --input token_attacks.csv --output token_attacks.pdf
python denomark/evaluation/plot_figures.py trajectory --input trajectory.csv --output trajectory.pdf
```

The tidy-CSV schemas are:

- `sensitivity`: `sweep,x,dataset,tpr_at_1pct,auc,delta_mean_logppl`
- `scan-range`: `scan_min,scan_max,condition,tpr_at_0_5pct,tpr_at_1pct,tpr_at_5pct,auc`
- `max-length`: `base,dataset,max_length,tpr_at_0_5pct,tpr_at_1pct,tpr_at_5pct,auc`
- `temperature-position`: `base,dataset,position_selection,temperature,tpr_at_1pct,auc`
- `open-source-attacks`: `family,setting,base,dataset,method,tpr_at_0_5pct,tpr_at_1pct`
- `token-level-attacks`: `attack,ratio,base,dataset,method,tpr_at_0_5pct,tpr_at_1pct,tpr_at_5pct,auc`
- `trajectory`: `base,prompt_id,step,cumulative_advantage`

Every subcommand validates its required columns before plotting.
Open-source attack plots average the LLaDA-8B/Dream and Finance/Alpaca cells;
token-level plots average five ratios, LLaDA-8B/Dream, and Finance/LongForm,
matching the paper captions. Trajectory input contains `base`, `prompt_id`,
`step`, and `cumulative_advantage`; the renderer draws individual trajectories
and a deterministic 2,000-resample 95% bootstrap interval.

## 14. Release checks

Before publishing a result or source bundle:

```bash
python -m compileall -q denomark scripts tests
python -m pytest
ruff check denomark scripts tests
python denomark/evaluation/audit_reproducibility.py
python denomark/evaluation/build_release.py --output /tmp/denomark-release
```

The release builder is allow-listed, rejects credentials, personal/cluster
absolute paths, model weights, archives, and generated artifacts, and writes a
SHA-256 manifest for every included file.
