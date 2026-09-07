# DenoMark

Semantic watermarking for diffusion language models. This repository contains
the method, baselines, and experiments from the paper, without bundled datasets.

## Install

```bash
python -m pip install -e ".[eval,attacks,data,plots]"
python -m denomark --help
```

Python 3.10+ is required. Generator checkpoints and external baseline repositories
are supplied through command-line paths; they are not bundled here.

## Encoder

Download **DenoMark-Encoder**, the paper's fine-tuned E5-base-v2 checkpoint:

```bash
curl -L -o DenoMark-Encoder.tar.gz \
  https://github.com/TianhaoMa5/DenoMark/releases/download/v0.2.0/DenoMark-Encoder.tar.gz
tar -xzf DenoMark-Encoder.tar.gz
```

The archive includes the weights, tokenizer, model card, and checksums.
[Encoder construction and training](docs/DENOMARK_ENCODER.md) describes how to
construct paraphrase pairs and train it. No training examples are distributed.

## Generate And Detect

First construct prompt inputs from locally obtained upstream data:

```bash
python -m denomark data prompts \
  --finance /path/to/finance.jsonl --alpaca /path/to/alpaca.jsonl \
  --longform /path/to/longform.jsonl --output-dir runs/prompts --limit 500
```

Generate watermarked text with a LLaDA-family checkpoint:

```bash
python -m denomark generate \
  --prompts_jsonl runs/prompts/finance_qa.jsonl \
  --llada_model GSAI-ML/LLaDA-8B-Instruct \
  --encoder_model /path/to/DenoMark-Encoder \
  --output runs/watermarked.jsonl --retokenize_prompts \
  --gen_length 300 --block_size 25 --cand_block_size 1 \
  --num_candidates 16 --num_message_bits 2 --channels_per_step 2 \
  --rollouts_per_cand 3 --rollout_schedule linear_decay \
  --temperature 0.5 --perturb_temperature 0.6 --rollout_temperature 0.5 \
  --position_selection random --per_cand_positions \
  --dedup_candidates --shared_rollout_seeds \
  --direction_seed 42 --message_seed 0 --generator_family llada
```

Use `--generator_family llada2` for LLaDA2.0-mini, or `dream` for Dream:
`python -m denomark generate --generator_family dream --help`.

```bash
python -m denomark data filter --help
python -m denomark data negatives --help
python -m denomark detect --help
```

The paper detector uses per-size empirical calibration plus Bonferroni over
unit sizes 12--37. Calibration and ROC evaluation pools are disjoint. See the
[reproduction guide](docs/PAPER_REPRODUCTION.md) for complete filtering,
negative-pool, and detection commands.

## Baselines

Clean only generates unwatermarked text; it has no watermark detector.
Other methods have generation and detection entry points:

| Method | Directory | Command |
| --- | --- | --- |
| Clean | `baselines/clean` | `baseline clean generate` (select Dream with `--generator_family dream`) |
| DLM-KGW | `baselines/dlm_kgw` | `baseline dlm_kgw generate` / `detect` |
| DGMark | `baselines/dgmark` | `baseline dgmark generate` / `detect` |
| PatternMark | `baselines/patternmark` | `baseline patternmark generate` / `detect` |
| UMR | `baselines/umr` | `baseline umr generate` / `detect` |
| Block Best-of-K | `baselines/block_best_of_k` | `baseline block_best_of_k generate` / `detect` |
| PMark-style | `baselines/pmark` | `baseline pmark generate` |
| SemStamp-style | `baselines/semstamp` | `baseline semstamp generate` |

Commands are prefixed with `python -m denomark`; append `--help` for options.
PMark-style and SemStamp-style share `baseline semantic-detect`.

## Attacks

| Entry point | Purpose |
| --- | --- |
| `attacks/gpt_sentence_level.py` | GPT rewriting, compression, and expansion, one sentence per request |
| `attacks/gpt_document_level.py` | GPT rewriting, compression, and expansion, one document per request |
| `attacks/parrot.py` | Parrot with 1, 4, 7, or 10 candidates and bigram selection |

Run these with `python -m denomark attack sentence`, `attack document`, or
`attack parrot`, respectively. Append `--help` for arguments.
Pegasus is used only to construct encoder training pairs, not as an attack.

## Layout

```text
denomark/
  core/          DenoMark generation, backbone adapters, scoring, calibration
  baselines/     Clean and the paper baselines
  attacks/       GPT, Parrot, DIPPER, translation, token-level attacks
  data/          Data construction and filtering code only
  encoder/       Encoder data construction and training
  evaluation/    Detection metrics, PPL, judge, plots
  experiments/   Paper downstream, trajectory, and block-candidate experiments
```

`core` has five implementation files: `generate.py` (unified generation entry),
`model.py` (LLaDA-family and Dream model adapters),
`selectors.py` (candidate selection), `scoring.py` (semantic scores), and
`calibration.py` (scan calibration). It contains no training or plotting code.

All paper experiment commands are in [the reproduction guide](docs/PAPER_REPRODUCTION.md),
with a machine-readable index in [configs/paper_experiments.json](configs/paper_experiments.json).

## Tests

`tests/` contains synthetic regression checks, not datasets or model weights.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m denomark.evaluation.audit_reproducibility
```

## License

MIT. See [LICENSE](LICENSE).
