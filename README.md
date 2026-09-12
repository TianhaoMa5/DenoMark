# 🇩🇰 DenMark

Semantic watermarking for diffusion language models. Code for the method,
baselines, attacks, and experiments evaluated in the paper.

Supported backbones: **LLaDA-8B, LLaDA1.5-8B, LLaDA2.0-mini, and Dream**.

## Installation

Python 3.10+ is required. Generation requires suitable GPU resources.

```bash
git clone https://github.com/TianhaoMa5/DenMark.git
cd DenMark
python -m pip install -e ".[eval,attacks,data,plots]"
```

## Encoder and Data

Download the paper's fine-tuned **DenMark-Encoder**:

```bash
curl -fL -o DenMark-Encoder.tar.gz \
  https://github.com/TianhaoMa5/DenMark/releases/download/v0.2.0/DenMark-Encoder.tar.gz
tar -xzf DenMark-Encoder.tar.gz
```

Datasets and backbone weights are not bundled. Prepare prompt-only inputs from
locally obtained upstream datasets:

```bash
python -m denmark data prompts \
  --finance /path/to/finance.jsonl --alpaca /path/to/alpaca.jsonl \
  --longform /path/to/longform.jsonl --output-dir runs/prompts --limit 500
```

See [data construction](docs/PAPER_REPRODUCTION.md#2-paper-matrix) and
[encoder training](docs/DENMARK_ENCODER.md) for the full recipes.

## Generate

Example using LLaDA-8B and the paper's DenMark settings:

```bash
python -m denmark generate \
  --prompts_jsonl runs/prompts/finance_qa.jsonl \
  --llada_model GSAI-ML/LLaDA-8B-Instruct \
  --encoder_model /path/to/DenMark-Encoder \
  --output runs/watermarked.jsonl --retokenize_prompts \
  --gen_length 300 --block_size 25 --cand_block_size 1 \
  --num_candidates 16 --num_message_bits 2 --channels_per_step 2 \
  --rollouts_per_cand 3 --rollout_schedule linear_decay \
  --temperature 0.5 --perturb_temperature 0.6 --rollout_temperature 0.5 \
  --position_selection random --per_cand_positions \
  --dedup_candidates --shared_rollout_seeds \
  --direction_seed 42 --message_seed 0 --generator_family llada
```

LLaDA-8B and LLaDA1.5 use `llada`; LLaDA2.0-mini uses `llada2`.
For Dream options: `python -m denmark generate --generator_family dream --help`.

## Reproduce the Paper

The [reproduction guide](docs/PAPER_REPRODUCTION.md) provides generation,
filtering, calibration, detection, quality evaluation, and figure commands.
The [experiment configuration](configs/paper_experiments.json) lists the settings.

- **Baselines:** Clean, DLM-KGW, DGMark, PatternMark, UMR; Block Best-of-K,
  PMark-style and SemStamp-style for the blockwise comparison.
- **Attacks:** sentence/document GPT attacks, nonuniform length changes,
  Parrot, DIPPER, back-translation, and token-level perturbations.
- **Experiments:** detection, PPL and GPT judge, ablations, rollout diagnostics,
  and downstream tasks.

Filter positives before attack; detect attacked text without length refiltering
or reusing original token IDs. DenMark uses per-size empirical calibration and
Bonferroni correction over unit sizes 12--37, with disjoint calibration and ROC
negative pools. See the [evaluation protocol](docs/REPRODUCIBILITY.md).

## Checks

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m denmark.evaluation.audit_reproducibility
```

Tests use synthetic inputs without downloading model weights or calling APIs.

## License

[MIT](LICENSE). External models, datasets, and baseline implementations retain
their respective licenses.
