# DenMark-Encoder

`DenMark-Encoder` is the semantic encoder checkpoint used for the paper's
reported DenMark results. Download it from:

```text
https://github.com/TianhaoMa5/DenMark/releases/latest/download/DenMark-Encoder.tar.gz
```

The release archive SHA-256 is recorded on the GitHub Release page. After
extraction, verify every checkpoint component with:

```bash
cd DenMark-Encoder
shasum -a 256 -c SHA256SUMS
```

## Architecture and training

- Initialization: `intfloat/e5-base-v2`
- Valid training pairs in the reported run: 7,993
- Maximum source length before paraphrasing: 25 whitespace-delimited words
- Paraphraser: `tuner007/pegasus_paraphrase`
- Pegasus beams and maximum length: 10 and 60
- Encoder maximum sequence length: 64 tokens
- Pooling: attention-mask mean pooling followed by L2 normalization
- Objective: InfoNCE with in-batch negatives
- Contrastive temperature: 0.05
- Batch size: 128
- Optimizer and learning rate: AdamW, `1e-5`
- Epochs: 3
- Warmup: 5% followed by linear learning-rate decay
- Seed: 42

## Construct the training pairs

No training corpus or pair file is included in this repository. Start from a
locally supplied JSONL corpus whose rows contain a `text` field:

```bash
python -m denmark.encoder.prepare \
  --input /path/to/source.jsonl \
  --output-dir runs/encoder_pairs \
  --n 8000 --num-shards 8 --max-words 25 --seed 42
```

Generate one sentence-wise Pegasus paraphrase for every selected document in
each shard:

```bash
python -m denmark.encoder.paraphrase \
  --input runs/encoder_pairs/source_shards/shard_00.jsonl \
  --output runs/encoder_pairs/generated/shard_00.jsonl \
  --model tuner007/pegasus_paraphrase \
  --num-beams 10 --max-length 60 --seed 42
```

Run that command for all eight shards, then merge and remove empty, unchanged,
or duplicate pairs:

```bash
python -m denmark.encoder.merge \
  --inputs runs/encoder_pairs/generated/shard_*.jsonl \
  --output runs/encoder_pairs/paraphrase_pairs.jsonl
```

Train from the resulting local pair file:

```bash
python -m denmark.encoder.train \
  --model_name_or_path intfloat/e5-base-v2 \
  --train_file runs/encoder_pairs/paraphrase_pairs.jsonl \
  --output_dir runs/DenMark-Encoder \
  --max_seq_length 64 --batch_size 128 \
  --learning_rate 1e-5 --num_train_epochs 3 \
  --temperature 0.05 --warmup_ratio 0.05 --seed 42 --bf16
```

The released checkpoint is provided for exact evaluation; reconstructing the
training pairs may vary slightly across library or upstream checkpoint
versions, so record those revisions when retraining.
