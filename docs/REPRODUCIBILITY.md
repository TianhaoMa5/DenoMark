# Reproducibility protocol

This document records the paper-facing protocol. It is intentionally separate
from historical experiment scripts, which may preserve older settings for
auditability.

## Positive samples

Generate up to 300 watermarked outputs for every method, backbone, and dataset.
Before applying an attack, retain outputs that satisfy all of the following:

- at least 150 tokens under the matching generator tokenizer;
- word-level 4-gram repetition ratio at most 0.2;
- non-empty and non-degenerate text;
- unique output text.

Apply every attack to this fixed retained positive set. Do not remove an
attacked output merely because the attack changes its token length.

## Calibration and ROC negatives

For each backbone, prepare 40,000 unique C4 RealNewsLike passages using that
backbone's tokenizer. Token lengths should be approximately uniform over the
integer range 150 through 300.

- 30,000 passages form DenoMark's calibration pool.
- 10,000 disjoint passages form the held-out empirical ROC-negative pool.
- The 10,000 held-out negatives are shared by all methods for that backbone.
- Calibration and held-out source IDs must have zero overlap.

The calibration pool defines the per-size empirical p-values. It is never used
to choose or evaluate a point on the final ROC curve.

## Detection metrics

Use the detector ranking statistic recorded for each method. For DenoMark this
is `-log(p_scan)`. Report rank AUC with half credit for ties and linearly
interpolated TPR at the requested exact false-positive rates.

The main paper uses FPR targets 0.5%, 1%, and 5%. Some diagnostics also include
0.1%; diagnostic columns must not silently replace the paper protocol.

## Attacked text

Delete cached generation token IDs from attack outputs or explicitly force
re-tokenization. Detection must use the attacked string, not the pre-attack
tokens. Preserve stable source IDs so every attacked output can be joined back
to the corresponding unattacked positive.

## Quality

Perplexity is completion-only. Report

```text
mean(log PPL_watermarked) - mean(log PPL_clean)
```

using a clean reference from the same backbone and dataset. Keep raw per-sample
values and source IDs so paired and unpaired summaries can be audited.

## Required provenance

Every released result should include:

- input paths or immutable dataset identifiers;
- model and tokenizer identifiers or revisions;
- encoder identifier or checksum;
- complete generation and watermark configuration;
- direction and message key identifiers (not secret deployment keys);
- filter counts and rejection reasons;
- calibration and held-out pool sizes and source-ID overlap count;
- attack model, prompt version, temperature, and cache key definition;
- code commit and package versions.

