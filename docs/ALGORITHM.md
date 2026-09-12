# DenMark algorithm

## Notation

- `m`: semantic-unit size in generator-tokenizer tokens; 25 in the paper.
- `K_t`: local candidates at denoising step `t`; 16 in the paper.
- `C`: keyed semantic channels per unit; 2 in the paper.
- `r_t`: token positions changed by one candidate; 1 in the paper.
- `R_t`: one-step rollouts used to estimate a candidate's unit semantics.
- `E_eta`: L2-normalized semantic encoder.
- `theta[b, j]`, `sign[b, j]`: key-derived direction and target sign for unit
  `b`, channel `j`.

The complete paper configuration is machine-readable in
`configs/denmark_paper.json`.

## Generation

The continuation is partitioned into fixed token regions of size `m`. At an
update of active unit `b`:

1. Sample `K_t` candidate local updates. Each candidate independently chooses
   `r_t` unresolved positions in the unit and samples replacement tokens from
   the generator logits at candidate temperature 0.6.
2. For each unique candidate state, complete the unresolved positions of the
   active unit `R_t` times using a candidate-conditioned model forward pass.
   Rollout tokens are used only for scoring and are discarded.
3. Decode each completed unit and compute its normalized embedding with
   `E_eta`.
4. Average the rollout embeddings for each candidate.
5. Score the candidate by its mean signed projection across the `C` channels:

   ```text
   W[t,b,k] = mean_j sign[b,j] * <mean_r E_eta(rollout[t,b,k,r]), theta[b,j]>
   ```

6. Commit only the local token update from the candidate with maximum score.
7. Continue until the unit is complete, then continue the model's native
   denoising schedule.

The paper uses a linear rollout schedule with target average `R=3`. If `q`
positions remain in a unit of length `m`, the implementation computes

```text
R_t = round(1 + 2 * (R - 1) * (q - 1) / (m - 1))
```

and clips the result to at least one. For `m=25`, this decreases from five
rollouts near the beginning of the unit to one near the end, averaging three
over the 25 single-position updates.

## Detection

The detector always tokenizes the evaluated text again with the matching
generator tokenizer. For a candidate unit size `m'`, it partitions the first
`gen_length` tokens into consecutive units and computes

```text
S_m'(y) = mean over non-empty units b and channels j of
          sign[b,j] * <E_eta(unit[b]), theta[b,j]>
```

For every `m'` in `G = {12, ..., 37}`, a fixed clean calibration pool `D_cal`
defines a right-tail empirical p-value:

```text
p_m'(y) = (1 + sum_{y0 in D_cal} 1[S_m'(y0) >= S_m'(y)])
          / (|D_cal| + 1)
```

The scanned p-value and ranking score are

```text
p_scan(y) = min(1, |G| * min_m' p_m'(y))
S_rob(y) = -log(p_scan(y))
```

This is per-size empirical calibration followed by a Bonferroni correction. It
is not a max-z statistic calibrated once after scanning.

Paper metrics use a second, disjoint clean pool to construct the empirical ROC.
AUC is rank based, and TPR at a target FPR is obtained by linear interpolation
on that ROC. The calibration pool is never used as the ROC-negative pool.

## Invariants

- Generation and detection must use the same encoder, tokenizer, channel count,
  direction seed, message seed, and maximum scored length.
- Attacked text must be re-tokenized; pre-attack token IDs are invalid.
- The scan range is fixed before evaluation.
- Calibration and held-out ROC negatives must be disjoint.
- Empty final units are ignored, not converted to zero-score evidence.
