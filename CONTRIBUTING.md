# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
ruff check denomark scripts tests
```

GPU models and experiment datasets are not required for the unit-test suite.
Please add a focused unit test for changes to scoring, calibration, scheduling,
token alignment, or generator adapters.

## Experiment changes

- Record every generation and detection parameter in the output metadata.
- Keep calibration samples disjoint from held-out ROC negatives.
- Re-tokenize attacked text instead of reusing pre-attack token IDs.
- Filter positives before attack; do not length-filter attacked outputs again.
- Put machine-specific paths in environment variables or command-line flags.
- Do not commit credentials, model weights, scheduler logs, or result bundles.

## Pull requests

Keep changes narrow and include the exact verification commands. Changes that
alter a paper-facing metric must document the old and new statistical protocol.
