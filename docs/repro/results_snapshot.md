# Result Snapshot (Report-Oriented)

This file records compact metrics that map to the main report tables.

## Baseline Reference

- Task: FinQA test
- Evaluator: math-verify
- Primary protocol: oracle + no-thinking + final-answer tag

## Key Outcome Summary

- Best zero-shot baseline in the verified matrix is below full task saturation.
- Stage-1 SFT outcomes are sensitive to prompt protocol alignment.
- The repository includes scripts for:
  - baseline matrix reporting
  - clean strict SFT matrix
  - prompt-aligned re-evaluation
  - 8B full-steps retrain and dual-protocol evaluation

For exact per-run values, read each run's `summary.json` through `stage1/scripts/summary_utils.py`.
