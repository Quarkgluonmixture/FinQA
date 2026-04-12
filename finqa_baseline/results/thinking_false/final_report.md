# Final Baseline Report (FinQA, Zero-Shot)

## Scope
- Dataset: FinQA test split
- Decoding: greedy (`do_sample=False`)
- Default baseline: thinking=true + `[FINAL_ANSWER]...[/FINAL_ANSWER]` + `math_verify`
- Also reported: legacy numeric evaluator for side-by-side comparison

## Baseline Table (Latest per Model/Setting)
| model | setting | split | n | acc_mathverify | acc_legacy | delta | parse_fail_mv | parse_fail_legacy | tag_open_only_rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-4B | oracle | test | 1147 | 0.196164 | 0.313862 | -0.117698 | 0.042720 | 0.011334 | 0.118570 |
| Qwen/Qwen3-4B | full | test | 1147 | 0.092415 | 0.114211 | -0.021796 | 0.031386 | 0.000000 | 0.305144 |
| Qwen/Qwen3-8B | oracle | test | 1147 | 0.155187 | 0.204010 | -0.048823 | 0.031386 | 0.000000 | 0.001744 |
| Qwen/Qwen3-8B | full | test | 1147 | 0.096774 | 0.122058 | -0.025283 | 0.031386 | 0.000000 | 0.002616 |

## Key Findings
- Best math-verify run: Qwen/Qwen3-4B / oracle @ 0.1962
- Avg parse_fail (math-verify): 0.0342
- Avg parse_fail (legacy): 0.0028
- Tag-status distribution is recorded in `summary.json` (`tag_status_counts`) for truncation diagnostics.
- `open_only` tags indicate truncated close tags; extraction fallback still provides answer text.

## Artifacts
- `results/regression_final_answer_mathverify.md`
- `results/summary.json`
- `results/error_cases.md`