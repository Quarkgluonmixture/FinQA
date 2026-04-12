# Zero-shot Baseline Discrepancy Diagnosis (2026-04-04)

## Compared Sources
- Old baseline report: `finqa_baseline/results_recovered/a656988/final_report.md`
- Old adjusted records: `finqa_baseline/results_recovered/a656988/summary_latest_adjusted.json`
- Current baseline report: `finqa_baseline/results/thinking_false/final_report.md`
- Current raw runs: `finqa_baseline/results/thinking_false/summary.json`

## Metric Snapshot (oracle/test)
| source | model | primary metric shown | value |
|---|---|---|---:|
| old recovered | Qwen/Qwen3-4B | `acc_base` | 0.165650 |
| old recovered | Qwen/Qwen3-4B | `acc_adjusted` | 0.258065 |
| current thinking_false | Qwen/Qwen3-4B | `acc_mathverify` | 0.196164 |
| current thinking_false | Qwen/Qwen3-4B | `acc_legacy` | 0.313862 |
| old recovered | Qwen/Qwen3-8B | `acc_base` | 0.076722 |
| old recovered | Qwen/Qwen3-8B | `acc_adjusted` | 0.108108 |
| current thinking_false | Qwen/Qwen3-8B | `acc_mathverify` | 0.155187 |
| current thinking_false | Qwen/Qwen3-8B | `acc_legacy` | 0.204010 |

## Root Cause (Not Apples-to-Apples)
The discrepancy comes from comparing different evaluation protocols/eras:

1. **Different primary metric definitions**
- Old report uses `acc_base/acc_adjusted` (legacy numeric pipeline + percent auto-scale table).
- Current report uses `acc_mathverify` as primary and separately reports legacy metrics.

2. **Different decoding/output policy**
- Old recovered runs were generated under older baseline settings (report states numeric-only output policy).
- Current runs enforce `final_answer_tag` format and `math_verify` evaluator.

3. **Different generation length settings**
- Old recovered entries in `summary.json` are from `max_new_tokens=32` runs.
- Current baseline uses `max_new_tokens=256`.

4. **Recovered folder is historical snapshot**
- `results_recovered/a656988` is a backfilled historical bundle, not the same run configuration used by current stage1 matrix.

## Recommendation for Stage1 Comparison
Use one consistent baseline family for all stage1 conclusions:
- Preferred: current `thinking_false` family (`math_verify`, `final_answer_tag`, `max_new_tokens=256`).
- Do not mix metrics from `results_recovered/*` with current `sft_matrix` comparisons.
