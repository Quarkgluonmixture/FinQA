# Error Shift Analysis

## Run Summary
| run_name | model_size | train_size | supervision_style | n | acc | parse_fail_rate | wrong_n |
|---|---|---|---|---:|---:|---:|---:|
| 8b_clean_1000_answer_only | 8B | 1000 | answer_only | 1147 | 0.160418 | 0.031386 | 963 |
| 8b_clean_250_answer_only | 8B | 250 | answer_only | 1147 | 0.167393 | 0.031386 | 955 |
| 8b_clean_full_answer_only | 8B | full | answer_only | 1147 | 0.168265 | 0.031386 | 954 |
| 8b_clean_full_formula_rationale | 8B | full | formula_rationale | 1147 | 0.167393 | 0.031386 | 955 |

## Style Comparison: answer_only vs formula_rationale
- close-score threshold: 0.0100 (if |delta| below threshold, use error composition as primary signal)
| model_size | train_size | acc_answer_only | acc_formula_rationale | delta_formula_minus_answer | close_score_gap |
|---|---|---:|---:|---:|---|
| 8B | full | 0.168265 | 0.167393 | -0.000872 | True |

### Error Composition Delta (formula_rationale - answer_only)
- 8B / full: {"numeric mismatch": 0.0}
