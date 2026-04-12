# Error Shift Analysis

## Run Summary
| run_name | model_size | train_size | supervision_style | n | acc | parse_fail_rate | wrong_n |
|---|---|---|---|---:|---:|---:|---:|
| 4b_clean_full_answer_only | 4B | full | answer_only | 1147 | 0.215344 | 0.130776 | 900 |
| 4b_clean_full_formula_rationale | 4B | full | formula_rationale | 1147 | 0.215344 | 0.102877 | 900 |
| 8b_clean_1000_answer_only | 8B | 1000 | answer_only | 1147 | 0.122058 | 0.091543 | 1007 |
| 8b_clean_250_answer_only | 8B | 250 | answer_only | 1147 | 0.103749 | 0.042720 | 1028 |
| 8b_clean_full_answer_only | 8B | full | answer_only | 1147 | 0.126417 | 0.060157 | 1002 |
| 8b_clean_full_formula_rationale | 8B | full | formula_rationale | 1147 | 0.090671 | 0.213601 | 1043 |

## Style Comparison: answer_only vs formula_rationale
- close-score threshold: 0.0100 (if |delta| below threshold, use error composition as primary signal)
| model_size | train_size | acc_answer_only | acc_formula_rationale | delta_formula_minus_answer | close_score_gap |
|---|---|---:|---:|---:|---|
| 4B | full | 0.215344 | 0.215344 | 0.000000 | True |
| 8B | full | 0.126417 | 0.090671 | -0.035745 | False |

### Error Composition Delta (formula_rationale - answer_only)
- 4B / full: {"numeric mismatch": -0.003333333333333334, "parse fail": -0.03555555555555555, "percent scaling": 0.025555555555555554, "unit confusion": 0.013333333333333308}
- 8B / full: {"numeric mismatch": 0.24355316213211162, "parse fail": 0.16462951374336657, "percent scaling": -0.07592867955364435, "unit confusion": -0.33225399632183383}
