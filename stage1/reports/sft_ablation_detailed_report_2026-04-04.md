# FinQA Stage1 SFT Ablation 详细实验报告

- 生成时间: 2026-04-04
- 执行窗口: 2026-04-03 03:23:48 -> 2026-04-04 05:40:33 (总计约 26.28 小时)
- 主日志: `stage1/logs/train_eval_matrix_full.log`
- 结果目录: `finqa_baseline/results/sft_matrix/`

## 1. 执行摘要

- 实验流程已完整跑通，主流程退出码为 `0`，日志存在 `All done.`。
- 按计划完成：Debug 4 组、Full 4 组、Zero-shot 锚点、8B winner-style 的 250/1000 消融、mandatory error-shift 分析。
- 8B winner-style 判定结果：`answer_only`。
- 结论上，`answer_only` 与 `formula_rationale` 分数差距都较小（close-gap），主要信号来自 error composition。

## 2. 计划对齐与范围

- 监督分支：`answer_only` vs `formula_rationale`。
- 推理配置：`--no-enable_thinking` + `final_answer_tag` + `math_verify`。
- LoRA 兼容检查：debug 阶段执行 base vs adapter parseability 对比。
- Stage B 策略：`250 ⊂ 1000 ⊂ full`，并按 program steps 分层采样。
- fallback 要求：已强制执行 error-shift 分析。

## 3. 时间线（关键阶段）

| 阶段 | 时间 |
|---|---|
| Stage 0: | 2026-04-03 03:23:49 |
| Stage 1: | 2026-04-03 03:23:53 |
| Stage 2: | 2026-04-03 13:31:30 |
| Stage 2b: | 2026-04-03 18:54:14 |
| Stage 3: | 2026-04-04 02:08:49 |
| Stage 4: | 2026-04-04 02:08:49 |
| Stage 5: | 2026-04-04 05:40:33 |
| All done. | 2026-04-04 05:40:33 |

## 4. 数据构建与采样验收

### 4.1 Percent/自洽修正

| split | rows | formula_exec_ok_rate | answer_corrected_rate | scale_relation_counts |
|---|---:|---:|---:|---|
| train | 25185 | 0.9317 | 0.2509 | {'consistent': 16989, 'mismatch': 8196} |
| dev | 3417 | 0.9306 | 0.2543 | {'consistent': 2297, 'mismatch': 1120} |
| debug | 100 | 0.9200 | 0.1800 | {'consistent': 72, 'mismatch': 28} |

- 说明：当前统计仅出现 `consistent/mismatch`，未出现 `x100_match/div100_match` 标签。

### 4.2 子集采样（分层 + 嵌套）

- full bucket: `{'single': 15003, 'double': 7775, 'multi': 2407}`
- 250 bucket: `{'single': 149, 'double': 77, 'multi': 24}`
- 1000 bucket: `{'single': 595, 'double': 309, 'multi': 96}`
- 嵌套关系校验：`250 ⊂ 1000` 通过（ID 检查无差异）。

## 5. 评测与兼容性验收

- debug compat 检查通过次数: `4/4`（`base_parseable=4, adapter_parseable=4`）。
- 评测格式可解析：大部分 run 的 `closed/open_only` 合计接近 1147，`absent` 显著下降于 full+8B。

## 6. 结果总表（manifest 12 runs）

| run | model | train_size | style | acc_mathverify | acc_adjusted | parse_fail | percent_recovered | tag(closed/open/absent) |
|---|---|---|---|---:|---:|---:|---:|---|
| 4b_debug_answer_only | 4B | debug | answer_only | 0.169137 | 0.270270 | 0.033130 | 177 | 976/143/28 |
| 4b_debug_formula_rationale | 4B | debug | formula_rationale | 0.163906 | 0.270270 | 0.033130 | 179 | 948/172/27 |
| 8b_debug_answer_only | 8B | debug | answer_only | 0.158675 | 0.200523 | 0.031386 | 93 | 1146/1/0 |
| 8b_debug_formula_rationale | 8B | debug | formula_rationale | 0.158675 | 0.198779 | 0.031386 | 91 | 1146/1/0 |
| 4b_full_answer_only | 4B | full | answer_only | 0.132520 | 0.132520 | 0.031386 | 49 | 1087/10/50 |
| 4b_full_formula_rationale | 4B | full | formula_rationale | 0.127289 | 0.134263 | 0.033130 | 49 | 1129/3/15 |
| 8b_full_answer_only | 8B | full | answer_only | 0.178727 | 0.187446 | 0.032258 | 81 | 1140/6/1 |
| 8b_full_formula_rationale | 8B | full | formula_rationale | 0.175240 | 0.183958 | 0.031386 | 77 | 1140/7/0 |
| 4b_zeroshot | 4B | 0 | zero_shot | 0.196164 | 0.313862 | 0.042720 | 211 | 985/136/26 |
| 8b_zeroshot | 8B | 0 | zero_shot | 0.155187 | 0.204010 | 0.031386 | 96 | 1144/2/1 |
| 8b_250_answer_only | 8B | 250 | answer_only | 0.162162 | 0.186574 | 0.031386 | 79 | 1133/14/0 |
| 8b_1000_answer_only | 8B | 1000 | answer_only | 0.173496 | 0.177855 | 0.031386 | 74 | 1142/5/0 |

## 7. 关键对比与发现

### 7.1 Full 主实验：answer_only vs formula_rationale

- 4B full: answer_only=0.132520, formula_rationale=0.127289, delta=-0.005231
- 8B full: answer_only=0.178727, formula_rationale=0.175240, delta=-0.003487
- 两组均属于 close-gap（|delta| < 0.01），符合 fallback 到 error-shift 的触发条件。

### 7.2 Full SFT 相对 Zero-shot

- 4B: zeroshot=0.196164 -> full_answer_only=0.132520 (-0.063644), full_formula=0.127289 (-0.068875)
- 8B: zeroshot=0.155187 -> full_answer_only=0.178727 (+0.023540), full_formula=0.175240 (+0.020052)
- 观察：8B full 有提升；4B full 相对 zeroshot 退化。

### 7.3 8B winner-style 消融（answer_only）

- 8B zeroshot(0): 0.155187
- 8B 250: 0.162162
- 8B 1000: 0.173496
- 8B full: 0.178727
- 趋势：0 -> 250 -> 1000 -> full 总体上升，但 1000 与 full 差距不大。

### 7.4 Error-shift（mandatory）

- 4B / debug: delta=-0.005231, close_gap=True, error_delta={'numeric mismatch': 0.005524511257463671, 'parse fail': -1.3130151532890488e-05, 'percent scaling': -0.002841583627576416, 'unit confusion': -0.0026697974783543993}
- 4B / full: delta=-0.005231, close_gap=True, error_delta={'numeric mismatch': 0.011205879547588005, 'percent scaling': -0.011199855420960948, 'unit confusion': -6.024126627141805e-06}
- 8B / debug: delta=+0.000000, close_gap=True, error_delta={'numeric mismatch': 0.0}
- 8B / full: delta=-0.003487, close_gap=True, error_delta={'numeric mismatch': 0.0}

## 8. 强制验收清单

| 验收项 | 状态 | 证据 |
|---|---|---|
| Debug 4 组跑通 (4B/8B × 2 styles) | PASS | manifest + 对应 summary 存在 |
| Full 4 组跑通 (4B/8B × 2 styles) | PASS | manifest + 对应 summary 存在 |
| 8B winner-style 决策 + 250/1000 消融 | PASS | 日志 winner=answer_only + 8b_250/8b_1000 summary |
| 4B anchor (0/full) | PASS | 4b_zeroshot + 4b_full_* summary |
| no-thinking + final_answer_tag + math_verify | PASS | 各 summary 中 `enable_thinking=false` + evaluator/format 字段 |
| LoRA tokenizer 兼容检查 | PASS | `[compat] base_parseable=4, adapter_parseable=4` ×4 |
| mandatory error-shift 报告 | PASS | `error_shift_report.md/json` 已生成 |
| run 产物完整性 (checkpoint/jsonl/summary) | PASS | manifest 逐条校验缺失数=0 |

## 9. 风险与备注

- 4B full 相对 zeroshot 出现退化，建议后续优先排查 4B 配置与数据配比。
- `scale_relation` 统计未出现 `x100_match/div100_match`，与最初规格预期不一致，建议确认标签判定阈值或逻辑路径。
- 部分 run 的 `absent` 仍非 0（如 4B full、8B full answer_only），建议纳入格式回归的 hard fail 条件。

## 10. 结论

- 从本轮结果看，8B + answer_only 是当前更稳的主线。
- 两种 supervision 的总分差异普遍较小，主结论应以 error-shift 组成差异来支撑，而非仅看 acc。
- 实验流水线与必交付物已达到“可复现、可审计、可汇报”状态。

## 11. 关键产物路径

- 主日志: `stage1/logs/train_eval_matrix_full.log`
- 运行清单: `finqa_baseline/results/sft_matrix/run_manifest.jsonl`
- error-shift: `finqa_baseline/results/sft_matrix/error_shift_report.md`
- 核心结果目录: `finqa_baseline/results/sft_matrix/`
- 数据审计: `stage1/data/unified/derived/*.summary.json`
- 子集采样审计: `stage1/data/unified/derived/subsets/train_subset_summary.json`
