# FinQA SFT Ablation Report (Strict Consistent-Only, 2026-04-04)

## 1. 执行摘要
本次实验按 `strict consistent-only (3277)` 策略完成了 6 个目标 run（`4B/8B × answer_only/formula_rationale + 8B 250/1000 answer_only`），评测协议统一为 `oracle + no-thinking + final_answer_tag + math_verify`。

整体结论：
- 6/6 run 均产出完整评测文件（`summary.json`、`finqa_*.jsonl`、`error_cases.md`）。
- 在当前这组结果下，SFT 相对 zero-shot baseline 未带来提升，4B/8B 的 `accuracy_adjusted` 均低于对应 zero-shot。
- `8B full` 上 `answer_only` 略优于 `formula_rationale`（差距很小）。
- `error shift` 已完成且可用。

## 2. 实验设置
- 数据策略：`source=finqa` + `formula_exec_ok=true` + `scale_relation=consistent`
- 训练集规模：`3277`
- 开发集规模：`475`
- 子集：`250 ⊂ 1000 ⊂ full`
- 推理设置：`--no-enable_thinking` + `FINAL_ANSWER` 标签 + `math_verify`
- 运行时间窗口（日志）：`2026-04-04 12:47:25` 到 `2026-04-04 21:41:10`

数据审计来源：
- `stage1/data/finqa_clean/clean_summary.json`

## 3. Run 清单
- `4b_clean_full_answer_only`
- `4b_clean_full_formula_rationale`
- `8b_clean_full_answer_only`
- `8b_clean_full_formula_rationale`
- `8b_clean_250_answer_only`
- `8b_clean_1000_answer_only`

## 4. 主结果（oracle/test）
| run | model | train_size | acc_mathverify | acc_adjusted | acc_base | parse_fail_rate |
|---|---|---|---:|---:|---:|---:|
| 4b_clean_full_answer_only | 4B | full | 13.25% | 16.13% | 9.42% | 3.14% |
| 4b_clean_full_formula_rationale | 4B | full | 14.04% | 17.09% | 9.68% | 3.14% |
| 8b_clean_full_answer_only | 8B | full | 16.83% | 17.52% | 10.99% | 3.14% |
| 8b_clean_full_formula_rationale | 8B | full | 16.74% | 16.83% | 11.07% | 3.14% |
| 8b_clean_250_answer_only | 8B | 250 | 16.74% | 15.95% | 10.55% | 3.14% |
| 8b_clean_1000_answer_only | 8B | 1000 | 16.04% | 15.69% | 10.64% | 3.14% |

## 5. 相对 Zero-shot 对比（当前协议基线）
使用当前协议 zero-shot 基线（来自 `results/thinking_false`）：
- 4B: `acc_base=19.62%`, `acc_adjusted=31.39%`
- 8B: `acc_base=15.52%`, `acc_adjusted=20.40%`

| run | model | acc_adjusted | delta vs zero-shot adjusted |
|---|---|---:|---:|
| 4b_clean_full_answer_only | 4B | 16.13% | -15.26 pp |
| 4b_clean_full_formula_rationale | 4B | 17.09% | -14.30 pp |
| 8b_clean_full_answer_only | 8B | 17.52% | -2.88 pp |
| 8b_clean_full_formula_rationale | 8B | 16.83% | -3.57 pp |
| 8b_clean_250_answer_only | 8B | 15.95% | -4.45 pp |
| 8b_clean_1000_answer_only | 8B | 15.69% | -4.71 pp |

## 6. Error-shift 结果
来自自动报告 `error_shift_report.md`：
- 4B full: `formula_rationale - answer_only = +0.007847`（0.78pp，属于 close-gap）
- 8B full: `formula_rationale - answer_only = -0.000872`（-0.09pp，close-gap）
- 4B full error composition 变化：`numeric mismatch` 降低，`percent scaling` 上升（幅度近似对冲）
- 8B full error composition 变化：几乎无差异

解释：在总分接近时，error composition 的变化比总分更有区分价值，当前结果支持这一点。

## 7. 执行偏差与风险说明
本轮存在一个重要执行偏差，需要在结论中明确：

- 训练步数并未完全统一。
- `4b_clean_full_answer_only` 的 `config_snapshot.yaml` 为 `max_steps=-1`。
- 其余 5 个 run 的 `config_snapshot.yaml` 为 `max_steps=100`。

这会影响不同 run 间可比性，特别是 4B 两个 supervision style 的直接比较（步数预算不一致）。

另外，主日志末尾出现：
- `run_train_eval_matrix_clean_strict.sh: line 430: unexpected EOF while looking for matching '"'`

该错误出现在 `All done.` 之后；产物已落盘，但脚本本身建议修复后再复用。

## 8. 产物索引
- 实验根目录：`finqa_baseline/results/sft_clean_strict`
- Manifest：`finqa_baseline/results/sft_clean_strict/run_manifest.jsonl`
- Error-shift：`finqa_baseline/results/sft_clean_strict/error_shift_report.md`
- 主日志：`stage1/logs/train_eval_clean_strict.log`
- 训练输出：`stage1/outputs/sft_clean_strict/*`

各 run 目录（均含 `summary.json` / `finqa_*.jsonl` / `error_cases.md`）：
- `finqa_baseline/results/sft_clean_strict/4b_clean_full_answer_only`
- `finqa_baseline/results/sft_clean_strict/4b_clean_full_formula_rationale`
- `finqa_baseline/results/sft_clean_strict/8b_clean_full_answer_only`
- `finqa_baseline/results/sft_clean_strict/8b_clean_full_formula_rationale`
- `finqa_baseline/results/sft_clean_strict/8b_clean_250_answer_only`
- `finqa_baseline/results/sft_clean_strict/8b_clean_1000_answer_only`

## 9. 建议的下一步（最小重跑）
为保证结论可发表，建议至少补一轮“统一步数预算”的对照：
- 方案A：全部 run 统一 `max_steps=-1`（按 epoch 跑满）
- 方案B：全部 run 统一 `max_steps=100`

优先级建议：先补 `4b_full_answer_only` 与 `4b_full_formula_rationale` 的对齐重跑（同一步数），再决定是否补 8B。
