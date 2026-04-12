# FinQA SFT 实验：完整时间线、交付物与报告撰写指南

**最后更新**: 2026-04-12
**Deadline**: 2026-04-17
**实验状态**: **全部完成** — 可以开始写报告
**Repository**: https://github.com/Quarkgluonmixture/FinQA
**Machine**: DGX Spark, NVIDIA GB10, 121.7GB GPU RAM, 1×GPU

---

## 〇、研究问题与协议锁定

**RQ**: 在 no-thinking + controlled answer format + math-verify 评测协议下，低数据量 task-specific SFT 能否恢复 Qwen3-8B 在 FinQA 上的异常低表现？与 Qwen3-4B 的恢复能力如何对比？

**协议锁定**:
- 推理模式: `no-thinking` (`--no-enable_thinking`)
- 答案格式: `[FINAL_ANSWER]<number>[/FINAL_ANSWER]`
- 评测器: `math-verify` (percent_auto_scale=true, atol=0.001, rtol=0.001)
- 基准: FinQA test split (n=1147)
- 模型: Qwen3-8B (主), Qwen3-4B (对照)
- 微调方法: LoRA / QLoRA SFT (via TRL)
- 分析: 数据量消融 + error shift + supervision style 对比

---

## 一、实验环境

### 硬件
- DGX Spark, NVIDIA GB10 GPU (121.7 GB VRAM), 单卡

### 软件
| 组件 | 版本 |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Transformers | 5.4.0 |
| PEFT | 0.18.1 |
| TRL | 1.0.0 |
| Datasets | 4.8.4 |
| bitsandbytes | (QLoRA 量化) |

### 虚拟环境
- 训练: `stage1/.venv/` (TRL + PEFT + bitsandbytes)
- 评测: `finqa_baseline/.venv/` (math-verify + PEFT)

### 模型
| 模型 | HuggingFace ID | 本地缓存 |
|---|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B` | `/home/jiaming/workspace/.cache/huggingface/models--Qwen--Qwen3-4B/` |
| Qwen3-8B | `Qwen/Qwen3-8B` | `/home/jiaming/workspace/.cache/huggingface/models--Qwen--Qwen3-8B/` |

---

## 二、实验时间线

### Phase 0: Baseline 建立 (3/28 – 3/31)

| 日期 | 事项 | 产物 |
|---|---|---|
| 3/28–3/31 | Qwen3-4B/8B zero-shot baseline (thinking_true + thinking_false, chat 格式) | `finqa_baseline/results/thinking_true/`<br>`finqa_baseline/results/thinking_false/` |
| 3/30 | Robust verification report | `finqa_baseline/results/robust_verification_report.md` |
| 3/31 | Proposal findings 整理 | `docs/proposal_findings_cn.md` |
| 3/31 | Sanity regression 验证 | `finqa_baseline/results/sanity/` |

**Baseline 完整数字 (chat 格式, oracle, n=1147)**:

| 模型 | thinking | acc_base | acc_adjusted | parse_fail | percent_recovered | tag: closed/open/absent |
|---|---|---|---|---|---|---|
| 4B | false | 12.99% | 31.39% | 4.27% (49) | 211 | 985 / 136 / 26 |
| 8B | false | 12.03% | 20.40% | 3.14% (36) | 96 | 1144 / 2 / 1 |

**Baseline (chat 格式, full context, n=1147)**:

| 模型 | acc_base | acc_adjusted | parse_fail |
|---|---|---|---|
| 4B | 6.28% | 11.42% | 3.14% |
| 8B | 6.71% | 12.21% | 3.14% |

**关键观察**: 8B 在 FinQA 上表现反常地低于或持平 4B，尤其 oracle 设定下 acc_adjusted 差 11pp (31.39% vs 20.40%)。

---

### Phase 1: SFT Pipeline + 第一轮实验 — 脏数据 (4/2 – 4/3)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/2 | Stage1 SFT pipeline (TRL + LoRA, config-driven) | `stage1/scripts/`, `stage1/configs/` |
| 4/2 | 三库合并数据 (FinQA 6251 + ConvFinQA 11104 + MultiHiertt 7830 = 25185) | `stage1/data/unified/` (train.jsonl 209M) |
| 4/3 | 实验矩阵设计 | `docs/experiment_matrix.md` |
| 4/3 | 修订 SFT 消融计划 | `docs/revised_sft_ablation_plan.md` |
| 4/3 | 第一轮训练+评测 (chat prompt 评测) | `stage1/outputs/sft_matrix/` (12 ckpts, 1.7G)<br>`finqa_baseline/results/sft_matrix/` (20 eval dirs) |

**第一轮完整结果 (脏数据 25185 条, chat prompt 评测, oracle)**:

| Run | acc_base | acc_adjusted | parse_fail | vs zero-shot (adj) |
|---|---|---|---|---|
| 4b_zeroshot | 12.99% | 31.39% | 4.27% | — |
| 4b_full_answer_only | 8.98% | 13.25% | 3.14% | **-18.14pp** |
| 4b_full_formula_rationale | 9.15% | 13.43% | 3.31% | **-17.96pp** |
| 8b_zeroshot | 12.03% | 20.40% | 3.14% | — |
| 8b_full_answer_only | 11.68% | 18.74% | 3.23% | -1.66pp |
| 8b_full_formula_rationale | 11.68% | 18.40% | 3.14% | -2.00pp |
| 8b_250_answer_only | 11.77% | 18.66% | 3.14% | -1.74pp |
| 8b_1000_answer_only | 11.33% | 17.79% | 3.14% | -2.61pp |

**Phase 1 发现**: 4B 灾难性退化 (-18pp)。8B 略退化。

---

### Phase 2: 问题诊断 + 数据清洗 (4/4)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/4 上午 | 发现数据是三库混合 (非纯 FinQA), 32% mismatch (8196 条) | `stage1/reports/sft_ablation_detailed_report_2026-04-04.md` |
| 4/4 上午 | Zero-shot 数字差异诊断 | `stage1/reports/zero_shot_discrepancy_diagnosis_2026-04-04.md` |
| 4/4 中午 | 生成 FinQA-clean strict 子集 | `stage1/data/finqa_clean/` + `clean_summary.json` |

**数据清洗**: `source=finqa` + `formula_exec_ok=true` + `scale_relation=consistent`

| | 原始 | 过滤后 |
|---|---|---|
| train | 25185 (finqa 6251 + convfinqa 11104 + multihiertt 7830) | **3277** |
| dev | 3417 | 475 |

Train 分布: single 1962 / double 873 / multi 442。子集: 250 ⊂ 1000 ⊂ full。

---

### Phase 3: 第二轮实验 — Clean Strict + Chat Eval (4/4)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/4 下午 | 6 个 SFT run 训练 + 评测 (clean strict, chat prompt 评测) | `stage1/outputs/sft_clean_strict/` (6 ckpts)<br>`finqa_baseline/results/sft_clean_strict/` (6 dirs) |

**训练超参**:

| 参数 | 4B runs | 8B runs |
|---|---|---|
| base model | Qwen3-4B | Qwen3-8B |
| use_qlora | false (full LoRA) | true (QLoRA) |
| lora_r / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| target_modules | [q_proj, v_proj] | [q_proj, v_proj] |
| learning_rate | 5e-05 | 2e-04 |
| num_train_epochs | 2 | 2 |
| **实际 max_steps** | **100** (仅 12% epoch) | **100** (仅 12% epoch) |
| batch_size | 4 | 4 |
| max_seq_length | 512 | 512 |

**重要**: 所有 6 个 run 都只训练了 100 steps (~12% of 1 epoch)，3277 条数据大部分未被使用。

**Phase 3 结果 (clean 数据, chat prompt 评测)**:

| Run | acc_base | acc_adjusted | parse_fail | vs chat ZS (adj) |
|---|---|---|---|---|
| 4b_clean_full_answer_only | 9.42% | 16.13% | 3.14% | **-15.26pp** |
| 4b_clean_full_formula_rationale | 9.68% | 17.09% | 3.14% | **-14.30pp** |
| 8b_clean_full_answer_only | 10.99% | 17.52% | 3.14% | -2.88pp |
| 8b_clean_full_formula_rationale | 11.07% | 16.83% | 3.14% | -3.57pp |
| 8b_clean_250_answer_only | 10.55% | 15.96% | 3.14% | -4.44pp |
| 8b_clean_1000_answer_only | 10.64% | 15.69% | 3.14% | -4.71pp |

全部退化 → 问题不在数据。

---

### Phase 4: 根因诊断 (4/4 – 4/5)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/4–4/5 | 三项诊断 | `stage1/reports/diagnostic_prompt_adapter_8b_clean_full_answer_only_2026-04-05.md` |

**训练 prompt 格式** (纯文本拼接):
```
Instruction:
Solve the financial numerical reasoning problem. Return exactly one
tagged final answer as [FINAL_ANSWER]...[/FINAL_ANSWER] in the output.

Input:
Context:
{table + text}

Question:
{question}

Output:
{answer}
```

**评测 prompt 格式** (chat system+user):
```
System: You are a financial QA assistant...
User: Context: ... Question: ... Return exactly one final answer as [FINAL_ANSWER]<number>[/FINAL_ANSWER].
```

**诊断结论**: 训练/评测 prompt 协议完全错配是退化主因。Adapter 已正确加载 (8/10 输出不同于 base)。`<think>` 不是问题 (两边都没用)。

---

### Phase 5: Prompt 对齐重评测 (4/5 – 4/11)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/5 | eval_finqa.py 新增 `--prompt_protocol stage1_train_text` | `finqa_baseline/eval_finqa.py` |
| 4/5–4/11 | 8 个 run 评测完成 | `finqa_baseline/results/sft_clean_strict_prompt_aligned_eval/` (8 dirs) |

**Phase 5 完整结果 (clean 数据, prompt 对齐评测)**:

| Run | acc_base | acc_adjusted | parse_fail | tag: closed/open/absent | vs text ZS (base) |
|---|---|---|---|---|---|
| **4b_zeroshot (text)** | **24.93%** | **32.78%** | 8.98% (103) | 673/65/409 | — |
| 4b_full_answer_only | 31.21% | 33.22% | 13.08% (150) | 1032/31/84 | **+6.28pp** |
| 4b_full_formula_rationale | 32.43% | 34.61% | 10.29% (118) | 1017/34/96 | **+7.50pp** |
| **8b_zeroshot (text)** | **7.15%** | **7.32%** | **64.43%** (739) | 852/6/289 | — |
| 8b_full_answer_only | 10.03% | 13.16% | 6.02% (69) | 1120/14/13 | +2.88pp |
| 8b_full_formula_rationale | 7.59% | 9.94% | 21.36% (245) | 1121/17/9 | +0.44pp |
| 8b_250_answer_only | 9.15% | 11.25% | 4.27% (49) | 1122/17/8 | +2.00pp |
| 8b_1000_answer_only | 9.94% | 12.47% | 9.15% (105) | 1133/11/3 | +2.79pp |

**Error Shift (prompt 对齐版)**:
- 4B: answer_only acc=21.53% vs formula_rationale acc=21.53% (完全持平)
  - error composition 差异: parse_fail -3.6pp, percent_scaling +2.6pp
- 8B: answer_only acc=12.64% vs formula_rationale acc=9.07% (**-3.57pp**)
  - formula_rationale parse_fail 高很多 (21.4% vs 6.0%)
- 产物: `finqa_baseline/results/sft_clean_strict_prompt_aligned_eval/error_shift_report.{md,json}`

**8B 失败模式**: 8B base 在纯文本 prompt 下输出 `[FINAL_ANSWER]...[/FINAL_ANSWER]`（字面省略号），然后在 tag 之后开始 CoT 推理。739/1147 (64.4%) 是这种空 tag 模式。

---

### Phase 6: 8B Chat 格式重评测 (4/11 – 4/12)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/11–4/12 | 4 个 8B SFT checkpoint 用 chat_default 重评 | `finqa_baseline/results/sft_clean_strict_chat_eval/` (4 dirs) |

**目的**: 验证 8B SFT 是否学到推理能力但被 text prompt 掩盖。

**Phase 6 结果 (clean 数据, chat 格式评测)**:

| Run | acc_base | acc_adjusted | parse_fail | vs chat ZS (adj=20.40%) |
|---|---|---|---|---|
| 8b_full_answer_only | 10.99% | **17.52%** | 3.14% | **-2.88pp** |
| 8b_full_formula_rationale | 11.07% | **16.83%** | 3.14% | -3.57pp |
| 8b_250_answer_only | 10.55% | **15.95%** | 3.14% | -4.45pp |
| 8b_1000_answer_only | 10.64% | **15.69%** | 3.14% | -4.71pp |

**结论**: 所有 8B SFT run 在 chat eval 下也低于 zero-shot (20.40%)。**SFT 未学到有效推理能力**，不是 prompt 的问题。

Error shift: `finqa_baseline/results/sft_clean_strict_chat_eval/error_shift_report.{md,json}`

---

### Phase 7: 8B 增加训练步数验证 (4/12)

| 日期 | 事项 | 产物 |
|---|---|---|
| 4/12 | 8B answer_only 重训 max_steps=-1 (1640 steps, 2 full epochs) + 双协议评测 | `stage1/outputs/sft_clean_strict/8b_clean_full_answer_only_fullsteps/`<br>`finqa_baseline/results/sft_clean_strict_fullsteps_eval/` |

**目的**: 验证 100 steps 是否欠训练。

训练: 1640 steps, 2 epochs, train_loss 1.528→1.190, 训练时长 4.44h

**Phase 7 结果**:

| 评测协议 | acc_base | acc_adjusted | parse_fail | tag: closed/open/absent |
|---|---|---|---|---|
| chat_default | **12.73%** | **18.74%** | 3.14% (36) | 1146/1/0 |
| stage1_train_text | **12.12%** | **14.39%** | 3.40% (39) | 946/2/199 |

**与 100 steps 对比 (chat eval)**:

| 条件 | acc_base | acc_adjusted | train_loss | steps |
|---|---|---|---|---|
| 8B chat zero-shot | 12.03% | **20.40%** | — | 0 |
| 8B 100 steps | 10.99% | 17.52% | 1.528 | 100 |
| **8B 1640 steps (full)** | **12.73%** | **18.74%** | **1.190** | **1640** |

**结论**: 增加训练步数有帮助 (acc_adjusted +1.22pp)，但仍未超过 zero-shot baseline (18.74% < 20.40%)。单纯增加步数不足以让 8B SFT 超越 zero-shot。

---

## 三、最终结果总表

### 所有可比条件汇总 (oracle, n=1147)

#### 4B

| 条件 | Prompt | acc_base | acc_adjusted | parse_fail | 说明 |
|---|---|---|---|---|---|
| Zero-shot | chat | 12.99% | 31.39% | 4.27% | Phase 0 baseline |
| Zero-shot | text | 24.93% | 32.78% | 8.98% | Phase 5 |
| SFT 100steps answer_only | text | **31.21%** | 33.22% | 13.08% | Phase 5, **+6.28pp base** |
| SFT 100steps formula_rationale | text | **32.43%** | **34.61%** | 10.29% | Phase 5, **+7.50pp base** |

#### 8B

| 条件 | Prompt | acc_base | acc_adjusted | parse_fail | 说明 |
|---|---|---|---|---|---|
| Zero-shot | chat | 12.03% | 20.40% | 3.14% | Phase 0 baseline |
| Zero-shot | text | 7.15% | 7.32% | 64.43% | Phase 5, 灾难 |
| SFT 100steps answer_only | chat | 10.99% | 17.52% | 3.14% | Phase 6, -2.88pp |
| SFT 100steps formula_rationale | chat | 11.07% | 16.83% | 3.14% | Phase 6, -3.57pp |
| SFT 100steps 250 answer_only | chat | 10.55% | 15.95% | 3.14% | Phase 6, -4.45pp |
| SFT 100steps 1000 answer_only | chat | 10.64% | 15.69% | 3.14% | Phase 6, -4.71pp |
| **SFT 1640steps answer_only** | **chat** | **12.73%** | **18.74%** | 3.14% | **Phase 7, -1.66pp** |
| SFT 1640steps answer_only | text | 12.12% | 14.39% | 3.40% | Phase 7 |

---

## 四、完整交付物索引

### 代码与脚本

| 路径 | 说明 |
|---|---|
| `finqa_baseline/eval_finqa.py` | **评测主脚本** (支持 `--prompt_protocol chat_default\|stage1_train_text`, `--adapter_path`, `--evaluator math_verify`) |
| `finqa_baseline/utils/prompting.py` | Prompt 构建逻辑 (chat 模式 + train-text 模式) |
| `finqa_baseline/utils/__init__.py` | 工具函数 |
| `stage1/train_sft.py` | **SFT 训练主脚本** (TRL SFTTrainer + LoRA/QLoRA) |
| `stage1/scripts/run_train.sh` | 单次训练 wrapper |
| `stage1/scripts/build_finqa_clean_splits.py` | FinQA-clean 数据过滤+子集生成 |
| `stage1/scripts/build_formula_rationale_targets.py` | Formula rationale target 构建 |
| `stage1/scripts/build_stratified_subsets.py` | 分层嵌套子集构建 |
| `stage1/scripts/analyze_error_shift.py` | Error shift 分析 |
| `stage1/scripts/diagnose_prompt_eval_adapter.py` | Prompt/adapter 诊断 |
| `stage1/scripts/run_train_eval_matrix.sh` | Phase 1 全矩阵脚本 |
| `stage1/scripts/run_train_eval_matrix_clean_strict.sh` | Phase 3 全矩阵脚本 |
| `stage1/scripts/run_eval_matrix_prompt_aligned.sh` | Phase 5 重评测脚本 |
| `stage1/scripts/run_eval_8b_chat_and_errorshift.sh` | Phase 6 8B chat eval + error shift |
| `stage1/scripts/run_8b_fullsteps_train_eval.sh` | Phase 7 fullsteps 训练+双评测 |
| `stage1/scripts/run_train_eval_background.sh` | 通用后台 watchdog+ntfy wrapper |
| `stage1/scripts/prompting.py` | 训练侧 prompt 生成 |

### 配置文件

| 路径 | 说明 |
|---|---|
| `stage1/configs/generated_clean_strict/*.yaml` | Phase 3 训练配置 (6 个 + 1 个 fullsteps) |
| `stage1/configs/generated/*.yaml` | Phase 1 训练配置 (10 个) |
| `stage1/configs/{debug,small,full}.yaml` | 基础配置模板 |

### 数据

| 路径 | 说明 | 大小 |
|---|---|---|
| `stage1/data/unified/` | 三库合并原始数据 | 474M |
| `stage1/data/unified/derived/` | 归一化 + 子集 | ~260M |
| `stage1/data/finqa_clean/` | **Strict clean 子集 (3277 train, 475 dev)** | 25M |
| `stage1/data/finqa_clean/clean_summary.json` | 过滤统计+抽检结果 | 6K |
| `stage1/data/finqa_clean_consistent_or_corrected/` | Relaxed 版本 (5933 train, 未使用) | 40M |

### Checkpoints

| 路径 | 说明 | 大小 |
|---|---|---|
| `stage1/outputs/sft_matrix/` | Phase 1 脏数据 (12 模型) | 1.7G |
| `stage1/outputs/sft_clean_strict/` | **Phase 3+7 clean strict (7 模型)** | ~1.5G |

每个 checkpoint 含: `config_snapshot.yaml`, `trainer_state.json`, `checkpoint-last/` (adapter_model.safetensors + optimizer.pt)

### 评测结果

| 路径 | Phase | Prompt | 内容 | 状态 |
|---|---|---|---|---|
| `finqa_baseline/results/thinking_false/` | 0 | chat | 4B/8B zero-shot baseline | 完成 |
| `finqa_baseline/results/thinking_true/` | 0 | chat | thinking=true baseline | 完成 |
| `finqa_baseline/results/sft_matrix/` | 1 | chat | 脏数据 SFT eval (20 dirs) | 完成 |
| `finqa_baseline/results/sft_clean_strict/` | 3 | chat | clean SFT eval (6 dirs) + error_shift | 完成 |
| `finqa_baseline/results/sft_clean_strict_prompt_aligned_eval/` | 5 | text | **clean SFT + prompt 对齐 (8 dirs) + error_shift** | 完成 |
| `finqa_baseline/results/sft_clean_strict_chat_eval/` | 6 | chat | **8B SFT chat 重评 (4 dirs) + error_shift** | 完成 |
| `finqa_baseline/results/sft_clean_strict_fullsteps_eval/` | 7 | both | **8B fullsteps 双协议 (2 dirs)** | 完成 |

每个 eval 目录含: `summary.json` (完整指标), `finqa_*.jsonl` (逐条结果含 raw_output), `error_cases.md`

### 报告与文档

| 路径 | 说明 |
|---|---|
| **`docs/experiment_timeline_and_deliverables.md`** | **本文件 — 实验全貌** |
| `docs/experiment_matrix.md` | 实验矩阵设计 (v2.0) |
| `docs/revised_sft_ablation_plan.md` | 修订消融计划 |
| `docs/proposal_findings_cn.md` | Proposal findings |
| `docs/实验过程.md` | 实验过程对话记录 |
| `stage1/reports/sft_ablation_detailed_report_2026-04-04.md` | Phase 1 脏数据报告 |
| `stage1/reports/zero_shot_discrepancy_diagnosis_2026-04-04.md` | Zero-shot 差异诊断 |
| `stage1/reports/sft_ablation_strict_consistent_report_2026-04-04.md` | Phase 3 报告 |
| `stage1/reports/diagnostic_prompt_adapter_8b_clean_full_answer_only_2026-04-05.md` | Prompt/adapter 诊断 |

### 日志

| 路径 | 说明 |
|---|---|
| `stage1/logs/train_eval_matrix_full.log` | Phase 1 完整日志 |
| `stage1/logs/train_eval_clean_strict.log` | Phase 3 完整日志 |
| `stage1/logs/eval_prompt_aligned.log` | Phase 5 主日志 |
| `stage1/logs/eval_8b_chat_and_errorshift.log` | Phase 6 日志 |
| `stage1/logs/8b_fullsteps_train_eval.log` | Phase 7 日志 |

---

## 五、Findings 总结 (报告素材)

### Finding 1: 数据质量对小模型影响极大
- 脏数据 (25185 混合, 32% mismatch) 导致 4B acc_adjusted -18pp
- 清洗后 (3277 条 strict consistent) 配合 prompt 对齐，4B SFT 从退化变为 +7.5pp 提升

### Finding 2: 训练-评测 Prompt 一致性是 SFT 成功的前提
- 训练用 Instruction/Input/Output 纯文本，评测用 chat system+user → 协议错配
- 对齐后 4B SFT 从 -15pp 退化变为 +7.5pp 提升
- 方法论 lesson: SFT 实验必须保证端到端 prompt 一致

### Finding 3: 模型规模与 Prompt 格式敏感度的交互
- 4B 对纯文本 prompt 友好: text ZS acc_base 24.93% > chat ZS 12.99%
- 8B 对纯文本 prompt 灾难: text ZS acc_base 7.15% (64% parse fail) << chat ZS 12.03%
- 8B 在预训练/对齐阶段更深度学习了 chat template 结构

### Finding 4: SFT 对 4B 有效
- 4B + clean data + prompt aligned: acc_base 24.93% → 32.43% (+7.5pp)
- formula_rationale 略优于 answer_only (+1.2pp base, +1.4pp adjusted)
- SFT 提升了格式遵从 (absent tag 409→96)

### Finding 5: 8B SFT 失败 — 负面发现
- 8B SFT 在所有评测条件下均低于 zero-shot baseline
- Chat eval: best = 18.74% (fullsteps) vs ZS 20.40% (-1.66pp)
- Text eval: best = 13.16% (100steps) vs text ZS 7.32% (+5.84pp，但绝对值仍很低)
- 8B SFT 学会了格式 (parse fail 64%→6%)，但推理能力未恢复
- 增加训练步数 (100→1640) 有帮助 (+1.22pp) 但不足以逆转

### Finding 6: 8B 数据量消融趋势 (chat eval)
- 0 (ZS): 20.40% → 250: 15.95% → 1000: 15.69% → full (100步): 17.52% → full (1640步): 18.74%
- 更多数据和更多步数都帮助 8B 接近但始终未超过 ZS
- 反直觉: SFT 在 8B 上是有害的 (在当前配置下)

### 已知 Confounds / Limitations
1. **max_steps 不一致**: 大部分 run 100 steps (12% epoch), 仅 Phase 7 的 8B 跑满
2. **LoRA vs QLoRA**: 4B 用 full LoRA, 8B 用 QLoRA — 量化可能影响 8B
3. **训练格式次优**: 纯文本 Instruction/Input 不是 Qwen3 的最优 SFT 格式
4. **lr 差异**: 4B=5e-5, 8B=2e-4 — 8B lr 可能过高
5. **单一 test split**: 无交叉验证
6. **4B 未做 fullsteps 实验**: 4B 也只训了 100 steps，fullsteps 可能更好

---

## 六、RQ 回答

> **RQ**: 低数据量 task-specific SFT 能否恢复 Qwen3-8B 在 FinQA 上的异常低表现？

**答案**: 在当前实验配置下 (QLoRA, text-format training, 3277 clean samples)，**不能**。

- **4B (Qwen3-4B)**: SFT 有效。Clean data + prompt 对齐后 acc_base 从 24.93% 提升到 32.43% (+7.5pp)，达到所有条件下的最高分。formula_rationale 略优于 answer_only。
- **8B (Qwen3-8B)**: SFT 无效甚至有害。即使跑满 2 个 epoch (1640 steps)，chat eval acc_adjusted 仍从 20.40% 降到 18.74%。8B 更依赖 chat template 结构，纯文本训练格式可能是核心瓶颈。

**核心洞察**: 更大的模型不等于更容易通过 SFT 恢复。8B 的强 chat template 依赖性使得在纯文本格式下的 SFT 适得其反。这提示 SFT 的成功不仅取决于数据质量和数量，还高度依赖于训练格式与模型预训练分布的匹配程度。

---

## 七、如何复现实验

### 环境搭建
```bash
git clone https://github.com/Quarkgluonmixture/FinQA.git
cd FinQA

# 训练环境
cd stage1 && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # TRL, PEFT, bitsandbytes, etc.

# 评测环境
cd ../finqa_baseline && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  # math-verify, PEFT, etc.
```

### 训练
```bash
cd stage1
bash scripts/run_train.sh configs/generated_clean_strict/8b_clean_full_answer_only.yaml
```

### 评测
```bash
cd finqa_baseline
.venv/bin/python eval_finqa.py \
  --model_name Qwen/Qwen3-8B \
  --adapter_path ../stage1/outputs/sft_clean_strict/8b_clean_full_answer_only/checkpoint-last \
  --setting oracle --split test \
  --evaluator math_verify --answer_format final_answer_tag --final_answer_tag FINAL_ANSWER \
  --prompt_protocol chat_default \
  --no-enable_thinking --max_new_tokens 256 --seed 42 --num_samples -1 \
  --results_dir results/my_eval/
```

### Error Shift 分析
```bash
cd stage1
python scripts/analyze_error_shift.py \
  --manifest_jsonl <manifest.jsonl> \
  --output_md <output.md> --output_json <output.json>
```

### 后台运行 (watchdog + ntfy)
```bash
bash stage1/scripts/run_8b_fullsteps_background.sh start   # 启动
bash stage1/scripts/run_8b_fullsteps_background.sh status  # 查状态
bash stage1/scripts/run_8b_fullsteps_background.sh stop    # 停止
# ntfy 通知: https://ntfy.sh/<topic>
```
