# Proposal 结果发现（中文详细版，基于已跑实验）

## 1. 已运行实验清单（矩阵与配置）

本项目当前已完成的可复用 baseline 结果主要来自 `finqa_baseline/results/` 下的汇总文件，核心运行矩阵如下：

- 数据集：FinQA test split（`n=1147`）
- 评估器：`math_verify`（同时保留 legacy 指标用于对照）
- 模型：`Qwen/Qwen3-4B`、`Qwen/Qwen3-8B`
- 设置：`oracle`、`full`
- thinking 维度：`true`、`false`
- 已完成预算：`max_new_tokens=256`（当前已完成结果中仅这一档）
- 已完成格式：`final_answer_tag`（`[FINAL_ANSWER]...[/FINAL_ANSWER]`）

已完成组合（8 个唯一配置）：

| thinking | model | setting | max_new_tokens | answer_format | n |
|---|---|---|---:|---|---:|
| true | Qwen/Qwen3-4B | oracle | 256 | final_answer_tag | 1147 |
| true | Qwen/Qwen3-4B | full | 256 | final_answer_tag | 1147 |
| true | Qwen/Qwen3-8B | oracle | 256 | final_answer_tag | 1147 |
| true | Qwen/Qwen3-8B | full | 256 | final_answer_tag | 1147 |
| false | Qwen/Qwen3-4B | oracle | 256 | final_answer_tag | 1147 |
| false | Qwen/Qwen3-4B | full | 256 | final_answer_tag | 1147 |
| false | Qwen/Qwen3-8B | oracle | 256 | final_answer_tag | 1147 |
| false | Qwen/Qwen3-8B | full | 256 | final_answer_tag | 1147 |

来源：
- `finqa_baseline/results/robust_verification_summary.json`（`rows`）
- `finqa_baseline/results/thinking_false/summary.json`、`finqa_baseline/results/thinking_true/summary.json`

---

## 2. 关键结果表（含核心数字与来源）

| 主题 | 关键数值 | 来源（文件 + 字段/依据） | 含义 |
|---|---:|---|---|
| 最佳 math-verify 结果 | `0.1961639`（19.62%） | `robust_verification_summary.json` -> `best_run.accuracy_mathverify` | 最佳结果仍低于 20%，任务远未饱和。 |
| 最佳配置 | `Qwen3-4B + thinking=false + oracle` | `robust_verification_summary.json` -> `best_run` | 当前最强组合不是 8B。 |
| no-thinking oracle 下 4B 相对 8B 优势 | `+0.040976`（约 +4.10 个百分点） | `robust_verification_summary.json` -> `fourb_gt_eightb.pairs`（`thinking=false, setting=oracle`） | 在该子场景中 4B 显著高于 8B。 |
| 4B>8B 是否全局成立 | `False` | `robust_verification_summary.json` -> `fourb_gt_eightb.all_pairs_hold` | 4B 优势是“条件成立”，不是普遍结论。 |
| thinking=true + 256 截断饱和率（8B-oracle） | `1034/1147 = 90.15%` | `results/thinking_true/finqa_Qwen_Qwen3-8B_oracle_test.jsonl` 逐条 token 长度统计（`len==256`） | 多数样本撞到长度上限，结果稳定性差。 |
| thinking=true + 256 标签缺失（8B-oracle） | `tag_status absent=1104` | 同上 jsonl（每条 `tag_status`） | 大量输出未形成可控最终答案标签。 |
| 2048 运行成本（8B-oracle，日志观测） | `5/1147 -> ETA 169:51:58`、`7/1147 -> ETA 198:03:43`、`8/1147 -> ETA 164:59:30`、`10/1147 -> ETA 101:52:40` | `logs/thinking_true_Qwen_Qwen3-8B_oracle_test.log` 的 tqdm 进度行 | 观测到百小时级，早期多次超过 170 小时估计。 |
| evaluator 差异（最佳 run） | `delta_mathverify_minus_legacy=-0.117698`（约 -11.77 个百分点） | `robust_verification_summary.json` -> `best_run.delta_mathverify_minus_legacy` | legacy 指标明显高估性能，math-verify 更严格。 |

---

## 3. 主要发现（可直接写入 proposal）

1. **任务远未饱和**：当前最优 math-verify 仅 19.62%，说明 FinQA 数值推理仍有很大提升空间。  
2. **4B 并非全局劣于 8B**：在 no-thinking + oracle 上，4B 比 8B 高约 4.10 个百分点；但在其他子设置中并不总是领先，因此应避免“4B 全面优于 8B”的过度结论。  
3. **thinking=true + 256 在当前 pipeline 下不稳定**：8B-oracle 出现 90.15% 长度饱和与大量 tag 缺失（absent=1104），使得评测噪声和提取偏差显著增加。  
4. **2048 预算的工程代价很高**：日志中出现多次 100~198 小时级 ETA，说明“直接把 thinking 预算拉高”在现阶段不是高性价比路径。  
5. **评估器选择会改变结论强度**：math-verify 相比 legacy 更保守（最佳 run 约低 11.77 个百分点），应以 math-verify 为主报告口径。

---

## 4. 对 Stage 0 的启示（no-thinking-first）

1. Stage 0 采用 **no-thinking-first** 是合理的工程决策：先在可控成本下完成预算/格式对照，避免被长思维链路拖垮。  
2. 后续 SFT 默认配置应优先从 no-thinking 路径选取（尤其面向 8B 目标模型时）。  
3. thinking=true 更适合放到后续小规模敏感性实验中（可选扩展），而不是当前主线必做项。  
4. Stage 0 报告中应同时呈现“效果 + 成本”，即 accuracy/parse_fail 与 sec-per-sample、ETA 证据并列。

---

## 5. 已知局限与风险

1. 当前完成结果在预算维度只覆盖 `256`，尚未完成完整预算敏感性矩阵。  
2. 当前完成结果在格式维度只覆盖 `final_answer_tag`，尚未完成 `plain_numeric` 对照。  
3. thinking=true 的高预算日志是在线估计（tqdm ETA），虽然可用于工程风险判断，但仍应在报告中注明其“运行中估计”属性。  
4. 4B 与 8B 的相对优劣受 setting/mode 影响明显，结论必须写成“分条件比较”，避免简单排名叙事。

---

## 6. 可直接粘贴到 proposal 的段落（Findings from Preliminary Runs）

**Findings from Preliminary Runs (Chinese Draft)**  
在已完成的 FinQA baseline 矩阵中（2 模型 × 2 setting × thinking true/false，均为 max_new_tokens=256），最优 math-verify 配置为 Qwen3-4B + thinking=false + oracle，accuracy 为 0.1962，表明任务仍远未饱和。我们观察到“4B>8B”并非全局成立，而是取决于 setting 与 thinking mode；例如在 no-thinking + oracle 条件下，4B 比 8B 高约 4.10 个百分点，但在其他子设置中并不稳定。与此同时，thinking=true + 256 在 8B-oracle 上出现明显截断饱和（1034/1147 样本顶到 256 token）与标签缺失（tag_status absent=1104），使得结果稳定性较差。我们还在 2048 预算运行日志中观察到百小时级 ETA（早期多次超过 170 小时估计），说明高预算 thinking 路线当前工程成本过高。基于以上证据，Stage 0 采用 no-thinking-first 作为主线更具可行性，后续 SFT 默认配置应优先从 no-thinking 路径中选择，并将 thinking=true 作为可选扩展实验。

---

## 附：数字复核抽查（按 Test Plan）

抽查项 1：`19.62%`  
- 结果：通过  
- 依据：`robust_verification_summary.json` -> `best_run.accuracy_mathverify = 0.1961639058413252`

抽查项 2：`90.15%`（`1034/1147`）  
- 结果：通过  
- 依据：`results/thinking_true/finqa_Qwen_Qwen3-8B_oracle_test.jsonl` 逐条 token 长度统计，`len==256` 计数为 1034

抽查项 3：`169:51:58`  
- 结果：通过  
- 依据：`logs/thinking_true_Qwen_Qwen3-8B_oracle_test.log` 中存在进度行 `5/1147 [35:16<169:51:58, ...]`
