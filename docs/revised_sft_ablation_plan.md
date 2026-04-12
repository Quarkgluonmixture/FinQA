# 4B/8B SFT + 分层消融执行计划（Oracle / No-Thinking / Math-Verify）

## Summary
- 目标不再是只做 `4B/8B` 的单一路径 SFT，而是先回答一个更强的问题：**8B 的落后主要是输出格式未对齐，还是缺少 computation-structure supervision？**
- 因此正式训练条件改为两个**对等 supervision 条件**：
  - `answer_only`
  - `formula_rationale`（formula-grounded visible rationale，不使用长 CoT / hidden thinking）
- 先完成 **4 个 full-data 主实验**：`4B/8B × {answer_only, formula_rationale}`。
- 再根据 full-data 结果做**分层决策**：只对 **8B 的 winner condition** 进入 `0/250/1000/full` 数据量消融；4B 只保留 matched anchor（至少 `0/full`，若算力允许再跑同样消融）。
- 所有正式评测固定为：`oracle` + `--no-enable_thinking` + `final_answer_tag` + `math_verify`。

---

## Public Interfaces / CLI 变更

### 1) `eval_finqa.py`
在 `finqa_baseline/eval_finqa.py` 增加：

- `--adapter_path`（可选）  
  约定：`--model_name` 始终表示 base model；若提供 `--adapter_path`，则先加载 LoRA adapter 后再推理。

### 2) `prompting.py`
在 `stage1/scripts/prompting.py` 增加统一 supervision 开关：

- `--supervision_style {answer_only,formula_rationale}`

约定：
- 默认模板仍是 **no-thinking**
- 默认输出格式仍带 `[FINAL_ANSWER]...[/FINAL_ANSWER]`
- 两种 style 的区别只在 **target/output 字段**：
  - `answer_only`:
    - `[FINAL_ANSWER]0.14[/FINAL_ANSWER]`
  - `formula_rationale`:
    - `Formula: (8.1 / 56.0)\n[FINAL_ANSWER]0.1446[/FINAL_ANSWER]`

### 3) 新增统一编排脚本
新增：

- `stage1/scripts/run_train_eval_matrix.sh`

职责：
- 统一跑 pipeline 验证
- 跑 4 个 full-data 主实验
- 根据结果决定后续消融
- 每个训练 run 结束后立即触发 `oracle` 评测

### 4) 新增数据构造/转换脚本（如需要）
若当前 `train/dev/debug.jsonl` 只含 gold answer、不含 formula supervision 所需字段，则新增：

- `stage1/scripts/build_formula_rationale_targets.py`

职责：
- 从 FinQA gold program / structured derivation 生成标准化 `Formula: ...` target
- 不生成长自然语言 step，不生成 hidden thinking 风格文本

---

## Implementation Changes

### 1. 协议锁定
训练侧与评测侧统一为：

- `no-thinking`
- `[FINAL_ANSWER]` 标签
- `math_verify`
- `oracle` 作为主评测 setting

### 2. supervision 条件改为两个对等分支
不再把 SFT 主线直接锁为单一 `answer_only`，而是并行比较：

- `answer_only`
- `formula_rationale`

注意：
- `formula_rationale` 指 **formula-grounded visible rationale**
- 不使用长 CoT
- 不使用“Step 1: subtract the two values”这类不绑定具体数字的弱 supervision
- 优先采用显式数值与运算绑定的 target，例如：
  - `Formula: (100690000 - 92710000) / 92710000`
  - `[FINAL_ANSWER]0.086[/FINAL_ANSWER]`

### 3. 数据组织建议
优先保持单一数据源，避免 debug/正式数据路径分裂。

推荐二选一：

#### 方案 A：单 raw 数据 + 动态 target formatting
- `stage1/data/unified/{train,dev,debug}.jsonl`
- 原始样本中保留：
  - context
  - question
  - gold_answer
  - formula/program（若有）
- 训练时由 `prompting.py --supervision_style ...` 动态生成 target

#### 方案 B：物化两套 supervision 版本
- `stage1/data/unified/answer_only/{train,dev,debug}.jsonl`
- `stage1/data/unified/formula_rationale/{train,dev,debug}.jsonl`

若时间紧，**优先方案 A**，因为它更少重复数据、更不容易漂移。

---

## Pipeline 验证阶段（必须先完成）

### 验证目标
在正式 full-data 训练前，先确认两种 supervision、两种模型、统一 evaluator 全都能跑通。

### 最小必做验证任务
执行以下 4 个 debug 验证任务：

- `4B-debug-answer_only`
- `4B-debug-formula_rationale`
- `8B-debug-answer_only`
- `8B-debug-formula_rationale`

每个任务必须完整经过：

- 训练启动
- checkpoint 产出
- `eval_finqa.py` 执行
- `summary.json` 落盘

### 推荐扩展验证
若时间允许，再补：

- `4B-small-answer_only`
- `8B-small-answer_only`

用途：
- 检查较长训练是否稳定
- 提前暴露显存 / token 长度 / logging 问题

> 这里不再强制 `debug/small/full × {4B,8B}` 全部做满 6 个或 12 个，只保留对正式问题最必要的验证。

---

## 主实验阶段（第一优先级）

### Stage A：4 个 full-data 主实验
先跑以下 4 个正式 SFT：

- `4B-full-answer_only`
- `4B-full-formula_rationale`
- `8B-full-answer_only`
- `8B-full-formula_rationale`

每个 run 完成后立刻执行 `oracle` 测试集评测，并写入独立 `results_dir`。

### Stage A 的核心问题
这 4 个 run 要回答：

1. 8B 能否被 task-specific SFT 修复？
2. `answer_only` 与 `formula_rationale` 哪个更有效？
3. 4B 与 8B 对 supervision style 的响应是否不同？

### Stage A 输出
至少整理出一张主表：

| model | supervision_style | train_size | acc_adjusted | parse_fail | recovered | notes |
|---|---|---:|---:|---:|---:|---|
| 4B | answer_only | full | ? | ? | ? | |
| 4B | formula_rationale | full | ? | ? | ? | |
| 8B | answer_only | full | ? | ? | ? | |
| 8B | formula_rationale | full | ? | ? | ? | |

---

## 分层决策门（Decision Gate）

在 full-data 4-run 完成后，**先不立即把所有消融全展开**，而是根据结果决定下一步。

### Decision Rule
对 8B，比较：

- `8B-full-answer_only`
- `8B-full-formula_rationale`

优先依据：
1. `acc_adjusted`
2. error shift（是否真正减少 wrong-number / magnitude / numeric mismatch）
3. 训练稳定性与输出兼容性

### Winner Condition
定义 8B 的更优 supervision 条件为：

- `winner_style ∈ {answer_only, formula_rationale}`

之后的数据量消融只围绕 **8B-winner_style** 展开。

---

## 消融阶段（第二优先级）

### Stage B：8B winner-style 数据量消融
固定 `winner_style` 后，运行：

- `8B-250-{winner_style}`
- `8B-1000-{winner_style}`
- `8B-full-{winner_style}`（已在主实验中获得）
- `8B-0-shot`（若历史 baseline 协议一致则复用，否则重跑）

得到一条 8B 的主曲线：

- `acc_adjusted` vs `0/250/1000/full`

### 4B 的角色
4B 主要作为 matched anchor。

#### 最低要求（必须）
- `4B-0-shot`
- `4B-full-{winner_style}`（若主实验已有对应 style，则直接复用）

#### 理想情况（算力允许再做）
- `4B-250-{winner_style}`
- `4B-1000-{winner_style}`

也就是说：

- **8B** 是消融主角
- **4B** 不是必须完整展开所有点位，除非时间和 GPU 允许

### 最终目标
- 至少得到一条完整的 8B 曲线
- 至少有 4B 的 matched anchor 点用于解释规模差异

---

## 评测命令锁定（训练后自动调用）

固定参数：

- `--setting oracle`
- `--split test`
- `--no-enable_thinking`
- `--evaluator math_verify`
- `--answer_format final_answer_tag`
- `--final_answer_tag FINAL_ANSWER`

约定：
- `--model_name` 传 base model（`Qwen3-4B` / `Qwen3-8B`）
- `--adapter_path` 指向当前 run 的 `checkpoint-last` 或 best checkpoint

---

## Test Plan / Acceptance

### 1. 格式回归
先跑：

- `regression_final_answer_mathverify.py`

要求：
- `answer_only` 样本全通过
- `formula_rationale` 样本全通过
- open tag / absent tag / extra formula line 均不导致解析崩溃

### 2. 每个训练 run 的通过标准
必须满足：

- 存在 checkpoint 目录
- 评测成功生成 `finqa_*.jsonl` 与 `summary.json`
- run metadata 中记录：
  - `evaluator=math_verify`
  - `answer_format=final_answer_tag`
  - `enable_thinking=false`
  - `supervision_style=...`

### 3. 兼容性确认
`summary` 中应包含或可追溯得到：

- `tag_status_counts`
- `parse_fail_rate_mathverify`
- `acc_adjusted`
- `recovered`

并保证：
- 评测流程无解析崩溃
- `formula_rationale` 不会破坏 final-answer extraction

### 4. Stage A 完成标准
- 4 个 full-data 主实验全部完成
- 能回答 `answer_only vs formula_rationale` 对 4B/8B 的影响
- 能明确选出 `8B winner_style`

### 5. Stage B 完成标准
最低完成标准：

- `8B winner_style` 的 `0/250/1000/full` 完成
- `4B` 至少有 matched `0/full` anchor
- 产出可直接画图的数据表

---

## Assumptions and Defaults

- 主评测范围固定为 `oracle`
- inference 固定 `no-thinking`
- 输出格式固定 `[FINAL_ANSWER]...[/FINAL_ANSWER]`
- 两种 supervision condition：
  - `answer_only`
  - `formula_rationale`
- `formula_rationale` 使用**短公式监督**，不使用长 CoT
- 数据量主消融点位固定：`0/250/1000/full`
- LoRA checkpoint 通过 `eval_finqa --adapter_path` 直接评测
- 先做 full-data 4-run 再做分层消融，不预先把所有 run 全部摊开

---

## Hard Decision Rules

### Rule 1
如果 debug 验证阶段未能让两种 supervision style 都稳定通过，则**先冻结 style 复杂度**，优先保住 `answer_only` 主线，不要强推 formula variant 到崩。

### Rule 2
如果 4 个 full-data 主实验到 deadline 中段仍未跑完，则**直接砍掉全部数据量消融**，先保住主比较表。

### Rule 3
如果 `formula_rationale` 明显优于 `answer_only`，则后续消融只围绕 `formula_rationale` 展开；反之亦然。

### Rule 4
如果 8B 训练显著不稳或速度过慢，则保住：
- `8B-full-answer_only`
- `8B-full-formula_rationale`
- `4B-full-*` matched 对照  
不要为了追求更多点位而毁掉主结论。

---

## 最终一句话版本
> 先用 `4B/8B × answer_only/formula_rationale` 的 4 个 full-data run 回答“8B 的欠表现主要是格式未对齐还是缺少 computation-structure supervision”，再只对 8B 的 winner condition 做 `0/250/1000/full` 数据量消融，并用 4B 作为 matched anchor。
