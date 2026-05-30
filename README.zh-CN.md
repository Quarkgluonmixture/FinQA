# FinReason-Lab — 面向金融数值推理的「思考感知」基线与低数据后训练研究

[English](./README.md) · **简体中文**

> 一项受控研究:**监督微调(SFT)何时有效,又何时会悄悄帮倒忙。**
> 以 FinQA 为锁定测试床,我们发现较小的模型(Qwen3-4B)在低数据 SFT 下能够*恢复*,
> 而更大的模型(Qwen3-8B)却*不能*;并将这一差异归因于 **prompt 格式对齐** 与
> **模型自带的对话先验**,而非单纯的参数规模。

<p align="left">
  <img alt="Task" src="https://img.shields.io/badge/task-FinQA%20numerical%20reasoning-1f6feb">
  <img alt="Models" src="https://img.shields.io/badge/models-Qwen3--4B%20%7C%20Qwen3--8B-8957e5">
  <img alt="Method" src="https://img.shields.io/badge/method-LoRA%20%2F%20QLoRA%20SFT-2da44e">
  <img alt="Eval" src="https://img.shields.io/badge/eval-math--verify%20%7C%20locked%20protocol-d29922">
  <img alt="License" src="https://img.shields.io/badge/code-Apache--2.0-555">
</p>

UCL COMP0087(统计自然语言处理)研究项目 · 单 GPU、低资源场景 · 本仓库含全部代码、配置与编排脚本。

---

## 📊 一页可见证据 — 结果速览

*判断这份工作所需的一切,一屏放下。所有数字均基于 FinQA 测试集(n = 1,147)、oracle 上下文、
no-thinking 推理,并用 `math-verify` 评分。*

**核心结果 —— 同一套 SFT 配方,截然相反的结局**

| 模型 | Prompt | 零样本 | 最佳 SFT | Δ(相对零样本) | Parse-fail(零样本 → SFT) |
|------:|:------:|:------:|:--------:|:--------------:|:-------------------------:|
| **Qwen3-4B**(LoRA)  | text | 24.93% | **32.43%** | **▲ +7.50 pp** | 8.98% → 10.29% |
| **Qwen3-8B**(QLoRA) | chat | 20.40% | **18.74%** | **▼ −1.66 pp** | 3.14% → 3.14% |

> 8B 在我们尝试的**每一个数据规模与训练长度下,都低于其自身的零样本基线**(15.69%–18.74% 区间)。
> 在这里,更大 ≠ 更易适配。

**三条经得起对照的发现**

1. **规模不保证可适配性。** 在同一套锁定协议下,4B 零样本就*高于* 8B,且只有 4B 能被 SFT 恢复。
2. **4B 的增益来自格式修复,而非新增推理能力。** 缺失标签的输出从 **409 → 84–96**;准确率随格式合规度走。
3. **8B 的失败在推理而非解析。** Parse-fail 恒定为 **3.14%**;残余错误是多步聚合上的 `numeric_mismatch`
   (例如返回单年数值,而非三年求和)。

**让结论可信的方法学要害**

> 仅仅是训练/评测的 **prompt 模板错配,就足以抹平全部 SFT 增益。** 因此我们把 *prompt 协议对齐*
> 当作一等实验变量,并在与各 checkpoint 训练格式相匹配的模板下评测它。

**数据质量漏斗**(严格清洗过滤 —— 见 `stage1/data/finqa_clean/clean_summary.json`)

```
合并语料池        25,185  ──┐
  去非 FinQA 源  −18,934    │  仅保留 formula_exec_ok = true
  去刻度不一致    −2,778    │  且 scale_relation = consistent
  去公式执行失败    −196    │
                 ─────────  │
  保留训练集       3,277  ◀─┘   (+ 475 dev)   约 32% 的 FinQA 被剔除
```

**技术栈:** Qwen3-4B/8B · LoRA & QLoRA(PEFT)· TRL `SFTTrainer` · BF16 · `math-verify` ·
1× NVIDIA GB10(121.7 GB)· 固定随机种子、全程脚本化的实验矩阵。

---

## 项目缘起

金融问答是 *数值* 推理的一个干净压力测试:答案需要同时落地到**表格与文本**,而正确性对刻度、
百分比、正负号、以及"该选哪个数"毫不留情。FinQA 提供了金标准推理程序,因此我们可以**符号化地验证**
答案,而非靠字符串匹配。

项目始于一个在固定协议下的反常观察:**Qwen3-4B 零样本竟然胜过 Qwen3-8B。** 这引出一个精确、可证伪的问题:

> *更大模型偏弱的基线,究竟是根本性的容量上限,还是仅仅是轻量监督就能恢复的任务对齐不足?*

我们用**「可恢复性」框架**来回答它:不问"SFT 之后谁赢",而问"在数据、协议、微调设计都匹配的条件下,
每个模型相对**自身零样本基线**恢复了多少"。

## 我做了什么(所展示的能力)

本仓库是论文背后的端到端实验框架 —— 不是一堆 notebook 堆砌。
**在这个六人团队项目中,全部代码由我(Jiaming Wei)负责** —— 下述每一个训练、评测、数据清洗、
编排与分析组件都是我的实现。它展示了:

- **大模型后训练** —— 通过 TRL + PEFT 做参数高效 SFT:**LoRA(4B)** 与 **QLoRA 4-bit(8B)**,
  适配单 GPU 显存预算。
- **严谨的评测设计** —— 一套*锁定*协议(no-thinking 推理、标签化答案抽取、`math-verify` 符号评分
  并自动百分比重缩放),使零样本与 SFT 数字可直接对比;三个互补指标
  (`acc_base`、`acc_adjusted`、`parse_fail`)。
- **受控实验** —— 跨模型规模匹配超参,显式命名并隔离混淆因素,嵌套的**数据规模消融**
  (250 ⊂ 1,000 ⊂ full)与**训练长度**检验(100 vs 1,640 步),把*更多数据*与*更长训练*分开。
- **数据工程** —— 一条严格清洗流水线,执行金标准公式以剔除标注不一致的样本,并用**分层子采样**
  保持单步/双步/多步推理的分布(种子 42)。
- **诊断分析** —— 一个自动化的**误差迁移(error-shift)**分类器
  (`parse_fail` / `numeric_mismatch` / `percent_scaling` / `unit_confusion`),解释准确率*为何*变化,
  而不只是变了。
- **可复现与工程化** —— 可移植、不依赖绝对路径的脚本(缓存走环境变量)、一条命令的健全性自检、
  完整的训练/评测矩阵编排,以及用于长时无人值守任务的 watchdog/`ntfy` 运行器。

## 方法概览

| 组件 | 选择 | 说明 |
|---|---|---|
| **评测协议** | no-thinking、单遍生成 | 答案包裹于 `[FINAL_ANSWER]…[/FINAL_ANSWER]`,由 `math-verify` 抽取并评分(atol/rtol = 1e-3,自动百分比重缩放) |
| **指标** | `acc_base`、`acc_adjusted`、`parse_fail` | 严格匹配 · 重缩放后匹配 · 无法抽出数字的比例 |
| **PEFT** | LoRA(4B)/ QLoRA-4bit(8B) | r = 16、α = 32、dropout 0.05,目标 `q_proj`,`v_proj`;8B 需 QLoRA 才能塞进单 GPU |
| **监督格式** | `answer_only` vs `formula_rationale` | 后者在标签前加**一行**内联公式 —— 简短、完全由数值落地、无自由文本 |
| **prompt 协议对齐** | 训练模板 ↔ 评测模板 | 一等变量;错配单独一项即可抹平 SFT 增益 |
| **数据** | 严格 `finqa_clean` | 3,277 训练 / 475 dev,仅保留 `formula_exec_ok` ∧ `scale_relation=consistent` |

共享训练配置:batch size 4、最大序列长度 512、2 个 epoch、BF16。两项**有意未匹配**的设置 ——
适配方法(LoRA vs QLoRA)与学习率(5e-5 vs 2e-4)—— 被显式列为混淆因素,并在*局限性*中重新审视。

## 结果详情

**零样本基线**(oracle、`thinking=false`、n = 1,147)

| 模型 | `acc_base` | `acc_adjusted` | `parse_fail` |
|---|---|---|---|
| Qwen3-4B(text prompt) | 24.93% | 32.78% | 8.98% |
| Qwen3-8B(chat prompt) | 12.03% | 20.40% | 3.14% |

**8B 的数据规模 & 训练长度消融**(`acc_adjusted`,虚线 = 20.40% 零样本)

| 条件 | 250 | 1,000 | full(100 步) | full(1,640 步) |
|---|---|---|---|---|
| Qwen3-8B | 15.95% | 15.69% | 17.52% | **18.74%** |
| Δ 相对零样本 | −4.45 | −4.71 | −2.88 | −1.66 |

→ **更多监督**与**更长训练**都没能合拢差距;8B 从未越过自身基线。

**误差迁移分析**

- **4B:** 缺失标签的输出从 **409 → 84–96**;`formula_rationale` 比 `answer_only` 少约 3.6 pp 的
  `parse_fail`,但多约 2.6 pp 的 `percent_scaling`(例如金标准是 `−0.176` 却输出 `−17.6%`)。
- **8B:** `parse_fail` **恒定为 3.14%**;上下文回显伪影收缩(64% → 6%),但准确率并不跟随。
  残余失败是多步组合上的 `numeric_mismatch`。

**要点。** 4B 的零样本弱点主要是**接口/格式**问题,干净且对齐的 SFT 即可修复;8B 的缺口在
**多步算术组合**,而低数据 SFT —— 尤其当它要对抗强烈的 chat 先验时 —— 无法重新加权那些通路。

## 仓库结构

```
.
├── README.md / README.zh-CN.md     # 英文 / 中文说明
├── FinReason-Lab.pdf               # 本 README 所总结的论文
├── finqa_baseline/                 # 零样本 + adapter 的「评测」流水线
│   ├── eval_finqa.py               #   单次评测器(math-verify、多 prompt 协议)
│   ├── run_verification_matrix.sh  #   基线矩阵 + 稳健报告
│   └── utils/                      #   prompting、numeric、answer-eval 辅助
├── stage1/                         # SFT「训练」流水线 + 编排
│   ├── train_sft.py                #   LoRA/QLoRA SFT 入口(TRL + PEFT)
│   ├── configs/                    #   debug / small / full YAML 配置
│   ├── data/finqa_clean/           #   纳入版本控制的子集 ID + clean_summary.json
│   ├── scripts/                    #   矩阵运行器、数据清洗、误差迁移分析
│   └── src/                        #   数据加载、预处理、trainer、配置
└── docs/
    ├── paper/                      #   报告 PDF + 项目提案
    └── repro/                      #   可复现指南 + 结果快照
```

仓库只纳入**最小可复现资产**(debug 样本、子集 ID 列表、清洗摘要、代码/配置)。大型再生成产物 ——
checkpoints、日志、完整数据集、生成的配置 —— 被有意 git-ignore。

## 快速开始

### 1)环境

```bash
# 评测
cd finqa_baseline && bash setup.sh

# 训练
cd ../stage1
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

缓存默认指向 `${HOME}/.cache/huggingface`;可用 `HF_HOME` / `HF_CACHE_ROOT` /
`HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE` 覆盖。**没有任何脚本依赖机器特定的绝对路径。**

### 2)一条命令的健全性自检(配置 → 预处理 → 训练 → checkpoint → dry-run 推理)

```bash
cd stage1 && bash scripts/run_debug.sh        # 输出位于 stage1/outputs/debug_run/
```

### 3)复现一条零样本基线

```bash
cd finqa_baseline
.venv/bin/python eval_finqa.py \
  --model_name Qwen/Qwen3-8B --setting oracle --split test \
  --no-enable_thinking --answer_format final_answer_tag
```

### 4)主实验入口

| 目标 | 命令 |
|---|---|
| 基线矩阵 | `finqa_baseline/run_verification_matrix.sh` |
| 严格清洗 SFT 矩阵 | `stage1/scripts/run_train_eval_matrix_clean_strict.sh` |
| prompt 对齐再评测 | `stage1/scripts/run_eval_matrix_prompt_aligned.sh` |
| 8B 全步数重训 + 双协议评测 | `stage1/scripts/run_8b_fullsteps_train_eval.sh` |
| 误差迁移分析 | `stage1/scripts/analyze_error_shift.py` |

完整可复现指南与 summary 读取约定见 **`docs/repro/README.md`**。

## 算力与可复现性

- **硬件:** 1× NVIDIA GB10,121.7 GB。4B-LoRA ≈ 30 GB;8B-QLoRA ≈ 55 GB。
- **墙钟时间:** ≈ 1.5 h(4B,2 epoch)/ ≈ 4 h(8B,2 epoch)/ ≈ 7 h(8B,1,640 步);评测 20–40 分钟/次。
- **软件:** TRL `SFTTrainer`、PEFT 0.11、Hugging Face `transformers`、`math-verify==0.9.0`。
- **确定性:** 所有训练与子采样使用**种子 42**;每个配置单次运行(未做多种子重复 —— 见*局限性*)。

## 诚实的局限性

论文中有完整记录,这里也先讲清楚:

- 多数 SFT 运行为 **100 步(约一个 epoch 的 12%)**,以模拟低资源预算;1,640 步的运行改善了 8B,
  但仍未越过基线。
- **LoRA(4B)vs QLoRA(8B)** 与**学习率**是未匹配的混淆因素。
- **8B 没有做原生 chat 模板的 SFT** —— 开放问题是:格式对齐的监督是否会解锁 8B 的容量。
- **单一基准、单一测试切分、无显著性检验** —— 差异相对基线较小,换种子/切分可能变化。

这些是给结论划定边界,而非推翻它:**prompt 格式对齐与模型自带先验在低资源 SFT 中起决定性作用**
这一证据是干净且可复现的。

## 引用

```bibtex
@misc{finreasonlab2025,
  title  = {Thinking-Aware Baselines and Low-Data Post-Training for Financial Numerical Reasoning},
  author = {Zhang, Yike and Wei, Jiaming and Wang, Yuhao and Liu, Yuxin and Wang, Jiaqi and Lang, Victor},
  note   = {UCL COMP0087 Statistical NLP project},
  year   = {2025},
  url    = {https://github.com/Quarkgluonmixture/FinQA}
}
```

## 许可与数据

代码以 **Apache-2.0** 发布。Qwen3-4B/8B 为 Apache-2.0;FinQA(Chen et al., 2021)以 MIT 许可证发布,
供研究使用。我们对这些资源的使用符合各自条款。
