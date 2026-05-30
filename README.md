# FinReason-Lab — Thinking-Aware Baselines and Low-Data Post-Training for Financial Numerical Reasoning

**English** · [简体中文](./README.zh-CN.md)

> A controlled study of **when supervised fine-tuning helps — and when it quietly hurts**.
> Using FinQA as a locked testbed, we show that a smaller model (Qwen3-4B) *recovers* under
> low-data SFT while a larger one (Qwen3-8B) *does not*, and trace the difference to
> **prompt-format alignment** and **model-specific conversational priors** rather than raw scale.

<p align="left">
  <img alt="Task" src="https://img.shields.io/badge/task-FinQA%20numerical%20reasoning-1f6feb">
  <img alt="Models" src="https://img.shields.io/badge/models-Qwen3--4B%20%7C%20Qwen3--8B-8957e5">
  <img alt="Method" src="https://img.shields.io/badge/method-LoRA%20%2F%20QLoRA%20SFT-2da44e">
  <img alt="Eval" src="https://img.shields.io/badge/eval-math--verify%20%7C%20locked%20protocol-d29922">
  <img alt="License" src="https://img.shields.io/badge/code-Apache--2.0-555">
</p>

UCL COMP0087 (Statistical NLP) research project · single-GPU, low-resource regime ·
full code, configs, and orchestration scripts in this repository.

---

## 📊 One-Page Evidence — Results at a Glance

*Everything a reader needs to judge the work, on one screen. All numbers are on the FinQA
test split (n = 1,147), oracle context, no-thinking inference, scored with `math-verify`.*

**Headline result — same SFT recipe, opposite outcomes**

| Model | Prompt | Zero-shot | Best SFT | Δ vs zero-shot | Parse-fail (ZS → SFT) |
|------:|:------:|:---------:|:--------:|:--------------:|:---------------------:|
| **Qwen3-4B** (LoRA)  | text | 24.93% | **32.43%** | **▲ +7.50 pp** | 8.98% → 10.29% |
| **Qwen3-8B** (QLoRA) | chat | 20.40% | **18.74%** | **▼ −1.66 pp** | 3.14% → 3.14% |

> The 8B model stays **below its own zero-shot baseline across every data size and training
> length we tried** (15.69%–18.74% band). Larger ≠ more adaptable here.

**Three findings that survive the controls**

1. **Scale does not guarantee adaptability.** Under an identical, locked protocol, 4B *out-scores*
   8B zero-shot, and only 4B is recovered by SFT.
2. **4B gains are format repair, not new reasoning.** Missing-tag outputs collapse from **409 → 84–96**;
   accuracy follows format compliance.
3. **8B failures are reasoning, not parsing.** Parse-fail stays flat at **3.14%**; the residual errors are
   `numeric_mismatch` on multi-step aggregations (e.g. returns a single-year value instead of a 3-year sum).

**The methodological catch that makes the result trustworthy**

> A train/eval **prompt-template mismatch alone was enough to erase all SFT gains.** We therefore treat
> *prompt-protocol alignment* as a first-class experimental variable and evaluate every checkpoint under
> the template that matches its training format.

**Data quality funnel** (strict clean filter — see `stage1/data/finqa_clean/clean_summary.json`)

```
merged pool         25,185  ──┐
  drop non-FinQA   −18,934    │  keep only formula_exec_ok = true
  drop scale mism.  −2,778    │  AND scale_relation = consistent
  drop exec fail      −196    │
                  ─────────   │
  retained train     3,277  ◀─┘   (+ 475 dev)   ~32% of FinQA dropped
```

**Stack:** Qwen3-4B/8B · LoRA & QLoRA (PEFT) · TRL `SFTTrainer` · BF16 · `math-verify` ·
1× NVIDIA GB10 (121.7 GB) · seed-fixed, fully scripted matrix.

---

## Why this project

Financial QA is a clean stress test for *numerical* reasoning: the answer is grounded in **both tables
and text**, and correctness is unforgiving about scale, percentages, signs, and which number to pick.
FinQA gives us gold reasoning programs, so we can verify answers symbolically instead of by string match.

The project started from a surprising observation under a fixed protocol: **Qwen3-4B beats Qwen3-8B
zero-shot.** That raises a precise, falsifiable question:

> *Is the larger model's weaker baseline a fundamental capacity limit, or just weaker task alignment
> that lightweight supervision can recover?*

We answer it with a **recoverability-framed** comparison: instead of asking "who wins after SFT?", we ask
"how much does each model recover *from its own zero-shot baseline* under matched data, protocol, and
fine-tuning design?"

## What I built (skills demonstrated)

This repo is the end-to-end experimental harness behind the paper — not a notebook dump.
**In this six-author team project I (Jiaming Wei) owned the entire codebase** — every training,
evaluation, data-cleaning, orchestration, and analysis component below is my implementation.
It demonstrates:

- **LLM post-training** — parameter-efficient SFT with **LoRA (4B)** and **QLoRA 4-bit (8B)** via TRL +
  PEFT, fit to a single-GPU memory budget.
- **Rigorous evaluation design** — a *locked* protocol (no-thinking inference, tagged-answer extraction,
  `math-verify` symbolic scoring with percentage rescaling) so zero-shot and SFT numbers are directly
  comparable; three complementary metrics (`acc_base`, `acc_adjusted`, `parse_fail`).
- **Controlled experimentation** — matched hyperparameters across model sizes, confounds named and
  isolated, a nested **data-size ablation** (250 ⊂ 1,000 ⊂ full) and a **training-length** check
  (100 vs 1,640 steps) to separate *more data* from *longer training*.
- **Data engineering** — a strict cleaning pipeline that executes gold formulas to drop label-inconsistent
  rows, with **stratified subsampling** that preserves the single/double/multi-step distribution (seed 42).
- **Diagnostic analysis** — an automated **error-shift** classifier (`parse_fail` / `numeric_mismatch` /
  `percent_scaling` / `unit_confusion`) that explains *why* accuracy moves, not just that it moved.
- **Reproducibility & ops** — portable, path-agnostic scripts (cache via env vars), a one-command sanity
  run, full train/eval matrix orchestrators, and watchdog/`ntfy` runners for long unattended jobs.

## Method overview

| Component | Choice | Notes |
|---|---|---|
| **Evaluation protocol** | no-thinking, single pass | answer wrapped in `[FINAL_ANSWER]…[/FINAL_ANSWER]`, extracted and scored by `math-verify` (atol/rtol = 1e-3, auto percentage rescaling) |
| **Metrics** | `acc_base`, `acc_adjusted`, `parse_fail` | strict match · match-after-rescaling · fraction with no extractable number |
| **PEFT** | LoRA (4B) / QLoRA-4bit (8B) | r = 16, α = 32, dropout 0.05, targets `q_proj`,`v_proj`; QLoRA needed to fit 8B on one GPU |
| **Supervision formats** | `answer_only` vs `formula_rationale` | the latter adds **one** inline formula line before the tag — short, fully grounded, no free-form prose |
| **Prompt-protocol alignment** | train template ↔ eval template | a first-class variable; mismatch alone nullifies SFT gains |
| **Data** | strict `finqa_clean` | 3,277 train / 475 dev, kept only `formula_exec_ok` ∧ `scale_relation=consistent` |

Shared training config: batch size 4, max sequence length 512, 2 epochs, BF16. The two **intentionally
unmatched** settings — adaptation method (LoRA vs QLoRA) and learning rate (5e-5 vs 2e-4) — are named as
confounds and revisited in *Limitations*.

## Results in detail

**Zero-shot baselines** (oracle, `thinking=false`, n = 1,147)

| Model | `acc_base` | `acc_adjusted` | `parse_fail` |
|---|---|---|---|
| Qwen3-4B (text prompt) | 24.93% | 32.78% | 8.98% |
| Qwen3-8B (chat prompt) | 12.03% | 20.40% | 3.14% |

**Data-size & training-length ablation for 8B** (`acc_adjusted`, dashed line = 20.40% zero-shot)

| Condition | 250 | 1,000 | full (100 steps) | full (1,640 steps) |
|---|---|---|---|---|
| Qwen3-8B | 15.95% | 15.69% | 17.52% | **18.74%** |
| Δ vs zero-shot | −4.45 | −4.71 | −2.88 | −1.66 |

→ Neither **more supervision** nor **longer training** closes the gap; 8B never crosses its own baseline.

**Error-shift analysis**

- **4B:** absent-tag outputs drop **409 → 84–96**; `formula_rationale` yields ~3.6 pp fewer `parse_fail`
  than `answer_only`, but ~2.6 pp more `percent_scaling` (e.g. emits `−17.6%` when gold is `−0.176`).
- **8B:** `parse_fail` is **flat at 3.14%**; context-echo artifacts shrink (64% → 6%) yet accuracy does not
  follow. Residual failures are `numeric_mismatch` on multi-step composition.

**Takeaway.** 4B's zero-shot weakness was largely an **interface/format** problem that clean, aligned SFT
repairs. 8B's deficit is in **multi-step arithmetic composition**, and low-data SFT — especially when it
fights a strong chat prior — does not re-weight those pathways.

## Repository structure

```
.
├── README.md                       # you are here
├── FinReason-Lab.pdf               # the paper this README summarizes
├── finqa_baseline/                 # zero-shot + adapter EVALUATION pipeline
│   ├── eval_finqa.py               #   single-run evaluator (math-verify, prompt protocols)
│   ├── run_verification_matrix.sh  #   baseline matrix + robust report
│   └── utils/                      #   prompting, numeric, answer-eval helpers
├── stage1/                         # SFT TRAINING pipeline + orchestration
│   ├── train_sft.py                #   LoRA/QLoRA SFT entrypoint (TRL + PEFT)
│   ├── configs/                    #   debug / small / full YAML configs
│   ├── data/finqa_clean/           #   tracked subset IDs + clean_summary.json
│   ├── scripts/                    #   matrix runners, data cleaning, error-shift analysis
│   └── src/                        #   data loaders, preprocessing, trainer, config
└── docs/
    ├── paper/                      #   report PDF + project proposal
    └── repro/                      #   reproducibility guide + result snapshot
```

The repo tracks only **minimal reproducibility assets** (debug samples, subset ID lists, cleaning
summaries, code/configs). Large regenerated artifacts — checkpoints, logs, full datasets, generated
configs — are intentionally git-ignored.

## Quick start

### 1) Environments

```bash
# Evaluation
cd finqa_baseline && bash setup.sh

# Training
cd ../stage1
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
```

Caches default to `${HOME}/.cache/huggingface`; override with `HF_HOME` / `HF_CACHE_ROOT` /
`HUGGINGFACE_HUB_CACHE` / `TRANSFORMERS_CACHE`. **No script needs a machine-specific absolute path.**

### 2) One-command sanity check (config → preprocess → train → checkpoint → dry-run infer)

```bash
cd stage1 && bash scripts/run_debug.sh        # outputs under stage1/outputs/debug_run/
```

### 3) Reproduce a zero-shot baseline

```bash
cd finqa_baseline
.venv/bin/python eval_finqa.py \
  --model_name Qwen/Qwen3-8B --setting oracle --split test \
  --no-enable_thinking --answer_format final_answer_tag
```

### 4) Main experiment entrypoints

| Goal | Command |
|---|---|
| Baseline matrix | `finqa_baseline/run_verification_matrix.sh` |
| Clean-strict SFT matrix | `stage1/scripts/run_train_eval_matrix_clean_strict.sh` |
| Prompt-aligned re-evaluation | `stage1/scripts/run_eval_matrix_prompt_aligned.sh` |
| 8B full-steps retrain + dual-protocol eval | `stage1/scripts/run_8b_fullsteps_train_eval.sh` |
| Error-shift analysis | `stage1/scripts/analyze_error_shift.py` |

See **`docs/repro/README.md`** for the full reproducibility guide and the summary-reading contract.

## Compute & reproducibility

- **Hardware:** 1× NVIDIA GB10, 121.7 GB. 4B-LoRA ≈ 30 GB; 8B-QLoRA ≈ 55 GB.
- **Wall-clock:** ≈ 1.5 h (4B, 2 epochs) / ≈ 4 h (8B, 2 epochs) / ≈ 7 h (8B, 1,640 steps); eval 20–40 min/run.
- **Software:** TRL `SFTTrainer`, PEFT 0.11, Hugging Face `transformers`, `math-verify==0.9.0`.
- **Determinism:** all training and subsampling use **seed 42**; results are single-run per configuration
  (no multi-seed repetition — see *Limitations*).

## Honest limitations

Reported in full in the paper, and worth stating up front:

- Most SFT runs are **100 steps (~12% of one epoch)** to simulate a low-resource budget; the 1,640-step run
  improves 8B but still does not cross its baseline.
- **LoRA (4B) vs QLoRA (8B)** and **learning rate** are unmatched confounds.
- **No native chat-template SFT for 8B** — the open question is whether format-aligned supervision would
  unlock 8B's capacity.
- **Single benchmark, single test split, no significance testing** — differences are small relative to the
  baseline and could shift under a different seed/split.

These bound the claims rather than undermine them: the evidence that **prompt-format alignment and
model-specific priors are decisive in low-resource SFT** is clean and reproducible.

## Citation

```bibtex
@misc{finreasonlab2025,
  title  = {Thinking-Aware Baselines and Low-Data Post-Training for Financial Numerical Reasoning},
  author = {Zhang, Yike and Wei, Jiaming and Wang, Yuhao and Liu, Yuxin and Wang, Jiaqi and Lang, Victor},
  note   = {UCL COMP0087 Statistical NLP project},
  year   = {2025},
  url    = {https://github.com/Quarkgluonmixture/FinQA}
}
```

## License & data

Code is released under **Apache-2.0**. Qwen3-4B/8B are Apache-2.0; FinQA (Chen et al., 2021) is released
for research use under the MIT license. Use of these resources complies with their respective terms.
