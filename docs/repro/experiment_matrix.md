# COMP0087 FinQA — Updated Experiment Matrix and Execution Plan (Repo-Aligned)

**Version**: v2.0  
**Date**: 2026-04-03  
**Deadline**: 2026-04-17  
**Repository**: `Quarkgluonmixture/FinQA`

---

## 1. Project lock-in: what we are actually doing

### 1.1 Final main question

> **RQ:** Under a fixed **no-thinking + controlled answer format + math-verify** evaluation protocol, can **low-data task-specific SFT** recover the anomalous underperformance of **Qwen3-8B** on FinQA, and how does that recovery compare with **Qwen3-4B**?

This is the cleanest research version of the project because it does **not** assume 8B must become better than 4B. Instead, it asks whether the larger model is **more recoverable** through task-specific alignment.

### 1.2 What is now fixed

The project should now be treated as **protocol-locked**:

- **Primary inference mode**: `no-thinking`
- **Primary answer format**: controlled final answer tag / numeric extraction-compatible format
- **Primary evaluator**: `math-verify`
- **Primary benchmark**: `FinQA`
- **Primary target model**: `Qwen3-8B`
- **Comparator model**: `Qwen3-4B`
- **Primary method**: `LoRA/QLoRA SFT`
- **Primary analysis**: `data ablation + error shift`

### 1.3 What is no longer on the main path

The following are explicitly **de-scoped from the core contribution**:

- `thinking=true` as a main baseline
- RL / GRPO
- full robustness study on MATH / GPQA
- multi-dataset expansion as a main story
- synthetic-data pipeline as a required component

These can remain **optional tail experiments** only if the core results finish early.

---

## 2. Repo-aware update: what has already been built

The repository already contains a usable **Stage 1 SFT skeleton**, which means the project has moved beyond planning and into execution.

### 2.1 Existing Stage 1 capability

Current repo capability already covers:

- configuration-driven training modes: `debug / small / full`
- data loading and prompt preprocessing
- TRL + LoRA / QLoRA training
- checkpoint saving
- post-training smoke inference
- end-to-end pipeline verification

This changes the project plan in an important way:

> the bottleneck is **no longer building the training framework**, but **using the framework to answer a clean research question**.

### 2.2 Local data status

You currently already have local dataset files prepared:

- `debug.jsonl`
- `train.jsonl`
- `dev.jsonl`

These are sufficient to start the core project. Because they are large and not uploaded, this document assumes:

- `train.jsonl` is the master supervised training file
- `dev.jsonl` is used for validation / model selection
- `debug.jsonl` is used only for end-to-end sanity checks

So the project no longer needs a separate “data construction first” phase as the top priority unless the current `train.jsonl` schema turns out to be unsuitable.

---

## 3. Final experimental story

## 3.1 Core thesis

The thesis is **not**:

> “8B should be stronger, so we will make it stronger.”

The thesis is:

> “8B is unexpectedly weak under the current zero-shot protocol. We test whether this weakness is mainly an alignment problem that can be repaired by low-data SFT.”

That makes the project falsifiable.

### 3.2 Why 4B must stay in the story

4B must remain in the project for two reasons:

1. it is the strongest current anchor under the controlled zero-shot setting
2. without 4B SFT as a comparator, any 8B improvement is hard to interpret

If only 8B is trained, then an examiner can always ask:

> “How do you know this is 8B-specific recoverability rather than the generic effect of SFT on FinQA?”

So **4B SFT is not optional**. It is the minimum scientific control.

---

## 4. Updated experiment matrix

## 4.0 Priority structure

### Tier 1 — must finish
- EXP-0: protocol lock + repo smoke verification
- EXP-1: zero-shot anchor table finalization
- EXP-2: 4B full SFT
- EXP-3: 8B full SFT
- EXP-4: data ablation
- EXP-5: error shift analysis
- report writing

### Tier 2 — do if time permits
- EXP-6: oracle vs full transfer check
- EXP-7: thinking-at-inference ablation on best checkpoints

### Tier 3 — almost certainly cut
- EXP-8: robustness on MATH / GPQA
- EXP-9: RL / GRPO

---

## 5. Detailed experiments

## EXP-0 — Protocol lock and repo verification

**Goal**  
Freeze the experimental protocol and verify the repository can produce comparable outputs for all later runs.

**Input**
- existing repo
- `debug.jsonl`
- current training configs

**Actions**
1. verify `debug` config runs end-to-end
2. verify training -> checkpoint -> smoke inference -> evaluation path
3. verify output format matches evaluator expectations
4. verify both 4B and 8B configs can at least initialize correctly
5. freeze a single evaluation script for all later experiments

**Deliverable**
- one short run log
- one confirmed checkpoint path
- one confirmed evaluation command
- one frozen protocol note in repo

**Why this matters**
This prevents the project from drifting into “different models, different scripts, different evaluation assumptions.”

---

## EXP-1 — Zero-shot anchor table finalization

**Goal**  
Freeze the baseline table that all SFT results will be compared against.

**Required dimensions**
- model: `4B`, `8B`
- setting: `oracle`, `full`
- mode: `no-thinking`
- evaluator: `math-verify`
- answer format: controlled tag format

**Output**
A final baseline table with:
- `acc_base`
- `acc_adjusted`
- `parse_fail_rate`
- `recovered`
- concise interpretation

**Important**
This table should be treated as **read-only** after lock. No silent metric changes later.

---

## EXP-2 — 4B full SFT (control model)

**Goal**  
Train `Qwen3-4B` with the current SFT pipeline and establish the control gain from task-specific supervision.

**Why it matters**
This is the scientific control for interpreting 8B gains.

**Data**
- training: `train.jsonl`
- validation: `dev.jsonl`

**Training mode**
- use current repo’s stable SFT path
- LoRA if enough memory
- QLoRA if needed for consistency with 8B
- keep training recipe as matched as possible with 8B

**Output**
- trained 4B checkpoint
- evaluation on FinQA test
- delta from 4B zero-shot baseline

**Interpretation**
This tells you how much of the final gain is simply “FinQA-specific SFT helps” rather than “8B is specially recoverable.”

---

## EXP-3 — 8B full SFT (primary model)

**Goal**  
Train `Qwen3-8B` using the same protocol and compare its recoverability with 4B.

**Data**
- training: same `train.jsonl`
- validation: same `dev.jsonl`

**Training mode**
- preferably QLoRA
- hyperparameters should be matched to 4B as much as possible
- only change what hardware forces you to change

**Primary comparison**
- 8B zero-shot -> 8B SFT delta
- compare this delta with 4B zero-shot -> 4B SFT delta

**What counts as a meaningful result**
Any of the following are publishable-coursework-grade findings:

1. **8B gains much more than 4B**  
   -> suggests 8B was under-aligned and SFT unlocks more of its task capacity

2. **8B gains, but not enough to overtake 4B SFT**  
   -> suggests larger size alone does not dominate task adaptation efficiency

3. **8B gains only slightly or not at all**  
   -> suggests the zero-shot gap is not easily repairable by small-scale SFT

All three outcomes are scientifically usable.

---

## EXP-4 — Data ablation (main analysis)

**Goal**  
Measure how much data is needed before SFT yields meaningful gains.

### 4.1 Recommended ablation points

Because the repo already uses `debug / small / full`, and because time is limited, the matrix should be simplified.

**Recommended points**
- `0` = zero-shot
- `debug` = sanity only, not for headline result
- `small`
- `mid` (derived subset from `train.jsonl`)
- `full`

### 4.2 Practical recommendation

If you want a concrete operational plan, use:

- `0`
- `250`
- `1000`
- `full`

or, if the repo is already strongly built around named configs:

- `0`
- `small`
- `medium`
- `full`

### 4.3 Important reproducibility rule

Do **not** create arbitrary random subsets each time.

Use one of these two reproducible strategies:

#### Option A — deterministic subset index files
Create files such as:
- `splits/train_ids_small.txt`
- `splits/train_ids_medium.txt`

#### Option B — nested materialized JSONL subsets
Create:
- `train_small.jsonl`
- `train_medium.jsonl`

Either way, enforce:

`small ⊂ medium ⊂ full`

so that data size is the only intended variable.

### 4.4 What to plot

For both 4B and 8B:
- x-axis: training size
- y-axis: `acc_adjusted`
- optional secondary y-axis or table: `parse_fail_rate`

The main quantity of interest is **gain slope**, not just absolute endpoint.

---

## EXP-5 — Error shift analysis

**Goal**  
Explain *why* performance changed after SFT.

This experiment is extremely valuable because it gives mechanism, not just scoreboard.

### Key questions
- Did SFT mainly reduce formatting / extraction errors?
- Did it reduce wrong-number selection?
- Did it reduce scale / unit mistakes?
- Did it reduce deeper numeric mismatch?
- Did 8B and 4B improve in the same way?

### Required outputs
1. pre/post error taxonomy table
2. relative change by error type
3. 6–10 representative cases

### Interpretation rule
If most of the gain comes from formatting and scale recovery, say so clearly.  
Do not oversell it as “reasoning got much better” unless the case evidence supports that.

---

## EXP-6 — Optional oracle/full follow-up

**Goal**  
Check whether SFT gains transfer similarly across cleaner and noisier contexts.

**Recommended position**
This is a **secondary** experiment, not the main story.

**Why**
The main scientific question is whether SFT repairs the 8B gap under a controlled protocol.  
That question is cleanest in `oracle`.

So the recommended reporting order is:

- **primary main-table result**: `oracle`
- **secondary validation**: `full`

---

## EXP-7 — Optional thinking-at-inference ablation

**Goal**  
Test whether the SFT checkpoint behaves differently under `thinking=true` vs `thinking=false` at inference time.

**Strict rule**
This must not replace the main protocol.

**Reason**
Your current evidence already says the thinking path is unstable and expensive. So this can only be a side ablation, never the core evaluation route.

---

## 6. Updated data plan

## 6.1 Current assumption

Because you already have:
- `debug.jsonl`
- `train.jsonl`
- `dev.jsonl`

the data plan should now be:

### Training master file
`train.jsonl`

### Validation file
`dev.jsonl`

### Pipeline sanity file
`debug.jsonl`

### Derived files only if necessary
- `train_small.jsonl`
- `train_medium.jsonl`

Do **not** spend multiple days rebuilding the entire data pipeline unless the current schema is broken.

## 6.2 Data-format principle

The core question now is no longer “can we build training data?”  
It is “can our current training data support a defensible SFT comparison?”

So only check the minimum necessary:
- field names are consistent
- prompt formatting is stable
- final answer extraction is evaluator-compatible
- percentage representation is consistent

---

## 7. Updated report structure

## 7.1 Recommended title direction

A stronger project title would now look like one of these:

- **Can Low-Data SFT Recover Larger-Model Underperformance in Financial Numerical Reasoning?**
- **Recovering Qwen3-8B on FinQA: A Controlled Study of Low-Data Task-Specific SFT**
- **When Bigger Is Worse: Can Task-Specific SFT Repair Larger-Model Underperformance on FinQA?**

## 7.2 Paper structure

### 1. Introduction
- FinQA task
- surprising 4B > 8B zero-shot observation
- research question: recoverability under SFT

### 2. Related Work
- financial QA / FinQA
- reasoning and numerical QA
- SFT for task alignment
- model scale vs adaptation

### 3. Method
- locked evaluation protocol
- repo training setup
- dataset split usage
- matched 4B / 8B training design

### 4. Results
- zero-shot anchor table
- 4B vs 8B SFT main result
- data ablation
- error shift analysis

### 5. Discussion
- what kind of failure SFT repaired
- whether 8B had more recoverable headroom
- limitations

### 6. Conclusion
- what was learned
- what was intentionally left out

---

## 8. Hard decision rules

These rules help prevent scope drift.

### Rule 1
If by **4/8** the full 4B and full 8B SFT runs are not both working, drop all optional experiments.

### Rule 2
If by **4/10** the main 4B/8B SFT comparison is not complete, shrink ablation to only:
- `0`
- `small`
- `full`

### Rule 3
If by **4/12** the report still lacks the main comparison figure, stop running new experiments and switch fully to writing.

### Rule 4
If 8B training becomes too unstable or too slow, keep the project scientifically valid by reframing around:
- “matched SFT comparison under tight compute constraints”
rather than forcing more scale.

---

## 9. Updated timeline to 4/17

| Date | Main target |
|---|---|
| 4/3 | Freeze protocol + update experiment doc + verify repo configs |
| 4/4 | Run debug pipeline end-to-end; confirm evaluator compatibility |
| 4/5 | Launch 4B full SFT |
| 4/6 | Evaluate 4B; launch 8B full SFT |
| 4/7 | Evaluate 8B; lock main comparison |
| 4/8–4/10 | Run reduced data ablation |
| 4/10–4/11 | Run error shift analysis |
| 4/11–4/13 | Write main report sections |
| 4/13–4/14 | Optional small ablation / oracle-full supplement if safe |
| 4/15 | Final figures and discussion |
| 4/16 | Final review and polish |
| 4/17 | Submit |

---

## 10. Final project sentence

> We will use the already-built SFT training skeleton in our FinQA repository to run a protocol-locked comparison of 4B and 8B models under no-thinking evaluation, testing whether low-data task-specific SFT can recover the anomalous zero-shot underperformance of the larger model and how much training data is required for that recovery.

---

## 11. Checklist

### Protocol
- [ ] no-thinking locked as the primary inference mode
- [ ] evaluator locked to math-verify
- [ ] answer extraction format frozen

### Repo
- [ ] debug config runs
- [ ] small/full config paths confirmed
- [ ] 4B and 8B training configs both initialize
- [ ] smoke inference output is evaluator-compatible

### Core science
- [ ] final zero-shot anchor table frozen
- [ ] 4B full SFT complete
- [ ] 8B full SFT complete
- [ ] matched comparison written
- [ ] ablation curve produced
- [ ] error shift table produced

### Writing
- [ ] title finalized
- [ ] introduction drafted
- [ ] method drafted
- [ ] result figures ready
- [ ] discussion written
- [ ] scope limitations stated clearly
