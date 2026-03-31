# COMP0087 Statistical NLP

## Project Proposal 

**Working Title**
 **Thinking-Aware Baselines and Low-Data Post-Training for Financial Numerical Reasoning**

**Course**
 COMP0087 Statistical NLP

**Project Type**
 6-person group project

**Primary task domain**
 Financial numerical reasoning over text–table hybrid inputs

**Primary benchmark**
 FinQA, an EMNLP 2021 benchmark for numerical reasoning over financial reports with gold reasoning programs. ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.300/))

**Extension benchmarks / data sources**
 ConvFinQA, a conversational numerical reasoning extension in finance, and MultiHiertt, a benchmark focused on multi-hierarchical tabular and textual financial data. ([ACL Anthology](https://aclanthology.org/2022.emnlp-main.421/))

**Core infrastructure**
 Math-Verify for robust mathematical answer verification, and Hugging Face TRL for SFT, with optional GRPO-based RL as a stretch goal. ([GitHub](https://github.com/huggingface/Math-Verify))





## 1. Project Motivation

Financial QA is not just a standard reading-comprehension problem. It requires retrieving relevant evidence from financial reports, handling heterogeneous inputs such as text and tables, and performing multi-step numerical reasoning with stable treatment of percentages, ratios, signs, and magnitudes. FinQA was introduced specifically to study this setting and includes gold reasoning programs, which makes it especially suitable for controlled evaluation and training design. ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.300/))

Our group’s current baseline results already suggest that, under a strict direct-answer protocol, the task is far from saturated and that the dominant failure modes are not simple parse failures but scale errors, percent/ratio instability, and wrong number selection. Based on TA feedback, the project should now move from a purely non-thinking baseline analysis into a fuller research pipeline:

1. build the **correct no-thinking-first baseline space** (thinking=false primary), with explicit runtime-quality trade-offs;
2. upgrade evaluation from brittle string matching to **mathematical equivalence verification**;
3. study whether **low-data post-training** can improve FinQA;
4. examine **how much data is actually needed**;
5. ensure improvements on FinQA do not come at the cost of broader reasoning ability on out-of-domain benchmarks such as MATH and GPQA. ([GitHub](https://github.com/huggingface/Math-Verify))





## 2. Why the Project Was Revised

Our earlier direction was too method-heavy too early. It leaned toward verifier/DSL-style ideas before fully establishing:

- what the modern baseline actually is,
- where the errors really come from,
- whether the benchmark is already saturated,
- and whether a simpler intervention would already solve most of     the problem.

After TA feedback, the revised principle is:

**No complex training or RL story before a strong mode-aware baseline, a robust evaluator, and a clear failure diagnosis.**

That does **not** mean your earlier non-thinking experiments were wasted. They remain valuable. In the revised proposal, they become:

- an **existing ablation**,
- an important point of comparison,
- and evidence for how much explicit reasoning mode matters.

But from now on:

- **Stage 0 execution default:** thinking=false only;
- the **default SFT path** is a compute-efficient answer-only recipe with strict final-answer tags;
- thinking=true is postponed to a bounded optional pilot after Stage 0 if time/compute allow.





## 3. Core Research Questions

### RQ0. Baseline Question

Under a controlled evaluation setup, how do **generation budget** and **output format constraints** affect performance on financial numerical reasoning under a no-thinking default?

This is now the baseline question we execute first. Prompting choices are not treated as “the method”; they are treated as part of the baseline space. Thinking-enabled generation is treated as a later optional extension rather than a Stage 0 requirement.



### RQ1. Data / Training Question

Can **verifiable financial numerical reasoning data** improve FinQA performance through lightweight post-training?

This question has two possible routes:

- using **existing     related datasets** such as ConvFinQA and MultiHiertt as augmented     sources, or
- constructing     **new FinQA-like verifiable data** from raw financial text via     retrieval, generation, and filtering.

ConvFinQA extends financial numerical reasoning into conversational chains, while MultiHiertt adds harder multi-step reasoning over multiple hierarchical tables and text, making them natural extensions rather than random benchmark additions. ([ACL Anthology](https://aclanthology.org/2022.emnlp-main.421/))



### RQ2. Sample Efficiency Question

How little data is needed to obtain meaningful improvement on FinQA?

This becomes a clean ablation question:

- 0 /     200 / 500 / 1k / 2k / 5k samples, or a similar progression depending on     compute.



### RQ3. Robustness / No-Forgetting Question

Can we improve FinQA without harming broader reasoning performance on external benchmarks?

We use:

- **MATH** as a math reasoning benchmark,     introduced as a 12,500-problem dataset with step-by-step solutions, and
- **GPQA** as a difficult graduate-level     “Google-proof” science QA benchmark with 448 expert-written     multiple-choice questions. ([NeurIPS 数据集基准](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html))



### Optional RQ4. RL Question

If the training targets are verifiable, does **RL with verifiable reward** provide extra gains beyond SFT?

This is optional. If time allows, we can use TRL’s GRPO framework with reward based on correctness and output-format compliance. TRL officially supports both SFT training and GRPO training flows. ([Hugging Face](https://huggingface.co/docs/trl/main/en/sft_trainer))





## 4. Main Hypotheses

### H0

Thinking-enabled baselines are **not guaranteed** to outperform thinking-disabled baselines once runtime budget, truncation, and output-format compliance are controlled.



### H1

A meaningful part of current FinQA error comes from unstable scale handling and answer formatting, so stronger baseline control plus robust verification will change the measured error profile.



### H2

Low-data post-training on verifiable finance-reasoning data will improve FinQA performance, especially on percent/ratio-sensitive errors.



### H3

Performance gains will show diminishing returns as training data size increases.



### H4

Naive finance-only fine-tuning may hurt out-of-domain reasoning, but careful adapter-based tuning and/or small anchor mixing can reduce that degradation.



### H5 (optional)

GRPO on verifiable rewards may improve over plain SFT, but this will be higher-risk and is therefore treated as a stretch goal.





## 5. Dataset Strategy

### 5.1 Primary Dataset: FinQA

FinQA is the core benchmark because it was designed precisely for numerical reasoning over financial reports and includes gold reasoning programs, which makes it ideal for both evaluation and potentially verifiable supervision. ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.300/))

We will keep two input settings:

- **oracle**: gold evidence only
- **full**: full context with distractors

These two settings are important because your existing results already show that distractor numbers matter.



### 5.2 Extension Dataset 1: ConvFinQA

ConvFinQA is a natural extension because it keeps the finance numerical reasoning setting but shifts it into a conversational form. This makes it useful both as:

- a possible auxiliary training source,
- and an external transfer benchmark after training. ([ACL Anthology](https://aclanthology.org/2022.emnlp-main.421/))



### 5.3 Extension Dataset 2: MultiHiertt

MultiHiertt is also relevant because it makes the reasoning setting harder in a specific way: multiple tables, hierarchical tables, longer text, and more complex reasoning processes. It is therefore appropriate as:

- an optional harder augmentation source,
- or a transfer evaluation source. ([ACL Anthology](https://aclanthology.org/2022.acl-long.454/))



### 5.4 Optional New Data Construction

Following TA’s suggestion, we may also construct a **small verifiable synthetic dataset**:

- retrieve financial passages similar to FinQA-style inputs,
- generate question–answer–reasoning triples,
- and keep only those examples whose answers can be automatically     verified.

This “retrieval → generation → verification” route would be our clearest data novelty component if we have enough time.



### Scope rule

To avoid scope explosion:

- FinQA remains the **primary     benchmark**,
- ConvFinQA and MultiHiertt     are **extensions**, not equal co-primary datasets,
- and synthetic data     construction is **recommended but minimal** rather than huge by     default.





## 6. Model Strategy

### 6.1 Student / Target Models

We will use the Qwen-family models already explored in the baseline stage as the student models for controlled comparison.



### 6.2 Baseline and training policy

The proposal explicitly distinguishes baseline analysis from training default:

- **current Stage 0 scope:** thinking=false baselines only;
- **default SFT training path:** answer-only supervision with strict tagged output (`[FINAL_ANSWER]...[/FINAL_ANSWER]`), typically with thinking disabled for efficiency and stability;
- **thinking=true** remains a controlled ablation and optional advanced branch on smaller pilot subsets.

This preserves the value of your completed work while keeping the main training loop feasible.



### 6.3 Teacher model for data generation

If we perform synthetic-data construction, a larger open instruction model is a plausible teacher choice. Qwen2.5-32B-Instruct is a realistic candidate because it is publicly available, instruction-tuned, and designed for structured outputs and long context. ([Hugging Face](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct))

We are **not** claiming it must be used; only that it is a sensible default teacher candidate if compute permits.





## 7. Evaluation Design

### 7.1 Baseline factors

The baseline space (Stage 0 default) will include:

- thinking=false only
- short / medium / long generation budget
- numeric-only output vs tagged output
- oracle vs full context
- model size comparisons where feasible



### 7.2 Output format control

We will compare at least:

- unrestricted numeric answer prompt
- strict tagged output, e.g. FINAL_ANSWER: <number>

This is not treated as the main method; it is treated as part of the baseline control space.



### 7.3 Verification upgrade: Math-Verify

A major change after TA feedback is replacing brittle string matching with **Math-Verify**, which supports robust answer extraction and mathematical equivalence checking across expressions such as 1/2 and 0.5, handles percentage conversion best-effort, and is designed to avoid underestimating model performance due to formatting differences. ([GitHub](https://github.com/huggingface/Math-Verify))

This will become the **primary evaluator** wherever applicable.



### 7.4 Core metrics

We will report:

- verified accuracy,
- parse/tag success rate,
- oracle vs full gap,
- model-wise error breakdown,
- and efficiency metrics (tokens per sample, seconds per sample, and end-to-end wall-clock).



### 7.5 Diagnostic metrics

We will retain your current strengths by reporting:

- percent-scale mistakes,
- wrong-number selection,
- sign and magnitude errors,
- and optional oracle-in-output upper bound diagnostics.

This helps separate:

- real reasoning failures,
- extraction/format artifacts,
- and scale-expression instability.



### 7.6 Robustness metrics

Before and after fine-tuning, we will evaluate on:

- MATH,
- GPQA.

We will summarize the trade-off between FinQA gain and any out-of-domain loss. ([NeurIPS 数据集基准](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html))





### 7.7 Success criteria

To keep claims concrete, we define target thresholds:

- primary target: FinQA verified accuracy improves by at least **+3 points** over the chosen baseline;
- robustness target: MATH/GPQA drop is at most **1–2 points**;
- engineering target: chosen training recipe must complete within the team’s scheduled compute window.

If these criteria are not met, we report negative findings and prefer simpler, better-controlled settings over adding extra method complexity.

## 8. Method Plan



## Stage 0 — Baseline Rebuild and Diagnostic Upgrade

This stage is now mandatory and foundational.

We will:

1. run a controlled no-thinking baseline matrix over generation budgets;
2. identify a practical default for SFT (quality + runtime) under thinking=false;
3. introduce output-format control;
4. replace string match with Math-Verify;
5. update the error taxonomy under the new evaluator.

**Main goal:** establish the real baseline that later training must beat.

**Expected output:**

- baseline comparison tables,
- updated taxonomy,
- and a justified choice of which model/setup becomes the     post-training target.


## Stage 1A — Data Preparation

This stage prepares training data for SFT.

We will use one or more of:

- FinQA train split,
- filtered/verifiable subsets from ConvFinQA,
- filtered/verifiable subsets from MultiHiertt,
- optionally a small synthetic FinQA-like set built by retrieval     → generation → verification.

The key principle is:

training data should be as **verifiable** as possible.

That matters because it strengthens both the SFT story and the optional RL story.





## Stage 1B / 1C — SFT and Data-Size Ablation

We will run SFT using the TRL SFTTrainer, which supports supervised fine-tuning of causal language models and is appropriate for adapter-based workflows. ([Hugging Face](https://huggingface.co/docs/trl/main/en/sft_trainer))

We will likely use:

- LoRA or QLoRA,
- stable hyperparameters across ablations,
- a single target model at first to keep compute manageable.
- answer-only supervision with strict final-answer tags as the default target format.

The main experiment is:

- train with progressively larger dataset sizes,
- evaluate FinQA gain,
- analyze which error buckets improve,
- and compare a small thinking=true branch only if the default path underperforms.

**Main question:** how much verifiable data is enough?





## Stage 2 — Robustness / No-Forgetting

We will evaluate the pre- and post-fine-tuning model on:

- MATH,
- GPQA. ([NeurIPS 数据集基准](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html))

If degradation appears, we will test a minimal mitigation, such as:

- smaller LoRA rank,
- more conservative learning rate,
- early stopping,
- or mixing a small anchor set.

The point is not to build a whole anti-forgetting framework, but to make sure our FinQA gain is not a misleading narrow overfit.





## Stage 3 — Optional RL with Verifiable Reward

If Stage 1 is stable and time remains, we will try a small GRPO experiment using TRL’s GRPO framework. Reward will be based on:

- correctness under verification,
- and output-format compliance. ([GitHub](https://github.com/huggingface/Math-Verify))

This stage is explicitly optional.





## 9. Team Division and Responsibilities

This section is fully aligned with your uploaded team-division document. The project is organized around six distinct roles and stage ownerships.

### Member 1 — Project Lead

- Leads     Stage 0 overall coordination
- Runs     thinking=true / thinking=false baseline comparisons
- Maintains     experiment logs and stage milestones
- Coordinates     the team and integrates the final report



### Member 2 — Evaluation Engineer

- Co-leads     Stage 0 evaluation infrastructure
- Integrates     Math-Verify and replaces string matching
- Designs     the FINAL_ANSWER output standard
- Maintains     the error taxonomy and evaluation scripts



### Member 3 — Data Engineer

- Leads     Stage 1A data construction
- Organizes     ConvFinQA / MultiHiertt augmentation sources
- Builds     the retrieval → generation → filtering pipeline
- Produces     the verifiable synthetic dataset if that route is used



### Member 4 — Training Engineer

- Leads     Stage 1B / 1C SFT implementation
- Uses     TRL SFTTrainer
- Configures     LoRA / QLoRA for feasible training
- Runs     the 0 / 200 / 500 / 1k / 2k / 5k data-size ablations



### Member 5 — Robustness Researcher

- Leads     Stage 2 no-forgetting evaluation
- Runs     pre/post tuning evaluation on MATH and GPQA
- Analyzes     catastrophic forgetting
- Produces     the FinQA↑ vs MATH/GPQA↔ trade-off table



### Member 6 — RL Researcher

- Leads     optional Stage 3 GRPO exploration
- Starts     only after Stage 1 is stable
- Designs     correctness + format-based reward
- Produces     the SFT vs RL comparison if time permits



### Shared responsibilities

- weekly     progress sync,
- shared     experiment log updates,
- common use     of the Math-Verify evaluation infrastructure,
- close     collaboration between data preparation and training formatting,
- report     writing by stage, with final integration by Member 1.





## 10. Timeline and Stage Ownership

This also follows your uploaded team plan.

### Stage 0

**Content:** baseline building with thinking / format / Math-Verify
 **Lead:** Member 1 + Member 2
 **Key deliverables:** baseline tables + error analysis report



### Stage 1A

**Content:** data construction from ConvFinQA / MultiHiertt / synthetic retrieval-generation-filter pipeline
 **Lead:** Member 3
 **Key deliverable:** verifiable training set



### Stage 1B / 1C

**Content:** answer-only SFT with LoRA and data-size ablation (thinking=false default, thinking=true optional branch)
 **Lead:** Member 4
 **Key deliverable:** performance-vs-data-size curve



### Stage 2

**Content:** robustness testing on MATH / GPQA
 **Lead:** Member 5
 **Key deliverable:** trade-off table



### Stage 3 (optional)

**Content:** GRPO RL with verifiable reward
 **Lead:** Member 6
 **Key deliverable:** SFT vs RL comparison





## 11. Key Deliverables

By the end of the project, the intended outputs are:

1. a reproducible **evaluation pipeline** with mode-aware baselines and Math-Verify;
2. a **baseline diagnostic report** with updated taxonomy;
3. a **verifiable training dataset** assembled from extension     data and/or synthetic generation;
4. an **SFT data-size ablation curve**;
5. a **robustness trade-off table** showing whether FinQA gains     preserve broader reasoning;
6. a **mode-and-runtime decision memo** documenting why the final SFT recipe is chosen;
7. optionally, an **SFT vs RL comparison**.





## 12. Risks and Mitigations

### Risk 1: Measured improvements are actually evaluator artifacts

**Mitigation:** use Math-Verify, tagged output formats, and explicit parse/tag reporting. ([GitHub](https://github.com/huggingface/Math-Verify))



### Risk 2: ConvFinQA / MultiHiertt increase engineering complexity too much

**Mitigation:** treat them first as extension sources and filtered subsets, not full co-primary pipelines.



### Risk 3: Fine-tuning improves FinQA but hurts general reasoning

**Mitigation:** run MATH/GPQA before-and-after checks and use conservative adapter settings. ([NeurIPS 数据集基准](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html))



### Risk 4: RL consumes too much time

**Mitigation:** keep RL strictly optional and dependent on Stage 1 stability.


### Risk 5: thinking-enabled training/inference causes prohibitive runtime

**Mitigation:** keep answer-only thinking=false as the default SFT path, and run thinking=true only as a bounded pilot or ablation.



### Risk 6: Team coordination drift

**Mitigation:** retain weekly sync and a shared experiment log, exactly as already specified in the team division plan.





## 13. Decision Gates

### Gate A — before large-scale training

We do **not** expand training until:

- no-thinking baseline matrix is completed for budget and format controls,
- Math-Verify is integrated,
- output-format control is tested,
- and the target post-training setup is clearly chosen with runtime evidence.



### Gate B — before adding full extension scope

We do **not** over-expand ConvFinQA / MultiHiertt usage until:

- SFT on the core FinQA-centered setup is stable,
- data formatting is working,
- and one complete ablation loop has finished.



### Gate C — before RL

We do **not** start GRPO until:

- baseline is     stable,
- SFT has run     successfully,
- and there is     time left for a fair comparison.





## 14. Expected Final Paper Storyline

The final report should tell one clear story:

1. **Problem setup:** financial numerical     reasoning is hard because it combines heterogeneous evidence and     multi-step arithmetic reasoning. FinQA is the main controlled benchmark,     and ConvFinQA / MultiHiertt serve as relevant extensions. ([ACL Anthology](https://aclanthology.org/2021.emnlp-main.300/))
2. **Baseline correction:** we first establish a no-thinking baseline space with robust mathematical verification, then choose a compute-feasible SFT default. ([GitHub](https://github.com/huggingface/Math-Verify))
3. **Error diagnosis:** the task is not     saturated, and the key failures concern scale, percentages, and number     selection rather than trivial parse issues.
4. **Low-data post-training:** verifiable     data can improve FinQA, and we quantify how much data is needed.
5. **Robustness:** those gains are only     meaningful if they do not substantially damage broader reasoning     performance on MATH and GPQA. ([NeurIPS 数据集基准](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html))
6. **Optional extension:** RL with     verifiable reward may add further gains, but only after the simpler story     is complete. ([GitHub](https://github.com/huggingface/Math-Verify))

 
