# Unified Data Format for FinReason Project

## Overview

This document describes the unified JSONL format used for all datasets in the FinReason project.  
Each line in the JSONL file is a JSON object with the following fields.

## Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier of the sample. For FinQA and ConvFinQA, it may contain the original dataset ID; for MultiHiertt, it is the `uid`. |
| `question` | string | The question (or concatenated questions for multi‑turn dialogues). |
| `context` | string | The full document context (text + tables) from which the answer can be derived. Constructed from the original pre_text, table, post_text (FinQA/ConvFinQA) or paragraphs + tables HTML (MultiHiertt). |
| `answer` | string | The gold answer (numeric or text). For multi‑turn dialogues, answers are concatenated with newline. |
| `source_dataset` | string | Dataset origin: `"finqa"`, `"convfinqa"`, or `"multihiertt"`. |
| `program` | list | The reasoning program. For FinQA it is a flat list of operations; for ConvFinQA it is a list of programs (one per turn); for MultiHiertt it is a flat list. May be empty if not available. |
| `evidence_indices` | object / list / null | Indices of supporting evidence. - **FinQA**: list of integers (`gold_inds`). <br> - **ConvFinQA**: always `null`. <br> - **MultiHiertt**: object with `"text"` and `"table"` keys, each holding a list of indices. |

## Usage Notes

- The `context` field contains the **raw document** (text + tables) as provided by the original dataset. For Oracle evidence construction, the `evidence_indices` field should be used to extract the relevant portions.
- For ConvFinQA, the `question` and `answer` fields concatenate all turns. This preserves the full conversational context.
- `program` is kept in its original representation (list of operations) to support execution evaluation.
- All answer values are stored as strings to maintain consistency across datasets.

## Files

- `train.jsonl`: Merged training set from all three datasets.
- `dev.jsonl`: Merged validation set.
- `debug.jsonl`: First 100 samples of the training set for quick experimentation.

## Generation

To regenerate these files, run:
```bash
python prepare_finqa.py
python prepare_convfinqa.py
python prepare_multihiertt.py
python merge_datasets.py