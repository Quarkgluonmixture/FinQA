# Stage1 Data Notes

This directory documents the unified JSONL schema and the repository data policy.

## Unified Record Schema

Each JSONL row follows this structure:

- `id` (string): sample identifier
- `question` (string): question text
- `context` (string): combined text and table context
- `answer` (string): gold answer string
- `source_dataset` (string): one of `finqa`, `convfinqa`, `multihiertt`
- `program` (list): program representation when available
- `evidence_indices` (object/list/null): evidence pointers when available

## Tracked vs Regenerated Data

Tracked minimal assets:

- `data/debug/train.jsonl`
- `data/finqa_clean/train_debug_50.jsonl`
- subset ID lists under `data/finqa_clean/` and `data/finqa_clean_consistent_or_corrected/`
- cleaning summaries (`clean_summary.json`)

Regenerated and not tracked:

- `data/unified/`
- full clean split JSONL files (`train_full`, `train_250`, `train_1000`, `dev_full`)

## Regeneration

Use these scripts when you need to rebuild local data artifacts:

```bash
python stage1/data/prepare_finqa.py
python stage1/data/prepare_convfinqa.py
python stage1/data/prepare_multihiertt.py
python stage1/data/merge_datasets.py
python stage1/scripts/build_finqa_clean_splits.py
```
