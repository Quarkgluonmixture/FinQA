from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import jsonlines


def prepare_finqa(input_path: str, output_path: str, dataset_name: str = "finqa") -> None:
    """Convert FinQA JSON format to the unified JSONL schema."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for idx, item in enumerate(data):
            example_id = item.get("id", f"{dataset_name}_{idx}")
            pre_text = item.get("pre_text", [])
            post_text = item.get("post_text", [])
            table = item.get("table", [])

            qa = item.get("qa", {})
            question = qa.get("question", "")
            program = qa.get("program", [])
            gold_inds = qa.get("gold_inds", [])
            exe_ans = qa.get("exe_ans", "")

            pre_text_str = "\n".join(pre_text) if isinstance(pre_text, list) else str(pre_text)
            post_text_str = "\n".join(post_text) if isinstance(post_text, list) else str(post_text)

            table_str = ""
            if isinstance(table, list):
                for row in table:
                    if isinstance(row, list):
                        table_str += " | ".join(str(cell) for cell in row) + "\n"
                    else:
                        table_str += str(row) + "\n"
            else:
                table_str = str(table)

            context_parts: List[str] = []
            if pre_text_str:
                context_parts.append(pre_text_str)
            if table_str:
                context_parts.append(table_str)
            if post_text_str:
                context_parts.append(post_text_str)
            context = "\n\n".join(context_parts)

            writer.write(
                {
                    "id": example_id,
                    "question": question,
                    "context": context,
                    "answer": str(exe_ans),
                    "source_dataset": dataset_name,
                    "program": program,
                    "evidence_indices": gold_inds,
                }
            )


if __name__ == "__main__":
    prepare_finqa("raw_data/finqa/train.json", "processed/finqa_train.jsonl")
    prepare_finqa("raw_data/finqa/dev.json", "processed/finqa_dev.jsonl")
