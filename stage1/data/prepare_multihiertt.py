from __future__ import annotations

import json
import os

import jsonlines


def prepare_multihiertt(input_path: str, output_path: str, dataset_name: str = "multihiertt") -> None:
    """Convert MultiHiertt JSON format to the unified JSONL schema."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for item in data:
            uid = item.get("uid", "")
            paragraphs = item.get("paragraphs", [])
            tables = item.get("tables", [])
            qa = item.get("qa", {})

            context_parts = []
            if paragraphs:
                context_parts.append("\n".join(paragraphs))
            if tables:
                context_parts.append("\n".join(tables))
            context = "\n\n".join(context_parts)

            writer.write(
                {
                    "id": uid,
                    "question": qa.get("question", ""),
                    "context": context,
                    "answer": str(qa.get("answer", "")),
                    "source_dataset": dataset_name,
                    "program": qa.get("program", []),
                    "evidence_indices": {
                        "text": qa.get("text_evidence", []),
                        "table": qa.get("table_evidence", []),
                    },
                }
            )


if __name__ == "__main__":
    prepare_multihiertt("raw_data/multihiertt/train.json", "processed/multihiertt_train.jsonl")
    prepare_multihiertt("raw_data/multihiertt/dev.json", "processed/multihiertt_dev.jsonl")
