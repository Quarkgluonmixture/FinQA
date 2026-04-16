from __future__ import annotations

import json
import os

import jsonlines


def prepare_convfinqa(input_path: str, output_path: str, dataset_name: str = "convfinqa") -> None:
    """Convert ConvFinQA conversation-level JSON into unified JSONL rows."""
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

            pre_text_str = "\n".join(pre_text) if isinstance(pre_text, list) else str(pre_text)
            post_text_str = "\n".join(post_text) if isinstance(post_text, list) else str(post_text)

            if isinstance(table, list):
                table_str = "\n".join(" | ".join(str(cell) for cell in row) for row in table if isinstance(row, list))
            else:
                table_str = str(table)

            context_parts = []
            if pre_text_str:
                context_parts.append(pre_text_str)
            if table_str:
                context_parts.append(table_str)
            if post_text_str:
                context_parts.append(post_text_str)
            context = "\n\n".join(context_parts)

            annotation = item.get("annotation", {})
            dialogue_break = annotation.get("dialogue_break", [])
            turn_program = annotation.get("turn_program", [])
            exe_ans_list = annotation.get("exe_ans_list", [])

            min_len = min(len(dialogue_break), len(turn_program), len(exe_ans_list))

            for turn_idx in range(min_len):
                question_text = dialogue_break[turn_idx]
                program = turn_program[turn_idx] if turn_program[turn_idx] is not None else []
                answer = str(exe_ans_list[turn_idx]) if exe_ans_list[turn_idx] is not None else ""

                writer.write(
                    {
                        "id": f"{example_id}_turn_{turn_idx}",
                        "question": question_text,
                        "context": context,
                        "answer": answer,
                        "source_dataset": dataset_name,
                        "program": program,
                        "evidence_indices": None,
                    }
                )


if __name__ == "__main__":
    prepare_convfinqa("raw_data/convfinqa/train.json", "processed/convfinqa_train.jsonl")
    prepare_convfinqa("raw_data/convfinqa/dev.json", "processed/convfinqa_dev.jsonl")
