import json
import jsonlines
import os

def prepare_convfinqa(input_path, output_path, dataset_name="convfinqa"):
    """
    读取 ConvFinQA 对话级别 JSON 文件（conversation level），
    将每个对话中的每个问题轮次转换为一个统一格式的样本。
    输出 JSONL 格式，每个样本包含：
        id: 原始对话id + "_turn_" + 轮次索引
        question: 当前轮次的问题文本
        context: 拼接的 pre_text + table + post_text
        answer: 当前轮次的执行答案 (exe_ans_list[i])
        source_dataset: 数据集名称
        program: 当前轮次的程序 (turn_program[i])，若无则为空列表
        evidence_indices: None（ConvFinQA 不提供证据索引）
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for idx, item in enumerate(data):
            # 提取基础字段
            example_id = item.get("id", f"{dataset_name}_{idx}")
            pre_text = item.get("pre_text", [])
            post_text = item.get("post_text", [])
            table = item.get("table", [])

            # 构建 context：拼接 pre_text、table、post_text
            pre_text_str = "\n".join(pre_text) if isinstance(pre_text, list) else str(pre_text)
            post_text_str = "\n".join(post_text) if isinstance(post_text, list) else str(post_text)

            # 表格转换：每行用 | 分隔单元格，行间换行
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

            # 提取 annotation 字段
            annotation = item.get("annotation", {})
            dialogue_break = annotation.get("dialogue_break", [])   # 问题列表
            turn_program = annotation.get("turn_program", [])       # 每个问题的程序
            exe_ans_list = annotation.get("exe_ans_list", [])       # 每个问题的答案

            # 确保三个列表长度一致，否则取最小长度
            min_len = min(len(dialogue_break), len(turn_program), len(exe_ans_list))

            # 为每个轮次生成一个样本
            for turn_idx in range(min_len):
                question_text = dialogue_break[turn_idx]
                program = turn_program[turn_idx] if turn_program[turn_idx] is not None else []
                answer = str(exe_ans_list[turn_idx]) if exe_ans_list[turn_idx] is not None else ""

                unified_sample = {
                    "id": f"{example_id}_turn_{turn_idx}",
                    "question": question_text,
                    "context": context,
                    "answer": answer,
                    "source_dataset": dataset_name,
                    "program": program,
                    "evidence_indices": None   # ConvFinQA 没有提供证据索引
                }
                writer.write(unified_sample)

if __name__ == "__main__":
    # 处理训练集和验证集
    prepare_convfinqa("raw_data/convfinqa/train.json", "processed/convfinqa_train.jsonl")
    prepare_convfinqa("raw_data/convfinqa/dev.json", "processed/convfinqa_dev.jsonl")