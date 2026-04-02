import json
import jsonlines
import os

def prepare_finqa(input_path, output_path, dataset_name="finqa"):
    """
    读取 FinQA 原始 JSON 文件，转换为统一格式的 JSONL。
    每个样本输出格式：
    {
        "id": "finqa_xxx",
        "question": str,
        "context": str (拼接 pre_text, table, post_text),
        "answer": str (exe_ans),
        "source_dataset": str,
        "program": list,
        "evidence_indices": list (gold_inds)
    }
    """
    # 确保输入文件存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for idx, item in enumerate(data):
            # 提取基础字段
            example_id = item.get("id", f"{dataset_name}_{idx}")
            pre_text = item.get("pre_text", [])
            post_text = item.get("post_text", [])
            table = item.get("table", [])  # 表格通常是列表的列表

            # 提取 qa 部分
            qa = item.get("qa", {})
            question = qa.get("question", "")
            program = qa.get("program", [])          # 程序列表
            gold_inds = qa.get("gold_inds", [])      # 证据索引
            exe_ans = qa.get("exe_ans", "")          # 执行结果，可能是数字或字符串

            # 构建 context：将 pre_text、table、post_text 拼接
            # 处理 pre_text 和 post_text 为字符串列表的情况
            pre_text_str = "\n".join(pre_text) if isinstance(pre_text, list) else str(pre_text)
            post_text_str = "\n".join(post_text) if isinstance(post_text, list) else str(post_text)

            # 将表格转换为可读字符串（每个单元格用空格分隔，行之间换行）
            table_str = ""
            if isinstance(table, list):
                for row in table:
                    if isinstance(row, list):
                        table_str += " | ".join(str(cell) for cell in row) + "\n"
                    else:
                        table_str += str(row) + "\n"
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

            # 统一格式样本
            unified_sample = {
                "id": example_id,
                "question": question,
                "context": context,
                "answer": str(exe_ans),
                "source_dataset": dataset_name,
                "program": program,
                "evidence_indices": gold_inds
            }
            writer.write(unified_sample)

if __name__ == "__main__":
    # 示例：处理训练集和验证集
    prepare_finqa("raw_data/finqa/train.json", "processed/finqa_train.jsonl")
    prepare_finqa("raw_data/finqa/dev.json", "processed/finqa_dev.jsonl")