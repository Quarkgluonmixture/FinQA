import json
import jsonlines
import os

def prepare_multihiertt(input_path, output_path, dataset_name="multihiertt"):
    """
    读取 MultiHiertt 原始 JSON 文件，转换为统一格式的 JSONL。
    每个样本输出格式：
    {
        "id": str (uid),
        "question": str,
        "context": str (段落文本 + 表格HTML),
        "answer": str,
        "source_dataset": str,
        "program": list,
        "evidence_indices": {"text": [...], "table": [...]}
    }
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        for item in data:
            uid = item.get("uid", "")
            paragraphs = item.get("paragraphs", [])
            tables = item.get("tables", [])
            qa = item.get("qa", {})
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            program = qa.get("program", [])
            text_evidence = qa.get("text_evidence", [])
            table_evidence = qa.get("table_evidence", [])

            # 构建 context：段落文本 + 表格HTML
            context_parts = []
            if paragraphs:
                # 段落按行连接
                context_parts.append("\n".join(paragraphs))
            if tables:
                # 每个表格保留原始 HTML 字符串，用换行分隔
                context_parts.append("\n".join(tables))
            context = "\n\n".join(context_parts)

            # 合并两种证据索引，保留类型信息
            evidence_indices = {
                "text": text_evidence,
                "table": table_evidence
            }

            unified_sample = {
                "id": uid,
                "question": question,
                "context": context,
                "answer": str(answer),
                "source_dataset": dataset_name,
                "program": program,
                "evidence_indices": evidence_indices
            }
            writer.write(unified_sample)

if __name__ == "__main__":
    # 处理训练集和验证集
    prepare_multihiertt("raw_data/multihiertt/train.json", "processed/multihiertt_train.jsonl")
    prepare_multihiertt("raw_data/multihiertt/dev.json", "processed/multihiertt_dev.jsonl")