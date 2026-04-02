import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    """
    输入:
        无
    输出:
        argparse.Namespace
            - config: str, 配置文件路径
            - checkpoint_dir: str | None, 手动覆盖 checkpoint 路径
            - input_file: str | None, 手动覆盖推理输入文件
            - output_file: str | None, 手动覆盖推理输出文件
    """
    parser = argparse.ArgumentParser(description="Minimal post-training inference check.")
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    输入:
        config_path: str, yaml 配置文件路径
    输出:
        Dict[str, Any], 配置字典
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid yaml config: {config_path}")
    return config


def load_json_or_jsonl(data_path: str) -> List[Dict[str, Any]]:
    """
    输入:
        data_path: str, 数据文件路径
    输出:
        List[Dict[str, Any]], 样本列表
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON file must contain a list: {data_path}")

    raise ValueError(f"Unsupported data format: {data_path}")


def save_jsonl(rows: List[Dict[str, Any]], save_path: str) -> None:
    """
    输入:
        rows: List[Dict[str, Any]], 预测结果列表
        save_path: str, 输出 jsonl 路径
    输出:
        无
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], save_path: str) -> None:
    """
    输入:
        data: Dict[str, Any], 字典数据
        save_path: str, 输出 json 路径
    输出:
        无
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def build_smoke_prediction(sample: Dict[str, Any]) -> Dict[str, str]:
    """
    输入:
        sample: Dict[str, Any], 单条输入样本
    输出:
        Dict[str, str]
            - prediction: str, 预测文本
            - prediction_source: str, 预测来源说明

    说明:
        当前是最小 smoke 版本，不做真实模型生成。
        目的是先验证：
        checkpoint 存在 -> 可读输入 -> 可写输出 -> 端到端脚本可跑通。
    """
    if sample.get("answer") is not None:
        return {
            "prediction": str(sample.get("answer")),
            "prediction_source": "echo_answer_field",
        }

    if sample.get("output") is not None:
        return {
            "prediction": str(sample.get("output")),
            "prediction_source": "echo_output_field",
        }

    if sample.get("final_answer") is not None:
        return {
            "prediction": str(sample.get("final_answer")),
            "prediction_source": "echo_final_answer_field",
        }

    return {
        "prediction": "",
        "prediction_source": "empty_stub",
    }


def main() -> None:
    """
    输入:
        无
    输出:
        无
    """
    args = parse_args()
    config = load_yaml_config(args.config)

    inference_cfg = config.get("inference", {})
    checkpoint_dir = Path(args.checkpoint_dir or inference_cfg.get("checkpoint_dir", "outputs/debug_run/checkpoint-last"))
    input_file = args.input_file or inference_cfg.get("input_file")
    output_file = args.output_file or inference_cfg.get("output_file", "outputs/debug_run/infer_predictions.jsonl")
    summary_file = inference_cfg.get("summary_file", "outputs/debug_run/infer_summary.json")
    max_samples = int(inference_cfg.get("max_samples", 3))
    infer_mode = inference_cfg.get("mode", "smoke_echo_gold")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    if not input_file:
        raise ValueError("No inference input file provided.")

    samples = load_json_or_jsonl(input_file)
    samples = samples[:max_samples]

    prediction_rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        smoke_result = build_smoke_prediction(sample)

        prediction_rows.append(
            {
                "sample_index": index,
                "id": sample.get("id", f"sample_{index}"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "question": sample.get("question", ""),
                "prediction": smoke_result["prediction"],
                "reference_answer": sample.get("answer", None),
                "prediction_source": smoke_result["prediction_source"],
                "checkpoint_dir": str(checkpoint_dir),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    save_jsonl(prediction_rows, output_file)

    summary = {
        "config_path": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "input_file": input_file,
        "output_file": output_file,
        "num_examples": len(samples),
        "mode": infer_mode,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(summary, summary_file)

    print(f"[Info] Inference input loaded from: {input_file}")
    print(f"[Info] Checkpoint dir: {checkpoint_dir}")
    print(f"[Info] Prediction file saved to: {output_file}")
    print(f"[Info] Summary file saved to: {summary_file}")


if __name__ == "__main__":
    main()