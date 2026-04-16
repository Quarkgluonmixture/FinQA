from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

LEGACY_DRY_RUN_MODE = "smoke_echo_gold"
DRY_RUN_MODE = "dry_run_echo_reference"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Post-training dry-run inference entrypoint.")
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return config


def load_json_or_jsonl(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON file must contain a list: {data_path}")

    raise ValueError(f"Unsupported data format: {data_path}")


def save_jsonl(rows: List[Dict[str, Any]], save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], save_path: str) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def normalize_mode(mode: str) -> str:
    mode = str(mode or "").strip()
    if mode == LEGACY_DRY_RUN_MODE:
        print(f"[Warning] '{LEGACY_DRY_RUN_MODE}' is deprecated; using '{DRY_RUN_MODE}'.")
        return DRY_RUN_MODE
    return mode or DRY_RUN_MODE


def build_dry_run_prediction(sample: Dict[str, Any]) -> Dict[str, str]:
    """Build a deterministic dry-run prediction for I/O pipeline checks."""
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
    args = parse_args()
    config = load_yaml_config(args.config)

    inference_cfg = config.get("inference", {})
    checkpoint_dir = Path(args.checkpoint_dir or inference_cfg.get("checkpoint_dir", "outputs/debug_run/checkpoint-last"))
    input_file = args.input_file or inference_cfg.get("input_file")
    output_file = args.output_file or inference_cfg.get("output_file", "outputs/debug_run/infer_predictions.jsonl")
    summary_file = inference_cfg.get("summary_file", "outputs/debug_run/infer_summary.json")
    max_samples = int(inference_cfg.get("max_samples", 3))
    infer_mode = normalize_mode(inference_cfg.get("mode", DRY_RUN_MODE))

    if infer_mode != DRY_RUN_MODE:
        raise ValueError(f"Unsupported inference mode: {infer_mode}")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    if not input_file:
        raise ValueError("No inference input file provided.")

    samples = load_json_or_jsonl(input_file)[:max_samples]

    prediction_rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        dry_run_result = build_dry_run_prediction(sample)
        prediction_rows.append(
            {
                "sample_index": index,
                "id": sample.get("id", f"sample_{index}"),
                "source_dataset": sample.get("source_dataset", "unknown"),
                "question": sample.get("question", ""),
                "prediction": dry_run_result["prediction"],
                "reference_answer": sample.get("answer", None),
                "prediction_source": dry_run_result["prediction_source"],
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
