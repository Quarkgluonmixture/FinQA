from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ablation config generation."""
    parser = argparse.ArgumentParser(description="Generate data-size ablation config files.")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/full.yaml",
        help="Base YAML config to clone.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[0, 200, 500, 1000, 2000, 5000],
        help="Subset sizes for generated configs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs",
        help="Directory to save generated YAML files.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML config: {config_path}")
    return config


def save_yaml_config(config: Dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def build_ablation_config(base_config: Dict[str, Any], subset_size: int) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)

    run_name = f"ablation_{subset_size}"
    output_dir = f"outputs/{run_name}"

    config["run_name"] = run_name

    config.setdefault("data", {})
    config["data"]["max_train_samples"] = subset_size

    config.setdefault("training", {})
    config["training"]["output_dir"] = output_dir

    config.setdefault("inference", {})
    config["inference"]["checkpoint_dir"] = f"{output_dir}/checkpoint-last"
    config["inference"]["output_file"] = f"{output_dir}/infer_predictions.jsonl"
    config["inference"]["summary_file"] = f"{output_dir}/infer_summary.json"

    config.setdefault("logging", {})
    config["logging"]["log_dir"] = f"{output_dir}/logs"
    config["logging"]["train_log_file"] = f"{output_dir}/logs/train.log"
    config["logging"]["infer_log_file"] = f"{output_dir}/logs/infer.log"

    return config


def main() -> None:
    args = parse_args()
    base_config = load_yaml_config(args.base_config)
    output_dir = Path(args.output_dir)

    for subset_size in args.sizes:
        ablation_config = build_ablation_config(base_config, subset_size)
        save_path = output_dir / f"ablation_{subset_size}.yaml"
        save_yaml_config(ablation_config, save_path)
        print(f"[Info] Generated: {save_path}")


if __name__ == "__main__":
    main()
