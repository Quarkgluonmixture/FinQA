import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml


def parse_args() -> argparse.Namespace:
    """
    输入:
        无
    输出:
        argparse.Namespace
            - base_config: str, 基础配置文件路径
            - sizes: List[int], 需要生成的训练数据规模
            - output_dir: str, 输出目录
    """
    parser = argparse.ArgumentParser(description="Generate data-size ablation config files.")
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/full.yaml",
        help="Base yaml config to clone.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[0, 200, 500, 1000, 2000, 5000],
        help="Subset sizes for ablation configs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs",
        help="Directory to save generated yaml files.",
    )
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


def save_yaml_config(config: Dict[str, Any], save_path: Path) -> None:
    """
    输入:
        config: Dict[str, Any], 配置字典
        save_path: Path, 输出路径
    输出:
        无
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def build_ablation_config(base_config: Dict[str, Any], subset_size: int) -> Dict[str, Any]:
    """
    输入:
        base_config: Dict[str, Any], 基础配置
        subset_size: int, 当前消融训练样本数
    输出:
        Dict[str, Any], 新配置
    """
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
    """
    输入:
        无
    输出:
        无
    """
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