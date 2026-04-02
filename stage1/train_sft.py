import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.loaders import load_data
from src.preprocessing.preprocess import preprocess_data
from src.training.trainer_sft import build_trainer
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """
    输入:
        无
    输出:
        argparse.Namespace
            - config: str, 配置文件路径
    """
    parser = argparse.ArgumentParser(description="Stage 1 SFT training entry.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="Path to yaml config file.",
    )
    return parser.parse_args()


def ensure_parent_dir(file_path: Path) -> None:
    """
    输入:
        file_path: Path, 文件路径
    输出:
        无
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    输入:
        config: Dict[str, Any], 配置字典
    输出:
        Dict[str, Path]
            - output_dir: 主训练输出目录
            - log_dir: 日志目录
            - checkpoint_dir: 推理阶段默认读取的 checkpoint 目录
    """
    training_cfg = config.get("training", {})
    logging_cfg = config.get("logging", {})
    inference_cfg = config.get("inference", {})

    output_dir = Path(training_cfg.get("output_dir", "outputs/default_run"))
    log_dir = Path(logging_cfg.get("log_dir", output_dir / "logs"))
    checkpoint_dir = Path(inference_cfg.get("checkpoint_dir", output_dir / "checkpoint-last"))

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
    }


def copy_config_snapshot(config_path: str, output_dir: Path) -> None:
    """
    输入:
        config_path: str, 原始配置文件路径
        output_dir: Path, 输出目录
    输出:
        无
    """
    snapshot_path = output_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, snapshot_path)


def apply_sample_limit(
    raw_samples: List[Dict[str, Any]],
    max_train_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """
    输入:
        raw_samples: List[Dict[str, Any]], 原始样本列表
        max_train_samples: Optional[int], 最大训练样本数
    输出:
        List[Dict[str, Any]], 截断后的样本列表
    """
    if max_train_samples is None:
        return raw_samples
    if max_train_samples <= 0:
        return []
    return raw_samples[:max_train_samples]


def save_json(data: Dict[str, Any], save_path: Path) -> None:
    """
    输入:
        data: Dict[str, Any], 字典数据
        save_path: Path, 输出文件路径
    输出:
        无
    """
    ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(rows: List[Dict[str, Any]], save_path: Path) -> None:
    """
    输入:
        rows: List[Dict[str, Any]], 记录列表
        save_path: Path, 输出 jsonl 文件路径
    输出:
        无
    """
    ensure_parent_dir(save_path)
    with open(save_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_processed_preview(processed_samples: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    输入:
        processed_samples: List[Dict[str, Any]], 预处理后的样本
        output_dir: Path, 输出目录
    输出:
        无
    """
    preview_path = output_dir / "processed_preview.jsonl"
    preview_rows = processed_samples[:5]
    save_jsonl(preview_rows, preview_path)


def validate_processed_samples(processed_samples: List[Dict[str, Any]]) -> None:
    """
    输入:
        processed_samples: List[Dict[str, Any]], 预处理后的样本
    输出:
        无

    说明:
        成员5的 trainer_sft.py 期望输入格式为:
            [{"text": "..."}]
        这里做一次显式校验，避免训练阶段因为字段不一致报错。
    """
    if not isinstance(processed_samples, list):
        raise TypeError("processed_samples must be a list.")

    for index, sample in enumerate(processed_samples[:10]):
        if not isinstance(sample, dict):
            raise TypeError(f"processed_samples[{index}] must be a dict.")
        if "text" not in sample:
            raise KeyError(
                f"processed_samples[{index}] missing key 'text'. "
                "preprocess_data() must return a list of {'text': ...} dicts."
            )


def save_checkpoint_artifacts(trainer: Any, checkpoint_dir: Path, output_dir: Path, config: Dict[str, Any]) -> None:
    """
    输入:
        trainer: Any, 训练器对象，期望至少具备 save_model()，并尽量提供 processing_class / tokenizer
        checkpoint_dir: Path, 训练后最小推理验证默认读取的 checkpoint 目录
        output_dir: Path, 主输出目录
        config: Dict[str, Any], 配置字典
    输出:
        无

    说明:
        目标:
        1. 保留成员5真实 trainer 的训练输出
        2. 为成员6的 run_infer.py 提供一个稳定可读的 checkpoint 目录
    """
    # 保存 adapter / model 到 checkpoint_dir
    if hasattr(trainer, "save_model") and callable(trainer.save_model):
        trainer.save_model(str(checkpoint_dir))

    # 尝试保存 tokenizer / processing_class
    tokenizer_like = None
    if hasattr(trainer, "processing_class"):
        tokenizer_like = getattr(trainer, "processing_class")
    elif hasattr(trainer, "tokenizer"):
        tokenizer_like = getattr(trainer, "tokenizer")

    if tokenizer_like is not None and hasattr(tokenizer_like, "save_pretrained"):
        tokenizer_like.save_pretrained(str(checkpoint_dir))

    # 保存 trainer state（如果可用）
    if hasattr(trainer, "save_state") and callable(trainer.save_state):
        trainer.save_state()

    checkpoint_meta = {
        "checkpoint_type": "stage1_train_output",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": config.get("run_name", "unknown_run"),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "note": "Checkpoint directory prepared for minimal post-training inference validation.",
    }
    save_json(checkpoint_meta, checkpoint_dir / "checkpoint_meta.json")


def save_run_meta(
    config: Dict[str, Any],
    output_dir: Path,
    num_raw_samples: int,
    num_processed_samples: int,
) -> None:
    """
    输入:
        config: Dict[str, Any], 配置字典
        output_dir: Path, 输出目录
        num_raw_samples: int, 原始样本数
        num_processed_samples: int, 预处理样本数
    输出:
        无
    """
    run_meta = {
        "run_name": config.get("run_name", "unknown_run"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_sample_count": num_raw_samples,
        "processed_sample_count": num_processed_samples,
        "output_dir": str(output_dir),
        "model_name_or_path": config.get("model", {}).get("model_name_or_path", "unknown"),
    }
    save_json(run_meta, output_dir / "run_meta.json")


def maybe_run_inference(config_path: str, config: Dict[str, Any]) -> None:
    """
    输入:
        config_path: str, 当前配置文件路径
        config: Dict[str, Any], 配置字典
    输出:
        无
    """
    if not bool(config.get("run_infer_after_train", False)):
        print("[Info] Skip post-training inference.")
        return

    run_infer_path = Path("run_infer.py")
    if not run_infer_path.exists():
        print("[Warning] run_infer.py not found. Skip post-training inference.")
        return

    print("[Info] Running minimal post-training inference check...")
    subprocess.run(
        [sys.executable, "run_infer.py", "--config", config_path],
        check=True,
    )
    print("[Info] Minimal inference check finished.")


def main() -> None:
    """
    输入:
        无
    输出:
        无
    """
    args = parse_args()
    config = load_config(args.config)

    dirs = ensure_output_dirs(config)
    output_dir = dirs["output_dir"]
    checkpoint_dir = dirs["checkpoint_dir"]

    copy_config_snapshot(args.config, output_dir)

    # 1) 读取原始数据
    data_cfg = config.get("data", {})
    train_file = data_cfg.get("train_file")
    if not train_file:
        raise ValueError("Config missing data.train_file")

    raw_samples = load_data(train_file)
    raw_samples = apply_sample_limit(raw_samples, data_cfg.get("max_train_samples"))
    print(f"[Info] Raw training samples loaded: {len(raw_samples)}")
    print(f"[Info] Training file: {train_file}")

    # 2) 预处理
    processed_samples = preprocess_data(raw_samples, config)
    validate_processed_samples(processed_samples)
    print(f"[Info] Processed training samples: {len(processed_samples)}")

    save_processed_preview(processed_samples, output_dir)

    # 3) 构建 trainer（成员5真实训练逻辑）
    trainer = build_trainer(processed_samples, config)
    print("[Info] Trainer initialized successfully.")

    # 4) 启动训练
    trainer.train()
    print("[Info] Training finished successfully.")

    # 5) 保存最小可验证 checkpoint 产物
    save_checkpoint_artifacts(
        trainer=trainer,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        config=config,
    )
    print(f"[Info] Checkpoint artifacts saved to: {checkpoint_dir}")

    # 6) 保存运行元信息
    save_run_meta(
        config=config,
        output_dir=output_dir,
        num_raw_samples=len(raw_samples),
        num_processed_samples=len(processed_samples),
    )
    print(f"[Info] Run metadata saved to: {output_dir / 'run_meta.json'}")

    # 7) 训练后最小推理验证
    maybe_run_inference(args.config, config)

    print("[Info] Stage 1 training pipeline finished successfully.")


if __name__ == "__main__":
    main()