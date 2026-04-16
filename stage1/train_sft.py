from __future__ import annotations

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
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Stage 1 SFT training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/debug.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create output, log, and checkpoint directories."""
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
    snapshot_path = output_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, snapshot_path)


def apply_sample_limit(raw_samples: List[Dict[str, Any]], max_train_samples: Optional[int]) -> List[Dict[str, Any]]:
    if max_train_samples is None:
        return raw_samples
    if max_train_samples <= 0:
        return []
    return raw_samples[:max_train_samples]


def save_json(data: Dict[str, Any], save_path: Path) -> None:
    ensure_parent_dir(save_path)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_jsonl(rows: List[Dict[str, Any]], save_path: Path) -> None:
    ensure_parent_dir(save_path)
    with save_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_processed_preview(processed_samples: List[Dict[str, Any]], output_dir: Path) -> None:
    preview_path = output_dir / "processed_preview.jsonl"
    save_jsonl(processed_samples[:5], preview_path)


def validate_processed_samples(processed_samples: List[Dict[str, Any]]) -> None:
    """Ensure trainer input follows `[{"text": ...}]` schema."""
    if not isinstance(processed_samples, list):
        raise TypeError("processed_samples must be a list.")

    for index, sample in enumerate(processed_samples[:10]):
        if not isinstance(sample, dict):
            raise TypeError(f"processed_samples[{index}] must be a dict.")
        if "text" not in sample:
            raise KeyError(
                f"processed_samples[{index}] missing key 'text'. "
                "preprocess_data() must return a list of {'text': ...} dictionaries."
            )


def save_checkpoint_artifacts(trainer: Any, checkpoint_dir: Path, output_dir: Path, config: Dict[str, Any]) -> None:
    """Save model artifacts needed for post-training evaluation."""
    if hasattr(trainer, "save_model") and callable(trainer.save_model):
        trainer.save_model(str(checkpoint_dir))

    tokenizer_like = None
    if hasattr(trainer, "processing_class"):
        tokenizer_like = getattr(trainer, "processing_class")
    elif hasattr(trainer, "tokenizer"):
        tokenizer_like = getattr(trainer, "tokenizer")

    if tokenizer_like is not None and hasattr(tokenizer_like, "save_pretrained"):
        tokenizer_like.save_pretrained(str(checkpoint_dir))

    if hasattr(trainer, "save_state") and callable(trainer.save_state):
        trainer.save_state()

    checkpoint_meta = {
        "checkpoint_type": "stage1_train_output",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": config.get("run_name", "unknown_run"),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "note": "Checkpoint directory prepared for post-training inference validation.",
    }
    save_json(checkpoint_meta, checkpoint_dir / "checkpoint_meta.json")


def save_run_meta(
    config: Dict[str, Any],
    output_dir: Path,
    num_raw_samples: int,
    num_processed_samples: int,
) -> None:
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
    if not bool(config.get("run_infer_after_train", False)):
        print("[Info] Skip post-training inference.")
        return

    run_infer_path = Path("run_infer.py")
    if not run_infer_path.exists():
        print("[Warning] run_infer.py not found. Skip post-training inference.")
        return

    print("[Info] Running post-training inference dry run...")
    subprocess.run(
        [sys.executable, "run_infer.py", "--config", config_path],
        check=True,
    )
    print("[Info] Inference dry run completed.")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dirs = ensure_output_dirs(config)
    output_dir = dirs["output_dir"]
    checkpoint_dir = dirs["checkpoint_dir"]

    copy_config_snapshot(args.config, output_dir)

    data_cfg = config.get("data", {})
    train_file = data_cfg.get("train_file")
    if not train_file:
        raise ValueError("Config missing data.train_file")

    raw_samples = load_data(train_file)
    raw_samples = apply_sample_limit(raw_samples, data_cfg.get("max_train_samples"))
    print(f"[Info] Raw training samples loaded: {len(raw_samples)}")
    print(f"[Info] Training file: {train_file}")

    processed_samples = preprocess_data(raw_samples, config)
    validate_processed_samples(processed_samples)
    print(f"[Info] Processed training samples: {len(processed_samples)}")
    save_processed_preview(processed_samples, output_dir)

    trainer = build_trainer(processed_samples, config)
    print("[Info] Trainer initialized successfully.")

    trainer.train()
    print("[Info] Training finished successfully.")

    save_checkpoint_artifacts(
        trainer=trainer,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        config=config,
    )
    print(f"[Info] Checkpoint artifacts saved to: {checkpoint_dir}")

    save_run_meta(
        config=config,
        output_dir=output_dir,
        num_raw_samples=len(raw_samples),
        num_processed_samples=len(processed_samples),
    )
    print(f"[Info] Run metadata saved to: {output_dir / 'run_meta.json'}")

    maybe_run_inference(args.config, config)

    print("[Info] Stage 1 training pipeline finished successfully.")


if __name__ == "__main__":
    main()
