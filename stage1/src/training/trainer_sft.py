"""Factory for building a TRL `SFTTrainer` from processed text samples."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from src.training.lora_utils import load_model_and_tokenizer


def build_trainer(processed_data: List[Dict[str, Any]], config: Dict[str, Any]) -> SFTTrainer:
    """Instantiate model, tokenizer, SFT config, and trainer."""
    model, tokenizer = load_model_and_tokenizer(config)
    dataset = Dataset.from_list(processed_data)

    train_cfg = config.get("training", {})
    output_dir = train_cfg.get("output_dir", "outputs/default_run")
    os.makedirs(output_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=config.get("run_name", "sft_run"),
        seed=int(config.get("seed", 42)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        bf16=bool(train_cfg.get("bf16", torch.cuda.is_available())),
        fp16=False,
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=int(train_cfg.get("save_steps", 50)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        max_length=int(train_cfg.get("max_seq_length", 1024)),
    )

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
