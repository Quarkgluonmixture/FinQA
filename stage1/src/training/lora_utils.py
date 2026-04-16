"""Model and tokenizer loading helpers for LoRA / QLoRA training."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load a base CausalLM model and wrap it with LoRA adapters."""
    model_cfg = config["model"]
    model_name = model_cfg["model_name_or_path"]
    use_qlora = bool(model_cfg.get("use_qlora", False))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_cfg.get("lora_r", 16)),
        lora_alpha=int(model_cfg.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
        target_modules=model_cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer
