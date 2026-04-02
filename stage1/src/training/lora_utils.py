"""
lora_utils.py — Model loading and LoRA/QLoRA configuration.

Yike Zhang @Arcadia_Ebendie

Usage:
    model, tokenizer = load_model_and_tokenizer(config)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def load_model_and_tokenizer(config):
    """
    Load base model + tokenizer, then wrap with LoRA (or QLoRA).

    Expected config keys:
        config["model"]["model_name_or_path"]  — HF model id or local path
        config["model"].get("use_qlora", False) — whether to use 4-bit QLoRA
        config["model"].get("lora_r", 16)
        config["model"].get("lora_alpha", 32)
        config["model"].get("lora_dropout", 0.05)
        config["model"].get("lora_target_modules", ["q_proj", "v_proj"])
    """
    model_cfg = config["model"]
    model_name = model_cfg["model_name_or_path"]
    use_qlora = model_cfg.get("use_qlora", False)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # quantization (QLoRA)
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # base model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        target_modules=model_cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer
