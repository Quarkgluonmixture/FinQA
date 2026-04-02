"""
preprocess.py — Convert raw data dicts into training-ready {"text": ...} format.

This bridges Member 4's prompting logic into the main pipeline.
Member 4 should update the prompt template as needed.
"""

import sys
import os

# Add project root to path so we can import scripts/prompting.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from prompting import build_prompt


def preprocess_data(data, config):
    """
    Convert raw data list into [{"text": "..."}] for SFTTrainer.

    Args:
        data: list of dicts with keys: id, question, context, answer, source_dataset
        config: full config dict

    Returns:
        list of {"text": "full prompt string"} dicts
    """
    thinking = config.get("preprocessing", {}).get("thinking", True)

    processed = []
    for example in data:
        text = build_prompt(example, thinking=thinking)
        processed.append({"text": text})

    return processed
