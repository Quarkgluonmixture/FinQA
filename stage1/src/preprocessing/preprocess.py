"""Convert raw examples into TRL-compatible `{"text": ...}` records."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

# Import shared prompt builder from stage1/scripts.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from prompting import build_prompt


def preprocess_data(data: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Render prompts for supervised fine-tuning."""
    preprocessing_cfg = config.get("preprocessing", {})
    thinking = bool(preprocessing_cfg.get("thinking", False))
    supervision_style = str(preprocessing_cfg.get("supervision_style", "answer_only"))
    final_answer_tag = str(preprocessing_cfg.get("final_answer_tag", "FINAL_ANSWER"))

    processed: List[Dict[str, str]] = []
    for example in data:
        text = build_prompt(
            example,
            thinking=thinking,
            supervision_style=supervision_style,
            final_answer_tag=final_answer_tag,
        )
        processed.append({"text": text})

    return processed
