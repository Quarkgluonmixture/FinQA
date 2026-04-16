from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load line-delimited JSON records from a JSONL file."""
    records: List[Dict[str, Any]] = []
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
