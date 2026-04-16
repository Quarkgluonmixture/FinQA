#!/usr/bin/env python3
"""Helpers for reading evaluation `summary.json` files consistently."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_summary_json(path: str | Path) -> Dict[str, Any]:
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as file:
        obj = json.load(file)

    if not isinstance(obj, dict):
        raise ValueError(f"Invalid summary format in {summary_path}")
    return obj


def latest_run_record(path: str | Path) -> Dict[str, Any]:
    """Return the latest run payload, supporting both top-level and `runs` formats."""
    obj = load_summary_json(path)
    runs = obj.get("runs")
    if isinstance(runs, list) and runs:
        last = runs[-1]
        if not isinstance(last, dict):
            raise ValueError(f"Invalid run record in {path}")
        return last
    return obj


def metric(path: str | Path, key: str, default: Any = None) -> Any:
    return latest_run_record(path).get(key, default)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Read the latest metric from summary.json")
    parser.add_argument("summary_json", type=str)
    parser.add_argument("metric_key", type=str)
    parser.add_argument("--default", type=str, default="")
    args = parser.parse_args()

    value = metric(args.summary_json, args.metric_key, args.default)
    print(value)


if __name__ == "__main__":
    main()
