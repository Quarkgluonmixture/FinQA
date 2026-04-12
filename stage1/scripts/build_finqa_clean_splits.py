#!/usr/bin/env python3
"""
Build FinQA-only clean splits from normalized jsonl files.

Default policy (first-round clean run):
- source_dataset == finqa
- formula_exec_ok == True
- scale_relation == consistent

Outputs:
- train_full.jsonl / dev_full.jsonl
- train_debug_50.jsonl
- train_250.jsonl / train_1000.jsonl (nested, stratified by program_steps_bucket)
- id lists and summary json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

CURRENT_DIR = Path(__file__).resolve().parent
STAGE1_ROOT = CURRENT_DIR.parent
if str(STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(STAGE1_ROOT))

from src.preprocessing.formula_utils import parse_number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FinQA clean train/dev/debug/subsets.")
    parser.add_argument(
        "--input_train",
        type=str,
        default=str(STAGE1_ROOT / "data/unified/derived/train_norm.jsonl"),
    )
    parser.add_argument(
        "--input_dev",
        type=str,
        default=str(STAGE1_ROOT / "data/unified/derived/dev_norm.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(STAGE1_ROOT / "data/finqa_clean"),
    )
    parser.add_argument("--source_dataset", type=str, default="finqa")
    parser.add_argument("--require_formula_exec_ok", action="store_true", default=True)
    parser.add_argument("--allow_non_exec", dest="require_formula_exec_ok", action="store_false")
    parser.add_argument("--require_scale_relation", type=str, default="consistent")
    parser.add_argument(
        "--allow_corrected",
        action="store_true",
        help="If set, keep rows with answer_corrected=true when formula_exec_ok=true even if scale_relation != require_scale_relation.",
    )
    parser.add_argument("--debug_size", type=int, default=50)
    parser.add_argument("--subset_sizes", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spot_check_n", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_ids(path: Path, ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in ids:
            f.write(f"{sid}\n")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _bucket_of(row: Dict[str, Any]) -> str:
    bucket = str(row.get("program_steps_bucket", "")).strip()
    if bucket in {"single", "double", "multi"}:
        return bucket
    steps = int(row.get("program_steps", 1) or 1)
    if steps <= 1:
        return "single"
    if steps == 2:
        return "double"
    return "multi"


def _subset_stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter[str] = Counter()
    for row in rows:
        c[_bucket_of(row)] += 1
    return dict(c)


def _stratified_master_order(rows: List[Dict[str, Any]], seed: int) -> List[int]:
    by_bucket: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_bucket[_bucket_of(row)].append(idx)

    rng = random.Random(seed)
    for bucket in by_bucket:
        rng.shuffle(by_bucket[bucket])

    total = len(rows)
    if total == 0:
        return []

    keys = sorted(by_bucket.keys())
    target_ratio: Dict[str, float] = {k: (len(by_bucket[k]) / total) for k in keys}
    produced: Dict[str, int] = {k: 0 for k in keys}
    consumed: Dict[str, int] = {k: 0 for k in keys}

    order: List[int] = []
    for pos in range(total):
        candidates: List[Tuple[float, float, str]] = []
        for k in keys:
            if consumed[k] >= len(by_bucket[k]):
                continue
            desired_so_far = (pos + 1) * target_ratio[k]
            deficit = desired_so_far - produced[k]
            remain_ratio = (len(by_bucket[k]) - consumed[k]) / max(1, total - pos)
            candidates.append((deficit, remain_ratio, k))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        chosen = candidates[0][2]
        order.append(by_bucket[chosen][consumed[chosen]])
        consumed[chosen] += 1
        produced[chosen] += 1
    return order


def _is_close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def _filter_rows(
    rows: List[Dict[str, Any]],
    source_dataset: str,
    require_formula_exec_ok: bool,
    require_scale_relation: str,
    allow_corrected: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    kept: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()

    for row in rows:
        if str(row.get("source_dataset", "")).strip().lower() != source_dataset:
            reasons["drop_non_target_source"] += 1
            continue
        if require_formula_exec_ok and not bool(row.get("formula_exec_ok", False)):
            reasons["drop_formula_exec_fail"] += 1
            continue
        if require_scale_relation:
            relation_ok = str(row.get("scale_relation", "")).strip() == require_scale_relation
            corrected_ok = allow_corrected and bool(row.get("answer_corrected", False))
            if not (relation_ok or corrected_ok):
                reasons["drop_scale_relation_mismatch"] += 1
                continue
        kept.append(row)
        reasons["kept"] += 1

    return kept, dict(reasons)


def main() -> None:
    args = parse_args()
    input_train = Path(args.input_train)
    input_dev = Path(args.input_dev)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(input_train)
    dev_rows = load_jsonl(input_dev)

    train_kept, train_reason = _filter_rows(
        train_rows,
        source_dataset=args.source_dataset,
        require_formula_exec_ok=bool(args.require_formula_exec_ok),
        require_scale_relation=str(args.require_scale_relation),
        allow_corrected=bool(args.allow_corrected),
    )
    dev_kept, dev_reason = _filter_rows(
        dev_rows,
        source_dataset=args.source_dataset,
        require_formula_exec_ok=bool(args.require_formula_exec_ok),
        require_scale_relation=str(args.require_scale_relation),
        allow_corrected=bool(args.allow_corrected),
    )

    if not train_kept:
        raise RuntimeError("Filtered train set is empty.")
    if not dev_kept:
        raise RuntimeError("Filtered dev set is empty.")

    train_full_path = out_dir / "train_full.jsonl"
    dev_full_path = out_dir / "dev_full.jsonl"
    save_jsonl(train_full_path, train_kept)
    save_jsonl(dev_full_path, dev_kept)

    order = _stratified_master_order(train_kept, seed=int(args.seed))
    ordered_train = [train_kept[i] for i in order]
    ordered_ids = [str(row.get("id", f"row_{i}")) for i, row in enumerate(ordered_train)]
    save_ids(out_dir / "train_full_ids.txt", ordered_ids)

    debug_n = min(int(args.debug_size), len(ordered_train))
    debug_rows = ordered_train[:debug_n]
    save_jsonl(out_dir / f"train_debug_{debug_n}.jsonl", debug_rows)
    save_ids(out_dir / f"train_debug_{debug_n}_ids.txt", [str(r.get("id", "")) for r in debug_rows])

    subset_summary: Dict[str, Any] = {}
    for n in sorted({int(x) for x in args.subset_sizes if int(x) > 0}):
        if n > len(ordered_train):
            continue
        subset = ordered_train[:n]
        subset_ids = [str(r.get("id", "")) for r in subset]
        save_jsonl(out_dir / f"train_{n}.jsonl", subset)
        save_ids(out_dir / f"train_{n}_ids.txt", subset_ids)
        subset_summary[str(n)] = {
            "size": n,
            "bucket_counts": _subset_stats(subset),
        }

    # Spot-check 20 examples: target_answer must stay numerically aligned with formula_value.
    rng = random.Random(int(args.seed))
    check_pool = train_kept[:] if len(train_kept) <= args.spot_check_n else rng.sample(train_kept, args.spot_check_n)
    spot_rows: List[Dict[str, Any]] = []
    spot_fail = 0
    for row in check_pool:
        target = parse_number(str(row.get("target_answer", "")))
        formula = row.get("formula_value")
        ok = (target is not None) and (formula is not None) and _is_close(float(target), float(formula))
        if not ok:
            spot_fail += 1
        spot_rows.append(
            {
                "id": row.get("id", ""),
                "program_steps_bucket": row.get("program_steps_bucket", ""),
                "scale_relation": row.get("scale_relation", ""),
                "target_answer": row.get("target_answer", ""),
                "formula_value": row.get("formula_value"),
                "spot_ok": ok,
            }
        )

    summary = {
        "input_train": str(input_train),
        "input_dev": str(input_dev),
        "policy": {
            "source_dataset": args.source_dataset,
            "require_formula_exec_ok": bool(args.require_formula_exec_ok),
            "require_scale_relation": str(args.require_scale_relation),
            "allow_corrected": bool(args.allow_corrected),
        },
        "counts": {
            "train_total": len(train_rows),
            "train_kept": len(train_kept),
            "dev_total": len(dev_rows),
            "dev_kept": len(dev_kept),
        },
        "drop_reasons": {
            "train": train_reason,
            "dev": dev_reason,
        },
        "bucket_counts": {
            "train_full": _subset_stats(train_kept),
            "dev_full": _subset_stats(dev_kept),
            f"train_debug_{debug_n}": _subset_stats(debug_rows),
        },
        "subsets": subset_summary,
        "nested_check": {
            "250_in_1000": None,
        },
        "spot_check": {
            "n": len(check_pool),
            "fail": spot_fail,
            "rows": spot_rows,
        },
    }

    path_250 = out_dir / "train_250_ids.txt"
    path_1000 = out_dir / "train_1000_ids.txt"
    if path_250.exists() and path_1000.exists():
        ids_250 = set(path_250.read_text(encoding="utf-8").splitlines())
        ids_1000 = set(path_1000.read_text(encoding="utf-8").splitlines())
        summary["nested_check"]["250_in_1000"] = ids_250.issubset(ids_1000)

    summary_path = out_dir / "clean_summary.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
