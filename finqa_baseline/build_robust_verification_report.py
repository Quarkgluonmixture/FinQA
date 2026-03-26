#!/usr/bin/env python3
"""Aggregate thinking=true/false FinQA runs into one robust verification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        return {"runs": []}
    runs = obj.get("runs", [])
    if not isinstance(runs, list):
        return {"runs": []}
    return {"runs": runs}


def _latest_runs(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for run in summary.get("runs", []):
        if run.get("task") != "finqa":
            continue
        key = (
            str(run.get("model", "")),
            str(run.get("setting", "")),
            str(run.get("split", "")),
        )
        latest[key] = run
    return list(latest.values())


def _safe_float(run: Dict[str, Any], key: str, fallback: float = 0.0) -> float:
    try:
        return float(run.get(key, fallback))
    except Exception:
        return fallback


def _open_only_rate(run: Dict[str, Any]) -> float:
    counts = run.get("tag_status_counts", {})
    if not isinstance(counts, dict):
        return 0.0
    total = 0
    for v in counts.values():
        try:
            total += int(v)
        except Exception:
            pass
    if total <= 0:
        return 0.0
    try:
        open_only = int(counts.get("open_only", 0))
    except Exception:
        open_only = 0
    return float(open_only) / float(total)


def _thinking_label(run: Dict[str, Any]) -> str:
    return "true" if bool(run.get("enable_thinking", False)) else "false"


def _sort_key(run: Dict[str, Any]) -> Tuple[int, int, int]:
    thinking_order = 0 if _thinking_label(run) == "true" else 1
    setting = str(run.get("setting", ""))
    setting_order = 0 if setting == "oracle" else 1
    model = str(run.get("model", ""))
    model_order = 0 if "4B" in model else 1
    return (thinking_order, setting_order, model_order)


def _build_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in sorted(runs, key=_sort_key):
        acc_mv = _safe_float(run, "accuracy_mathverify", _safe_float(run, "accuracy"))
        acc_legacy = _safe_float(run, "accuracy_legacy", _safe_float(run, "accuracy_adjusted"))
        pf_mv = _safe_float(run, "parse_fail_rate_mathverify", _safe_float(run, "parse_fail_rate"))
        pf_legacy = _safe_float(run, "parse_fail_rate_legacy")
        open_rate = _open_only_rate(run)
        rows.append(
            {
                "thinking": _thinking_label(run),
                "model": str(run.get("model", "")),
                "setting": str(run.get("setting", "")),
                "split": str(run.get("split", "")),
                "num_samples": int(run.get("num_samples", 0)),
                "accuracy_mathverify": acc_mv,
                "accuracy_legacy": acc_legacy,
                "delta_mathverify_minus_legacy": acc_mv - acc_legacy,
                "parse_fail_rate_mathverify": pf_mv,
                "parse_fail_rate_legacy": pf_legacy,
                "tag_open_only_rate": open_rate,
            }
        )
    return rows


def _find_best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda r: float(r.get("accuracy_mathverify", 0.0)))


def _pairwise_4b_vs_8b(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        key = (str(r["thinking"]), str(r["setting"]))
        cell = index.setdefault(key, {})
        model = str(r["model"])
        if "4B" in model:
            cell["4B"] = float(r["accuracy_mathverify"])
        elif "8B" in model:
            cell["8B"] = float(r["accuracy_mathverify"])

    pairs: List[Dict[str, Any]] = []
    for (thinking, setting), cell in sorted(index.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if "4B" not in cell or "8B" not in cell:
            continue
        delta = float(cell["4B"]) - float(cell["8B"])
        pairs.append(
            {
                "thinking": thinking,
                "setting": setting,
                "acc_4b_mathverify": float(cell["4B"]),
                "acc_8b_mathverify": float(cell["8B"]),
                "delta_4b_minus_8b": delta,
                "holds_4b_gt_8b": delta > 0.0,
            }
        )
    return pairs


def build_outputs(rows: List[Dict[str, Any]], open_only_threshold: float) -> Dict[str, Any]:
    best = _find_best(rows)
    best_acc = float(best.get("accuracy_mathverify", 0.0)) if best else 0.0
    far_from_saturated = best_acc < 0.5

    pairwise = _pairwise_4b_vs_8b(rows)
    all_pairs_hold = bool(pairwise) and all(bool(p["holds_4b_gt_8b"]) for p in pairwise)

    open_only_alerts = [r for r in rows if float(r["tag_open_only_rate"]) >= open_only_threshold]

    return {
        "best_run": best,
        "best_accuracy_mathverify": best_acc,
        "far_from_saturated": far_from_saturated,
        "fourb_gt_eightb": {
            "all_pairs_hold": all_pairs_hold,
            "pairs": pairwise,
        },
        "open_only_threshold": open_only_threshold,
        "open_only_alerts": open_only_alerts,
        "rows": rows,
    }


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def _to_markdown(payload: Dict[str, Any]) -> str:
    rows = payload["rows"]
    best = payload.get("best_run", {})
    best_acc = float(payload.get("best_accuracy_mathverify", 0.0))
    far_from_saturated = bool(payload.get("far_from_saturated", False))
    pairwise = payload.get("fourb_gt_eightb", {}).get("pairs", [])
    all_pairs_hold = bool(payload.get("fourb_gt_eightb", {}).get("all_pairs_hold", False))
    open_only_threshold = float(payload.get("open_only_threshold", 0.0))
    open_only_alerts = payload.get("open_only_alerts", [])

    lines: List[str] = []
    lines.append("# Robust Verification Report (FinQA)")
    lines.append("")
    lines.append("## Run Matrix Summary")
    lines.append(
        "| thinking | model | setting | split | n | acc_mathverify | acc_legacy | delta_mv_minus_legacy | parse_fail_mv | parse_fail_legacy | tag_open_only_rate |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['thinking']} | {r['model']} | {r['setting']} | {r['split']} | {r['num_samples']} | "
            f"{_fmt(float(r['accuracy_mathverify']))} | {_fmt(float(r['accuracy_legacy']))} | "
            f"{_fmt(float(r['delta_mathverify_minus_legacy']))} | {_fmt(float(r['parse_fail_rate_mathverify']))} | "
            f"{_fmt(float(r['parse_fail_rate_legacy']))} | {_fmt(float(r['tag_open_only_rate']))} |"
        )

    lines.append("")
    lines.append("## 0B Conclusions")
    if best:
        lines.append(
            f"- Best math-verify run: `{best['model']}` / `{best['setting']}` / thinking={best['thinking']} @ {_fmt(best_acc)}"
        )
    lines.append(
        f"- Is FinQA still far from saturated? **{'Yes' if far_from_saturated else 'No'}** "
        f"(best accuracy_mathverify={_fmt(best_acc)})."
    )
    lines.append(
        f"- Does 4B > 8B still hold after verification/format changes? **{'Yes' if all_pairs_hold else 'Not fully'}**."
    )

    lines.append("")
    lines.append("### 4B vs 8B by thinking/setting")
    lines.append("| thinking | setting | acc_4B_mathverify | acc_8B_mathverify | delta_4B_minus_8B | holds_4B_gt_8B |")
    lines.append("|---|---|---:|---:|---:|---|")
    for p in pairwise:
        lines.append(
            f"| {p['thinking']} | {p['setting']} | {_fmt(float(p['acc_4b_mathverify']))} | "
            f"{_fmt(float(p['acc_8b_mathverify']))} | {_fmt(float(p['delta_4b_minus_8b']))} | {p['holds_4b_gt_8b']} |"
        )

    lines.append("")
    lines.append("## Truncation / Extraction Diagnostics")
    lines.append(
        f"- Open-tag-only alert threshold: {open_only_threshold:.3f}. "
        f"Alerted runs: {len(open_only_alerts)}."
    )
    if open_only_alerts:
        lines.append("| thinking | model | setting | tag_open_only_rate |")
        lines.append("|---|---|---|---:|")
        for r in open_only_alerts:
            lines.append(
                f"| {r['thinking']} | {r['model']} | {r['setting']} | {_fmt(float(r['tag_open_only_rate']))} |"
            )
        lines.append(
            "- Recommendation: rerun alerted runs with `max_new_tokens=512` for sensitivity check."
        )
    else:
        lines.append("- No open-only truncation alert triggered.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build robust verification report from thinking true/false runs.")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--output_md", type=str, default="results/robust_verification_report.md")
    p.add_argument("--output_json", type=str, default="results/robust_verification_summary.json")
    p.add_argument("--open_only_threshold", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)

    true_summary = _load_summary(root / "thinking_true" / "summary.json")
    false_summary = _load_summary(root / "thinking_false" / "summary.json")

    runs = _latest_runs(true_summary) + _latest_runs(false_summary)
    rows = _build_rows(runs)
    payload = build_outputs(rows, open_only_threshold=float(args.open_only_threshold))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(_to_markdown(payload))

    print(f"[saved] {output_md}")
    print(f"[saved] {output_json}")


if __name__ == "__main__":
    main()
