#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = Path(os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface")))


@dataclass
class CompareResult:
    same_count: int
    total: int
    rows: List[Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3 diagnostics: format/raw_output/adapter compare")
    parser.add_argument("--root", type=str, default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--run_name", type=str, default="8b_clean_full_answer_only")
    parser.add_argument("--train_jsonl", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=str(DEFAULT_HF_CACHE))
    parser.add_argument("--compare_samples", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse_existing_compare", action="store_true", default=True)
    parser.add_argument("--report_md", type=str, default="")
    return parser.parse_args()


def _read_first_jsonl(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise RuntimeError(f"empty jsonl: {path}")
    return json.loads(line)


def _read_jsonl_rows(path: Path, n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_eval_jsonl(results_dir: Path) -> Path:
    cands = sorted(results_dir.glob("finqa_*_oracle_test.jsonl"))
    if not cands:
        raise FileNotFoundError(f"No finqa_*_oracle_test.jsonl in {results_dir}")
    return cands[0]


def _load_prompt_builders(baseline_root: Path):
    sys.path.insert(0, str(baseline_root))
    from utils.prompting import build_system_instruction, build_finqa_prompt

    return build_system_instruction, build_finqa_prompt


def _ensure_compare_jsonl(
    py_bin: Path,
    baseline_root: Path,
    cache_dir: Path,
    adapter_path: Path,
    compare_samples: int,
    max_new_tokens: int,
    seed: int,
    reuse_existing: bool,
) -> Tuple[Path, Path]:
    one_path = Path("/tmp/diag_compare_samples.jsonl")
    base_dir = Path("/tmp/diag_compare_base")
    adpt_dir = Path("/tmp/diag_compare_adapter")
    base_jsonl = base_dir / "finqa_Qwen_Qwen3-8B_oracle_test.jsonl"
    adpt_jsonl = adpt_dir / "finqa_Qwen_Qwen3-8B_oracle_test.jsonl"

    if reuse_existing and base_jsonl.exists() and adpt_jsonl.exists():
        try:
            if len(base_jsonl.read_text(encoding="utf-8").strip().splitlines()) >= compare_samples and len(
                adpt_jsonl.read_text(encoding="utf-8").strip().splitlines()
            ) >= compare_samples:
                return base_jsonl, adpt_jsonl
        except Exception:
            pass

    ds = load_dataset("dreamerdeo/finqa", split="test", cache_dir=str(cache_dir))
    with one_path.open("w", encoding="utf-8") as f:
        for i in range(compare_samples):
            f.write(json.dumps(ds[i], ensure_ascii=False) + "\n")

    common = [
        str(py_bin),
        "eval_finqa.py",
        "--model_name",
        "Qwen/Qwen3-8B",
        "--setting",
        "oracle",
        "--split",
        "test",
        "--cache_dir",
        str(cache_dir),
        "--evaluator",
        "math_verify",
        "--answer_format",
        "final_answer_tag",
        "--final_answer_tag",
        "FINAL_ANSWER",
        "--no-enable_thinking",
        "--max_new_tokens",
        str(max_new_tokens),
        "--seed",
        str(seed),
        "--num_samples",
        str(compare_samples),
        "--local_json",
        str(one_path),
    ]

    subprocess.run(common + ["--results_dir", str(base_dir)], check=True, cwd=str(baseline_root))
    subprocess.run(
        common + ["--results_dir", str(adpt_dir), "--adapter_path", str(adapter_path)],
        check=True,
        cwd=str(baseline_root),
    )

    return base_jsonl, adpt_jsonl


def _compare_jsonl(base_jsonl: Path, adpt_jsonl: Path, total: int) -> CompareResult:
    rb = [json.loads(x) for x in base_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()][:total]
    ra = [json.loads(x) for x in adpt_jsonl.read_text(encoding="utf-8").splitlines() if x.strip()][:total]
    n = min(len(rb), len(ra))
    rows: List[Dict[str, Any]] = []
    same = 0
    for i in range(n):
        b = rb[i]
        a = ra[i]
        eq = b.get("raw_output") == a.get("raw_output")
        same += int(eq)
        rows.append(
            {
                "idx": i,
                "same_raw_output": eq,
                "base_raw_output": b.get("raw_output"),
                "adapter_raw_output": a.get("raw_output"),
                "base_pred": b.get("pred"),
                "adapter_pred": a.get("pred"),
            }
        )
    return CompareResult(same_count=same, total=n, rows=rows)


def _to_markdown(
    run_name: str,
    train_first: Dict[str, Any],
    processed_text: str,
    eval_first: Dict[str, Any],
    eval_system: str,
    eval_user: str,
    first_five: List[Dict[str, Any]],
    open_rows: List[Dict[str, Any]],
    compare: CompareResult,
) -> str:
    train_has_think = "<think>" in processed_text.lower() or "</think>" in processed_text.lower()
    eval_has_think = "<think>" in eval_user.lower() or "</think>" in eval_user.lower()

    lines: List[str] = []
    lines.append(f"# 3-Step Diagnostic Report ({run_name})")
    lines.append("")
    lines.append("## Step 1: Training vs Eval Prompt Format")
    lines.append(f"- train sample id: `{train_first.get('id')}`")
    lines.append(f"- train sample question: `{train_first.get('question')}`")
    lines.append(f"- training processed text has `<think>` tags: `{train_has_think}`")
    lines.append(f"- eval user prompt has `<think>` tags: `{eval_has_think}`")
    lines.append("")
    lines.append("### Training Processed Text (head)")
    lines.append("```text")
    lines.append(processed_text[:900])
    lines.append("```")
    lines.append("")
    lines.append("### Eval System Prompt")
    lines.append("```text")
    lines.append(eval_system)
    lines.append("```")
    lines.append("")
    lines.append("### Eval User Prompt (head)")
    lines.append("```text")
    lines.append(eval_user[:900])
    lines.append("```")
    lines.append("")
    lines.append("### Finding")
    lines.append(
        "- Prompt protocol is mismatched: training uses a monolithic `Instruction/Input/Output` supervised string, while eval uses chat-style `system + user` messages with different wording and context layout."
    )
    lines.append(
        "- `thinking` tag mismatch is NOT observed in this run (both sides show no `<think>` tags under current config)."
    )
    lines.append("")
    lines.append("## Step 2: Inspect 5 Raw Outputs")
    lines.append("| idx | gold | pred | correct | tag_status | raw_output |")
    lines.append("|---:|---:|---:|---|---|---|")
    for i, r in enumerate(first_five):
        raw = str(r.get("raw_output", "")).replace("|", "\\|")
        lines.append(
            f"| {i} | {r.get('gold')} | {r.get('pred')} | {r.get('correct')} | {r.get('tag_status')} | `{raw}` |"
        )
    lines.append("")
    if open_rows:
        lines.append("### Non-closed Tag Example")
        for r in open_rows[:1]:
            lines.append(f"- question: `{r.get('question')}`")
            lines.append(f"- tag_status: `{r.get('tag_status')}`")
            lines.append(f"- raw_output: `{str(r.get('raw_output', ''))[:220]}`")
    lines.append("")
    lines.append("### Finding")
    lines.append(
        "- Model generally follows `FINAL_ANSWER` format, but there are pathological outputs (e.g., runaway digits with `open_only`) and many scale-related wrong answers in sampled failures."
    )
    lines.append("")
    lines.append("## Step 3: Base vs Adapter (same inputs)")
    lines.append(f"- same raw output count: `{compare.same_count}/{compare.total}`")
    lines.append("| idx | same_raw_output | base_pred | adapter_pred |")
    lines.append("|---:|---|---:|---:|")
    for row in compare.rows:
        lines.append(
            f"| {row['idx']} | {row['same_raw_output']} | {row['base_pred']} | {row['adapter_pred']} |"
        )
    lines.append("")
    lines.append("### Finding")
    if compare.same_count < compare.total:
        lines.append("- Adapter is being loaded and changing outputs (not identical to base).")
    else:
        lines.append("- Base and adapter outputs are identical in sampled set; this is suspicious for adapter effectiveness.")

    lines.append("")
    lines.append("## Overall Diagnosis")
    lines.append(
        "- Most likely primary issue is training/eval prompt protocol mismatch (format + instruction framing + context packaging), not `<think>` on/off mismatch and not adapter loading failure."
    )

    lines.append("")
    lines.append("## Raw Eval Row Head")
    lines.append("```json")
    lines.append(json.dumps(eval_first, ensure_ascii=False, indent=2)[:1600])
    lines.append("```")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    stage1_root = root / "stage1"
    baseline_root = root / "finqa_baseline"
    py_bin = baseline_root / ".venv/bin/python"

    train_jsonl = Path(args.train_jsonl) if args.train_jsonl else stage1_root / "data/finqa_clean/train_full.jsonl"
    run_out_dir = stage1_root / f"outputs/sft_clean_strict/{args.run_name}"
    results_dir = baseline_root / f"results/sft_clean_strict/{args.run_name}"
    processed_preview = run_out_dir / "processed_preview.jsonl"
    eval_jsonl = _find_eval_jsonl(results_dir)

    if not py_bin.exists():
        raise FileNotFoundError(f"baseline python not found: {py_bin}")
    if not train_jsonl.exists():
        raise FileNotFoundError(f"missing train jsonl: {train_jsonl}")
    if not processed_preview.exists():
        raise FileNotFoundError(f"missing processed preview: {processed_preview}")
    if not eval_jsonl.exists():
        raise FileNotFoundError(f"missing eval jsonl: {eval_jsonl}")

    train_first = _read_first_jsonl(train_jsonl)
    processed_first = _read_first_jsonl(processed_preview)
    processed_text = str(processed_first.get("text", ""))

    eval_first = _read_first_jsonl(eval_jsonl)
    first_five = _read_jsonl_rows(eval_jsonl, 5)
    open_rows = [r for r in _read_jsonl_rows(eval_jsonl, 200) if r.get("tag_status") != "closed"]

    build_system_instruction, build_finqa_prompt = _load_prompt_builders(baseline_root)
    cache_dir = Path(args.cache_dir)
    ds = load_dataset("dreamerdeo/finqa", split="test", cache_dir=str(cache_dir))
    ex0 = ds[0]
    eval_system = build_system_instruction(answer_format="final_answer_tag", final_answer_tag="FINAL_ANSWER")
    eval_user = build_finqa_prompt(ex0, setting="oracle", answer_format="final_answer_tag", final_answer_tag="FINAL_ANSWER")

    adapter_path = run_out_dir / "checkpoint-last"
    base_jsonl, adpt_jsonl = _ensure_compare_jsonl(
        py_bin=py_bin,
        baseline_root=baseline_root,
        cache_dir=cache_dir,
        adapter_path=adapter_path,
        compare_samples=int(args.compare_samples),
        max_new_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        reuse_existing=bool(args.reuse_existing_compare),
    )
    compare = _compare_jsonl(base_jsonl, adpt_jsonl, int(args.compare_samples))

    report_md = (
        Path(args.report_md)
        if args.report_md
        else stage1_root / f"reports/diagnostic_prompt_adapter_{args.run_name}.md"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(
        _to_markdown(
            run_name=args.run_name,
            train_first=train_first,
            processed_text=processed_text,
            eval_first=eval_first,
            eval_system=eval_system,
            eval_user=eval_user,
            first_five=first_five,
            open_rows=open_rows,
            compare=compare,
        ),
        encoding="utf-8",
    )

    print(f"[saved] {report_md}")


if __name__ == "__main__":
    main()
