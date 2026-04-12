# 3-Step Diagnostic Report (8b_clean_full_answer_only)

## Step 1: Training vs Eval Prompt Format
- train sample id: `ADI/2009/page_49.pdf-1`
- train sample question: `what is the the interest expense in 2009?`
- training processed text has `<think>` tags: `False`
- eval user prompt has `<think>` tags: `False`

### Training Processed Text (head)
```text
Instruction:
Solve the financial numerical reasoning problem. Return exactly one tagged final answer as [FINAL_ANSWER]...[/FINAL_ANSWER] in the output.

Input:
Context:
interest rate to a variable interest rate based on the three-month libor plus 2.05% ( 2.05 % ) ( 2.34% ( 2.34 % ) as of october 31 , 2009 ) .
if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million .
foreign currency exposure as more fully described in note 2i .
in the notes to consolidated financial statements contained in item 8 of this annual report on form 10-k , we regularly hedge our non-u.s .
dollar-based exposures by entering into forward foreign currency exchange contracts .
the terms of these contracts are for periods matching the duration of the underlying exposure and generally range from one month to twelve months .
currently , our largest foreign currency exposure is 
```

### Eval System Prompt
```text
You are a financial QA assistant. Output exactly one final answer wrapped as [FINAL_ANSWER]...[/FINAL_ANSWER]. Inside the tag, include only the final numeric answer (no explanation, no units). Do not output anything outside the tag.
```

### Eval User Prompt (head)
```text
Context:
the 2014 net revenue of amount ( in millions ) is $ 5735 ;
the 2015 net revenue of amount ( in millions ) is $ 5829 ;

Question:
what is the net change in net revenue during 2015 for entergy corporation?

Return exactly one final answer as [FINAL_ANSWER]<number>[/FINAL_ANSWER]. No text outside the tag.
```

### Finding
- Prompt protocol is mismatched: training uses a monolithic `Instruction/Input/Output` supervised string, while eval uses chat-style `system + user` messages with different wording and context layout.
- `thinking` tag mismatch is NOT observed in this run (both sides show no `<think>` tags under current config).

## Step 2: Inspect 5 Raw Outputs
| idx | gold | pred | correct | tag_status | raw_output |
|---:|---:|---:|---|---|---|
| 0 | 94.0 | 94.0 | True | closed | `[FINAL_ANSWER]94[/FINAL_ANSWER]` |
| 1 | 0.14 | 14.5 | False | closed | `[FINAL_ANSWER]14.5[/FINAL_ANSWER]` |
| 2 | 0.099 | 8.4 | False | closed | `[FINAL_ANSWER]8.4[/FINAL_ANSWER]` |
| 3 | 0.028999999999999998 | 0.29 | False | closed | `[FINAL_ANSWER]0.29[/FINAL_ANSWER]` |
| 4 | None | 11.0 | False | closed | `[FINAL_ANSWER]11.0[/FINAL_ANSWER]` |

### Non-closed Tag Example
- question: `by how much more is the net gains from sales of available-for-sale securities in 2009 compare to 2008?`
- tag_status: `open_only`
- raw_output: `[FINAL_ANSWER]29999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999`

### Finding
- Model generally follows `FINAL_ANSWER` format, but there are pathological outputs (e.g., runaway digits with `open_only`) and many scale-related wrong answers in sampled failures.

## Step 3: Base vs Adapter (same inputs)
- same raw output count: `2/10`
| idx | same_raw_output | base_pred | adapter_pred |
|---:|---|---:|---:|
| 0 | True | 94.0 | 94.0 |
| 1 | False | 14.57142857 | 14.5 |
| 2 | False | 4.15 | 8.4 |
| 3 | False | 0.971 | 0.29 |
| 4 | False | 11.43 | 11.0 |
| 5 | False | 19.33 | 19.3 |
| 6 | False | 0.1003 | 0.1 |
| 7 | False | 0.12 | 30.1 |
| 8 | True | 1.0 | 1.0 |
| 9 | False | -35.0 | -45.0 |

### Finding
- Adapter is being loaded and changing outputs (not identical to base).

## Overall Diagnosis
- Most likely primary issue is training/eval prompt protocol mismatch (format + instruction framing + context packaging), not `<think>` on/off mismatch and not adapter loading failure.

## Raw Eval Row Head
```json
{
  "question": "what is the net change in net revenue during 2015 for entergy corporation?",
  "gold": 94.0,
  "gold_text": "94",
  "raw_output": "[FINAL_ANSWER]94[/FINAL_ANSWER]",
  "final_answer_text": "94",
  "tag_status": "closed",
  "pred": 94.0,
  "pred_legacy_raw": 94.0,
  "pred_legacy_adjusted": 94.0,
  "correct": true,
  "correct_mathverify": true,
  "correct_legacy": true,
  "correct_legacy_base": true,
  "parse_fail": false,
  "parse_fail_mathverify": false,
  "parse_fail_legacy": false,
  "percent_recovered": false,
  "mathverify_error": "",
  "evaluator": "math_verify"
}
```
