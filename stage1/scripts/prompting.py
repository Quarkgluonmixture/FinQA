def build_prompt(example, thinking=True):
    question = example["question"]
    context = example["context"]
    answer = example["answer"]

    instruction = "Solve the financial numerical reasoning problem."

    input_text = f"""Context:
{context}

Question:
{question}
"""

    if thinking:
        output_text = f"""Reasoning:
<step-by-step reasoning here>

FINAL_ANSWER: {answer}
"""
    else:
        output_text = f"""FINAL_ANSWER: {answer}
"""

    return f"""Instruction:
{instruction}

Input:
{input_text}

Output:
{output_text}
"""