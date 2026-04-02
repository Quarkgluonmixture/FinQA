import json
from prompting import build_prompt


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_training_examples(data, thinking=True):
    processed = []

    for example in data:
        text = build_prompt(example, thinking=thinking)
        processed.append({"text": text})

    return processed


def tokenize_examples(processed_data, tokenizer=None, max_length=1024):
    """
    If tokenizer is provided, convert text into tokenized features.
    If tokenizer is None, keep the original text format.
    """
    if tokenizer is None:
        return processed_data

    tokenized_data = []
    for item in processed_data:
        encoded = tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokenized_data.append(encoded)

    return tokenized_data


def main():
    raw_data = load_jsonl("debug.jsonl")

    # Step 1: build text examples
    processed_data = build_training_examples(raw_data, thinking=True)
    save_jsonl(processed_data, "processed.jsonl")

    # Step 2: tokenizer interface placeholder
    # If you have a real tokenizer later, pass it here.
    tokenized_data = tokenize_examples(processed_data, tokenizer=None)

    # Save tokenizer-stage output as well
    save_jsonl(tokenized_data, "tokenized.jsonl")

    print("Done!")


if __name__ == "__main__":
    main()