from __future__ import annotations

import glob
import os
import random

import jsonlines


def merge_files(pattern: str, output_path: str) -> None:
    """Merge all JSONL files matching `pattern` into `output_path`."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for file_path in sorted(glob.glob(pattern)):
            print(f"Adding {file_path}...")
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    writer.write(obj)
    print(f"Merged into {output_path}")


def create_debug(input_path: str, output_path: str, n: int = 100, seed: int = 42) -> None:
    """Create a reproducible debug subset by random sampling from a JSONL file."""
    random.seed(seed)

    samples = []
    with jsonlines.open(input_path) as reader:
        for obj in reader:
            samples.append(obj)

    total = len(samples)
    if total <= n:
        debug_samples = samples
        print(f"Total samples {total} <= {n}, using all samples for debug.")
    else:
        debug_samples = random.sample(samples, n)
        print(f"Randomly selected {n} samples from {total} total samples (seed={seed}).")

    with jsonlines.open(output_path, mode="w") as writer:
        for sample in debug_samples:
            writer.write(sample)
    print(f"Debug set created: {output_path} ({len(debug_samples)} samples)")


if __name__ == "__main__":
    merge_files("processed/*_train.jsonl", "unified/train.jsonl")
    merge_files("processed/*_dev.jsonl", "unified/dev.jsonl")
    create_debug("unified/train.jsonl", "unified/debug.jsonl", n=100, seed=42)
