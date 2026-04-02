import jsonlines
import glob
import os
import random

def merge_files(pattern, output_path):
    """合并所有匹配 pattern 的 JSONL 文件到 output_path"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for file in sorted(glob.glob(pattern)):
            print(f"Adding {file}...")
            with jsonlines.open(file) as reader:
                for obj in reader:
                    writer.write(obj)
    print(f"Merged into {output_path}")

def create_debug(input_path, output_path, n=100, seed=42):
    """从 input_path 的 JSONL 中随机抽取 n 条作为 debug 集"""
    # 设置随机种子以保证可复现性
    random.seed(seed)

    # 读取所有样本
    samples = []
    with jsonlines.open(input_path) as reader:
        for obj in reader:
            samples.append(obj)

    total = len(samples)
    # 如果样本数少于 n，则取全部；否则随机抽取 n 条
    if total <= n:
        debug_samples = samples
        print(f"Total samples {total} <= {n}, using all samples for debug.")
    else:
        debug_samples = random.sample(samples, n)
        print(f"Randomly selected {n} samples from {total} total samples (seed={seed}).")

    # 写入 debug 文件
    with jsonlines.open(output_path, mode="w") as writer:
        for s in debug_samples:
            writer.write(s)
    print(f"Debug set created: {output_path} ({len(debug_samples)} samples)")

if __name__ == "__main__":
    # 合并训练集
    merge_files("processed/*_train.jsonl", "unified/train.jsonl")
    # 合并验证集
    merge_files("processed/*_dev.jsonl", "unified/dev.jsonl")
    # 从训练集中随机抽取 100 条生成 debug 集（可修改 n 和 seed）
    create_debug("unified/train.jsonl", "unified/debug.jsonl", n=100, seed=42)