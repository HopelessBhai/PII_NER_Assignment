from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to source JSONL file")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--dev_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_out", default="data/train.jsonl")
    ap.add_argument("--dev_out", default="data/dev.jsonl")
    ap.add_argument("--test_out", default="data/test.jsonl")
    return ap.parse_args()


def write_subset(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    if abs(args.train_ratio + args.dev_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/dev/test ratios must sum to 1.0")

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))

    random.Random(args.seed).shuffle(data)

    n = len(data)
    train_end = int(n * args.train_ratio)
    dev_end = train_end + int(n * args.dev_ratio)

    train_records = data[:train_end]
    dev_records = data[train_end:dev_end]
    test_records = data[dev_end:]

    write_subset(train_records, args.train_out)
    write_subset(dev_records, args.dev_out)
    write_subset(test_records, args.test_out)

    print(
        f"Wrote {len(train_records)} train, {len(dev_records)} dev, {len(test_records)} test examples "
        f"from {n} total."
    )


if __name__ == "__main__":
    main()

