from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.input)
    counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            for ent in obj.get("privacy_mask", []):
                counter[ent.get("label", "").upper()] += 1
            if args.limit and idx + 1 >= args.limit:
                break

    print(f"Counted {sum(counter.values())} entities across {len(counter)} labels")
    for label, count in counter.most_common():
        print(f"{label:25s} {count}")


if __name__ == "__main__":
    main()

