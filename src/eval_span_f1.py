import argparse
import json

from span_utils import span_metrics


def load_gold(path):
    gold = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["id"]
            spans = []
            for e in obj.get("entities", []):
                spans.append((e["start"], e["end"], e["label"]))
            gold[uid] = spans
    return gold


def load_pred(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pred = {}
    for uid, ents in obj.items():
        spans = []
        for e in ents:
            spans.append((e["start"], e["end"], e["label"]))
        pred[uid] = spans
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    pred = load_pred(args.pred)

    metrics = span_metrics(gold, pred)

    print("Per-entity metrics:")
    for lab, scores in metrics["per_label"].items():
        print(
            f"{lab:15s} P={scores['precision']:.3f} "
            f"R={scores['recall']:.3f} F1={scores['f1']:.3f}"
        )

    print(f"\nMacro-F1: {metrics['macro_f1']:.3f}")

    pii = metrics["pii"]
    non = metrics["non_pii"]
    print(f"\nPII-only metrics: P={pii['precision']:.3f} R={pii['recall']:.3f} F1={pii['f1']:.3f}")
    print(f"Non-PII metrics: P={non['precision']:.3f} R={non['recall']:.3f} F1={non['f1']:.3f}")


if __name__ == "__main__":
    main()
